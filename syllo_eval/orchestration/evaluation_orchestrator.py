import asyncio
import logging
from datetime import datetime, timezone
from uuid import UUID, uuid4

from pydantic import BaseModel, Field

from syllo_eval.execution.sample_executor import SampleExecutionResult, SampleExecutor
from syllo_eval.infrastructure.database import DatabaseManager
from syllo_eval.infrastructure.unit_of_work import TransactionalUnitOfWork, UnitOfWork
from syllo_eval.logging_utils import bind_run_id, bind_sample_id
from syllo_eval.model import (
  Agent,
  EvaluationRun,
  EvaluationRunSample,
  EvaluationSampleStatus,
  EvaluationStatus,
  Sample,
)

logger = logging.getLogger(__name__)


class EvaluationConfig(BaseModel):
  """Configuration required to start a new evaluation run.

  Attributes:
      agent_id: The ID of the agent being evaluated.
      dataset_id: The ID of the dataset to evaluate against.
  """

  agent_id: UUID
  dataset_id: UUID
  selected_metric_names: list[str] = Field(default_factory=list)
  planned_sample_ids: list[UUID] = Field(default_factory=list)
  config: dict[str, object] | None = None
  source_run_id: UUID | None = None


class EvaluationOrchestrator:
  """Orchestrates a full dataset evaluation run.

  Manages the EvaluationRun lifecycle end-to-end:
    1. Creates the EvaluationRun record (RUNNING).
    2. Fetches all samples for the configured dataset.
    3. Runs each sample in SampleExecutor.
    4. Finalises the run:
       * COMPLETED when every scheduled sample succeeds.
       * PARTIALLY_COMPLETED when at least one sample succeeds and at least one fails.
       * FAILED when every sample fails or an unrecoverable error occurs.
  """

  def __init__(
    self,
    db_manager: DatabaseManager,
    sample_executor: SampleExecutor,
    max_concurrent_samples: int = 5,
    sample_trace_timeout_seconds: float | None = None,
    sample_compute_timeout_seconds: float | None = None,
  ) -> None:
    if max_concurrent_samples < 1:
      raise ValueError(f'max_concurrent_samples must be >= 1, got {max_concurrent_samples}')
    if sample_trace_timeout_seconds is not None and sample_trace_timeout_seconds <= 0:
      raise ValueError(f'sample_trace_timeout_seconds must be > 0 when set, got {sample_trace_timeout_seconds}')
    if sample_compute_timeout_seconds is not None and sample_compute_timeout_seconds <= 0:
      raise ValueError(f'sample_compute_timeout_seconds must be > 0 when set, got {sample_compute_timeout_seconds}')

    self._db_manager = db_manager
    self._sample_executor = sample_executor
    self._max_concurrent_samples = max_concurrent_samples
    self._sample_trace_timeout_seconds = sample_trace_timeout_seconds
    self._sample_compute_timeout_seconds = sample_compute_timeout_seconds

  async def run(self, config: EvaluationConfig) -> EvaluationRun:
    evaluation_run = await self.create_run(config)
    return await self.execute_run(evaluation_run)

  async def create_run(self, config: EvaluationConfig) -> EvaluationRun:
    """Create and persist a RUNNING evaluation run."""
    evaluation_run = await self._create_evaluation_run(config)
    logger.info(
      'Evaluation run started run_id=%s agent_id=%s dataset_id=%s',
      evaluation_run.id,
      config.agent_id,
      config.dataset_id,
    )
    return evaluation_run

  async def execute_run(self, evaluation_run: EvaluationRun) -> EvaluationRun:
    """Execute a complete evaluation run.

    Args:
        evaluation_run: Existing RUNNING evaluation run to continue.

    Returns:
        The finalised EvaluationRun record.
    """
    with bind_run_id(evaluation_run.id):
      try:
        current_run = await self._fetch_run(evaluation_run.id)
        if current_run.status != EvaluationStatus.RUNNING:
          logger.info(
            'Evaluation run already terminal run_id=%s status=%s; skipping execution.',
            current_run.id,
            current_run.status.value,
          )
          return current_run

        planned_sample_ids = await self._fetch_planned_sample_ids(current_run.id)
        if (
          not planned_sample_ids
          and current_run.source_run_id is None
          and (current_run.config or {}).get('plan_snapshot_version') != 1
        ):
          samples = await self._fetch_samples(current_run.dataset_id)
          planned_sample_ids = [sample.id for sample in samples]
        elif current_run.source_run_id is None:
          samples = await self._fetch_samples_by_ids(planned_sample_ids)
        else:
          samples = []

        if not planned_sample_ids:
          logger.warning(
            'No samples found for run_id=%s dataset_id=%s; finalizing as %s.',
            current_run.id,
            current_run.dataset_id,
            EvaluationStatus.FAILED.value,
          )
          return await self._finalize_run(current_run.id, EvaluationStatus.FAILED)

        if current_run.source_run_id is None:
          agent = await self._fetch_agent(current_run.agent_id)
          outcomes = await self._execute_samples(current_run.id, agent, samples)
        else:
          source_run_samples = await self._fetch_source_run_samples(current_run.source_run_id)
          traces_by_sample_id = {
            run_sample.sample_id: run_sample.trace_id
            for run_sample in source_run_samples
            if run_sample.trace_id is not None
          }
          recompute_items = [
            (sample_id, traces_by_sample_id[sample_id])
            for sample_id in planned_sample_ids
            if sample_id in traces_by_sample_id
          ]
          outcomes = await self._execute_recomputations(current_run.id, recompute_items)
          outcomes.extend(
            [
              await self._create_missing_source_trace_result(current_run.id, sample_id)
              for sample_id in planned_sample_ids
              if sample_id not in traces_by_sample_id
            ]
          )
        final_status = self._determine_final_status(outcomes)

      except Exception as exc:
        logger.exception('Unrecoverable error during evaluation run_id=%s', evaluation_run.id)
        try:
          return await self._finalize_run(evaluation_run.id, EvaluationStatus.FAILED)
        except Exception:
          logger.exception(
            'Failed to finalize run as FAILED run_id=%s original_error=%s',
            evaluation_run.id,
            exc,
          )
          raise

      succeeded = sum(1 for outcome in outcomes if outcome.succeeded)
      unsuccessful = len(outcomes) - succeeded
      logger.info(
        'Evaluation run finishing run_id=%s status=%s samples=%d succeeded=%d unsuccessful=%d',
        evaluation_run.id,
        final_status.value,
        len(outcomes),
        succeeded,
        unsuccessful,
      )
      return await self._finalize_run(evaluation_run.id, final_status)

  async def _create_evaluation_run(self, config: EvaluationConfig) -> EvaluationRun:
    async with TransactionalUnitOfWork(self._db_manager) as uow:
      evaluation_run = await uow.evaluation_runs.create(
        EvaluationRun(
          id=uuid4(),
          agent_id=config.agent_id,
          dataset_id=config.dataset_id,
          status=EvaluationStatus.RUNNING,
          start_time=datetime.now(timezone.utc),
          source_run_id=config.source_run_id,
          config=config.config,
        )
      )
      await uow.evaluation_run_metrics.create_many(evaluation_run.id, config.selected_metric_names)
      await uow.evaluation_run_plan_samples.create_many(evaluation_run.id, config.planned_sample_ids)
      return evaluation_run

  async def _fetch_run(self, run_id: UUID) -> EvaluationRun:
    async with UnitOfWork(self._db_manager) as uow:
      return await uow.evaluation_runs.get_by_id_or_raise(run_id)

  async def _fetch_samples(self, dataset_id: UUID) -> list[Sample]:
    async with UnitOfWork(self._db_manager) as uow:
      return await uow.samples.list_by_dataset(dataset_id)

  async def _fetch_samples_by_ids(self, sample_ids: list[UUID]) -> list[Sample]:
    async with UnitOfWork(self._db_manager) as uow:
      return await uow.samples.list_by_ids(sample_ids)

  async def _fetch_planned_sample_ids(self, run_id: UUID) -> list[UUID]:
    async with UnitOfWork(self._db_manager) as uow:
      return await uow.evaluation_run_plan_samples.list_sample_ids(run_id)

  async def _fetch_source_run_samples(self, source_run_id: UUID) -> list[EvaluationRunSample]:
    async with UnitOfWork(self._db_manager) as uow:
      return await uow.evaluation_run_samples.list_by_evaluation_run(source_run_id)

  async def _fetch_agent(self, agent_id: UUID) -> Agent:
    async with UnitOfWork(self._db_manager) as uow:
      agent = await uow.agents.get_by_id(agent_id)
    if agent is None:
      raise ValueError(f'Agent not found for agent_id={agent_id}')
    return agent

  async def _execute_samples(
    self,
    evaluation_run_id: UUID,
    agent: Agent,
    samples: list[Sample],
  ) -> list[SampleExecutionResult]:
    """Run scheduled samples through a fixed-size worker pool over a shared iterator."""
    if not samples:
      return []

    outcomes: list[SampleExecutionResult] = []
    samples_iter = iter(samples)

    async def _worker() -> None:
      for sample in samples_iter:
        outcomes.append(await self._execute_one_sample(evaluation_run_id, agent, sample))

    worker_count = min(self._max_concurrent_samples, len(samples))
    await asyncio.gather(*(_worker() for _ in range(worker_count)))
    return outcomes

  async def _execute_recomputations(
    self,
    evaluation_run_id: UUID,
    recompute_items: list[tuple[UUID, str]],
  ) -> list[SampleExecutionResult]:
    if not recompute_items:
      return []

    outcomes: list[SampleExecutionResult] = []
    items_iter = iter(recompute_items)

    async def _worker() -> None:
      for sample_id, trace_id in items_iter:
        outcomes.append(await self._execute_one_recompute(evaluation_run_id, sample_id, trace_id))

    worker_count = min(self._max_concurrent_samples, len(recompute_items))
    await asyncio.gather(*(_worker() for _ in range(worker_count)))
    return outcomes

  async def _execute_one_recompute(
    self,
    evaluation_run_id: UUID,
    sample_id: UUID,
    trace_id: str,
  ) -> SampleExecutionResult:
    with bind_sample_id(sample_id):
      try:
        return await self._sample_executor.recompute(
          evaluation_run_id=evaluation_run_id,
          sample_id=sample_id,
          trace_id=trace_id,
          compute_timeout_seconds=self._sample_compute_timeout_seconds,
        )
      except Exception as exc:
        logger.error(
          'Sample recomputation crashed sample_id=%s trace_id=%s: %s',
          sample_id,
          trace_id,
          exc,
        )
        return self._build_unpersisted_crashed_result_for_sample_id(evaluation_run_id, sample_id, exc)

  async def _execute_one_sample(
    self,
    evaluation_run_id: UUID,
    agent: Agent,
    sample: Sample,
  ) -> SampleExecutionResult:
    with bind_sample_id(sample.id):
      try:
        return await self._sample_executor.execute(
          evaluation_run_id,
          agent,
          sample,
          trace_timeout_seconds=self._sample_trace_timeout_seconds,
          compute_timeout_seconds=self._sample_compute_timeout_seconds,
        )
      except Exception as exc:
        logger.error(
          'Sample execution crashed sample_id=%s agent_id=%s: %s',
          sample.id,
          agent.id,
          exc,
        )
        try:
          return await self._create_crashed_sample_result(evaluation_run_id, sample, exc)
        except Exception:
          logger.exception(
            'Failed to persist crashed-sample row sample_id=%s; using in-memory failure result',
            sample.id,
          )
          return self._build_unpersisted_crashed_result(evaluation_run_id, sample, exc)

  async def _create_crashed_sample_result(
    self,
    evaluation_run_id: UUID,
    sample: Sample,
    error: Exception,
  ) -> SampleExecutionResult:
    error_message = str(error)
    now = datetime.now(timezone.utc)
    metadata = {'failure_phase': 'sample_setup'}
    async with UnitOfWork(self._db_manager) as uow:
      existing = await uow.evaluation_run_samples.get_by_run_and_sample(
        evaluation_run_id=evaluation_run_id,
        sample_id=sample.id,
      )
      if existing is not None:
        run_sample = await uow.evaluation_run_samples.update_status(
          run_sample_id=existing.id,
          status=EvaluationSampleStatus.FAILED,
          ended_at=now,
          error_message=error_message,
          metadata=metadata,
        )
      else:
        run_sample = await uow.evaluation_run_samples.create(
          EvaluationRunSample(
            id=uuid4(),
            evaluation_run_id=evaluation_run_id,
            sample_id=sample.id,
            status=EvaluationSampleStatus.FAILED,
            started_at=now,
            ended_at=now,
            error_message=error_message,
            metadata=metadata,
          )
        )
    return SampleExecutionResult(evaluation_run_sample=run_sample, computations=[], error=error_message)

  async def _create_missing_source_trace_result(
    self,
    evaluation_run_id: UUID,
    sample_id: UUID,
  ) -> SampleExecutionResult:
    error_message = 'Source run sample has no stored trace to recompute'
    now = datetime.now(timezone.utc)
    try:
      async with UnitOfWork(self._db_manager) as uow:
        run_sample = await uow.evaluation_run_samples.create(
          EvaluationRunSample(
            id=uuid4(),
            evaluation_run_id=evaluation_run_id,
            sample_id=sample_id,
            status=EvaluationSampleStatus.FAILED,
            started_at=now,
            ended_at=now,
            error_message=error_message,
            metadata={'failure_phase': 'metric_compute'},
          )
        )
    except Exception as exc:
      return self._build_unpersisted_crashed_result_for_sample_id(evaluation_run_id, sample_id, exc)

    return SampleExecutionResult(evaluation_run_sample=run_sample, computations=[], error=error_message)

  @staticmethod
  def _build_unpersisted_crashed_result(
    evaluation_run_id: UUID,
    sample: Sample,
    error: Exception,
  ) -> SampleExecutionResult:
    error_message = str(error)
    now = datetime.now(timezone.utc)
    return SampleExecutionResult(
      evaluation_run_sample=EvaluationRunSample(
        id=uuid4(),
        evaluation_run_id=evaluation_run_id,
        sample_id=sample.id,
        status=EvaluationSampleStatus.FAILED,
        started_at=now,
        ended_at=now,
        error_message=error_message,
        metadata={'failure_phase': 'sample_setup', 'persistence_failed': True},
      ),
      computations=[],
      error=error_message,
    )

  @staticmethod
  def _build_unpersisted_crashed_result_for_sample_id(
    evaluation_run_id: UUID,
    sample_id: UUID,
    error: Exception,
  ) -> SampleExecutionResult:
    return SampleExecutionResult(
      evaluation_run_sample=EvaluationRunSample(
        id=uuid4(),
        evaluation_run_id=evaluation_run_id,
        sample_id=sample_id,
        status=EvaluationSampleStatus.FAILED,
        started_at=datetime.now(timezone.utc),
        ended_at=datetime.now(timezone.utc),
        error_message=str(error),
        metadata={'failure_phase': 'sample_setup', 'persistence_failed': True},
      ),
      computations=[],
      error=str(error),
    )

  @staticmethod
  def _determine_final_status(outcomes: list[SampleExecutionResult]) -> EvaluationStatus:
    """Compute final run status from aggregated sample outcomes."""
    if not outcomes:
      return EvaluationStatus.FAILED

    succeeded = sum(1 for outcome in outcomes if outcome.succeeded)
    if succeeded == len(outcomes):
      return EvaluationStatus.COMPLETED
    if succeeded > 0:
      return EvaluationStatus.PARTIALLY_COMPLETED
    return EvaluationStatus.FAILED

  async def _finalize_run(self, run_id: UUID, status: EvaluationStatus) -> EvaluationRun:
    async with UnitOfWork(self._db_manager) as uow:
      finalized_run = await uow.evaluation_runs.update_status_if_current(
        run_id=run_id,
        current_statuses=[EvaluationStatus.RUNNING],
        status=status,
        end_time=datetime.now(timezone.utc),
      )
      if finalized_run is not None:
        return finalized_run

      return await uow.evaluation_runs.get_by_id_or_raise(run_id)
