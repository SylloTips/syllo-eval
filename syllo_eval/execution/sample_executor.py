import asyncio
import logging
from datetime import datetime, timezone
from typing import Any
from uuid import UUID, uuid4

from pydantic import BaseModel

from syllo_eval.execution.agent_caller import AgentCallDispatcher
from syllo_eval.evaluation.metric_planner import MetricPlanner
from syllo_eval.evaluation.plan_executor import PlanExecutor
from syllo_eval.evaluation.trace_processor import TraceIntegration
from syllo_eval.infrastructure.database import DatabaseManager
from syllo_eval.infrastructure.unit_of_work import UnitOfWork
from syllo_eval.model import Agent, EvaluationRunSample, EvaluationSampleStatus, MetricComputation, Sample

logger = logging.getLogger(__name__)


class SampleExecutionResult(BaseModel):
  """Outcome of processing a single sample through the evaluation pipeline."""

  evaluation_run_sample: EvaluationRunSample
  computations: list[MetricComputation]
  error: str | None = None

  @property
  def succeeded(self) -> bool:
    return self.error is None


class SampleExecutor:
  """Executes the full evaluation pipeline for a single sample.

  Responsibilities:
    * Create the EvaluationRunSample record.
    * Invoke TraceProcessor to fetch trace/span data.
    * Invoke MetricPlanner to collect plan items (async iterator).
    * Invoke PlanExecutor to compute metrics and persist the results.
  """

  def __init__(
    self,
    db_manager: DatabaseManager,
    agent_call_dispatcher: AgentCallDispatcher | None,
    trace_processor: TraceIntegration | None,
    metric_planner: MetricPlanner,
    plan_executor: PlanExecutor,
  ) -> None:
    self._db_manager = db_manager
    self._agent_call_dispatcher = agent_call_dispatcher
    self._trace_processor = trace_processor
    self._metric_planner = metric_planner
    self._plan_executor = plan_executor

  async def execute(
    self,
    evaluation_run_id: UUID,
    agent: Agent,
    sample: Sample,
    request_id: str | None = None,
    trace_timeout_seconds: float | None = None,
    compute_timeout_seconds: float | None = None,
  ) -> SampleExecutionResult:
    """Run the evaluation pipeline for one sample.

    Args:
        evaluation_run_id: ID of the parent EvaluationRun.
        agent: Agent being evaluated.
        sample: The sample being evaluated.
        request_id: Optional precomputed request ID. When omitted, the agent is called for the sample.
        trace_timeout_seconds: Optional timeout for the agent call plus trace processing.
        compute_timeout_seconds: Optional timeout for metric computation.

    Returns:
        SampleExecutionResult containing the run-sample record and computed metrics computations.

    Raises:
        Exception: Propagates failures that occur before the run-sample record
            is created. Once the record exists, downstream pipeline failures,
            including timeout, are captured in the returned result (`error`
            field).
    """

    if trace_timeout_seconds is not None and trace_timeout_seconds <= 0:
      raise ValueError(f'trace_timeout_seconds must be > 0 when set, got {trace_timeout_seconds}')
    if compute_timeout_seconds is not None and compute_timeout_seconds <= 0:
      raise ValueError(f'compute_timeout_seconds must be > 0 when set, got {compute_timeout_seconds}')

    evaluation_run_sample = await self._create_evaluation_run_sample(
      evaluation_run_id=evaluation_run_id,
      sample_id=sample.id,
    )
    trace_failure_phase = 'trace_fetch' if request_id is not None else 'agent_call'

    try:
      async with asyncio.timeout(trace_timeout_seconds):
        if request_id is None:
          if self._agent_call_dispatcher is None:
            raise ValueError('Agent caller is required for fresh execution')
          request_id = await self._agent_call_dispatcher.call(agent=agent, sample=sample)

        trace_failure_phase = 'trace_fetch'
        if self._trace_processor is None:
          raise ValueError('Trace integration is required for fresh execution')
        processed_trace = await self._trace_processor.process_trace(request_id)
        trace_id = processed_trace.trace.external_id
      evaluation_run_sample = await self._update_evaluation_run_sample(
        run_sample_id=evaluation_run_sample.id,
        status=EvaluationSampleStatus.RUNNING,
        trace_id=trace_id,
      )
    except asyncio.CancelledError:
      await self._mark_cancelled_sample(evaluation_run_sample.id, 'trace', trace_failure_phase)
      raise
    except asyncio.TimeoutError:
      error_msg = f'Trace phase timed out after {trace_timeout_seconds}s'
      logger.error(
        'Sample trace phase timed out sample_id=%s trace_timeout_seconds=%s',
        sample.id,
        trace_timeout_seconds,
      )
      evaluation_run_sample = await self._update_evaluation_run_sample(
        run_sample_id=evaluation_run_sample.id,
        status=EvaluationSampleStatus.FAILED,
        ended_at=datetime.now(timezone.utc),
        error_message=error_msg,
        metadata={'failure_phase': trace_failure_phase},
      )
      return SampleExecutionResult(
        evaluation_run_sample=evaluation_run_sample,
        computations=[],
        error=error_msg,
      )
    except Exception as exc:
      error_msg = str(exc)
      logger.error(
        'Sample trace phase failed sample_id=%s: %s',
        sample.id,
        error_msg,
        exc_info=True,
      )
      evaluation_run_sample = await self._update_evaluation_run_sample(
        run_sample_id=evaluation_run_sample.id,
        status=EvaluationSampleStatus.FAILED,
        ended_at=datetime.now(timezone.utc),
        error_message=error_msg,
        metadata={'failure_phase': trace_failure_phase},
      )
      return SampleExecutionResult(
        evaluation_run_sample=evaluation_run_sample,
        computations=[],
        error=error_msg,
      )

    return await self._run_compute_phase(evaluation_run_sample, sample.id, trace_id, compute_timeout_seconds)

  async def recompute(
    self,
    evaluation_run_id: UUID,
    sample_id: UUID,
    trace_id: str,
    compute_timeout_seconds: float | None = None,
  ) -> SampleExecutionResult:
    if compute_timeout_seconds is not None and compute_timeout_seconds <= 0:
      raise ValueError(f'compute_timeout_seconds must be > 0 when set, got {compute_timeout_seconds}')

    evaluation_run_sample = await self._create_evaluation_run_sample(
      evaluation_run_id=evaluation_run_id,
      sample_id=sample_id,
      trace_id=trace_id,
    )
    return await self._run_compute_phase(evaluation_run_sample, sample_id, trace_id, compute_timeout_seconds)

  async def _execute_pipeline(
    self,
    evaluation_run_sample_id: UUID,
    sample_id: UUID,
    trace_id: str,
  ) -> list[MetricComputation]:
    return await self._plan_executor.execute(
      plan_items=self._metric_planner.iter_plan_items(
        trace_id=trace_id,
        sample_id=sample_id,
      ),
      evaluation_run_sample_id=evaluation_run_sample_id,
    )

  async def _run_compute_phase(
    self,
    evaluation_run_sample: EvaluationRunSample,
    sample_id: UUID,
    trace_id: str,
    compute_timeout_seconds: float | None,
  ) -> SampleExecutionResult:
    try:
      async with asyncio.timeout(compute_timeout_seconds):
        computations = await self._execute_pipeline(
          evaluation_run_sample_id=evaluation_run_sample.id,
          sample_id=sample_id,
          trace_id=trace_id,
        )

      logger.info(
        'Sample evaluation complete sample_id=%s trace_id=%s computations=%d',
        sample_id,
        trace_id,
        len(computations),
      )
      evaluation_run_sample = await self._update_evaluation_run_sample(
        run_sample_id=evaluation_run_sample.id,
        status=EvaluationSampleStatus.COMPLETED,
        ended_at=datetime.now(timezone.utc),
      )
      return SampleExecutionResult(evaluation_run_sample=evaluation_run_sample, computations=computations)

    except asyncio.CancelledError:
      await self._mark_cancelled_sample(evaluation_run_sample.id, 'compute', 'metric_compute')
      raise

    except asyncio.TimeoutError:
      error_msg = f'Compute phase timed out after {compute_timeout_seconds}s'
      logger.error(
        'Sample compute phase timed out sample_id=%s trace_id=%s compute_timeout_seconds=%s',
        sample_id,
        trace_id,
        compute_timeout_seconds,
      )
      evaluation_run_sample = await self._update_evaluation_run_sample(
        run_sample_id=evaluation_run_sample.id,
        status=EvaluationSampleStatus.FAILED,
        ended_at=datetime.now(timezone.utc),
        error_message=error_msg,
        metadata={'failure_phase': 'metric_compute'},
      )
      return SampleExecutionResult(evaluation_run_sample=evaluation_run_sample, computations=[], error=error_msg)

    except Exception as exc:
      error_msg = str(exc)
      logger.error(
        'Sample evaluation failed sample_id=%s trace_id=%s: %s',
        sample_id,
        trace_id,
        error_msg,
        exc_info=True,
      )
      evaluation_run_sample = await self._update_evaluation_run_sample(
        run_sample_id=evaluation_run_sample.id,
        status=EvaluationSampleStatus.FAILED,
        ended_at=datetime.now(timezone.utc),
        error_message=error_msg,
        metadata={'failure_phase': 'metric_compute'},
      )
      return SampleExecutionResult(evaluation_run_sample=evaluation_run_sample, computations=[], error=error_msg)

  async def _mark_cancelled_sample(self, run_sample_id: UUID, phase: str, failure_phase: str) -> None:
    error_msg = f'Sample execution cancelled during {phase} phase'
    logger.warning(
      'Sample execution cancelled run_sample_id=%s phase=%s',
      run_sample_id,
      phase,
    )
    await asyncio.shield(
      self._update_evaluation_run_sample(
        run_sample_id=run_sample_id,
        status=EvaluationSampleStatus.FAILED,
        ended_at=datetime.now(timezone.utc),
        error_message=error_msg,
        metadata={'failure_phase': failure_phase},
      )
    )

  async def _create_evaluation_run_sample(
    self,
    evaluation_run_id: UUID,
    sample_id: UUID,
    trace_id: str | None = None,
  ) -> EvaluationRunSample:
    async with UnitOfWork(self._db_manager) as uow:
      return await uow.evaluation_run_samples.create(
        EvaluationRunSample(
          id=uuid4(),
          evaluation_run_id=evaluation_run_id,
          sample_id=sample_id,
          trace_id=trace_id,
          status=EvaluationSampleStatus.RUNNING,
          started_at=datetime.now(timezone.utc),
        )
      )

  async def _update_evaluation_run_sample(
    self,
    run_sample_id: UUID,
    status: EvaluationSampleStatus,
    trace_id: str | None = None,
    ended_at: datetime | None = None,
    error_message: str | None = None,
    metadata: dict[str, Any] | None = None,
  ) -> EvaluationRunSample:
    async with UnitOfWork(self._db_manager) as uow:
      return await uow.evaluation_run_samples.update_status(
        run_sample_id=run_sample_id,
        status=status,
        trace_id=trace_id,
        ended_at=ended_at,
        error_message=error_message,
        metadata=metadata,
      )
