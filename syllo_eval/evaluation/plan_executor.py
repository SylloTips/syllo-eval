"""Plan execution for asynchronous metric computation."""

import asyncio
import logging
import time
from collections.abc import AsyncIterable
from uuid import UUID, uuid4

from syllo_eval.evaluation.metric_planner import MetricPlanItem
from syllo_eval.evaluation.metrics.contracts import MetricComputationResult
from syllo_eval.infrastructure.database import DatabaseManager
from syllo_eval.infrastructure.unit_of_work import UnitOfWork
from syllo_eval.model import MetricComputation, MetricComputationStatus

logger = logging.getLogger(__name__)


class PlanExecutor:
  """Executes metric plan items asynchronously and persists results."""

  def __init__(
    self,
    db_manager: DatabaseManager,
    max_concurrent_tasks: int = 10,
  ):
    self.db_manager = db_manager
    self._max_concurrent_tasks = max_concurrent_tasks

  async def execute(
    self,
    plan_items: AsyncIterable[MetricPlanItem],
    evaluation_run_sample_id: UUID,
  ) -> list[MetricComputation]:
    """
    Execute plan items concurrently and persist the results.

    Args:
        plan_items: Async stream of plan items to execute.
        evaluation_run_sample_id: The ID of the evaluation run sample these
            computations belong to.

    Returns:
        List of persisted metric computation records, in plan item order.
    """
    pending: dict[asyncio.Task[MetricComputation], tuple[int, MetricPlanItem]] = {}
    computations_by_index: dict[int, MetricComputation] = {}
    failures: list[BaseException] = []
    next_index = 0

    async def collect_completed(tasks: set[asyncio.Task[MetricComputation]]) -> None:
      for task in tasks:
        index, item = pending.pop(task)
        try:
          computations_by_index[index] = task.result()
        except BaseException as result:
          failures.append(result)
          logger.error(
            'Failed to persist plan item %s metric=%s: %s',
            item.log_target,
            item.metric_name,
            result,
          )

    try:
      async for item in plan_items:
        task = asyncio.create_task(self._execute_item(item, evaluation_run_sample_id))
        pending[task] = (next_index, item)
        next_index += 1
        if len(pending) >= self._max_concurrent_tasks:
          done, _ = await asyncio.wait(pending.keys(), return_when=asyncio.FIRST_COMPLETED)
          await collect_completed(done)

      if pending:
        done, _ = await asyncio.wait(pending.keys())
        await collect_completed(done)
    except BaseException:
      for task in pending:
        task.cancel()
      await asyncio.gather(*pending.keys(), return_exceptions=True)
      raise

    if failures:
      raise failures[0]

    return [computations_by_index[index] for index in sorted(computations_by_index)]

  async def _execute_item(
    self,
    item: MetricPlanItem,
    evaluation_run_sample_id: UUID,
  ) -> MetricComputation:
    """
    Execute a single plan item and persist the result.

    Args:
        item: The plan item to execute.
        evaluation_run_sample_id: The owning evaluation run sample ID.

    Returns:
        Persisted MetricComputation.
    """
    started = time.monotonic()
    try:
      computation_result = await self._compute_result(item)
      metric_computation = self._to_metric_computation(
        item=item,
        evaluation_run_sample_id=evaluation_run_sample_id,
        computation_result=computation_result,
      )
    except Exception as exc:
      logger.error(
        'Failed to execute plan item %s metric=%s: %s',
        item.log_target,
        item.metric_name,
        exc,
      )
      metric_computation = self._to_failed_metric_computation(
        item=item,
        evaluation_run_sample_id=evaluation_run_sample_id,
        error_message=str(exc),
      )
    if item.skip_reason is None:
      # Skipped items compute nothing; computed and failed items record how long the metric took.
      metric_computation.metadata = {
        **(metric_computation.metadata or {}),
        'latency_seconds': round(time.monotonic() - started, 3),
      }

    async with UnitOfWork(self.db_manager) as uow:
      record = await uow.metric_computations.create(metric_computation)

    logger.debug(
      'Computed metric=%s %s status=%s score=%s',
      item.metric_name,
      item.log_target,
      record.status.value,
      record.score,
    )
    return record

  @staticmethod
  async def _compute_result(item: MetricPlanItem) -> MetricComputationResult:
    if item.skip_reason is not None:
      return MetricComputationResult(
        score=None,
        status=MetricComputationStatus.SKIPPED,
        error_message=item.skip_reason,
      )

    return await item.metric.compute(item.target.compute_input, item.ground_truth)

  @staticmethod
  def _to_metric_computation(
    item: MetricPlanItem,
    evaluation_run_sample_id: UUID,
    computation_result: MetricComputationResult,
  ) -> MetricComputation:
    return MetricComputation(
      id=uuid4(),
      evaluation_run_sample_id=evaluation_run_sample_id,
      metric=item.metric_name,
      ground_truth_id=item.ground_truth.id if item.ground_truth is not None else None,
      targeting_mode=item.targeting_mode,
      target_span_type=item.target_span_type,
      span_ids=item.span_ids,
      score=computation_result.score,
      status=computation_result.status,
      reasoning=computation_result.reasoning,
      metadata=computation_result.metadata,
      error_message=computation_result.error_message,
      raw_output=computation_result.raw_output,
    )

  @staticmethod
  def _to_failed_metric_computation(
    item: MetricPlanItem,
    evaluation_run_sample_id: UUID,
    error_message: str,
  ) -> MetricComputation:
    return MetricComputation(
      id=uuid4(),
      evaluation_run_sample_id=evaluation_run_sample_id,
      metric=item.metric_name,
      ground_truth_id=item.ground_truth.id if item.ground_truth is not None else None,
      targeting_mode=item.targeting_mode,
      target_span_type=item.target_span_type,
      span_ids=item.span_ids,
      score=None,
      status=MetricComputationStatus.FAILED,
      error_message=error_message,
    )
