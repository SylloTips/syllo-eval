"""Metric registry for loading and validating metric implementations."""

from uuid import uuid4

from syllo_eval.evaluation.metrics.contracts import EvaluationMetric
from syllo_eval.infrastructure.database import DatabaseManager
from syllo_eval.infrastructure.unit_of_work import TransactionalUnitOfWork, UnitOfWork
from syllo_eval.model import Metric as MetricModel
from syllo_eval.model import MetricTargetSpanType
from syllo_eval.model import SpanType


class MetricRegistry:
  """
  Registry that resolves metric implementations and syncs persistence metadata.
  """

  def __init__(self, db_manager: DatabaseManager):
    self.db_manager = db_manager
    self._implementations: dict[str, EvaluationMetric] = {}

  def register(self, metric: EvaluationMetric) -> None:
    """
    Register a metric implementation in memory.

    Raises:
        ValueError: If another implementation already uses the same metric name.
    """
    key = self._normalize(metric.name)
    existing = self._implementations.get(key)
    if existing is not None and existing is not metric:
      raise ValueError(f'Metric {metric.name!r} is already registered')

    self._implementations[key] = metric

  def register_many(self, metrics: list[EvaluationMetric]) -> None:
    """Register multiple metrics."""
    for metric in metrics:
      self.register(metric)

  def get(self, metric_name: str) -> EvaluationMetric:
    """
    Get a metric implementation by name.

    Raises:
        KeyError: If no implementation is registered for metric_name.
    """
    key = self._normalize(metric_name)
    metric = self._implementations.get(key)
    if metric is None:
      raise KeyError(f'No metric implementation registered for {metric_name!r}')
    return metric

  def list_registered(self) -> list[EvaluationMetric]:
    """Return all registered metric implementations in registration order."""
    return list(self._implementations.values())

  async def sync_with_persistence(self) -> None:
    """
    Persist registered metric definitions and metric-span_type mappings.

    - Ensures each registered metric exists in `metric`
    - Updates metric description when a non-null value is provided
    - Updates metric ground_truth_keys when changed
    - Ensures each metric/span_type target mapping exists in `metric_target_span_type`
      with the correct targeting mode
    """
    async with TransactionalUnitOfWork(self.db_manager) as uow:
      for metric in self._implementations.values():
        await self._create_or_update_metric(uow, metric)
        await self._create_or_update_metric_target_span_types(uow, metric)

  async def _create_or_update_metric(self, uow: UnitOfWork, metric: EvaluationMetric) -> None:
    await uow.metrics.upsert_from_registry(
      MetricModel(
        name=metric.name,
        description=metric.description,
        ground_truth_keys=list(metric.ground_truth_keys),
      )
    )

  async def _create_or_update_metric_target_span_types(self, uow: UnitOfWork, metric: EvaluationMetric) -> None:
    desired_span_types = set(metric.target_span_types)
    persisted_span_types = set(await uow.metric_target_span_types.get_span_types_for_metric(metric.name))

    for span_type in desired_span_types:
      await uow.span_types.upsert_from_registry(SpanType(name=span_type))
      await uow.metric_target_span_types.upsert_from_registry(
        MetricTargetSpanType(
          id=uuid4(),
          metric=metric.name,
          span_type=span_type,
          targeting_mode=metric.targeting_mode,
        )
      )

    stale_span_types = persisted_span_types - desired_span_types
    for span_type in stale_span_types:
      await uow.metric_target_span_types.delete_by_metric_and_span_type(metric.name, span_type)

  @staticmethod
  def _normalize(metric_name: str) -> str:
    return metric_name.strip().lower()
