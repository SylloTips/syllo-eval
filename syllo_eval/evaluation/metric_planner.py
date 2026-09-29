"""Metric planning for metric execution."""

import logging
from abc import ABC, abstractmethod
from collections import defaultdict
from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import Any
from uuid import UUID

from syllo_eval.evaluation.metric_registry import MetricRegistry
from syllo_eval.evaluation.metrics.contracts import EvaluationMetric
from syllo_eval.infrastructure.database import DatabaseManager
from syllo_eval.infrastructure.unit_of_work import UnitOfWork
from syllo_eval.model import GroundTruth, MetricTargetingMode, Span

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class MetricTarget(ABC):
  """Span target selected for a metric computation."""

  target_span_type: str

  @property
  @abstractmethod
  def targeting_mode(self) -> MetricTargetingMode:
    raise NotImplementedError

  @property
  @abstractmethod
  def span_ids(self) -> list[str]:
    raise NotImplementedError

  @property
  @abstractmethod
  def log_target(self) -> str:
    raise NotImplementedError

  @property
  @abstractmethod
  def compute_input(self) -> Any:
    raise NotImplementedError


@dataclass(frozen=True, slots=True)
class SpanTarget(MetricTarget):
  """One span selected for a single-span metric."""

  span: Span

  @property
  def targeting_mode(self) -> MetricTargetingMode:
    return MetricTargetingMode.SINGLE

  @property
  def span_ids(self) -> list[str]:
    return [self.span.external_id]

  @property
  def log_target(self) -> str:
    return f'span_id={self.span.external_id}'

  @property
  def compute_input(self) -> Span:
    return self.span


@dataclass(frozen=True, slots=True)
class SpanGroupTarget(MetricTarget):
  """Ordered span group selected for a group metric."""

  spans: tuple[Span, ...]

  @property
  def targeting_mode(self) -> MetricTargetingMode:
    return MetricTargetingMode.GROUP

  @property
  def span_ids(self) -> list[str]:
    return [span.external_id for span in self.spans]

  @property
  def log_target(self) -> str:
    return f'span_ids={",".join(self.span_ids)}'

  @property
  def compute_input(self) -> tuple[Span, ...]:
    return self.spans


@dataclass(frozen=True, slots=True)
class MissingTarget(MetricTarget):
  """A metric target whose configured span type is absent from the trace."""

  target_mode: MetricTargetingMode

  @property
  def targeting_mode(self) -> MetricTargetingMode:
    return self.target_mode

  @property
  def span_ids(self) -> list[str]:
    return []

  @property
  def log_target(self) -> str:
    return f'span_type={self.target_span_type}'

  @property
  def compute_input(self) -> None:
    return None


@dataclass(frozen=True, slots=True)
class MetricPlanItem:
  """Metric paired with its target and any pre-compute skip reason."""

  metric: EvaluationMetric
  target: MetricTarget
  ground_truth: GroundTruth | None
  skip_reason: str | None = None

  @property
  def metric_name(self) -> str:
    return self.metric.name

  @property
  def target_span_type(self) -> str:
    return self.target.target_span_type

  @property
  def targeting_mode(self) -> MetricTargetingMode:
    return self.target.targeting_mode

  @property
  def span_ids(self) -> list[str]:
    return self.target.span_ids

  @property
  def log_target(self) -> str:
    return self.target.log_target


class MetricPlanner:
  """Builds metric plan items for all applicable spans and span groups in a trace."""

  def __init__(self, metric_registry: MetricRegistry, db_manager: DatabaseManager):
    self.metric_registry = metric_registry
    self.db_manager = db_manager

  async def iter_plan_items(self, trace_id: str, sample_id: UUID) -> AsyncIterator[MetricPlanItem]:
    """
    Yield metric plan items for all applicable span/metric pairs in a trace.

    Notes:
      - For metrics requiring ground truth, missing rows are skipped and logged.
      - For metrics not requiring ground truth, missing rows are allowed and yield ground_truth=None.
    """
    async with UnitOfWork(self.db_manager) as uow:
      spans = await uow.spans.list_by_trace(trace_id)
      ground_truths = await uow.ground_truths.list_by_sample(sample_id)

      ground_truth_by_key = {gt.key: gt for gt in ground_truths}
      warned_missing_required_metrics: set[str] = set()
      spans_by_span_type: dict[str, list[Span]] = defaultdict(list)

      for span in spans:
        spans_by_span_type[span.span_type].append(span)

      for metric in self.metric_registry.list_registered():
        normalized_metric_name = self._normalize_metric_name(metric.name)
        ground_truth = ground_truth_by_key.get(metric.ground_truth_key) if metric.ground_truth_key else None

        for span_type in metric.target_span_types:
          grouped_spans = [span for span in spans_by_span_type.get(span_type, []) if metric.matches_span(span)]
          targets: tuple[MetricTarget, ...]
          skip_reason = None
          if not grouped_spans:
            targets = (
              MissingTarget(
                target_span_type=span_type,
                target_mode=metric.targeting_mode,
              ),
            )
            skip_reason = f'No spans found for target span type "{span_type}".'
          else:
            targets = self._build_targets(metric, span_type, grouped_spans)
            if ground_truth is None and metric.requires_ground_truth:
              skip_reason = 'Missing required ground truth.'
              if normalized_metric_name not in warned_missing_required_metrics:
                logger.warning(
                  'Missing required ground truth for sample_id=%s, metric=%s. Skipping metric planning item.',
                  sample_id,
                  metric.name,
                )
                warned_missing_required_metrics.add(normalized_metric_name)

          for target in targets:
            yield MetricPlanItem(
              metric=metric,
              target=target,
              ground_truth=ground_truth,
              skip_reason=skip_reason or self._input_skip_reason(metric, target),
            )

  @staticmethod
  def _build_targets(metric: EvaluationMetric, span_type: str, grouped_spans: list[Span]) -> tuple[MetricTarget, ...]:
    ordered_spans = sorted(grouped_spans, key=lambda span: (span.start_time, span.external_id))
    if metric.targeting_mode == MetricTargetingMode.SINGLE:
      return tuple(SpanTarget(target_span_type=span_type, span=span) for span in ordered_spans)

    if metric.targeting_mode == MetricTargetingMode.GROUP:
      groups = metric.group_spans(ordered_spans)
      return tuple(SpanGroupTarget(target_span_type=span_type, spans=tuple(group)) for group in groups if group)

    raise ValueError(f'Unsupported targeting mode for metric={metric.name}: {metric.targeting_mode!r}')

  @staticmethod
  def _input_skip_reason(metric: EvaluationMetric, target: MetricTarget) -> str | None:
    if isinstance(target, MissingTarget):
      return None
    spans = (target.span,) if isinstance(target, SpanTarget) else target.compute_input
    return next((reason for span in spans if (reason := metric.input_skip_reason(span))), None)

  @staticmethod
  def _normalize_metric_name(metric_name: str) -> str:
    return metric_name.strip().lower()
