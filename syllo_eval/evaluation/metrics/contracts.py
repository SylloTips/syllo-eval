from abc import ABC, abstractmethod
from collections.abc import Sequence
from typing import Any, ClassVar

from pydantic import BaseModel

from syllo_eval.model import GroundTruth, MetricComputationStatus, MetricTargetingMode, Span


class MetricComputationResult(BaseModel):
  """Output produced by a metric computation."""

  score: float | None
  status: MetricComputationStatus = MetricComputationStatus.COMPLETED
  reasoning: str | None = None
  metadata: dict[str, Any] | None = None
  error_message: str | None = None
  raw_output: dict[str, Any] | None = None


class EvaluationMetric(ABC):
  """Common interface for all metric implementations.

  The class-level attributes are read *before* an instance exists: the runtime inspects
  ``requires_*_client`` to decide which provider clients a run needs, and those clients are
  constructor arguments. Declare them on the class (usually once on a shared base class), not as
  instance properties. Metrics declaring a dependency are constructed with it as a keyword
  argument of the same name, so a custom metric can reuse the configured client.
  """

  metric_name: ClassVar[str] = ''
  metric_description: ClassVar[str | None] = None
  requires_judge_client: ClassVar[bool] = False
  requires_claim_extractor_client: ClassVar[bool] = False

  @property
  def name(self) -> str:
    """Unique metric name used by persistence."""
    return self.metric_name

  @property
  def description(self) -> str | None:
    """Human-readable metric description."""
    return self.metric_description

  @property
  def requires_ground_truth(self) -> bool:
    """Whether this metric requires a ground-truth payload."""
    return True

  @property
  def ground_truth_key(self) -> str | None:
    """Ground-truth key (see GroundTruthKey) this metric consumes, or None when it uses none."""
    return None

  @property
  def target_span_types(self) -> tuple[str, ...]:
    """Span types this metric can evaluate."""
    return ()

  @property
  def targeting_mode(self) -> MetricTargetingMode:
    """How this metric targets spans for each configured span type."""
    return MetricTargetingMode.SINGLE

  def matches_span(self, span: Span) -> bool:
    """Filter targets within the declared span types using canonical semantic data."""
    return True

  def input_skip_reason(self, span: Span) -> str | None:
    """Describe unavailable required observations; observed empty values remain evaluable."""
    return None

  def group_spans(self, spans: Sequence[Span]) -> Sequence[Sequence[Span]]:
    """Override to partition GROUP targets, for example by execution step or query."""
    return (spans,)

  @abstractmethod
  async def compute(self, target: Any, ground_truth: GroundTruth | None) -> MetricComputationResult:
    """Compute a metric score for a span or span group, depending on targeting mode."""
    raise NotImplementedError


class SpanEvaluationMetric(EvaluationMetric, ABC):
  """Metric implementation that evaluates one span at a time."""

  @property
  def targeting_mode(self) -> MetricTargetingMode:
    return MetricTargetingMode.SINGLE

  @abstractmethod
  async def compute(self, span: Span, ground_truth: GroundTruth | None) -> MetricComputationResult:
    """Compute a metric score for a span-ground truth pair."""
    raise NotImplementedError


class SpanGroupEvaluationMetric(EvaluationMetric, ABC):
  """Metric implementation that evaluates one ordered span group at a time."""

  @property
  def targeting_mode(self) -> MetricTargetingMode:
    return MetricTargetingMode.GROUP

  @abstractmethod
  async def compute(self, spans: Sequence[Span], ground_truth: GroundTruth | None) -> MetricComputationResult:
    """Compute a metric score for an ordered span group and optional ground truth."""
    raise NotImplementedError
