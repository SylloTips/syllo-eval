"""Demo metric that scores spans by output length."""

from syllo_eval.evaluation.metrics.contracts import MetricComputationResult, SpanEvaluationMetric
from syllo_eval.model import MetricComputationStatus, GroundTruth, Span


class ResponseLengthMetric(SpanEvaluationMetric):
  """Simple metric: score equals the number of characters in span output."""

  metric_name = 'response_length'
  metric_description = 'Returns the character length of the canonical answer.'

  def input_skip_reason(self, span: Span) -> str | None:
    return 'Trace does not provide an answer.' if span.semantics.answer is None else None

  @property
  def requires_ground_truth(self) -> bool:
    return False

  @property
  def target_span_types(self) -> tuple[str, ...]:
    # Demo default: score the root agent output.
    return ('agent_root',)

  async def compute(self, span: Span, ground_truth: GroundTruth | None) -> MetricComputationResult:
    # Also support direct compute() calls that bypass the planner.
    if reason := self.input_skip_reason(span):
      return MetricComputationResult(score=None, status=MetricComputationStatus.SKIPPED, error_message=reason)
    del ground_truth
    assert span.semantics.answer is not None
    score = float(len(span.semantics.answer.text))
    return MetricComputationResult(
      score=score,
      reasoning=f'Output length is {int(score)} characters.',
      metadata={'output_length': int(score)},
    )
