from collections.abc import Mapping, Sequence

from syllo_eval.evaluation.metrics.contracts import MetricComputationResult, SpanGroupEvaluationMetric
from syllo_eval.model import GroundTruth, Span


class LlmCallsMetric(SpanGroupEvaluationMetric):
  """Score equals the number of LLM spans in the group."""

  metric_name = 'llm_calls'
  metric_description = 'Returns the number of llm spans in the trace.'

  @property
  def target_span_types(self) -> tuple[str, ...]:
    return ('llm',)

  async def compute(self, spans: Sequence[Span], ground_truths: Mapping[str, GroundTruth]) -> MetricComputationResult:
    del ground_truths
    llm_call_count = len(spans)
    return MetricComputationResult(
      score=float(llm_call_count),
      reasoning=f'Found {llm_call_count} llm span(s) in the trace.',
      metadata={'llm_call_count': llm_call_count},
    )
