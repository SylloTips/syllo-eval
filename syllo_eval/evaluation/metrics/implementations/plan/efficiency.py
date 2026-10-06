from collections.abc import Mapping

from syllo_eval.evaluation.metrics.contracts import MetricComputationResult, SpanEvaluationMetric
from syllo_eval.evaluation.metric_support.agent_outputs import extract_actual_plan_steps
from syllo_eval.model import MetricComputationStatus, GroundTruth, GroundTruthKey, Span


class PlanEfficiencyMetric(SpanEvaluationMetric):
  """Scores plan efficiency by comparing executed step count to the expected plan length."""

  metric_name = 'plan_efficiency'
  metric_description = 'Expected plan steps divided by observed execution steps, capped at 1.'

  def input_skip_reason(self, span: Span) -> str | None:
    return (
      'Trace does not provide executed planning steps.'
      if span.semantics.planning is None or span.semantics.planning.executed_steps is None
      else None
    )

  @property
  def target_span_types(self) -> tuple[str, ...]:
    return ('agent_root',)

  @property
  def ground_truth_keys(self) -> tuple[str, ...]:
    return (GroundTruthKey.EXPECTED_PLAN.value,)

  async def compute(self, span: Span, ground_truths: Mapping[str, GroundTruth]) -> MetricComputationResult:
    # Also support direct compute() calls that bypass the planner.
    if reason := self.input_skip_reason(span):
      return MetricComputationResult(score=None, status=MetricComputationStatus.SKIPPED, error_message=reason)
    expected_step_count = len(ground_truths[self.ground_truth_keys[0]].ground_truth_value['expected_plan'])
    actual_step_count = len(extract_actual_plan_steps(span))
    score = min(1.0, expected_step_count / actual_step_count) if actual_step_count else 0.0

    return MetricComputationResult(
      score=score,
      reasoning=f'Expected {expected_step_count} plan steps; executed {actual_step_count} plan steps.',
      metadata={
        'expected_step_count': expected_step_count,
        'actual_step_count': actual_step_count,
      },
    )
