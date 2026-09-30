from syllo_eval.evaluation.metric_support.agent_outputs import (
  display_text,
  extract_actual_plan,
  ground_truth_text,
)
from syllo_eval.evaluation.judge.metric_base import BaseLlmJudgeMetric
from syllo_eval.evaluation.metrics.prompts import render_prompt
from syllo_eval.model import GroundTruth, GroundTruthKey, Span


class PlanCorrectnessJudgeMetric(BaseLlmJudgeMetric):
  """Scores whether the agent plan correctly solves the user request."""

  metric_name = 'plan_correctness_judge'
  metric_description = (
    'Uses an LLM judge to score whether the agent plan correctly solves the user request, '
    'optionally informed by an expected plan.'
  )
  prompt_version = 'v1'

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
  def requires_ground_truth(self) -> bool:
    return False

  @property
  def ground_truth_key(self) -> str:
    return GroundTruthKey.EXPECTED_PLAN.value

  def build_system_prompt(self) -> str:
    return render_prompt(f'plan_correctness/{self.prompt_version}/system.md', rubric_addition=self._rubric_addition)

  def build_user_prompt(self, span: Span, ground_truth: GroundTruth | None) -> str:
    return render_prompt(
      f'plan_correctness/{self.prompt_version}/user.md',
      request=display_text(span.semantics.request),
      actual_plan=extract_actual_plan(span),
      expected_plan=ground_truth_text(ground_truth, 'expected_plan'),
      rubric=ground_truth_text(ground_truth, 'rubric'),
      notes=ground_truth_text(ground_truth, 'notes'),
    )
