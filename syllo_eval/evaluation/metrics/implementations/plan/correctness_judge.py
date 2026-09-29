from syllo_eval.evaluation.metric_support.agent_outputs import (
  display_text,
  extract_actual_plan,
  ground_truth_text,
)
from syllo_eval.evaluation.judge.metric_base import BaseLlmJudgeMetric
from syllo_eval.model import GroundTruth, GroundTruthKey, Span


class PlanCorrectnessJudgeMetric(BaseLlmJudgeMetric):
  """Scores whether the agent plan correctly solves the user request."""

  metric_name = 'plan_correctness_judge'
  metric_description = (
    'Uses an LLM judge to score whether the agent plan correctly solves the user request, '
    'optionally informed by an expected plan.'
  )

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
    return (
      'You are grading the correctness of the plan executed by an agent for a benchmark sample. '
      'A plan is the ordered sequence of steps (actions with their inputs and outputs) '
      'the agent took to handle the input. '
      'Judge correctness only: do the steps actually solve the user request? '
      'A plan can be considered correct even if it differs significantly from the expected plan '
      '(when one is provided), as long as it reaches a valid solution. Multiple plans can be correct. '
      'Penalize hallucinated steps, unsupported actions, missing critical steps, contradictions with '
      'the request, and dead-ends that never produce an answer. '
      'Do NOT penalize inefficiency, redundancy, or excessive steps here — efficiency is graded by a '
      'separate metric. Focus solely on whether the plan reaches a valid solution to the request. '
      'Treat the expected plan (if provided) as one valid reference, not as the only acceptable plan. '
      'Return a single score in [0.0, 1.0]. '
      'Return only the JSON object required by the schema, '
      'and use the `reasoning` field to briefly explain your correctness assessment.'
    )

  def build_user_prompt(self, span: Span, ground_truth: GroundTruth | None) -> str:
    expected_plan = ground_truth_text(ground_truth, 'expected_plan')
    rubric = ground_truth_text(ground_truth, 'rubric')
    notes = ground_truth_text(ground_truth, 'notes')

    sections = [
      'Grade the correctness of the actual plan executed by the agent. '
      'Ignore efficiency concerns — only judge whether the plan validly solves the request.',
      f'Sample input:\n{display_text(span.semantics.request)}',
      f'Actual plan (ordered steps the agent took):\n{extract_actual_plan(span)}',
      'Scoring guidance (correctness only):\n'
      '- 1.0 = plan is fully correct: every necessary step is present and the plan reaches a valid solution\n'
      '- 0.7 = plan is mostly correct with small omissions or minor errors that do not prevent a valid solution\n'
      '- 0.4 = plan partially solves the request: missing important steps or containing notable errors\n'
      '- 0.0 = plan is wrong, contradictory, hallucinated, never produces an answer, or is empty\n'
      'Remember: a plan that differs from the expected plan can still score highly if it correctly '
      'solves the request.',
    ]

    if expected_plan is not None:
      sections.append('Expected plan (one valid reference — not the only acceptable solution):\n' + expected_plan)
    if rubric is not None:
      sections.append(f'Additional rubric:\n{rubric}')
    if notes is not None:
      sections.append(f'Ground-truth notes:\n{notes}')

    return '\n\n'.join(sections)
