from collections.abc import Mapping

from syllo_eval.evaluation.metric_support.agent_outputs import (
  display_text,
  extract_final_answer,
  ground_truth_text,
)
from syllo_eval.evaluation.judge.metric_base import BaseLlmJudgeMetric
from syllo_eval.evaluation.metrics.prompts import render_prompt
from syllo_eval.model import GroundTruth, GroundTruthKey, Span


class AnswerCorrectnessJudgeMetric(BaseLlmJudgeMetric):
  """Scores whether the agent answer matches the expected answer."""

  metric_name = 'answer_correctness_judge'
  metric_description = 'Uses an LLM judge to score how well the agent answer matches the expected answer.'
  prompt_version = 'v1'

  def input_skip_reason(self, span: Span) -> str | None:
    return 'Trace does not provide an answer.' if span.semantics.answer is None else None

  @property
  def target_span_types(self) -> tuple[str, ...]:
    return ('agent_root',)

  @property
  def ground_truth_keys(self) -> tuple[str, ...]:
    return (GroundTruthKey.EXPECTED_OUTPUT.value,)

  def build_system_prompt(self) -> str:
    return render_prompt(f'answer_correctness/{self.prompt_version}/system.md', rubric_addition=self._rubric_addition)

  def build_user_prompt(self, span: Span, ground_truths: Mapping[str, GroundTruth]) -> str:
    ground_truth = ground_truths.get(self.ground_truth_keys[0])
    return render_prompt(
      f'answer_correctness/{self.prompt_version}/user.md',
      expected_answer=self._extract_expected_answer(ground_truth),
      request=display_text(span.semantics.request),
      actual_answer=extract_final_answer(span),
      rubric=ground_truth_text(ground_truth, 'rubric'),
      notes=ground_truth_text(ground_truth, 'notes'),
    )

  @staticmethod
  def _extract_expected_answer(ground_truth: GroundTruth | None) -> str:
    if ground_truth is None:
      raise ValueError('Missing ground truth for answer correctness judge metric.')

    expected_answer = ground_truth_text(ground_truth, 'expected_output')
    if expected_answer is not None:
      return expected_answer

    raise ValueError('ground_truth_value must include expected_output.')
