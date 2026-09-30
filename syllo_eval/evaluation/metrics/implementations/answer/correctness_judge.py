from syllo_eval.evaluation.metric_support.agent_outputs import (
  display_text,
  extract_final_answer,
  ground_truth_text,
)
from syllo_eval.evaluation.judge.metric_base import BaseLlmJudgeMetric
from syllo_eval.model import GroundTruth, GroundTruthKey, Span


class AnswerCorrectnessJudgeMetric(BaseLlmJudgeMetric):
  """Scores whether the agent answer matches the expected answer."""

  metric_name = 'answer_correctness_judge'
  metric_description = 'Uses an LLM judge to score how well the agent answer matches the expected answer.'

  def input_skip_reason(self, span: Span) -> str | None:
    return 'Trace does not provide an answer.' if span.semantics.answer is None else None

  @property
  def target_span_types(self) -> tuple[str, ...]:
    return ('agent_root',)

  @property
  def ground_truth_key(self) -> str:
    return GroundTruthKey.EXPECTED_OUTPUT.value

  def build_system_prompt(self) -> str:
    return (
      'You are grading an agent answer against the expected answer for a benchmark sample. '
      'Return a score between 0.0 and 1.0 where 1.0 means fully correct, 0.0 means incorrect, '
      'irrelevant, contradictory, or missing. Focus on semantic correctness instead of exact wording, '
      'but penalize hallucinations and material omissions.\n\n'
      'Scoring guidance:\n'
      '- 1.0 = fully correct\n'
      '- 0.7 = mostly correct with only minor omissions\n'
      '- 0.4 = partially correct but misses important details\n'
      '- 0.0 = wrong, unsupported, contradictory, or no answer\n\n'
      'Return only the JSON object required by the schema.'
    )

  def build_user_prompt(self, span: Span, ground_truth: GroundTruth | None) -> str:
    expected_answer = self._extract_expected_answer(ground_truth)
    rubric = ground_truth_text(ground_truth, 'rubric')
    notes = ground_truth_text(ground_truth, 'notes')

    sections = [
      'Grade the actual answer against the expected answer.',
      f'Sample input:\n{display_text(span.semantics.request)}',
      f'Expected answer:\n{expected_answer}',
      f'Actual answer:\n{extract_final_answer(span)}',
    ]

    if rubric is not None:
      sections.append(f'Additional rubric:\n{rubric}')
    if notes is not None:
      sections.append(f'Ground-truth notes:\n{notes}')

    return '\n\n'.join(sections)

  @staticmethod
  def _extract_expected_answer(ground_truth: GroundTruth | None) -> str:
    if ground_truth is None:
      raise ValueError('Missing ground truth for answer correctness judge metric.')

    expected_answer = ground_truth_text(ground_truth, 'expected_output')
    if expected_answer is not None:
      return expected_answer

    raise ValueError('ground_truth_value must include expected_output.')
