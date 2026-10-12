import unittest
from typing import Any, cast
from unittest.mock import MagicMock
from uuid import uuid4

from syllo_eval.evaluation.metrics.available import build_available_metrics
from syllo_eval.evaluation.metrics.implementations.plan.correctness_judge import PlanCorrectnessJudgeMetric
from syllo_eval.evaluation.metrics.prompts import render_prompt
from syllo_eval.evaluation.metrics.test_support import span
from syllo_eval.model import GroundTruth
from syllo_eval.trace_semantics import ExecutionStep, PlanningData


class RenderPromptTest(unittest.TestCase):
  def test_paragraphs_with_missing_values_are_omitted_and_values_are_not_templated(self) -> None:
    prompt = render_prompt(
      'answer_correctness/v1/user.md',
      request='Cost in $USD?',
      expected_answer='{"usd": 5}',
      actual_answer='$5',
      rubric=None,
      notes='Round up',
    )

    self.assertEqual(
      prompt,
      'Grade the actual answer against the expected answer.\n\n'
      'Sample input:\nCost in $USD?\n\n'
      'Expected answer:\n{"usd": 5}\n\n'
      'Actual answer:\n$5\n\n'
      'Ground-truth notes:\nRound up',
    )
    with self.assertRaises(KeyError):
      render_prompt('answer_correctness/v1/user.md', request='Q')

  def test_plan_judge_sees_step_inputs_status_outputs_and_expected_parameters(self) -> None:
    target = span(
      planning=PlanningData(
        executed_steps=[
          ExecutionStep(
            id='a', operation='return_items', status='error', input={'order_id': '#W1'}, output='Order not found'
          ),
          ExecutionStep(id='b', operation='respond'),
        ]
      ),
    )
    expected = GroundTruth(
      id=uuid4(),
      sample_id=uuid4(),
      key='expected_plan',
      ground_truth_value={
        'expected_plan': [{'operation': 'return_items', 'instruction': '', 'parameters': {'order_id': '#W2'}}]
      },
    )

    prompt = PlanCorrectnessJudgeMetric(judge_client=MagicMock()).build_user_prompt(target, {'expected_plan': expected})

    self.assertIn(
      'Step 1: operation=return_items\n'
      '  status: error\n'
      '  input: {"order_id": "#W1"}\n'
      '  output: Order not found\n'
      'Step 2: operation=respond\n\n',
      prompt,
    )
    self.assertTrue(prompt.endswith('Step 1: operation=return_items\n  parameters: {"order_id": "#W2"}'))

  def test_rubric_addition_closes_every_judge_system_prompt_without_replacing_it(self) -> None:
    plain = build_available_metrics(judge_client=MagicMock(), claim_extractor_client=MagicMock())
    judges = [metric for metric in plain if metric.requires_judge_client]
    stricter = build_available_metrics(
      judge_client=MagicMock(),
      claim_extractor_client=MagicMock(),
      rubric_additions={metric.name: 'Cite a source.' for metric in judges},
    )
    stricter_by_name = {metric.name: metric for metric in stricter}

    self.assertEqual(len(judges), 10)
    for metric in judges:
      prompt = cast(Any, metric).build_system_prompt()
      self.assertNotIn('Additional requirements', prompt)
      self.assertEqual(
        cast(Any, stricter_by_name[metric.name]).build_system_prompt(),
        f'{prompt}\n\nAdditional requirements (they apply on top of the criteria above and can only make your '
        'judgment stricter, never more lenient):\nCite a source.',
      )


if __name__ == '__main__':
  unittest.main()
