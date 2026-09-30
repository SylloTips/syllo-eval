import unittest

from syllo_eval.evaluation.metrics.prompts import render_prompt


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


if __name__ == '__main__':
  unittest.main()
