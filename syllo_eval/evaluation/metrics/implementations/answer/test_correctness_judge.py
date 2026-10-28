import unittest

from pydantic import ValidationError

from syllo_eval.evaluation.metrics.test_support import Judge, span, truth
from syllo_eval.evaluation.metrics.implementations.answer.correctness_judge import (
  AnswerCorrectnessJudgeMetric,
  AnswerCorrectnessPayload,
)
from syllo_eval.evaluation.metrics.implementations.demo.response_length import ResponseLengthMetric
from syllo_eval.trace_semantics import Answer
from syllo_eval.model import MetricComputationStatus


class AnswerMetricTest(unittest.IsolatedAsyncioTestCase):
  async def test_judge_uses_canonical_answer_and_missing_skips_without_call(self):
    judge = Judge({'score': 1.0, 'reasoning': 'correct'})
    metric = AnswerCorrectnessJudgeMetric(judge_client=judge)
    result = await metric.compute(span(answer=Answer(text='Canonical answer')), truth(expected_output='Expected'))
    self.assertEqual(result.score, 1.0)
    self.assertIn('Canonical answer', judge.requests[0].user_prompt)
    self.assertEqual(
      (await metric.compute(span(), truth(expected_output='Expected'))).status, MetricComputationStatus.SKIPPED
    )
    self.assertEqual(len(judge.requests), 1)

  def test_score_must_be_binary(self):
    for score in (0.0, 1.0):
      self.assertEqual(AnswerCorrectnessPayload.model_validate({'score': score, 'reasoning': 'r'}).score, score)
    for score in (0.5, 0.7, -1.0, 2.0):
      with self.assertRaises(ValidationError):
        AnswerCorrectnessPayload.model_validate({'score': score, 'reasoning': 'r'})

  async def test_empty_answer_length_is_zero_and_missing_is_skipped(self):
    metric = ResponseLengthMetric()
    self.assertEqual((await metric.compute(span(answer=Answer(text='')), {})).score, 0)
    self.assertEqual((await metric.compute(span(), {})).status, MetricComputationStatus.SKIPPED)
