import unittest

from pydantic import ValidationError

from syllo_eval.evaluation.metrics.test_support import Judge, span, truth
from syllo_eval.evaluation.metrics.implementations.plan.correctness_judge import PlanCorrectnessJudgeMetric
from syllo_eval.trace_semantics import PlanningData, ExecutionStep


class PlanningJudgeTest(unittest.IsolatedAsyncioTestCase):
  async def test_renders_generic_expected_and_executed_operations(self):
    judge = Judge({'score': 1, 'reasoning': 'correct'})
    target = span(
      planning=PlanningData(executed_steps=[ExecutionStep(id='1', operation='search', instruction='Find evidence')])
    )
    metric = PlanCorrectnessJudgeMetric(judge_client=judge)
    result = await metric.compute(
      target, truth(expected_plan=[{'operation': 'search', 'instruction': 'Find evidence'}])
    )
    self.assertEqual(result.score, 1)
    self.assertIn('operation=search', judge.requests[0].user_prompt)
    self.assertIn('Find evidence', judge.requests[0].user_prompt)
    self.assertEqual(metric.ground_truth_skip_reason({}), 'Missing required ground truth: expected_plan.')

  def test_score_must_be_between_zero_and_one(self):
    payload = PlanCorrectnessJudgeMetric(judge_client=Judge()).response_model
    for score in (0.0, 0.7, 1.0):
      self.assertEqual(payload.model_validate({'score': score, 'reasoning': 'r'}).score, score)
    for score in (-0.1, 1.1, 7.0, float('nan')):
      with self.assertRaises(ValidationError):
        payload.model_validate({'score': score, 'reasoning': 'r'})
