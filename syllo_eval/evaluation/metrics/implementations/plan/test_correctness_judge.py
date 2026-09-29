import unittest
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
