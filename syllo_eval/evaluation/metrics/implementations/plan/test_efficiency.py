import unittest
from syllo_eval.evaluation.metrics.test_support import span, truth
from syllo_eval.evaluation.metrics.implementations.plan.efficiency import PlanEfficiencyMetric
from syllo_eval.trace_semantics import PlanningData, ExecutionStep
from syllo_eval.model import MetricComputationStatus


class EfficiencyTest(unittest.IsolatedAsyncioTestCase):
  async def test_caps_score_when_fewer_steps_are_executed_than_expected(self):
    target = span(
      planning=PlanningData(
        executed_steps=[ExecutionStep(id=str(i), operation='search', status='completed') for i in range(2)]
      )
    )
    expected = truth(expected_plan=[{'operation': 'search'}, {'operation': 'search'}, {'operation': 'answer'}])
    self.assertEqual((await PlanEfficiencyMetric().compute(target, expected)).score, 1.0)

  async def test_counts_observed_attempts_including_errors_without_agent_specific_rules(self):
    target = span(
      planning=PlanningData(
        executed_steps=[
          ExecutionStep.model_validate({'id': str(i), 'operation': 'search', 'status': status})
          for i, status in enumerate(['completed', 'unsuccessful', 'error'])
        ]
      )
    )
    metric = PlanEfficiencyMetric()
    expected = truth(expected_plan=[{'operation': 'search'}, {'operation': 'answer'}])
    self.assertEqual((await metric.compute(target, expected)).score, 2 / 3)
    self.assertEqual((await metric.compute(span(), expected)).status, MetricComputationStatus.SKIPPED)
