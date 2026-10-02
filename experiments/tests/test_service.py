import unittest
from typing import Any, cast

from syllo_eval.evaluation.metrics.implementations.plan.efficiency import PlanEfficiencyMetric
from syllo_eval.settings import Settings
from service import build_service


class BuildServiceTest(unittest.TestCase):
  def test_service_runs_only_the_metrics_it_is_given(self) -> None:
    settings = Settings.model_validate({'llm_judge': {'provider': 'gemini', 'gemini': {'api_key': 'test-key'}}})

    service = build_service(settings, cast(Any, object()), metrics=[PlanEfficiencyMetric()])

    # The judge provider is configured, yet no built-in judge metric is offered: the service never builds a judge.
    self.assertEqual(service.available_metric_names(), ['plan_efficiency'])


if __name__ == '__main__':
  unittest.main()
