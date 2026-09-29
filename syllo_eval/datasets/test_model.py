import unittest

from syllo_eval.datasets.model import DatasetJsonPlanStep


class DatasetJsonPlanStepTest(unittest.TestCase):
  def test_accepts_custom_operations_and_rejects_unknown_fields(self):
    step = DatasetJsonPlanStep(
      operation='custom_analyze', instruction='Analyze', parameters={'limit': 3, 'tags': ['a']}
    )
    assert step.parameters is not None
    self.assertEqual(step.parameters['limit'], 3)
    for payload in [
      {'kind': 'search', 'instruction': 'Find'},
      {'operation': 'search', 'kind': 'search', 'instruction': 'Find'},
      {'operation': '', 'instruction': 'Find'},
    ]:
      with self.subTest(payload=payload), self.assertRaises(ValueError):
        DatasetJsonPlanStep.model_validate(payload)
