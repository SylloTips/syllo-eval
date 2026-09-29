import unittest
from syllo_eval.evaluation.metrics.test_support import span
from syllo_eval.evaluation.metric_support.agent_outputs import extract_final_answer, extract_actual_plan
from syllo_eval.trace_semantics import Answer, PlanningData, ExecutionStep


class CanonicalOutputTest(unittest.TestCase):
  def test_answer_and_execution_are_typed_and_independent_of_raw_payload(self):
    target = span(
      answer=Answer(text='Please clarify', kind='clarification'),
      planning=PlanningData(
        executed_steps=[ExecutionStep(id='attempt', operation='search', status='error', instruction='Find evidence')]
      ),
    )
    target.output_data = {'unrelated': 'payload'}
    self.assertEqual(extract_final_answer(target), 'Please clarify')
    self.assertIn('operation=search', extract_actual_plan(target))
    self.assertIn('Find evidence', extract_actual_plan(target))
