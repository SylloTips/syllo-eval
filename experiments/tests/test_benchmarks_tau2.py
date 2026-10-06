import json
import tempfile
import unittest
from pathlib import Path
from typing import Any

from benchmarks.common import check_benchmark
from benchmarks.tau2 import build, convert, render_user_scenario


def _scenario(**instructions: Any) -> dict[str, Any]:
  fields = {
    'domain': 'retail',
    'reason_for_call': 'You want to return a lamp.',
    'known_info': 'You are sam_lee_1234.',
    'unknown_info': None,
    'task_instructions': '.',
  }
  return {'persona': None, 'instructions': {**fields, **instructions}}


def _task(task_id: str, actions: list[dict[str, Any]], **criteria: Any) -> dict[str, Any]:
  return {
    'id': task_id,
    'description': {'purpose': None, 'relevant_policies': None, 'notes': None},
    'user_scenario': _scenario(reason_for_call=f'Reason {task_id}.'),
    'initial_state': None,
    'evaluation_criteria': {
      'actions': actions,
      'communicate_info': [],
      'nl_assertions': None,
      'reward_basis': ['DB', 'NL_ASSERTION'],
      **criteria,
    },
  }


LOOKUP = {'action_id': '0_0', 'name': 'get_order_details', 'arguments': {'order_id': '#W1'}, 'info': None}
RETURN = {
  'action_id': '0_1',
  'name': 'return_delivered_order_items',
  'arguments': {'payment_method_id': 'card_1', 'order_id': '#W1', 'item_ids': ['9', '3']},
  'info': None,
}
TRANSFER = {
  'action_id': '2_0',
  'name': 'transfer_to_human_agents',
  'arguments': {'summary': 'Needs help.'},
  'info': None,
  'compare_args': [],
}
TASKS = [
  _task('0', [LOOKUP, RETURN]),
  _task('1', [], reward_basis=['DB'], nl_assertions=[]),
  _task('2', [TRANSFER], nl_assertions=['Agent says a human will follow up.', 'Agent apologizes.']),
  _task('3', [{**RETURN, 'action_id': '3_0'}]),
]
SPLITS = {'base': ['3', '1', '0', '2'], 'train': ['0', '1'], 'test': ['2', '3']}


class RenderUserScenarioTest(unittest.TestCase):
  def test_matches_tau2s_rendering_with_tab_indents(self) -> None:
    scenario = _scenario(reason_for_call='Line one\n\nLine two', unknown_info='Your zip code.')

    self.assertEqual(
      render_user_scenario(scenario),
      'Instructions:\n'
      '\tDomain: retail\n'
      '\tReason for call:\n'
      '\t\tLine one\n'
      '\n'  # textwrap.indent leaves blank lines unindented, at both levels
      '\t\tLine two\n'
      '\tKnown info:\n'
      '\t\tYou are sam_lee_1234.\n'
      '\tUnknown info:\n'
      '\t\tYour zip code.\n'
      '\tTask instructions:\n'
      '\t\t.',
    )

  def test_omits_missing_sections_and_renders_a_persona_and_plain_instructions(self) -> None:
    rendered = render_user_scenario({'persona': 'Calm.\nPatient.', 'instructions': 'Ask for a refund.'})

    self.assertEqual(rendered, 'Persona:\n\tCalm.\n\tPatient.\nInstructions:\n\tAsk for a refund.')
    self.assertNotIn('Unknown info', render_user_scenario(_scenario()))


class ConvertTest(unittest.TestCase):
  def test_reference_tool_calls_become_the_expected_plan(self) -> None:
    benchmark = convert(TASKS, SPLITS)

    plan = benchmark.samples[0].plan
    assert plan is not None
    self.assertEqual([step.operation for step in plan], ['get_order_details', 'return_delivered_order_items'])
    self.assertEqual(plan[1].instruction, '')
    self.assertEqual(plan[1].parameters, RETURN['arguments'])
    self.assertIsNone(benchmark.samples[1].plan)
    self.assertEqual(benchmark.samples[0].input_prompt, render_user_scenario(TASKS[0]['user_scenario']))
    self.assertIsNone(benchmark.samples[0].ground_truth_output)

  def test_records_keep_grading_fields_splits_and_compare_args(self) -> None:
    records = convert(TASKS, SPLITS).records

    self.assertEqual([record['split'] for record in records], ['train', 'train', 'test', 'test'])
    self.assertEqual(records[1]['reward_basis'], ['DB'])
    gold_actions = records[2]['gold_actions']
    assert isinstance(gold_actions, list) and isinstance(gold_actions[0], dict)
    self.assertEqual(gold_actions[0]['compare_args'], [])
    first = records[0]['gold_actions']
    assert isinstance(first, list) and isinstance(first[0], dict)
    self.assertIsNone(first[0]['compare_args'])
    self.assertEqual(records[0]['gold_tool_types'], {'read': 1, 'write': 1, 'generic': 0})

  def test_reports_tasks_without_a_plan_and_with_writes_only_plans(self) -> None:
    benchmark = convert(TASKS, SPLITS)

    self.assertEqual(benchmark.stats['samples_without_plan'], ['1'])
    self.assertEqual(benchmark.stats['samples_with_writes_only_plan'], ['3'])
    self.assertEqual(benchmark.stats['reward_basis'], {'DB+NL_ASSERTION': 3, 'DB': 1})
    # NL_ASSERTION is in three reward bases, but only task 2 has assertions for the judge to grade.
    self.assertEqual(benchmark.stats['samples_with_nl_assertions'], ['2'])
    self.assertEqual(benchmark.stats['nl_assertions'], 2)
    report = check_benchmark(benchmark, expected_samples=4)
    self.assertTrue(report.passed, report.errors)
    self.assertEqual(report.warnings, [])

  def test_rejects_unknown_tools_and_tasks_outside_the_base_split(self) -> None:
    with self.assertRaises(KeyError):
      convert([_task('0', [{**LOOKUP, 'name': 'unknown_tool'}])], {'base': ['0'], 'train': ['0'], 'test': []})
    with self.assertRaisesRegex(ValueError, 'base'):
      convert(TASKS, {**SPLITS, 'base': ['0', '1']})


class BuildTest(unittest.TestCase):
  def test_reads_the_task_and_split_files(self) -> None:
    with tempfile.TemporaryDirectory() as directory:
      paths = {'tasks': Path(directory) / 'tasks.json', 'splits': Path(directory) / 'split_tasks.json'}
      paths['tasks'].write_text(json.dumps(TASKS), encoding='utf-8')
      paths['splits'].write_text(json.dumps(SPLITS), encoding='utf-8')

      self.assertEqual(len(build(paths).samples), 4)


if __name__ == '__main__':
  unittest.main()
