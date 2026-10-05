"""tau2-bench retail: 114 customer-service tasks whose reference tool calls are the expected plan.

A sample's input prompt is the task's user scenario exactly as tau2 renders it for its user simulator: it identifies the
task uniquely and is what the agent's customer acts out. Each reference tool call becomes one plan step whose
instruction renders the call's arguments, because the plan judge shows only a step's operation, instruction and output.
The Phase 5 adapter must render executed tool calls with the same ``render_arguments``.
"""

import json
import textwrap
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from pydantic import JsonValue

from syllo_eval.datasets.model import DatasetJsonPlanStep, DatasetJsonSample

from benchmarks.common import ConvertedBenchmark, percentiles

_TAB = '\t'

# Retail tool types at the pinned revision (tau2's get_tool_types); a tool missing here means the pin has drifted.
TOOL_TYPES = {
  'find_user_id_by_name_zip': 'read',
  'find_user_id_by_email': 'read',
  'get_order_details': 'read',
  'get_product_details': 'read',
  'get_item_details': 'read',
  'get_user_details': 'read',
  'list_all_product_types': 'read',
  'cancel_pending_order': 'write',
  'exchange_delivered_order_items': 'write',
  'modify_pending_order_address': 'write',
  'modify_pending_order_items': 'write',
  'modify_pending_order_payment': 'write',
  'modify_user_address': 'write',
  'return_delivered_order_items': 'write',
  'calculate': 'generic',
  'transfer_to_human_agents': 'generic',
}


def render_user_scenario(user_scenario: Mapping[str, Any]) -> str:
  """``str(UserScenario)`` of tau2 at the pinned revision; blank lines stay unindented, as with ``textwrap.indent``."""
  lines = []
  if user_scenario.get('persona') is not None:
    lines += ['Persona:', textwrap.indent(user_scenario['persona'], _TAB)]
  instructions = user_scenario['instructions']
  if not isinstance(instructions, str):
    instructions = _render_instructions(instructions)
  lines += ['Instructions:', textwrap.indent(instructions, _TAB)]
  return '\n'.join(lines)


def render_arguments(arguments: Mapping[str, JsonValue]) -> str:
  """Arguments as sorted-key JSON: key order varies across reference calls, list order is meaningful."""
  return json.dumps(arguments, ensure_ascii=False, sort_keys=True)


def convert(tasks: Sequence[Mapping[str, Any]], splits: Mapping[str, Sequence[str]]) -> ConvertedBenchmark:
  """Convert the tasks in file order; every task must belong to the ``base`` split, and to train or test."""
  task_ids = [task['id'] for task in tasks]
  if sorted(task_ids) != sorted(splits['base']) or len(set(task_ids)) != len(task_ids):
    raise ValueError('Task ids must be unique and equal the "base" split')
  split_of = {task_id: name for name in ('train', 'test') for task_id in splits[name]}

  samples: list[DatasetJsonSample] = []
  records: list[dict[str, JsonValue]] = []
  for task in tasks:
    criteria = task['evaluation_criteria']
    actions = criteria.get('actions') or []
    plan = [
      DatasetJsonPlanStep(
        operation=action['name'], instruction=render_arguments(action['arguments']), parameters=action['arguments']
      )
      for action in actions
    ]
    samples.append(DatasetJsonSample(input_prompt=render_user_scenario(task['user_scenario']), plan=plan or None))
    tool_types = Counter(TOOL_TYPES[action['name']] for action in actions)
    records.append(
      {
        'sample_key': task['id'],
        'split': split_of[task['id']],
        'reward_basis': criteria.get('reward_basis'),
        'nl_assertions': criteria.get('nl_assertions'),
        'communicate_info': criteria.get('communicate_info'),
        # Steps are keyed by their index: action ids are irregular (gaps, off-by-one prefixes).
        'gold_actions': [
          {
            'index': index,
            'action_id': action['action_id'],
            'name': action['name'],
            'arguments': action['arguments'],
            # None compares every argument of the call, [] only the tool name.
            'compare_args': action.get('compare_args'),
          }
          for index, action in enumerate(actions)
        ],
        'gold_tool_types': {kind: tool_types[kind] for kind in ('read', 'write', 'generic')},
        'issues': [
          {'id': issue['id'], 'status': issue['status'], 'title': issue['title']} for issue in task.get('issues') or []
        ],
      }
    )

  plan_lengths = [len(sample.plan or []) for sample in samples]
  stats: dict[str, JsonValue] = {
    'samples_by_split': dict(Counter(str(record['split']) for record in records)),
    'plan_steps': sum(plan_lengths),
    'plan_steps_per_sample': percentiles(plan_lengths),
    'samples_without_plan': [str(record['sample_key']) for sample, record in zip(samples, records) if not sample.plan],
    # The reference lists only the writes for these tasks, with no authentication or lookups.
    'samples_with_writes_only_plan': [
      str(record['sample_key'])
      for sample, record in zip(samples, records)
      if sample.plan and all(TOOL_TYPES[step.operation] == 'write' for step in sample.plan)
    ],
    'operations': dict(Counter(step.operation for sample in samples for step in sample.plan or []).most_common()),
    'reward_basis': dict(Counter('+'.join(task['evaluation_criteria'].get('reward_basis') or []) for task in tasks)),
    # tau2 scores missing assertions as met, so an LLM judges the reward only where a task has assertions.
    'samples_with_nl_assertions': [
      str(record['sample_key'])
      for record in records
      if isinstance(record['nl_assertions'], list) and record['nl_assertions']
    ],
    'nl_assertions': sum(
      len(record['nl_assertions'] or []) for record in records if isinstance(record['nl_assertions'], list)
    ),
    'prompt_chars': percentiles([len(sample.input_prompt) for sample in samples]),
  }
  return ConvertedBenchmark(samples=samples, records=records, stats=stats)


def build(paths: Mapping[str, Path]) -> ConvertedBenchmark:
  tasks = json.loads(paths['tasks'].read_text(encoding='utf-8'))
  splits = json.loads(paths['splits'].read_text(encoding='utf-8'))
  return convert(tasks, splits)


def _render_instructions(instructions: Mapping[str, Any]) -> str:
  """``str(StructuredUserInstructions)`` of tau2 at the pinned revision."""
  lines = [
    f'Domain: {instructions["domain"]}',
    f'Reason for call:\n{textwrap.indent(instructions["reason_for_call"], _TAB)}',
  ]
  if instructions.get('known_info') is not None:
    lines.append(f'Known info:\n{textwrap.indent(instructions["known_info"], _TAB)}')
  if instructions.get('unknown_info') is not None:
    lines.append(f'Unknown info:\n{textwrap.indent(instructions["unknown_info"], _TAB)}')
  lines.append(f'Task instructions:\n{textwrap.indent(instructions["task_instructions"], _TAB)}')
  return '\n'.join(lines)
