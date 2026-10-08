import asyncio
import hashlib
import json
import os
import sys
import tempfile
import textwrap
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

from syllo_eval.evaluation.trace_import import ImportedTrace, match_traces_to_samples, normalize_imported_trace
from syllo_eval.model import Sample
from syllo_eval.settings import PhoenixSettings

from agents.tau2 import Tau2Caller, Tau2Settings, Tau2SimulationError, Tau2TraceAdapter, build_caller
from benchmarks.download import PinnedFileMismatchError
from benchmarks.tau2 import render_user_scenario
from config import ExperimentConfig
from indexing.embedding import AzureFoundrySettings
from run_outputs import RunOutputs

PROMPT = 'Instructions:\n\tDomain: retail\n\tReason for call:\n\t\tWhere is my order?'
TRACE_ID = 'a' * 32
T0 = datetime(2026, 10, 1, 12, 0, tzinfo=timezone.utc)


def _record(
  span_id: str,
  name: str,
  kind: str,
  parent: str | None,
  second: int,
  attributes: dict[str, Any] | None = None,
  *,
  error: str | None = None,
) -> dict[str, Any]:
  """A span as the Phoenix client returns it: attributes flattened under ``attributes.``."""
  record: dict[str, Any] = {
    'name': name,
    'span_kind': kind,
    'status_code': 'ERROR' if error else 'OK',
    'status_message': error or '',
    'parent_id': parent,
    'start_time': T0 + timedelta(seconds=second),
    'end_time': T0 + timedelta(seconds=second + 1),
    'context.span_id': span_id,
    'context.trace_id': TRACE_ID,
    'attributes.openinference.span.kind': kind,
  }
  record.update({f'attributes.{key}': value for key, value in (attributes or {}).items()})
  return record


def _llm(span_id: str, parent: str, second: int, tokens: int, cost: float | None = None) -> dict[str, Any]:
  attributes: dict[str, Any] = {
    'llm.model_name': 'claude-sonnet-5-5',
    'llm.provider': 'anthropic',
    'llm.token_count.prompt': tokens,
    'llm.token_count.completion': 10,
    'llm.token_count.total': tokens + 10,
  }
  if cost is not None:
    attributes['llm.cost.total'] = cost
  return _record(span_id, 'completion', 'LLM', parent, second, attributes)


def _tool(span_id: str, name: str, call_id: str, arguments: str, output: str, second: int, **kw: Any) -> dict[str, Any]:
  attributes = {'input.value': arguments, 'output.value': output, 'tool.name': name, 'tool.id': call_id}
  return _record(span_id, name, 'TOOL', 'turn1', second, attributes, **kw)


def _simulation() -> list[dict[str, Any]]:
  return [
    _record(
      'root',
      'tau2.simulation',
      'CHAIN',
      None,
      0,
      {'input.value': PROMPT, 'output.value': '{"termination_reason": "user_stop", "reward": 1.0}', 'request_id': 'r1'},
    ),
    _record('agent', 'tau2-llm-agent', 'AGENT', 'root', 0, {'input.value': 'Hi', 'output.value': 'It ships today.'}),
    _record('user1', 'user_turn', 'CHAIN', 'root', 1),
    _llm('user-llm', 'user1', 1, tokens=50),
    _record('turn1', 'agent_turn', 'CHAIN', 'agent', 2),
    _llm('llm1', 'turn1', 2, tokens=100, cost=0.0004),
    # litellm's retry: a nested call repeating its parent's usage.
    _llm('retry', 'llm1', 2, tokens=100, cost=0.0004),
    _tool(
      'tool1',
      'find_user_id_by_name_zip',
      'call_1',
      '{"zip": "19122", "first_name": "Yusuf", "last_name": "Rossi"}',
      'yusuf_rossi_9620',
      3,
    ),
    _tool(
      'tool2',
      'get_order_details',
      'call_2',
      '{"order_id": "#W0000000"}',
      'Error: Order not found',
      4,
      error='Error: Order not found',
    ),
    _record('turn2', 'agent_turn', 'CHAIN', 'agent', 5),
    # litellm reports 0 for a model it cannot price.
    _llm('llm2', 'turn2', 5, tokens=200, cost=0.0),
    _record('evaluation', 'evaluate_simulation', 'EVALUATOR', 'root', 7),
    _llm('grader-llm', 'evaluation', 7, tokens=300),
  ]


class Tau2TraceAdapterTest(unittest.TestCase):
  def setUp(self) -> None:
    self.result = Tau2TraceAdapter().normalize(TRACE_ID, _simulation())
    self.spans = {span.external_id: span for span in self.result.spans}

  def test_the_agent_span_is_the_root_and_the_simulation_and_its_llm_calls_get_their_own_types(self) -> None:
    self.assertEqual(self.result.trace.adapter, 'tau2:1')
    self.assertEqual(
      {span_id: span.span_type for span_id, span in self.spans.items()},
      {
        'root': 'simulation',
        'agent': 'agent_root',
        'user1': 'chain',
        'user-llm': 'customer_llm',
        'turn1': 'chain',
        'llm1': 'llm',
        'retry': 'llm_internal',
        'tool1': 'tool',
        'tool2': 'tool',
        'turn2': 'chain',
        'llm2': 'llm',
        'evaluation': 'evaluator',
        'grader-llm': 'grader_llm',
      },
    )

  def test_the_agent_root_requests_the_user_scenario_and_answers_with_its_last_message(self) -> None:
    semantics = self.spans['agent'].semantics
    self.assertEqual(semantics.request, PROMPT)
    assert semantics.answer is not None
    self.assertEqual(semantics.answer.text, 'It ships today.')

  def test_executed_steps_are_the_agent_tool_calls_with_arguments_rendered_as_the_expected_plan(self) -> None:
    planning = self.spans['agent'].semantics.planning
    assert planning is not None and planning.executed_steps is not None
    steps = [step.model_dump(exclude={'start_time', 'end_time'}) for step in planning.executed_steps]
    self.assertEqual(
      steps,
      [
        {
          'id': 'tool1',
          'operation': 'find_user_id_by_name_zip',
          'instruction': '{"first_name": "Yusuf", "last_name": "Rossi", "zip": "19122"}',
          'plan_id': None,
          'planned_step_id': None,
          'status': 'completed',
          'input': {'zip': '19122', 'first_name': 'Yusuf', 'last_name': 'Rossi'},
          'output': 'yusuf_rossi_9620',
          'span_ids': ['tool1'],
          'attributes': {'tool_call_id': 'call_1'},
        },
        {
          'id': 'tool2',
          'operation': 'get_order_details',
          'instruction': '{"order_id": "#W0000000"}',
          'plan_id': None,
          'planned_step_id': None,
          'status': 'error',
          'input': {'order_id': '#W0000000'},
          'output': 'Error: Order not found',
          'span_ids': ['tool2'],
          'attributes': {'tool_call_id': 'call_2'},
        },
      ],
    )

  def test_a_simulation_without_tool_calls_observed_no_steps(self) -> None:
    records = [record for record in _simulation() if record['span_kind'] != 'TOOL']

    [agent] = [span for span in Tau2TraceAdapter().normalize(TRACE_ID, records).spans if span.span_type == 'agent_root']

    assert agent.semantics.planning is not None
    self.assertEqual(agent.semantics.planning.executed_steps, [])

  def test_only_the_agent_own_llm_calls_carry_usage_with_their_cost(self) -> None:
    usages = {span_id: span.semantics.usage for span_id, span in self.spans.items() if span.semantics.usage}
    self.assertEqual(set(usages), {'llm1', 'llm2'})
    self.assertEqual((usages['llm1'].input_tokens, usages['llm1'].cost, usages['llm1'].currency), (100, 0.0004, 'USD'))
    self.assertEqual((usages['llm2'].input_tokens, usages['llm2'].cost), (200, None))

  def test_imports_bind_the_trace_to_the_sample_of_its_task(self) -> None:
    trace = ImportedTrace.model_validate_json(ImportedTrace(trace_id=TRACE_ID, spans=_simulation()).model_dump_json())
    sample = Sample(id=uuid4(), dataset_id=uuid4(), input_prompt=PROMPT)
    other = Sample(id=uuid4(), dataset_id=sample.dataset_id, input_prompt='Instructions:\n\tAnother task')

    result = normalize_imported_trace(Tau2TraceAdapter(), trace)

    self.assertEqual(match_traces_to_samples([result], [sample, other]), {sample.id: TRACE_ID})

  def test_rejects_a_trace_that_is_not_one_simulation(self) -> None:
    second_agent = _record('agent2', 'tau2-llm-agent', 'AGENT', 'root', 0)
    renamed_root = {**_simulation()[0], 'name': 'other'}
    for case, records, message in (
      ('two agents', [*_simulation(), second_agent], '2 agent spans'),
      ('another root', [renamed_root, *_simulation()[1:]], 'one root span'),
    ):
      with self.subTest(case=case), self.assertRaisesRegex(ValueError, message):
        Tau2TraceAdapter().normalize(TRACE_ID, records)


# Stands in for run_simulation.py: the spec's metadata says how each attempt ends.
_FAKE_RUNNER = textwrap.dedent(
  """
  import json, os, pathlib, sys, time

  arguments = dict(zip(sys.argv[1::2], sys.argv[2::2]))
  spec = json.loads(pathlib.Path(arguments['--spec']).read_text())
  outcome = spec['metadata']['outcomes'][spec['metadata']['attempt'] - 1]
  print('runner of', spec['task_id'], flush=True)
  pathlib.Path(spec['metadata']['pid_file']).write_text(str(os.getpid()))
  if outcome == 'hang':
    time.sleep(60)
  if outcome == 0:
    simulation = {
      'termination_reason': 'user_stop',
      'reward_info': {'reward': 1.0, 'db_check': {'db_match': True, 'db_reward': 1.0}},
      'agent_cost': 0.5,
      'user_cost': 0.1,
      'duration': 12.5,
    }
    pathlib.Path(arguments['--result']).write_text(json.dumps(simulation))
  sys.exit(outcome)
  """
)


class Tau2CallerTest(unittest.IsolatedAsyncioTestCase):
  def setUp(self) -> None:
    self._directory = tempfile.TemporaryDirectory()
    self.root = Path(self._directory.name)
    self.runner = self.root / 'fake_runner.py'
    self.runner.write_text(_FAKE_RUNNER, encoding='utf-8')
    self.outputs = RunOutputs(self.root / 'outputs')
    self.run_id = uuid4()
    self.outputs.bind(self.run_id)
    self.sample = Sample(id=uuid4(), dataset_id=uuid4(), input_prompt=PROMPT)

  def tearDown(self) -> None:
    self._directory.cleanup()

  def _caller(self, outcomes: list[int | str], max_attempts: int = 3) -> Tau2Caller:
    metadata = {'configuration': 'tau2/test', 'trial': 2, 'outcomes': outcomes, 'pid_file': str(self.root / 'pid')}
    return Tau2Caller(
      python=Path(sys.executable),
      spec={'seed': 373753, 'agent_name': 'tau2-llm-agent', 'metadata': metadata},
      task_ids={PROMPT: '7'},
      outputs=self.outputs,
      environment=dict(os.environ),
      max_attempts=max_attempts,
      retry_delay_seconds=0,
      runner=self.runner,
    )

  def _simulations(self) -> Path:
    return self.root / 'outputs' / 'tau2' / str(self.run_id) / 'simulations'

  def _rewards(self) -> list[dict[str, Any]]:
    path = self.root / 'outputs' / 'tau2' / str(self.run_id) / 'rewards.jsonl'
    return [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines()]

  async def test_runs_the_task_of_the_sample_and_records_its_reward(self) -> None:
    request_id = await self._caller([0]).call(self.sample)

    self.assertEqual(len(request_id), 32)
    spec = json.loads((self._simulations() / f'{request_id}.spec.json').read_text(encoding='utf-8'))
    self.assertEqual((spec['request_id'], spec['task_id'], spec['seed']), (request_id, '7', 373753))
    self.assertEqual((spec['metadata']['sample_id'], spec['metadata']['attempt']), (str(self.sample.id), 1))
    self.assertIn('runner of 7', (self._simulations() / f'{request_id}.log').read_text(encoding='utf-8'))
    [record] = self._rewards()
    self.assertEqual(
      {key: record[key] for key in ('run_id', 'sample_id', 'request_id', 'task_id', 'configuration', 'trial', 'seed')},
      {
        'run_id': str(self.run_id),
        'sample_id': str(self.sample.id),
        'request_id': request_id,
        'task_id': '7',
        'configuration': 'tau2/test',
        'trial': 2,
        'seed': 373753,
      },
    )
    self.assertEqual(
      (record['attempt'], record['termination_reason'], record['reward'], record['agent_cost'], record['duration']),
      (1, 'user_stop', 1.0, 0.5, 12.5),
    )
    self.assertEqual(record['reward_info']['db_check'], {'db_match': True, 'db_reward': 1.0})

  async def test_a_transient_failure_runs_a_new_simulation_with_a_new_request_id(self) -> None:
    with self.assertLogs('agents.tau2', 'WARNING'):
      request_id = await self._caller([75, 0]).call(self.sample)

    specs = sorted(self._simulations().glob('*.spec.json'))
    self.assertEqual(len(specs), 2)
    [record] = self._rewards()
    self.assertEqual((record['request_id'], record['attempt']), (request_id, 2))

  async def test_any_other_failure_fails_the_sample_at_once(self) -> None:
    with self.assertRaisesRegex(Tau2SimulationError, 'tau2 task 7 failed in attempt 1 with exit code 1; see .*log'):
      await self._caller([1, 0]).call(self.sample)

    self.assertEqual(len(list(self._simulations().glob('*.spec.json'))), 1)

  async def test_a_transient_failure_in_the_last_attempt_fails_the_sample(self) -> None:
    with self.assertLogs('agents.tau2', 'WARNING'), self.assertRaisesRegex(Tau2SimulationError, 'attempt 2'):
      await self._caller([75, 75, 0], max_attempts=2).call(self.sample)

    self.assertEqual(len(list(self._simulations().glob('*.spec.json'))), 2)

  async def test_a_prompt_that_is_no_task_fails_without_running(self) -> None:
    sample = Sample(id=uuid4(), dataset_id=uuid4(), input_prompt='What is the refund policy?')

    with self.assertRaisesRegex(ValueError, 'not the user scenario of a tau2 retail task'):
      await self._caller([0]).call(sample)

  async def test_cancelling_the_call_stops_the_simulation(self) -> None:
    call = asyncio.create_task(self._caller(['hang']).call(self.sample))
    pid_file = self.root / 'pid'
    for _ in range(200):
      if pid_file.exists() and pid_file.read_text():
        break
      await asyncio.sleep(0.05)
    call.cancel()

    with self.assertRaises(asyncio.CancelledError):
      await call
    with self.assertRaises(ProcessLookupError):
      os.kill(int(pid_file.read_text()), 0)


class BuildCallerTest(unittest.TestCase):
  def setUp(self) -> None:
    self._directory = tempfile.TemporaryDirectory()
    self.data_dir = Path(self._directory.name) / 'data'
    tasks = [
      {'id': '0', 'user_scenario': {'instructions': 'Return the lamp.'}},
      {'id': '1', 'user_scenario': {'persona': 'Calm.', 'instructions': 'Cancel order #W1.'}},
    ]
    contents = {
      'tasks': ('data/tau2/domains/retail/tasks.json', json.dumps(tasks)),
      'database': ('data/tau2/domains/retail/db.json', '{}'),
    }
    files = {}
    for role, (path, text) in contents.items():
      file = self.data_dir / 'tau2' / 'raw' / path
      file.parent.mkdir(parents=True, exist_ok=True)
      file.write_text(text, encoding='utf-8')
      files[role] = {'path': path, 'size': file.stat().st_size, 'sha256': hashlib.sha256(text.encode()).hexdigest()}
    self.config = ExperimentConfig.model_validate(
      {
        'models': {
          'agents': {'model-a': {'provider': 'anthropic', 'model': 'claude-a', 'label': 'A'}},
          'customer_simulator': {'provider': 'azure', 'model': 'gpt-c', 'label': 'C'},
          'judge': {
            'provider': 'gemini',
            'model': 'judge-1',
            'label': 'Judge',
            'temperature': 0.0,
            'timeout_seconds': 300,
            'max_retries': 5,
            'max_concurrent_requests': 4,
            'output_token_limit': 65536,
          },
          'embedding': {
            'provider': 'azure_foundry',
            'model': 'embed-1',
            'label': 'Embedder',
            'output_dimension': 1024,
            'timeout_seconds': 60,
            'max_retries': 3,
            'max_concurrent_requests': 2,
            'tokens_per_minute': 100_000,
            'max_request_tokens': 50_000,
          },
        },
        'configurations': [
          {'id': 'tau2/test/trial-2', 'benchmark': 'tau2', 'agent': 'tau2-llm-agent', 'model': 'model-a', 'trial': 2}
        ],
        'benchmarks': {
          'tau2': {
            'dataset_name': 'tasks-test',
            'expected_samples': 2,
            'base_url': f'https://example.org/tau2/{"a" * 40}',
            'files': files,
          }
        },
        'tau2': {
          'agent_llm_args': {'model-a': {'max_tokens': 100}},
          'customer_llm_args': {'temperature': 0.0, 'reasoning_effort': 'none'},
          'max_steps': 50,
          'max_errors': 5,
          'base_seed': 300,
          'cost_map_url': f'https://example.org/litellm/{"c" * 40}/model_prices.json',
        },
      }
    )
    self.configuration = self.config.configurations[0]
    self.phoenix = PhoenixSettings(base_url='http://phoenix:6006/', project_id='tasks-test')
    self.settings = Tau2Settings(python=Path(sys.executable))

  def tearDown(self) -> None:
    self._directory.cleanup()

  def _build(self, **overrides: Any) -> Tau2Caller:
    arguments: dict[str, Any] = {
      'data_dir': self.data_dir,
      'phoenix': self.phoenix,
      'outputs': RunOutputs(Path(self._directory.name) / 'outputs'),
      'settings': self.settings,
      'foundry': AzureFoundrySettings(base_url='https://resource.services.ai.azure.com', api_key='key'),
    }
    return build_caller(self.configuration, self.config, **{**arguments, **overrides})

  def test_the_spec_runs_the_configuration_models_on_its_trial_seed_and_sends_spans_to_the_project(self) -> None:
    caller = self._build()

    agent = {'model': 'anthropic/claude-a', 'llm_args': {'max_tokens': 100}}
    self.assertEqual(
      caller._spec,
      {
        'seed': 373753,
        'agent_name': 'tau2-llm-agent',
        'agent': agent,
        'customer': {'model': 'azure/gpt-c', 'llm_args': {'temperature': 0.0, 'reasoning_effort': 'none'}},
        'grader': agent,
        'max_steps': 50,
        'max_errors': 5,
        'otlp_endpoint': 'http://phoenix:6006/v1/traces',
        'project': 'tasks-test',
        'cost_map_url': f'https://example.org/litellm/{"c" * 40}/model_prices.json',
        'metadata': {'configuration': 'tau2/test/trial-2', 'trial': 2},
      },
    )
    self.assertEqual(
      caller._task_ids,
      {
        render_user_scenario({'instructions': 'Return the lamp.'}): '0',
        render_user_scenario({'persona': 'Calm.', 'instructions': 'Cancel order #W1.'}): '1',
      },
    )
    environment = caller._environment
    self.assertEqual(environment['TAU2_DATA_DIR'], str((self.data_dir / 'tau2' / 'raw' / 'data').resolve()))
    self.assertEqual(
      {key: environment[key] for key in environment if key.startswith(('AZURE_API_', 'AZURE_AI_API_'))},
      {
        'AZURE_API_BASE': 'https://resource.services.ai.azure.com',
        'AZURE_API_KEY': 'key',
        'AZURE_API_VERSION': '2025-04-01-preview',
        'AZURE_AI_API_BASE': 'https://resource.services.ai.azure.com/models',
        'AZURE_AI_API_KEY': 'key',
      },
    )

  def test_needs_the_runtime_and_the_pinned_files(self) -> None:
    with self.assertRaisesRegex(FileNotFoundError, 'run scripts/setup-tau2.sh'):
      self._build(settings=Tau2Settings(python=Path(self._directory.name) / 'missing' / 'python'))
    (self.data_dir / 'tau2' / 'raw' / 'data' / 'tau2' / 'domains' / 'retail' / 'db.json').write_text('{"x": 1}')
    with self.assertRaises(PinnedFileMismatchError):
      self._build()


if __name__ == '__main__':
  unittest.main()
