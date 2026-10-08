"""Offline check of the spans of one simulation, with litellm's mock responses: no model is called.

Runs in the tau2 environment, on the fetched benchmark files (``syllo-exp benchmarks fetch --only tau2``):

  .venv-tau2/bin/python -m unittest discover -s tau2_runtime
"""

import json
import os
import unittest
from collections.abc import Sequence
from pathlib import Path
from typing import Any

os.environ.setdefault('LITELLM_LOCAL_MODEL_COST_MAP', 'True')
DATA_DIR = Path(
  os.environ.setdefault('TAU2_DATA_DIR', str(Path(__file__).parents[1] / 'data' / 'tau2' / 'raw' / 'data'))
)

from opentelemetry.sdk.trace import ReadableSpan  # noqa: E402
from opentelemetry.sdk.trace.export import SpanExportResult  # noqa: E402
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter  # noqa: E402

from run_simulation import EXIT_FAILED, EXIT_TRANSIENT, TransientError, _is_transient  # noqa: E402
from telemetry import CheckedExporter, configure_tracing  # noqa: E402

EXPORTER = InMemorySpanExporter()
PROVIDER = configure_tracing('test-project', EXPORTER)


def _tool_call(call_id: str, name: str, **arguments: Any) -> dict[str, Any]:
  return {'id': call_id, 'type': 'function', 'function': {'name': name, 'arguments': json.dumps(arguments)}}


@unittest.skipUnless((DATA_DIR / 'tau2' / 'domains' / 'retail' / 'tasks.json').exists(), 'tau2 files not fetched')
class TracedSimulationTest(unittest.TestCase):
  spans: Sequence[ReadableSpan]
  simulation: Any
  task_scenario: str

  @classmethod
  def setUpClass(cls) -> None:
    from tau2.data_model.simulation import TextRunConfig
    from tau2.orchestrator.orchestrator import Role
    from tau2.runner.build import build_text_orchestrator
    from tau2.runner.helpers import get_tasks

    from traced_simulation import TracedOrchestrator, configure_grader, run_traced_simulation

    agent_script: list[dict[str, Any]] = [
      {
        'mock_response': '',
        'mock_tool_calls': [
          _tool_call('call_1', 'find_user_id_by_name_zip', zip='19122', first_name='Yusuf', last_name='Rossi'),
          _tool_call('call_2', 'get_order_details', order_id='#W0000000'),  # no such order: the tool fails
        ],
      },
      {'mock_response': 'Your user id is yusuf_rossi_9620; order #W0000000 does not exist.'},
    ]
    customer_script: list[dict[str, Any]] = [
      {'mock_response': 'Hi, I am Yusuf Rossi, zip 19122. Where is order #W0000000?'},
      {'mock_response': 'Thanks. ###STOP###'},
    ]

    class ScriptedOrchestrator(TracedOrchestrator):
      def step(self) -> None:
        scripts: dict[Any, tuple[list[dict[str, Any]], Any]] = {
          Role.AGENT: (agent_script, self.agent),
          Role.USER: (customer_script, self.user),
        }
        script, participant = scripts.get(self.to_role, ([], None))
        if script:
          participant.llm_args.pop('mock_tool_calls', None)
          participant.llm_args.update(script.pop(0))
        super().step()

    # The grader answers in a Markdown fence, which tau2 alone could not parse.
    verdict = {'results': [{'expectedOutcome': 'Ten options', 'reasoning': 'Said so.', 'metExpectation': True}]}
    configure_grader('gpt-4.1', {'mock_response': f'```json\n{json.dumps(verdict)}\n```'})
    (task,) = get_tasks('retail', task_split_name='base', task_ids=['2'])
    config = TextRunConfig(
      domain='retail', agent='llm_agent', llm_agent='gpt-4.1', user='user_simulator', llm_user='gpt-4.1'
    )
    built = build_text_orchestrator(config, task, seed=300, simulation_id='request-1')
    tracer = PROVIDER.get_tracer('test')
    orchestrator = ScriptedOrchestrator.wrap(built, tracer=tracer, agent_name='tau2-llm-agent')
    cls.simulation = run_traced_simulation(
      orchestrator, tracer=tracer, request_id='request-1', metadata={'configuration': 'tau2/test', 'trial': 1}
    )
    cls.task_scenario = str(task.user_scenario)
    PROVIDER.force_flush()
    cls.spans = EXPORTER.get_finished_spans()

  def _named(self, name: str) -> list[ReadableSpan]:
    return [span for span in self.spans if span.name == name]

  def _parent(self, span: ReadableSpan) -> ReadableSpan | None:
    if span.parent is None:
      return None
    return next(other for other in self.spans if other.context.span_id == span.parent.span_id)

  def test_one_trace_whose_root_carries_the_request_id_scenario_and_reward(self) -> None:
    self.assertEqual(len({span.context.trace_id for span in self.spans}), 1)
    [root] = [span for span in self.spans if span.parent is None]
    attributes = dict(root.attributes or {})
    self.assertEqual((root.name, attributes['openinference.span.kind']), ('tau2.simulation', 'CHAIN'))
    self.assertEqual(attributes['request_id'], 'request-1')
    self.assertEqual(attributes['input.value'], self.task_scenario)
    metadata = json.loads(str(attributes['metadata']))
    self.assertEqual(
      {key: metadata[key] for key in ('task_id', 'seed', 'configuration', 'trial')},
      {'task_id': '2', 'seed': 300, 'configuration': 'tau2/test', 'trial': 1},
    )
    self.assertEqual(
      json.loads(str(attributes['output.value'])),
      {'termination_reason': 'user_stop', 'reward': self.simulation.reward_info.reward},
    )

  def test_the_agent_span_holds_its_turns_their_llm_calls_and_tool_calls(self) -> None:
    [agent] = self._named('tau2-llm-agent')
    self.assertEqual(dict(agent.attributes or {})['openinference.span.kind'], 'AGENT')
    self.assertEqual(self._parent(agent).name, 'tau2.simulation')  # type: ignore[union-attr]
    turns = self._named('agent_turn')
    self.assertEqual(len(turns), 2)
    self.assertTrue(all(self._parent(turn) is agent for turn in turns))
    tools = [span for span in self.spans if dict(span.attributes or {}).get('openinference.span.kind') == 'TOOL']
    self.assertEqual([span.name for span in tools], ['find_user_id_by_name_zip', 'get_order_details'])
    self.assertTrue(all(self._parent(span) is turns[0] for span in tools))
    found, missing = (dict(span.attributes or {}) for span in tools)
    self.assertEqual(found['tool.id'], 'call_1')
    self.assertEqual(
      json.loads(str(found['input.value'])), {'zip': '19122', 'first_name': 'Yusuf', 'last_name': 'Rossi'}
    )
    self.assertEqual(found['output.value'], 'yusuf_rossi_9620')
    self.assertEqual(json.loads(str(found['metadata'])), {'requestor': 'assistant', 'tool_type': 'read'})
    self.assertEqual(tools[1].status.description, 'Error: Order not found')
    llm_parents = {self._parent(span).name for span in self.spans if span.name == 'completion'}  # type: ignore[union-attr]
    self.assertEqual(llm_parents, {'agent_turn', 'user_turn', 'evaluate_simulation'})

  def test_the_grader_reply_in_a_fence_is_read(self) -> None:
    [check] = self.simulation.reward_info.nl_assertions
    self.assertEqual((check.nl_assertion, check.met, check.justification), ('Ten options', True, 'Said so.'))

  def test_the_customer_and_the_grader_stay_outside_the_agent(self) -> None:
    self.assertTrue(all(self._parent(span).name == 'tau2.simulation' for span in self._named('user_turn')))  # type: ignore[union-attr]
    [evaluation] = self._named('evaluate_simulation')
    self.assertEqual(self._parent(evaluation).name, 'tau2.simulation')  # type: ignore[union-attr]
    reward_info = json.loads(str(dict(evaluation.attributes or {})['output.value']))
    self.assertEqual(reward_info['reward'], self.simulation.reward_info.reward)


@unittest.skipUnless((DATA_DIR / 'tau2' / 'domains' / 'retail' / 'tasks.json').exists(), 'tau2 files not fetched')
class JsonContentTest(unittest.TestCase):
  def test_json_stays_as_it_is_and_other_replies_become_the_json_inside_them(self) -> None:
    from traced_simulation import json_content

    fence_in_a_string = json.dumps({'results': [{'reasoning': 'It wrote ```json``` itself.'}]})
    cases = {
      '{"results": []}': '{"results": []}',
      fence_in_a_string: fence_in_a_string,
      '```json\n{"results": []}\n```': '{"results": []}',
      'Here it is: {"results": []} Done.': '{"results": []}',
      'No verdict.': 'No verdict.',
    }
    for content, expected in cases.items():
      with self.subTest(content=content):
        self.assertEqual(json_content(content), expected)
    self.assertIsNone(json_content(None))


class CheckedExporterTest(unittest.TestCase):
  def test_remembers_a_batch_that_failed_to_export(self) -> None:
    class Failing(InMemorySpanExporter):
      def export(self, spans: Sequence[ReadableSpan]) -> SpanExportResult:
        return SpanExportResult.FAILURE

    working, failing = CheckedExporter(InMemorySpanExporter()), CheckedExporter(Failing())
    for exporter in (working, failing):
      exporter.export([])

    self.assertEqual((working.failed, failing.failed), (False, True))


class TransientTest(unittest.TestCase):
  def test_rate_limits_outages_and_an_unpinned_cost_map_are_transient(self) -> None:
    import litellm

    transient = [
      TransientError('cost map'),
      litellm.exceptions.RateLimitError('slow down', llm_provider='anthropic', model='m'),
      litellm.exceptions.InternalServerError('overloaded', llm_provider='anthropic', model='m'),
      litellm.exceptions.APIConnectionError('reset', llm_provider='azure', model='m'),
    ]
    permanent = [
      litellm.exceptions.BadRequestError('bad', llm_provider='azure', model='m'),
      litellm.exceptions.AuthenticationError('denied', llm_provider='azure', model='m'),
      ValueError('Expecting value'),
    ]
    self.assertEqual([_is_transient(error) for error in transient], [True] * len(transient))
    self.assertEqual([_is_transient(error) for error in permanent], [False] * len(permanent))
    self.assertNotEqual(EXIT_FAILED, EXIT_TRANSIENT)


if __name__ == '__main__':
  unittest.main()
