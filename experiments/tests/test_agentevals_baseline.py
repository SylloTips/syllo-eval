import asyncio
import json
import unittest
from typing import Any, Literal, cast
from unittest.mock import MagicMock

import httpx
from langchain_google_genai import ChatGoogleGenerativeAI

from syllo_eval.evaluation.metric_support.agent_outputs import extract_actual_plan
from syllo_eval.evaluation.metrics.implementations.plan.correctness_judge import PlanCorrectnessJudgeMetric
from syllo_eval.model import GroundTruth, GroundTruthKey, MetricComputationStatus, Span
from syllo_eval.trace_semantics import ExecutionStep, PlanningData

from agentevals_baseline import AGENTEVALS_VERSION, AgentEvalsTrajectoryAccuracy, agent_trajectory, reference_trajectory
from agents.tau2 import Tau2TraceAdapter
from judge_fakes import FakeChatModel, Unparsed, prompt_text
from metric_fakes import ground_truths, span, truth
from openevals_baseline import OpenEvalsJudge
from test_agents_tau2 import PROMPT, TRACE_ID, _simulation

OUTPUT_LIMIT = 1_000
SCENARIO = 'You are Sam. Cancel order #W1, which you no longer need.'
EXPECTED_PLAN = [
  {'operation': 'find_user_id_by_email', 'instruction': '', 'parameters': {'email': 'sam@example.com'}},
  {
    'operation': 'cancel_pending_order',
    'instruction': '',
    'parameters': {'order_id': '#W1', 'reason': 'no longer needed'},
  },
]


def judge_of(chat: FakeChatModel, *, max_attempts: int = 3) -> OpenEvalsJudge:
  return OpenEvalsJudge(
    cast(ChatGoogleGenerativeAI, chat), temperature=0.0, max_concurrent_requests=2, max_attempts=max_attempts
  )


def verdict(accurate: bool, reasoning: str = 'Thus, the score should be: true.') -> dict[str, Any]:
  return {'reasoning': reasoning, 'score': accurate}


def step(
  step_id: str,
  operation: str,
  arguments: Any,
  output: Any = None,
  *,
  status: Literal['completed', 'error'] = 'completed',
  **fields: Any,
) -> ExecutionStep:
  return ExecutionStep(id=step_id, operation=operation, status=status, input=arguments, output=output, **fields)


def plan_span(*steps: ExecutionStep, request: str | None = SCENARIO, external_id: str = 'agent') -> Span:
  return span(
    external_id, 'agent_root', parent=None, request=request, planning=PlanningData(executed_steps=list(steps))
  )


def trajectory_span(external_id: str = 'agent') -> Span:
  return plan_span(
    step(
      's1',
      'find_user_id_by_email',
      {'email': 'sam@example.com'},
      'sam_1',
      attributes={'tool_call_id': f'call-{external_id}'},
    ),
    step('s2', 'cancel_pending_order', {'order_id': '#W1', 'reason': 'no longer needed'}, '{"status": "cancelled"}'),
    external_id=external_id,
  )


def plan_truth() -> dict[str, GroundTruth]:
  return ground_truths(truth(GroundTruthKey.EXPECTED_PLAN.value, expected_plan=EXPECTED_PLAN))


class TrajectoryTest(unittest.TestCase):
  def test_the_trajectory_is_the_request_then_each_step_call_and_its_result(self) -> None:
    target = plan_span(
      step('s1', 'get_order_details', {'order_id': '#W1'}, '{"status": "pending"}', attributes={'tool_call_id': 'c1'}),
      step('s2', 'calculate', {'expression': '2 + 2'}, {'total': 4, 'currency': '€'}, instruction='Add the totals.'),
      step('s3', 'transfer_to_human_agents', {'summary': 'Refund.'}),
      step('s4', 'list_all_product_types', {}, ''),
      step('s5', 'get_order_details', {'order_id': '#W2'}, 'Error: Order not found', status='error'),
      step('s6', 'get_user_details', {'user_id': 'sam'}, 'Sam', attributes={'tool_call_id': ''}),
    )

    self.assertEqual(
      agent_trajectory(target),
      [
        {'role': 'user', 'content': SCENARIO},
        _call('get_order_details', {'order_id': '#W1'}, call_id='c1'),
        {'role': 'tool', 'tool_call_id': 'c1', 'content': '{"status": "pending"}'},
        # A result that is not text is JSON as Plan Correctness renders it.
        _call('calculate', {'expression': '2 + 2'}, call_id='s2', content='Add the totals.'),
        {'role': 'tool', 'tool_call_id': 's2', 'content': '{"currency": "€", "total": 4}'},
        # A result the trace did not observe gets no tool message; an observed empty one does.
        _call('transfer_to_human_agents', {'summary': 'Refund.'}, call_id='s3'),
        _call('list_all_product_types', {}, call_id='s4'),
        {'role': 'tool', 'tool_call_id': 's4', 'content': ''},
        # A failed call stays, with the error the agent read; a missing or empty tool-call id gives the step's id.
        _call('get_order_details', {'order_id': '#W2'}, call_id='s5'),
        {'role': 'tool', 'tool_call_id': 's5', 'content': 'Error: Order not found'},
        _call('get_user_details', {'user_id': 'sam'}, call_id='s6'),
        {'role': 'tool', 'tool_call_id': 's6', 'content': 'Sam'},
      ],
    )

  def test_the_trajectory_of_a_tau2_trace_holds_every_call_of_the_agent(self) -> None:
    spans = Tau2TraceAdapter().normalize(TRACE_ID, _simulation()).spans
    [agent] = [span for span in spans if span.span_type == 'agent_root']

    self.assertEqual(
      agent_trajectory(agent),
      [
        {'role': 'user', 'content': PROMPT},
        _call(
          'find_user_id_by_name_zip', {'first_name': 'Yusuf', 'last_name': 'Rossi', 'zip': '19122'}, call_id='call_1'
        ),
        {'role': 'tool', 'tool_call_id': 'call_1', 'content': 'yusuf_rossi_9620'},
        _call('get_order_details', {'order_id': '#W0000000'}, call_id='call_2'),
        {'role': 'tool', 'tool_call_id': 'call_2', 'content': 'Error: Order not found'},
      ],
    )

  def test_arguments_are_json_with_sorted_keys_and_their_characters_kept(self) -> None:
    target = plan_span(step('s1', 'modify_user_address', {'zip': '80331', 'city': 'München'}, 'ok'))
    reference = reference_trajectory(
      truth(
        GroundTruthKey.EXPECTED_PLAN.value,
        expected_plan=[{'operation': 'modify_user_address', 'parameters': {'city': 'München', 'zip': '80331'}}],
      )
    )

    [_, call, _] = agent_trajectory(target)
    [expected] = reference
    for message in (call, expected):
      self.assertEqual(message['tool_calls'][0]['function']['arguments'], '{"city": "München", "zip": "80331"}')

  def test_a_trajectory_without_a_request_starts_with_its_first_call(self) -> None:
    target = plan_span(step('s1', 'get_user_details', {'user_id': 'sam'}, 'Sam'), request='  ')

    self.assertEqual(agent_trajectory(target)[0], _call('get_user_details', {'user_id': 'sam'}, call_id='s1'))

  def test_the_reference_is_one_assistant_call_per_expected_step(self) -> None:
    reference = reference_trajectory(plan_truth()[GroundTruthKey.EXPECTED_PLAN.value])

    self.assertEqual(
      reference,
      [
        _call('find_user_id_by_email', {'email': 'sam@example.com'}),
        _call('cancel_pending_order', {'order_id': '#W1', 'reason': 'no longer needed'}),
      ],
    )
    self.assertEqual(reference_trajectory(None), [])

  def test_arguments_that_are_not_an_object_are_refused(self) -> None:
    for arguments in (None, 'order #W1', ['#W1']):
      with self.subTest(arguments=arguments):
        with self.assertRaises(ValueError):
          agent_trajectory(plan_span(step('s1', 'get_order_details', arguments, 'pending')))
        with self.assertRaises(ValueError):
          reference_trajectory(
            truth(
              GroundTruthKey.EXPECTED_PLAN.value,
              expected_plan=[{'operation': 'get_order_details', 'parameters': arguments}],
            )
          )


def _call(name: str, arguments: dict[str, Any], *, call_id: str | None = None, content: str = '') -> dict[str, Any]:
  rendered = json.dumps(arguments, ensure_ascii=False, sort_keys=True)
  call: dict[str, Any] = {'type': 'function', 'function': {'name': name, 'arguments': rendered}}
  if call_id is not None:
    call['id'] = call_id
  return {'role': 'assistant', 'content': content, 'tool_calls': [call]}


class AgentEvalsTrajectoryAccuracyTest(unittest.IsolatedAsyncioTestCase):
  def metric(self, chat: FakeChatModel) -> AgentEvalsTrajectoryAccuracy:
    return AgentEvalsTrajectoryAccuracy(judge=judge_of(chat), output_token_limit=OUTPUT_LIMIT)

  async def test_grades_the_trajectory_against_the_expected_plan_in_one_call(self) -> None:
    for accurate, score in ((True, 1.0), (False, 0.0)):
      with self.subTest(accurate=accurate):
        chat = FakeChatModel(verdict(accurate, 'The trajectory cancels the order. Thus, the score should be: x.'))

        result = await self.metric(chat).compute(trajectory_span(), plan_truth())

        self.assertEqual(result.status, MetricComputationStatus.COMPLETED)
        self.assertEqual(result.score, score)
        self.assertEqual(result.reasoning, 'The trajectory cancels the order. Thus, the score should be: x.')
        assert result.metadata is not None
        self.assertEqual(result.metadata['judge_calls'], 1)
        self.assertEqual(result.metadata['judge_usage'], {'input_tokens': 100, 'output_tokens': 7, 'total_tokens': 107})

        [call] = chat.calls
        self.assertEqual(call['schema']['properties']['score']['type'], 'boolean')
        prompt = prompt_text(call['messages'])
        # AgentEvals' reference prompt, with the expected plan as the reference and the trace's steps as the trajectory.
        self.assertIn('Is semantically equivalent to the provided reference trajectory', prompt)
        reference, actual = prompt.split('Grade this actual trajectory:')
        self.assertIn('<name>cancel_pending_order</name>', reference)
        self.assertNotIn(SCENARIO, reference)
        self.assertIn(f'<user>\n{SCENARIO}\n</user>', actual)
        self.assertIn('<id>call-agent</id>\n<content>sam_1</content>', actual)
        self.assertIn('<id>s2</id>\n<content>{"status": "cancelled"}</content>', actual)

  async def test_the_judge_reads_arguments_as_plan_correctness_renders_them(self) -> None:
    chat = FakeChatModel(verdict(True))
    target = plan_span(step('s1', 'modify_user_address', {'zip': '80331', 'city': 'München'}, 'ok'))
    expected_plan = [{'operation': 'modify_user_address', 'parameters': {'zip': '80331', 'city': 'München'}}]

    await self.metric(chat).compute(target, ground_truths(truth('expected_plan', expected_plan=expected_plan)))

    reference, actual = prompt_text(chat.calls[0]['messages']).split('Grade this actual trajectory:')
    # LangChain parses and re-serializes each call's arguments, keeping their key order and their characters.
    for part in (reference, actual):
      self.assertIn('<arguments>{"city": "München", "zip": "80331"}</arguments>', part)
    self.assertIn('input: {"city": "München", "zip": "80331"}', extract_actual_plan(target))

  async def test_keeps_the_units_skip_rules_and_reference_of_plan_correctness(self) -> None:
    chat = FakeChatModel()
    metric = self.metric(chat)
    built_in = PlanCorrectnessJudgeMetric(judge_client=MagicMock())

    self.assertIsInstance(metric, PlanCorrectnessJudgeMetric)
    self.assertEqual(metric.name, 'agentevals_trajectory_accuracy')
    self.assertEqual(metric.target_span_types, built_in.target_span_types)
    self.assertEqual(metric.ground_truth_keys, built_in.ground_truth_keys)

    no_steps = span('agent', 'agent_root', parent=None, request=SCENARIO)
    result = await metric.compute(no_steps, plan_truth())

    self.assertEqual(result.status, MetricComputationStatus.SKIPPED)
    self.assertEqual(result.error_message, built_in.input_skip_reason(no_steps))
    self.assertEqual(chat.calls, [])

  async def test_failures_record_the_library_classes(self) -> None:
    cases = [
      (Unparsed(ValueError('bad JSON')), 7, 'invalid_output'),
      (Unparsed(ValueError('cut off')), OUTPUT_LIMIT, 'truncated'),
      ({'reasoning': 'No score.'}, 7, 'invalid_output'),
      (None, 7, 'invalid_output'),
      (httpx.ReadTimeout('slow'), 7, 'timeout'),
      (RuntimeError('500 internal'), 7, 'provider'),
    ]
    for reply, output_tokens, failure in cases:
      with self.subTest(reply=reply):
        chat = FakeChatModel(reply, output_tokens=output_tokens)

        result = await self.metric(chat).compute(trajectory_span(), plan_truth())

        self.assertEqual(result.status, MetricComputationStatus.FAILED)
        self.assertIsNone(result.score)
        assert result.metadata is not None
        self.assertEqual(result.metadata['failure'], failure)
        self.assertEqual((result.metadata['judge_calls'], result.metadata['judge_failed_calls']), (1, 1))
        # An output that came back still cost its tokens; a call that raised reported none.
        self.assertEqual('judge_usage' in result.metadata, not isinstance(reply, Exception))

  async def test_concurrent_units_each_report_their_own_calls(self) -> None:
    def reply(prompt: str) -> dict[str, Any]:
      return verdict('call-u1' in prompt)

    # Usage depends on the prompt, so a call recorded for the wrong unit shows in its usage.
    chat = FakeChatModel(reply, input_tokens=len)
    metric = self.metric(chat)
    units = [trajectory_span(f'u{index}') for index in range(4)]

    results = await asyncio.gather(*(metric.compute(unit, plan_truth()) for unit in units))

    self.assertEqual([result.score for result in results], [0.0, 1.0, 0.0, 0.0])
    for index, result in enumerate(results):
      with self.subTest(unit=index):
        [own] = [call for call in chat.calls if f'<id>call-u{index}</id>' in prompt_text(call['messages'])]
        assert result.metadata is not None
        self.assertEqual(result.metadata['judge_response_id'], own['id'])
        self.assertEqual(result.metadata['judge_usage']['input_tokens'], len(prompt_text(own['messages'])))
    # The units overlapped, up to the judge's limit of two calls in flight.
    self.assertEqual(chat.load['peak'], 2)


class AgentEvalsSetupTest(unittest.TestCase):
  def test_runs_the_pinned_version(self) -> None:
    self.assertEqual(AGENTEVALS_VERSION, '0.0.9')


if __name__ == '__main__':
  unittest.main()
