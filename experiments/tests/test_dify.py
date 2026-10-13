import asyncio
import json
import unittest
from collections.abc import Callable, Sequence
from datetime import datetime, timedelta, timezone
from typing import Any
from uuid import uuid4

import httpx

from syllo_eval.model import Sample

from agents.dify import DifyCaller, DifyError, DifySettings, DifyTraceAdapter


def _stream(*events: dict[str, Any]) -> bytes:
  """Dify's server-sent events: one ``data:`` line per event, with pings in between."""
  lines = ['event: ping', '']
  for event in events:
    lines += [f'data: {json.dumps(event)}', '']
  return '\n'.join(lines).encode()


def _event(kind: str, **fields: Any) -> dict[str, Any]:
  return {'event': kind, 'message_id': 'm1', 'conversation_id': 'c1', 'task_id': 't1', **fields}


class DifyCallerTest(unittest.TestCase):
  def setUp(self) -> None:
    self.requests: list[httpx.Request] = []
    self.sample = Sample(id=uuid4(), dataset_id=uuid4(), input_prompt='Where is my order?')

  def _call(self, respond: Callable[[httpx.Request], httpx.Response], times: int = 1) -> list[str]:
    def handle(request: httpx.Request) -> httpx.Response:
      self.requests.append(request)
      return respond(request)

    caller = DifyCaller(
      DifySettings(base_url='http://dify.test/v1', api_key='app-key'), transport=httpx.MockTransport(handle)
    )

    async def call() -> list[str]:
      return [await caller.call(self.sample) for _ in range(times)]

    return asyncio.run(call())

  def _replying(self, *events: dict[str, Any]) -> Callable[[httpx.Request], httpx.Response]:
    return lambda request: httpx.Response(200, content=_stream(*events))

  def test_streams_a_new_conversation_to_its_end(self) -> None:
    [request_id] = self._call(
      self._replying(
        _event('agent_thought', thought='', tool='search'),
        _event('agent_message', answer='It ships tomorrow.'),
        _event('message_end', metadata={'usage': {'total_tokens': 12}}),
      )
    )
    [request] = self.requests
    self.assertEqual(request.url, 'http://dify.test/v1/chat-messages')
    self.assertEqual(request.headers['Authorization'], 'Bearer app-key')
    self.assertEqual(
      json.loads(request.content),
      {
        'query': 'Where is my order?',
        'inputs': {},
        'user': 'syllo-eval',
        'response_mode': 'streaming',
        'trace_id': request_id,
      },
    )

  def test_each_call_sends_a_new_request_id_as_trace_id(self) -> None:
    request_ids = self._call(self._replying(_event('message_end')), times=2)
    self.assertEqual(request_ids, [json.loads(request.content)['trace_id'] for request in self.requests])
    self.assertNotEqual(request_ids[0], request_ids[1])

  def test_failures_raise(self) -> None:
    cases: Sequence[tuple[Callable[[httpx.Request], httpx.Response], str]] = [
      (lambda request: httpx.Response(400, json={'code': 'invalid_param'}), 'Dify returned 400: .*invalid_param'),
      (self._replying(_event('error', status=500, message='model quota')), 'model quota'),
      (self._replying(_event('agent_message', answer='Half')), 'before the end of the message'),
    ]
    for respond, message in cases:
      with self.subTest(message), self.assertRaisesRegex(DifyError, message):
        self._call(respond)

  def test_needs_an_api_key(self) -> None:
    with self.assertRaisesRegex(ValueError, 'DIFY_API_KEY'):
      DifyCaller(DifySettings(api_key=None))


T0 = datetime(2026, 10, 1, 12, 0, tzinfo=timezone.utc)
SEARCH = {
  'call_id': 'call-1',
  'query': 'refund policy',
  'results': [
    {'rank': 1, 'id': 'doc-7', 'title': 'Refunds', 'text': 'Refunds take 5 days.'},
    {'rank': 2, 'id': 'doc-2', 'title': 'Returns', 'text': 'Returns are free.'},
  ],
}


def _node(span_id: str, name: str, kind: str, parent: str | None, start: float, end: float, **attributes: Any):
  return {
    'name': name,
    'status_code': 'OK',
    'parent_id': parent,
    'start_time': T0 + timedelta(seconds=start),
    'end_time': T0 + timedelta(seconds=end),
    'context.span_id': span_id,
    'attributes.openinference.span.kind': kind,
    **{f'attributes.{key}': json.dumps(v) if key.endswith('.value') else v for key, v in attributes.items()},
  }


PARSE_FAILED = {
  'id': 'parse',
  'parent_id': 'round',
  'label': 'Action parse failed',
  'status': 'error',
  'metadata': {},
  'data': {'error': 'Model output did not contain a valid Action or FinalAnswer.', 'model_output': '{"x": 1}'},
}


def _log_entry(
  entry_id: str, label: str, started: float, finished: float, parent: str = 'round', **data: Any
) -> dict[str, Any]:
  clock: dict[str, Any] = {'started_at': 5000 + started, 'finished_at': 5000 + finished}
  if label.endswith('Thought'):
    clock['provider'] = 'langgenius/azure_ai_studio/azure_ai_studio'
  return {'id': entry_id, 'parent_id': parent, 'label': label, 'status': 'success', 'metadata': clock, 'data': data}


def _dify_trace(question: str = 'How long do refunds take?') -> list[dict[str, Any]]:
  log = [
    _log_entry('round', 'ROUND 1', 0.0, 3.0, parent='', thought='Search first.'),
    _log_entry('thought', 'model Thought', 0.0, 1.0, action='search_knowledge_base'),
    _log_entry(
      'ok',
      'CALL search_knowledge_base',
      1.0,
      1.5,
      tool_name='search_knowledge_base',
      tool_call_args={'query': 'refund policy'},
      output=f'tool response: {json.dumps(SEARCH)}.',
    ),
    _log_entry(
      'fail',
      'CALL search_knowledge_base',
      2.0,
      2.5,
      tool_name='search_knowledge_base',
      tool_call_args={'query': ''},
      output='tool response: The search failed: empty query.',
    ),
    PARSE_FAILED,
    {'data': []},
  ]
  answer = {'answer': '5 days.'}
  return [
    # Dify exports the root last, after its workflow has ended.
    _node(
      'r',
      'run-1',
      'CHAIN',
      None,
      20,
      20,
      **{'dify_trace_id': 'req-1', 'input.value': {'sys.query': question}, 'output.value': answer},
    ),
    _node('wf', 'workflow_run-1', 'CHAIN', 'r', 0, 10, **{'output.value': answer}),
    _node(
      'agent',
      'agent_Agent',
      'AGENT',
      'wf',
      1,
      9,
      **{'metadata.node_type': 'agent', 'output.value': {'text': '5 days.', 'json': log}},
    ),
    _node('answer', 'answer_Answer', 'CHAIN', 'wf', 9, 9, **{'metadata.node_type': 'answer', 'output.value': answer}),
  ]


class DifyTraceAdapterTest(unittest.TestCase):
  def test_the_root_answers_and_the_agent_log_becomes_spans_under_the_agent_node(self) -> None:
    result = DifyTraceAdapter().normalize('d' * 32, _dify_trace())
    spans = {span.external_id: span for span in result.spans}
    self.assertEqual(set(spans), {'r', 'wf', 'agent', 'answer', 'round', 'thought', 'ok', 'fail'})
    self.assertEqual(spans['agent'].output_data, {'text': '5 days.'})
    self.assertEqual(spans['agent'].metadata['dify.agent_log.untimed'], [PARSE_FAILED])  # type: ignore[index]
    self.assertEqual((spans['round'].span_type, spans['round'].parent_span_id), ('chain', 'agent'))
    self.assertEqual(spans['round'].output_data, {'thought': 'Search first.'})
    self.assertEqual((spans['thought'].span_type, spans['thought'].parent_span_id), ('llm', 'round'))
    self.assertIsNone(spans['thought'].semantics.usage)

    root = spans['r']
    self.assertEqual(root.span_type, 'agent_root')
    self.assertEqual(root.semantics.request, 'How long do refunds take?')
    assert root.semantics.answer is not None
    self.assertEqual(root.semantics.answer.text, '5 days.')

    search = spans['ok']
    self.assertEqual((search.span_type, search.parent_span_id), ('retrieval', 'round'))
    self.assertEqual(search.semantics.request, 'How long do refunds take?')
    [retrieval] = search.semantics.retrieval
    self.assertEqual((retrieval.kind, retrieval.stage, retrieval.query), ('document', 'selected', 'refund policy'))
    self.assertEqual([item.id for item in retrieval.items], ['doc-7', 'doc-2'])
    self.assertEqual(retrieval.items[0].content, 'Refunds take 5 days.')
    self.assertEqual((search.start_time, search.end_time), (T0 + timedelta(seconds=2), T0 + timedelta(seconds=2.5)))

    self.assertEqual((spans['fail'].span_type, spans['fail'].semantics.retrieval), ('tool', []))
    self.assertEqual(result.trace.adapter, 'dify:1')

  def test_a_trace_without_the_request_id_or_the_question_is_refused(self) -> None:
    message_trace = _dify_trace()
    del message_trace[0]['attributes.dify_trace_id']
    with self.assertRaisesRegex(ValueError, 'dify_trace_id'):
      DifyTraceAdapter().normalize('d' * 32, message_trace)
    with self.assertRaisesRegex(ValueError, 'no question'):
      DifyTraceAdapter().normalize('d' * 32, _dify_trace(question=' '))


if __name__ == '__main__':
  unittest.main()
