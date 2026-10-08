import asyncio
import json
import unittest
from collections.abc import Callable, Sequence
from typing import Any
from uuid import uuid4

import httpx

from syllo_eval.model import Sample

from agents.dify import DifyCaller, DifyError, DifySettings


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

  def _record(self, respond: Callable[[httpx.Request], httpx.Response]) -> Callable[[httpx.Request], httpx.Response]:
    def handle(request: httpx.Request) -> httpx.Response:
      self.requests.append(request)
      return respond(request)

    return handle

  def _chat(self, respond: Callable[[httpx.Request], httpx.Response], query: str = 'Where is my order?') -> Any:
    async def chat() -> Any:
      transport = httpx.MockTransport(self._record(respond))
      async with httpx.AsyncClient(base_url='http://dify.test/v1', transport=transport) as client:
        return await DifyCaller(client, DifySettings(api_key='app-key')).chat(query)

    return asyncio.run(chat())

  def _replying(self, *events: dict[str, Any]) -> Callable[[httpx.Request], httpx.Response]:
    return lambda request: httpx.Response(200, content=_stream(*events))

  def test_streams_a_new_conversation_and_joins_the_answer(self) -> None:
    reply = self._chat(
      self._replying(
        _event('agent_thought', thought='', tool='search'),
        _event('agent_message', answer='It ships '),
        _event('agent_message', answer='tomorrow.'),
        _event('message_end', metadata={'usage': {'total_tokens': 12}}),
      )
    )
    self.assertEqual((reply.message_id, reply.conversation_id, reply.answer), ('m1', 'c1', 'It ships tomorrow.'))
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
        'trace_id': reply.request_id,
      },
    )

  def test_reads_chatflow_message_events(self) -> None:
    reply = self._chat(self._replying(_event('message', answer='Yes.'), _event('message_end')))
    self.assertEqual(reply.answer, 'Yes.')

  def test_call_returns_the_trace_id_it_sent_as_request_id(self) -> None:
    sample = Sample(id=uuid4(), dataset_id=uuid4(), input_prompt='Where is my order?')

    async def call() -> list[str]:
      transport = httpx.MockTransport(self._record(self._replying(_event('message_end'))))
      async with httpx.AsyncClient(base_url='http://dify.test/v1', transport=transport) as client:
        caller = DifyCaller(client, DifySettings(api_key='app-key'))
        return [await caller.call(sample), await caller.call(sample)]

    request_ids = asyncio.run(call())
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
        self._chat(respond)

  def test_needs_an_api_key(self) -> None:
    async def build() -> None:
      async with httpx.AsyncClient() as client:
        DifyCaller(client, DifySettings(api_key=None))

    with self.assertRaisesRegex(ValueError, 'DIFY_API_KEY'):
      asyncio.run(build())


if __name__ == '__main__':
  unittest.main()
