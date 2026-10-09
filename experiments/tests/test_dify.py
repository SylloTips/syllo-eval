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


if __name__ == '__main__':
  unittest.main()
