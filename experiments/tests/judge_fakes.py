"""A scripted chat model in place of the shared judge's, for the judges of the baseline frameworks."""

import asyncio
from collections.abc import Callable
from typing import Any
from unittest.mock import AsyncMock, MagicMock

from google.genai.errors import ClientError
from langchain_core.messages import AIMessage


class Unparsed:
  """A reply whose structured output did not parse."""

  def __init__(self, error: Exception):
    self.error = error


class FakeChatModel:
  """Stands in for the judge's chat model: records each structured-output call and answers from a script.

  A reply is the parsed output, ``None`` for no output, an ``Unparsed`` output, or an exception the call raises; a
  single function of the prompt answers every call. ``input_tokens`` is a count, or a function of the prompt. Copies
  share the script, the calls and ``load``, the peak number of calls in flight, as copies of a real chat model share
  its client.
  """

  model = 'judge-1'

  def __init__(self, *replies: Any, output_tokens: int = 7, input_tokens: int | Callable[[str], int] = 100):
    self.replies = list(replies)
    self.output_tokens = output_tokens
    self.input_tokens = input_tokens
    self.options: dict[str, Any] = {}
    self.calls: list[dict[str, Any]] = []
    self.load = {'in_flight': 0, 'peak': 0}
    self.async_client = MagicMock(aclose=AsyncMock())

  def model_copy(self, *, update: dict[str, Any]) -> 'FakeChatModel':
    copy = FakeChatModel.__new__(FakeChatModel)
    copy.__dict__.update(self.__dict__)
    copy.options = {**self.options, **update}
    return copy

  def with_structured_output(self, *, schema: Any, method: str, include_raw: bool) -> '_Runnable':
    return _Runnable(self, {'schema': schema, 'method': method, 'include_raw': include_raw, 'options': self.options})


class _Runnable:
  def __init__(self, chat: FakeChatModel, call: dict[str, Any]):
    self._chat = chat
    self._call = call

  async def ainvoke(self, messages: list[Any]) -> dict[str, Any]:
    chat = self._chat
    prompt = prompt_text(messages)
    call_id = f'response-{len(chat.calls) + 1}'
    chat.calls.append({**self._call, 'messages': messages, 'id': call_id})
    chat.load['in_flight'] += 1
    chat.load['peak'] = max(chat.load['peak'], chat.load['in_flight'])
    try:
      await _pause()
    finally:
      chat.load['in_flight'] -= 1
    reply = chat.replies[0](prompt) if len(chat.replies) == 1 and callable(chat.replies[0]) else chat.replies.pop(0)
    if isinstance(reply, Exception):
      raise reply
    input_tokens = chat.input_tokens(prompt) if callable(chat.input_tokens) else chat.input_tokens
    raw = AIMessage(
      content='',
      id=call_id,
      response_metadata={'model_name': 'judge-1-001'},
      usage_metadata={
        'input_tokens': input_tokens,
        'output_tokens': chat.output_tokens,
        'total_tokens': input_tokens + chat.output_tokens,
      },
    )
    if isinstance(reply, Unparsed):
      return {'raw': raw, 'parsed': None, 'parsing_error': reply.error}
    return {'raw': raw, 'parsed': reply, 'parsing_error': None}


def prompt_text(messages: list[Any]) -> str:
  """The text of a call's only message: a (role, text) pair, as the DeepEval judge sends it, or an OpenAI message."""
  [message] = messages
  return message[1] if isinstance(message, tuple) else message['content']


async def _pause() -> None:
  """Let other tasks run, as a request in flight does, without asyncio.sleep, which some tests patch."""
  future = asyncio.get_running_loop().create_future()
  future.get_loop().call_soon(future.set_result, None)
  await future


def rate_limit() -> ClientError:
  return ClientError(
    429,
    {
      'error': {
        'code': 429,
        'message': 'Quota exceeded. Please retry in 2s.',
        'status': 'RESOURCE_EXHAUSTED',
        'details': [{'@type': 'type.googleapis.com/google.rpc.RetryInfo', 'retryDelay': '2s'}],
      }
    },
    None,
  )
