"""The Dify agent: a chat app built in Dify's UI, called through Dify's service API.

Each sample starts a new conversation under a request ID the caller draws and sends as Dify's ``trace_id``. Dify's
Phoenix tracing writes it as ``dify_trace_id`` on the root of the workflow trace, which holds the agent node and its
tool calls, so trace lookup needs ``PHOENIX_REQUEST_ID_ATTRIBUTE=dify_trace_id``. The message ID would find Dify's
separate message trace instead, which holds only the question and the answer.
"""

import json
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass
from uuid import uuid4

import httpx
from pydantic import Field
from pydantic_settings import SettingsConfigDict

from syllo_eval.model import Sample
from syllo_eval.settings import EnvSettings


class DifySettings(EnvSettings):
  model_config = SettingsConfigDict(env_prefix='DIFY_')

  # Dify's service API, behind its nginx.
  base_url: str = 'http://localhost/v1'
  api_key: str | None = None
  # The end-user identifier Dify records for every message.
  user: str = Field(default='syllo-eval', min_length=1)
  timeout_seconds: float = Field(default=600.0, gt=0)


class DifyError(RuntimeError):
  """A message Dify rejected, or a stream that failed or ended without an answer."""


@dataclass(frozen=True, slots=True)
class DifyReply:
  request_id: str
  message_id: str
  conversation_id: str
  answer: str


class DifyCaller:
  """Sends a prompt as a new conversation and reads the streamed answer: agent apps do not answer in blocking mode."""

  def __init__(self, client: httpx.AsyncClient, settings: DifySettings) -> None:
    if settings.api_key is None:
      raise ValueError('DIFY_API_KEY must be set to the API key of the app under test')
    self._client = client
    self._headers = {'Authorization': f'Bearer {settings.api_key}'}
    self._user = settings.user

  async def call(self, sample: Sample) -> str:
    return (await self.chat(sample.input_prompt)).request_id

  async def chat(self, query: str) -> DifyReply:
    request_id = str(uuid4())
    body = {'query': query, 'inputs': {}, 'user': self._user, 'response_mode': 'streaming', 'trace_id': request_id}
    async with self._client.stream('POST', '/chat-messages', headers=self._headers, json=body) as response:
      if response.status_code != 200:
        await response.aread()
        raise DifyError(f'Dify returned {response.status_code}: {response.text[:500]}')
      answer: list[str] = []
      async for line in response.aiter_lines():
        if not line.startswith('data:'):
          continue
        event = json.loads(line.removeprefix('data:'))
        kind = event.get('event')
        if kind == 'error':
          raise DifyError(f'Dify failed the message: {event.get("message")}')
        # Agent apps stream agent_message chunks, chatflows message chunks.
        if kind in ('agent_message', 'message'):
          answer.append(event.get('answer') or '')
        elif kind == 'message_end':
          return DifyReply(request_id, event['message_id'], event['conversation_id'], ''.join(answer))
    raise DifyError('Dify ended the stream before the end of the message')


@asynccontextmanager
async def open_dify_client(settings: DifySettings) -> AsyncIterator[httpx.AsyncClient]:
  async with httpx.AsyncClient(base_url=settings.base_url, timeout=settings.timeout_seconds) as client:
    yield client
