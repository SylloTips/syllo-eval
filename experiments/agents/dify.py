"""The Dify agent: a chat app built in Dify's UI, called through Dify's service API.

Each sample starts a new conversation under a request ID the caller draws and sends as Dify's ``trace_id``. Dify's
Phoenix tracing writes it as ``dify_trace_id`` on the root of the workflow trace, which holds the agent node and its
tool calls, so collection looks traces up by that attribute. The message ID would find Dify's separate message trace
instead, which holds only the question and the answer.
"""

import json
from uuid import uuid4

import httpx
from pydantic import Field
from pydantic_settings import SettingsConfigDict

from syllo_eval.model import Sample
from syllo_eval.settings import EnvSettings

REQUEST_ID_ATTRIBUTE = 'dify_trace_id'


class DifySettings(EnvSettings):
  model_config = SettingsConfigDict(env_prefix='DIFY_')

  # Dify's service API, behind its nginx.
  base_url: str = 'http://localhost/v1'
  api_key: str | None = None
  # The end-user identifier Dify records for every message.
  user: str = Field(default='syllo-eval', min_length=1)
  timeout_seconds: float = Field(default=600.0, gt=0)
  # The search server that the app calls, as this host reaches it: collect checks what it serves.
  search_url: str = Field(default='http://127.0.0.1:8101/mcp', min_length=1)


class DifyError(RuntimeError):
  """A message Dify rejected, or a stream that failed or ended without an answer."""


class DifyCaller:
  """Sends a prompt as a new conversation and reads the stream to its end: agent apps do not answer in blocking mode.

  Each call opens its own client, so the caller holds nothing to close.
  """

  def __init__(self, settings: DifySettings, transport: httpx.AsyncBaseTransport | None = None) -> None:
    if settings.api_key is None:
      raise ValueError('DIFY_API_KEY must be set to the API key of the app under test')
    self._settings = settings
    self._transport = transport
    self._headers = {'Authorization': f'Bearer {settings.api_key}'}

  async def call(self, sample: Sample) -> str:
    request_id = str(uuid4())
    body = {
      'query': sample.input_prompt,
      'inputs': {},
      'user': self._settings.user,
      'response_mode': 'streaming',
      'trace_id': request_id,
    }
    async with (
      httpx.AsyncClient(
        base_url=self._settings.base_url, timeout=self._settings.timeout_seconds, transport=self._transport
      ) as client,
      client.stream('POST', '/chat-messages', headers=self._headers, json=body) as response,
    ):
      if response.status_code != 200:
        await response.aread()
        raise DifyError(f'Dify returned {response.status_code}: {response.text[:500]}')
      async for line in response.aiter_lines():
        if not line.startswith('data:'):
          continue
        event = json.loads(line.removeprefix('data:'))
        if event.get('event') == 'error':
          raise DifyError(f'Dify failed the message: {event.get("message")}')
        if event.get('event') == 'message_end':
          return request_id
    raise DifyError('Dify ended the stream before the end of the message')
