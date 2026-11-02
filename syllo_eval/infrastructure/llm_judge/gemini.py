import asyncio
import logging
import random
import re
import time
from typing import Any

from google.genai import types
from google.genai.errors import APIError
from langchain_google_genai import ChatGoogleGenerativeAI
from pydantic import BaseModel

from syllo_eval.evaluation.judge import LlmJudgeRequest, LlmJudgeResponse
from syllo_eval.evaluation.judge.prefix import SharedPromptPrefix, shared_prefix_of
from syllo_eval.infrastructure.exceptions import ExternalServiceError
from syllo_eval.infrastructure.llm_judge.base import LangChainLlmJudgeClient
from syllo_eval.settings import GeminiJudgeSettings

_RETRY_HINT_PATTERN = re.compile(r'retry in (\d+(?:\.\d+)?)(ms|s)\b', re.IGNORECASE)
_RETRYABLE_CODES = frozenset({429, 503, 504})
_DEFAULT_RETRYABLE_DELAY_SECONDS = 60.0
_CACHE_TTL = '600s'

logger = logging.getLogger(__name__)


class GeminiLlmJudgeClient(LangChainLlmJudgeClient):
  """Judge client backed by LangChain's Gemini chat integration."""

  def __init__(
    self,
    config: GeminiJudgeSettings,
    max_concurrent_requests: int,
    chat_model: Any | None = None,
  ):
    client_args: dict[str, Any] | None = None
    if config.base_url is not None:
      client_args = {'http_options': {'base_url': config.base_url}}

    self._max_attempts = config.max_retries
    default_chat_model = chat_model or ChatGoogleGenerativeAI(
      model=config.model,
      api_key=config.api_key,
      timeout=config.timeout_seconds,
      max_retries=1,
      convert_system_message_to_human=True,
      client_args=client_args,
    )
    super().__init__(
      provider_name='gemini',
      model=config.model,
      max_concurrent_requests=max_concurrent_requests,
      chat_model=default_chat_model,
      structured_output_method='json_schema',
      max_tokens_option='max_output_tokens',
      close=default_chat_model.async_client.aclose if chat_model is None else None,
    )

  async def judge[Payload: BaseModel](self, request: LlmJudgeRequest[Payload]) -> LlmJudgeResponse[Payload]:
    prefix = shared_prefix_of(request.system_prompt, request.user_prompt)
    task = asyncio.current_task()
    if prefix is None or prefix.requests < 2 or task is None:
      return await super().judge(request)
    prefix.users.add(task)
    try:
      async with prefix.cache_lock:
        if not prefix.cache_attempted:
          prefix.cache_attempted = True
          await self._create_cache(prefix)
      return await super().judge(request)
    finally:
      prefix.users.discard(task)

  async def _create_cache(self, prefix: SharedPromptPrefix) -> None:
    started = time.monotonic()
    try:
      cache = await self._chat_model.async_client.caches.create(
        model=self.model,
        config=types.CreateCachedContentConfig(
          contents=types.Content(
            role='user', parts=[types.Part(text=prefix.system_prompt), types.Part(text=prefix.user_prefix)]
          ),
          ttl=_CACHE_TTL,
        ),
      )
    except Exception as error:
      if not isinstance(error, APIError) or error.code != 400:
        logger.warning('Judge calls sharing a prompt prefix run uncached, the cache could not be created: %s', error)
    else:
      prefix.cache_name = cache.name
      prefix.cache_creation_input_tokens = cache.usage_metadata.total_token_count if cache.usage_metadata else None
      prefix.cache_close = lambda: self._delete_cache(prefix)
    prefix.cache_seconds += time.monotonic() - started

  async def _delete_cache(self, prefix: SharedPromptPrefix) -> None:
    started = time.monotonic()
    try:
      await self._chat_model.async_client.caches.delete(name=prefix.cache_name)
    except Exception as error:
      logger.warning('Cache %s could not be deleted and expires after %s: %s', prefix.cache_name, _CACHE_TTL, error)
    prefix.cache_seconds += time.monotonic() - started

  def _model_options(self, request: LlmJudgeRequest) -> dict[str, Any]:
    options = super()._model_options(request)
    prefix = shared_prefix_of(request.system_prompt, request.user_prompt)
    if prefix is not None and prefix.cache_name is not None:
      options['cached_content'] = prefix.cache_name
    return options

  def _messages(self, system_prompt: str, user_prompt: str) -> list[tuple[str, str]]:
    prefix = shared_prefix_of(system_prompt, user_prompt)
    if prefix is None:
      return super()._messages(system_prompt, user_prompt)
    suffix = user_prompt[len(prefix.user_prefix) :]
    if prefix.cache_name is None:
      return [('system', system_prompt), ('human', prefix.user_prefix), ('human', suffix)]
    return [('human', suffix)]

  async def _invoke(self, request: LlmJudgeRequest) -> Any:
    attempt = 1
    while True:
      try:
        return {**await super()._invoke(request), 'attempts': attempt}
      except ExternalServiceError as error:
        retry_delay = self._retry_delay_seconds(error)
        if retry_delay is None or attempt >= self._max_attempts:
          raise
        attempt += 1
        await asyncio.sleep(retry_delay)

  @staticmethod
  def _retry_delay_seconds(error: ExternalServiceError) -> float | None:
    api_error = GeminiLlmJudgeClient._google_api_error(error)
    if api_error is None or api_error.code not in _RETRYABLE_CODES:
      return None
    payload = api_error.details.get('error') if isinstance(api_error.details, dict) else None
    if not isinstance(payload, dict):
      return None
    hint = max(
      GeminiLlmJudgeClient._parse_message_hint(payload.get('message')),
      GeminiLlmJudgeClient._parse_structured_hint(payload.get('details')),
    )
    if hint <= 0:
      return _DEFAULT_RETRYABLE_DELAY_SECONDS
    return hint * random.uniform(1.0, 1.5)

  @staticmethod
  def _parse_structured_hint(details: Any) -> float:
    if not isinstance(details, list):
      return 0.0
    for detail in details:
      if not isinstance(detail, dict):
        continue
      retry_delay = detail.get('retryDelay')
      if isinstance(retry_delay, str) and retry_delay.endswith('s'):
        try:
          return float(retry_delay.removesuffix('s'))
        except ValueError:
          continue
    return 0.0

  @staticmethod
  def _parse_message_hint(message: Any) -> float:
    if not isinstance(message, str):
      return 0.0
    match = _RETRY_HINT_PATTERN.search(message)
    if match is None:
      return 0.0
    value, unit = match.groups()
    delay = float(value)
    return delay / 1000 if unit == 'ms' else delay

  @staticmethod
  def _google_api_error(error: ExternalServiceError) -> APIError | None:
    original_error = error.original_error
    if isinstance(original_error, APIError):
      return original_error

    cause = getattr(original_error, '__cause__', None)
    return cause if isinstance(cause, APIError) else None
