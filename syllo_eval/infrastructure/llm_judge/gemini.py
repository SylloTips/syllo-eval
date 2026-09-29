import asyncio
import random
import re
from typing import Any

from google.genai.errors import APIError
from langchain_google_genai import ChatGoogleGenerativeAI

from syllo_eval.evaluation.judge import LlmJudgeRequest
from syllo_eval.infrastructure.exceptions import ExternalServiceError
from syllo_eval.infrastructure.llm_judge.base import LangChainLlmJudgeClient
from syllo_eval.settings import GeminiJudgeSettings

_RETRY_HINT_PATTERN = re.compile(r'retry in (\d+(?:\.\d+)?)(ms|s)\b', re.IGNORECASE)
_RETRYABLE_CODES = frozenset({429, 503, 504})
_DEFAULT_RETRYABLE_DELAY_SECONDS = 60.0


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

  async def _invoke(self, request: LlmJudgeRequest) -> Any:
    attempt = 1
    while True:
      try:
        return await super()._invoke(request)
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
