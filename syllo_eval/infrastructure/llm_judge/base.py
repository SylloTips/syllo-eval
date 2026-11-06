import asyncio
import json
import logging
import time
from collections.abc import Awaitable, Callable
from typing import Any, Literal, TypeVar

from langchain_core.messages import AIMessage
from pydantic import BaseModel, ValidationError

from syllo_eval.evaluation.judge import LlmJudgeRequest, LlmJudgeResponse
from syllo_eval.evaluation.judge.base import aggregate_usage
from syllo_eval.evaluation.judge.failures import output_reached_limit
from syllo_eval.infrastructure.exceptions import (
  ConfigurationError,
  DataMappingError,
  ExternalServiceError,
  InfrastructureError,
  JudgeOutputError,
)

Payload = TypeVar('Payload', bound=BaseModel)

logger = logging.getLogger(__name__)


class LangChainLlmJudgeClient:
  """Structured judge calls with provider options and one validation boundary."""

  def __init__(
    self,
    provider_name: str,
    model: str,
    max_concurrent_requests: int,
    chat_model: Any,
    *,
    structured_output_method: Literal['json_mode', 'json_schema'],
    max_tokens_option: str,
    output_attempts: int,
    close: Callable[[], Awaitable[None]] | None = None,
  ):
    self._provider_name = provider_name
    self._model = model.strip()
    if not self._model:
      raise ConfigurationError(provider_name, 'model', 'must be configured')
    if max_concurrent_requests < 1:
      raise ConfigurationError(provider_name, 'max_concurrent_requests', 'must be at least 1')
    self._semaphore = asyncio.Semaphore(max_concurrent_requests)
    self._chat_model = chat_model
    self._structured_output_method = structured_output_method
    self._max_tokens_option = max_tokens_option
    self._output_attempts = output_attempts
    self._close = close

  @property
  def provider(self) -> str:
    return self._provider_name

  @property
  def model(self) -> str:
    return self._model

  @property
  def models(self) -> tuple[str, ...]:
    return (self._model,)

  async def judge(self, request: LlmJudgeRequest[Payload]) -> LlmJudgeResponse[Payload]:
    usages: list[dict[str, int] | None] = []
    attempts = 0
    while True:
      try:
        async with self._semaphore:
          result = await self._invoke(request)
      except InfrastructureError as error:
        error.retry_usage = aggregate_usage(usages)
        raise
      attempts += result.get('attempts', 1)

      raw = result['raw']
      if not isinstance(raw, AIMessage):
        raise DataMappingError(self._provider_name, 'structured output runnable did not return an AI message')
      # A response that cannot be used still cost its tokens.
      usages.append(self._usage(raw))
      try:
        output = self._output(request, result, raw)
      except JudgeOutputError as error:
        logger.warning(
          'Judge output rejected: %s',
          json.dumps(
            {
              'provider': self._provider_name,
              'model': raw.response_metadata.get('model_name') or self._model,
              'attempt': len(usages),
              'finish_reason': raw.response_metadata.get('finish_reason'),
              'max_output_tokens': request.max_output_tokens,
              'usage': usages[-1],
              'error': str(error),
              'text': raw.text,
            },
            ensure_ascii=False,
          ),
        )
        if (
          len(usages) >= self._output_attempts
          or _refusal(raw) is not None
          or output_reached_limit(usages[-1], request.max_output_tokens)
        ):
          error.usage = usages[-1]
          error.retry_usage = aggregate_usage(usages[:-1])
          raise
        continue

      return LlmJudgeResponse(
        provider=self._provider_name,
        model=raw.response_metadata.get('model_name') or self._model,
        output=output,
        response_id=(
          raw.id
          or raw.response_metadata.get('id')
          or raw.additional_kwargs.get('response_id')
          or raw.additional_kwargs.get('id')
        ),
        usage=usages[-1],
        retry_usage=aggregate_usage(usages[:-1]),
        latency_seconds=result.get('latency_seconds'),
        attempts=attempts,
      )

  def _output(self, request: LlmJudgeRequest[Payload], result: dict[str, Any], raw: AIMessage) -> Payload:
    refusal = _refusal(raw)
    if result['parsing_error'] is not None:
      reason = f'model refused judge request: {refusal}' if refusal is not None else 'failed to parse structured output'
      raise JudgeOutputError(self._provider_name, reason, result['parsing_error'])
    if result['parsed'] is None:
      reason = (
        f'model refused judge request: {refusal}' if refusal is not None else 'model returned no structured output'
      )
      raise JudgeOutputError(self._provider_name, reason)
    try:
      return request.response_model.model_validate(result['parsed'])
    except ValidationError as error:
      raise JudgeOutputError(self._provider_name, 'invalid structured output', error) from error

  async def aclose(self) -> None:
    """Close an owned provider client; injected chat models remain caller-owned."""
    if self._close is not None:
      await self._close()

  def _model_options(self, request: LlmJudgeRequest) -> dict[str, Any]:
    """The request's sampling options, as fields of the chat model."""
    options: dict[str, Any] = {'temperature': request.temperature}
    if request.max_output_tokens is not None:
      options[self._max_tokens_option] = request.max_output_tokens
    return options

  def _messages(self, system_prompt: str, user_prompt: str) -> list[tuple[str, str]]:
    return [('system', system_prompt), ('human', user_prompt)]

  async def _invoke(self, request: LlmJudgeRequest) -> dict[str, Any]:
    schema: type[BaseModel] | dict[str, Any] = request.response_model
    system_prompt = request.system_prompt
    if self._structured_output_method == 'json_mode':
      schema_json = json.dumps(request.response_model.model_json_schema(), ensure_ascii=False)
      system_prompt += (
        '\n\nRespond with a single valid JSON object that matches the following JSON schema. '
        'Return only the JSON object, with no markdown code fences or surrounding text.\n'
        f'JSON schema: {schema_json}'
      )
    else:
      schema = request.response_model.model_json_schema()

    # The options are set on a copy of the model: `with_structured_output` drops options bound with `bind`.
    runnable = self._chat_model.model_copy(update=self._model_options(request)).with_structured_output(
      schema=schema,
      method=self._structured_output_method,
      include_raw=True,
    )
    started = time.monotonic()
    try:
      result = await runnable.ainvoke(self._messages(system_prompt, request.user_prompt))
    except Exception as error:
      raise ExternalServiceError(self._provider_name, 'judge', error) from error
    # Timed per attempt, so retry back-off waits never count as provider latency.
    return {**result, 'latency_seconds': time.monotonic() - started}

  @staticmethod
  def _usage(message: AIMessage) -> dict[str, int] | None:
    usage: dict[str, Any] = dict(message.usage_metadata or {})
    if not usage:
      legacy = message.response_metadata.get('token_usage', {})
      usage = {
        'input_tokens': legacy.get('prompt_tokens'),
        'output_tokens': legacy.get('completion_tokens'),
        'total_tokens': legacy.get('total_tokens'),
      }
    # Detail counts are kept only when the provider reports them: a missing count is unknown, not zero.
    usage['cached_input_tokens'] = (usage.get('input_token_details') or {}).get('cache_read')
    usage['reasoning_tokens'] = (usage.get('output_token_details') or {}).get('reasoning')
    return {
      key: value
      for key in ('input_tokens', 'output_tokens', 'total_tokens', 'cached_input_tokens', 'reasoning_tokens')
      if isinstance(value := usage.get(key), int)
    } or None


def _refusal(message: AIMessage) -> str | None:
  """Why the model refused the request, if it did.

  OpenAI reports a refusal in its own field. From langchain-core 1.7.5, LangChain also turns each provider's report,
  such as Gemini's safety stop, into a refusal block.
  """
  if refusal := message.additional_kwargs.get('refusal'):
    return str(refusal)
  for block in message.content_blocks:
    fields: dict[str, Any] = dict(block)
    if fields.get('type') == 'refusal':
      return str(fields.get('refusal') or 'no reason given')
  return None
