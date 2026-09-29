import asyncio
import json
from collections.abc import Awaitable, Callable
from typing import Any, Literal, TypeVar

from langchain_core.messages import AIMessage
from pydantic import BaseModel, ValidationError

from syllo_eval.evaluation.judge import LlmJudgeRequest, LlmJudgeResponse
from syllo_eval.infrastructure.exceptions import ConfigurationError, DataMappingError, ExternalServiceError

Payload = TypeVar('Payload', bound=BaseModel)


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
    self._close = close

  @property
  def model(self) -> str:
    return self._model

  async def judge(self, request: LlmJudgeRequest[Payload]) -> LlmJudgeResponse[Payload]:
    async with self._semaphore:
      result = await self._invoke(request)

    if result['parsing_error'] is not None:
      raise DataMappingError(self._provider_name, 'failed to parse structured output', result['parsing_error'])
    raw = result['raw']
    if not isinstance(raw, AIMessage):
      raise DataMappingError(self._provider_name, 'structured output runnable did not return an AI message')
    if result['parsed'] is None:
      refusal = raw.additional_kwargs.get('refusal')
      reason = f'model refused judge request: {refusal}' if refusal else 'model returned no structured output'
      raise DataMappingError(self._provider_name, reason)
    try:
      output = request.response_model.model_validate(result['parsed'])
    except ValidationError as error:
      raise DataMappingError(self._provider_name, 'invalid structured output', error) from error

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
      usage=self._usage(raw),
    )

  async def aclose(self) -> None:
    """Close an owned provider client; injected chat models remain caller-owned."""
    if self._close is not None:
      await self._close()

  async def _invoke(self, request: LlmJudgeRequest) -> dict[str, Any]:
    options: dict[str, Any] = {'temperature': request.temperature}
    if request.max_output_tokens is not None:
      options[self._max_tokens_option] = request.max_output_tokens
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

    runnable = self._chat_model.bind(**options).with_structured_output(
      schema=schema,
      method=self._structured_output_method,
      include_raw=True,
    )
    try:
      return await runnable.ainvoke([('system', system_prompt), ('human', request.user_prompt)])
    except Exception as error:
      raise ExternalServiceError(self._provider_name, 'judge', error) from error

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
    return {
      key: value
      for key in ('input_tokens', 'output_tokens', 'total_tokens')
      if isinstance(value := usage.get(key), int)
    } or None
