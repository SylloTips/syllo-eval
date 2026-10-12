"""The single judge client shared by every experiment metric.

Experiment services disable the built-in metrics, so they never allocate a judge of their own: built-in metric classes
and the experiment metrics all receive this client. It pins the model, thinking level, timeout and retries from the
experiment config, and applies one concurrency limit to every judge call of a run. A request identical to an earlier
one of the same sample gets its verdict without a new call, reporting no tokens.
"""

import asyncio
import hashlib
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

from langchain_google_genai import ChatGoogleGenerativeAI
from pydantic import BaseModel

from syllo_eval.evaluation.judge import LlmJudgeClient, LlmJudgeRequest, LlmJudgeResponse
from syllo_eval.evaluation.plan_executor import sample_cache
from syllo_eval.infrastructure.llm_judge.gemini import GeminiLlmJudgeClient
from syllo_eval.settings import GeminiJudgeSettings

from config import JudgeConfig


class ConfiguredTemperatureJudgeClient:
  """Applies the configured temperature to every request; built-in metrics hardcode 0.0."""

  def __init__(self, inner: LlmJudgeClient, temperature: float):
    self._inner = inner
    self._temperature = temperature

  async def judge[Payload: BaseModel](self, request: LlmJudgeRequest[Payload]) -> LlmJudgeResponse[Payload]:
    return await self._inner.judge(request.model_copy(update={'temperature': self._temperature}))

  async def aclose(self) -> None:
    await self._inner.aclose()


class ReusingJudgeClient:
  def __init__(self, inner: LlmJudgeClient):
    self._inner = inner

  async def judge[Payload: BaseModel](self, request: LlmJudgeRequest[Payload]) -> LlmJudgeResponse[Payload]:
    calls: dict[Any, asyncio.Task[Any]] | None = sample_cache.get(None)
    if calls is None:
      return await self._inner.judge(request)
    model = request.response_model
    fields = request.model_dump_json(exclude={'response_model'})
    key = ('judge', hashlib.sha256(f'{model.__module__}.{model.__qualname__}\0{fields}'.encode()).hexdigest())
    task = calls.get(key)
    if task is None:
      task = calls[key] = asyncio.create_task(self._inner.judge(request))
      task.add_done_callback(lambda done: (done.cancelled() or done.exception()) and calls.pop(key, None))
      return await asyncio.shield(task)
    response = await asyncio.shield(task)
    usage = None if response.usage is None else dict.fromkeys(response.usage, 0)
    return response.model_copy(update={'usage': usage, 'latency_seconds': 0.0})

  async def aclose(self) -> None:
    await self._inner.aclose()


def build_chat_model(config: JudgeConfig, settings: GeminiJudgeSettings) -> ChatGoogleGenerativeAI:
  """The judge model of every framework: the API key comes from ``settings``, the rest from the experiment config."""
  return ChatGoogleGenerativeAI(
    model=config.model,
    api_key=settings.api_key,
    timeout=config.timeout_seconds,
    # The callers retry rate limits themselves, honouring the server's retry hints.
    max_retries=1,
    convert_system_message_to_human=True,
    thinking_level=config.thinking_level,
  )


@asynccontextmanager
async def open_judge_client(config: JudgeConfig, settings: GeminiJudgeSettings) -> AsyncIterator[LlmJudgeClient]:
  """Yield the shared judge. ``settings`` supplies the API key; everything else comes from the experiment config."""
  chat_model = build_chat_model(config, settings)
  client = GeminiLlmJudgeClient(
    settings.model_copy(update={'model': config.model, 'max_retries': config.max_retries}),
    max_concurrent_requests=config.max_concurrent_requests,
    chat_model=chat_model,
  )
  try:
    yield ReusingJudgeClient(ConfiguredTemperatureJudgeClient(client, config.temperature))
  finally:
    # The judge client leaves an injected chat model to its owner.
    await chat_model.async_client.aclose()
