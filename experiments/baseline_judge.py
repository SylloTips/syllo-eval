"""The shared judge model as the baseline frameworks call it, and the documents they give it to judge.

A framework sends its own prompt, with the JSON schema of its output as structured output. Everything else mirrors
Syllo-eval's judge client: the temperature, the concurrency limit and the retries of rate limits come from the
experiment config. Usage is read and the output validated the same way, so an output that does not parse or validate
raises with its usage and never reaches the framework's own parsing. Every call must run inside
``recorded_judge_calls``, which receives the successful ones, so that run reports count a framework's calls and tokens
as they count Syllo-eval's, and failed units record the same failure classes.
"""

import asyncio
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Any

from langchain_core.messages import AIMessage
from langchain_core.runnables import Runnable
from langchain_google_genai import ChatGoogleGenerativeAI
from pydantic import BaseModel, ValidationError

from syllo_eval.evaluation.judge import LlmJudgeRequest, LlmJudgeResponse
from syllo_eval.evaluation.metric_support.retrieved_context import RetrievedItem
from syllo_eval.infrastructure.exceptions import DataMappingError, ExternalServiceError, JudgeOutputError
from syllo_eval.infrastructure.llm_judge.base import LangChainLlmJudgeClient
from syllo_eval.infrastructure.llm_judge.gemini import GeminiLlmJudgeClient

PROVIDER = 'gemini'

_recorded_calls: ContextVar[list[LlmJudgeResponse]] = ContextVar('baseline_judge_calls')


@contextmanager
def recorded_judge_calls() -> Iterator[list[LlmJudgeResponse]]:
  """Collect the judge calls that a framework makes in this context, so each unit reports its own calls."""
  calls: list[LlmJudgeResponse] = []
  token = _recorded_calls.set(calls)
  try:
    yield calls
  finally:
    _recorded_calls.reset(token)


class BaselineJudge:
  """The shared judge's chat model, at the configured temperature, with its concurrency limit and retries."""

  def __init__(
    self,
    chat_model: ChatGoogleGenerativeAI,
    *,
    temperature: float,
    max_concurrent_requests: int,
    max_attempts: int,
  ):
    self.chat_model = chat_model.model_copy(update={'temperature': temperature})
    self._semaphore = asyncio.Semaphore(max_concurrent_requests)
    self._max_attempts = max_attempts

  async def generate[Output: BaseModel](
    self, runnable: Runnable[Any, Any], messages: Any, validate: Callable[[Any], Output]
  ) -> Output:
    """Send ``messages`` through ``runnable``, the chat model with structured output and its raw message, and record
    the call. ``validate`` turns the parsed output into the framework's schema, raising ``ValidationError``."""
    calls = _recorded_calls.get()
    async with self._semaphore:
      result, latency_seconds, attempts = await self._invoke(runnable, messages)

    raw = result['raw']
    if not isinstance(raw, AIMessage):
      raise DataMappingError(PROVIDER, 'structured output runnable did not return an AI message')
    # A response that cannot be used still cost its tokens.
    usage = LangChainLlmJudgeClient._usage(raw)
    if result['parsing_error'] is not None:
      raise JudgeOutputError(PROVIDER, 'failed to parse structured output', result['parsing_error'], usage=usage)
    if result['parsed'] is None:
      refusal = raw.additional_kwargs.get('refusal')
      reason = f'model refused judge request: {refusal}' if refusal else 'model returned no structured output'
      raise JudgeOutputError(PROVIDER, reason, usage=usage)
    try:
      output = validate(result['parsed'])
    except ValidationError as error:
      raise JudgeOutputError(PROVIDER, 'invalid structured output', error, usage=usage) from error

    calls.append(
      LlmJudgeResponse(
        provider=PROVIDER,
        model=raw.response_metadata.get('model_name') or self.chat_model.model,
        output=output,
        response_id=(
          raw.id
          or raw.response_metadata.get('id')
          or raw.additional_kwargs.get('response_id')
          or raw.additional_kwargs.get('id')
        ),
        usage=usage,
        latency_seconds=latency_seconds,
        attempts=attempts,
      )
    )
    return output

  async def _invoke(self, runnable: Runnable[Any, Any], messages: Any) -> tuple[dict[str, Any], float, int]:
    """The structured-output result, the latency of the attempt that succeeded, and the number of attempts."""
    attempt = 1
    while True:
      started = time.monotonic()
      try:
        return await runnable.ainvoke(messages), time.monotonic() - started, attempt
      except Exception as error:
        failure = ExternalServiceError(PROVIDER, 'judge', error)
        retry_delay = GeminiLlmJudgeClient._retry_delay_seconds(failure)
        if retry_delay is None or attempt >= self._max_attempts:
          raise failure from error
        attempt += 1
        await asyncio.sleep(retry_delay)


class _NoSylloEvalJudge:
  """The judge client of the compared metrics' constructors: the baseline metrics judge through their framework."""

  async def judge[Payload: BaseModel](self, request: LlmJudgeRequest[Payload]) -> LlmJudgeResponse[Payload]:
    raise RuntimeError('Baseline metrics judge through their framework, not through a Syllo-eval judge client.')

  async def aclose(self) -> None:
    pass


NO_SYLLO_EVAL_JUDGE = _NoSylloEvalJudge()


def node_text(item: RetrievedItem) -> str:
  """A retrieved document as the baselines give it to their judges, as DeepEval's retrieval-context nodes carry it:
  its title, if any, a blank line, then its content."""
  return '\n\n'.join(part for part in (item.item.title, item.item.content) if part)
