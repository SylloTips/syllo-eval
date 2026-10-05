from collections.abc import Sequence
from typing import Any, Protocol
from pydantic import BaseModel, Field


class LlmJudgeRequest[Payload: BaseModel](BaseModel):
  """A structured judge request produced by an evaluation metric."""

  system_prompt: str = Field(min_length=1)
  user_prompt: str = Field(min_length=1)
  response_model: type[Payload]
  temperature: float = Field(default=0.0, ge=0.0)
  max_output_tokens: int | None = Field(default=None, gt=0)


class LlmJudgeResponse[Payload: BaseModel](BaseModel):
  """Structured response returned by a judge provider."""

  provider: str
  model: str
  output: Payload
  response_id: str | None = None
  usage: dict[str, int] | None = None
  # Wall time of the attempt that succeeded, and how many client-level attempts the call took.
  latency_seconds: float | None = Field(default=None, ge=0.0)
  attempts: int = Field(default=1, ge=1)


class LlmJudgeClient(Protocol):
  """Common client interface used by all LLM judge metrics."""

  async def judge[Payload: BaseModel](self, request: LlmJudgeRequest[Payload]) -> LlmJudgeResponse[Payload]:
    """Execute one judge request and return the parsed structured output."""

  async def aclose(self) -> None:
    """Release any provider-specific resources held by the client."""


def judge_metadata(
  responses: Sequence[LlmJudgeResponse], failed_usages: Sequence[dict[str, int] | None] = ()
) -> dict[str, Any]:
  """Judge usage of a computation. ``failed_usages`` holds one entry per call that raised, ``None`` when unknown."""
  if not responses and not failed_usages:
    return {}

  metadata: dict[str, Any] = {'judge_calls': len(responses) + len(failed_usages)}
  if failed_usages:
    metadata['judge_failed_calls'] = len(failed_usages)
  usage = _aggregate_usage([response.usage for response in responses] + list(failed_usages))
  if usage is not None:
    metadata['judge_usage'] = usage
  if not responses:
    return metadata

  metadata.update(
    {
      'judge_provider': responses[0].provider,
      'judge_model': responses[0].model,
      'judge_attempts': sum(response.attempts for response in responses),
    }
  )
  latencies = [response.latency_seconds for response in responses if response.latency_seconds is not None]
  if latencies:
    metadata['judge_latency_seconds'] = round(sum(latencies), 3)
  response_ids = [response.response_id for response in responses if response.response_id is not None]
  if len(responses) == 1:
    if response_ids:
      metadata['judge_response_id'] = response_ids[0]
  elif response_ids:
    metadata['judge_response_ids'] = response_ids
  return metadata


def _aggregate_usage(usages: Sequence[dict[str, int] | None]) -> dict[str, int] | None:
  total: dict[str, int] = {}
  for usage in usages:
    for key, value in (usage or {}).items():
      total[key] = total.get(key, 0) + value
  return total or None
