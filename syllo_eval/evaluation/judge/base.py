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


class LlmJudgeClient(Protocol):
  """Common client interface used by all LLM judge metrics."""

  async def judge[Payload: BaseModel](self, request: LlmJudgeRequest[Payload]) -> LlmJudgeResponse[Payload]:
    """Execute one judge request and return the parsed structured output."""

  async def aclose(self) -> None:
    """Release any provider-specific resources held by the client."""


def judge_metadata(responses: Sequence[LlmJudgeResponse]) -> dict[str, Any]:
  if not responses:
    return {}

  metadata: dict[str, Any] = {
    'judge_provider': responses[0].provider,
    'judge_model': responses[0].model,
  }
  response_ids = [response.response_id for response in responses if response.response_id is not None]
  if len(responses) == 1:
    if response_ids:
      metadata['judge_response_id'] = response_ids[0]
  elif response_ids:
    metadata['judge_response_ids'] = response_ids

  usage = _aggregate_usage(responses)
  if usage is not None:
    metadata['judge_usage'] = usage
  return metadata


def _aggregate_usage(responses: Sequence[LlmJudgeResponse]) -> dict[str, int] | None:
  usage: dict[str, int] = {}
  for response in responses:
    if response.usage is None:
      continue
    for key, value in response.usage.items():
      usage[key] = usage.get(key, 0) + value
  return usage or None
