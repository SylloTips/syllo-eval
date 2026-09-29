from abc import ABC, abstractmethod
from typing import Any

from pydantic import BaseModel

from syllo_eval.evaluation.judge.base import LlmJudgeClient, LlmJudgeRequest, judge_metadata
from syllo_eval.evaluation.metrics.contracts import MetricComputationResult, SpanEvaluationMetric
from syllo_eval.model import GroundTruth, MetricComputationStatus, Span


class JudgeScorePayload(BaseModel):
  """Default structured payload returned by judge metrics."""

  score: float
  reasoning: str
  metadata: dict[str, Any] | None = None


class BaseLlmJudgeMetric(SpanEvaluationMetric, ABC):
  """Reusable base class for metrics backed by a shared LLM judge client."""

  requires_judge_client = True

  def __init__(self, *, judge_client: LlmJudgeClient):
    self._judge_client = judge_client

  @property
  def temperature(self) -> float:
    """Sampling temperature used by the judge request."""
    return 0.0

  @property
  def max_output_tokens(self) -> int | None:
    """Maximum response tokens requested from the judge model."""
    return None

  @property
  def response_model(self) -> type[JudgeScorePayload]:
    """Pydantic model used for structured output."""
    return JudgeScorePayload

  @abstractmethod
  def build_system_prompt(self) -> str:
    """Build the system prompt for the judge request."""

  @abstractmethod
  def build_user_prompt(self, span: Span, ground_truth: GroundTruth | None) -> str:
    """Build the user prompt for the judge request."""

  async def compute(self, span: Span, ground_truth: GroundTruth | None) -> MetricComputationResult:
    # Also support direct compute() calls that bypass the planner.
    if reason := self.input_skip_reason(span):
      return MetricComputationResult(score=None, status=MetricComputationStatus.SKIPPED, error_message=reason)
    result = await self._judge_client.judge(
      LlmJudgeRequest(
        system_prompt=self.build_system_prompt(),
        user_prompt=self.build_user_prompt(span, ground_truth),
        response_model=self.response_model,
        temperature=self.temperature,
        max_output_tokens=self.max_output_tokens,
      )
    )
    payload = result.output

    metadata = dict(payload.metadata or {})
    metadata.update(judge_metadata([result]))

    return MetricComputationResult(
      score=payload.score,
      reasoning=payload.reasoning,
      metadata=metadata or None,
    )
