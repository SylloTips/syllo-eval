from abc import ABC, abstractmethod
from collections.abc import Mapping
from typing import Any

from pydantic import BaseModel

from syllo_eval.evaluation.judge.base import LlmJudgeClient, LlmJudgeRequest, judge_metadata
from syllo_eval.evaluation.judge.batch import judge_batch
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

  def __init__(self, *, judge_client: LlmJudgeClient, rubric_addition: str | None = None):
    self._judge_client = judge_client
    self._rubric_addition = rubric_addition

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
  def build_user_prompt(self, span: Span, ground_truths: Mapping[str, GroundTruth]) -> str:
    """Build the user prompt for the judge request."""

  async def judge_input(
    self, span: Span, ground_truths: Mapping[str, GroundTruth]
  ) -> tuple[str, dict[str, Any]] | MetricComputationResult:
    """The user prompt and the metadata it adds to the result, or a result that ends the computation.

    Override to build the prompt from data that must be awaited; the default is ``build_user_prompt``.
    """
    return self.build_user_prompt(span, ground_truths), {}

  async def compute(self, span: Span, ground_truths: Mapping[str, GroundTruth]) -> MetricComputationResult:
    # Also support direct compute() calls that bypass the planner.
    if reason := self.input_skip_reason(span):
      return MetricComputationResult(score=None, status=MetricComputationStatus.SKIPPED, error_message=reason)
    judge_input = await self.judge_input(span, ground_truths)
    if isinstance(judge_input, MetricComputationResult):
      return judge_input
    user_prompt, input_metadata = judge_input
    request = LlmJudgeRequest(
      system_prompt=self.build_system_prompt(),
      user_prompt=user_prompt,
      response_model=self.response_model,
      temperature=self.temperature,
      max_output_tokens=self.max_output_tokens,
    )
    responses, failure = await judge_batch(
      [self._judge_client.judge(request)], max_output_tokens=self.max_output_tokens
    )
    if failure is not None:
      failure.metadata = {**input_metadata, **(failure.metadata or {})}
      return failure
    payload = responses[0].output

    metadata = dict(payload.metadata or {})
    metadata.update(judge_metadata(responses))
    metadata.update(input_metadata)

    return MetricComputationResult(
      score=payload.score,
      reasoning=payload.reasoning,
      metadata=metadata or None,
    )
