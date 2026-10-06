"""Why a judge computation failed, recorded as ``metadata['failure']`` so that analyses can tell outcomes apart."""

from collections.abc import Iterator, Sequence
from enum import StrEnum
from typing import Any

import httpx
from langchain_core.exceptions import ContextOverflowError

from syllo_eval.evaluation.judge.base import LlmJudgeResponse, judge_metadata
from syllo_eval.evaluation.metrics.contracts import MetricComputationResult
from syllo_eval.infrastructure.exceptions import DataMappingError
from syllo_eval.model import MetricComputationStatus


class JudgeFailureKind(StrEnum):
  # The judgments do not match the judged items one to one: a wrong count, or ranks the items do not have.
  MISALIGNED = 'misaligned'
  # A wrong judgment count, or an output that does not parse, with the output at its token limit.
  TRUNCATED = 'truncated'
  # The output does not parse or validate, or the model refused.
  INVALID_OUTPUT = 'invalid_output'
  CONTEXT_OVERFLOW = 'context_overflow'
  TIMEOUT = 'timeout'
  # Any other provider error, after the judge client's retries.
  PROVIDER = 'provider'


def output_reached_limit(usage: dict[str, int] | None, max_output_tokens: int | None) -> bool:
  output_tokens = (usage or {}).get('output_tokens')
  return max_output_tokens is not None and output_tokens is not None and output_tokens >= max_output_tokens


def mismatch_kind(response: LlmJudgeResponse, max_output_tokens: int | None) -> JudgeFailureKind:
  """The failure of a response with the wrong judgments: truncated when it used its whole output limit."""
  if output_reached_limit(response.usage, max_output_tokens):
    return JudgeFailureKind.TRUNCATED
  return JudgeFailureKind.MISALIGNED


def classify_judge_error(error: BaseException, max_output_tokens: int | None = None) -> JudgeFailureKind:
  if isinstance(error, DataMappingError):
    if output_reached_limit(getattr(error, 'usage', None), max_output_tokens):
      return JudgeFailureKind.TRUNCATED
    return JudgeFailureKind.INVALID_OUTPUT
  causes = list(_causes(error))
  if any(isinstance(cause, ContextOverflowError) for cause in causes):
    return JudgeFailureKind.CONTEXT_OVERFLOW
  # A deadline the provider enforces surfaces as a 504.
  if any(
    isinstance(cause, TimeoutError | httpx.TimeoutException) or getattr(cause, 'code', None) == 504 for cause in causes
  ):
    return JudgeFailureKind.TIMEOUT
  return JudgeFailureKind.PROVIDER


def judge_failure_result(
  error: BaseException,
  responses: Sequence[LlmJudgeResponse],
  *,
  max_output_tokens: int | None = None,
  metadata: dict[str, Any] | None = None,
  failed_errors: Sequence[BaseException] = (),
) -> MetricComputationResult:
  """Classify the first error and keep usage from all successful and failed calls."""
  return MetricComputationResult(
    score=None,
    status=MetricComputationStatus.FAILED,
    error_message=str(error),
    metadata={
      **(metadata or {}),
      'failure': classify_judge_error(error, max_output_tokens).value,
      **judge_metadata(responses, failed_usages=[getattr(item, 'usage', None) for item in failed_errors or (error,)]),
    },
  )


def _causes(error: BaseException) -> Iterator[BaseException]:
  """The error and every error it wraps, through ``original_error`` and the exception chain."""
  seen: set[int] = set()
  pending: list[Any] = [error]
  while pending:
    current = pending.pop()
    if not isinstance(current, BaseException) or id(current) in seen:
      continue
    seen.add(id(current))
    yield current
    pending.extend((getattr(current, 'original_error', None), current.__cause__, current.__context__))
