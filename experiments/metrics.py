"""The metrics the ablations are compared with: those of the main pass, and the gold-claim pass's recall.

The built-in contextual precision and recall score the selected context of the agent root, and recall decomposes the
expected answer into claims on every run. The protocol (METHODOLOGY.md) instead makes each search call one unit and
reads claims stored as ground truth. These subclasses change only that: the judge prompts and the scoring are the
built-ins'. Judging is a separate step, so that the single-call ablations (``ablation_metrics.py``) override nothing
else. Answer and Plan Correctness are the built-ins with the same judge-failure handling as the rest: in every arm of
every comparison, a judge failure becomes a FAILED result with a failure class and the usage of every response.
"""

import re
from abc import ABC
from collections.abc import Coroutine, Iterable, Iterator, Sequence
from dataclasses import dataclass
from enum import StrEnum
from typing import Any, ClassVar

import httpx
from langchain_core.exceptions import ContextOverflowError
from pydantic import BaseModel

from syllo_eval.evaluation.judge import LlmJudgeRequest, LlmJudgeResponse, judge_metadata
from syllo_eval.evaluation.judge.batch import judge_batch
from syllo_eval.evaluation.judge.metric_base import BaseLlmJudgeMetric
from syllo_eval.evaluation.metric_support.retrieved_context import (
  RetrievedItem,
  extract_retrieved_items,
  retrieval_skip_result,
)
from syllo_eval.evaluation.metrics.contracts import MetricComputationResult
from syllo_eval.evaluation.metrics.implementations.answer.correctness_judge import AnswerCorrectnessJudgeMetric
from syllo_eval.evaluation.metrics.implementations.plan.correctness_judge import PlanCorrectnessJudgeMetric
from syllo_eval.evaluation.metrics.implementations.rag.contextual_precision_judge import (
  ContextualPrecisionDocumentJudgeMetric,
  ContextualPrecisionJudgment,
)
from syllo_eval.evaluation.metrics.implementations.rag.contextual_recall_judge import (
  ContextualRecallDocumentJudgeMetric,
  ContextualRecallJudgment,
)
from syllo_eval.infrastructure.exceptions import DataMappingError
from syllo_eval.model import GroundTruth, MetricComputationStatus, Span

from benchmarks.common import Claim, claims_from_value
from benchmarks.erb import CLAIMS_KEY as ERB_GOLD_CLAIMS_KEY

# Span type the agent adapters give each search call. Its semantics carry the user question as the request, and the
# documents the call returned as one ranked document result at the `selected` stage.
SEARCH_SPAN_TYPE = 'retrieval'


class FailureKind(StrEnum):
  """Why a computation failed, recorded as ``metadata['failure']``; METHODOLOGY.md says which kinds count as wrong."""

  # The judgments do not match the unit one to one: a wrong count, or a missing, unknown or repeated rank.
  MISALIGNED = 'misaligned'
  TRUNCATED = 'truncated'  # a wrong judgment count, with the judge output at its token budget
  # The output did not parse or validate, or the model refused. Includes outputs cut inside a judgment.
  INVALID_OUTPUT = 'invalid_output'
  CONTEXT_OVERFLOW = 'context_overflow'
  TIMEOUT = 'timeout'
  PROVIDER = 'provider'  # any other provider error, after the judge client's retries
  TRACE_LOAD = 'trace_load'  # the stored trace could not be loaded or rendered


@dataclass(frozen=True, slots=True)
class JudgeFailure:
  kind: FailureKind
  message: str


@dataclass(frozen=True, slots=True)
class JudgedUnit[Judgment: BaseModel]:
  """The judge responses of one unit, and one judgment per input document or claim, in input order."""

  responses: list[LlmJudgeResponse]
  judgments: list[Judgment]


def classify_judge_error(error: Exception) -> FailureKind:
  if isinstance(error, DataMappingError):
    return FailureKind.INVALID_OUTPUT
  causes = list(_causes(error))
  if any(isinstance(cause, ContextOverflowError) for cause in causes):
    return FailureKind.CONTEXT_OVERFLOW
  # A deadline the provider enforces surfaces as a 504.
  if any(
    isinstance(cause, TimeoutError | httpx.TimeoutException) or getattr(cause, 'code', None) == 504 for cause in causes
  ):
    return FailureKind.TIMEOUT
  return FailureKind.PROVIDER


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


async def judge_each[Payload: BaseModel](
  calls: Iterable[Coroutine[Any, Any, LlmJudgeResponse[Payload]]],
) -> tuple[list[LlmJudgeResponse[Payload]], JudgeFailure | None]:
  """Run judge calls as ``judge_batch`` does, the first failure cancelling the others, and classify that failure."""
  failures: list[JudgeFailure] = []

  async def classified(call: Coroutine[Any, Any, LlmJudgeResponse[Payload]]) -> LlmJudgeResponse[Payload]:
    try:
      return await call
    except Exception as error:
      failures.append(JudgeFailure(classify_judge_error(error), str(error)))
      raise

  responses, failed = await judge_batch(classified(call) for call in calls)
  if failed is None:
    return responses, None
  return responses, failures[0] if failures else JudgeFailure(FailureKind.PROVIDER, failed.error_message or '')


def output_reached_budget(response: LlmJudgeResponse, budget: int) -> bool:
  output_tokens = (response.usage or {}).get('output_tokens')
  return output_tokens is not None and output_tokens >= budget


def count_failure(
  counts: Sequence[int], expected: int, responses: Sequence[LlmJudgeResponse], budget: int
) -> FailureKind:
  """Misaligned, or truncated when a response with the wrong number of judgments used its whole output budget."""
  truncated = any(
    count != expected and output_reached_budget(response, budget) for count, response in zip(counts, responses)
  )
  return FailureKind.TRUNCATED if truncated else FailureKind.MISALIGNED


def failed_result(
  metric_name: str,
  failure: JudgeFailure,
  metadata: dict[str, Any],
  responses: Sequence[LlmJudgeResponse] = (),
  raw_output: dict[str, Any] | None = None,
) -> MetricComputationResult:
  """A FAILED result that keeps the failure class, the unit's inputs and the usage of every judge response."""
  return MetricComputationResult(
    score=None,
    status=MetricComputationStatus.FAILED,
    reasoning=f'Failed to compute {metric_name}: {failure.message}',
    metadata={**metadata, 'failure': failure.kind.value, **judge_metadata(responses)},
    error_message=failure.message,
    raw_output=raw_output,
  )


class ClassifiedJudgeMetric(BaseLlmJudgeMetric, ABC):
  """``BaseLlmJudgeMetric`` making its one judge call through ``judge_each``, so a failure is a classified result.

  ``judge_input`` returns the user prompt and the metadata it adds to the result: the built-in prompt by default.
  """

  async def judge_input(
    self, span: Span, ground_truth: GroundTruth | None
  ) -> tuple[str, dict[str, Any]] | MetricComputationResult:
    return self.build_user_prompt(span, ground_truth), {}

  async def compute(self, span: Span, ground_truth: GroundTruth | None) -> MetricComputationResult:
    if reason := self.input_skip_reason(span):
      return MetricComputationResult(score=None, status=MetricComputationStatus.SKIPPED, error_message=reason)
    judge_input = await self.judge_input(span, ground_truth)
    if isinstance(judge_input, MetricComputationResult):
      return judge_input
    user_prompt, metadata = judge_input
    responses, failure = await judge_each(
      [
        self._judge_client.judge(
          LlmJudgeRequest(
            system_prompt=self.build_system_prompt(),
            user_prompt=user_prompt,
            response_model=self.response_model,
            temperature=self.temperature,
            max_output_tokens=self.max_output_tokens,
          )
        )
      ]
    )
    if failure is not None:
      return failed_result(self.name, failure, metadata, responses)
    payload = responses[0].output
    return MetricComputationResult(
      score=payload.score,
      reasoning=payload.reasoning,
      metadata={**(payload.metadata or {}), **judge_metadata(responses), **metadata},
    )


class AnswerCorrectness(ClassifiedJudgeMetric, AnswerCorrectnessJudgeMetric):
  """The built-in Answer Correctness: same name, prompts and scoring, with judge failures classified."""


class PlanCorrectness(ClassifiedJudgeMetric, PlanCorrectnessJudgeMetric):
  """The built-in Plan Correctness: same name, prompts and scoring, with judge failures classified."""


class SearchContextualPrecision(ContextualPrecisionDocumentJudgeMetric):
  """Built-in contextual precision over the documents of one search call, with one judge call per document."""

  metric_name = 'contextual_precision_search'
  metric_description = 'Contextual precision of the documents one search call returned; one judge call per document.'

  @property
  def target_span_types(self) -> tuple[str, ...]:
    return (SEARCH_SPAN_TYPE,)

  async def compute(self, span: Span, ground_truth: GroundTruth | None) -> MetricComputationResult:
    assert ground_truth is not None
    skip_result = retrieval_skip_result(
      span,
      metric_name=self.name,
      variant=self.variant,
      stage=self.retrieval_stage,
      require_content=True,
      require_rank=True,
    )
    if skip_result is not None:
      return skip_result
    items = extract_retrieved_items(span, self.variant, self.retrieval_stage)
    if not items:
      # An observed empty result scores 0 without calling the judge, as in the built-in.
      return self._result(items, JudgedUnit([], []))
    judged = await self.judge_items(span, ground_truth, items)
    return judged if isinstance(judged, MetricComputationResult) else self._result(items, judged)

  async def judge_items(
    self, span: Span, ground_truth: GroundTruth, items: list[RetrievedItem]
  ) -> JudgedUnit[ContextualPrecisionJudgment] | MetricComputationResult:
    """One built-in judge call per document, each returning exactly one judgment."""
    responses, failure = await judge_each(
      self._judge_retrieved_item(span, ground_truth, rank, item) for rank, item in enumerate(items, start=1)
    )
    if failure is not None:
      return self.unit_failure(failure, items, responses)
    counts = [len(response.output.judgments) for response in responses]
    if any(count != 1 for count in counts):
      rank = next(rank for rank, count in enumerate(counts, start=1) if count != 1)
      message = (
        f'Judge returned {counts[rank - 1]} contextual precision judgments '
        f'for 1 retrieved {self.variant} at rank {rank}.'
      )
      return self.unit_failure(
        JudgeFailure(count_failure(counts, 1, responses, self.max_output_tokens), message),
        items,
        responses,
        raw_output={'outputs': [response.output.model_dump() for response in responses]},
      )
    return JudgedUnit(responses, [response.output.judgments[0] for response in responses])

  def unit_failure(
    self,
    failure: JudgeFailure,
    items: list[RetrievedItem],
    responses: Sequence[LlmJudgeResponse],
    raw_output: dict[str, Any] | None = None,
    **details: Any,
  ) -> MetricComputationResult:
    """A failed unit, listing its documents in rank order so that each can be counted as a wrong decision."""
    metadata = {
      'variant': self.variant,
      'retrieval_kind': self.variant,
      'counts': {'retrieved': len(items)},
      'expected_retrieved_ids': [item.retrieved_id for item in items],
      **details,
    }
    return failed_result(self.name, failure, metadata, responses, raw_output)

  def _result(
    self, items: list[RetrievedItem], judged: JudgedUnit[ContextualPrecisionJudgment]
  ) -> MetricComputationResult:
    """The built-in score, with each judgment keyed to its input document rather than to the id the judge echoed."""
    keyed = [
      judgment.model_copy(update={'rank': rank, 'retrieved_id': item.retrieved_id})
      for rank, (item, judgment) in enumerate(zip(items, judged.judgments), start=1)
    ]
    score, rank_results, relevant_count = self._score(keyed)
    echo_mismatches = sum(
      (judgment.rank, judgment.retrieved_id) != (rank, item.retrieved_id)
      for rank, (item, judgment) in enumerate(zip(items, judged.judgments), start=1)
    )
    metadata: dict[str, Any] = {
      'variant': self.variant,
      'retrieval_kind': self.variant,
      'counts': {'retrieved': len(items), 'relevant': relevant_count, 'echo_mismatches': echo_mismatches},
      'rank_results': rank_results,
    }
    metadata.update(judge_metadata(judged.responses))
    return MetricComputationResult(
      score=score,
      reasoning=(
        f'Contextual precision over {len(items)} retrieved {self.variant}s '
        f'with {relevant_count} relevant {self.variant}s.'
      ),
      metadata=metadata,
      raw_output={'judgments': [judgment.model_dump() for judgment in judged.judgments]},
    )


class StoredClaimsContextualRecall(ContextualRecallDocumentJudgeMetric):
  """Built-in contextual recall over the documents of one search call, for claims stored as ground truth.

  Nothing is decomposed during a run, so a unit makes one judge call per claim. Subclasses set ``claims_key`` and the
  metric name; the value under the key is written by ``benchmarks.common.claims_value``.
  """

  claims_key: ClassVar[str]

  @property
  def target_span_types(self) -> tuple[str, ...]:
    return (SEARCH_SPAN_TYPE,)

  @property
  def ground_truth_key(self) -> str:
    return self.claims_key

  async def compute(self, span: Span, ground_truth: GroundTruth | None) -> MetricComputationResult:
    assert ground_truth is not None
    skip_result = retrieval_skip_result(
      span, metric_name=self.name, variant=self.variant, stage=self.retrieval_stage, require_content=True
    )
    if skip_result is not None:
      return skip_result
    claims = claims_from_value(ground_truth.ground_truth_value)
    items = extract_retrieved_items(span, self.variant, self.retrieval_stage)
    if not items:
      # Nothing retrieved can support a claim, so recall is 0 without calling the judge, as in the built-in.
      return self._result(items, claims, JudgedUnit([], []))
    judged = await self.judge_claims(span, items, claims)
    return judged if isinstance(judged, MetricComputationResult) else self._result(items, claims, judged)

  async def judge_claims(
    self, span: Span, items: list[RetrievedItem], claims: list[Claim]
  ) -> JudgedUnit[ContextualRecallJudgment] | MetricComputationResult:
    """One built-in judge call per claim, each returning exactly one judgment."""
    responses, failure = await judge_each(self._judge_claim(span, items, claim.text) for claim in claims)
    if failure is not None:
      return self.unit_failure(failure, items, claims, responses)
    counts = [len(response.output.judgments) for response in responses]
    if any(count != 1 for count in counts):
      count = next(count for count in counts if count != 1)
      return self.unit_failure(
        JudgeFailure(
          count_failure(counts, 1, responses, self.max_output_tokens),
          f'Judge returned {count} contextual recall judgments for 1 statement.',
        ),
        items,
        claims,
        responses,
        raw_output={'outputs': [response.output.model_dump() for response in responses]},
      )
    return JudgedUnit(responses, [response.output.judgments[0] for response in responses])

  def unit_failure(
    self,
    failure: JudgeFailure,
    items: list[RetrievedItem],
    claims: list[Claim],
    responses: Sequence[LlmJudgeResponse],
    raw_output: dict[str, Any] | None = None,
    **details: Any,
  ) -> MetricComputationResult:
    """A failed unit, listing its claims so that each can be counted as a wrong decision."""
    metadata = {
      'variant': self.variant,
      'retrieval_kind': self.variant,
      'claims_key': self.claims_key,
      'counts': {'retrieved': len(items), 'expected_statements': len(claims)},
      'expected_claim_ids': [claim.id for claim in claims],
      **details,
    }
    return failed_result(self.name, failure, metadata, responses, raw_output)

  def _result(
    self, items: list[RetrievedItem], claims: list[Claim], judged: JudgedUnit[ContextualRecallJudgment]
  ) -> MetricComputationResult:
    """The built-in score, with each judgment keyed to its stored claim, not to the statement the judge echoed."""
    keyed = [judgment.model_copy(update={'statement': claim.text}) for claim, judgment in zip(claims, judged.judgments)]
    _, statement_results, attributable_count = self._score(keyed)
    for statement_result, claim in zip(statement_results, claims):
      statement_result['claim_id'] = claim.id
    echo_mismatches = sum(
      _comparable(judgment.statement) != _comparable(claim.text) for claim, judgment in zip(claims, judged.judgments)
    )
    metadata: dict[str, Any] = {
      'variant': self.variant,
      'retrieval_kind': self.variant,
      'claims_key': self.claims_key,
      'counts': {
        'retrieved': len(items),
        'expected_statements': len(claims),
        'attributable': attributable_count,
        'echo_mismatches': echo_mismatches,
      },
      'statement_results': statement_results,
    }
    metadata.update(judge_metadata(judged.responses))
    return MetricComputationResult(
      score=attributable_count / len(claims),
      reasoning=(
        f'Contextual recall over {len(claims)} expected statements with {attributable_count} attributable statements.'
      ),
      metadata=metadata,
      raw_output={'judgments': [judgment.model_dump() for judgment in judged.judgments]},
    )


def _comparable(statement: str) -> str:
  """The words of a statement, without case, punctuation or the list number a single-call prompt puts before it."""
  return ' '.join(re.findall(r'\w+', re.sub(r'^\s*\d+\.\s+', '', statement))).casefold()


class GoldClaimsContextualRecall(StoredClaimsContextualRecall):
  """Contextual recall over the ERB gold claims (RQ1 attribution)."""

  claims_key = ERB_GOLD_CLAIMS_KEY
  metric_name = 'contextual_recall_gold_claims'
  metric_description = (
    'Contextual recall of the documents one search call returned, over the ERB gold claims; one judge call per claim.'
  )
