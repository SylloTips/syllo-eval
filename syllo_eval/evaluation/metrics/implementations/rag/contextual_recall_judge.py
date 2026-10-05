import re
from collections.abc import Mapping, Sequence
from typing import Any

from pydantic import BaseModel

from syllo_eval.evaluation.judge.batch import judge_batch

from syllo_eval.evaluation.judge import LlmJudgeClient, LlmJudgeRequest, LlmJudgeResponse, judge_metadata
from syllo_eval.evaluation.judge.failures import mismatch_kind
from syllo_eval.evaluation.metrics.prompts import render_prompt
from syllo_eval.evaluation.metrics.contracts import MetricComputationResult, SpanEvaluationMetric
from syllo_eval.evaluation.metric_support.retrieved_context import (
  RetrievedItem,
  extract_retrieved_items,
  retrieval_skip_reason,
  retrieval_skip_result,
)
from syllo_eval.model import GroundTruth, GroundTruthKey, MetricComputationStatus, Span


class ContextualRecallClaims(BaseModel):
  claims: list[str]


class ContextualRecallJudgment(BaseModel):
  statement: str
  attributable: bool
  supporting_retrieved_ids: list[str]
  reasoning: str


class ContextualRecallJudgePayload(BaseModel):
  judgments: list[ContextualRecallJudgment]


class BaseContextualRecallJudgeMetric(SpanEvaluationMetric):
  """Scores whether expected-answer statements are supported by retrieved context."""

  variant: str = ''
  prompt_version = 'v1'
  retrieval_stage: str = 'selected'
  requires_judge_client = True
  accepts_target_span_types = True

  def __init__(
    self,
    *,
    judge_client: LlmJudgeClient,
    rubric_addition: str | None = None,
    target_span_types: Sequence[str] = ('agent_root',),
  ):
    self._judge_client = judge_client
    self._rubric_addition = rubric_addition
    self._target_span_types = tuple(target_span_types)

  @property
  def target_span_types(self) -> tuple[str, ...]:
    return self._target_span_types

  def matches_span(self, span: Span) -> bool:
    return any(
      result.kind == self.variant and result.stage == self.retrieval_stage for result in span.semantics.retrieval
    )

  def input_skip_reason(self, span: Span) -> str | None:
    return retrieval_skip_reason(span, self.variant, stage=self.retrieval_stage, require_content=True)

  @property
  def ground_truth_keys(self) -> tuple[str, ...]:
    return (GroundTruthKey.EXPECTED_OUTPUT.value,)

  @property
  def max_output_tokens(self) -> int:
    return 2_000

  async def compute(self, span: Span, ground_truths: Mapping[str, GroundTruth]) -> MetricComputationResult:
    ground_truth = ground_truths[self.ground_truth_keys[0]]

    skip_result = retrieval_skip_result(
      span, metric_name=self.name, variant=self.variant, stage=self.retrieval_stage, require_content=True
    )
    if skip_result is not None:
      return skip_result
    retrieved_items = self._extract_retrieved_items(span)
    if not retrieved_items:
      # Nothing retrieved can support a statement, so recall is 0 without calling the judge.
      return self._build_result(retrieved_items, [], [], [])
    decomposition_result = await self._decompose_claims(ground_truth)
    claims = decomposition_result.output.claims

    if not claims:
      return self._build_result(retrieved_items, [], [], [decomposition_result])

    judged = await self.judge_claims(span, retrieved_items, claims, prior_responses=[decomposition_result])
    if isinstance(judged, MetricComputationResult):
      return judged
    judge_results, judgments = judged
    return self._build_result(retrieved_items, claims, judgments, [decomposition_result, *judge_results])

  async def judge_claims(
    self,
    span: Span,
    retrieved_items: list[RetrievedItem],
    claims: list[str],
    *,
    prior_responses: Sequence[LlmJudgeResponse] = (),
  ) -> tuple[list[LlmJudgeResponse], list[ContextualRecallJudgment]] | MetricComputationResult:
    """Judge whether each claim is attributable to the retrieved items, one judge call per claim.

    Returns the responses and one judgment per claim in input order, or a FAILED result that also counts
    ``prior_responses``. Override to judge claims another way; scoring and result metadata stay the same.
    """
    judge_results, failure = await judge_batch(
      (self._judge_claim(span, retrieved_items, claim) for claim in claims),
      prior_responses=prior_responses,
      max_output_tokens=self.max_output_tokens,
    )
    if failure is not None:
      return failure

    judgments: list[ContextualRecallJudgment] = []
    for claim, result in zip(claims, judge_results):
      payload = result.output
      if len(payload.judgments) != 1:
        return self._failed_judgment_count_result(claim, payload, result, [*prior_responses, *judge_results])
      judgments.append(payload.judgments[0])
    return judge_results, judgments

  def build_decomposition_system_prompt(self) -> str:
    return render_prompt(f'contextual_recall/{self.prompt_version}/decomposition_system.md')

  def build_decomposition_user_prompt(self, ground_truth: GroundTruth) -> str:
    return render_prompt(
      f'contextual_recall/{self.prompt_version}/decomposition_user.md',
      expected_answer=ground_truth.ground_truth_value['expected_output'],
    )

  def build_system_prompt(self) -> str:
    return render_prompt(
      f'contextual_recall/{self.prompt_version}/system.md', variant=self.variant, rubric_addition=self._rubric_addition
    )

  def build_user_prompt(self, span: Span, retrieved_items: list[RetrievedItem], claim: str) -> str:
    return render_prompt(
      f'contextual_recall/{self.prompt_version}/user.md',
      variant=self.variant,
      items='\n\n'.join(item.render_for_prompt(rank) for rank, item in enumerate(retrieved_items, 1)),
      claim=claim,
    )

  async def _decompose_claims(self, ground_truth: GroundTruth) -> LlmJudgeResponse[ContextualRecallClaims]:
    return await self._judge_client.judge(
      LlmJudgeRequest(
        system_prompt=self.build_decomposition_system_prompt(),
        user_prompt=self.build_decomposition_user_prompt(ground_truth),
        response_model=ContextualRecallClaims,
        temperature=0.0,
        max_output_tokens=self.max_output_tokens,
      )
    )

  async def _judge_claim(
    self,
    span: Span,
    retrieved_items: list[RetrievedItem],
    claim: str,
  ) -> LlmJudgeResponse[ContextualRecallJudgePayload]:
    return await self._judge_client.judge(
      LlmJudgeRequest(
        system_prompt=self.build_system_prompt(),
        user_prompt=self.build_user_prompt(span, retrieved_items, claim),
        response_model=ContextualRecallJudgePayload,
        temperature=0.0,
        max_output_tokens=self.max_output_tokens,
      )
    )

  def _extract_retrieved_items(self, span: Span) -> list[RetrievedItem]:
    return extract_retrieved_items(span, self.variant, self.retrieval_stage)

  def _build_result(
    self,
    retrieved_items: list[RetrievedItem],
    claims: list[str],
    judgments: list[ContextualRecallJudgment],
    responses: list[LlmJudgeResponse],
    claim_ids: Sequence[str] | None = None,
  ) -> MetricComputationResult:
    """Score the judgments, one per claim, keyed to the claim rather than to the statement the judge echoed."""
    score, statement_results, attributable_count = self._score(
      [judgment.model_copy(update={'statement': claim}) for claim, judgment in zip(claims, judgments)]
    )
    for statement_result, claim_id in zip(statement_results, claim_ids or ()):
      statement_result['claim_id'] = claim_id
    expected_statement_count = len(claims)
    echo_mismatches = sum(
      _comparable(judgment.statement) != _comparable(claim) for claim, judgment in zip(claims, judgments)
    )

    metadata: dict[str, Any] = {
      'variant': self.variant,
      'retrieval_kind': self.variant,
      'counts': {
        'retrieved': len(retrieved_items),
        'expected_statements': expected_statement_count,
        'attributable': attributable_count,
        'echo_mismatches': echo_mismatches,
      },
      'statement_results': statement_results,
    }
    metadata.update(judge_metadata(responses))

    reasoning = (
      f'Contextual recall over {expected_statement_count} expected statements '
      f'with {attributable_count} attributable statements.'
    )
    return MetricComputationResult(
      score=score,
      reasoning=reasoning,
      metadata=metadata,
      raw_output={'judgments': [judgment.model_dump() for judgment in judgments]},
    )

  def _failed_judgment_count_result(
    self,
    claim: str,
    payload: ContextualRecallJudgePayload,
    judge_response: LlmJudgeResponse,
    responses: list[LlmJudgeResponse],
  ) -> MetricComputationResult:
    error_message = f'Judge returned {len(payload.judgments)} contextual recall judgments for 1 statement.'
    error_metadata: dict[str, Any] = {
      'variant': self.variant,
      'retrieval_kind': self.variant,
      'counts': {'judgments': len(payload.judgments)},
      'statement': claim,
      'failure': mismatch_kind(judge_response, self.max_output_tokens).value,
    }
    error_metadata.update(judge_metadata(responses))

    return MetricComputationResult(
      score=None,
      status=MetricComputationStatus.FAILED,
      reasoning=f'Failed to compute {self.name}: {error_message}',
      metadata=error_metadata,
      error_message=error_message,
      raw_output=judge_response.output.model_dump(),
    )

  @staticmethod
  def _score(judgments: list[ContextualRecallJudgment]) -> tuple[float, list[dict[str, Any]], int]:
    attributable_count = 0
    statement_results: list[dict[str, Any]] = []

    for judgment in judgments:
      if judgment.attributable:
        attributable_count += 1

      statement_results.append(
        {
          'statement': judgment.statement,
          'attributable': judgment.attributable,
          'supporting_retrieved_ids': judgment.supporting_retrieved_ids,
          'reasoning': judgment.reasoning,
        }
      )

    score = attributable_count / len(judgments) if judgments else 0.0
    return score, statement_results, attributable_count


def _comparable(statement: str) -> str:
  """The words of a statement, without case, punctuation or a leading list number."""
  return ' '.join(re.findall(r'\w+', re.sub(r'^\s*\d+\.\s+', '', statement))).casefold()


class ContextualRecallDocumentJudgeMetric(BaseContextualRecallJudgeMetric):
  """Scores whether retrieved documents support the expected answer."""

  variant = 'document'
  metric_name = 'contextual_recall_document_judge'
  metric_description = 'Uses an LLM judge to compute contextual recall over retrieved documents.'


class ContextualRecallSnippetJudgeMetric(BaseContextualRecallJudgeMetric):
  """Scores whether retrieved snippets support the expected answer."""

  variant = 'snippet'
  metric_name = 'contextual_recall_snippet_judge'
  metric_description = 'Uses an LLM judge to compute contextual recall over retrieved snippets.'
