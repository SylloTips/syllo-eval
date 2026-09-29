from typing import Any

from pydantic import BaseModel

from syllo_eval.evaluation.judge.batch import judge_batch

from syllo_eval.evaluation.judge import LlmJudgeClient, LlmJudgeRequest, LlmJudgeResponse, judge_metadata
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
  retrieval_stage: str = 'selected'
  requires_judge_client = True

  def __init__(self, *, judge_client: LlmJudgeClient):
    self._judge_client = judge_client

  @property
  def target_span_types(self) -> tuple[str, ...]:
    return ('agent_root',)

  def matches_span(self, span: Span) -> bool:
    return any(
      result.kind == self.variant and result.stage == self.retrieval_stage for result in span.semantics.retrieval
    )

  def input_skip_reason(self, span: Span) -> str | None:
    return retrieval_skip_reason(span, self.variant, stage=self.retrieval_stage, require_content=True)

  @property
  def ground_truth_key(self) -> str:
    return GroundTruthKey.EXPECTED_OUTPUT.value

  @property
  def max_output_tokens(self) -> int:
    return 2_000

  async def compute(self, span: Span, ground_truth: GroundTruth | None) -> MetricComputationResult:
    assert ground_truth is not None

    skip_result = retrieval_skip_result(
      span, metric_name=self.name, variant=self.variant, stage=self.retrieval_stage, require_content=True
    )
    if skip_result is not None:
      return skip_result
    retrieved_items = self._extract_retrieved_items(span)
    if not retrieved_items:
      # Nothing retrieved can support a statement, so recall is 0 without calling the judge.
      return self._build_result(retrieved_items, [], [])
    decomposition_result = await self._decompose_claims(ground_truth)
    claims = decomposition_result.output.claims

    if not claims:
      return self._build_result(retrieved_items, [], [decomposition_result])

    judge_results, failure = await judge_batch(
      (self._judge_claim(span, retrieved_items, claim) for claim in claims),
      prior_responses=[decomposition_result],
    )
    if failure is not None:
      return failure

    judgments: list[ContextualRecallJudgment] = []
    responses: list[LlmJudgeResponse] = [decomposition_result, *judge_results]
    for claim, result in zip(claims, judge_results):
      payload = result.output
      if len(payload.judgments) != 1:
        return self._failed_judgment_count_result(claim, payload, result, responses)
      judgments.append(payload.judgments[0])

    return self._build_result(retrieved_items, judgments, responses)

  def build_decomposition_system_prompt(self) -> str:
    return (
      'You decompose an expected answer into atomic factual statements. '
      'Each statement must be self-contained, independently verifiable, and minimal in scope. '
      'Do not infer facts that are not stated in the expected answer, including entity-type claims such as '
      '"X is a person". If the expected answer is a name, list, fragment, or value, preserve each answer item '
      'verbatim instead of rewriting it into an inferred sentence. '
      'Return only the JSON object required by the schema.'
    )

  def build_decomposition_user_prompt(self, ground_truth: GroundTruth) -> str:
    return (
      'Decompose the expected answer into atomic factual statements.\n\n'
      f'Expected answer:\n{ground_truth.ground_truth_value["expected_output"]}'
    )

  def build_system_prompt(self) -> str:
    return (
      f'You judge whether one factual statement from the expected answer is attributable to '
      f'retrieved {self.variant}s. A statement is attributable when the retrieved {self.variant}s '
      f'contain enough information to directly support it. Return exactly one binary attribution '
      f'judgment for the statement. Return only the JSON object required by the schema.'
    )

  def build_user_prompt(self, span: Span, retrieved_items: list[RetrievedItem], claim: str) -> str:
    rendered_items = '\n\n'.join(item.render_for_prompt(rank) for rank, item in enumerate(retrieved_items, 1))
    sections = [
      f'Judge whether the statement is attributable to the retrieved {self.variant}s.',
      f'Statement:\n{claim}',
      f'Retrieved {self.variant}s:\n{rendered_items}',
    ]
    return '\n\n'.join(sections)

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
    judgments: list[ContextualRecallJudgment],
    responses: list[LlmJudgeResponse],
  ) -> MetricComputationResult:
    score, statement_results, attributable_count = self._score(judgments)
    expected_statement_count = len(judgments)

    metadata: dict[str, Any] = {
      'variant': self.variant,
      'retrieval_kind': self.variant,
      'counts': {
        'retrieved': len(retrieved_items),
        'expected_statements': expected_statement_count,
        'attributable': attributable_count,
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
