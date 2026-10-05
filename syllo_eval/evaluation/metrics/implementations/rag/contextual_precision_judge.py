from typing import Any

from pydantic import BaseModel

from syllo_eval.evaluation.judge.batch import judge_batch

from syllo_eval.evaluation.judge import LlmJudgeClient, LlmJudgeRequest, LlmJudgeResponse, judge_metadata
from syllo_eval.evaluation.metrics.prompts import render_prompt
from syllo_eval.evaluation.metrics.contracts import MetricComputationResult, SpanEvaluationMetric
from syllo_eval.evaluation.metric_support.retrieved_context import (
  RetrievedItem,
  extract_retrieval_results,
  retrieval_skip_reason,
  retrieval_skip_result,
)
from syllo_eval.trace_semantics import RetrievalResult
from syllo_eval.model import GroundTruth, GroundTruthKey, MetricComputationStatus, Span


class ContextualPrecisionJudgment(BaseModel):
  rank: int
  retrieved_id: str
  relevant: bool
  reasoning: str


class ContextualPrecisionJudgePayload(BaseModel):
  judgments: list[ContextualPrecisionJudgment]


class _BaseContextualPrecisionJudgeMetric(SpanEvaluationMetric):
  """Scores whether relevant retrieved items appear early in the ranked list."""

  variant: str = ''
  prompt_version = 'v1'
  retrieval_stage: str = 'selected'
  requires_judge_client = True

  def __init__(self, *, judge_client: LlmJudgeClient, rubric_addition: str | None = None):
    self._judge_client = judge_client
    self._rubric_addition = rubric_addition

  @property
  def target_span_types(self) -> tuple[str, ...]:
    return ('agent_root',)

  def matches_span(self, span: Span) -> bool:
    return any(
      result.kind == self.variant and result.stage == self.retrieval_stage for result in span.semantics.retrieval
    )

  def input_skip_reason(self, span: Span) -> str | None:
    return retrieval_skip_reason(
      span, self.variant, stage=self.retrieval_stage, require_content=True, require_rank=True
    )

  @property
  def ground_truth_key(self) -> str:
    return GroundTruthKey.EXPECTED_OUTPUT.value

  @property
  def max_output_tokens(self) -> int:
    return 2_000

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
    # Each ranked result set is its own ranking: several are scored separately, never joined into one list.
    rankings = [
      (result, [RetrievedItem(item) for item in result.items])
      for result in extract_retrieval_results(span, self.variant, self.retrieval_stage)
    ]
    ranked_items = [(rank, item) for _, items in rankings for rank, item in enumerate(items, start=1)]

    judge_responses, failure = await judge_batch(
      self._judge_retrieved_item(span, ground_truth, rank, item) for rank, item in ranked_items
    )

    if failure is not None:
      return failure

    judgments: list[ContextualPrecisionJudgment] = []
    for (rank, _), result in zip(ranked_items, judge_responses):
      payload = result.output
      if len(payload.judgments) != 1:
        return self._failed_judgment_count_result(
          [item for _, item in ranked_items], rank, payload, result, judge_responses
        )
      judgments.append(payload.judgments[0])

    return self._build_result(rankings, judgments, judge_responses)

  def _build_result(
    self,
    rankings: list[tuple[RetrievalResult, list[RetrievedItem]]],
    judgments: list[ContextualPrecisionJudgment],
    responses: list[LlmJudgeResponse],
  ) -> MetricComputationResult:
    """Score each ranking, keying every judgment to its input item rather than to the rank and id the judge echoed."""
    scored: list[dict[str, Any]] = []
    echo_mismatches = 0
    remaining = iter(judgments)
    for result, items in rankings:
      echoed = [next(remaining) for _ in items]
      echo_mismatches += sum(
        (judgment.rank, judgment.retrieved_id) != (rank, item.retrieved_id)
        for rank, (item, judgment) in enumerate(zip(items, echoed), start=1)
      )
      score, rank_results, relevant_count = self._score(
        [
          judgment.model_copy(update={'rank': rank, 'retrieved_id': item.retrieved_id})
          for rank, (item, judgment) in enumerate(zip(items, echoed), start=1)
        ]
      )
      scored.append(
        {
          'query': result.query,
          'score': score,
          'counts': {'retrieved': len(items), 'relevant': relevant_count},
          'rank_results': rank_results,
        }
      )

    retrieved_count = sum(ranking['counts']['retrieved'] for ranking in scored)
    relevant_count = sum(ranking['counts']['relevant'] for ranking in scored)
    metadata: dict[str, Any] = {
      'variant': self.variant,
      'retrieval_kind': self.variant,
      'counts': {'retrieved': retrieved_count, 'relevant': relevant_count, 'echo_mismatches': echo_mismatches},
    }
    if len(scored) == 1:
      score = scored[0]['score']
      metadata['rank_results'] = scored[0]['rank_results']
      reasoning = (
        f'Contextual precision over {retrieved_count} retrieved {self.variant}s '
        f'with {relevant_count} relevant {self.variant}s.'
      )
    else:
      score = sum(ranking['score'] for ranking in scored) / len(scored)
      metadata['counts']['rankings'] = len(scored)
      metadata['rankings'] = scored
      reasoning = (
        f'Mean contextual precision over {len(scored)} rankings of {retrieved_count} retrieved {self.variant}s '
        f'with {relevant_count} relevant {self.variant}s.'
      )
    metadata.update(judge_metadata(responses))

    return MetricComputationResult(
      score=score,
      reasoning=reasoning,
      metadata=metadata,
      raw_output={'judgments': [judgment.model_dump() for judgment in judgments]},
    )

  def build_system_prompt(self) -> str:
    return render_prompt(
      f'contextual_precision/{self.prompt_version}/system.md',
      variant=self.variant,
      rubric_addition=self._rubric_addition,
    )

  def build_user_prompt(self, span: Span, ground_truth: GroundTruth, rank: int, item: RetrievedItem) -> str:
    return render_prompt(
      f'contextual_precision/{self.prompt_version}/user.md',
      variant=self.variant,
      request=span.semantics.request or '<empty>',
      expected_answer=ground_truth.ground_truth_value['expected_output'],
      item=item.render_for_prompt(rank),
    )

  async def _judge_retrieved_item(
    self,
    span: Span,
    ground_truth: GroundTruth,
    rank: int,
    item: RetrievedItem,
  ) -> LlmJudgeResponse[ContextualPrecisionJudgePayload]:
    return await self._judge_client.judge(
      LlmJudgeRequest(
        system_prompt=self.build_system_prompt(),
        user_prompt=self.build_user_prompt(span, ground_truth, rank, item),
        response_model=ContextualPrecisionJudgePayload,
        temperature=0.0,
        max_output_tokens=self.max_output_tokens,
      )
    )

  def _failed_judgment_count_result(
    self,
    retrieved_items: list[RetrievedItem],
    rank: int,
    payload: ContextualPrecisionJudgePayload,
    judge_response: LlmJudgeResponse,
    responses: list[LlmJudgeResponse],
  ) -> MetricComputationResult:
    error_message = (
      f'Judge returned {len(payload.judgments)} contextual precision judgments '
      f'for 1 retrieved {self.variant} at rank {rank}.'
    )
    error_metadata: dict[str, Any] = {
      'variant': self.variant,
      'retrieval_kind': self.variant,
      'counts': {'retrieved': len(retrieved_items), 'judgments': len(payload.judgments)},
      'rank': rank,
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
  def _score(judgments: list[ContextualPrecisionJudgment]) -> tuple[float, list[dict[str, Any]], int]:
    relevant_count = 0
    precision_sum = 0.0
    rank_results: list[dict[str, Any]] = []

    for rank, judgment in enumerate(judgments, 1):
      if judgment.relevant:
        relevant_count += 1
        precision_sum += relevant_count / rank

      rank_results.append(
        {
          'rank': rank,
          'retrieved_id': judgment.retrieved_id,
          'relevant': judgment.relevant,
          'precision_at_k': relevant_count / rank,
          'reasoning': judgment.reasoning,
        }
      )

    score = precision_sum / relevant_count if relevant_count else 0.0
    return score, rank_results, relevant_count


class ContextualPrecisionDocumentJudgeMetric(_BaseContextualPrecisionJudgeMetric):
  """Scores whether relevant retrieved documents appear early in the ranked list."""

  variant = 'document'
  metric_name = 'contextual_precision_document_judge'
  metric_description = 'Uses an LLM judge to compute contextual precision over ranked retrieved documents.'


class ContextualPrecisionSnippetJudgeMetric(_BaseContextualPrecisionJudgeMetric):
  """Scores whether relevant retrieved snippets appear early in the ranked list."""

  variant = 'snippet'
  metric_name = 'contextual_precision_snippet_judge'
  metric_description = 'Uses an LLM judge to compute contextual precision over ranked retrieved snippets.'
