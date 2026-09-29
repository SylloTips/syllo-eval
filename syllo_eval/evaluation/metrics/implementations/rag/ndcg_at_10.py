import math
from abc import ABC
from typing import Any

from syllo_eval.evaluation.metrics.contracts import MetricComputationResult, SpanEvaluationMetric
from syllo_eval.evaluation.metric_support.retrieved_context import (
  extract_retrieved_items,
  normalize_relevant_ids,
  retrieval_skip_reason,
  retrieval_skip_result,
)
from syllo_eval.model import GroundTruth, GroundTruthKey, MetricComputationStatus, Span

TOP_K = 10


class BaseNdcgAt10Metric(SpanEvaluationMetric, ABC):
  """Computes binary-relevance NDCG@10 over observed retrieval order."""

  variant: str = ''
  retrieval_stage: str = 'selected'
  metric_kind = 'ndcg_at_10'

  @property
  def target_span_types(self) -> tuple[str, ...]:
    return ('agent_root',)

  def matches_span(self, span: Span) -> bool:
    return any(
      result.kind == self.variant and result.stage == self.retrieval_stage for result in span.semantics.retrieval
    )

  def input_skip_reason(self, span: Span) -> str | None:
    return retrieval_skip_reason(span, self.variant, stage=self.retrieval_stage, require_rank=True)

  @property
  def ground_truth_key(self) -> str:
    key = GroundTruthKey.RELEVANT_SNIPPET_IDS if self.variant == 'snippet' else GroundTruthKey.RELEVANT_DOCUMENT_IDS
    return key.value

  async def compute(self, span: Span, ground_truth: GroundTruth | None) -> MetricComputationResult:
    skip_result = retrieval_skip_result(
      span, metric_name=self.name, variant=self.variant, stage=self.retrieval_stage, require_rank=True
    )
    if skip_result is not None:
      return skip_result
    if ground_truth is None:
      return self._error_result('Missing ground truth for metric computation.')

    ranked_ids = self._ranked_ids(span)
    relevant_ids, relevant_error = self._extract_relevant_ids(ground_truth)
    if relevant_error is not None:
      return self._error_result(relevant_error)

    dcg_at_k, rank_results, matched_at_k = self._discounted_cumulative_gain(ranked_ids, relevant_ids)
    idcg_at_k = self._ideal_discounted_cumulative_gain(len(relevant_ids))
    score = dcg_at_k / idcg_at_k if idcg_at_k else 0.0

    return MetricComputationResult(
      score=score,
      reasoning=(
        f'NDCG@10 computed with {len(relevant_ids)} relevant IDs across {len(ranked_ids)} ranked {self.variant} IDs.'
      ),
      metadata={
        'metric_kind': self.metric_kind,
        'variant': self.variant,
        'top_k': TOP_K,
        'ranked_ids': ranked_ids,
        'relevant_ids': sorted(relevant_ids),
        'dcg_at_k': dcg_at_k,
        'idcg_at_k': idcg_at_k,
        'rank_results': rank_results,
        'counts': {
          'ranked': len(ranked_ids),
          'relevant': len(relevant_ids),
          'matched_at_k': matched_at_k,
          'top_k': TOP_K,
        },
      },
    )

  def _ranked_ids(self, span: Span) -> list[str]:
    seen: set[str] = set()
    ranked_ids: list[str] = []
    for item in extract_retrieved_items(span, self.variant, self.retrieval_stage):
      item_id = item.retrieved_id
      if item_id not in seen:
        seen.add(item_id)
        ranked_ids.append(item_id)
    return ranked_ids

  @staticmethod
  def _extract_relevant_ids(ground_truth: GroundTruth) -> tuple[set[str], str | None]:
    value = ground_truth.ground_truth_value
    if not isinstance(value, dict):
      return set(), 'ground_truth_value must be an object.'
    relevant_ids = value.get('relevant_ids')
    if not isinstance(relevant_ids, list):
      return set(), 'ground_truth_value.relevant_ids must be a list.'
    return normalize_relevant_ids(relevant_ids), None

  @staticmethod
  def _discounted_cumulative_gain(
    ranked_ids: list[str], relevant_ids: set[str]
  ) -> tuple[float, list[dict[str, Any]], int]:
    dcg = 0.0
    matched = 0
    rank_results: list[dict[str, Any]] = []
    for rank, retrieved_id in enumerate(ranked_ids[:TOP_K], 1):
      gain = int(retrieved_id in relevant_ids)
      discount = 1 / math.log2(rank + 1)
      discounted_gain = gain * discount
      dcg += discounted_gain
      matched += gain
      rank_results.append(
        {
          'rank': rank,
          'retrieved_id': retrieved_id,
          'gain': gain,
          'discount': discount,
          'discounted_gain': discounted_gain,
        }
      )
    return dcg, rank_results, matched

  @staticmethod
  def _ideal_discounted_cumulative_gain(relevant_count: int) -> float:
    return sum(1 / math.log2(rank + 1) for rank in range(1, min(relevant_count, TOP_K) + 1))

  def _error_result(self, error_message: str) -> MetricComputationResult:
    return MetricComputationResult(
      score=None,
      status=MetricComputationStatus.FAILED,
      reasoning=f'Failed to compute {self.name}: {error_message}',
      metadata={'metric_kind': self.metric_kind, 'variant': self.variant, 'top_k': TOP_K, 'error': error_message},
      error_message=error_message,
    )


class NdcgAt10DocumentMetric(BaseNdcgAt10Metric):
  variant = 'document'
  metric_name = 'ndcg_at_10_document'
  metric_description = 'Binary-relevance NDCG@10 on observed document retrieval order.'


class NdcgAt10SnippetMetric(BaseNdcgAt10Metric):
  variant = 'snippet'
  metric_name = 'ndcg_at_10_snippet'
  metric_description = 'Binary-relevance NDCG@10 on observed snippet retrieval order.'
