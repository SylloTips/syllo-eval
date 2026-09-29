"""Shared logic for set-based RAG metrics."""

from abc import ABC, abstractmethod

from syllo_eval.evaluation.metrics.contracts import MetricComputationResult, SpanEvaluationMetric
from syllo_eval.evaluation.metric_support.retrieved_context import (
  extract_retrieved_ids,
  retrieval_skip_reason,
  retrieval_skip_result,
  normalize_relevant_ids,
)
from syllo_eval.model import GroundTruth, GroundTruthKey, MetricComputationStatus, Span


class BaseSetRagMetric(SpanEvaluationMetric, ABC):
  """Base class with parsing/validation/metadata logic for set RAG metrics."""

  metric_kind: str = ''  # precision or recall
  variant: str = ''
  retrieval_stage: str = 'selected'

  @property
  def target_span_types(self) -> tuple[str, ...]:
    return ('agent_root',)

  def matches_span(self, span: Span) -> bool:
    return any(
      result.kind == self.variant and result.stage == self.retrieval_stage for result in span.semantics.retrieval
    )

  def input_skip_reason(self, span: Span) -> str | None:
    return retrieval_skip_reason(span, self.variant, stage=self.retrieval_stage)

  @property
  def ground_truth_key(self) -> str:
    key = GroundTruthKey.RELEVANT_SNIPPET_IDS if self.variant == 'snippet' else GroundTruthKey.RELEVANT_DOCUMENT_IDS
    return key.value

  async def compute(self, span: Span, ground_truth: GroundTruth | None) -> MetricComputationResult:
    """Compute score comparing predicted IDs to ground-truth relevant IDs."""
    skip_result = retrieval_skip_result(span, metric_name=self.name, variant=self.variant, stage=self.retrieval_stage)
    if skip_result is not None:
      return skip_result

    if ground_truth is None:
      return self._error_result('Missing ground truth for metric computation.')

    predicted_ids = extract_retrieved_ids(span, self.variant, self.retrieval_stage)

    relevant_ids, relevant_error = self._extract_relevant_ids(ground_truth)
    if relevant_error is not None:
      return self._error_result(relevant_error)

    matched_ids = predicted_ids.intersection(relevant_ids)
    false_positive_ids = predicted_ids - relevant_ids
    false_negative_ids = relevant_ids - predicted_ids
    denominator_kind = self._denominator_kind()
    denominator = len(predicted_ids) if denominator_kind == 'predicted' else len(relevant_ids)
    score = self._compute_score(predicted_ids=predicted_ids, relevant_ids=relevant_ids, matched_ids=matched_ids)

    return MetricComputationResult(
      score=score,
      reasoning=(
        f'{self.metric_kind} computed with {len(matched_ids)} matched IDs, '
        f'{len(predicted_ids)} predicted IDs, and {len(relevant_ids)} relevant IDs.'
      ),
      metadata={
        'metric_kind': self.metric_kind,
        'variant': self.variant,
        'retrieval_kind': self.variant,
        'score_parts': {
          'numerator': len(matched_ids),
          'denominator': denominator,
          'denominator_kind': denominator_kind,
        },
        'predicted_ids': sorted(predicted_ids),
        'relevant_ids': sorted(relevant_ids),
        'matched_ids': sorted(matched_ids),
        'true_positive_ids': sorted(matched_ids),
        'false_positive_ids': sorted(false_positive_ids),
        'false_negative_ids': sorted(false_negative_ids),
        'counts': {
          'predicted': len(predicted_ids),
          'relevant': len(relevant_ids),
          'matched': len(matched_ids),
          'true_positive': len(matched_ids),
          'false_positive': len(false_positive_ids),
          'false_negative': len(false_negative_ids),
        },
      },
    )

  @abstractmethod
  def _compute_score(self, predicted_ids: set[str], relevant_ids: set[str], matched_ids: set[str]) -> float:
    """Compute metric-specific score from ID sets."""

  def _denominator_kind(self) -> str:
    return 'predicted' if self.metric_kind == 'precision' else 'relevant'

  @staticmethod
  def _extract_relevant_ids(ground_truth: GroundTruth) -> tuple[set[str], str | None]:
    value = ground_truth.ground_truth_value
    if not isinstance(value, dict):
      return set(), 'ground_truth_value must be an object.'

    relevant_ids = value.get('relevant_ids')
    if not isinstance(relevant_ids, list):
      return set(), 'ground_truth_value.relevant_ids must be a list.'

    return normalize_relevant_ids(relevant_ids), None

  def _error_result(self, error_message: str) -> MetricComputationResult:
    return MetricComputationResult(
      score=None,
      status=MetricComputationStatus.FAILED,
      reasoning=f'Failed to compute {self.name}: {error_message}',
      metadata={
        'metric_kind': self.metric_kind,
        'variant': self.variant,
        'retrieval_kind': self.variant,
        'error': error_message,
      },
      error_message=error_message,
    )
