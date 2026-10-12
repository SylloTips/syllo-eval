"""Set-based recall metrics for RAG output IDs."""

from collections.abc import Mapping

from syllo_eval.evaluation.metrics.implementations.rag._set_base import BaseSetRagMetric
from syllo_eval.model import GroundTruth


class _BaseSetRecallMetric(BaseSetRagMetric):
  metric_kind = 'recall'

  def ground_truth_skip_reason(self, ground_truths: Mapping[str, GroundTruth]) -> str | None:
    # Malformed labels are left to compute, which fails them; with no relevant IDs there is nothing to recall.
    ground_truth = ground_truths.get(self.ground_truth_keys[0])
    if ground_truth is not None and self._extract_relevant_ids(ground_truth) == (set(), None):
      return 'No relevant IDs, so recall is undefined.'
    return super().ground_truth_skip_reason(ground_truths)

  def _compute_score(self, predicted_ids: set[str], relevant_ids: set[str], matched_ids: set[str]) -> float:
    del predicted_ids
    if not relevant_ids:
      return 1.0
    return len(matched_ids) / len(relevant_ids)


class SetRecallDocumentMetric(_BaseSetRecallMetric):
  variant = 'document'
  metric_name = 'set_recall_document'
  metric_description = 'Set-based recall on retrieved document IDs.'


class SetRecallSnippetMetric(_BaseSetRecallMetric):
  variant = 'snippet'
  metric_name = 'set_recall_snippet'
  metric_description = 'Set-based recall on retrieved snippet IDs.'
