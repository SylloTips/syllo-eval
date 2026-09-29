"""Set-based recall metrics for RAG output IDs."""

from syllo_eval.evaluation.metrics.implementations.rag._set_base import BaseSetRagMetric


class _BaseSetRecallMetric(BaseSetRagMetric):
  metric_kind = 'recall'

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
