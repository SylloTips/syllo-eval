"""Set-based precision metrics for RAG output IDs."""

from syllo_eval.evaluation.metrics.implementations.rag._set_base import BaseSetRagMetric


class _BaseSetPrecisionMetric(BaseSetRagMetric):
  metric_kind = 'precision'

  def _score_skip_reason(self, predicted_ids: set[str], relevant_ids: set[str]) -> str | None:
    if not predicted_ids and relevant_ids:
      return 'Nothing selected, so precision is undefined.'
    return None

  def _compute_score(self, predicted_ids: set[str], relevant_ids: set[str], matched_ids: set[str]) -> float:
    del relevant_ids
    # Nothing selected and nothing relevant scores 1.0 by convention.
    if not predicted_ids:
      return 1.0
    return len(matched_ids) / len(predicted_ids)


class SetPrecisionDocumentMetric(_BaseSetPrecisionMetric):
  variant = 'document'
  metric_name = 'set_precision_document'
  metric_description = 'Set-based precision on retrieved document IDs.'


class SetPrecisionSnippetMetric(_BaseSetPrecisionMetric):
  variant = 'snippet'
  metric_name = 'set_precision_snippet'
  metric_description = 'Set-based precision on retrieved snippet IDs.'
