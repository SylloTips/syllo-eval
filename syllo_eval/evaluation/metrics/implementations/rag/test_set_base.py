import unittest
from syllo_eval.evaluation.metrics.test_support import span, truth
from syllo_eval.evaluation.metrics.implementations.rag.set_precision import (
  SetPrecisionDocumentMetric,
  SetPrecisionSnippetMetric,
)
from syllo_eval.evaluation.metrics.implementations.rag.set_recall import SetRecallDocumentMetric
from syllo_eval.model import MetricComputationStatus
from syllo_eval.trace_semantics import RetrievalResult, RetrievalItem


class SetMetricsTest(unittest.IsolatedAsyncioTestCase):
  async def test_precision_recall_duplicates_and_empty_sets(self):
    target = span(
      retrieval=[
        RetrievalResult(stage='selected', kind='document', items=[RetrievalItem(id=i) for i in ['a', 'b', 'b']])
      ]
    )
    for metric, expected in [(SetPrecisionDocumentMetric(), 0.5), (SetRecallDocumentMetric(), 1 / 3)]:
      result = await metric.compute(target, truth(relevant_ids=['b', 'c', 'd']))
      self.assertEqual(result.score, expected)
      assert result.metadata is not None
      self.assertEqual(result.metadata['counts']['predicted'], 2)
    target.semantics.retrieval[0].items = []
    self.assertEqual((await SetPrecisionDocumentMetric().compute(target, truth(relevant_ids=[]))).score, 1)
    self.assertEqual((await SetRecallDocumentMetric().compute(target, truth(relevant_ids=['a']))).score, 0)
    result = await SetPrecisionDocumentMetric().compute(span(), truth(relevant_ids=[]))
    self.assertEqual(result.status, MetricComputationStatus.SKIPPED)

  async def test_empty_set_conventions(self):
    # (relevant, retrieved) -> (recall, precision); None is a skip. Precision scores 1.0 when both sets are empty.
    cases = {
      (('d1',), ()): (0.0, None),
      (('d1',), ('x',)): (0.0, 0.0),
      ((), ()): (None, 1.0),
      ((), ('x',)): (None, 0.0),
    }
    for (relevant, retrieved), (expected_recall, expected_precision) in cases.items():
      target = span(
        retrieval=[RetrievalResult(stage='selected', kind='document', items=[RetrievalItem(id=i) for i in retrieved])]
      )
      ground_truths = truth(relevant_ids=list(relevant))
      with self.subTest(relevant=relevant, retrieved=retrieved):
        recall = SetRecallDocumentMetric()
        if expected_recall is None:
          self.assertEqual(recall.ground_truth_skip_reason(ground_truths), 'No relevant IDs, so recall is undefined.')
        else:
          self.assertIsNone(recall.ground_truth_skip_reason(ground_truths))
          self.assertEqual((await recall.compute(target, ground_truths)).score, expected_recall)
        precision = SetPrecisionDocumentMetric()
        self.assertIsNone(precision.ground_truth_skip_reason(ground_truths))
        result = await precision.compute(target, ground_truths)
        if expected_precision is None:
          self.assertEqual(result.status, MetricComputationStatus.SKIPPED)
          self.assertEqual(result.error_message, 'Nothing selected, so precision is undefined.')
        else:
          self.assertEqual(result.score, expected_precision)

  def test_recall_leaves_malformed_labels_to_compute(self):
    self.assertIsNone(SetRecallDocumentMetric().ground_truth_skip_reason(truth(relevant_ids='invalid')))

  async def test_snippets_are_not_collapsed_to_parent_documents_or_filtered(self):
    target = span(
      retrieval=[
        RetrievalResult(
          stage='selected',
          kind='snippet',
          items=[
            RetrievalItem(id='s1', document_id='d'),
            RetrievalItem(id='s2', document_id='d', attributes={'example.is_adjacent': True}),
          ],
        )
      ]
    )
    result = await SetPrecisionSnippetMetric().compute(target, truth(relevant_ids=['s2']))
    self.assertEqual(result.score, 0.5)

  async def test_invalid_ground_truth_is_reported(self):
    target = span(retrieval=[RetrievalResult(stage='selected', kind='document')])
    result = await SetRecallDocumentMetric().compute(target, truth(relevant_ids='invalid'))
    self.assertEqual(result.status, MetricComputationStatus.FAILED)
