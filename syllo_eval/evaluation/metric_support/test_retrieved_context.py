import unittest
from syllo_eval.evaluation.metrics.test_support import span
from syllo_eval.evaluation.metric_support.retrieved_context import (
  extract_retrieved_items,
  retrieval_skip_reason,
  normalize_relevant_ids,
)
from syllo_eval.trace_semantics import RetrievalResult, RetrievalItem


class CanonicalRetrievalTest(unittest.TestCase):
  def test_identity_order_and_unknown_score_are_preserved(self):
    target = span(
      retrieval=[
        RetrievalResult(
          stage='selected',
          kind='snippet',
          items=[
            RetrievalItem(id='s1', document_id='d', score=None, attributes={'vendor.adjacent': True}),
            RetrievalItem(id='s2', document_id='d', score=100),
          ],
        )
      ]
    )
    items = extract_retrieved_items(target, 'snippet')
    self.assertEqual([item.retrieved_id for item in items], ['s1', 's2'])
    self.assertIsNone(items[0].score)
    self.assertIsNotNone(retrieval_skip_reason(target, 'snippet', require_content=True))

  def test_missing_differs_from_observed_empty_and_unavailable(self):
    self.assertIsNotNone(retrieval_skip_reason(span(), 'document'))
    empty = span(retrieval=[RetrievalResult(stage='selected', kind='document')])
    self.assertIsNone(retrieval_skip_reason(empty, 'document', require_content=True))
    empty.semantics.retrieval[0].availability = 'unavailable'
    self.assertIsNotNone(retrieval_skip_reason(empty, 'document'))

  def test_relevant_ids_normalize_strings_and_id_objects_consistently(self):
    self.assertEqual(normalize_relevant_ids([' a ', 'a', {'id': ' b '}, '', None, 12, {}]), {'a', 'b'})
