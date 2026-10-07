import math
import unittest
from syllo_eval.evaluation.metrics.test_support import span, truth
from syllo_eval.evaluation.metrics.implementations.rag.ndcg_at_10 import (
  NdcgAt10DocumentMetric,
  NdcgAt10SnippetMetric,
)
from syllo_eval.trace_semantics import RetrievalResult, RetrievalItem
from syllo_eval.model import MetricComputationStatus


class NdcgTest(unittest.IsolatedAsyncioTestCase):
  async def test_observed_order_wins_over_scores_and_duplicates_count_once(self):
    target = span(
      retrieval=[
        RetrievalResult(
          stage='selected',
          kind='document',
          items=[RetrievalItem(id='wrong', score=0), RetrievalItem(id='right', score=100), RetrievalItem(id='right')],
        )
      ]
    )
    result = await NdcgAt10DocumentMetric().compute(target, truth(relevant_ids=['right']))
    assert result.score is not None
    self.assertAlmostEqual(result.score, 1 / math.log2(3))
    assert result.metadata is not None
    self.assertEqual(result.metadata['ranked_ids'], ['wrong', 'right'])

  async def test_snippet_ids_and_top_ten_cutoff(self):
    target = span(
      retrieval=[
        RetrievalResult(
          stage='selected', kind='snippet', items=[RetrievalItem(id=str(i), document_id='shared') for i in range(11)]
        )
      ]
    )
    metric = NdcgAt10SnippetMetric()
    self.assertEqual((await metric.compute(target, truth(relevant_ids=['10']))).score, 0)
    self.assertEqual((await metric.compute(target, truth(relevant_ids=['0']))).score, 1)

  async def test_unranked_context_skips(self):
    target = span(
      retrieval=[RetrievalResult(stage='selected', kind='snippet', items=[RetrievalItem(id='a')], ranked=False)]
    )
    metric = NdcgAt10SnippetMetric()
    self.assertEqual(metric.input_skip_reason(target), 'Snippet selected context is not ranked.')
    self.assertEqual((await metric.compute(target, truth(relevant_ids=['a']))).status, MetricComputationStatus.SKIPPED)

  async def test_each_ranking_is_scored_on_its_own(self):
    target = span(
      retrieval=[
        RetrievalResult(
          stage='selected', kind='document', query='first', items=[RetrievalItem(id='x'), RetrievalItem(id='right')]
        ),
        RetrievalResult(stage='selected', kind='document', query='second', items=[RetrievalItem(id='right')]),
      ]
    )
    result = await NdcgAt10DocumentMetric().compute(target, truth(relevant_ids=['right']))
    assert result.score is not None and result.metadata is not None
    # Joined and deduplicated, the second search would vanish; scored apart, it is a perfect ranking.
    self.assertAlmostEqual(result.score, (1 / math.log2(3) + 1) / 2)
    self.assertEqual([ranking['ranked_ids'] for ranking in result.metadata['rankings']], [['x', 'right'], ['right']])
    self.assertEqual(result.metadata['counts']['rankings'], 2)

  def test_empty_relevant_ids_skip_and_malformed_ones_are_left_to_compute(self):
    metric = NdcgAt10DocumentMetric()
    self.assertEqual(
      metric.ground_truth_skip_reason(truth(relevant_ids=[])), 'No relevant IDs, so NDCG@10 is undefined.'
    )
    self.assertIsNone(metric.ground_truth_skip_reason(truth(relevant_ids='invalid')))
    self.assertIsNone(metric.ground_truth_skip_reason(truth(relevant_ids=['d1'])))
