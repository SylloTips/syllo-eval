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
