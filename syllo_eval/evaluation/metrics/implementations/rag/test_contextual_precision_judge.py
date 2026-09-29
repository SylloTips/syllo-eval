import unittest
from syllo_eval.evaluation.metrics.test_support import Judge, span, truth
from syllo_eval.evaluation.metrics.implementations.rag.contextual_precision_judge import (
  ContextualPrecisionDocumentJudgeMetric,
)
from syllo_eval.trace_semantics import RetrievalResult, RetrievalItem
from syllo_eval.model import MetricComputationStatus


class ContextualPrecisionTest(unittest.IsolatedAsyncioTestCase):
  async def test_ranked_relevance_and_usage(self):
    judge = Judge(
      *[
        {'judgments': [{'rank': i + 1, 'retrieved_id': str(i), 'relevant': relevant, 'reasoning': 'evidence'}]}
        for i, relevant in enumerate([True, False, True])
      ]
    )
    metric = ContextualPrecisionDocumentJudgeMetric(judge_client=judge)
    target = span(
      retrieval=[
        RetrievalResult(
          stage='selected', kind='document', items=[RetrievalItem(id=str(i), content='Evidence') for i in range(3)]
        )
      ]
    )
    result = await metric.compute(target, truth(expected_output='Expected'))
    assert result.score is not None
    self.assertAlmostEqual(result.score, (1 + 2 / 3) / 2)
    assert result.metadata is not None
    self.assertEqual(result.metadata['judge_usage']['total_tokens'], 9)
    self.assertIn('Evidence', judge.requests[0].user_prompt)

  async def test_empty_is_scored_missing_content_skips_and_bad_judgments_fail(self):
    judge = Judge({'judgments': []})
    metric = ContextualPrecisionDocumentJudgeMetric(judge_client=judge)
    target = span(retrieval=[RetrievalResult(stage='selected', kind='document')])
    self.assertEqual((await metric.compute(target, truth(expected_output='Expected'))).score, 0)
    self.assertEqual(len(judge.requests), 0)
    target.semantics.retrieval[0].items = [RetrievalItem(id='1')]
    self.assertEqual(
      (await metric.compute(target, truth(expected_output='Expected'))).status, MetricComputationStatus.SKIPPED
    )
    target.semantics.retrieval[0].items[0].content = 'Evidence'
    self.assertEqual(
      (await metric.compute(target, truth(expected_output='Expected'))).status, MetricComputationStatus.FAILED
    )

  async def test_unranked_context_skips_without_judging(self):
    judge = Judge()
    metric = ContextualPrecisionDocumentJudgeMetric(judge_client=judge)
    target = span(
      retrieval=[
        RetrievalResult(stage='selected', kind='document', items=[RetrievalItem(id='d', content='A')], ranked=False)
      ]
    )
    result = await metric.compute(target, truth(expected_output='A'))
    self.assertEqual(result.status, MetricComputationStatus.SKIPPED)
    self.assertEqual(len(judge.requests), 0)
