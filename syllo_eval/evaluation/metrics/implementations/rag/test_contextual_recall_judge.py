import unittest
from syllo_eval.evaluation.metrics.test_support import Judge, span, truth
from syllo_eval.evaluation.metrics.implementations.rag.contextual_recall_judge import (
  ContextualRecallDocumentJudgeMetric,
)
from syllo_eval.trace_semantics import RetrievalResult, RetrievalItem
from syllo_eval.model import MetricComputationStatus


class ContextualRecallTest(unittest.IsolatedAsyncioTestCase):
  async def test_claim_attribution_and_usage(self):
    judge = Judge(
      {'claims': ['A', 'B']},
      *[
        {
          'judgments': [
            {
              'statement': text,
              'attributable': supported,
              'supporting_retrieved_ids': ['d'] if supported else [],
              'reasoning': 'evidence',
            }
          ]
        }
        for text, supported in [('A', True), ('B', False)]
      ],
    )
    metric = ContextualRecallDocumentJudgeMetric(judge_client=judge)
    target = span(
      retrieval=[RetrievalResult(stage='selected', kind='document', items=[RetrievalItem(id='d', content='A')])]
    )
    result = await metric.compute(target, truth(expected_output='A and B'))
    self.assertEqual(result.score, 0.5)
    assert result.metadata is not None
    self.assertEqual(result.metadata['judge_usage']['total_tokens'], 9)

  async def test_missing_context_skips_and_bad_judgment_count_fails(self):
    judge = Judge({'claims': ['A']}, {'judgments': []})
    metric = ContextualRecallDocumentJudgeMetric(judge_client=judge)
    expected = truth(expected_output='A')
    self.assertEqual((await metric.compute(span(), expected)).status, MetricComputationStatus.SKIPPED)
    self.assertEqual(len(judge.requests), 0)
    target = span(
      retrieval=[RetrievalResult(stage='selected', kind='document', items=[RetrievalItem(id='d', content='A')])]
    )
    self.assertEqual((await metric.compute(target, expected)).status, MetricComputationStatus.FAILED)

  async def test_empty_context_scores_zero_without_judging(self):
    judge = Judge()
    metric = ContextualRecallDocumentJudgeMetric(judge_client=judge)
    target = span(retrieval=[RetrievalResult(stage='selected', kind='document', items=[])])
    result = await metric.compute(target, truth(expected_output='A'))
    self.assertEqual((result.status, result.score), (MetricComputationStatus.COMPLETED, 0.0))
    self.assertEqual(len(judge.requests), 0)
