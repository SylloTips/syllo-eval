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

  async def test_results_name_the_claims_not_the_echoed_statements(self):
    judge = Judge(
      {'claims': ['Dana approved it.', 'In May.']},
      {
        'judgments': [
          {'statement': '1. dana approved it', 'attributable': True, 'supporting_retrieved_ids': [], 'reasoning': 'r'}
        ]
      },
      {
        'judgments': [
          {'statement': 'In March.', 'attributable': False, 'supporting_retrieved_ids': [], 'reasoning': 'r'}
        ]
      },
    )
    metric = ContextualRecallDocumentJudgeMetric(judge_client=judge)
    target = span(
      retrieval=[RetrievalResult(stage='selected', kind='document', items=[RetrievalItem(id='d', content='A')])]
    )
    result = await metric.compute(target, truth(expected_output='Dana approved it in May.'))
    assert result.metadata is not None and result.raw_output is not None
    self.assertEqual(
      [entry['statement'] for entry in result.metadata['statement_results']], ['Dana approved it.', 'In May.']
    )
    # Case, punctuation and a list number are no mismatch; a changed fact is.
    self.assertEqual(result.metadata['counts']['echo_mismatches'], 1)
    self.assertEqual(result.raw_output['judgments'][1]['statement'], 'In March.')
