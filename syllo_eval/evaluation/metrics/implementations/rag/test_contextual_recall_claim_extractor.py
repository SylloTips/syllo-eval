import unittest
from unittest.mock import AsyncMock
from syllo_eval.evaluation.metrics.test_support import Judge, span, truth
from syllo_eval.evaluation.claim_extractor import ClaimExtractionResult, ExtractedClaim
from syllo_eval.evaluation.metrics.implementations.rag.contextual_recall_claim_extractor import (
  ContextualRecallDocumentClaimExtractorMetric,
)
from syllo_eval.trace_semantics import RetrievalResult, RetrievalItem
from syllo_eval.model import MetricComputationStatus


class ClaimExtractorTest(unittest.IsolatedAsyncioTestCase):
  async def test_verifiable_claims_use_canonical_request_and_context(self):
    extractor = AsyncMock()
    extractor.extract.return_value = ClaimExtractionResult(
      claims=[
        ExtractedClaim(subtype='Factoid', content='A', evidences=[]),
        ExtractedClaim(subtype='Unverifiable', content='Nice', evidences=[]),
      ],
      model='fake',
      usage={},
      time_taken=0.1,
    )
    judge = Judge(
      {
        'judgments': [
          {'statement': 'A', 'attributable': True, 'supporting_retrieved_ids': ['d'], 'reasoning': 'evidence'}
        ]
      }
    )
    metric = ContextualRecallDocumentClaimExtractorMetric(judge_client=judge, claim_extractor_client=extractor)
    target = span(
      retrieval=[RetrievalResult(stage='selected', kind='document', items=[RetrievalItem(id='d', content='A')])]
    )
    result = await metric.compute(target, truth(expected_output='A'))
    self.assertEqual(result.score, 1)
    assert result.metadata is not None
    self.assertEqual(result.metadata['counts']['excluded_unverifiable_claims'], 1)
    self.assertEqual(extractor.extract.call_args.args[0][0].content, 'Question')
    result = await metric.compute(span(), truth(expected_output='A'))
    self.assertEqual(result.status, MetricComputationStatus.SKIPPED)
    self.assertEqual(extractor.extract.await_count, 1)

  async def test_empty_context_scores_zero_without_extracting_or_judging(self):
    extractor = AsyncMock()
    judge = Judge()
    metric = ContextualRecallDocumentClaimExtractorMetric(judge_client=judge, claim_extractor_client=extractor)
    target = span(retrieval=[RetrievalResult(stage='selected', kind='document', items=[])])
    result = await metric.compute(target, truth(expected_output='A'))
    self.assertEqual((result.status, result.score), (MetricComputationStatus.COMPLETED, 0.0))
    extractor.extract.assert_not_awaited()
    self.assertEqual(len(judge.requests), 0)
