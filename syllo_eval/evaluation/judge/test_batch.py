import asyncio
import unittest
from typing import Any
from unittest.mock import AsyncMock, patch

from syllo_eval.evaluation.judge.base import LlmJudgeResponse
from syllo_eval.evaluation.judge.batch import judge_batch
from syllo_eval.evaluation.claim_extractor import ClaimExtractionResult, ExtractedClaim
from syllo_eval.evaluation.metrics.implementations.rag.contextual_precision_judge import (
  ContextualPrecisionDocumentJudgeMetric,
)
from syllo_eval.evaluation.metrics.implementations.rag.contextual_recall_judge import (
  ContextualRecallDocumentJudgeMetric,
  ContextualRecallClaims,
)
from syllo_eval.evaluation.metrics.implementations.rag.contextual_recall_claim_extractor import (
  ContextualRecallDocumentClaimExtractorMetric,
)
from syllo_eval.evaluation.metrics.test_support import span, truth, Judge
from syllo_eval.model import MetricComputationStatus
from syllo_eval.trace_semantics import RetrievalResult, RetrievalItem


def response(output, tokens=3):
  return LlmJudgeResponse(provider='fake', model='test', output=output, usage={'total_tokens': tokens})


class JudgeBatchTest(unittest.IsolatedAsyncioTestCase):
  async def test_all_contextual_metrics_drain_siblings_and_keep_completed_usage_on_failure(self):
    for variant in ('precision', 'recall', 'extractor'):
      with self.subTest(variant=variant):
        blocked = asyncio.Event()
        cancelled = asyncio.Event()
        calls = 0

        async def invoke(*args):
          nonlocal calls
          index = calls
          calls += 1
          if index == 0:
            return response(ContextualRecallClaims(claims=[]))
          if index == 1:
            await blocked.wait()
            raise RuntimeError('judge unavailable')
          blocked.set()
          try:
            await asyncio.Future()
          finally:
            cancelled.set()

        extractor = AsyncMock()
        extractor.extract.return_value = ClaimExtractionResult(
          claims=[ExtractedClaim(content=c, subtype='Factoid', evidences=[]) for c in ('a', 'b', 'c')],
          model='fake',
          usage={'tokens': 5},
          time_taken=0.1,
        )
        metric: Any
        if variant == 'precision':
          metric = ContextualPrecisionDocumentJudgeMetric(judge_client=AsyncMock())
          method = '_judge_retrieved_item'
        elif variant == 'recall':
          metric = ContextualRecallDocumentJudgeMetric(judge_client=AsyncMock())
          metric._decompose_claims = AsyncMock(return_value=response(ContextualRecallClaims(claims=['a', 'b', 'c']), 7))
          method = '_judge_claim'
        else:
          metric = ContextualRecallDocumentClaimExtractorMetric(
            judge_client=AsyncMock(), claim_extractor_client=extractor
          )
          method = '_judge_claim'
        target = span(
          retrieval=[
            RetrievalResult(
              kind='document', stage='selected', items=[RetrievalItem(id=c, content=c) for c in ('a', 'b', 'c')]
            )
          ]
        )
        with patch.object(metric, method, side_effect=invoke):
          result = await metric.compute(target, truth(expected_output='answer'))
        self.assertTrue(cancelled.is_set())
        self.assertEqual(result.status, MetricComputationStatus.FAILED)
        self.assertEqual(result.error_message, 'judge unavailable')
        assert result.metadata is not None
        self.assertEqual(result.metadata['judge_usage']['total_tokens'], 10 if variant == 'recall' else 3)
        if variant == 'extractor':
          self.assertEqual(result.metadata['claim_extractor']['usage'], {'tokens': 5})

  async def test_cancellation_propagates_after_child_cleanup(self):
    started = asyncio.Event()
    cleaned = asyncio.Event()

    async def blocked():
      started.set()
      try:
        await asyncio.Future()
      finally:
        cleaned.set()
      return response(ContextualRecallClaims(claims=[]))

    task = asyncio.create_task(judge_batch([blocked()]))
    await started.wait()
    task.cancel()
    with self.assertRaises(asyncio.CancelledError):
      await task
    self.assertTrue(cleaned.is_set())

  async def test_invalid_judgment_count_keeps_usage_from_every_completed_request(self):
    judge = Judge({'judgments': []}, {'judgments': []})
    metric = ContextualPrecisionDocumentJudgeMetric(judge_client=judge)
    target = span(
      retrieval=[
        RetrievalResult(kind='document', stage='selected', items=[RetrievalItem(id=c, content=c) for c in ('a', 'b')])
      ]
    )
    result = await metric.compute(target, truth(expected_output='answer'))
    self.assertEqual(result.status, MetricComputationStatus.FAILED)
    assert result.metadata is not None
    self.assertEqual(result.metadata['judge_usage']['total_tokens'], 6)
