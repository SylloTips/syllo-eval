import re
import unittest

from syllo_eval.evaluation.judge import LlmJudgeRequest, LlmJudgeResponse
from syllo_eval.evaluation.metrics.test_support import Judge, span, truth
from syllo_eval.evaluation.metrics.implementations.rag.contextual_precision_judge import (
  ContextualPrecisionDocumentJudgeMetric,
)
from syllo_eval.trace_semantics import RetrievalResult, RetrievalItem
from syllo_eval.model import MetricComputationStatus


class _RelevanceJudge:
  """Judges each prompt's document relevant when its id is listed, echoing the rank and id it was shown."""

  def __init__(self, relevant: set[str]):
    self.relevant = relevant
    self.requests: list[LlmJudgeRequest] = []

  async def judge(self, request: LlmJudgeRequest) -> LlmJudgeResponse:
    self.requests.append(request)
    shown = re.search(r'Rank (\d+)\nretrieved_id: (\S+)', request.user_prompt)
    assert shown is not None
    rank, document = int(shown.group(1)), shown.group(2)
    judgment = {'rank': rank, 'retrieved_id': document, 'relevant': document in self.relevant, 'reasoning': 'r'}
    return LlmJudgeResponse(
      provider='fake', model='test', output=request.response_model.model_validate({'judgments': [judgment]})
    )

  async def aclose(self) -> None:
    pass


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

  async def test_each_ranking_is_scored_on_its_own(self):
    judge = _RelevanceJudge(relevant={'a2', 'b1'})
    metric = ContextualPrecisionDocumentJudgeMetric(judge_client=judge)
    target = span(
      retrieval=[
        RetrievalResult(
          stage='selected',
          kind='document',
          query='first',
          items=[RetrievalItem(id=i, content='C') for i in ('a1', 'a2')],
        ),
        RetrievalResult(
          stage='selected',
          kind='document',
          query='second',
          items=[RetrievalItem(id=i, content='C') for i in ('b1', 'b2')],
        ),
      ]
    )
    result = await metric.compute(target, truth(expected_output='Expected'))
    assert result.metadata is not None and result.score is not None
    # Joined into one list, b1 would sit at rank 3; scored apart, it ranks first in its own search.
    self.assertAlmostEqual(result.score, (1 / 2 + 1) / 2)
    self.assertEqual([ranking['query'] for ranking in result.metadata['rankings']], ['first', 'second'])
    self.assertEqual(result.metadata['counts'], {'retrieved': 4, 'relevant': 2, 'echo_mismatches': 0, 'rankings': 2})
    self.assertEqual(sorted(request.user_prompt.count('Rank 1') for request in judge.requests), [0, 0, 1, 1])

  async def test_results_name_the_input_documents_not_the_echoed_ids(self):
    judge = Judge(
      {'judgments': [{'rank': 1, 'retrieved_id': 'a', 'relevant': True, 'reasoning': 'r'}]},
      {'judgments': [{'rank': 7, 'retrieved_id': 'typo', 'relevant': False, 'reasoning': 'r'}]},
    )
    metric = ContextualPrecisionDocumentJudgeMetric(judge_client=judge)
    target = span(
      retrieval=[
        RetrievalResult(stage='selected', kind='document', items=[RetrievalItem(id=i, content='C') for i in 'ab'])
      ]
    )
    result = await metric.compute(target, truth(expected_output='Expected'))
    assert result.metadata is not None and result.raw_output is not None
    self.assertEqual([(r['rank'], r['retrieved_id']) for r in result.metadata['rank_results']], [(1, 'a'), (2, 'b')])
    self.assertEqual(result.metadata['counts']['echo_mismatches'], 1)
    self.assertEqual(result.raw_output['judgments'][1]['retrieved_id'], 'typo')
