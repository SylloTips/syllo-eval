import unittest
from typing import Any
from uuid import uuid4

from syllo_eval.evaluation.metric_support.retrieved_context import extract_retrieved_items
from syllo_eval.evaluation.metrics.implementations.rag.contextual_recall_judge import (
  ContextualRecallDocumentJudgeMetric,
)
from syllo_eval.evaluation.metrics.implementations.rag.contextual_recall_stored_claims import (
  ContextualRecallDocumentStoredClaimsMetric,
)
from syllo_eval.evaluation.metrics.test_support import Judge, span
from syllo_eval.model import GroundTruth, GroundTruthKey
from syllo_eval.trace_semantics import RetrievalItem, RetrievalResult

CLAIMS = {'claims': [{'id': 'c1', 'text': 'Dana approved it.'}, {'id': 'c2', 'text': 'In May.'}]}


def claims_truth(value: dict[str, Any]) -> GroundTruth:
  return GroundTruth(id=uuid4(), sample_id=uuid4(), key=GroundTruthKey.EXPECTED_CLAIMS.value, ground_truth_value=value)


def judgment(statement: str, attributable: bool) -> dict[str, Any]:
  return {
    'judgments': [
      {'statement': statement, 'attributable': attributable, 'supporting_retrieved_ids': [], 'reasoning': 'r'}
    ]
  }


def context() -> RetrievalResult:
  return RetrievalResult(stage='selected', kind='document', items=[RetrievalItem(id='d', content='Dana approved it.')])


class ContextualRecallStoredClaimsTest(unittest.IsolatedAsyncioTestCase):
  async def test_judges_each_stored_claim_with_the_built_in_prompt_and_no_decomposition(self):
    judge = Judge(judgment('Dana approved it.', True), judgment('In May.', False))
    metric = ContextualRecallDocumentStoredClaimsMetric(judge_client=judge)
    target = span(retrieval=[context()])

    result = await metric.compute(target, claims_truth(CLAIMS))

    self.assertEqual(metric.ground_truth_key, 'expected_claims')
    self.assertEqual(result.score, 0.5)
    assert result.metadata is not None
    self.assertEqual(
      [(entry['claim_id'], entry['statement']) for entry in result.metadata['statement_results']],
      [('c1', 'Dana approved it.'), ('c2', 'In May.')],
    )
    self.assertEqual(result.metadata['judge_calls'], 2)
    built_in = ContextualRecallDocumentJudgeMetric(judge_client=judge)
    items = extract_retrieved_items(target, 'document')
    self.assertEqual(
      [request.user_prompt for request in judge.requests],
      [built_in.build_user_prompt(target, items, claim['text']) for claim in CLAIMS['claims']],
    )

  async def test_nothing_retrieved_scores_zero_without_judging(self):
    judge = Judge()
    metric = ContextualRecallDocumentStoredClaimsMetric(judge_client=judge)

    result = await metric.compute(
      span(retrieval=[RetrievalResult(stage='selected', kind='document')]), claims_truth(CLAIMS)
    )

    self.assertEqual(result.score, 0.0)
    assert result.metadata is not None
    self.assertEqual(result.metadata['counts']['expected_statements'], 2)
    self.assertEqual(judge.requests, [])

  async def test_malformed_claims_raise_instead_of_scoring(self):
    metric = ContextualRecallDocumentStoredClaimsMetric(judge_client=Judge())
    values: list[dict[str, Any]] = [
      {},
      {'claims': []},
      {'claims': [{'id': 'c1'}]},
      {'claims': [{'id': '', 'text': 'A.'}]},
    ]
    for value in values:
      with self.subTest(value=value), self.assertRaises(ValueError):
        await metric.compute(span(retrieval=[context()]), claims_truth(value))


if __name__ == '__main__':
  unittest.main()
