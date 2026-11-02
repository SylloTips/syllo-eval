import unittest
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

import httpx
from google.genai.errors import APIError
from langchain_core.messages import AIMessage

from syllo_eval.evaluation.metric_support.retrieved_context import extract_retrieved_items
from syllo_eval.evaluation.metrics.implementations.rag.contextual_recall_stored_claims import (
  ContextualRecallDocumentStoredClaimsMetric,
)
from syllo_eval.evaluation.metrics.test_support import span
from syllo_eval.infrastructure.llm_judge.gemini import GeminiLlmJudgeClient
from syllo_eval.model import GroundTruth, GroundTruthKey, MetricComputationStatus
from syllo_eval.settings import GeminiJudgeSettings
from syllo_eval.trace_semantics import RetrievalItem, RetrievalResult

CLAIMS = [{'id': 'c1', 'text': 'Dana approved it.'}, {'id': 'c2', 'text': 'In May.'}]


def claims_truth(*claims: dict[str, str]) -> dict[str, GroundTruth]:
  key = GroundTruthKey.EXPECTED_CLAIMS.value
  return {key: GroundTruth(id=uuid4(), sample_id=uuid4(), key=key, ground_truth_value={'claims': list(claims)})}


def reply(messages: list[tuple[str, str]]) -> dict[str, Any]:
  judgment = {'statement': messages[-1][1], 'attributable': True, 'supporting_retrieved_ids': ['d'], 'reasoning': 'r'}
  return {
    'raw': AIMessage(content='', usage_metadata={'input_tokens': 10, 'output_tokens': 2, 'total_tokens': 12}),
    'parsed': {'judgments': [judgment]},
    'parsing_error': None,
  }


class GeminiSharedPromptPrefixTest(unittest.IsolatedAsyncioTestCase):
  def setUp(self) -> None:
    self.chat_model = MagicMock()
    runnable = self.chat_model.model_copy.return_value.with_structured_output.return_value
    self.ainvoke = runnable.ainvoke = AsyncMock(side_effect=reply)
    self.caches = self.chat_model.async_client.caches
    self.caches.create = AsyncMock(
      return_value=SimpleNamespace(name='cachedContents/c1', usage_metadata=SimpleNamespace(total_token_count=5000))
    )
    self.caches.delete = AsyncMock()
    client = GeminiLlmJudgeClient(
      GeminiJudgeSettings(api_key='key', model='judge-1', max_retries=1), 4, chat_model=self.chat_model
    )
    self.metric = ContextualRecallDocumentStoredClaimsMetric(judge_client=client)
    self.target = span(
      retrieval=[RetrievalResult(stage='selected', kind='document', items=[RetrievalItem(id='d', content='Text.')])]
    )
    self.system = self.metric.build_system_prompt()
    self.prefix = self.metric.build_user_prompt(self.target, extract_retrieved_items(self.target, 'document'), '')

  def sent(self) -> list[list[tuple[str, str]]]:
    return [call.args[0] for call in self.ainvoke.call_args_list]

  def cached_content(self) -> list[str | None]:
    return [call.kwargs['update'].get('cached_content') for call in self.chat_model.model_copy.call_args_list]

  async def test_claims_are_judged_against_one_cache_of_the_documents(self) -> None:
    result = await self.metric.compute(self.target, claims_truth(*CLAIMS))

    self.assertEqual(result.score, 1.0)
    self.caches.create.assert_awaited_once()
    contents = self.caches.create.call_args.kwargs['config'].contents
    self.assertEqual((contents.role, [part.text for part in contents.parts]), ('user', [self.system, self.prefix]))
    self.assertCountEqual(self.sent(), [[('human', claim['text'])] for claim in CLAIMS])
    self.assertEqual(self.cached_content(), ['cachedContents/c1'] * 2)
    self.caches.delete.assert_awaited_once_with(name='cachedContents/c1')
    assert result.metadata is not None
    self.assertEqual(result.metadata['judge_usage']['cache_creation_input_tokens'], 5000)
    self.assertEqual(result.metadata['judge_usage']['input_tokens'], 20)
    self.assertIn('judge_cache_seconds', result.metadata)

  async def test_a_single_claim_keeps_the_two_turn_layout_without_a_cache(self) -> None:
    result = await self.metric.compute(self.target, claims_truth(CLAIMS[0]))

    self.caches.create.assert_not_awaited()
    self.assertEqual(self.sent(), [[('system', self.system), ('human', self.prefix), ('human', CLAIMS[0]['text'])]])
    self.assertEqual(self.cached_content(), [None])
    assert result.metadata is not None
    self.assertNotIn('judge_cache_seconds', result.metadata)

  async def test_a_cache_that_cannot_be_created_leaves_the_calls_uncached(self) -> None:
    for error in (APIError(500, {'error': {'message': 'unavailable'}}), httpx.ReadTimeout('slow')):
      with self.subTest(error=type(error).__name__):
        self.setUp()
        self.caches.create.side_effect = error

        with self.assertLogs('syllo_eval.infrastructure.llm_judge.gemini', 'WARNING'):
          result = await self.metric.compute(self.target, claims_truth(*CLAIMS))

        self.assertEqual(result.score, 1.0)
        self.assertCountEqual(
          self.sent(),
          [[('system', self.system), ('human', self.prefix), ('human', claim['text'])] for claim in CLAIMS],
        )
        self.assertEqual(self.cached_content(), [None, None])
        self.caches.delete.assert_not_awaited()
        assert result.metadata is not None
        self.assertNotIn('cache_creation_input_tokens', result.metadata['judge_usage'])
        self.assertIn('judge_cache_seconds', result.metadata)

  async def test_a_cache_that_cannot_be_deleted_keeps_the_judgments(self) -> None:
    self.caches.delete.side_effect = httpx.ReadTimeout('slow')

    with self.assertLogs('syllo_eval.infrastructure.llm_judge.gemini', 'WARNING'):
      result = await self.metric.compute(self.target, claims_truth(*CLAIMS))

    self.assertEqual((result.status, result.score), (MetricComputationStatus.COMPLETED, 1.0))
    assert result.metadata is not None
    self.assertEqual(result.metadata['judge_usage']['cache_creation_input_tokens'], 5000)

  async def test_a_failed_unit_still_counts_the_cache_creation(self) -> None:
    self.ainvoke.side_effect = ValueError('bad request')

    result = await self.metric.compute(self.target, claims_truth(*CLAIMS))

    self.assertEqual(result.status, MetricComputationStatus.FAILED)
    self.caches.delete.assert_awaited_once_with(name='cachedContents/c1')
    assert result.metadata is not None
    self.assertEqual(result.metadata['judge_usage'], {'cache_creation_input_tokens': 5000})
    self.assertIn('judge_cache_seconds', result.metadata)


if __name__ == '__main__':
  unittest.main()
