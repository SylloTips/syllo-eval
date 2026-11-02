import asyncio
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

from langchain_core.messages import AIMessage
from pydantic import BaseModel

from syllo_eval.evaluation.judge import LlmJudgeRequest, LlmJudgeResponse
from syllo_eval.evaluation.metrics.implementations.rag.contextual_precision_judge import (
  ContextualPrecisionItemJudgePayload,
  ContextualPrecisionItemJudgment,
)
from syllo_eval.evaluation.plan_executor import sample_cache
from syllo_eval.infrastructure.llm_judge.gemini import GeminiLlmJudgeClient
from syllo_eval.model import MetricComputationStatus
from syllo_eval.settings import GeminiJudgeSettings
from ablation_metrics import SEARCH_SPAN_TYPE, GoldClaimsContextualRecall
from benchmarks.common import Claim
from config import JudgeConfig
from judge import ConfiguredTemperatureJudgeClient, ReusingJudgeClient, StoredPrecisionJudgeClient, open_judge_client
from metric_fakes import claims_truth, search_span, truths


class _Payload(BaseModel):
  value: int


class _RecordingJudge:
  def __init__(self) -> None:
    self.requests: list[LlmJudgeRequest] = []
    self.closed = False

  async def judge(self, request: LlmJudgeRequest) -> LlmJudgeResponse:
    self.requests.append(request)
    return LlmJudgeResponse(provider='fake', model='judge', output=_Payload(value=1))

  async def aclose(self) -> None:
    self.closed = True


def _judge_config(**overrides: Any) -> JudgeConfig:
  fields: dict[str, Any] = {
    'provider': 'gemini',
    'model': 'judge-1',
    'label': 'Judge',
    'temperature': 1.0,
    'thinking_level': 'low',
    'timeout_seconds': 300,
    'max_retries': 7,
    'max_concurrent_requests': 3,
    'output_token_limit': 65536,
  }
  return JudgeConfig(**{**fields, **overrides})


class ConfiguredTemperatureJudgeClientTest(unittest.IsolatedAsyncioTestCase):
  async def test_overrides_the_temperature_metrics_request_and_closes_the_inner_client(self) -> None:
    inner = _RecordingJudge()
    client = ConfiguredTemperatureJudgeClient(inner, temperature=1.0)
    request = LlmJudgeRequest(system_prompt='system', user_prompt='user', response_model=_Payload, temperature=0.0)

    await client.judge(request)
    await client.aclose()

    self.assertEqual(inner.requests[0].temperature, 1.0)
    self.assertEqual(inner.requests[0].user_prompt, 'user')
    self.assertEqual(request.temperature, 0.0)
    self.assertTrue(inner.closed)


class _ScriptedJudge:
  def __init__(self, *results: LlmJudgeResponse | Exception) -> None:
    self.calls = 0
    self._results = list(results)

  async def judge(self, request: LlmJudgeRequest) -> LlmJudgeResponse:
    self.calls += 1
    await asyncio.sleep(0)
    result = self._results.pop(0)
    if isinstance(result, Exception):
      raise result
    return result

  async def aclose(self) -> None:
    pass


def _request(user_prompt: str = 'user') -> LlmJudgeRequest:
  return LlmJudgeRequest(system_prompt='system', user_prompt=user_prompt, response_model=_Payload)


def _response(value: int) -> LlmJudgeResponse:
  return LlmJudgeResponse(
    provider='fake', model='judge', output=_Payload(value=value), usage={'input_tokens': 9}, latency_seconds=2.0
  )


class ReusingJudgeClientTest(unittest.IsolatedAsyncioTestCase):
  async def test_an_identical_request_of_the_sample_reuses_the_first_verdict_and_reports_no_tokens(self) -> None:
    inner = _ScriptedJudge(_response(1), _response(2))
    client = ReusingJudgeClient(inner)
    sample_cache.set({})

    first, again, other = await asyncio.gather(
      client.judge(_request()), client.judge(_request()), client.judge(_request('other'))
    )

    self.assertEqual(inner.calls, 2)
    self.assertEqual((first.output, first.usage), (_Payload(value=1), {'input_tokens': 9}))
    self.assertEqual((again.output, again.usage, again.latency_seconds), (_Payload(value=1), {'input_tokens': 0}, 0.0))
    self.assertEqual(other.output, _Payload(value=2))

  async def test_another_sample_or_no_sample_calls_the_judge_again(self) -> None:
    inner = _ScriptedJudge(_response(1), _response(2), _response(3))
    client = ReusingJudgeClient(inner)

    await client.judge(_request())
    for _ in range(2):
      sample_cache.set({})
      await client.judge(_request())

    self.assertEqual(inner.calls, 3)

  async def test_a_failed_call_is_not_reused(self) -> None:
    inner = _ScriptedJudge(RuntimeError('busy'), _response(1))
    client = ReusingJudgeClient(inner)
    sample_cache.set({})

    with self.assertRaises(RuntimeError):
      await client.judge(_request())
    response = await client.judge(_request())

    self.assertEqual((inner.calls, response.output), (2, _Payload(value=1)))


class StoredPrecisionJudgeClientTest(unittest.IsolatedAsyncioTestCase):
  async def test_a_precision_verdict_is_reused_across_runs_of_its_scope_with_its_original_usage(self) -> None:
    precision = LlmJudgeRequest(
      system_prompt='system', user_prompt='doc', response_model=ContextualPrecisionItemJudgePayload
    )
    verdict = LlmJudgeResponse(
      provider='fake',
      model='judge',
      output=ContextualPrecisionItemJudgePayload(
        judgments=[ContextualPrecisionItemJudgment(retrieved_id='d1', relevant=True, reasoning='r')]
      ),
      usage={'input_tokens': 9},
      latency_seconds=2.0,
    )
    with tempfile.TemporaryDirectory() as folder:
      path = Path(folder) / 'outputs' / 'verdicts.sqlite'
      first_run = StoredPrecisionJudgeClient(_ScriptedJudge(verdict, _response(1)), path, 'repetition-1')
      self.assertEqual(await first_run.judge(precision), verdict)
      await first_run.judge(_request())
      await first_run.aclose()

      later_inner, retest_inner = _ScriptedJudge(_response(2)), _ScriptedJudge(verdict)
      later_run = StoredPrecisionJudgeClient(later_inner, path, 'repetition-1')
      retest = StoredPrecisionJudgeClient(retest_inner, path, 'repetition-2')
      self.assertEqual(await later_run.judge(precision), verdict)
      await later_run.judge(_request())
      await retest.judge(precision)
      await later_run.aclose()
      await retest.aclose()

    self.assertEqual((later_inner.calls, retest_inner.calls), (1, 1))

  async def test_runs_that_miss_the_same_verdict_at_once_both_return_the_stored_one(self) -> None:
    precision = LlmJudgeRequest(
      system_prompt='system', user_prompt='doc', response_model=ContextualPrecisionItemJudgePayload
    )

    def verdict(relevant: bool) -> LlmJudgeResponse:
      judgment = ContextualPrecisionItemJudgment(retrieved_id='d1', relevant=relevant, reasoning='r')
      return LlmJudgeResponse(
        provider='fake', model='judge', output=ContextualPrecisionItemJudgePayload(judgments=[judgment])
      )

    with tempfile.TemporaryDirectory() as folder:
      path = Path(folder) / 'verdicts.sqlite'
      first = StoredPrecisionJudgeClient(_ScriptedJudge(verdict(True)), path, 'repetition-1')

      class _SlowerJudge:
        async def judge(self, request: LlmJudgeRequest) -> LlmJudgeResponse:
          await first.judge(request)
          return verdict(False)

        async def aclose(self) -> None:
          await first.aclose()

      second = StoredPrecisionJudgeClient(_SlowerJudge(), path, 'repetition-1')
      response = await second.judge(precision)
      await second.aclose()

    self.assertEqual(response, verdict(True))


class ReusedCallsAndPromptCacheTest(unittest.IsolatedAsyncioTestCase):
  async def test_a_failed_unit_cancels_reused_calls_still_using_the_cache_before_deleting_it(self) -> None:
    events: list[str] = []
    claims = (Claim(id='q1-f01', text='Dana approved the budget.'), Claim(id='q1-f02', text='It was approved in May.'))

    async def reply(messages: list[tuple[str, str]]) -> dict[str, Any]:
      if messages[-1][1] == claims[0].text:
        raise ValueError('bad request')
      try:
        await asyncio.Event().wait()
      except asyncio.CancelledError:
        events.append('cancelled')
        raise
      return {'raw': AIMessage(content=''), 'parsed': None, 'parsing_error': None}

    chat_model = MagicMock()
    chat_model.model_copy.return_value.with_structured_output.return_value.ainvoke = AsyncMock(side_effect=reply)
    chat_model.async_client.caches.create = AsyncMock(
      return_value=SimpleNamespace(name='cachedContents/c1', usage_metadata=SimpleNamespace(total_token_count=5000))
    )
    chat_model.async_client.caches.delete = AsyncMock(side_effect=lambda **_: events.append('deleted'))
    client = GeminiLlmJudgeClient(GeminiJudgeSettings(api_key='key', model='judge-1'), 4, chat_model=chat_model)
    metric = GoldClaimsContextualRecall(judge_client=ReusingJudgeClient(client), target_span_types=(SEARCH_SPAN_TYPE,))
    sample_cache.set({})

    result = await metric.compute(search_span('s', ['d1', 'd2'], stage='retrieved'), truths(claims_truth(*claims)))

    self.assertEqual(result.status, MetricComputationStatus.FAILED)
    self.assertEqual(events, ['cancelled', 'deleted'])

  async def test_a_unit_cancelled_while_the_cache_is_created_leaves_no_reused_call_running(self) -> None:
    events: list[str] = []
    claims = (Claim(id='q1-f01', text='Dana approved the budget.'), Claim(id='q1-f02', text='It was approved in May.'))
    creating = asyncio.Event()

    async def create(**_: Any) -> Any:
      creating.set()
      try:
        await asyncio.Event().wait()
      except asyncio.CancelledError:
        events.append('creation cancelled')
        raise

    chat_model = MagicMock()
    ainvoke = chat_model.model_copy.return_value.with_structured_output.return_value.ainvoke = AsyncMock()
    chat_model.async_client.caches.create = AsyncMock(side_effect=create)
    client = GeminiLlmJudgeClient(GeminiJudgeSettings(api_key='key', model='judge-1'), 4, chat_model=chat_model)
    metric = GoldClaimsContextualRecall(judge_client=ReusingJudgeClient(client), target_span_types=(SEARCH_SPAN_TYPE,))
    calls: dict[Any, Any] = {}
    sample_cache.set(calls)

    unit = asyncio.create_task(
      metric.compute(search_span('s', ['d1', 'd2'], stage='retrieved'), truths(claims_truth(*claims)))
    )
    await creating.wait()
    unit.cancel()
    with self.assertRaises(asyncio.CancelledError):
      await unit

    self.assertEqual(events, ['creation cancelled'])
    self.assertTrue(all(task.done() for task in calls.values()))
    ainvoke.assert_not_awaited()


class OpenJudgeClientTest(unittest.IsolatedAsyncioTestCase):
  async def test_builds_the_chat_model_from_the_experiment_config_and_closes_it(self) -> None:
    chat_model = MagicMock()
    chat_model.async_client.aclose = AsyncMock()
    with patch('judge.ChatGoogleGenerativeAI', return_value=chat_model) as chat_model_class:
      async with open_judge_client(_judge_config(), GeminiJudgeSettings(api_key='test-key')) as client:
        self.assertIsInstance(client, ReusingJudgeClient)
        chat_model.async_client.aclose.assert_not_awaited()

    kwargs = chat_model_class.call_args.kwargs
    self.assertEqual(kwargs['model'], 'judge-1')
    self.assertEqual(kwargs['api_key'], 'test-key')
    self.assertEqual(kwargs['timeout'], 300)
    self.assertEqual(kwargs['thinking_level'], 'low')
    chat_model.async_client.aclose.assert_awaited_once()


if __name__ == '__main__':
  unittest.main()
