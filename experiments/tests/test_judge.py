import asyncio
import unittest
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

from pydantic import BaseModel

from syllo_eval.evaluation.judge import LlmJudgeRequest, LlmJudgeResponse
from syllo_eval.evaluation.plan_executor import sample_cache
from syllo_eval.settings import GeminiJudgeSettings
from config import JudgeConfig
from judge import ConfiguredTemperatureJudgeClient, ReusingJudgeClient, open_judge_client


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
