import unittest
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

from pydantic import BaseModel

from syllo_eval.evaluation.judge import LlmJudgeRequest, LlmJudgeResponse
from syllo_eval.settings import GeminiJudgeSettings
from config import JudgeConfig
from judge import ConfiguredTemperatureJudgeClient, open_judge_client


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


class OpenJudgeClientTest(unittest.IsolatedAsyncioTestCase):
  async def test_builds_the_chat_model_from_the_experiment_config_and_closes_it(self) -> None:
    chat_model = MagicMock()
    chat_model.async_client.aclose = AsyncMock()
    with patch('judge.ChatGoogleGenerativeAI', return_value=chat_model) as chat_model_class:
      async with open_judge_client(_judge_config(), GeminiJudgeSettings(api_key='test-key')) as client:
        self.assertIsInstance(client, ConfiguredTemperatureJudgeClient)
        chat_model.async_client.aclose.assert_not_awaited()

    kwargs = chat_model_class.call_args.kwargs
    self.assertEqual(kwargs['model'], 'judge-1')
    self.assertEqual(kwargs['api_key'], 'test-key')
    self.assertEqual(kwargs['timeout'], 300)
    self.assertEqual(kwargs['thinking_level'], 'low')
    chat_model.async_client.aclose.assert_awaited_once()


if __name__ == '__main__':
  unittest.main()
