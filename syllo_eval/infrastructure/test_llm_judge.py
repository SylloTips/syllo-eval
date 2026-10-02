import asyncio
import unittest
from datetime import datetime, timezone
from typing import Any
from unittest.mock import AsyncMock, patch

from google.genai.errors import ClientError
from langchain_core.messages import AIMessage
from pydantic import BaseModel

from syllo_eval.evaluation.judge import LlmJudgeRequest, LlmJudgeResponse
from syllo_eval.evaluation.judge.metric_base import BaseLlmJudgeMetric, JudgeScorePayload
from syllo_eval.infrastructure.llm_judge import build_llm_judge_client
from syllo_eval.infrastructure.exceptions import DataMappingError
from syllo_eval.infrastructure.llm_judge.gemini import GeminiLlmJudgeClient
from syllo_eval.infrastructure.llm_judge.openai import OpenAILlmJudgeClient
from syllo_eval.model import GroundTruth, Span
from syllo_eval.settings import GeminiJudgeSettings, LlmJudgeSettings, OpenAIJudgeSettings


class _FakeStructuredRunnable:
  def __init__(self, result: dict[str, Any] | list[Any], recorded_inputs: list[Any]):
    self._result = result
    self._recorded_inputs = recorded_inputs

  async def ainvoke(self, input: Any) -> dict[str, Any]:
    self._recorded_inputs.append(input)
    if isinstance(self._result, list):
      result = self._result.pop(0)
    else:
      result = self._result
    if isinstance(result, Exception):
      raise result
    return result


class _FakeBoundChatModel:
  def __init__(self, parent: '_FakeChatModel', bound_kwargs: dict[str, Any]):
    self._parent = parent
    self._bound_kwargs = bound_kwargs

  def bind(self, **kwargs: Any) -> '_FakeBoundChatModel':
    merged_kwargs = dict(self._bound_kwargs)
    merged_kwargs.update(kwargs)
    return _FakeBoundChatModel(self._parent, merged_kwargs)

  def with_structured_output(
    self,
    schema: dict[str, Any] | type[BaseModel] | None = None,
    *,
    method: str = 'function_calling',
    include_raw: bool = False,
    **kwargs: Any,
  ) -> Any:
    self._parent.calls.append(
      {
        'schema': schema,
        'method': method,
        'include_raw': include_raw,
        'kwargs': kwargs,
        'bound_kwargs': dict(self._bound_kwargs),
      }
    )
    return _FakeStructuredRunnable(self._parent._result, self._parent.inputs)


class _FakeChatModel:
  def __init__(self, result: dict[str, Any] | list[Any]):
    self._result = result
    self.calls: list[dict[str, Any]] = []
    self.inputs: list[Any] = []
    self.bind_calls: list[dict[str, Any]] = []

  def bind(self, **kwargs: Any) -> _FakeBoundChatModel:
    self.bind_calls.append(dict(kwargs))
    return _FakeBoundChatModel(self, kwargs)

  def with_structured_output(
    self,
    schema: dict[str, Any] | type[BaseModel] | None = None,
    *,
    method: str = 'function_calling',
    include_raw: bool = False,
    **kwargs: Any,
  ) -> Any:
    return _FakeBoundChatModel(self, {}).with_structured_output(
      schema=schema,
      method=method,
      include_raw=include_raw,
      **kwargs,
    )


class _FakeJudgeClient:
  def __init__(self, *, error: Exception):
    self._error = error

  async def judge(self, request: LlmJudgeRequest) -> LlmJudgeResponse:
    del request
    raise self._error

  async def aclose(self) -> None:
    return None


class _TestJudgeMetric(BaseLlmJudgeMetric):
  @property
  def name(self) -> str:
    return 'test_judge_metric'

  @property
  def description(self) -> str:
    return 'Metric used to test judge error handling.'

  @property
  def target_span_types(self) -> tuple[str, ...]:
    return ('agent',)

  def build_system_prompt(self) -> str:
    return 'system'

  def build_user_prompt(self, span: Span, ground_truth: GroundTruth | None) -> str:
    del span, ground_truth
    return 'user'


def _make_span() -> Span:
  now = datetime.now(timezone.utc)
  return Span(
    external_id='span_test',
    trace_id='trace_test',
    parent_span_id=None,
    span_type='agent',
    name='agent_span',
    start_time=now,
    end_time=now,
    input_data='{}',
    output_data='{}',
    metadata=None,
  )


class TestOpenAILlmJudgeClient(unittest.IsolatedAsyncioTestCase):
  def test_openai_client_passes_use_responses_api_flag_to_chat_openai(self) -> None:
    with patch('syllo_eval.infrastructure.llm_judge.openai.ChatOpenAI') as chat_model_cls:
      OpenAILlmJudgeClient(
        config=OpenAIJudgeSettings(api_key='test-key', model='deepseek-v4-flash', use_responses_api=False),
        max_concurrent_requests=2,
      )

    self.assertEqual(chat_model_cls.call_args.kwargs['use_responses_api'], False)

  def test_openai_client_defaults_use_responses_api_to_true(self) -> None:
    with patch('syllo_eval.infrastructure.llm_judge.openai.ChatOpenAI') as chat_model_cls:
      OpenAILlmJudgeClient(
        config=OpenAIJudgeSettings(api_key='test-key', model='gpt-5-mini'),
        max_concurrent_requests=2,
      )

    self.assertEqual(chat_model_cls.call_args.kwargs['use_responses_api'], True)

  async def test_judge_uses_pydantic_structured_output_and_parses_raw_message(self) -> None:
    fake_chat_model = _FakeChatModel(
      {
        'raw': AIMessage(
          content='',
          id='resp_123',
          response_metadata={'model_name': 'gpt-5-mini'},
          usage_metadata={'input_tokens': 10, 'output_tokens': 4, 'total_tokens': 14},
        ),
        'parsed': JudgeScorePayload(
          score=0.8,
          reasoning='Mostly correct.',
          metadata={'label': 'pass'},
        ),
        'parsing_error': None,
      }
    )

    client = OpenAILlmJudgeClient(
      config=OpenAIJudgeSettings(api_key='test-openai-key', model='gpt-5-mini'),
      max_concurrent_requests=2,
      chat_model=fake_chat_model,
    )
    response = await client.judge(
      LlmJudgeRequest(
        system_prompt='system',
        user_prompt='user',
        response_model=JudgeScorePayload,
        max_output_tokens=400,
      )
    )

    self.assertEqual(response.provider, 'openai')
    self.assertEqual(response.model, 'gpt-5-mini')
    self.assertEqual(response.response_id, 'resp_123')
    self.assertIsInstance(response.output, JudgeScorePayload)
    self.assertEqual(response.output.score, 0.8)
    self.assertEqual(response.usage, {'input_tokens': 10, 'output_tokens': 4, 'total_tokens': 14})
    self.assertEqual(response.attempts, 1)
    self.assertIsNotNone(response.latency_seconds)

    self.assertEqual(len(fake_chat_model.calls), 1)
    self.assertEqual(fake_chat_model.bind_calls, [{'temperature': 0.0, 'max_tokens': 400}])
    call = fake_chat_model.calls[0]
    self.assertIs(call['schema'], JudgeScorePayload)
    self.assertEqual(call['method'], 'json_mode')
    self.assertTrue(call['include_raw'])
    self.assertEqual(call['bound_kwargs'], {'temperature': 0.0, 'max_tokens': 400})
    self.assertEqual(call['kwargs'], {})

    self.assertEqual(len(fake_chat_model.inputs), 1)
    messages = fake_chat_model.inputs[0]
    self.assertEqual(messages[1], ('human', 'user'))
    system_role, system_content = messages[0]
    self.assertEqual(system_role, 'system')
    self.assertTrue(system_content.startswith('system'))
    self.assertIn('JSON', system_content)
    self.assertIn('"score"', system_content)

  async def test_invalid_outputs_fail_at_client_boundary(self) -> None:
    cases: list[dict[str, Any]] = [
      {'raw': AIMessage(content=''), 'parsed': {'score': 'invalid', 'reasoning': 'bad'}, 'parsing_error': None},
      {'raw': AIMessage(content=''), 'parsed': None, 'parsing_error': ValueError('invalid JSON')},
      {'raw': AIMessage(content='', additional_kwargs={'refusal': 'declined'}), 'parsed': None, 'parsing_error': None},
    ]
    for result in cases:
      with self.subTest(result=result):
        client = OpenAILlmJudgeClient(
          config=OpenAIJudgeSettings(), max_concurrent_requests=1, chat_model=_FakeChatModel(result)
        )
        with self.assertRaises(DataMappingError):
          await client.judge(
            LlmJudgeRequest(system_prompt='system', user_prompt='user', response_model=JudgeScorePayload)
          )

  async def test_client_limits_concurrent_calls(self) -> None:
    entered = asyncio.Event()
    release = asyncio.Event()

    async def invoke(request):
      entered.set()
      await release.wait()
      return {'raw': AIMessage(content=''), 'parsed': {'score': 1, 'reasoning': 'ok'}, 'parsing_error': None}

    client = OpenAILlmJudgeClient(
      config=OpenAIJudgeSettings(), max_concurrent_requests=1, chat_model=_FakeChatModel({})
    )
    request = LlmJudgeRequest(system_prompt='system', user_prompt='user', response_model=JudgeScorePayload)
    with patch.object(client, '_invoke', side_effect=invoke) as mocked_invoke:
      first = asyncio.create_task(client.judge(request))
      second = asyncio.create_task(client.judge(request))
      try:
        await asyncio.wait_for(entered.wait(), timeout=1)
        self.assertEqual(mocked_invoke.call_count, 1)
      finally:
        release.set()
        await asyncio.gather(first, second)
      self.assertEqual(mocked_invoke.call_count, 2)

  async def test_only_owned_provider_clients_are_closed(self) -> None:
    for client_type, config, model_path, close_path in [
      (OpenAILlmJudgeClient, OpenAIJudgeSettings(), 'openai.ChatOpenAI', 'root_async_client.close'),
      (GeminiLlmJudgeClient, GeminiJudgeSettings(), 'gemini.ChatGoogleGenerativeAI', 'async_client.aclose'),
    ]:
      with self.subTest(provider=client_type.__name__):
        with patch(f'syllo_eval.infrastructure.llm_judge.{model_path}') as model_class:
          close = AsyncMock()
          parent, attr = close_path.split('.')
          setattr(getattr(model_class.return_value, parent), attr, close)
          owned = client_type(config=config, max_concurrent_requests=1)
          await owned.aclose()
          close.assert_awaited_once()
          close.reset_mock()
          borrowed = client_type(config=config, max_concurrent_requests=1, chat_model=model_class.return_value)
          await borrowed.aclose()
          close.assert_not_awaited()


class TestGeminiLlmJudgeClient(unittest.IsolatedAsyncioTestCase):
  def test_gemini_disables_sdk_retries(self) -> None:
    with patch('syllo_eval.infrastructure.llm_judge.gemini.ChatGoogleGenerativeAI') as chat_model_cls:
      GeminiLlmJudgeClient(
        config=GeminiJudgeSettings(api_key='test-gemini-key', model='gemini-2.5-flash', max_retries=4),
        max_concurrent_requests=2,
      )

    self.assertEqual(chat_model_cls.call_args.kwargs['max_retries'], 1)

  async def test_judge_uses_json_schema_for_gemini_and_preserves_raw_message_id(self) -> None:
    fake_chat_model = _FakeChatModel(
      {
        'raw': AIMessage(
          content='',
          id='lc_run--123',
          response_metadata={'model_name': 'gemini-2.5-flash'},
          usage_metadata={
            'input_tokens': 9,
            'output_tokens': 6,
            'total_tokens': 15,
            'input_token_details': {'cache_read': 4},
            'output_token_details': {'reasoning': 2},
          },
        ),
        'parsed': {'score': 0.6, 'reasoning': 'Partially correct.', 'metadata': {'label': 'partial'}},
        'parsing_error': None,
      }
    )

    client = GeminiLlmJudgeClient(
      config=GeminiJudgeSettings(api_key='test-gemini-key', model='gemini-2.5-flash'),
      max_concurrent_requests=2,
      chat_model=fake_chat_model,
    )
    response = await client.judge(
      LlmJudgeRequest(
        system_prompt='system',
        user_prompt='user',
        response_model=JudgeScorePayload,
        max_output_tokens=400,
      )
    )

    self.assertEqual(response.provider, 'gemini')
    self.assertEqual(response.model, 'gemini-2.5-flash')
    self.assertEqual(response.response_id, 'lc_run--123')
    self.assertIsInstance(response.output, JudgeScorePayload)
    self.assertEqual(response.output.score, 0.6)
    self.assertEqual(
      response.usage,
      {'input_tokens': 9, 'output_tokens': 6, 'total_tokens': 15, 'cached_input_tokens': 4, 'reasoning_tokens': 2},
    )
    self.assertEqual(response.attempts, 1)
    self.assertIsNotNone(response.latency_seconds)

    self.assertEqual(len(fake_chat_model.calls), 1)
    self.assertEqual(fake_chat_model.bind_calls, [{'temperature': 0.0, 'max_output_tokens': 400}])
    call = fake_chat_model.calls[0]
    self.assertEqual(call['schema'], JudgeScorePayload.model_json_schema())
    self.assertEqual(call['method'], 'json_schema')
    self.assertTrue(call['include_raw'])
    self.assertEqual(call['bound_kwargs'], {'temperature': 0.0, 'max_output_tokens': 400})
    self.assertEqual(call['kwargs'], {})
    self.assertEqual(fake_chat_model.inputs, [[('system', 'system'), ('human', 'user')]])

  async def test_gemini_retries_429_after_api_retry_delay(self) -> None:
    fake_chat_model = _FakeChatModel(
      [
        ClientError(
          429,
          {
            'error': {
              'code': 429,
              'message': 'Quota exceeded. Please retry in 57.970153168s.',
              'status': 'RESOURCE_EXHAUSTED',
              'details': [
                {'@type': 'type.googleapis.com/google.rpc.RetryInfo', 'retryDelay': '57s'},
              ],
            }
          },
          None,
        ),
        {
          'raw': AIMessage(content='', id='lc_run--retry', response_metadata={'model_name': 'gemini-2.5-flash'}),
          'parsed': {'score': 0.9, 'reasoning': 'Correct after retry.', 'metadata': {}},
          'parsing_error': None,
        },
      ]
    )
    client = GeminiLlmJudgeClient(
      config=GeminiJudgeSettings(api_key='test-gemini-key', model='gemini-2.5-flash', max_retries=2),
      max_concurrent_requests=2,
      chat_model=fake_chat_model,
    )

    with (
      patch('syllo_eval.infrastructure.llm_judge.gemini.asyncio.sleep', new_callable=AsyncMock) as sleep_mock,
      patch('syllo_eval.infrastructure.llm_judge.gemini.random.uniform', return_value=1.0),
    ):
      response = await client.judge(
        LlmJudgeRequest(
          system_prompt='system',
          user_prompt='user',
          response_model=JudgeScorePayload,
        )
      )

    self.assertEqual(response.response_id, 'lc_run--retry')
    self.assertEqual(response.attempts, 2)
    sleep_mock.assert_awaited_once_with(57.970153168)
    self.assertEqual(len(fake_chat_model.inputs), 2)

  async def test_gemini_uses_message_hint_when_structured_retry_delay_is_zero(self) -> None:
    fake_chat_model = _FakeChatModel(
      [
        ClientError(
          429,
          {
            'error': {
              'code': 429,
              'message': 'Quota exceeded. Please retry in 762.168097ms.',
              'status': 'RESOURCE_EXHAUSTED',
              'details': [
                {'@type': 'type.googleapis.com/google.rpc.RetryInfo', 'retryDelay': '0s'},
              ],
            }
          },
          None,
        ),
        {
          'raw': AIMessage(content='', id='lc_run--retry-ms', response_metadata={'model_name': 'gemini-2.5-flash'}),
          'parsed': {'score': 0.5, 'reasoning': 'Recovered.', 'metadata': {}},
          'parsing_error': None,
        },
      ]
    )
    client = GeminiLlmJudgeClient(
      config=GeminiJudgeSettings(api_key='test-gemini-key', model='gemini-2.5-flash', max_retries=2),
      max_concurrent_requests=2,
      chat_model=fake_chat_model,
    )

    with (
      patch('syllo_eval.infrastructure.llm_judge.gemini.asyncio.sleep', new_callable=AsyncMock) as sleep_mock,
      patch('syllo_eval.infrastructure.llm_judge.gemini.random.uniform', return_value=1.5),
    ):
      response = await client.judge(
        LlmJudgeRequest(
          system_prompt='system',
          user_prompt='user',
          response_model=JudgeScorePayload,
        )
      )

    self.assertEqual(response.response_id, 'lc_run--retry-ms')
    sleep_mock.assert_awaited_once()
    assert sleep_mock.await_args is not None
    (delay,), _ = sleep_mock.await_args
    self.assertAlmostEqual(delay, 0.762168097 * 1.5)

  async def test_gemini_retries_503_when_message_carries_retry_hint(self) -> None:
    fake_chat_model = _FakeChatModel(
      [
        ClientError(
          503,
          {
            'error': {
              'code': 503,
              'message': 'Service overloaded. Please retry in 2s.',
              'status': 'UNAVAILABLE',
            }
          },
          None,
        ),
        {
          'raw': AIMessage(content='', id='lc_run--503-retry', response_metadata={'model_name': 'gemini-2.5-flash'}),
          'parsed': {'score': 0.5, 'reasoning': 'OK.', 'metadata': {}},
          'parsing_error': None,
        },
      ]
    )
    client = GeminiLlmJudgeClient(
      config=GeminiJudgeSettings(api_key='test-gemini-key', model='gemini-2.5-flash', max_retries=2),
      max_concurrent_requests=2,
      chat_model=fake_chat_model,
    )

    with (
      patch('syllo_eval.infrastructure.llm_judge.gemini.asyncio.sleep', new_callable=AsyncMock) as sleep_mock,
      patch('syllo_eval.infrastructure.llm_judge.gemini.random.uniform', return_value=1.0),
    ):
      response = await client.judge(
        LlmJudgeRequest(
          system_prompt='system',
          user_prompt='user',
          response_model=JudgeScorePayload,
        )
      )

    self.assertEqual(response.response_id, 'lc_run--503-retry')
    sleep_mock.assert_awaited_once_with(2.0)

  async def test_gemini_retries_503_without_retry_hint_after_default_delay(self) -> None:
    fake_chat_model = _FakeChatModel(
      [
        ClientError(
          503,
          {
            'error': {
              'code': 503,
              'message': 'This model is currently experiencing high demand.',
              'status': 'UNAVAILABLE',
            }
          },
          None,
        ),
        {
          'raw': AIMessage(
            content='', id='lc_run--503-default-retry', response_metadata={'model_name': 'gemini-2.5-flash'}
          ),
          'parsed': {'score': 0.5, 'reasoning': 'Recovered.', 'metadata': {}},
          'parsing_error': None,
        },
      ]
    )
    client = GeminiLlmJudgeClient(
      config=GeminiJudgeSettings(api_key='test-gemini-key', model='gemini-2.5-flash', max_retries=3),
      max_concurrent_requests=2,
      chat_model=fake_chat_model,
    )

    with patch('syllo_eval.infrastructure.llm_judge.gemini.asyncio.sleep', new_callable=AsyncMock) as sleep_mock:
      response = await client.judge(
        LlmJudgeRequest(
          system_prompt='system',
          user_prompt='user',
          response_model=JudgeScorePayload,
        )
      )

    self.assertEqual(response.response_id, 'lc_run--503-default-retry')
    sleep_mock.assert_awaited_once_with(60.0)


class TestLlmJudgeFactory(unittest.TestCase):
  def test_factory_returns_none_when_provider_is_disabled(self) -> None:
    config = LlmJudgeSettings(
      provider=None,
      max_concurrent_requests=5,
      openai=OpenAIJudgeSettings(),
      gemini=GeminiJudgeSettings(),
    )

    client = build_llm_judge_client(config)

    self.assertIsNone(client)

  def test_factory_builds_openai_client(self) -> None:
    config = LlmJudgeSettings(
      provider='openai',
      max_concurrent_requests=3,
      openai=OpenAIJudgeSettings(api_key='key'),
      gemini=GeminiJudgeSettings(),
    )

    client = build_llm_judge_client(config)

    self.assertIsInstance(client, OpenAILlmJudgeClient)

  def test_factory_builds_gemini_client(self) -> None:
    config = LlmJudgeSettings(
      provider='gemini',
      max_concurrent_requests=3,
      openai=OpenAIJudgeSettings(),
      gemini=GeminiJudgeSettings(api_key='key'),
    )

    client = build_llm_judge_client(config)

    self.assertIsInstance(client, GeminiLlmJudgeClient)


class TestBaseLlmJudgeMetric(unittest.IsolatedAsyncioTestCase):
  async def test_compute_propagates_judge_call_errors(self) -> None:
    metric = _TestJudgeMetric(judge_client=_FakeJudgeClient(error=RuntimeError('judge service unavailable')))

    with self.assertRaises(RuntimeError) as ctx:
      await metric.compute(_make_span(), None)

    self.assertIn('judge service unavailable', str(ctx.exception))

  async def test_compute_propagates_payload_validation_errors(self) -> None:
    metric = _TestJudgeMetric(
      judge_client=OpenAILlmJudgeClient(
        config=OpenAIJudgeSettings(),
        max_concurrent_requests=1,
        chat_model=_FakeChatModel(
          {
            'raw': AIMessage(content=''),
            'parsed': {'score': 'invalid', 'reasoning': 'bad payload'},
            'parsing_error': None,
          }
        ),
      )
    )

    with self.assertRaises(DataMappingError):
      await metric.compute(_make_span(), None)
