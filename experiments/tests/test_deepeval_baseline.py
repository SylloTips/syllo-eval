import asyncio
import os
import unittest
from collections.abc import Callable
from typing import Any, cast
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
from google.genai.errors import ClientError
from langchain_core.messages import AIMessage
from langchain_google_genai import ChatGoogleGenerativeAI

from deepeval_baseline import (
  CORRECTNESS_STEPS,
  DEEPEVAL_VERSION,
  METRIC_KEYS,
  DeepEvalAnswerCorrectness,
  DeepEvalContextualPrecision,
  DeepEvalGoldClaimsContextualRecall,
  DeepEvalJudge,
  deepeval_metrics,
  node_text,
  open_deepeval_judge,
  recorded_judge_calls,
)

# DeepEval modules come after the baseline module, which sets the switches DeepEval reads on import.
from deepeval.constants import HIDDEN_DIR
from deepeval.metrics.contextual_precision.schema import Verdicts as PrecisionVerdicts
from deepeval.metrics.contextual_recall.schema import Verdicts as RecallVerdicts
from deepeval.metrics.g_eval.schema import ReasonScore
from deepeval.telemetry import telemetry_opt_out

from syllo_eval.evaluation.metric_support.retrieved_context import RetrievedItem
from syllo_eval.evaluation.metrics.implementations.rag.contextual_precision_judge import (
  ContextualPrecisionDocumentJudgeMetric,
)
from syllo_eval.infrastructure.exceptions import ExternalServiceError, JudgeOutputError
from syllo_eval.model import GroundTruth, MetricComputationStatus, Span
from syllo_eval.settings import GeminiJudgeSettings
from syllo_eval.trace_semantics import Answer, RetrievalItem, RetrievalResult

from ablation_metrics import SEARCH_SPAN_TYPE, GoldClaimsContextualRecall
from benchmarks.common import Claim
from benchmarks.erb import CLAIMS_KEY
from config import EXPERIMENTS_DIR, load_config
from metric_fakes import (
  EXPECTED_ANSWER,
  QUESTION,
  ScriptedJudge,
  claims_truth,
  ground_truths,
  precision_judgments,
  recall_judgments,
  search_span,
  span,
  truth,
  without_judge_keys,
)

CLAIMS = (Claim('c1', 'Dana approved the budget.'), Claim('c2', 'The approval was in May.'))
OUTPUT_LIMIT = 1_000


class _Unparsed:
  """A reply whose structured output did not parse."""

  def __init__(self, error: Exception):
    self.error = error


class FakeChatModel:
  """Stands in for the judge's chat model: records each structured-output call and answers from a script.

  A reply is the parsed output, ``None`` for no output, an ``_Unparsed`` output, or an exception the call raises; a
  single function of the prompt answers every call. ``input_tokens`` is a count, or a function of the prompt. Copies
  share the script, the calls and ``load``, the peak number of calls in flight, as copies of a real chat model share
  its client.
  """

  model = 'judge-1'

  def __init__(self, *replies: Any, output_tokens: int = 7, input_tokens: int | Callable[[str], int] = 100):
    self.replies = list(replies)
    self.output_tokens = output_tokens
    self.input_tokens = input_tokens
    self.options: dict[str, Any] = {}
    self.calls: list[dict[str, Any]] = []
    self.load = {'in_flight': 0, 'peak': 0}
    self.async_client = MagicMock(aclose=AsyncMock())

  def model_copy(self, *, update: dict[str, Any]) -> 'FakeChatModel':
    copy = FakeChatModel.__new__(FakeChatModel)
    copy.__dict__.update(self.__dict__)
    copy.options = {**self.options, **update}
    return copy

  def with_structured_output(self, *, schema: Any, method: str, include_raw: bool) -> '_Runnable':
    return _Runnable(self, {'schema': schema, 'method': method, 'include_raw': include_raw, 'options': self.options})


class _Runnable:
  def __init__(self, chat: FakeChatModel, call: dict[str, Any]):
    self._chat = chat
    self._call = call

  async def ainvoke(self, messages: list[tuple[str, str]]) -> dict[str, Any]:
    chat = self._chat
    [(_, prompt)] = messages
    call_id = f'response-{len(chat.calls) + 1}'
    chat.calls.append({**self._call, 'messages': messages, 'id': call_id})
    chat.load['in_flight'] += 1
    chat.load['peak'] = max(chat.load['peak'], chat.load['in_flight'])
    try:
      await _pause()
    finally:
      chat.load['in_flight'] -= 1
    reply = chat.replies[0](prompt) if len(chat.replies) == 1 and callable(chat.replies[0]) else chat.replies.pop(0)
    if isinstance(reply, Exception):
      raise reply
    input_tokens = chat.input_tokens(prompt) if callable(chat.input_tokens) else chat.input_tokens
    raw = AIMessage(
      content='',
      id=call_id,
      response_metadata={'model_name': 'judge-1-001'},
      usage_metadata={
        'input_tokens': input_tokens,
        'output_tokens': chat.output_tokens,
        'total_tokens': input_tokens + chat.output_tokens,
      },
    )
    if isinstance(reply, _Unparsed):
      return {'raw': raw, 'parsed': None, 'parsing_error': reply.error}
    return {'raw': raw, 'parsed': reply, 'parsing_error': None}


async def _pause() -> None:
  """Let other tasks run, as a request in flight does, without asyncio.sleep, which some tests patch."""
  future = asyncio.get_running_loop().create_future()
  future.get_loop().call_soon(future.set_result, None)
  await future


def deepeval_judge(chat: FakeChatModel, *, max_attempts: int = 3) -> DeepEvalJudge:
  return DeepEvalJudge(
    cast(ChatGoogleGenerativeAI, chat), temperature=0.0, max_concurrent_requests=2, max_attempts=max_attempts
  )


def verdicts(*decisions: tuple[bool, str]) -> dict[str, Any]:
  return {'verdicts': [{'verdict': 'yes' if yes else 'no', 'reason': reason} for yes, reason in decisions]}


def rate_limit() -> ClientError:
  return ClientError(
    429,
    {
      'error': {
        'code': 429,
        'message': 'Quota exceeded. Please retry in 2s.',
        'status': 'RESOURCE_EXHAUSTED',
        'details': [{'@type': 'type.googleapis.com/google.rpc.RetryInfo', 'retryDelay': '2s'}],
      }
    },
    None,
  )


def prompt_of(call: dict[str, Any]) -> str:
  [(role, prompt)] = call['messages']
  assert role == 'human'
  return prompt


def answer_span(text: str | None = 'Dana approved it, in May.') -> Span:
  answer = Answer(text=text) if text is not None else None
  return span('root', 'agent_root', parent=None, request=QUESTION, answer=answer)


class DeepEvalJudgeTest(unittest.IsolatedAsyncioTestCase):
  async def test_sends_the_prompt_alone_with_the_output_schema_and_records_the_call(self) -> None:
    chat = FakeChatModel(verdicts((True, 'quoted')))
    judge = deepeval_judge(chat)

    with recorded_judge_calls() as calls:
      output = await judge.a_generate_with_schema('Judge these nodes.', schema=PrecisionVerdicts)

    self.assertIsInstance(output, PrecisionVerdicts)
    [call] = chat.calls
    self.assertEqual(call['messages'], [('human', 'Judge these nodes.')])
    self.assertEqual(call['schema'], PrecisionVerdicts.model_json_schema())
    self.assertEqual((call['method'], call['include_raw']), ('json_schema', True))
    # The configured temperature, and no output limit: DeepEval's own Gemini model sets none.
    self.assertEqual(call['options'], {'temperature': 0.0})
    [response] = calls
    self.assertEqual((response.provider, response.model, response.response_id), ('gemini', 'judge-1-001', 'response-1'))
    self.assertEqual(response.usage, {'input_tokens': 100, 'output_tokens': 7, 'total_tokens': 107})
    self.assertEqual(response.attempts, 1)
    self.assertIsNotNone(response.latency_seconds)
    self.assertEqual(judge.get_model_name(), 'judge-1')

  async def test_output_that_does_not_validate_raises_with_its_usage(self) -> None:
    replies = [{'verdicts': [{'verdict': 'maybe', 'reason': 'r'}]}, None, _Unparsed(ValueError('bad JSON'))]
    for reply in replies:
      with self.subTest(reply=reply):
        judge = deepeval_judge(FakeChatModel(reply))
        with recorded_judge_calls() as calls, self.assertRaises(JudgeOutputError) as raised:
          await judge.a_generate('Judge.', schema=PrecisionVerdicts)
        self.assertEqual(raised.exception.usage, {'input_tokens': 100, 'output_tokens': 7, 'total_tokens': 107})
        self.assertEqual(calls, [])

  async def test_retries_rate_limits_after_the_server_hint_and_counts_the_attempts(self) -> None:
    chat = FakeChatModel(rate_limit(), verdicts((False, 'r')))
    with (
      patch('deepeval_baseline.asyncio.sleep', new_callable=AsyncMock) as sleep,
      patch('syllo_eval.infrastructure.llm_judge.gemini.random.uniform', return_value=1.0),
      recorded_judge_calls() as calls,
    ):
      await deepeval_judge(chat).a_generate('Judge.', schema=PrecisionVerdicts)

    sleep.assert_awaited_once_with(2.0)
    self.assertEqual([response.attempts for response in calls], [2])

  async def test_gives_up_after_the_configured_attempts_and_never_retries_other_errors(self) -> None:
    cases = [(FakeChatModel(rate_limit(), rate_limit()), 2), (FakeChatModel(RuntimeError('500 internal')), 1)]
    for chat, attempts in cases:
      with self.subTest(attempts=attempts):
        with (
          patch('deepeval_baseline.asyncio.sleep', new_callable=AsyncMock),
          recorded_judge_calls(),
          self.assertRaises(ExternalServiceError),
        ):
          await deepeval_judge(chat, max_attempts=2).a_generate('Judge.', schema=PrecisionVerdicts)
        self.assertEqual(len(chat.calls), attempts)

  async def test_a_call_outside_a_recording_context_fails_before_reaching_the_model(self) -> None:
    chat = FakeChatModel(verdicts((True, 'r')))

    with self.assertRaises(LookupError):
      await deepeval_judge(chat).a_generate('Judge.', schema=PrecisionVerdicts)

    self.assertEqual(chat.calls, [])


class DeepEvalContextualPrecisionTest(unittest.IsolatedAsyncioTestCase):
  def metric(self, chat: FakeChatModel) -> DeepEvalContextualPrecision:
    return DeepEvalContextualPrecision(
      judge=deepeval_judge(chat), output_token_limit=OUTPUT_LIMIT, target_span_types=(SEARCH_SPAN_TYPE,)
    )

  async def test_judges_a_ranking_in_one_call_and_scores_it_as_the_compared_metric(self) -> None:
    chat = FakeChatModel(verdicts((True, 'd1 reasoning'), (False, 'd2 reasoning'), (True, 'd3 reasoning')))
    target = search_span('s1', ['d1', 'd2', 'd3'])

    result = await self.metric(chat).compute(target, ground_truths(truth()))

    built_in = ContextualPrecisionDocumentJudgeMetric(
      judge_client=ScriptedJudge(
        precision_judgments((1, 'd1', True)),
        precision_judgments((2, 'd2', False)),
        precision_judgments((3, 'd3', True)),
      ),
      target_span_types=(SEARCH_SPAN_TYPE,),
    )
    expected = await built_in.compute(target, ground_truths(truth()))
    self.assertEqual(result.status, MetricComputationStatus.COMPLETED)
    # DeepEval's weighted cumulative precision of yes, no, yes.
    self.assertAlmostEqual(result.score or 0.0, (1 + 2 / 3) / 2)
    self.assertEqual(result.score, expected.score)
    self.assertEqual(without_judge_keys(result.metadata), without_judge_keys(expected.metadata))
    self.assertEqual(result.raw_output, expected.raw_output)
    assert result.metadata is not None
    self.assertEqual(result.metadata['judge_calls'], 1)
    self.assertEqual(result.metadata['judge_usage'], {'input_tokens': 100, 'output_tokens': 7, 'total_tokens': 107})

    [call] = chat.calls
    prompt = prompt_of(call)
    self.assertIn('remotely useful in arriving at the expected output', prompt)
    self.assertIn(QUESTION, prompt)
    self.assertIn(EXPECTED_ANSWER, prompt)
    self.assertIn(str(['Text of d1.', 'Text of d2.', 'Text of d3.']), prompt)
    self.assertIn('(3 documents)', prompt)
    # include_reason=False: DeepEval writes no summary reason, so the unit is a single call.
    self.assertEqual(call['schema'], PrecisionVerdicts.model_json_schema())

  async def test_an_observed_empty_ranking_makes_no_call(self) -> None:
    chat = FakeChatModel(verdicts((True, 'r')))
    items = [RetrievalItem(id='d1', content='Text of d1.')]
    target = span(
      's1',
      SEARCH_SPAN_TYPE,
      request=QUESTION,
      retrieval=[
        RetrievalResult(kind='document', stage='selected', items=[], ranked=True),
        RetrievalResult(kind='document', stage='selected', items=items, ranked=True),
      ],
    )

    result = await self.metric(chat).compute(target, ground_truths(truth()))

    self.assertEqual(len(chat.calls), 1)
    self.assertEqual(result.score, 0.5)

  async def test_a_wrong_verdict_count_fails_the_unit_as_misaligned_or_truncated(self) -> None:
    # Too few verdicts, too many, and too few from an output that used its whole output limit.
    for count, output_tokens, failure in [(2, 7, 'misaligned'), (4, 7, 'misaligned'), (2, OUTPUT_LIMIT, 'truncated')]:
      with self.subTest(count=count, failure=failure):
        output = verdicts(*[(True, 'r')] * count)
        chat = FakeChatModel(output, output_tokens=output_tokens)

        result = await self.metric(chat).compute(search_span('s1', ['d1', 'd2', 'd3']), ground_truths(truth()))

        self.assertEqual(result.status, MetricComputationStatus.FAILED)
        self.assertIn(f'{count} contextual precision verdicts for 3 ranked documents', result.error_message or '')
        assert result.metadata is not None
        self.assertEqual(result.metadata['failure'], failure)
        self.assertEqual(result.metadata['judge_calls'], 1)
        self.assertEqual(result.raw_output, output)

  async def test_concurrent_units_each_report_their_own_calls(self) -> None:
    def reply(prompt: str) -> dict[str, Any]:
      return verdicts(*[(True, 'r')] * prompt.count("'Text of "))

    # Usage depends on the prompt, so a call recorded for the wrong unit shows in its usage.
    chat = FakeChatModel(reply, input_tokens=len)
    two_rankings = span(
      'u2',
      SEARCH_SPAN_TYPE,
      request=QUESTION,
      retrieval=[
        RetrievalResult(
          kind='document',
          stage='selected',
          ranked=True,
          items=[RetrievalItem(id=doc, content=f'Text of {doc}.') for doc in documents],
        )
        for documents in (['u2d0'], ['u2d1', 'u2d2'])
      ],
    )
    units = [search_span('u0', ['u0d0']), search_span('u1', ['u1d0', 'u1d1']), two_rankings]
    units.append(search_span('u3', ['u3d0', 'u3d1', 'u3d2']))
    metric = self.metric(chat)

    results = await asyncio.gather(*(metric.compute(unit, ground_truths(truth())) for unit in units))

    for index, result in enumerate(results):
      with self.subTest(unit=index):
        own = [call for call in chat.calls if f'Text of u{index}d' in prompt_of(call)]
        assert result.metadata is not None
        self.assertEqual(result.metadata['judge_calls'], len(own))
        self.assertEqual(result.metadata['judge_usage']['input_tokens'], sum(len(prompt_of(call)) for call in own))
        response_ids = result.metadata.get('judge_response_ids') or [result.metadata['judge_response_id']]
        self.assertEqual(response_ids, [call['id'] for call in own])
    self.assertEqual(len(chat.calls), 5)
    # The units overlapped, up to the judge's limit of two calls in flight.
    self.assertEqual(chat.load['peak'], 2)


class DeepEvalGoldClaimsContextualRecallTest(unittest.IsolatedAsyncioTestCase):
  def metric(self, chat: FakeChatModel) -> DeepEvalGoldClaimsContextualRecall:
    return DeepEvalGoldClaimsContextualRecall(
      judge=deepeval_judge(chat), output_token_limit=OUTPUT_LIMIT, target_span_types=(SEARCH_SPAN_TYPE,)
    )

  async def test_judges_every_claim_in_one_call_one_claim_per_line(self) -> None:
    chat = FakeChatModel(verdicts((True, 'why'), (False, 'why')))
    target = search_span('s1', ['d1', 'd2'])

    result = await self.metric(chat).compute(target, ground_truths(claims_truth(*CLAIMS)))

    built_in = GoldClaimsContextualRecall(
      judge_client=ScriptedJudge(recall_judgments((CLAIMS[0].text, True)), recall_judgments((CLAIMS[1].text, False))),
      target_span_types=(SEARCH_SPAN_TYPE,),
    )
    expected = await built_in.compute(target, ground_truths(claims_truth(*CLAIMS)))
    self.assertEqual(result.score, 0.5)
    self.assertEqual(without_judge_keys(result.metadata), without_judge_keys(expected.metadata))
    assert result.metadata is not None
    self.assertEqual([entry['claim_id'] for entry in result.metadata['statement_results']], ['c1', 'c2'])
    self.assertEqual(result.metadata['judge_calls'], 1)

    [call] = chat.calls
    prompt = prompt_of(call)
    self.assertIn('For EACH sentence in the given expected output', prompt)
    self.assertIn(f'Expected Output:\n{CLAIMS[0].text}\n{CLAIMS[1].text}\n', prompt)
    self.assertIn(str(['Text of d1.', 'Text of d2.']), prompt)
    self.assertEqual(call['schema'], RecallVerdicts.model_json_schema())

  async def test_a_wrong_verdict_count_fails_the_unit_as_misaligned(self) -> None:
    for count in (1, 3):
      with self.subTest(count=count):
        chat = FakeChatModel(verdicts(*[(True, 'why')] * count))

        result = await self.metric(chat).compute(search_span('s1', ['d1']), ground_truths(claims_truth(*CLAIMS)))

        self.assertEqual(result.status, MetricComputationStatus.FAILED)
        self.assertIn(f'{count} contextual recall verdicts for 2 statements', result.error_message or '')
        assert result.metadata is not None
        self.assertEqual((result.metadata['failure'], result.metadata['judge_calls']), ('misaligned', 1))

  async def test_a_claim_spanning_lines_is_refused_before_judging(self) -> None:
    chat = FakeChatModel()

    with self.assertRaises(ValueError):
      await self.metric(chat).compute(
        search_span('s1', ['d1']), ground_truths(claims_truth(Claim('c1', 'Dana approved it.\nIn May.')))
      )

    self.assertEqual(chat.calls, [])


class DeepEvalAnswerCorrectnessTest(unittest.IsolatedAsyncioTestCase):
  async def test_grades_with_the_documented_correctness_steps_in_one_call(self) -> None:
    chat = FakeChatModel({'reason': 'Matches the expected answer.', 'score': 7})
    metric = DeepEvalAnswerCorrectness(judge=deepeval_judge(chat), output_token_limit=OUTPUT_LIMIT)

    result = await metric.compute(answer_span(), ground_truths(truth()))

    self.assertEqual(result.status, MetricComputationStatus.COMPLETED)
    # G-Eval scores 0 to 10, and DeepEval divides by 10.
    self.assertEqual(result.score, 0.7)
    self.assertEqual(result.reasoning, 'Matches the expected answer.')
    assert result.metadata is not None
    self.assertEqual(result.metadata['judge_calls'], 1)
    # The steps are given, so DeepEval generates none and makes a single call.
    [call] = chat.calls
    self.assertEqual(call['schema'], ReasonScore.model_json_schema())
    prompt = prompt_of(call)
    for number, step in enumerate(CORRECTNESS_STEPS, start=1):
      self.assertIn(f'{number}. {step}\n', prompt)
    for value in (QUESTION, 'Dana approved it, in May.', EXPECTED_ANSWER):
      self.assertIn(value, prompt)

  async def test_skips_a_trace_without_an_answer(self) -> None:
    chat = FakeChatModel()
    metric = DeepEvalAnswerCorrectness(judge=deepeval_judge(chat), output_token_limit=OUTPUT_LIMIT)

    result = await metric.compute(answer_span(None), ground_truths(truth()))

    self.assertEqual(result.status, MetricComputationStatus.SKIPPED)
    self.assertEqual(chat.calls, [])


class DeepEvalFailureTest(unittest.IsolatedAsyncioTestCase):
  async def test_every_metric_records_the_library_failure_classes(self) -> None:
    units: dict[str, tuple[Span, dict[str, GroundTruth]]] = {
      'precision': (search_span('s1', ['d1']), ground_truths(truth())),
      'recall': (search_span('s1', ['d1']), ground_truths(claims_truth(*CLAIMS))),
      'answer': (answer_span(), ground_truths(truth())),
    }
    cases = [
      (_Unparsed(ValueError('bad JSON')), 7, 'invalid_output'),
      (_Unparsed(ValueError('cut off')), OUTPUT_LIMIT, 'truncated'),
      ({'reason': 'No verdicts and no score.'}, 7, 'invalid_output'),
      (httpx.ReadTimeout('slow'), 7, 'timeout'),
      (RuntimeError('500 internal'), 7, 'provider'),
    ]
    for key, (target, truths) in units.items():
      for reply, output_tokens, failure in cases:
        with self.subTest(metric=key, reply=reply):
          chat = FakeChatModel(reply, output_tokens=output_tokens)
          metric = deepeval_metrics(deepeval_judge(chat), output_token_limit=OUTPUT_LIMIT)[key]

          result = await metric.compute(target, truths)

          self.assertEqual(result.status, MetricComputationStatus.FAILED)
          assert result.metadata is not None
          self.assertEqual(result.metadata['failure'], failure)
          self.assertEqual((result.metadata['judge_calls'], result.metadata['judge_failed_calls']), (1, 1))
          # An output that came back still cost its tokens; a call that raised reported none.
          self.assertEqual('judge_usage' in result.metadata, not isinstance(reply, Exception))


class DeepEvalSetupTest(unittest.IsolatedAsyncioTestCase):
  def test_metrics_score_the_main_pass_units_under_their_own_names(self) -> None:
    metrics = deepeval_metrics(deepeval_judge(FakeChatModel()), output_token_limit=OUTPUT_LIMIT)

    self.assertEqual(tuple(metrics), METRIC_KEYS)
    self.assertEqual(
      [metric.name for metric in metrics.values()],
      ['deepeval_contextual_precision', 'deepeval_contextual_recall_gold_claims', 'deepeval_answer_correctness'],
    )
    self.assertEqual(metrics['precision'].target_span_types, (SEARCH_SPAN_TYPE,))
    self.assertEqual(metrics['recall'].target_span_types, (SEARCH_SPAN_TYPE,))
    self.assertEqual(metrics['recall'].ground_truth_keys, (CLAIMS_KEY,))
    self.assertEqual(metrics['answer'].target_span_types, ('agent_root',))
    self.assertEqual(metrics['answer'].ground_truth_keys, ('expected_output',))

  def test_runs_the_pinned_version_offline(self) -> None:
    self.assertEqual(DEEPEVAL_VERSION, '4.2.6')
    self.assertTrue(telemetry_opt_out())
    self.assertEqual(os.environ['DEEPEVAL_DISABLE_DOTENV'], '1')
    self.assertEqual(HIDDEN_DIR, str(EXPERIMENTS_DIR / 'outputs' / 'deepeval'))

  def test_a_node_is_the_document_title_then_its_content(self) -> None:
    titled = RetrievedItem(RetrievalItem(id='d1', title='Budget policy', content='Dana approves budgets.'))
    untitled = RetrievedItem(RetrievalItem(id='d2', content='Dana approves budgets.'))

    self.assertEqual(node_text(titled), 'Budget policy\n\nDana approves budgets.')
    self.assertEqual(node_text(untitled), 'Dana approves budgets.')

  async def test_open_builds_the_shared_chat_model_and_closes_it(self) -> None:
    config = load_config().models.judge.model_copy(update={'temperature': 1.0, 'thinking_level': 'low'})
    chat_model = MagicMock()
    chat_model.model = config.model
    chat_model.model_copy.return_value = chat_model
    chat_model.async_client.aclose = AsyncMock()
    with patch('judge.ChatGoogleGenerativeAI', return_value=chat_model) as chat_model_class:
      async with open_deepeval_judge(config, GeminiJudgeSettings(api_key='test-key')) as judge:
        self.assertEqual(judge.get_model_name(), config.model)
        chat_model.async_client.aclose.assert_not_awaited()

    kwargs = chat_model_class.call_args.kwargs
    self.assertEqual(
      {key: kwargs[key] for key in ('model', 'api_key', 'timeout', 'thinking_level', 'max_retries')},
      {
        'model': config.model,
        'api_key': 'test-key',
        'timeout': config.timeout_seconds,
        'thinking_level': 'low',
        'max_retries': 1,
      },
    )
    chat_model.model_copy.assert_called_once_with(update={'temperature': 1.0})
    chat_model.async_client.aclose.assert_awaited_once()

  async def test_open_takes_the_attempts_and_the_concurrency_limit_from_the_config(self) -> None:
    judge_config = load_config().models.judge
    for limit in (1, 2):
      with self.subTest(limit=limit):
        chat = FakeChatModel(verdicts((True, 'r')), verdicts((True, 'r')), verdicts((True, 'r')))
        config = judge_config.model_copy(update={'max_concurrent_requests': limit})
        with patch('judge.ChatGoogleGenerativeAI', return_value=chat):
          async with open_deepeval_judge(config, GeminiJudgeSettings(api_key='test-key')) as judge:
            with recorded_judge_calls():
              await asyncio.gather(*(judge.a_generate(f'Judge {n}.', PrecisionVerdicts) for n in range(3)))
        self.assertEqual(chat.load['peak'], limit)

    chat = FakeChatModel(rate_limit(), rate_limit(), verdicts((True, 'r')))
    config = judge_config.model_copy(update={'max_retries': 2})
    with (
      patch('judge.ChatGoogleGenerativeAI', return_value=chat),
      patch('deepeval_baseline.asyncio.sleep', new_callable=AsyncMock),
    ):
      async with open_deepeval_judge(config, GeminiJudgeSettings(api_key='test-key')) as judge:
        with recorded_judge_calls(), self.assertRaises(ExternalServiceError):
          await judge.a_generate('Judge.', PrecisionVerdicts)
    self.assertEqual(len(chat.calls), 2)


if __name__ == '__main__':
  unittest.main()
