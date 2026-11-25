import asyncio
import re
import unittest
from typing import Any, cast
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
from langchain_google_genai import ChatGoogleGenerativeAI

from syllo_eval.evaluation.metric_support.agent_outputs import extract_actual_plan, ground_truth_text
from syllo_eval.evaluation.metrics.implementations.answer.correctness_judge import AnswerCorrectnessJudgeMetric
from syllo_eval.evaluation.metrics.implementations.plan.correctness_judge import PlanCorrectnessJudgeMetric
from syllo_eval.evaluation.metrics.implementations.rag.contextual_precision_judge import (
  ContextualPrecisionDocumentJudgeMetric,
)
from syllo_eval.infrastructure.exceptions import ExternalServiceError, JudgeOutputError
from syllo_eval.model import GroundTruth, GroundTruthKey, MetricComputationStatus, Span
from syllo_eval.settings import GeminiJudgeSettings
from syllo_eval.trace_semantics import Answer, ExecutionStep, PlanningData, RetrievalItem, RetrievalResult

from ablation_metrics import SEARCH_SPAN_TYPE, GoldClaimsContextualRecall, OrbitalsClaimsContextualRecall
from baseline_judge import recorded_judge_calls
from benchmarks.common import Claim
from benchmarks.erb import CLAIMS_KEY as ERB_CLAIMS_KEY
from benchmarks.orbitals_claims import CLAIMS_KEY as ORBITALS_CLAIMS_KEY
from config import Benchmark, load_config
from judge_fakes import FakeChatModel, Unparsed, prompt_text, rate_limit
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
from openevals_baseline import (
  METRIC_KEYS,
  OPENEVALS_VERSION,
  OpenEvalsCorrectness,
  OpenEvalsGoldClaimsGroundedness,
  OpenEvalsJudge,
  OpenEvalsOrbitalsClaimsGroundedness,
  OpenEvalsPlanAdherence,
  OpenEvalsRetrievalRelevance,
  open_openevals_judge,
  openevals_metrics,
)

OUTPUT_LIMIT = 1_000
USAGE = {'input_tokens': 100, 'output_tokens': 7, 'total_tokens': 107}
CLAIMS = (Claim('c1', 'Dana approved the budget.'), Claim('c2', 'The approval was in May.'))
SCENARIO = 'You are Sam. Cancel order #W1, which you no longer need.'
EXPECTED_PLAN = [{'operation': 'cancel_pending_order', 'instruction': '', 'parameters': {'order_id': '#W1'}}]
# OpenEvals' default output: a reasoning, then a boolean score.
SCHEMA: dict[str, Any] = {
  'title': 'score',
  'description': 'A score that is true if criteria in the prompt are met, and false otherwise.',
  'type': 'object',
  'additionalProperties': False,
  'properties': {
    'reasoning': {'type': 'string', 'description': 'Why.'},
    'score': {'type': 'boolean', 'description': 'A score that is true if criteria in the prompt are met.'},
  },
  'required': ['reasoning', 'score'],
}


def openevals_judge(chat: FakeChatModel, *, max_attempts: int = 3) -> OpenEvalsJudge:
  return OpenEvalsJudge(
    cast(ChatGoogleGenerativeAI, chat), temperature=0.0, max_concurrent_requests=2, max_attempts=max_attempts
  )


def verdict(met: bool, reasoning: str = 'Thus, the score should be: true.') -> dict[str, Any]:
  return {'reasoning': reasoning, 'score': met}


def judged_document(prompt: str) -> str:
  """The id of the document a relevance prompt holds: each fake document's text names it."""
  [document] = re.findall(r'Text of (\w+)\.', prompt)
  return document


def answer_span(text: str | None = 'Dana approved it, in May.') -> Span:
  answer = Answer(text=text) if text is not None else None
  return span('root', 'agent_root', parent=None, request=QUESTION, answer=answer)


def plan_span() -> Span:
  steps = [
    ExecutionStep(
      id='s1',
      operation='cancel_pending_order',
      status='completed',
      input={'order_id': '#W1', 'reason': 'no longer needed'},
      output='{"status": "cancelled"}',
    )
  ]
  return span('root', 'agent_root', parent=None, request=SCENARIO, planning=PlanningData(executed_steps=steps))


def plan_truth() -> dict[str, GroundTruth]:
  return ground_truths(truth(GroundTruthKey.EXPECTED_PLAN.value, expected_plan=EXPECTED_PLAN))


class OpenEvalsJudgeTest(unittest.IsolatedAsyncioTestCase):
  async def test_sends_the_prompt_alone_with_the_default_output_schema_and_records_the_call(self) -> None:
    chat = FakeChatModel(verdict(True, 'Accurate.'))
    messages = [{'role': 'user', 'content': 'Grade this output.'}]

    with recorded_judge_calls() as calls:
      output = await openevals_judge(chat).with_structured_output(SCHEMA).ainvoke(messages)

    self.assertEqual(output, {'reasoning': 'Accurate.', 'score': True})
    [call] = chat.calls
    self.assertEqual(call['messages'], messages)
    self.assertIs(call['schema'], SCHEMA)
    self.assertEqual((call['method'], call['include_raw']), ('json_schema', True))
    # The configured temperature, and no output limit: OpenEvals sets none.
    self.assertEqual(call['options'], {'temperature': 0.0})
    [response] = calls
    self.assertEqual((response.provider, response.model, response.response_id), ('gemini', 'judge-1-001', 'response-1'))
    self.assertEqual(response.usage, USAGE)
    self.assertEqual(response.attempts, 1)

  def test_refuses_any_output_but_the_default_one(self) -> None:
    continuous = {**SCHEMA, 'properties': {**SCHEMA['properties'], 'score': {'type': 'number'}}}
    without_reasoning = {**SCHEMA, 'properties': {'score': {'type': 'boolean'}}, 'required': ['score']}
    judge = openevals_judge(FakeChatModel())
    cases: list[tuple[dict[str, Any], dict[str, Any]]] = [
      (continuous, {}),
      (without_reasoning, {}),
      (SCHEMA, {'include_raw': True}),
      (SCHEMA, {'method': 'function_calling'}),
    ]
    for schema, options in cases:
      with self.subTest(schema=schema, options=options), self.assertRaises(ValueError):
        judge.with_structured_output(schema, **options)

  async def test_output_that_does_not_validate_raises_with_its_usage(self) -> None:
    replies = [
      {'reasoning': 'r', 'score': 'true'},
      {'reasoning': 'r'},
      {'reasoning': 'r', 'score': True, 'confidence': 0.9},
      None,
      Unparsed(ValueError('bad JSON')),
    ]
    for reply in replies:
      with self.subTest(reply=reply):
        runnable = openevals_judge(FakeChatModel(reply)).with_structured_output(SCHEMA)
        with recorded_judge_calls() as calls, self.assertRaises(JudgeOutputError) as raised:
          await runnable.ainvoke([{'role': 'user', 'content': 'Judge.'}])
        self.assertEqual(raised.exception.usage, USAGE)
        self.assertEqual(calls, [])

  async def test_retries_rate_limits_after_the_server_hint_and_counts_the_attempts(self) -> None:
    chat = FakeChatModel(rate_limit(), verdict(True))
    runnable = openevals_judge(chat).with_structured_output(SCHEMA)
    with (
      patch('baseline_judge.asyncio.sleep', new_callable=AsyncMock) as sleep,
      patch('syllo_eval.infrastructure.llm_judge.gemini.random.uniform', return_value=1.0),
      recorded_judge_calls() as calls,
    ):
      await runnable.ainvoke([{'role': 'user', 'content': 'Judge.'}])

    sleep.assert_awaited_once_with(2.0)
    self.assertEqual([response.attempts for response in calls], [2])

  async def test_latency_is_that_of_the_attempt_that_succeeded(self) -> None:
    chat = FakeChatModel(rate_limit(), verdict(True))
    runnable = openevals_judge(chat).with_structured_output(SCHEMA)
    # The clock at each attempt's start, then at the successful attempt's end: the wait for a retry is left out.
    clock = iter([0.0, 10.0, 11.5])
    with (
      patch('baseline_judge.time', MagicMock(monotonic=lambda: next(clock))),
      patch('baseline_judge.asyncio.sleep', new_callable=AsyncMock),
      patch('syllo_eval.infrastructure.llm_judge.gemini.random.uniform', return_value=1.0),
      recorded_judge_calls() as calls,
    ):
      await runnable.ainvoke([{'role': 'user', 'content': 'Judge.'}])

    [response] = calls
    self.assertEqual((response.attempts, response.latency_seconds), (2, 1.5))


class OpenEvalsRetrievalRelevanceTest(unittest.IsolatedAsyncioTestCase):
  def metric(self, chat: FakeChatModel) -> OpenEvalsRetrievalRelevance:
    return OpenEvalsRetrievalRelevance(
      judge=openevals_judge(chat), output_token_limit=OUTPUT_LIMIT, target_span_types=(SEARCH_SPAN_TYPE,)
    )

  async def test_judges_each_document_in_its_own_call_and_scores_the_verdicts_as_precision(self) -> None:
    def reply(prompt: str) -> dict[str, Any]:
      document = judged_document(prompt)
      return verdict(document != 'd2', f'{document} reasoning')

    chat = FakeChatModel(reply)
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
    self.assertAlmostEqual(result.score or 0.0, (1 + 2 / 3) / 2)
    self.assertEqual(result.score, expected.score)
    self.assertEqual(without_judge_keys(result.metadata), without_judge_keys(expected.metadata))
    self.assertEqual(result.raw_output, expected.raw_output)
    assert result.metadata is not None
    self.assertEqual(result.metadata['judge_calls'], 3)

    self.assertEqual(sorted(judged_document(prompt_text(call['messages'])) for call in chat.calls), ['d1', 'd2', 'd3'])
    prompt = prompt_text(chat.calls[0]['messages'])
    self.assertIn('evaluating retrieved context for relevance to the input', prompt)
    self.assertIn(f'<inputs>\n{QUESTION}\n</inputs>', prompt)
    # The context as OpenEvals' examples give it, rendered by the prompt with str().
    self.assertRegex(prompt, r"<retrieved_context>\n\{'documents': \['Text of d\d\.'\]\}\n</retrieved_context>")
    # OpenEvals' relevance reads no reference.
    self.assertNotIn(EXPECTED_ANSWER, prompt)

  async def test_a_document_is_its_title_then_its_content_as_a_python_string(self) -> None:
    chat = FakeChatModel(verdict(True))
    items = [RetrievalItem(id='d1', title='Budget policy', content='Dana approves budgets.')]
    ranking = RetrievalResult(kind='document', stage='selected', items=items, ranked=True)
    target = span('s1', SEARCH_SPAN_TYPE, request=QUESTION, retrieval=[ranking])

    await self.metric(chat).compute(target, ground_truths(truth()))

    # The blank line between title and content reaches the judge escaped, as str() renders it.
    context = "{'documents': ['Budget policy\\n\\nDana approves budgets.']}"
    self.assertIn(f'<retrieved_context>\n{context}\n</retrieved_context>', prompt_text(chat.calls[0]['messages']))

  async def test_an_observed_empty_ranking_makes_no_call(self) -> None:
    chat = FakeChatModel(verdict(True))
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


class OpenEvalsGroundednessTest(unittest.IsolatedAsyncioTestCase):
  def metric(self, chat: FakeChatModel) -> OpenEvalsGoldClaimsGroundedness:
    return OpenEvalsGoldClaimsGroundedness(
      judge=openevals_judge(chat), output_token_limit=OUTPUT_LIMIT, target_span_types=(SEARCH_SPAN_TYPE,)
    )

  async def test_judges_each_claim_against_every_document_and_scores_the_verdicts_as_recall(self) -> None:
    def reply(prompt: str) -> dict[str, Any]:
      return verdict(f'<outputs>\n{CLAIMS[0].text}\n</outputs>' in prompt, 'why')

    chat = FakeChatModel(reply)
    target = search_span('s1', ['d1', 'd2'], stage='retrieved')

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
    self.assertEqual(result.metadata['judge_calls'], 2)

    prompts = [prompt_text(call['messages']) for call in chat.calls]
    self.assertEqual(
      sorted(claim.text for claim in CLAIMS for prompt in prompts if f'<outputs>\n{claim.text}\n</outputs>' in prompt),
      sorted(claim.text for claim in CLAIMS),
    )
    for prompt in prompts:
      self.assertIn('how well LLM output aligns with and is supported by the retrieved context', prompt)
      self.assertIn("<context>\n{'documents': ['Text of d1.', 'Text of d2.']}\n</context>", prompt)

  async def test_every_claim_reads_each_document_title_then_its_content(self) -> None:
    chat = FakeChatModel(verdict(True))
    items = [
      RetrievalItem(id='d1', title='Budget policy', content='Dana approves budgets.'),
      RetrievalItem(id='d2', content='Approvals happen in May.'),
    ]
    evidence = RetrievalResult(kind='document', stage='retrieved', items=items, ranked=False)
    target = span('root', 'agent_root', parent=None, request=QUESTION, retrieval=[evidence])

    await self.metric(chat).compute(target, ground_truths(claims_truth(CLAIMS[0])))

    context = "{'documents': ['Budget policy\\n\\nDana approves budgets.', 'Approvals happen in May.']}"
    self.assertIn(f'<context>\n{context}\n</context>', prompt_text(chat.calls[0]['messages']))

  async def test_the_wixqa_variant_reads_the_orbitals_claims_as_the_main_pass_does(self) -> None:
    chat = FakeChatModel(verdict(True), verdict(True))
    metric = OpenEvalsOrbitalsClaimsGroundedness(judge=openevals_judge(chat), output_token_limit=OUTPUT_LIMIT)

    self.assertEqual(
      metric.ground_truth_keys, OrbitalsClaimsContextualRecall(judge_client=MagicMock()).ground_truth_keys
    )
    self.assertEqual(metric.ground_truth_keys, (ORBITALS_CLAIMS_KEY,))
    self.assertEqual(metric.name, 'openevals_groundedness_orbitals_claims')
    result = await metric.compute(
      search_span('s1', ['d1'], stage='retrieved'), ground_truths(claims_truth(*CLAIMS, key=ORBITALS_CLAIMS_KEY))
    )
    self.assertEqual((result.score, len(chat.calls)), (1.0, 2))


class OpenEvalsCorrectnessTest(unittest.IsolatedAsyncioTestCase):
  async def test_grades_the_answer_against_the_expected_one_in_one_call(self) -> None:
    for correct, score in ((True, 1.0), (False, 0.0)):
      with self.subTest(correct=correct):
        chat = FakeChatModel(verdict(correct, 'Matches the reference.'))
        metric = OpenEvalsCorrectness(judge=openevals_judge(chat), output_token_limit=OUTPUT_LIMIT)

        result = await metric.compute(answer_span(), ground_truths(truth()))

        self.assertEqual((result.status, result.score), (MetricComputationStatus.COMPLETED, score))
        self.assertEqual(result.reasoning, 'Matches the reference.')
        assert result.metadata is not None
        self.assertEqual((result.metadata['judge_calls'], result.metadata['judge_usage']), (1, USAGE))
        [call] = chat.calls
        prompt = prompt_text(call['messages'])
        self.assertIn('evaluating model outputs for correctness', prompt)
        for tag, value in (('input', QUESTION), ('output', 'Dana approved it, in May.')):
          self.assertIn(f'<{tag}>\n{value}\n</{tag}>', prompt)
        self.assertIn(f'<reference_outputs>\n{EXPECTED_ANSWER}\n</reference_outputs>', prompt)

  async def test_skips_a_trace_without_an_answer(self) -> None:
    chat = FakeChatModel()
    metric = OpenEvalsCorrectness(judge=openevals_judge(chat), output_token_limit=OUTPUT_LIMIT)

    result = await metric.compute(answer_span(None), ground_truths(truth()))

    self.assertEqual(result.status, MetricComputationStatus.SKIPPED)
    self.assertEqual(chat.calls, [])


class OpenEvalsPlanAdherenceTest(unittest.IsolatedAsyncioTestCase):
  async def test_grades_the_steps_against_the_expected_plan_as_plan_correctness_renders_them(self) -> None:
    chat = FakeChatModel(verdict(True, 'Every planned step was executed.'))
    metric = OpenEvalsPlanAdherence(judge=openevals_judge(chat), output_token_limit=OUTPUT_LIMIT)
    target, truths = plan_span(), plan_truth()

    result = await metric.compute(target, truths)

    self.assertEqual((result.score, result.reasoning), (1.0, 'Every planned step was executed.'))
    [call] = chat.calls
    prompt = prompt_text(call['messages'])
    self.assertIn('whether an AI agent followed its declared plan', prompt)
    expected_plan = ground_truth_text(truths[GroundTruthKey.EXPECTED_PLAN.value], 'expected_plan')
    for tag, value in (('input', SCENARIO), ('plan', expected_plan), ('output', extract_actual_plan(target))):
      self.assertIn(f'<{tag}>\n{value}\n</{tag}>', prompt)

  async def test_skips_a_trace_without_executed_steps(self) -> None:
    chat = FakeChatModel()
    metric = OpenEvalsPlanAdherence(judge=openevals_judge(chat), output_token_limit=OUTPUT_LIMIT)

    result = await metric.compute(span('root', 'agent_root', parent=None, request=SCENARIO), plan_truth())

    self.assertEqual(result.status, MetricComputationStatus.SKIPPED)
    self.assertEqual(chat.calls, [])


class OpenEvalsFailureTest(unittest.IsolatedAsyncioTestCase):
  async def test_every_metric_records_the_library_failure_classes(self) -> None:
    units: dict[str, tuple[Span, dict[str, GroundTruth]]] = {
      'relevance': (search_span('s1', ['d1']), ground_truths(truth())),
      'recall': (search_span('s1', ['d1'], stage='retrieved'), ground_truths(claims_truth(CLAIMS[0]))),
      'answer': (answer_span(), ground_truths(truth())),
      'plan': (plan_span(), plan_truth()),
    }
    cases = [
      (Unparsed(ValueError('bad JSON')), 7, 'invalid_output'),
      (Unparsed(ValueError('cut off')), OUTPUT_LIMIT, 'truncated'),
      ({'reasoning': 'No score.'}, 7, 'invalid_output'),
      (None, 7, 'invalid_output'),
      (httpx.ReadTimeout('slow'), 7, 'timeout'),
      (RuntimeError('500 internal'), 7, 'provider'),
    ]
    for key, (target, truths) in units.items():
      for reply, output_tokens, failure in cases:
        with self.subTest(metric=key, reply=reply):
          chat = FakeChatModel(reply, output_tokens=output_tokens)
          benchmark: Benchmark = 'tau2' if key == 'plan' else 'erb'
          metric = openevals_metrics(benchmark, openevals_judge(chat), output_token_limit=OUTPUT_LIMIT)[key]

          result = await metric.compute(target, truths)

          self.assertEqual(result.status, MetricComputationStatus.FAILED)
          assert result.metadata is not None
          self.assertEqual(result.metadata['failure'], failure)
          self.assertEqual((result.metadata['judge_calls'], result.metadata['judge_failed_calls']), (1, 1))
          # An output that came back still cost its tokens; a call that raised reported none.
          self.assertEqual('judge_usage' in result.metadata, not isinstance(reply, Exception))


class OpenEvalsSetupTest(unittest.IsolatedAsyncioTestCase):
  def test_each_benchmark_gets_one_metric_for_each_judge_metric_of_the_main_arm(self) -> None:
    judge = openevals_judge(FakeChatModel())
    erb = openevals_metrics('erb', judge, output_token_limit=OUTPUT_LIMIT)
    wixqa = openevals_metrics('wixqa', judge, output_token_limit=OUTPUT_LIMIT)
    tau2 = openevals_metrics('tau2', judge, output_token_limit=OUTPUT_LIMIT)

    self.assertEqual(
      {key: metric.name for key, metric in erb.items()},
      {
        'relevance': 'openevals_retrieval_relevance',
        'recall': 'openevals_groundedness_gold_claims',
        'answer': 'openevals_correctness',
      },
    )
    self.assertEqual(wixqa['recall'].name, 'openevals_groundedness_orbitals_claims')
    self.assertEqual({key: metric.name for key, metric in tau2.items()}, {'plan': 'openevals_plan_adherence'})
    self.assertEqual(set(METRIC_KEYS), {*erb, *tau2})
    # Each metric scores the units of the metric it is compared with.
    compared = {
      'relevance': ContextualPrecisionDocumentJudgeMetric,
      'recall': GoldClaimsContextualRecall,
      'answer': AnswerCorrectnessJudgeMetric,
    }
    for key, compared_class in compared.items():
      self.assertIsInstance(erb[key], compared_class)
    self.assertIsInstance(tau2['plan'], PlanCorrectnessJudgeMetric)
    self.assertEqual(erb['relevance'].target_span_types, (SEARCH_SPAN_TYPE,))
    self.assertEqual(erb['recall'].target_span_types, ('agent_root',))
    self.assertEqual(OpenEvalsGoldClaimsGroundedness.retrieval_stage, 'retrieved')
    self.assertEqual(erb['recall'].ground_truth_keys, (ERB_CLAIMS_KEY,))
    self.assertEqual(wixqa['recall'].ground_truth_keys, (ORBITALS_CLAIMS_KEY,))
    self.assertEqual(erb['answer'].ground_truth_keys, ('expected_output',))

  def test_runs_the_pinned_version(self) -> None:
    self.assertEqual(OPENEVALS_VERSION, '0.2.2')

  async def test_open_builds_the_shared_chat_model_and_closes_it(self) -> None:
    config = load_config().models.judge.model_copy(update={'temperature': 1.0, 'thinking_level': 'low'})
    chat_model = MagicMock()
    chat_model.model = config.model
    chat_model.model_copy.return_value = chat_model
    chat_model.async_client.aclose = AsyncMock()
    with patch('judge.ChatGoogleGenerativeAI', return_value=chat_model) as chat_model_class:
      async with open_openevals_judge(config, GeminiJudgeSettings(api_key='test-key')) as judge:
        self.assertIsInstance(judge, OpenEvalsJudge)
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
        chat = FakeChatModel(verdict(True), verdict(True), verdict(True))
        config = judge_config.model_copy(update={'max_concurrent_requests': limit})
        with patch('judge.ChatGoogleGenerativeAI', return_value=chat):
          async with open_openevals_judge(config, GeminiJudgeSettings(api_key='test-key')) as judge:
            runnable = judge.with_structured_output(SCHEMA)
            with recorded_judge_calls():
              await asyncio.gather(*(runnable.ainvoke([{'role': 'user', 'content': f'Judge {n}.'}]) for n in range(3)))
        self.assertEqual(chat.load['peak'], limit)

    chat = FakeChatModel(rate_limit(), rate_limit(), verdict(True))
    config = judge_config.model_copy(update={'max_retries': 2})
    with (
      patch('judge.ChatGoogleGenerativeAI', return_value=chat),
      patch('baseline_judge.asyncio.sleep', new_callable=AsyncMock),
    ):
      async with open_openevals_judge(config, GeminiJudgeSettings(api_key='test-key')) as judge:
        runnable = judge.with_structured_output(SCHEMA)
        with recorded_judge_calls(), self.assertRaises(ExternalServiceError):
          await runnable.ainvoke([{'role': 'user', 'content': 'Judge.'}])
    self.assertEqual(len(chat.calls), 2)


if __name__ == '__main__':
  unittest.main()
