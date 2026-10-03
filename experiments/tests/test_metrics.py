import re
import unittest
from typing import Any

import httpx
from langchain_core.exceptions import ContextOverflowError

from syllo_eval.evaluation.judge import LlmJudgeRequest
from syllo_eval.evaluation.metric_support.retrieved_context import extract_retrieved_items
from syllo_eval.evaluation.metrics.implementations.answer.correctness_judge import AnswerCorrectnessJudgeMetric
from syllo_eval.evaluation.metrics.implementations.plan.correctness_judge import PlanCorrectnessJudgeMetric
from syllo_eval.evaluation.metrics.implementations.rag.contextual_precision_judge import (
  ContextualPrecisionDocumentJudgeMetric,
)
from syllo_eval.evaluation.metrics.implementations.rag.contextual_recall_judge import (
  ContextualRecallDocumentJudgeMetric,
)
from syllo_eval.infrastructure.exceptions import DataMappingError, ExternalServiceError
from syllo_eval.model import MetricComputationStatus
from syllo_eval.trace_semantics import Answer, ExecutionStep, PlanningData

from benchmarks.common import Claim
from benchmarks.erb import CLAIMS_KEY
from metric_fakes import (
  QUESTION,
  ScriptedJudge,
  claims_truth,
  precision_judgments,
  recall_judgments,
  search_span,
  span,
  truth,
)
from metrics import (
  SEARCH_SPAN_TYPE,
  AnswerCorrectness,
  FailureKind,
  GoldClaimsContextualRecall,
  PlanCorrectness,
  SearchContextualPrecision,
  classify_judge_error,
)

CLAIMS = (Claim(id='q1-f01', text='Dana approved the budget.'), Claim(id='q1-f02', text='It was approved in May.'))


def _document_in(request: LlmJudgeRequest) -> str:
  match = re.search(r'retrieved_id: (\S+)', request.user_prompt)
  assert match is not None
  return match.group(1)


def _rank_in(request: LlmJudgeRequest) -> int:
  match = re.search(r'Rank (\d+)', request.user_prompt)
  assert match is not None
  return int(match.group(1))


def _relevant_documents(*relevant: str) -> Any:
  """Per-document judge replies: each document judged relevant when listed, echoing its rank and id."""
  return lambda request: precision_judgments(
    (_rank_in(request), _document_in(request), _document_in(request) in relevant)
  )


class SearchContextualPrecisionTest(unittest.IsolatedAsyncioTestCase):
  async def test_scores_each_search_with_the_built_in_prompts_and_score(self) -> None:
    judge = ScriptedJudge(_relevant_documents('d1', 'd3'))
    metric = SearchContextualPrecision(judge_client=judge)
    target = search_span('search-1', ['d1', 'd2', 'd3'])

    result = await metric.compute(target, truth())

    self.assertEqual(result.status, MetricComputationStatus.COMPLETED)
    assert result.score is not None and result.metadata is not None
    self.assertAlmostEqual(result.score, (1 + 2 / 3) / 2)
    self.assertEqual(result.metadata['counts'], {'retrieved': 3, 'relevant': 2, 'echo_mismatches': 0})
    self.assertEqual([entry['retrieved_id'] for entry in result.metadata['rank_results']], ['d1', 'd2', 'd3'])
    self.assertEqual(result.metadata['judge_calls'], 3)
    built_in = ContextualPrecisionDocumentJudgeMetric(judge_client=judge)
    items = extract_retrieved_items(target, 'document')
    expected_prompts = {built_in.build_user_prompt(target, truth(), rank, item) for rank, item in enumerate(items, 1)}
    self.assertEqual({request.user_prompt for request in judge.requests}, expected_prompts)
    self.assertEqual({request.system_prompt for request in judge.requests}, {built_in.build_system_prompt()})

  def test_targets_search_spans_with_the_built_in_unit_rules(self) -> None:
    metric = SearchContextualPrecision(judge_client=ScriptedJudge())

    self.assertEqual(metric.target_span_types, (SEARCH_SPAN_TYPE,))
    self.assertTrue(metric.matches_span(search_span('s', ['d1'])))
    self.assertFalse(metric.matches_span(search_span('s', ['d1'], stage='retrieved')))
    self.assertIsNone(metric.input_skip_reason(search_span('s', ['d1'])))
    self.assertIn('not ranked', metric.input_skip_reason(search_span('s', ['d1'], ranked=False)) or '')
    self.assertIn('content', metric.input_skip_reason(search_span('s', ['d1'], with_content=False)) or '')

  async def test_decisions_are_keyed_to_the_input_documents_not_to_the_echoed_ids(self) -> None:
    judge = ScriptedJudge(precision_judgments((1, 'd1', True)), precision_judgments((2, 'other', False)))
    metric = SearchContextualPrecision(judge_client=judge)

    result = await metric.compute(search_span('s', ['d1', 'd2']), truth())

    assert result.metadata is not None and result.raw_output is not None
    self.assertEqual([entry['retrieved_id'] for entry in result.metadata['rank_results']], ['d1', 'd2'])
    self.assertEqual(result.metadata['counts']['echo_mismatches'], 1)
    self.assertEqual(result.raw_output['judgments'][1]['retrieved_id'], 'other')

  async def test_an_observed_empty_search_scores_zero_without_calling_the_judge(self) -> None:
    judge = ScriptedJudge()
    metric = SearchContextualPrecision(judge_client=judge)

    result = await metric.compute(search_span('s', []), truth())

    self.assertEqual(result.score, 0.0)
    self.assertEqual(judge.requests, [])

  async def test_a_wrong_judgment_count_fails_the_unit_as_misaligned(self) -> None:
    judge = ScriptedJudge(precision_judgments((1, 'd1', True)), precision_judgments((2, 'd2', True), (3, 'd3', True)))
    metric = SearchContextualPrecision(judge_client=judge)

    result = await metric.compute(search_span('s', ['d1', 'd2']), truth())

    self.assertEqual(result.status, MetricComputationStatus.FAILED)
    assert result.metadata is not None and result.raw_output is not None
    self.assertEqual(result.metadata['failure'], 'misaligned')
    self.assertEqual(result.metadata['expected_retrieved_ids'], ['d1', 'd2'])
    self.assertEqual(result.metadata['judge_calls'], 2)
    self.assertEqual(len(result.raw_output['outputs']), 2)
    self.assertIn('at rank 2', result.error_message or '')

  async def test_a_judge_error_fails_the_unit_with_its_failure_class(self) -> None:
    timeout = ExternalServiceError('gemini', 'judge', httpx.ReadTimeout('read timed out'))
    judge = ScriptedJudge(
      lambda request: timeout if _document_in(request) == 'd2' else precision_judgments((1, 'd1', True))
    )
    metric = SearchContextualPrecision(judge_client=judge)

    result = await metric.compute(search_span('s', ['d1', 'd2']), truth())

    self.assertEqual(result.status, MetricComputationStatus.FAILED)
    assert result.metadata is not None
    self.assertEqual(result.metadata['failure'], 'timeout')
    self.assertEqual(result.metadata['expected_retrieved_ids'], ['d1', 'd2'])
    self.assertIn('read timed out', result.error_message or '')
    # The call that completed before the failure keeps its usage.
    self.assertEqual(result.metadata['judge_calls'], 1)
    self.assertEqual(result.metadata['judge_usage'], {'input_tokens': 100, 'output_tokens': 7, 'total_tokens': 107})


class GoldClaimsContextualRecallTest(unittest.IsolatedAsyncioTestCase):
  async def test_judges_each_stored_claim_with_the_built_in_prompts(self) -> None:
    judge = ScriptedJudge(lambda request: recall_judgments(('Dana approved the budget', 'Dana' in request.user_prompt)))
    metric = GoldClaimsContextualRecall(judge_client=judge)
    target = search_span('search-1', ['d1', 'd2'])

    result = await metric.compute(target, claims_truth(*CLAIMS))

    assert result.score is not None and result.metadata is not None and result.raw_output is not None
    self.assertEqual(result.score, 0.5)
    self.assertEqual(
      [
        (entry['claim_id'], entry['statement'], entry['attributable']) for entry in result.metadata['statement_results']
      ],
      [('q1-f01', CLAIMS[0].text, True), ('q1-f02', CLAIMS[1].text, False)],
    )
    self.assertEqual(
      result.metadata['counts'],
      {'retrieved': 2, 'expected_statements': 2, 'attributable': 1, 'echo_mismatches': 1},
    )
    self.assertEqual(result.metadata['judge_calls'], 2)
    self.assertEqual(result.raw_output['judgments'][1]['statement'], 'Dana approved the budget')
    built_in = ContextualRecallDocumentJudgeMetric(judge_client=judge)
    items = extract_retrieved_items(target, 'document')
    self.assertEqual(
      [request.user_prompt for request in judge.requests],
      [built_in.build_user_prompt(target, items, claim.text) for claim in CLAIMS],
    )
    self.assertEqual({request.system_prompt for request in judge.requests}, {built_in.build_system_prompt()})

  def test_reads_the_gold_claims_of_each_search(self) -> None:
    metric = GoldClaimsContextualRecall(judge_client=ScriptedJudge())

    self.assertEqual(metric.ground_truth_key, CLAIMS_KEY)
    self.assertEqual(metric.target_span_types, (SEARCH_SPAN_TYPE,))

  async def test_nothing_retrieved_scores_zero_without_calling_the_judge(self) -> None:
    judge = ScriptedJudge()
    metric = GoldClaimsContextualRecall(judge_client=judge)

    result = await metric.compute(search_span('s', []), claims_truth(*CLAIMS))

    self.assertEqual(result.score, 0.0)
    assert result.metadata is not None
    self.assertEqual(result.metadata['counts']['expected_statements'], 2)
    self.assertEqual(judge.requests, [])

  async def test_malformed_claims_raise_instead_of_scoring(self) -> None:
    metric = GoldClaimsContextualRecall(judge_client=ScriptedJudge())

    with self.assertRaises(ValueError):
      await metric.compute(search_span('s', ['d1']), truth(key=CLAIMS_KEY, claims=[]))

  async def test_a_wrong_judgment_count_fails_the_unit_and_lists_its_claims(self) -> None:
    judge = ScriptedJudge(recall_judgments(), recall_judgments(('It was approved in May.', True)))
    metric = GoldClaimsContextualRecall(judge_client=judge)

    result = await metric.compute(search_span('s', ['d1']), claims_truth(*CLAIMS))

    self.assertEqual(result.status, MetricComputationStatus.FAILED)
    assert result.metadata is not None
    self.assertEqual(result.metadata['failure'], 'misaligned')
    self.assertEqual(result.metadata['expected_claim_ids'], ['q1-f01', 'q1-f02'])
    self.assertEqual(result.metadata['claims_key'], CLAIMS_KEY)

  async def test_a_judge_error_fails_the_unit_and_keeps_the_usage_of_completed_calls(self) -> None:
    timeout = ExternalServiceError('gemini', 'judge', httpx.ReadTimeout('read timed out'))
    judge = ScriptedJudge(
      lambda request: timeout if 'May' in request.user_prompt else recall_judgments((CLAIMS[0].text, True))
    )
    metric = GoldClaimsContextualRecall(judge_client=judge)

    result = await metric.compute(search_span('s', ['d1']), claims_truth(*CLAIMS))

    self.assertEqual(result.status, MetricComputationStatus.FAILED)
    assert result.metadata is not None
    self.assertEqual(result.metadata['failure'], 'timeout')
    self.assertEqual(result.metadata['judge_calls'], 1)


def _root(**semantics: Any) -> Any:
  steps = [ExecutionStep(id='1', operation='search', instruction='{"query": "budget"}', output='2 documents')]
  fields: dict[str, Any] = {
    'request': QUESTION,
    'answer': Answer(text='Dana approved it.'),
    'planning': PlanningData(executed_steps=steps),
  }
  return span('root', 'agent_root', parent=None, **{**fields, **semantics})


class MainPassAnswerAndPlanCorrectnessTest(unittest.IsolatedAsyncioTestCase):
  async def test_the_built_in_names_prompts_and_scores(self) -> None:
    plan_truth = truth(key='expected_plan', expected_plan=[{'operation': 'search', 'instruction': 'budget'}])
    for metric_class, built_in_class, ground_truth in (
      (AnswerCorrectness, AnswerCorrectnessJudgeMetric, truth()),
      (PlanCorrectness, PlanCorrectnessJudgeMetric, plan_truth),
    ):
      judge = ScriptedJudge({'score': 0.7, 'reasoning': 'Mostly correct.'})
      built_in = built_in_class(judge_client=judge, rubric_addition='Penalize missing dates.')
      with self.subTest(metric=metric_class.__name__):
        result = await metric_class(judge_client=judge, rubric_addition='Penalize missing dates.').compute(
          _root(), ground_truth
        )
        self.assertEqual(metric_class.metric_name, built_in_class.metric_name)
        self.assertEqual(judge.requests[0].system_prompt, built_in.build_system_prompt())
        self.assertEqual(judge.requests[0].user_prompt, built_in.build_user_prompt(_root(), ground_truth))
        self.assertEqual((result.score, result.reasoning), (0.7, 'Mostly correct.'))
        assert result.metadata is not None
        self.assertEqual(result.metadata['judge_calls'], 1)

  async def test_a_judge_failure_is_classified_where_the_built_in_raises(self) -> None:
    timeout = ExternalServiceError('gemini', 'judge', httpx.ReadTimeout('read timed out'))

    result = await AnswerCorrectness(judge_client=ScriptedJudge(timeout)).compute(_root(), truth())

    self.assertEqual(result.status, MetricComputationStatus.FAILED)
    assert result.metadata is not None
    self.assertEqual(result.metadata['failure'], 'timeout')
    with self.assertRaises(ExternalServiceError):
      await AnswerCorrectnessJudgeMetric(judge_client=ScriptedJudge(timeout)).compute(_root(), truth())

  async def test_skips_what_the_built_in_skips_without_calling_the_judge(self) -> None:
    judge = ScriptedJudge()

    result = await PlanCorrectness(judge_client=judge).compute(_root(planning=None), None)

    self.assertEqual(result.status, MetricComputationStatus.SKIPPED)
    self.assertEqual(result.error_message, 'Trace does not provide executed planning steps.')
    self.assertEqual(judge.requests, [])


class _GatewayTimeout(Exception):
  code = 504


class ClassifyJudgeErrorTest(unittest.TestCase):
  def test_failure_classes(self) -> None:
    cases: list[tuple[Exception, FailureKind]] = [
      (DataMappingError('gemini', 'invalid structured output'), FailureKind.INVALID_OUTPUT),
      (ExternalServiceError('gemini', 'judge', ContextOverflowError('too long')), FailureKind.CONTEXT_OVERFLOW),
      (ExternalServiceError('gemini', 'judge', httpx.ReadTimeout('slow')), FailureKind.TIMEOUT),
      (ExternalServiceError('gemini', 'judge', _GatewayTimeout('deadline exceeded')), FailureKind.TIMEOUT),
      (ExternalServiceError('gemini', 'judge', RuntimeError('500 internal')), FailureKind.PROVIDER),
    ]
    for error, kind in cases:
      with self.subTest(error=str(error)):
        self.assertEqual(classify_judge_error(error), kind)

  def test_follows_the_exception_chain(self) -> None:
    try:
      try:
        raise ContextOverflowError('too long')
      except ContextOverflowError as cause:
        raise ExternalServiceError('gemini', 'judge') from cause
    except ExternalServiceError as error:
      self.assertEqual(classify_judge_error(error), FailureKind.CONTEXT_OVERFLOW)


if __name__ == '__main__':
  unittest.main()
