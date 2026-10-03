import re
import unittest
from collections.abc import Sequence
from typing import Any

from langchain_core.exceptions import ContextOverflowError

from syllo_eval.evaluation.judge import LlmJudgeRequest
from syllo_eval.evaluation.metric_support.agent_outputs import extract_actual_plan, extract_final_answer
from syllo_eval.evaluation.metric_support.retrieved_context import RetrievedItem
from syllo_eval.evaluation.metrics.implementations.answer.correctness_judge import AnswerCorrectnessJudgeMetric
from syllo_eval.evaluation.metrics.implementations.plan.correctness_judge import PlanCorrectnessJudgeMetric
from syllo_eval.evaluation.metrics.implementations.rag.contextual_precision_judge import (
  ContextualPrecisionDocumentJudgeMetric,
)
from syllo_eval.evaluation.metrics.implementations.rag.contextual_recall_judge import (
  ContextualRecallDocumentJudgeMetric,
)
from syllo_eval.infrastructure.exceptions import DataMappingError, ExternalServiceError
from syllo_eval.model import MetricComputationStatus, Span
from syllo_eval.trace_semantics import Answer, ExecutionStep, PlanningData

from ablation_metrics import (
  AnswerCorrectnessWT,
  GoldClaimsContextualRecallSC,
  PlanCorrectnessWT,
  SearchContextualPrecisionSC,
  render_whole_trace,
)
from benchmarks.common import Claim
from metric_fakes import (
  QUESTION,
  ScriptedJudge,
  claims_truth,
  precision_judgments,
  recall_judgments,
  search_span,
  span,
  truth,
  without_judge_keys,
)
from metrics import GoldClaimsContextualRecall, SearchContextualPrecision

LIMIT = 65_536
CLAIMS = (Claim(id='q1-f01', text='Dana approved the budget.'), Claim(id='q1-f02', text='It was approved in May.'))
UNIT_SPANS = (
  search_span('s', ['d1', 'd2']),
  search_span('s', ['d1'], stage='retrieved'),
  search_span('s', ['d1'], ranked=False),
  search_span('s', ['d1'], with_content=False),
  search_span('s', []),
)


def _paragraphs(prompt: str) -> list[str]:
  return prompt.split('\n\n')


def _plan_items(metric: Any, spans: Sequence[Span]) -> list[tuple[bool, str | None]]:
  """What the planner reads from a metric for each candidate span: whether it is a target, and its skip reason."""
  return [(metric.matches_span(target), metric.input_skip_reason(target)) for target in spans]


class SearchContextualPrecisionSCTest(unittest.IsolatedAsyncioTestCase):
  def test_scores_the_same_units_as_the_main_pass(self) -> None:
    main = SearchContextualPrecision(judge_client=ScriptedJudge())
    ablation = SearchContextualPrecisionSC(judge_client=ScriptedJudge(), output_token_limit=LIMIT)

    self.assertEqual(ablation.target_span_types, main.target_span_types)
    self.assertEqual(ablation.ground_truth_key, main.ground_truth_key)
    self.assertEqual(_plan_items(ablation, UNIT_SPANS), _plan_items(main, UNIT_SPANS))

  async def test_one_call_gives_the_main_pass_result_for_the_same_decisions(self) -> None:
    decisions = [(1, 'd1', True), (2, 'd2', False), (3, 'd3', True)]
    target = search_span('s', ['d1', 'd2', 'd3'])
    main_judge = ScriptedJudge(*(precision_judgments(decision) for decision in decisions))
    # Out of rank order: single-call judgments are matched to documents by rank, not by position.
    ablation_judge = ScriptedJudge(precision_judgments(*reversed(decisions)))

    main = await SearchContextualPrecision(judge_client=main_judge).compute(target, truth())
    ablation = await SearchContextualPrecisionSC(judge_client=ablation_judge, output_token_limit=LIMIT).compute(
      target, truth()
    )

    self.assertEqual(len(main_judge.requests), 3)
    self.assertEqual(len(ablation_judge.requests), 1)
    self.assertEqual(ablation.score, main.score)
    self.assertEqual(without_judge_keys(ablation.metadata), without_judge_keys(main.metadata))
    self.assertEqual(ablation.raw_output, main.raw_output)
    assert ablation.metadata is not None
    self.assertEqual(ablation.metadata['judge_calls'], 1)

  async def test_budget_is_the_per_document_budget_capped_at_the_output_limit(self) -> None:
    for count, budget in ((3, 6_000), (80, LIMIT)):
      documents = [f'd{rank}' for rank in range(1, count + 1)]
      judge = ScriptedJudge(precision_judgments(*((rank, doc, False) for rank, doc in enumerate(documents, 1))))
      metric = SearchContextualPrecisionSC(judge_client=judge, output_token_limit=LIMIT)
      with self.subTest(count=count):
        result = await metric.compute(search_span('s', documents), truth())
        self.assertEqual(result.status, MetricComputationStatus.COMPLETED)
        self.assertEqual(judge.requests[0].max_output_tokens, budget)

  def test_prompts_keep_the_built_in_relevance_definition_and_rubric_addition(self) -> None:
    built_in = ContextualPrecisionDocumentJudgeMetric(judge_client=ScriptedJudge(), rubric_addition='Cite the policy.')
    ablation = SearchContextualPrecisionSC(
      judge_client=ScriptedJudge(), output_token_limit=LIMIT, rubric_addition='Cite the policy.'
    )
    target = search_span('s', ['d1', 'd2'])
    items = [RetrievedItem(item) for item in target.semantics.retrieval[0].items]

    built_in_system = _paragraphs(built_in.build_system_prompt())
    system = _paragraphs(ablation.build_single_call_system_prompt())
    definition = 'A document is relevant when it contains information that directly supports, verifies, or is '
    self.assertIn(definition, built_in_system[0])
    self.assertIn(definition, system[0])
    self.assertEqual(system[1:], built_in_system[1:])
    user = _paragraphs(ablation.build_single_call_user_prompt(target, truth(), items))
    built_in_user = _paragraphs(built_in.build_user_prompt(target, truth(), 1, items[0]))
    self.assertEqual(user[1:3], built_in_user[1:3])
    self.assertIn('Rank 1\nretrieved_id: d1', user[3])
    self.assertIn('Rank 2\nretrieved_id: d2', '\n\n'.join(user[3:]))

  async def test_a_missing_or_repeated_rank_fails_the_unit_and_keeps_the_response(self) -> None:
    judge = ScriptedJudge(precision_judgments((1, 'd1', True), (1, 'd2', False), (3, 'd3', True)))
    metric = SearchContextualPrecisionSC(judge_client=judge, output_token_limit=LIMIT)

    result = await metric.compute(search_span('s', ['d1', 'd2', 'd3']), truth())

    self.assertEqual(result.status, MetricComputationStatus.FAILED)
    assert result.metadata is not None and result.raw_output is not None
    self.assertEqual(result.metadata['failure'], 'misaligned')
    self.assertEqual(result.metadata['alignment'], {'missing_ranks': [2], 'unknown_ranks': [], 'repeated_ranks': [1]})
    self.assertEqual(result.metadata['expected_retrieved_ids'], ['d1', 'd2', 'd3'])
    self.assertEqual(result.metadata['max_output_tokens'], 6_000)
    self.assertEqual(result.metadata['judge_calls'], 1)
    self.assertEqual(result.metadata['judge_usage']['output_tokens'], judge.output_tokens)
    self.assertEqual(len(result.raw_output['judgments']), 3)

  async def test_a_short_output_that_used_its_whole_budget_is_truncated(self) -> None:
    judge = ScriptedJudge(precision_judgments((1, 'd1', True)), output_tokens=4_000)
    metric = SearchContextualPrecisionSC(judge_client=judge, output_token_limit=4_000)

    result = await metric.compute(search_span('s', ['d1', 'd2']), truth())

    assert result.metadata is not None
    self.assertEqual(result.metadata['failure'], 'truncated')

  async def test_an_unparsable_output_fails_the_unit_with_its_class(self) -> None:
    judge = ScriptedJudge(DataMappingError('gemini', 'invalid structured output'))
    metric = SearchContextualPrecisionSC(judge_client=judge, output_token_limit=LIMIT)

    result = await metric.compute(search_span('s', ['d1', 'd2']), truth())

    self.assertEqual(result.status, MetricComputationStatus.FAILED)
    assert result.metadata is not None
    self.assertEqual(result.metadata['failure'], 'invalid_output')
    self.assertEqual(result.metadata['expected_retrieved_ids'], ['d1', 'd2'])
    self.assertNotIn('judge_calls', result.metadata)

  async def test_an_observed_empty_search_scores_zero_without_calling_the_judge(self) -> None:
    judge = ScriptedJudge()
    metric = SearchContextualPrecisionSC(judge_client=judge, output_token_limit=LIMIT)

    result = await metric.compute(search_span('s', []), truth())

    self.assertEqual(result.score, 0.0)
    self.assertEqual(judge.requests, [])


class GoldClaimsContextualRecallSCTest(unittest.IsolatedAsyncioTestCase):
  def test_scores_the_same_units_and_claims_as_the_main_pass(self) -> None:
    main = GoldClaimsContextualRecall(judge_client=ScriptedJudge())
    ablation = GoldClaimsContextualRecallSC(judge_client=ScriptedJudge(), output_token_limit=LIMIT)

    self.assertEqual(ablation.target_span_types, main.target_span_types)
    self.assertEqual(ablation.ground_truth_key, main.ground_truth_key)
    self.assertEqual(_plan_items(ablation, UNIT_SPANS), _plan_items(main, UNIT_SPANS))

  async def test_one_call_gives_the_main_pass_result_for_the_same_decisions(self) -> None:
    target = search_span('s', ['d1', 'd2'])
    main_judge = ScriptedJudge(
      recall_judgments((CLAIMS[0].text, True)), recall_judgments(('It was approved in March.', False))
    )
    ablation_judge = ScriptedJudge(
      recall_judgments(('1. Dana approved the budget', True), ('It was approved in March.', False))
    )

    main = await GoldClaimsContextualRecall(judge_client=main_judge).compute(target, claims_truth(*CLAIMS))
    ablation = await GoldClaimsContextualRecallSC(judge_client=ablation_judge, output_token_limit=LIMIT).compute(
      target, claims_truth(*CLAIMS)
    )

    self.assertEqual(len(main_judge.requests), 2)
    self.assertEqual(len(ablation_judge.requests), 1)
    self.assertEqual(ablation.score, main.score)
    # The list number and the missing period are no echo mismatch; the rewritten month is one, in both arms.
    self.assertEqual(without_judge_keys(ablation.metadata), without_judge_keys(main.metadata))
    assert ablation.metadata is not None
    self.assertEqual(ablation.metadata['counts']['echo_mismatches'], 1)
    self.assertEqual([entry['claim_id'] for entry in ablation.metadata['statement_results']], ['q1-f01', 'q1-f02'])

  def test_prompts_keep_the_built_in_attribution_definition_and_number_the_claims(self) -> None:
    built_in = ContextualRecallDocumentJudgeMetric(judge_client=ScriptedJudge())
    ablation = GoldClaimsContextualRecallSC(judge_client=ScriptedJudge(), output_token_limit=LIMIT)
    items = [RetrievedItem(item) for item in search_span('s', ['d1']).semantics.retrieval[0].items]

    definition = 'A statement is attributable when the retrieved documents contain enough information to directly '
    self.assertIn(definition, built_in.build_system_prompt())
    self.assertIn(definition, ablation.build_single_call_system_prompt())
    user = _paragraphs(ablation.build_single_call_user_prompt(items, list(CLAIMS)))
    built_in_user = _paragraphs(built_in.build_user_prompt(search_span('s', ['d1']), items, CLAIMS[0].text))
    self.assertEqual(user[1], built_in_user[1])
    self.assertEqual(user[2], 'Statements:\n1. Dana approved the budget.\n2. It was approved in May.')

  async def test_a_wrong_judgment_count_fails_the_unit_and_lists_its_claims(self) -> None:
    judge = ScriptedJudge(recall_judgments((CLAIMS[0].text, True)))
    metric = GoldClaimsContextualRecallSC(judge_client=judge, output_token_limit=LIMIT)

    result = await metric.compute(search_span('s', ['d1']), claims_truth(*CLAIMS))

    self.assertEqual(result.status, MetricComputationStatus.FAILED)
    assert result.metadata is not None and result.raw_output is not None
    self.assertEqual(result.metadata['failure'], 'misaligned')
    self.assertEqual(result.metadata['expected_claim_ids'], ['q1-f01', 'q1-f02'])
    self.assertEqual(result.metadata['max_output_tokens'], 4_000)
    self.assertEqual(result.metadata['judge_calls'], 1)
    self.assertEqual(len(result.raw_output['judgments']), 1)


def _trace(trace_id: str = 'trace-1', prefix: str = '') -> list[Span]:
  """An agent run (root, LLM call, search, nested tool) plus a span outside the agent, out of order."""
  steps = [ExecutionStep(id='1', operation='search', instruction='{"query": "budget"}', output='2 documents')]

  def ident(name: str) -> str:
    return f'{prefix}{name}'

  root = span(
    ident('root'),
    'agent_root',
    parent=None,
    name='ReAct agent',
    trace_id=trace_id,
    input_data={'question': QUESTION},
    output_data={'answer': 'Dana did.'},
    request=QUESTION,
    answer=Answer(text='Dana approved it, in May.'),
    planning=PlanningData(executed_steps=steps),
  )
  search = search_span(ident('search'), ['d1', 'd2'], offset=2).model_copy(
    update={'parent_span_id': ident('root'), 'trace_id': trace_id}
  )
  llm = span(
    ident('llm'),
    'llm',
    parent=ident('root'),
    offset=1,
    trace_id=trace_id,
    input_data={'messages': [{'role': 'user', 'content': 'Qui a approuvé le budget ?'}]},
    output_data='Search the knowledge base.',
  )
  tool = span(
    ident('tool'), 'tool', parent=ident('llm'), offset=1, trace_id=trace_id, input_data='plan', output_data=None
  )
  grader = span(ident('grader'), 'llm', parent=None, offset=9, trace_id=trace_id, output_data='reward 1')
  return [grader, search, tool, root, llm]


class RenderWholeTraceTest(unittest.TestCase):
  def test_renders_the_target_subtree_depth_first_with_ordinal_ids(self) -> None:
    rendered, span_count = render_whole_trace(_trace(), 'root')

    self.assertEqual(span_count, 4)
    tags = re.findall(r'<span [^>]*>', rendered)
    self.assertEqual(
      tags,
      [
        '<span id="1" type="agent_root" name="ReAct agent" status="success">',
        '<span id="2" parent="1" type="llm" name="llm" status="success">',
        '<span id="3" parent="2" type="tool" name="tool" status="success">',
        '<span id="4" parent="1" type="retrieval" name="search" status="success">',
      ],
    )
    self.assertNotIn('reward 1', rendered)

  def test_typed_observations_appear_as_the_target_metrics_render_them(self) -> None:
    spans = _trace()
    root = next(item for item in spans if item.external_id == 'root')
    rendered, _ = render_whole_trace(spans, 'root')

    self.assertTrue(
      rendered.startswith(
        f'<span id="1" type="agent_root" name="ReAct agent" status="success">\nrequest:\n{QUESTION}\n'
      )
    )
    self.assertTrue(
      rendered.endswith(f'executed steps:\n{extract_actual_plan(root)}\nanswer:\n{extract_final_answer(root)}\n</span>')
    )
    first_document = RetrievedItem(spans[1].semantics.retrieval[0].items[0]).render_for_prompt(1)
    self.assertIn(f'retrieved documents (selected):\n{first_document}', rendered)
    # Raw input and output only where a span has no typed content, with text and non-ASCII kept as they are.
    self.assertIn('input:\n{"messages": [{"role": "user", "content": "Qui a approuvé le budget ?"}]}', rendered)
    self.assertIn('output:\nSearch the knowledge base.', rendered)
    self.assertNotIn('"answer": "Dana did."', rendered)
    self.assertNotIn('budget approval', rendered)

  def test_a_re_identified_copy_renders_the_same(self) -> None:
    self.assertEqual(
      render_whole_trace(_trace(), 'root'), render_whole_trace(_trace('trace-2', prefix='copy-'), 'copy-root')
    )

  def test_a_target_outside_the_trace_is_an_error(self) -> None:
    with self.assertRaises(ValueError):
      render_whole_trace(_trace(), 'missing')


class _SpanStore:
  def __init__(self, spans: Sequence[Span] | Exception):
    self.spans = spans
    self.loaded: list[str] = []

  async def load(self, trace_id: str) -> Sequence[Span]:
    self.loaded.append(trace_id)
    if isinstance(self.spans, Exception):
      raise self.spans
    return self.spans


def _score_reply(request: LlmJudgeRequest) -> dict[str, Any]:
  return {'score': 0.7, 'reasoning': 'Mostly correct.'}


class AnswerCorrectnessWTTest(unittest.IsolatedAsyncioTestCase):
  def test_system_prompt_is_the_built_in_one(self) -> None:
    for rubric_addition in (None, 'Penalize missing dates.'):
      built_in = AnswerCorrectnessJudgeMetric(judge_client=ScriptedJudge(), rubric_addition=rubric_addition)
      ablation = AnswerCorrectnessWT(
        judge_client=ScriptedJudge(), load_spans=_SpanStore([]).load, rubric_addition=rubric_addition
      )
      with self.subTest(rubric_addition=rubric_addition):
        self.assertEqual(ablation.build_system_prompt(), built_in.build_system_prompt())
        self.assertEqual(ablation.metric_name, 'answer_correctness_judge_wt')

  def test_user_prompt_replaces_only_the_answer_paragraph(self) -> None:
    root = _trace()[3]
    built_in = _paragraphs(AnswerCorrectnessJudgeMetric(judge_client=ScriptedJudge()).build_user_prompt(root, truth()))
    ablation = AnswerCorrectnessWT(judge_client=ScriptedJudge(), load_spans=_SpanStore([]).load)

    prompt = _paragraphs(ablation.build_whole_trace_user_prompt(root, truth(), 'TRACE'))

    self.assertEqual(built_in[3], f'Actual answer:\n{extract_final_answer(root)}')
    self.assertEqual(
      prompt,
      [
        f'{built_in[0]} The actual answer is the answer of the agent_root span in the agent trace.',
        built_in[1],
        built_in[2],
        'Agent trace:\nTRACE',
      ],
    )

  async def test_judges_the_rendered_trace_and_records_its_size(self) -> None:
    store = _SpanStore(_trace())
    judge = ScriptedJudge(_score_reply)
    metric = AnswerCorrectnessWT(judge_client=judge, load_spans=store.load)
    root = _trace()[3]

    result = await metric.compute(root, truth())

    rendered, span_count = render_whole_trace(_trace(), 'root')
    self.assertEqual(store.loaded, ['trace-1'])
    self.assertEqual(result.score, 0.7)
    self.assertEqual(result.reasoning, 'Mostly correct.')
    assert result.metadata is not None
    self.assertEqual(result.metadata['trace_render'], {'spans': span_count, 'chars': len(rendered)})
    self.assertEqual(result.metadata['judge_calls'], 1)
    self.assertIn(f'Agent trace:\n{rendered}', judge.requests[0].user_prompt)
    self.assertIsNone(judge.requests[0].max_output_tokens)

  async def test_a_unit_the_built_in_skips_is_skipped_without_loading_the_trace(self) -> None:
    store = _SpanStore(_trace())
    judge = ScriptedJudge()
    root = _trace()[3]
    root.semantics.answer = None

    result = await AnswerCorrectnessWT(judge_client=judge, load_spans=store.load).compute(root, truth())

    self.assertEqual(result.status, MetricComputationStatus.SKIPPED)
    self.assertEqual(result.error_message, AnswerCorrectnessJudgeMetric(judge_client=judge).input_skip_reason(root))
    self.assertEqual((store.loaded, judge.requests), ([], []))

  async def test_load_and_judge_failures_fail_the_unit_with_their_class(self) -> None:
    root = _trace()[3]
    load_failure = await AnswerCorrectnessWT(
      judge_client=ScriptedJudge(), load_spans=_SpanStore(ConnectionError('database unreachable')).load
    ).compute(root, truth())
    overflow = ExternalServiceError('gemini', 'judge', ContextOverflowError('input exceeds the maximum'))
    judge_failure = await AnswerCorrectnessWT(
      judge_client=ScriptedJudge(overflow), load_spans=_SpanStore(_trace()).load
    ).compute(root, truth())

    self.assertEqual((load_failure.status, judge_failure.status), (MetricComputationStatus.FAILED,) * 2)
    assert load_failure.metadata is not None and judge_failure.metadata is not None
    self.assertEqual(load_failure.metadata['failure'], 'trace_load')
    self.assertEqual(judge_failure.metadata['failure'], 'context_overflow')
    self.assertEqual(judge_failure.metadata['trace_render']['spans'], 4)


class PlanCorrectnessWTTest(unittest.IsolatedAsyncioTestCase):
  def test_user_prompt_replaces_only_the_plan_paragraph(self) -> None:
    root = _trace()[3]
    plan_truth = truth(key='expected_plan', expected_plan=[{'operation': 'search', 'instruction': '{"query": "x"}'}])
    built_in = _paragraphs(PlanCorrectnessJudgeMetric(judge_client=ScriptedJudge()).build_user_prompt(root, plan_truth))
    ablation = PlanCorrectnessWT(judge_client=ScriptedJudge(), load_spans=_SpanStore([]).load)

    prompt = _paragraphs(ablation.build_whole_trace_user_prompt(root, plan_truth, 'TRACE'))

    self.assertEqual(built_in[2], f'Actual plan (ordered steps the agent took):\n{extract_actual_plan(root)}')
    self.assertEqual(
      prompt,
      [
        f'{built_in[0]} The actual plan is the executed steps of the agent_root span in the agent trace.',
        built_in[1],
        'Agent trace:\nTRACE',
        built_in[3],
      ],
    )

  async def test_runs_without_an_expected_plan_like_the_built_in(self) -> None:
    judge = ScriptedJudge(_score_reply)
    metric = PlanCorrectnessWT(judge_client=judge, load_spans=_SpanStore(_trace()).load)

    result = await metric.compute(_trace()[3], None)

    self.assertFalse(metric.requires_ground_truth)
    self.assertEqual(result.score, 0.7)
    self.assertNotIn('Expected plan', judge.requests[0].user_prompt)
    self.assertEqual(
      judge.requests[0].system_prompt, PlanCorrectnessJudgeMetric(judge_client=judge).build_system_prompt()
    )
    self.assertIn('executed steps:\nStep 1: operation=search', judge.requests[0].user_prompt)


if __name__ == '__main__':
  unittest.main()
