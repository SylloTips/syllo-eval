"""The two ablations of Section 5.1. Each removes one design choice of Section 4 from the metric it is compared with.

- Syllo-eval-SC (single call) judges all the documents, or all the claims, of a search unit in one call instead of one
  call each. It subclasses its main-pass metric in ``metrics.py`` and replaces only the judging step.
- Syllo-eval-WT (whole trace) reads the agent's whole trace instead of the observations of its target span. It
  subclasses the built-in Answer or Plan Correctness and replaces only the paragraph holding the answer or the plan.

Units, skip rules, rubrics, scoring and result metadata therefore stay those of the main pass. The prompts that change
live in ``prompts/``: the built-in v1 wording, edited only where the design choice requires it.
"""

import json
from abc import ABC, abstractmethod
from collections import Counter, defaultdict
from collections.abc import Awaitable, Callable, Sequence
from functools import cache
from string import Template
from typing import Any

from pydantic import JsonValue

from syllo_eval.evaluation.judge import LlmJudgeClient, LlmJudgeRequest, judge_metadata
from syllo_eval.evaluation.judge.metric_base import BaseLlmJudgeMetric
from syllo_eval.evaluation.metric_support.agent_outputs import (
  display_text,
  extract_actual_plan,
  extract_final_answer,
  ground_truth_text,
  render_plan,
)
from syllo_eval.evaluation.metric_support.retrieved_context import RetrievedItem
from syllo_eval.evaluation.metrics.contracts import MetricComputationResult
from syllo_eval.evaluation.metrics.implementations.answer.correctness_judge import AnswerCorrectnessJudgeMetric
from syllo_eval.evaluation.metrics.implementations.plan.correctness_judge import PlanCorrectnessJudgeMetric
from syllo_eval.evaluation.metrics.implementations.rag.contextual_precision_judge import (
  ContextualPrecisionJudgePayload,
  ContextualPrecisionJudgment,
)
from syllo_eval.evaluation.metrics.implementations.rag.contextual_recall_judge import (
  ContextualRecallJudgePayload,
  ContextualRecallJudgment,
)
from syllo_eval.infrastructure import DatabaseManager
from syllo_eval.infrastructure.unit_of_work import UnitOfWork
from syllo_eval.model import GroundTruth, MetricComputationStatus, Span
from syllo_eval.trace_semantics import RetrievalResult

from benchmarks.common import Claim
from config import EXPERIMENTS_DIR
from metrics import (
  FailureKind,
  GoldClaimsContextualRecall,
  JudgedUnit,
  JudgeFailure,
  SearchContextualPrecision,
  StoredClaimsContextualRecall,
  count_failure,
  failed_result,
  judge_each,
)

PROMPTS_DIR = EXPERIMENTS_DIR / 'prompts'


@cache
def _paragraphs(name: str) -> tuple[Template, ...]:
  text = (PROMPTS_DIR / name).read_text(encoding='utf-8')
  return tuple(Template(paragraph) for paragraph in text.strip('\n').split('\n\n'))


def render_ablation_prompt(name: str, **values: str | None) -> str:
  """Render ``prompts/<name>`` as the library renders its templates: a paragraph using a None value is omitted."""
  return '\n\n'.join(
    paragraph.substitute(values)
    for paragraph in _paragraphs(name)
    if all(values[identifier] is not None for identifier in paragraph.get_identifiers())
  )


def _rendered_items(items: Sequence[RetrievedItem]) -> str:
  return '\n\n'.join(item.render_for_prompt(rank) for rank, item in enumerate(items, start=1))


class SearchContextualPrecisionSC(SearchContextualPrecision):
  """Syllo-eval-SC for contextual precision: every document of the search judged in one call.

  The output budget is the main pass's per-document budget times the number of documents, capped at the judge's
  output limit. Judgments are matched to documents by rank; a wrong count or rank fails the unit.
  """

  metric_name = 'contextual_precision_search_sc'
  metric_description = 'Syllo-eval-SC: contextual precision of one search call, judging all its documents in one call.'

  def __init__(self, *, judge_client: LlmJudgeClient, output_token_limit: int, rubric_addition: str | None = None):
    super().__init__(judge_client=judge_client, rubric_addition=rubric_addition)
    self._output_token_limit = output_token_limit

  async def judge_items(
    self, span: Span, ground_truth: GroundTruth, items: list[RetrievedItem]
  ) -> JudgedUnit[ContextualPrecisionJudgment] | MetricComputationResult:
    budget = min(self.max_output_tokens * len(items), self._output_token_limit)
    responses, failure = await judge_each(
      [
        self._judge_client.judge(
          LlmJudgeRequest(
            system_prompt=self.build_single_call_system_prompt(),
            user_prompt=self.build_single_call_user_prompt(span, ground_truth, items),
            response_model=ContextualPrecisionJudgePayload,
            temperature=0.0,
            max_output_tokens=budget,
          )
        )
      ]
    )
    if failure is not None:
      return self.unit_failure(failure, items, responses, max_output_tokens=budget)
    judgments = responses[0].output.judgments
    ranks = [judgment.rank for judgment in judgments]
    expected_ranks = range(1, len(items) + 1)
    if sorted(ranks) != list(expected_ranks):
      alignment = {
        'missing_ranks': sorted(set(expected_ranks) - set(ranks)),
        'unknown_ranks': sorted(set(ranks) - set(expected_ranks)),
        'repeated_ranks': sorted(rank for rank, count in Counter(ranks).items() if count > 1),
      }
      message = (
        f'Judge returned {len(judgments)} contextual precision judgments for {len(items)} ranked '
        f'{self.variant}s: {alignment}.'
      )
      return self.unit_failure(
        JudgeFailure(count_failure([len(judgments)], len(items), responses, budget), message),
        items,
        responses,
        raw_output=responses[0].output.model_dump(),
        max_output_tokens=budget,
        alignment=alignment,
      )
    by_rank = {judgment.rank: judgment for judgment in judgments}
    return JudgedUnit(responses, [by_rank[rank] for rank in expected_ranks])

  def build_single_call_system_prompt(self) -> str:
    return render_ablation_prompt(
      'contextual_precision_sc_system.md', variant=self.variant, rubric_addition=self._rubric_addition
    )

  def build_single_call_user_prompt(self, span: Span, ground_truth: GroundTruth, items: list[RetrievedItem]) -> str:
    return render_ablation_prompt(
      'contextual_precision_sc_user.md',
      variant=self.variant,
      request=span.semantics.request or '<empty>',
      expected_answer=ground_truth.ground_truth_value['expected_output'],
      items=_rendered_items(items),
    )


class StoredClaimsContextualRecallSC(StoredClaimsContextualRecall):
  """Syllo-eval-SC for contextual recall: every claim judged against the search's documents in one call.

  The output budget is the main pass's per-claim budget times the number of claims, capped at the judge's output
  limit. Judgments are matched to claims by position; a wrong count fails the unit.
  """

  def __init__(self, *, judge_client: LlmJudgeClient, output_token_limit: int, rubric_addition: str | None = None):
    super().__init__(judge_client=judge_client, rubric_addition=rubric_addition)
    self._output_token_limit = output_token_limit

  async def judge_claims(
    self, span: Span, items: list[RetrievedItem], claims: list[Claim]
  ) -> JudgedUnit[ContextualRecallJudgment] | MetricComputationResult:
    budget = min(self.max_output_tokens * len(claims), self._output_token_limit)
    responses, failure = await judge_each(
      [
        self._judge_client.judge(
          LlmJudgeRequest(
            system_prompt=self.build_single_call_system_prompt(),
            user_prompt=self.build_single_call_user_prompt(items, claims),
            response_model=ContextualRecallJudgePayload,
            temperature=0.0,
            max_output_tokens=budget,
          )
        )
      ]
    )
    if failure is not None:
      return self.unit_failure(failure, items, claims, responses, max_output_tokens=budget)
    judgments = responses[0].output.judgments
    if len(judgments) != len(claims):
      return self.unit_failure(
        JudgeFailure(
          count_failure([len(judgments)], len(claims), responses, budget),
          f'Judge returned {len(judgments)} contextual recall judgments for {len(claims)} statements.',
        ),
        items,
        claims,
        responses,
        raw_output=responses[0].output.model_dump(),
        max_output_tokens=budget,
      )
    return JudgedUnit(responses, list(judgments))

  def build_single_call_system_prompt(self) -> str:
    return render_ablation_prompt(
      'contextual_recall_sc_system.md', variant=self.variant, rubric_addition=self._rubric_addition
    )

  def build_single_call_user_prompt(self, items: list[RetrievedItem], claims: list[Claim]) -> str:
    return render_ablation_prompt(
      'contextual_recall_sc_user.md',
      variant=self.variant,
      items=_rendered_items(items),
      claims='\n'.join(f'{index}. {claim.text}' for index, claim in enumerate(claims, start=1)),
    )


class GoldClaimsContextualRecallSC(StoredClaimsContextualRecallSC):
  """Syllo-eval-SC for contextual recall over the ERB gold claims."""

  claims_key = GoldClaimsContextualRecall.claims_key
  metric_name = 'contextual_recall_gold_claims_sc'
  metric_description = (
    'Syllo-eval-SC: contextual recall of one search call over the ERB gold claims, judging all claims in one call.'
  )


SpanLoader = Callable[[str], Awaitable[Sequence[Span]]]


def stored_span_loader(db_manager: DatabaseManager) -> SpanLoader:
  """Load a stored trace through the caller's database pool, which stays owned by the caller."""

  async def load_spans(trace_id: str) -> Sequence[Span]:
    async with UnitOfWork(db_manager) as uow:
      return await uow.spans.list_by_trace(trace_id)

  return load_spans


def render_whole_trace(spans: Sequence[Span], target_span_id: str) -> tuple[str, int]:
  """The canonical trace a whole-trace judge reads, and how many spans it holds: the target span and its descendants.

  Spans appear depth-first, children by start time, with ordinal ids instead of source ids so that a re-identified
  copy renders the same. Typed observations are rendered with the helpers the target metrics use, so the target's
  request, answer and executed steps appear verbatim. A span's request comes before its children; its retrieval
  results, plans, executed steps and answer come after them. A span without typed content shows its raw input and
  output instead.
  """
  spans_by_id = {span.external_id: span for span in spans}
  if target_span_id not in spans_by_id:
    raise ValueError(f'Span {target_span_id} is not in its stored trace')
  children: dict[str, list[Span]] = defaultdict(list)
  for span in spans:
    if span.parent_span_id is not None:
      children[span.parent_span_id].append(span)
  lines: list[str] = []
  span_count = 0

  def visit(span: Span, parent: int | None) -> None:
    nonlocal span_count
    span_count += 1
    ordinal = span_count
    parent_attribute = '' if parent is None else f' parent="{parent}"'
    lines.append(
      f'<span id="{ordinal}"{parent_attribute} type={_quoted(span.span_type)} name={_quoted(span.name)} '
      f'status="{span.status}">'
    )
    before_children, after_children = _span_sections(span)
    lines.extend(before_children)
    for child in sorted(children[span.external_id], key=lambda child: (child.start_time, child.external_id)):
      visit(child, ordinal)
    lines.extend(after_children)
    lines.append('</span>')

  visit(spans_by_id[target_span_id], None)
  return '\n'.join(lines), span_count


def _span_sections(span: Span) -> tuple[list[str], list[str]]:
  semantics = span.semantics
  planning = semantics.planning
  if semantics.request is None and semantics.answer is None and not semantics.retrieval and planning is None:
    return _section('input', _raw_text(span.input_data)), _section('output', _raw_text(span.output_data))
  before_children = _section('request', display_text(semantics.request)) if semantics.request is not None else []
  after_children = [line for result in semantics.retrieval for line in _retrieval_section(result)]
  if planning is not None:
    for plan in planning.plans:
      supersedes = f' (supersedes {plan.supersedes_plan_id})' if plan.supersedes_plan_id is not None else ''
      after_children += _section(f'plan {plan.id}{supersedes}', render_plan([step.model_dump() for step in plan.steps]))
    if planning.executed_steps is not None:
      after_children += _section('executed steps', extract_actual_plan(span))
  if semantics.answer is not None:
    after_children += _section('answer', extract_final_answer(span))
  return before_children, after_children


def _retrieval_section(result: RetrievalResult) -> list[str]:
  label = f'retrieved {result.kind}s ({result.stage}{"" if result.ranked else ", unranked"})'
  if result.query is not None:
    label += f' for query {_quoted(result.query)}'
  if result.availability != 'available':
    return [f'{label}: {result.availability}' + (f' ({result.reason})' if result.reason else '')]
  if not result.items:
    return [f'{label}: none']
  return [f'{label}:', _rendered_items([RetrievedItem(item) for item in result.items])]


def _section(label: str, text: str | None) -> list[str]:
  return [] if text is None else [f'{label}:', text]


def _raw_text(value: JsonValue) -> str | None:
  if value is None:
    return None
  return value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)


def _quoted(value: str) -> str:
  return json.dumps(value, ensure_ascii=False)


class _WholeTraceJudgeMetric(BaseLlmJudgeMetric, ABC):
  """Syllo-eval-WT: the judge reads the agent's whole canonical trace in place of its target span's observations.

  The system prompt is the built-in one. The user prompt is the built-in one with the trace in place of the answer or
  plan paragraph, plus one sentence saying where in the trace the answer or plan is.
  """

  def __init__(self, *, judge_client: LlmJudgeClient, load_spans: SpanLoader, rubric_addition: str | None = None):
    super().__init__(judge_client=judge_client, rubric_addition=rubric_addition)
    self._load_spans = load_spans

  @abstractmethod
  def build_whole_trace_user_prompt(self, span: Span, ground_truth: GroundTruth | None, trace: str) -> str:
    """The built-in user prompt, with ``trace`` in place of the target's observations."""

  async def compute(self, span: Span, ground_truth: GroundTruth | None) -> MetricComputationResult:
    # The built-in skip rule, so that both arms score the same units.
    if reason := self.input_skip_reason(span):
      return MetricComputationResult(score=None, status=MetricComputationStatus.SKIPPED, error_message=reason)
    try:
      trace, span_count = render_whole_trace(await self._load_spans(span.trace_id), span.external_id)
    except Exception as error:
      return failed_result(self.name, JudgeFailure(FailureKind.TRACE_LOAD, str(error)), {})
    # Recorded for every unit, failed ones included, so that trace length never depends on the judge succeeding.
    trace_render: dict[str, Any] = {'spans': span_count, 'chars': len(trace)}
    responses, failure = await judge_each(
      [
        self._judge_client.judge(
          LlmJudgeRequest(
            system_prompt=self.build_system_prompt(),
            user_prompt=self.build_whole_trace_user_prompt(span, ground_truth, trace),
            response_model=self.response_model,
            temperature=self.temperature,
            max_output_tokens=self.max_output_tokens,
          )
        )
      ]
    )
    if failure is not None:
      return failed_result(self.name, failure, {'trace_render': trace_render}, responses)
    payload = responses[0].output
    metadata = {**(payload.metadata or {}), **judge_metadata(responses), 'trace_render': trace_render}
    return MetricComputationResult(score=payload.score, reasoning=payload.reasoning, metadata=metadata)


class AnswerCorrectnessWT(_WholeTraceJudgeMetric, AnswerCorrectnessJudgeMetric):
  """Syllo-eval-WT for Answer Correctness."""

  metric_name = 'answer_correctness_judge_wt'
  metric_description = 'Syllo-eval-WT: answer correctness judged from the whole canonical trace, not the final answer.'

  def build_whole_trace_user_prompt(self, span: Span, ground_truth: GroundTruth | None, trace: str) -> str:
    return render_ablation_prompt(
      'answer_correctness_wt_user.md',
      expected_answer=self._extract_expected_answer(ground_truth),
      request=display_text(span.semantics.request),
      trace=trace,
      rubric=ground_truth_text(ground_truth, 'rubric'),
      notes=ground_truth_text(ground_truth, 'notes'),
    )


class PlanCorrectnessWT(_WholeTraceJudgeMetric, PlanCorrectnessJudgeMetric):
  """Syllo-eval-WT for Plan Correctness."""

  metric_name = 'plan_correctness_judge_wt'
  metric_description = 'Syllo-eval-WT: plan correctness judged from the whole canonical trace, not the executed steps.'

  def build_whole_trace_user_prompt(self, span: Span, ground_truth: GroundTruth | None, trace: str) -> str:
    return render_ablation_prompt(
      'plan_correctness_wt_user.md',
      request=display_text(span.semantics.request),
      trace=trace,
      expected_plan=ground_truth_text(ground_truth, 'expected_plan'),
      rubric=ground_truth_text(ground_truth, 'rubric'),
      notes=ground_truth_text(ground_truth, 'notes'),
    )
