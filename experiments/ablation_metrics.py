"""The experiments' metrics that the library does not provide as they are: the two ablations of Section 5.1, and
recall over the ERB gold claims.

The main pass runs the built-in metrics: contextual precision on search spans (``SEARCH_SPAN_TYPE``), and Answer and
Plan Correctness. ``GoldClaimsContextualRecall`` is the built-in stored-claims recall, reading the claims key of the
benchmark import. Each ablation subclasses the metric it is compared with and overrides one library hook:

- Syllo-eval-SC (single call) replaces ``judge_items`` or ``judge_claims``: all the documents, or all the claims, of a
  search unit are judged in one call instead of one call each.
- Syllo-eval-WT (whole trace) replaces ``judge_input``: the user prompt holds the agent's whole canonical trace in
  place of the paragraph with the answer or plan, and adds one sentence saying where in the trace that is.

Units, skip rules, rubrics, scoring and result metadata therefore stay those of the compared metric; an ablation only
adds metadata fields (``trace_render`` for WT; the output budget, and for CP the rank alignment, of a failed SC unit).
The prompts that change live in ``prompts/``: the built-in v1 wording (v2 for Plan Correctness), edited
only where the design choice requires it.
"""

import json
from abc import ABC, abstractmethod
from collections import Counter, defaultdict
from collections.abc import Awaitable, Callable, Mapping, Sequence
from functools import cache
from string import Template
from typing import Any

from pydantic import JsonValue

from syllo_eval.evaluation.judge import LlmJudgeClient, LlmJudgeRequest, LlmJudgeResponse, judge_metadata
from syllo_eval.evaluation.judge.batch import judge_batch
from syllo_eval.evaluation.judge.failures import mismatch_kind
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
  ContextualPrecisionDocumentJudgeMetric,
  ContextualPrecisionJudgePayload,
  ContextualPrecisionJudgment,
)
from syllo_eval.evaluation.metrics.implementations.rag.contextual_recall_judge import (
  ContextualRecallJudgePayload,
  ContextualRecallJudgment,
)
from syllo_eval.evaluation.metrics.implementations.rag.contextual_recall_stored_claims import (
  ContextualRecallDocumentStoredClaimsMetric,
)
from syllo_eval.infrastructure import DatabaseManager
from syllo_eval.infrastructure.unit_of_work import UnitOfWork
from syllo_eval.model import GroundTruth, MetricComputationStatus, Span
from syllo_eval.trace_semantics import RetrievalResult

from benchmarks.erb import CLAIMS_KEY as ERB_GOLD_CLAIMS_KEY
from config import EXPERIMENTS_DIR

# Span type the agent adapters give each search call: its semantics carry the user question as the request, and the
# documents the call returned as one ranked document result at the `selected` stage. Retrieval metrics target it.
SEARCH_SPAN_TYPE = 'retrieval'
# Failure class of a whole-trace unit whose stored trace could not be loaded or rendered: an infrastructure failure.
TRACE_LOAD_FAILURE = 'trace_load'

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


class GoldClaimsContextualRecall(ContextualRecallDocumentStoredClaimsMetric):
  """The built-in stored-claims recall over the ERB gold claims (RQ1 attribution)."""

  claims_key = ERB_GOLD_CLAIMS_KEY
  metric_name = 'contextual_recall_gold_claims'
  metric_description = 'Contextual recall over the ERB gold claims; one judge call per claim.'


class ContextualPrecisionSC(ContextualPrecisionDocumentJudgeMetric):
  """Syllo-eval-SC for contextual precision: every document of a ranking judged in one call.

  The output budget is the per-document call's budget times the number of documents, capped at the judge's output
  limit. Judgments are matched to documents by rank; a wrong count or rank fails the unit.
  """

  metric_name = 'contextual_precision_document_judge_sc'
  metric_description = 'Syllo-eval-SC: contextual precision, judging all the documents of a ranking in one call.'

  def __init__(
    self,
    *,
    judge_client: LlmJudgeClient,
    output_token_limit: int,
    rubric_addition: str | None = None,
    target_span_types: Sequence[str] = ('agent_root',),
  ):
    super().__init__(judge_client=judge_client, rubric_addition=rubric_addition, target_span_types=target_span_types)
    self._output_token_limit = output_token_limit

  async def judge_items(
    self, span: Span, ground_truth: GroundTruth, rankings: list[list[RetrievedItem]]
  ) -> tuple[list[LlmJudgeResponse], list[ContextualPrecisionJudgment]] | MetricComputationResult:
    # An observed empty ranking needs no call.
    judged = [
      (items, min(self.max_output_tokens * len(items), self._output_token_limit)) for items in rankings if items
    ]
    budget = max((budget for _, budget in judged), default=None)
    responses, failure = await judge_batch(
      (
        self._judge_client.judge(
          LlmJudgeRequest(
            system_prompt=self.build_single_call_system_prompt(),
            user_prompt=self.build_single_call_user_prompt(span, ground_truth, items),
            response_model=ContextualPrecisionJudgePayload,
            temperature=0.0,
            max_output_tokens=ranking_budget,
          )
        )
        for items, ranking_budget in judged
      ),
      max_output_tokens=budget,
    )
    if failure is not None:
      failure.metadata = {**(failure.metadata or {}), 'max_output_tokens': budget}
      return failure

    judgments: list[ContextualPrecisionJudgment] = []
    for (items, ranking_budget), response in zip(judged, responses):
      ranks = [judgment.rank for judgment in response.output.judgments]
      expected_ranks = range(1, len(items) + 1)
      if sorted(ranks) != list(expected_ranks):
        alignment = {
          'missing_ranks': sorted(set(expected_ranks) - set(ranks)),
          'unknown_ranks': sorted(set(ranks) - set(expected_ranks)),
          'repeated_ranks': sorted(rank for rank, count in Counter(ranks).items() if count > 1),
        }
        message = (
          f'Judge returned {len(ranks)} contextual precision judgments for {len(items)} ranked '
          f'{self.variant}s: {alignment}.'
        )
        return _single_call_failure(
          self.name,
          message,
          response,
          responses,
          ranking_budget,
          variant=self.variant,
          retrieval_kind=self.variant,
          alignment=alignment,
        )
      by_rank = {judgment.rank: judgment for judgment in response.output.judgments}
      judgments.extend(by_rank[rank] for rank in expected_ranks)
    return responses, judgments

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


class GoldClaimsContextualRecallSC(GoldClaimsContextualRecall):
  """Syllo-eval-SC for recall over the ERB gold claims: every claim judged against the documents in one call.

  The output budget is the per-claim call's budget times the number of claims, capped at the judge's output limit.
  Judgments are matched to claims by position; a wrong count fails the unit.
  """

  metric_name = 'contextual_recall_gold_claims_sc'
  metric_description = 'Syllo-eval-SC: contextual recall over the ERB gold claims, judging all claims in one call.'

  def __init__(
    self,
    *,
    judge_client: LlmJudgeClient,
    output_token_limit: int,
    rubric_addition: str | None = None,
    target_span_types: Sequence[str] = ('agent_root',),
  ):
    super().__init__(judge_client=judge_client, rubric_addition=rubric_addition, target_span_types=target_span_types)
    self._output_token_limit = output_token_limit

  async def judge_claims(
    self,
    span: Span,
    retrieved_items: list[RetrievedItem],
    claims: list[str],
    *,
    prior_responses: Sequence[LlmJudgeResponse] = (),
  ) -> tuple[list[LlmJudgeResponse], list[ContextualRecallJudgment]] | MetricComputationResult:
    budget = min(self.max_output_tokens * len(claims), self._output_token_limit)
    responses, failure = await judge_batch(
      [
        self._judge_client.judge(
          LlmJudgeRequest(
            system_prompt=self.build_single_call_system_prompt(),
            user_prompt=self.build_single_call_user_prompt(retrieved_items, claims),
            response_model=ContextualRecallJudgePayload,
            temperature=0.0,
            max_output_tokens=budget,
          )
        )
      ],
      prior_responses=prior_responses,
      max_output_tokens=budget,
    )
    if failure is not None:
      failure.metadata = {**(failure.metadata or {}), 'max_output_tokens': budget}
      return failure
    judgments = responses[0].output.judgments
    if len(judgments) != len(claims):
      return _single_call_failure(
        self.name,
        f'Judge returned {len(judgments)} contextual recall judgments for {len(claims)} statements.',
        responses[0],
        [*prior_responses, *responses],
        budget,
        variant=self.variant,
        retrieval_kind=self.variant,
      )
    return responses, list(judgments)

  def build_single_call_system_prompt(self) -> str:
    return render_ablation_prompt(
      'contextual_recall_sc_system.md', variant=self.variant, rubric_addition=self._rubric_addition
    )

  def build_single_call_user_prompt(self, retrieved_items: list[RetrievedItem], claims: list[str]) -> str:
    return render_ablation_prompt(
      'contextual_recall_sc_user.md',
      variant=self.variant,
      items=_rendered_items(retrieved_items),
      claims='\n'.join(f'{index}. {claim}' for index, claim in enumerate(claims, start=1)),
    )


def _single_call_failure(
  metric_name: str,
  message: str,
  response: LlmJudgeResponse,
  responses: Sequence[LlmJudgeResponse],
  budget: int,
  **metadata: Any,
) -> MetricComputationResult:
  """A single-call unit whose judgments do not match its items: misaligned, or truncated at the output budget."""
  return MetricComputationResult(
    score=None,
    status=MetricComputationStatus.FAILED,
    reasoning=f'Failed to compute {metric_name}: {message}',
    metadata={
      **metadata,
      'failure': mismatch_kind(response, budget).value,
      'max_output_tokens': budget,
      **judge_metadata(responses),
    },
    error_message=message,
    raw_output=response.output.model_dump(),
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

  Spans appear depth-first with ordinal ids instead of source ids. Children are ordered by start time, then end time,
  type and name, so a re-identified copy renders the same unless two siblings agree on all four. Typed observations
  are rendered with the helpers the target metrics use, so the target's request, answer and executed steps appear
  verbatim. A span's request comes before its children; its retrieval results, plans, executed steps and answer come
  after them. A span without typed content shows its raw input and output instead.
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
    for child in sorted(children[span.external_id], key=_sibling_order):
      visit(child, ordinal)
    lines.extend(after_children)
    lines.append('</span>')

  visit(spans_by_id[target_span_id], None)
  return '\n'.join(lines), span_count


def _sibling_order(span: Span) -> tuple[Any, ...]:
  return span.start_time, span.end_time, span.span_type, span.name, span.external_id


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
      after_children += _section(
        f'plan {plan.id}{supersedes}', render_plan([step.model_dump(exclude_defaults=True) for step in plan.steps])
      )
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

  The skip rule, the system prompt and the failure handling are the built-in ones. The user prompt is the built-in one
  with the trace in place of the answer or plan paragraph, plus one sentence saying where in the trace that is.
  """

  def __init__(self, *, judge_client: LlmJudgeClient, load_spans: SpanLoader, rubric_addition: str | None = None):
    super().__init__(judge_client=judge_client, rubric_addition=rubric_addition)
    self._load_spans = load_spans

  @abstractmethod
  def build_whole_trace_user_prompt(self, span: Span, ground_truths: Mapping[str, GroundTruth], trace: str) -> str:
    """The built-in user prompt, with ``trace`` in place of the target's observations."""

  async def judge_input(
    self, span: Span, ground_truths: Mapping[str, GroundTruth]
  ) -> tuple[str, dict[str, Any]] | MetricComputationResult:
    try:
      trace, span_count = render_whole_trace(await self._load_spans(span.trace_id), span.external_id)
    except Exception as error:
      return MetricComputationResult(
        score=None,
        status=MetricComputationStatus.FAILED,
        reasoning=f'Failed to compute {self.name}: {error}',
        metadata={'failure': TRACE_LOAD_FAILURE},
        error_message=str(error),
      )
    # Recorded for every judged unit, failed ones included, so that trace length never depends on the judge succeeding.
    trace_render = {'spans': span_count, 'chars': len(trace)}
    return self.build_whole_trace_user_prompt(span, ground_truths, trace), {'trace_render': trace_render}


class AnswerCorrectnessWT(_WholeTraceJudgeMetric, AnswerCorrectnessJudgeMetric):
  """Syllo-eval-WT for Answer Correctness."""

  metric_name = 'answer_correctness_judge_wt'
  metric_description = 'Syllo-eval-WT: answer correctness judged from the whole canonical trace, not the final answer.'

  def build_whole_trace_user_prompt(self, span: Span, ground_truths: Mapping[str, GroundTruth], trace: str) -> str:
    ground_truth = ground_truths.get(self.ground_truth_keys[0])
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

  def build_whole_trace_user_prompt(self, span: Span, ground_truths: Mapping[str, GroundTruth], trace: str) -> str:
    ground_truth = ground_truths.get(self.ground_truth_keys[0])
    return render_ablation_prompt(
      'plan_correctness_wt_user.md',
      request=display_text(span.semantics.request),
      trace=trace,
      expected_plan=ground_truth_text(ground_truth, 'expected_plan'),
      rubric=ground_truth_text(ground_truth, 'rubric'),
      notes=ground_truth_text(ground_truth, 'notes'),
    )
