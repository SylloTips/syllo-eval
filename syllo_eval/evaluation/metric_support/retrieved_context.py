from dataclasses import dataclass
from typing import Any

from syllo_eval.evaluation.metric_support.agent_outputs import skipped_retrieval_result
from syllo_eval.model import Span
from syllo_eval.evaluation.metrics.contracts import MetricComputationResult
from syllo_eval.trace_semantics import RetrievalItem, RetrievalResult


@dataclass(frozen=True, slots=True)
class RetrievedItem:
  item: RetrievalItem

  @property
  def retrieved_id(self) -> str:
    return self.item.id

  @property
  def score(self) -> float | None:
    return self.item.score

  def render_for_prompt(self, rank: int) -> str:
    return (
      f'Rank {rank}\n'
      f'retrieved_id: {self.retrieved_id}\n'
      f'document_id: {self.item.document_id or self.item.id}\n'
      f'title: {self.item.title or ""}\n'
      f'location: {self.item.location or ""}\n'
      f'retrieval_score: {self.item.score}\n'
      f'content:\n{self.item.content or ""}'
    )


def extract_retrieval_results(span: Span, kind: str, stage: str = 'selected') -> list[RetrievalResult]:
  return [
    result
    for result in span.semantics.retrieval
    if result.kind == kind and result.stage == stage and result.availability == 'available'
  ]


def extract_retrieved_items(span: Span, variant: str, stage: str = 'selected') -> list[RetrievedItem]:
  return [RetrievedItem(item) for result in extract_retrieval_results(span, variant, stage) for item in result.items]


def extract_retrieved_ids(span: Span, variant: str, stage: str = 'selected') -> set[str]:
  return {item.retrieved_id for item in extract_retrieved_items(span, variant, stage)}


def retrieval_skip_reason(
  span: Span, variant: str, *, stage: str = 'selected', require_content: bool = False, require_rank: bool = False
) -> str | None:
  results = [result for result in span.semantics.retrieval if result.kind == variant and result.stage == stage]
  if not results:
    return f'Trace does not provide {stage} {variant} context.'

  available_results = [result for result in results if result.availability == 'available']
  if len(available_results) != len(results):
    reason = next((result.reason for result in results if result.reason), 'results were unavailable')
    return f'{variant.capitalize()} retrieval unavailable: {reason}.'
  if require_rank and not all(result.ranked for result in available_results):
    return f'{variant.capitalize()} {stage} context is not ranked.'

  items = [item for result in available_results for item in result.items]
  if require_content and any(item.content is None for item in items):
    return f'Retrieved {variant} items do not contain the content required by this metric.'

  return None


def retrieval_skip_result(
  span: Span,
  *,
  metric_name: str,
  variant: str,
  stage: str = 'selected',
  require_content: bool = False,
  require_rank: bool = False,
) -> MetricComputationResult | None:
  reason = retrieval_skip_reason(span, variant, stage=stage, require_content=require_content, require_rank=require_rank)
  if reason is None:
    return None
  return skipped_retrieval_result(metric_name, reason, variant=variant)


def normalize_relevant_ids(values: list[Any]) -> set[str]:
  """Accept string IDs and objects containing an id field."""
  candidates = (value.get('id') if isinstance(value, dict) else value for value in values)
  return {value.strip() for value in candidates if isinstance(value, str) and value.strip()}
