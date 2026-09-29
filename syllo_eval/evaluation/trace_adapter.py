"""Trace normalization boundary. Agent adapters return validated, provider-neutral data."""

import json
from graphlib import TopologicalSorter
from typing import Any, Literal, Protocol

from pydantic import BaseModel, model_validator

from syllo_eval.model import Span, Trace
from syllo_eval.trace_semantics import LlmUsage, SpanSemantics


class TraceProcessingResult(BaseModel):
  trace: Trace
  spans: list[Span]

  @model_validator(mode='after')
  def validate_trace(self) -> 'TraceProcessingResult':
    if self.trace.schema_version != 1:
      raise ValueError('Unsupported canonical trace schema version')
    if not self.spans:
      raise ValueError('A trace must contain spans')
    spans = {
      span.external_id: span for span in sorted(self.spans, key=lambda span: (span.start_time, span.external_id))
    }
    if len(spans) != len(self.spans):
      raise ValueError('Span IDs must be unique within a trace')
    for span in self.spans:
      if span.trace_id != self.trace.external_id:
        raise ValueError('Span belongs to a different trace')
      if span.end_time < span.start_time:
        raise ValueError('Span end_time precedes start_time')
      if span.parent_span_id is not None and span.parent_span_id not in spans:
        raise ValueError(f'Missing parent span: {span.parent_span_id}')
    graph = {key: [span.parent_span_id] if span.parent_span_id else [] for key, span in spans.items()}
    self.spans = [spans[key] for key in TopologicalSorter(graph).static_order()]
    if self.trace.end_time < self.trace.start_time:
      raise ValueError('Trace end_time precedes start_time')
    observed_plans = [plan for span in self.spans if span.semantics.planning for plan in span.semantics.planning.plans]
    plans = {plan.id: plan for plan in observed_plans}
    if len(plans) != len(observed_plans):
      raise ValueError('Plan snapshot IDs must be unique within a trace')
    TopologicalSorter(
      {key: [plan.supersedes_plan_id] if plan.supersedes_plan_id else [] for key, plan in plans.items()}
    ).prepare()
    for plan in plans.values():
      if plan.supersedes_plan_id is not None and plan.supersedes_plan_id not in plans:
        raise ValueError(f'Missing superseded plan: {plan.supersedes_plan_id}')
    for span in self.spans:
      planning = span.semantics.planning
      for step in (planning.executed_steps or []) if planning else []:
        if not set(step.span_ids) <= spans.keys():
          raise ValueError(f'Unknown execution span in step {step.id}')
        if step.plan_id is not None and step.plan_id not in plans:
          raise ValueError(f'Unknown plan in step {step.id}')
        if step.planned_step_id is not None:
          if step.plan_id is None or step.planned_step_id not in {s.id for s in plans[step.plan_id].steps}:
            raise ValueError(f'Unknown planned step in execution step {step.id}')
    return self


class TraceAdapter(Protocol):
  def normalize(self, trace_id: str, records: list[dict[str, Any]]) -> TraceProcessingResult: ...


def decode_payload(value: Any) -> Any:
  """Decode JSON when supplied as text; ordinary text stays text."""
  if isinstance(value, str):
    try:
      return json.loads(value)
    except json.JSONDecodeError:
      return value
  return value


class PhoenixTraceAdapter:
  """Decode Phoenix transport fields without interpreting agent-specific payloads."""

  def normalize(self, trace_id: str, records: list[dict[str, Any]]) -> TraceProcessingResult:
    spans = [self._span(trace_id, record) for record in records]
    if not spans:
      raise ValueError(f'No spans found for trace_id={trace_id}')
    return TraceProcessingResult(
      trace=Trace(
        external_id=trace_id,
        start_time=min(span.start_time for span in spans),
        end_time=max(span.end_time for span in spans),
        source='phoenix',
        adapter='phoenix:1',
      ),
      spans=spans,
    )

  def _span(self, trace_id: str, record: dict[str, Any]) -> Span:
    parent_id = record.get('parent_id')
    kind = str(record.get('attributes.openinference.span.kind') or record.get('span_kind') or 'unknown').lower()
    kind = {'retriever': 'retrieval'}.get(kind, kind)
    if parent_id is None and kind in {'agent', 'chain', 'unknown'}:
      kind = 'agent_root'
    metadata = {key: decode_payload(value) for key, value in record.items() if key.startswith('attributes.')}
    status: Literal['success', 'error', 'unset'] = 'unset'
    if record.get('status_code') == 'OK':
      status = 'success'
    elif record.get('status_code') == 'ERROR':
      status = 'error'
    if record.get('status_message'):
      metadata['phoenix.status_message'] = record['status_message']
    usage = None
    if kind == 'llm':
      usage = LlmUsage(
        call_id=str(record['context.span_id']),
        model=record.get('attributes.llm.model_name'),
        provider=record.get('attributes.llm.provider'),
        input_tokens=record.get('attributes.llm.token_count.prompt'),
        output_tokens=record.get('attributes.llm.token_count.completion'),
        total_tokens=record.get('attributes.llm.token_count.total'),
        cost=record.get('attributes.llm.cost.total_usd'),
        currency='USD' if record.get('attributes.llm.cost.total_usd') is not None else None,
      )
    return Span(
      external_id=str(record['context.span_id']),
      trace_id=trace_id,
      parent_span_id=str(parent_id) if parent_id is not None else None,
      span_type=kind,
      status=status,
      name=str(record.get('name') or record['context.span_id']),
      start_time=record['start_time'],
      end_time=record['end_time'],
      input_data=decode_payload(record.get('attributes.input.value')),
      output_data=decode_payload(record.get('attributes.output.value')),
      metadata=metadata,
      semantics=SpanSemantics(usage=usage),
    )
