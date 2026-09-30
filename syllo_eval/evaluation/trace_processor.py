"""Fetch, normalize, validate, and transactionally persist agent traces."""

from typing import Any, Protocol

from syllo_eval.evaluation.trace_adapter import PhoenixTraceAdapter, TraceAdapter, TraceProcessingResult
from syllo_eval.infrastructure.database import DatabaseManager
from syllo_eval.infrastructure.unit_of_work import TransactionalUnitOfWork
from syllo_eval.model import SpanType


class TraceSourceClient(Protocol):
  async def get_trace_id_by_request_id(self, request_id: str) -> str | None: ...

  async def get_trace_json(self, trace_id: str) -> list[dict[str, Any]]: ...


class TraceIntegration(Protocol):
  """Caller-owned integration returning a validated trace persisted in the evaluation database."""

  async def process_trace(self, request_id: str) -> TraceProcessingResult: ...


class TraceProcessor:
  def __init__(self, trace_client: TraceSourceClient, db_manager: DatabaseManager, adapter: TraceAdapter | None = None):
    self.trace_client = trace_client
    self.db_manager = db_manager
    self.adapter = adapter if adapter is not None else PhoenixTraceAdapter()

  async def process_trace(self, request_id: str) -> TraceProcessingResult:
    trace_id = await self.trace_client.get_trace_id_by_request_id(request_id)
    if trace_id is None:
      raise ValueError(f'No trace found for request_id={request_id}')
    records = await self.trace_client.get_trace_json(trace_id)
    result = normalize_trace(self.adapter, trace_id, records)
    async with TransactionalUnitOfWork(self.db_manager) as uow:
      return await persist_trace(uow, result)


def normalize_trace(adapter: TraceAdapter, trace_id: str, records: list[dict[str, Any]]) -> TraceProcessingResult:
  normalized = adapter.normalize(trace_id, records)
  # Validate even when an injected adapter used model_construct or mutated its result.
  return TraceProcessingResult.model_validate(normalized.model_dump())


async def persist_trace(uow: TransactionalUnitOfWork, result: TraceProcessingResult) -> TraceProcessingResult:
  existing = await uow.traces.get_by_id(result.trace.external_id)
  if existing is not None:
    stored = TraceProcessingResult(trace=existing, spans=await uow.spans.list_by_trace(existing.external_id))
    if stored != result:
      raise ValueError('Trace already exists with different normalized content; use a new trace identity')
    return stored
  await uow.traces.create(result.trace)
  for span_type in sorted({span.span_type for span in result.spans}):
    await uow.span_types.upsert_from_registry(SpanType(name=span_type))
  await uow.spans.bulk_create(result.spans)
  return result
