import os
import unittest
from datetime import datetime, timezone
from typing import Any, cast
from unittest.mock import AsyncMock, patch

from syllo_eval.evaluation.trace_processor import TraceProcessor
from syllo_eval.infrastructure import UnitOfWork
from syllo_eval.infrastructure.arize import PhoenixClient
from syllo_eval.infrastructure.exceptions import ExternalServiceError
from syllo_eval.settings import PhoenixSettings, load_settings_env
from syllo_eval.testing_database import setup_test_database as _setup_database


class _StaticTraceClient:
  def __init__(self, trace_id: str, records: list[dict[str, Any]]):
    self._trace_id = trace_id
    self._records = records

  async def get_trace_id_by_request_id(self, request_id: str) -> str | None:
    del request_id
    return self._trace_id

  async def get_trace_json(self, trace_id: str) -> list[dict[str, Any]]:
    assert trace_id == self._trace_id
    return self._records


class _TraceIdPhoenixClient:
  def __init__(self, trace_id: str, phoenix_client: PhoenixClient):
    self._trace_id = trace_id
    self._phoenix_client = phoenix_client

  async def get_trace_id_by_request_id(self, request_id: str) -> str | None:
    del request_id
    return self._trace_id

  async def get_trace_json(self, trace_id: str) -> list[dict[str, Any]]:
    return await self._phoenix_client.get_trace_json(trace_id)


class _TraceRepositoryStub:
  def __init__(self):
    self.get_by_id = AsyncMock(return_value=None)
    self.create = AsyncMock(side_effect=lambda trace: trace)
    self.update = AsyncMock(side_effect=lambda trace: trace)


class _SpanTypeRepositoryStub:
  def __init__(self):
    self.exists = AsyncMock(return_value=True)
    self.create = AsyncMock()
    self.upsert_from_registry = AsyncMock()


class _SpanRepositoryStub:
  def __init__(self):
    self.exists = AsyncMock(return_value=False)
    self.bulk_create = AsyncMock()
    self.update = AsyncMock()


class _TransactionalUnitOfWorkStub:
  def __init__(self):
    self.entered = False
    self.traces = _TraceRepositoryStub()
    self.span_types = _SpanTypeRepositoryStub()
    self.spans = _SpanRepositoryStub()

  async def __aenter__(self):
    self.entered = True
    return self

  async def __aexit__(self, exc_type, exc, tb):
    del exc_type, exc, tb


class TestTraceProcessorPersistence(unittest.IsolatedAsyncioTestCase):
  async def test_process_trace_persists_trace_and_spans_transactionally(self) -> None:
    now = datetime.now(timezone.utc)
    records: list[dict[str, Any]] = [
      {
        'context.span_id': 'span-transactional',
        'span_kind': 'bla',
        'context.trace_id': 'trace-transactional',
        'parent_id': None,
        'name': 'demo-agent',
        'start_time': now,
        'end_time': now,
        'attributes.input.value': {'prompt': 'hello'},
        'attributes.final_state': {'answer': 'hi'},
      }
    ]
    transactional_uow = _TransactionalUnitOfWorkStub()
    processor = TraceProcessor(
      trace_client=_StaticTraceClient(trace_id='trace-transactional', records=records),
      db_manager=cast(Any, object()),
    )

    with patch(
      'syllo_eval.evaluation.trace_processor.TransactionalUnitOfWork',
      return_value=transactional_uow,
      create=True,
    ):
      with patch(
        'syllo_eval.evaluation.trace_processor.UnitOfWork',
        side_effect=AssertionError('Trace ingestion must use TransactionalUnitOfWork'),
        create=True,
      ):
        result = await processor.process_trace('request-transactional')

    self.assertTrue(transactional_uow.entered)
    self.assertEqual(result.trace.external_id, 'trace-transactional')
    self.assertEqual([span.external_id for span in result.spans], ['span-transactional'])
    self.assertEqual(result.spans[0].span_type, 'bla')
    transactional_uow.span_types.upsert_from_registry.assert_awaited_once()
    self.assertEqual(transactional_uow.span_types.upsert_from_registry.call_args.args[0].name, 'bla')
    transactional_uow.span_types.exists.assert_not_awaited()
    transactional_uow.span_types.create.assert_not_awaited()
    transactional_uow.traces.create.assert_awaited_once()
    transactional_uow.spans.bulk_create.assert_awaited_once_with(result.spans)

  async def test_process_trace_marks_only_root_span_as_agent_root(self) -> None:
    now = datetime.now(timezone.utc)
    records: list[dict[str, Any]] = [
      {
        'context.span_id': 'span-root',
        'context.trace_id': 'trace-agent-root',
        'parent_id': None,
        'name': 'demo-agent',
        'start_time': now,
        'end_time': now,
        'attributes.input.value': {'prompt': 'hello'},
        'attributes.final_state': {'answer': 'hi'},
      },
      {
        'context.span_id': 'span-agent-op',
        'context.trace_id': 'trace-agent-root',
        'parent_id': 'span-root',
        'name': 'snippet_search',
        'attributes.openinference.span.kind': 'agent',
        'start_time': now,
        'end_time': now,
        'attributes.output.value': {'snippets': []},
      },
    ]
    transactional_uow = _TransactionalUnitOfWorkStub()
    processor = TraceProcessor(
      trace_client=_StaticTraceClient(trace_id='trace-agent-root', records=records),
      db_manager=cast(Any, object()),
    )

    with patch(
      'syllo_eval.evaluation.trace_processor.TransactionalUnitOfWork',
      return_value=transactional_uow,
      create=True,
    ):
      result = await processor.process_trace('request-agent-root')

    spans_by_id = {span.external_id: span for span in result.spans}
    self.assertEqual(spans_by_id['span-root'].span_type, 'agent_root')
    self.assertEqual(spans_by_id['span-agent-op'].span_type, 'agent')
    self.assertIsNone(spans_by_id['span-root'].output_data)
    self.assertEqual(spans_by_id['span-agent-op'].output_data, {'snippets': []})


class TestTraceProcessorLlmMetadata(unittest.IsolatedAsyncioTestCase):
  async def test_process_trace_captures_llm_token_counts_on_span_metadata(self) -> None:
    now = datetime.now(timezone.utc)
    records: list[dict[str, Any]] = [
      {
        'context.span_id': 'span-root',
        'context.trace_id': 'trace-tokens',
        'parent_id': None,
        'name': 'demo-agent',
        'start_time': now,
        'end_time': now,
      },
      {
        'context.span_id': 'span-llm-with-tokens',
        'context.trace_id': 'trace-tokens',
        'parent_id': 'span-root',
        'name': 'llm_call',
        'attributes.openinference.span.kind': 'llm',
        'start_time': now,
        'end_time': now,
        'attributes.llm.token_count.prompt': 12,
        'attributes.llm.token_count.completion': 5,
        'attributes.llm.token_count.total': 17,
      },
      {
        'context.span_id': 'span-llm-without-tokens',
        'context.trace_id': 'trace-tokens',
        'parent_id': 'span-root',
        'name': 'llm_followup',
        'attributes.openinference.span.kind': 'llm',
        'start_time': now,
        'end_time': now,
      },
    ]
    transactional_uow = _TransactionalUnitOfWorkStub()
    processor = TraceProcessor(
      trace_client=_StaticTraceClient(trace_id='trace-tokens', records=records),
      db_manager=cast(Any, object()),
    )

    with patch(
      'syllo_eval.evaluation.trace_processor.TransactionalUnitOfWork',
      return_value=transactional_uow,
      create=True,
    ):
      with patch(
        'syllo_eval.evaluation.trace_processor.UnitOfWork',
        side_effect=AssertionError('Trace ingestion must use TransactionalUnitOfWork'),
        create=True,
      ):
        result = await processor.process_trace('request-tokens')

    spans_by_id = {span.external_id: span for span in result.spans}
    assert spans_by_id['span-llm-with-tokens'].semantics.usage is not None
    self.assertEqual(
      spans_by_id['span-llm-with-tokens'].semantics.usage.model_dump(
        include={'input_tokens', 'output_tokens', 'total_tokens'}
      ),
      {'input_tokens': 12, 'output_tokens': 5, 'total_tokens': 17},
    )
    self.assertNotIn('token_usage', spans_by_id['span-llm-without-tokens'].metadata or {})


class TestTraceProcessorIntegration(unittest.IsolatedAsyncioTestCase):
  async def asyncSetUp(self) -> None:
    load_settings_env()
    self.trace_id = os.environ.get('TRACE_PROCESSOR_TEST_TRACE_ID')
    if not self.trace_id:
      self.skipTest('TRACE_PROCESSOR_TEST_TRACE_ID not configured')

    self.processed_trace_id: str | None = None
    self.created_span_types: set[str] = set()
    self._trace_preexisted = False

    try:
      self.db_manager = await _setup_database()
    except RuntimeError as error:
      self.skipTest(str(error))

    async with UnitOfWork(self.db_manager) as uow:
      self.preexisting_span_types = {span_type.name for span_type in await uow.span_types.list_all()}
      self._trace_preexisted = await uow.traces.get_by_id(self.trace_id) is not None

  async def asyncTearDown(self) -> None:
    if hasattr(self, 'db_manager'):
      if not self._trace_preexisted:
        async with UnitOfWork(self.db_manager) as uow:
          if self.processed_trace_id is not None:
            await uow.traces.delete(self.processed_trace_id)
          for span_type_name in sorted(self.created_span_types - self.preexisting_span_types):
            await uow.span_types.delete(span_type_name)
      await self.db_manager.close_async()

  async def test_process_trace_from_phoenix(self) -> None:
    phoenix_config = PhoenixSettings()
    if not phoenix_config.project_id:
      phoenix_config.project_id = 'default'

    assert self.trace_id is not None
    processor = TraceProcessor(
      trace_client=_TraceIdPhoenixClient(self.trace_id, PhoenixClient(phoenix_config)),
      db_manager=self.db_manager,
    )

    try:
      result = await processor.process_trace(request_id='trace-processor-live-test')
    except ExternalServiceError as error:
      self.skipTest(f'Phoenix service unavailable or not configured: {error}')

    self.processed_trace_id = result.trace.external_id
    self.created_span_types = {span.span_type for span in result.spans}

    self.assertIsNotNone(result.trace.external_id)
    self.assertGreater(len(result.spans), 0)

    async with UnitOfWork(self.db_manager) as uow:
      persisted_trace = await uow.traces.get_by_id(result.trace.external_id)
      self.assertIsNotNone(persisted_trace)

      persisted_spans = await uow.spans.list_by_trace(result.trace.external_id)
      self.assertGreaterEqual(len(persisted_spans), len(result.spans))


if __name__ == '__main__':
  unittest.main()
