"""
Repository for Span entity operations.
"""

import json
from typing import Dict, Any, List
from uuid import UUID

from syllo_eval.model import Span
from syllo_eval.infrastructure.repositories.base import BaseRepository
from psycopg.rows import class_row
from psycopg.types.json import Jsonb


class SpanRepository(BaseRepository[Span]):
  """Repository for managing Span entities."""

  @property
  def table_name(self) -> str:
    return 'span'

  @property
  def model_class(self) -> type[Span]:
    return Span

  @property
  def id_column(self) -> str:
    return 'external_id'

  @staticmethod
  def _normalize_span_metadata(span: Span) -> Span:
    """Normalize metadata payload returned by DB into a dict when needed."""
    if span.metadata and isinstance(span.metadata, str):
      span.metadata = json.loads(span.metadata)
    return span

  def _get_insert_fields(self, entity: Span) -> Dict[str, Any]:
    """Get fields for inserting a new span."""
    fields = {
      'external_id': entity.external_id,
      'trace_id': entity.trace_id,
      'span_type': entity.span_type,
      'name': entity.name,
      'start_time': entity.start_time,
      'end_time': entity.end_time,
      'input_data': Jsonb(entity.input_data),
      'output_data': Jsonb(entity.output_data),
      'status': entity.status,
      'semantics': Jsonb(entity.semantics.model_dump(mode='json')),
    }

    if entity.parent_span_id is not None:
      fields['parent_span_id'] = entity.parent_span_id
    if entity.metadata is not None:
      fields['metadata'] = json.dumps(entity.metadata)

    return fields

  def _get_update_fields(self, entity: Span) -> Dict[str, Any]:
    """Get fields for updating a span."""
    fields = {
      'trace_id': entity.trace_id,
      'parent_span_id': entity.parent_span_id,
      'span_type': entity.span_type,
      'name': entity.name,
      'start_time': entity.start_time,
      'end_time': entity.end_time,
      'input_data': Jsonb(entity.input_data),
      'output_data': Jsonb(entity.output_data),
      'status': entity.status,
      'semantics': Jsonb(entity.semantics.model_dump(mode='json')),
      'metadata': json.dumps(entity.metadata) if entity.metadata is not None else None,
    }

    return fields

  async def list_by_trace(self, trace_id: str) -> List[Span]:
    """
    List all spans in a trace.

    Args:
        trace_id: Trace ID

    Returns:
        List of spans ordered by start time
    """
    try:
      async with self._get_connection() as conn:
        query = """
                    SELECT * FROM span
                    WHERE trace_id = %s
                    ORDER BY start_time ASC
                """

        async with conn.cursor(row_factory=class_row(Span)) as cur:
          await cur.execute(query, (trace_id,))
          results = await cur.fetchall()
          return [self._normalize_span_metadata(result) for result in results]

    except Exception as e:
      self._handle_db_error(e, f'list_by_trace({trace_id})')

  async def list_usage_spans_by_evaluation_run(self, evaluation_run_id: UUID) -> List[Span]:
    """List canonical usage sources and legacy LLM spans, once per span."""
    try:
      async with self._get_connection() as conn:
        query = """
                    SELECT span.* FROM span
                    WHERE EXISTS (
                        SELECT 1 FROM evaluation_run_sample ers
                        WHERE ers.trace_id = span.trace_id AND ers.evaluation_run_id = %s
                    ) AND (
                        span.span_type = 'llm'
                        OR jsonb_typeof(span.semantics -> 'usage') = 'object'
                        OR jsonb_typeof(span.metadata -> 'token_usage') = 'object'
                    )
                    ORDER BY span.start_time ASC, span.external_id ASC
                """

        async with conn.cursor(row_factory=class_row(Span)) as cur:
          await cur.execute(query, (evaluation_run_id,))
          results = await cur.fetchall()
          return [self._normalize_span_metadata(result) for result in results]

    except Exception as e:
      self._handle_db_error(e, f'list_usage_spans_by_evaluation_run({evaluation_run_id})')

  async def list_by_span_type(self, trace_id: str, span_type: str) -> List[Span]:
    """
    List all spans of a specific type within a trace.

    Args:
        trace_id: Trace ID
        span_type: Span type to filter by

    Returns:
        List of spans
    """
    try:
      async with self._get_connection() as conn:
        query = """
                    SELECT * FROM span
                    WHERE trace_id = %s AND span_type = %s
                    ORDER BY start_time ASC
                """

        async with conn.cursor(row_factory=class_row(Span)) as cur:
          await cur.execute(query, (trace_id, span_type))
          results = await cur.fetchall()
          return [self._normalize_span_metadata(result) for result in results]

    except Exception as e:
      self._handle_db_error(e, f'list_by_span_type({trace_id}, {span_type})')

  async def get_root_spans(self, trace_id: str) -> List[Span]:
    """
    Get all root spans (spans with no parent) in a trace.

    Args:
        trace_id: Trace ID

    Returns:
        List of root spans
    """
    try:
      async with self._get_connection() as conn:
        query = """
                    SELECT * FROM span
                    WHERE trace_id = %s AND parent_span_id IS NULL
                    ORDER BY start_time ASC
                """

        async with conn.cursor(row_factory=class_row(Span)) as cur:
          await cur.execute(query, (trace_id,))
          results = await cur.fetchall()
          return [self._normalize_span_metadata(result) for result in results]

    except Exception as e:
      self._handle_db_error(e, f'get_root_spans({trace_id})')

  async def get_child_spans(self, parent_span_id: str) -> List[Span]:
    """
    Get all child spans of a parent span.

    Args:
        parent_span_id: Parent span ID

    Returns:
        List of child spans
    """
    try:
      async with self._get_connection() as conn:
        query = """
                    SELECT * FROM span
                    WHERE parent_span_id = %s
                    ORDER BY start_time ASC
                """

        async with conn.cursor(row_factory=class_row(Span)) as cur:
          await cur.execute(query, (parent_span_id,))
          results = await cur.fetchall()
          return [self._normalize_span_metadata(result) for result in results]

    except Exception as e:
      self._handle_db_error(e, f'get_child_spans({parent_span_id})')

  async def bulk_create(self, spans: List[Span]) -> List[Span]:
    """
    Create multiple spans in a single transaction.

    Args:
        spans: List of spans to create

    Returns:
        List of created spans

    Raises:
        DuplicateError: If any span violates unique constraints
        IntegrityError: If any span violates foreign key constraints
    """
    if not spans:
      return []

    try:
      if self._connection is not None:
        conn = self._connection
        created_spans = []

        for span in spans:
          fields = self._get_insert_fields(span)
          columns = ', '.join(fields.keys())
          placeholders = ', '.join(['%s'] * len(fields))
          values = tuple(fields.values())

          query = f"""
                        INSERT INTO {self.table_name} ({columns})
                        VALUES ({placeholders})
                        RETURNING *
                    """

          async with conn.cursor(row_factory=class_row(Span)) as cur:
            await cur.execute(query, values)
            result = await cur.fetchone()
            if result:
              created_spans.append(self._normalize_span_metadata(result))

        return created_spans

      async with self.db_manager.transaction() as conn:
        created_spans = []

        for span in spans:
          fields = self._get_insert_fields(span)
          columns = ', '.join(fields.keys())
          placeholders = ', '.join(['%s'] * len(fields))
          values = tuple(fields.values())

          query = f"""
                        INSERT INTO {self.table_name} ({columns})
                        VALUES ({placeholders})
                        RETURNING *
                    """

          async with conn.cursor(row_factory=class_row(Span)) as cur:
            await cur.execute(query, values)
            result = await cur.fetchone()
            if result:
              created_spans.append(self._normalize_span_metadata(result))

        return created_spans

    except Exception as e:
      self._handle_db_error(e, 'bulk_create')

  async def count_by_trace(self, trace_id: str) -> int:
    """
    Count spans in a trace.

    Args:
        trace_id: Trace ID

    Returns:
        Number of spans in the trace
    """
    try:
      async with self._get_connection() as conn:
        query = """
                    SELECT COUNT(*) FROM span
                    WHERE trace_id = %s
                """

        async with conn.cursor() as cur:
          await cur.execute(query, (trace_id,))
          result = await cur.fetchone()
          return result[0] if result else 0

    except Exception as e:
      self._handle_db_error(e, f'count_by_trace({trace_id})')
