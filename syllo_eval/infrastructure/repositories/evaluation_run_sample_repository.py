"""
Repository for EvaluationRunSample entity operations.
"""

import json
from datetime import datetime
from typing import Dict, Any, List
from uuid import UUID

from psycopg.rows import class_row

from syllo_eval.model import EvaluationRunSample, EvaluationSampleStatus
from syllo_eval.infrastructure.repositories.base import BaseRepository


class EvaluationRunSampleRepository(BaseRepository[EvaluationRunSample]):
  """Repository for managing EvaluationRunSample entities."""

  @property
  def table_name(self) -> str:
    return 'evaluation_run_sample'

  @property
  def model_class(self) -> type[EvaluationRunSample]:
    return EvaluationRunSample

  def _get_insert_fields(self, entity: EvaluationRunSample) -> Dict[str, Any]:
    fields: Dict[str, Any] = {
      'evaluation_run_id': entity.evaluation_run_id,
      'sample_id': entity.sample_id,
      'trace_id': entity.trace_id,
      'status': entity.status,
    }
    if entity.id is not None:
      fields['id'] = entity.id
    if entity.started_at is not None:
      fields['started_at'] = entity.started_at
    if entity.ended_at is not None:
      fields['ended_at'] = entity.ended_at
    if entity.error_message is not None:
      fields['error_message'] = entity.error_message
    if entity.metadata is not None:
      fields['metadata'] = json.dumps(entity.metadata)
    return fields

  def _get_update_fields(self, entity: EvaluationRunSample) -> Dict[str, Any]:
    fields: Dict[str, Any] = {
      'trace_id': entity.trace_id,
      'status': entity.status,
      'started_at': entity.started_at,
      'ended_at': entity.ended_at,
      'error_message': entity.error_message,
    }
    if entity.metadata is not None:
      fields['metadata'] = json.dumps(entity.metadata)
    return fields

  async def list_by_evaluation_run(self, evaluation_run_id: UUID) -> List[EvaluationRunSample]:
    """List all samples in an evaluation run."""
    try:
      async with self._get_connection() as conn:
        query = """
                    SELECT * FROM evaluation_run_sample
                    WHERE evaluation_run_id = %s
                    ORDER BY created_at ASC
                """
        async with conn.cursor(row_factory=class_row(EvaluationRunSample)) as cur:
          await cur.execute(query, (evaluation_run_id,))
          return [self._normalize_required(result) for result in await cur.fetchall()]
    except Exception as e:
      self._handle_db_error(e, f'list_by_evaluation_run({evaluation_run_id})')

  async def get_by_run_and_sample(
    self,
    evaluation_run_id: UUID,
    sample_id: UUID,
  ) -> EvaluationRunSample | None:
    """Get evaluation run sample by run and sample ID."""
    try:
      async with self._get_connection() as conn:
        query = """
                    SELECT * FROM evaluation_run_sample
                    WHERE evaluation_run_id = %s AND sample_id = %s
                """
        async with conn.cursor(row_factory=class_row(EvaluationRunSample)) as cur:
          await cur.execute(query, (evaluation_run_id, sample_id))
          return self._normalize(await cur.fetchone())
    except Exception as e:
      self._handle_db_error(
        e,
        f'get_by_run_and_sample({evaluation_run_id}, {sample_id})',
      )

  async def update_status(
    self,
    run_sample_id: UUID,
    status: EvaluationSampleStatus,
    trace_id: str | None = None,
    started_at: datetime | None = None,
    ended_at: datetime | None = None,
    error_message: str | None = None,
    metadata: dict[str, Any] | None = None,
  ) -> EvaluationRunSample:
    """Update sample execution status and lifecycle fields."""
    try:
      async with self._get_connection() as conn:
        fields: dict[str, Any] = {'status': status}
        if trace_id is not None:
          fields['trace_id'] = trace_id
        if started_at is not None:
          fields['started_at'] = started_at
        if ended_at is not None:
          fields['ended_at'] = ended_at
        if error_message is not None:
          fields['error_message'] = error_message
        if metadata is not None:
          fields['metadata'] = json.dumps(metadata)

        set_clause = ', '.join(f'{key} = %s' for key in fields)
        query = f"""
                    UPDATE evaluation_run_sample
                    SET {set_clause}
                    WHERE id = %s
                    RETURNING *
                """
        async with conn.cursor(row_factory=class_row(EvaluationRunSample)) as cur:
          await cur.execute(query, tuple(fields.values()) + (run_sample_id,))
          result = await cur.fetchone()
          if result is None:
            return await self.get_by_id_or_raise(run_sample_id)
          return self._normalize_required(result)
    except Exception as e:
      self._handle_db_error(e, f'update_status({run_sample_id}, {status})')

  @staticmethod
  def _normalize(run_sample: EvaluationRunSample | None) -> EvaluationRunSample | None:
    if run_sample is None:
      return None

    if run_sample.metadata and isinstance(run_sample.metadata, str):
      run_sample.metadata = json.loads(run_sample.metadata)

    return run_sample

  @staticmethod
  def _normalize_required(run_sample: EvaluationRunSample) -> EvaluationRunSample:
    normalized = EvaluationRunSampleRepository._normalize(run_sample)
    if normalized is None:
      raise ValueError('Expected evaluation run sample row.')
    return normalized
