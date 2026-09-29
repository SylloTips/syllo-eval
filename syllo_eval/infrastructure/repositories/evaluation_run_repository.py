"""
Repository for EvaluationRun entity operations.
"""

import json
from datetime import datetime
from typing import Dict, Any, Optional, List, Sequence
from uuid import UUID

from syllo_eval.model import EvaluationRun, EvaluationStatus
from syllo_eval.infrastructure.repositories.base import BaseRepository
from syllo_eval.infrastructure.exceptions import NotFoundError
from psycopg.rows import class_row


class EvaluationRunRepository(BaseRepository[EvaluationRun]):
  """Repository for managing EvaluationRun entities."""

  @property
  def table_name(self) -> str:
    return 'evaluation_run'

  @property
  def model_class(self) -> type[EvaluationRun]:
    return EvaluationRun

  def _get_insert_fields(self, entity: EvaluationRun) -> Dict[str, Any]:
    """Get fields for inserting a new evaluation run."""
    fields: Dict[str, Any] = {
      'agent_id': entity.agent_id,
      'dataset_id': entity.dataset_id,
      'status': entity.status.value,
      'start_time': entity.start_time,
    }

    if entity.id is not None:
      fields['id'] = entity.id
    if entity.end_time is not None:
      fields['end_time'] = entity.end_time
    if entity.source_run_id is not None:
      fields['source_run_id'] = entity.source_run_id
    if entity.config is not None:
      fields['config'] = json.dumps(entity.config)

    return fields

  def _get_update_fields(self, entity: EvaluationRun) -> Dict[str, Any]:
    """Get fields for updating an evaluation run."""
    fields: Dict[str, Any] = {
      'status': entity.status.value,
    }

    if entity.end_time is not None:
      fields['end_time'] = entity.end_time

    return fields

  async def update_status(
    self, run_id: UUID, status: EvaluationStatus, end_time: Optional[datetime] = None
  ) -> EvaluationRun:
    """
    Update the status of an evaluation run.

    Args:
        run_id: ID of the evaluation run
        status: New status
        end_time: Optional end time (for COMPLETED/FAILED status)

    Returns:
        Updated evaluation run

    Raises:
        NotFoundError: If run not found
    """
    try:
      async with self._get_connection() as conn:
        query = """
                    UPDATE evaluation_run
                    SET status = %s, end_time = %s
                    WHERE id = %s
                    RETURNING *
                """

        async with conn.cursor(row_factory=class_row(EvaluationRun)) as cur:
          await cur.execute(query, (status.value, end_time, run_id))
          result = await cur.fetchone()

          if result is None:
            raise NotFoundError('evaluation_run', run_id)

          return result

    except NotFoundError:
      raise
    except Exception as e:
      self._handle_db_error(e, f'update_status({run_id}, {status})')

  async def update_status_if_current(
    self,
    run_id: UUID,
    current_statuses: Sequence[EvaluationStatus],
    status: EvaluationStatus,
    end_time: Optional[datetime] = None,
  ) -> EvaluationRun | None:
    """Update run status only when its current status matches one of the provided values."""
    if not current_statuses:
      raise ValueError('current_statuses must contain at least one status')

    try:
      async with self._get_connection() as conn:
        placeholders = ', '.join(['%s'] * len(current_statuses))
        query = f"""
                    UPDATE evaluation_run
                    SET status = %s, end_time = %s
                    WHERE id = %s AND status IN ({placeholders})
                    RETURNING *
                """
        params = [status.value, end_time, run_id, *[current_status.value for current_status in current_statuses]]

        async with conn.cursor(row_factory=class_row(EvaluationRun)) as cur:
          await cur.execute(query, params)
          return await cur.fetchone()

    except Exception as e:
      self._handle_db_error(
        e,
        f'update_status_if_current({run_id}, {[current_status.value for current_status in current_statuses]}, \
        {status})',
      )

  async def list_runs(
    self,
    *,
    limit: int,
    offset: int = 0,
    status: EvaluationStatus | None = None,
  ) -> List[EvaluationRun]:
    """List evaluation runs in reverse chronological order."""
    try:
      async with self._get_connection() as conn:
        query = 'SELECT * FROM evaluation_run'
        params: List[Any] = []
        if status is not None:
          query += ' WHERE status = %s'
          params.append(status.value)

        query += ' ORDER BY start_time DESC, id DESC LIMIT %s OFFSET %s'
        params.extend([limit, offset])

        async with conn.cursor(row_factory=class_row(EvaluationRun)) as cur:
          await cur.execute(query, params)
          return await cur.fetchall()

    except Exception as e:
      self._handle_db_error(e, f'list_runs({limit}, {offset}, {status})')

  async def count_runs(self, status: EvaluationStatus | None = None) -> int:
    """Count evaluation runs, optionally filtered by status."""
    try:
      async with self._get_connection() as conn:
        query = 'SELECT COUNT(*) FROM evaluation_run'
        params: List[Any] = []
        if status is not None:
          query += ' WHERE status = %s'
          params.append(status.value)

        async with conn.cursor() as cur:
          await cur.execute(query, params)
          result = await cur.fetchone()
          return result[0] if result else 0

    except Exception as e:
      self._handle_db_error(e, f'count_runs({status})')

  async def fail_running(self, end_time: datetime) -> List[EvaluationRun]:
    """Mark all currently running evaluation runs as failed."""
    try:
      async with self._get_connection() as conn:
        query = """
                    UPDATE evaluation_run
                    SET status = %s, end_time = %s
                    WHERE status = %s
                    RETURNING *
                """

        async with conn.cursor(row_factory=class_row(EvaluationRun)) as cur:
          await cur.execute(query, (EvaluationStatus.FAILED.value, end_time, EvaluationStatus.RUNNING.value))
          return await cur.fetchall()

    except Exception as e:
      self._handle_db_error(e, 'fail_running')

  async def list_by_agent(
    self, agent_id: UUID, status: Optional[EvaluationStatus] = None, limit: Optional[int] = None
  ) -> List[EvaluationRun]:
    """
    List evaluation runs for a specific agent.

    Args:
        agent_id: Agent ID
        status: Optional status filter
        limit: Maximum number of results

    Returns:
        List of evaluation runs
    """
    try:
      async with self._get_connection() as conn:
        query = """
                    SELECT * FROM evaluation_run
                    WHERE agent_id = %s
                """
        params: List[Any] = [agent_id]

        if status is not None:
          query += ' AND status = %s'
          params.append(status.value)

        query += ' ORDER BY start_time DESC'

        if limit is not None:
          query += f' LIMIT {limit}'

        async with conn.cursor(row_factory=class_row(EvaluationRun)) as cur:
          await cur.execute(query, params)
          return await cur.fetchall()

    except Exception as e:
      self._handle_db_error(e, f'list_by_agent({agent_id})')

  async def list_by_dataset(
    self, dataset_id: UUID, status: Optional[EvaluationStatus] = None, limit: Optional[int] = None
  ) -> List[EvaluationRun]:
    """
    List evaluation runs for a specific dataset.

    Args:
        dataset_id: Dataset ID
        status: Optional status filter
        limit: Maximum number of results

    Returns:
        List of evaluation runs
    """
    try:
      async with self._get_connection() as conn:
        query = """
                    SELECT * FROM evaluation_run
                    WHERE dataset_id = %s
                """
        params: List[Any] = [dataset_id]

        if status is not None:
          query += ' AND status = %s'
          params.append(status.value)

        query += ' ORDER BY start_time DESC'

        if limit is not None:
          query += f' LIMIT {limit}'

        async with conn.cursor(row_factory=class_row(EvaluationRun)) as cur:
          await cur.execute(query, params)
          return await cur.fetchall()

    except Exception as e:
      self._handle_db_error(e, f'list_by_dataset({dataset_id})')

  async def list_by_status(self, status: EvaluationStatus, limit: Optional[int] = None) -> List[EvaluationRun]:
    """
    List evaluation runs by status.

    Args:
        status: Status to filter by
        limit: Maximum number of results

    Returns:
        List of evaluation runs
    """
    try:
      async with self._get_connection() as conn:
        query = """
                    SELECT * FROM evaluation_run
                    WHERE status = %s
                    ORDER BY start_time DESC
                """

        if limit is not None:
          query += f' LIMIT {limit}'

        async with conn.cursor(row_factory=class_row(EvaluationRun)) as cur:
          await cur.execute(query, (status.value,))
          return await cur.fetchall()

    except Exception as e:
      self._handle_db_error(e, f'list_by_status({status})')

  async def get_running_runs(self) -> List[EvaluationRun]:
    """
    Get all currently running evaluation runs.

    Returns:
        List of running evaluation runs
    """
    return await self.list_by_status(EvaluationStatus.RUNNING)

  async def count_by_status(self, status: EvaluationStatus) -> int:
    """
    Count evaluation runs by status.

    Args:
        status: Status to count

    Returns:
        Number of runs with the given status
    """
    try:
      async with self._get_connection() as conn:
        query = """
                    SELECT COUNT(*) FROM evaluation_run
                    WHERE status = %s
                """

        async with conn.cursor() as cur:
          await cur.execute(query, (status.value,))
          result = await cur.fetchone()
          return result[0] if result else 0

    except Exception as e:
      self._handle_db_error(e, f'count_by_status({status})')
