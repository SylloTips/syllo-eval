from collections.abc import Sequence
from uuid import UUID

from syllo_eval.infrastructure.database import DatabaseManager
from syllo_eval.infrastructure.repositories.base import BaseRepository


class EvaluationRunPlanSampleRepository(BaseRepository[UUID]):
  def __init__(self, db_manager: DatabaseManager, connection=None):
    super().__init__(db_manager, connection)

  @property
  def table_name(self) -> str:
    return 'evaluation_run_plan_sample'

  @property
  def model_class(self) -> type[UUID]:
    return UUID

  def _get_insert_fields(self, entity: UUID):
    raise NotImplementedError

  def _get_update_fields(self, entity: UUID):
    raise NotImplementedError

  async def create_many(self, evaluation_run_id: UUID, sample_ids: Sequence[UUID]) -> None:
    if not sample_ids:
      return

    try:
      async with self._get_connection() as conn:
        async with conn.cursor() as cur:
          await cur.executemany(
            """
            INSERT INTO evaluation_run_plan_sample (evaluation_run_id, sample_id)
            VALUES (%s, %s)
            ON CONFLICT DO NOTHING
            """,
            [(evaluation_run_id, sample_id) for sample_id in sample_ids],
          )
    except Exception as e:
      self._handle_db_error(e, f'create_many({evaluation_run_id})')

  async def list_sample_ids(self, evaluation_run_id: UUID) -> list[UUID]:
    try:
      async with self._get_connection() as conn:
        async with conn.cursor() as cur:
          await cur.execute(
            """
            SELECT sample_id
            FROM evaluation_run_plan_sample
            WHERE evaluation_run_id = %s
            ORDER BY created_at ASC, sample_id ASC
            """,
            (evaluation_run_id,),
          )
          return [row[0] for row in await cur.fetchall()]
    except Exception as e:
      self._handle_db_error(e, f'list_sample_ids({evaluation_run_id})')
