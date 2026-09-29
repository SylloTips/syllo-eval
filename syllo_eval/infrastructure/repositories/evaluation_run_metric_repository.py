from collections.abc import Sequence
from uuid import UUID

from syllo_eval.infrastructure.database import DatabaseManager
from syllo_eval.infrastructure.repositories.base import BaseRepository


class EvaluationRunMetricRepository(BaseRepository[str]):
  def __init__(self, db_manager: DatabaseManager, connection=None):
    super().__init__(db_manager, connection)

  @property
  def table_name(self) -> str:
    return 'evaluation_run_metric'

  @property
  def model_class(self) -> type[str]:
    return str

  def _get_insert_fields(self, entity: str):
    raise NotImplementedError

  def _get_update_fields(self, entity: str):
    raise NotImplementedError

  async def create_many(self, evaluation_run_id: UUID, metric_names: Sequence[str]) -> None:
    if not metric_names:
      return

    try:
      async with self._get_connection() as conn:
        async with conn.cursor() as cur:
          await cur.executemany(
            """
            INSERT INTO evaluation_run_metric (evaluation_run_id, metric)
            VALUES (%s, %s)
            ON CONFLICT DO NOTHING
            """,
            [(evaluation_run_id, metric_name) for metric_name in metric_names],
          )
    except Exception as e:
      self._handle_db_error(e, f'create_many({evaluation_run_id})')

  async def list_metrics(self, evaluation_run_id: UUID) -> list[str]:
    try:
      async with self._get_connection() as conn:
        async with conn.cursor() as cur:
          await cur.execute(
            """
            SELECT metric
            FROM evaluation_run_metric
            WHERE evaluation_run_id = %s
            ORDER BY created_at ASC, metric ASC
            """,
            (evaluation_run_id,),
          )
          return [row[0] for row in await cur.fetchall()]
    except Exception as e:
      self._handle_db_error(e, f'list_metrics({evaluation_run_id})')
