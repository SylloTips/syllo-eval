"""
Repository for Metric entity operations.
"""

from typing import Dict, Any

from psycopg.rows import class_row

from syllo_eval.model import Metric
from syllo_eval.infrastructure.repositories.base import BaseRepository


class MetricRepository(BaseRepository[Metric]):
  """Repository for managing Metric entities."""

  @property
  def table_name(self) -> str:
    return 'metric'

  @property
  def model_class(self) -> type[Metric]:
    return Metric

  @property
  def id_column(self) -> str:
    return 'name'

  def _get_insert_fields(self, entity: Metric) -> Dict[str, Any]:
    fields: Dict[str, Any] = {
      'name': entity.name,
      'ground_truth_keys': entity.ground_truth_keys,
    }
    if entity.description is not None:
      fields['description'] = entity.description
    return fields

  def _get_update_fields(self, entity: Metric) -> Dict[str, Any]:
    fields: Dict[str, Any] = {'ground_truth_keys': entity.ground_truth_keys}
    if entity.description is not None:
      fields['description'] = entity.description
    return fields

  async def upsert_from_registry(self, entity: Metric) -> Metric:
    """Create or update a registry-owned metric definition atomically."""
    try:
      async with self._get_connection() as conn:
        query = """
                    INSERT INTO metric (name, description, ground_truth_keys)
                    VALUES (%s, %s, %s)
                    ON CONFLICT (name) DO UPDATE
                    SET
                      description = COALESCE(EXCLUDED.description, metric.description),
                      ground_truth_keys = EXCLUDED.ground_truth_keys
                    RETURNING *
                """

        async with conn.cursor(row_factory=class_row(self.model_class)) as cur:
          await cur.execute(query, (entity.name, entity.description, entity.ground_truth_keys))
          result = await cur.fetchone()
          if result is None:
            raise RuntimeError(f'Upserted metric {entity.name!r} could not be reloaded')
          return result

    except Exception as e:
      self._handle_db_error(e, f'upsert_from_registry({entity.name})')
