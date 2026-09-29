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
      'requires_ground_truth': entity.requires_ground_truth,
    }
    if entity.description is not None:
      fields['description'] = entity.description
    if entity.ground_truth_key is not None:
      fields['ground_truth_key'] = entity.ground_truth_key
    return fields

  def _get_update_fields(self, entity: Metric) -> Dict[str, Any]:
    fields: Dict[str, Any] = {
      'requires_ground_truth': entity.requires_ground_truth,
      'ground_truth_key': entity.ground_truth_key,
    }
    if entity.description is not None:
      fields['description'] = entity.description
    return fields

  async def upsert_from_registry(self, entity: Metric) -> Metric:
    """Create or update a registry-owned metric definition atomically."""
    try:
      async with self._get_connection() as conn:
        query = """
                    INSERT INTO metric (name, description, requires_ground_truth, ground_truth_key)
                    VALUES (%s, %s, %s, %s)
                    ON CONFLICT (name) DO UPDATE
                    SET
                      description = COALESCE(EXCLUDED.description, metric.description),
                      requires_ground_truth = EXCLUDED.requires_ground_truth,
                      ground_truth_key = EXCLUDED.ground_truth_key
                    RETURNING *
                """

        async with conn.cursor(row_factory=class_row(self.model_class)) as cur:
          await cur.execute(
            query, (entity.name, entity.description, entity.requires_ground_truth, entity.ground_truth_key)
          )
          result = await cur.fetchone()
          if result is None:
            raise RuntimeError(f'Upserted metric {entity.name!r} could not be reloaded')
          return result

    except Exception as e:
      self._handle_db_error(e, f'upsert_from_registry({entity.name})')
