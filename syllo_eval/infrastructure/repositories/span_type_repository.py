"""
Repository for SpanType entity operations.
"""

from typing import Dict, Any

from psycopg.rows import class_row

from syllo_eval.model import SpanType
from syllo_eval.infrastructure.repositories.base import BaseRepository


class SpanTypeRepository(BaseRepository[SpanType]):
  """Repository for managing SpanType entities."""

  @property
  def table_name(self) -> str:
    return 'span_type'

  @property
  def model_class(self) -> type[SpanType]:
    return SpanType

  @property
  def id_column(self) -> str:
    return 'name'

  def _get_insert_fields(self, entity: SpanType) -> Dict[str, Any]:
    fields: Dict[str, Any] = {'name': entity.name}
    if entity.description is not None:
      fields['description'] = entity.description
    return fields

  def _get_update_fields(self, entity: SpanType) -> Dict[str, Any]:
    fields: Dict[str, Any] = {}
    if entity.description is not None:
      fields['description'] = entity.description
    return fields

  async def upsert_from_registry(self, entity: SpanType) -> SpanType:
    """Create a registry-declared span type, preserving existing descriptions."""
    try:
      async with self._get_connection() as conn:
        query = """
                    INSERT INTO span_type (name, description)
                    VALUES (%s, %s)
                    ON CONFLICT (name) DO UPDATE
                    SET description = COALESCE(EXCLUDED.description, span_type.description)
                    RETURNING *
                """

        async with conn.cursor(row_factory=class_row(self.model_class)) as cur:
          await cur.execute(query, (entity.name, entity.description))
          result = await cur.fetchone()
          if result is None:
            raise RuntimeError(f'Upserted span_type {entity.name!r} could not be reloaded')
          return result

    except Exception as e:
      self._handle_db_error(e, f'upsert_from_registry({entity.name})')
