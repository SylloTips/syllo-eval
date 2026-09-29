"""
Repository for Dataset entity operations.
"""

from typing import Dict, Any, List

from psycopg.rows import class_row

from syllo_eval.model import Dataset, DatasetSummary
from syllo_eval.infrastructure.repositories.base import BaseRepository


class DatasetRepository(BaseRepository[Dataset]):
  """Repository for managing Dataset entities."""

  @property
  def table_name(self) -> str:
    return 'dataset'

  @property
  def model_class(self) -> type[Dataset]:
    return Dataset

  def _get_insert_fields(self, entity: Dataset) -> Dict[str, Any]:
    fields: Dict[str, Any] = {'name': entity.name}
    if entity.id is not None:
      fields['id'] = entity.id
    return fields

  def _get_update_fields(self, entity: Dataset) -> Dict[str, Any]:
    return {'name': entity.name}

  async def get_by_name(self, name: str) -> Dataset | None:
    """Get dataset by name."""
    try:
      async with self._get_connection() as conn:
        query = 'SELECT * FROM dataset WHERE name = %s'
        async with conn.cursor(row_factory=class_row(Dataset)) as cur:
          await cur.execute(query, (name,))
          return await cur.fetchone()
    except Exception as e:
      self._handle_db_error(e, f'get_by_name({name})')

  async def list_with_sample_counts(self, limit: int, offset: int = 0) -> List[DatasetSummary]:
    """List datasets in reverse chronological order with their sample counts."""
    try:
      async with self._get_connection() as conn:
        query = (
          'SELECT d.id, d.name, d.created_at, COUNT(s.id)::int AS sample_count '
          'FROM dataset d LEFT JOIN sample s ON s.dataset_id = d.id '
          'GROUP BY d.id, d.name, d.created_at '
          'ORDER BY d.created_at DESC '
          'LIMIT %s OFFSET %s'
        )
        async with conn.cursor(row_factory=class_row(DatasetSummary)) as cur:
          await cur.execute(query, (limit, offset))
          return await cur.fetchall()
    except Exception as e:
      self._handle_db_error(e, f'list_with_sample_counts({limit}, {offset})')

  async def count_all(self) -> int:
    """Count all datasets."""
    try:
      async with self._get_connection() as conn:
        async with conn.cursor() as cur:
          await cur.execute('SELECT COUNT(*) FROM dataset')
          result = await cur.fetchone()
          return result[0] if result else 0
    except Exception as e:
      self._handle_db_error(e, 'count_all()')
