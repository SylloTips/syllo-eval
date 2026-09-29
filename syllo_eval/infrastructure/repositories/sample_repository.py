"""
Repository for Sample entity operations.
"""

from collections.abc import Sequence
from typing import Dict, Any, List, Optional
from uuid import UUID

from syllo_eval.model import Sample
from syllo_eval.infrastructure.repositories.base import BaseRepository
from psycopg.rows import class_row


class SampleRepository(BaseRepository[Sample]):
  """Repository for managing Sample entities."""

  @property
  def table_name(self) -> str:
    return 'sample'

  @property
  def model_class(self) -> type[Sample]:
    return Sample

  def _get_insert_fields(self, entity: Sample) -> Dict[str, Any]:
    """Get fields for inserting a new sample."""
    fields: Dict[str, Any] = {
      'dataset_id': entity.dataset_id,
      'input_prompt': entity.input_prompt,
    }

    if entity.id is not None:
      fields['id'] = entity.id
    if entity.ground_truth_output is not None:
      fields['ground_truth_output'] = entity.ground_truth_output

    return fields

  def _get_update_fields(self, entity: Sample) -> Dict[str, Any]:
    """Get fields for updating a sample."""
    fields: Dict[str, Any] = {
      'input_prompt': entity.input_prompt,
    }

    if entity.ground_truth_output is not None:
      fields['ground_truth_output'] = entity.ground_truth_output

    return fields

  async def list_by_dataset(self, dataset_id: UUID, limit: Optional[int] = None, offset: int = 0) -> List[Sample]:
    """
    List all samples in a dataset with optional pagination.

    Args:
        dataset_id: Dataset ID
        limit: Maximum number of samples to return
        offset: Number of samples to skip

    Returns:
        List of samples
    """
    try:
      async with self._get_connection() as conn:
        query = """
                    SELECT * FROM sample
                    WHERE dataset_id = %s
                    ORDER BY created_at ASC
                """

        if limit is not None:
          query += f' LIMIT {limit} OFFSET {offset}'

        async with conn.cursor(row_factory=class_row(Sample)) as cur:
          await cur.execute(query, (dataset_id,))
          return await cur.fetchall()

    except Exception as e:
      self._handle_db_error(e, f'list_by_dataset({dataset_id})')

  async def count_by_dataset(self, dataset_id: UUID) -> int:
    """
    Count samples in a dataset.

    Args:
        dataset_id: Dataset ID

    Returns:
        Number of samples in the dataset
    """
    try:
      async with self._get_connection() as conn:
        query = """
                    SELECT COUNT(*) FROM sample
                    WHERE dataset_id = %s
                """

        async with conn.cursor() as cur:
          await cur.execute(query, (dataset_id,))
          result = await cur.fetchone()
          return result[0] if result else 0

    except Exception as e:
      self._handle_db_error(e, f'count_by_dataset({dataset_id})')

  async def list_by_ids(self, sample_ids: Sequence[UUID]) -> list[Sample]:
    if not sample_ids:
      return []

    try:
      async with self._get_connection() as conn:
        placeholders = ', '.join(['%s'] * len(sample_ids))
        query = f"""
                    SELECT * FROM sample
                    WHERE id IN ({placeholders})
                """

        async with conn.cursor(row_factory=class_row(Sample)) as cur:
          await cur.execute(query, tuple(sample_ids))
          samples = await cur.fetchall()

      samples_by_id = {sample.id: sample for sample in samples}
      return [samples_by_id[sample_id] for sample_id in sample_ids if sample_id in samples_by_id]

    except Exception as e:
      self._handle_db_error(e, f'list_by_ids({sample_ids})')

  async def bulk_create(self, samples: List[Sample]) -> List[Sample]:
    """
    Create multiple samples in a single transaction.

    Args:
        samples: List of samples to create

    Returns:
        List of created samples with IDs

    Raises:
        DuplicateError: If any sample violates unique constraints
        IntegrityError: If any sample violates foreign key constraints
    """
    if not samples:
      return []

    try:
      if self._connection is not None:
        conn = self._connection
        created_samples = []
        for sample in samples:
          fields = self._get_insert_fields(sample)
          columns = ', '.join(fields.keys())
          placeholders = ', '.join(['%s'] * len(fields))
          values = tuple(fields.values())

          query = f"""
                        INSERT INTO {self.table_name} ({columns})
                        VALUES ({placeholders})
                        RETURNING *
                    """

          async with conn.cursor(row_factory=class_row(Sample)) as cur:
            await cur.execute(query, values)
            result = await cur.fetchone()
            if result:
              created_samples.append(result)

        return created_samples

      async with self.db_manager.transaction() as conn:
        created_samples = []

        for sample in samples:
          fields = self._get_insert_fields(sample)
          columns = ', '.join(fields.keys())
          placeholders = ', '.join(['%s'] * len(fields))
          values = tuple(fields.values())

          query = f"""
                        INSERT INTO {self.table_name} ({columns})
                        VALUES ({placeholders})
                        RETURNING *
                    """

          async with conn.cursor(row_factory=class_row(Sample)) as cur:
            await cur.execute(query, values)
            result = await cur.fetchone()
            if result:
              created_samples.append(result)

        return created_samples

    except Exception as e:
      self._handle_db_error(e, 'bulk_create')
