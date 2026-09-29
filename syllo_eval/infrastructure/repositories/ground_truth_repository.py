"""
Repository for GroundTruth entity operations.
"""

import json
from typing import Dict, Any, List, Optional
from uuid import UUID

from syllo_eval.model import GroundTruth
from syllo_eval.infrastructure.repositories.base import BaseRepository
from psycopg.rows import class_row


class GroundTruthRepository(BaseRepository[GroundTruth]):
  """Repository for managing GroundTruth entities."""

  @property
  def table_name(self) -> str:
    return 'ground_truth'

  @property
  def model_class(self) -> type[GroundTruth]:
    return GroundTruth

  def _get_insert_fields(self, entity: GroundTruth) -> Dict[str, Any]:
    """Get fields for inserting a new ground truth."""
    fields: Dict[str, Any] = {
      'sample_id': entity.sample_id,
      'key': entity.key,
      'ground_truth_value': json.dumps(entity.ground_truth_value),
    }

    if entity.id is not None:
      fields['id'] = entity.id

    return fields

  def _get_update_fields(self, entity: GroundTruth) -> Dict[str, Any]:
    """Get fields for updating a ground truth."""
    return {
      'ground_truth_value': json.dumps(entity.ground_truth_value),
    }

  async def get_by_sample_and_key(self, sample_id: UUID, key: str) -> Optional[GroundTruth]:
    """
    Get ground truth for a specific sample and ground-truth key.

    Args:
        sample_id: Sample ID
        key: Ground-truth key

    Returns:
        GroundTruth if found, None otherwise
    """
    try:
      async with self._get_connection() as conn:
        query = """
                    SELECT * FROM ground_truth
                    WHERE sample_id = %s AND key = %s
                """

        async with conn.cursor(row_factory=class_row(GroundTruth)) as cur:
          await cur.execute(query, (sample_id, key))
          result = await cur.fetchone()

          if result and isinstance(result.ground_truth_value, str):
            result.ground_truth_value = json.loads(result.ground_truth_value)

          return result

    except Exception as e:
      self._handle_db_error(e, f'get_by_sample_and_key({sample_id}, {key})')

  async def list_by_sample(self, sample_id: UUID) -> List[GroundTruth]:
    """
    List all ground truths for a sample.

    Args:
        sample_id: Sample ID

    Returns:
        List of ground truths
    """
    try:
      async with self._get_connection() as conn:
        query = """
                    SELECT * FROM ground_truth
                    WHERE sample_id = %s
                    ORDER BY key
                """

        async with conn.cursor(row_factory=class_row(GroundTruth)) as cur:
          await cur.execute(query, (sample_id,))
          results = await cur.fetchall()

          for result in results:
            if isinstance(result.ground_truth_value, str):
              result.ground_truth_value = json.loads(result.ground_truth_value)

          return results

    except Exception as e:
      self._handle_db_error(e, f'list_by_sample({sample_id})')

  async def list_by_key(self, key: str) -> List[GroundTruth]:
    """
    List all ground truths for a ground-truth key.

    Args:
        key: Ground-truth key

    Returns:
        List of ground truths
    """
    try:
      async with self._get_connection() as conn:
        query = """
                    SELECT * FROM ground_truth
                    WHERE key = %s
                    ORDER BY created_at
                """

        async with conn.cursor(row_factory=class_row(GroundTruth)) as cur:
          await cur.execute(query, (key,))
          results = await cur.fetchall()

          for result in results:
            if isinstance(result.ground_truth_value, str):
              result.ground_truth_value = json.loads(result.ground_truth_value)

          return results

    except Exception as e:
      self._handle_db_error(e, f'list_by_key({key})')

  async def bulk_create(self, ground_truths: List[GroundTruth]) -> List[GroundTruth]:
    """
    Create multiple ground truths in a single transaction.

    Args:
        ground_truths: List of ground truths to create

    Returns:
        List of created ground truths

    Raises:
        DuplicateError: If any ground truth violates unique constraints
        IntegrityError: If any ground truth violates foreign key constraints
    """
    if not ground_truths:
      return []

    try:
      if self._connection is not None:
        conn = self._connection
        created_gts = []

        for gt in ground_truths:
          fields = self._get_insert_fields(gt)
          columns = ', '.join(fields.keys())
          placeholders = ', '.join(['%s'] * len(fields))
          values = tuple(fields.values())

          query = f"""
                        INSERT INTO {self.table_name} ({columns})
                        VALUES ({placeholders})
                        RETURNING *
                    """

          async with conn.cursor(row_factory=class_row(GroundTruth)) as cur:
            await cur.execute(query, values)
            result = await cur.fetchone()
            if result:
              if isinstance(result.ground_truth_value, str):
                result.ground_truth_value = json.loads(result.ground_truth_value)
              created_gts.append(result)

        return created_gts

      async with self.db_manager.transaction() as conn:
        created_gts = []

        for gt in ground_truths:
          fields = self._get_insert_fields(gt)
          columns = ', '.join(fields.keys())
          placeholders = ', '.join(['%s'] * len(fields))
          values = tuple(fields.values())

          query = f"""
                        INSERT INTO {self.table_name} ({columns})
                        VALUES ({placeholders})
                        RETURNING *
                    """

          async with conn.cursor(row_factory=class_row(GroundTruth)) as cur:
            await cur.execute(query, values)
            result = await cur.fetchone()
            if result:
              if isinstance(result.ground_truth_value, str):
                result.ground_truth_value = json.loads(result.ground_truth_value)
              created_gts.append(result)

        return created_gts

    except Exception as e:
      self._handle_db_error(e, 'bulk_create')
