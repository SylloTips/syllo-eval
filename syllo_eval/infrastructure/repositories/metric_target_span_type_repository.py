"""
Repository for MetricTargetSpanType entity operations.
"""

from typing import Dict, Any, List

from psycopg.rows import class_row

from syllo_eval.model import MetricTargetSpanType
from syllo_eval.infrastructure.repositories.base import BaseRepository


class MetricTargetSpanTypeRepository(BaseRepository[MetricTargetSpanType]):
  """Repository for managing MetricTargetSpanType entities."""

  @property
  def table_name(self) -> str:
    return 'metric_target_span_type'

  @property
  def model_class(self) -> type[MetricTargetSpanType]:
    return MetricTargetSpanType

  def _get_insert_fields(self, entity: MetricTargetSpanType) -> Dict[str, Any]:
    """Get fields for inserting a new metric target span type."""
    fields: Dict[str, Any] = {
      'metric': entity.metric,
      'span_type': entity.span_type,
      'targeting_mode': entity.targeting_mode,
    }

    if entity.id is not None:
      fields['id'] = entity.id

    return fields

  def _get_update_fields(self, entity: MetricTargetSpanType) -> Dict[str, Any]:
    """Get fields for updating a metric target span type."""
    return {
      'targeting_mode': entity.targeting_mode,
    }

  async def upsert_from_registry(self, entity: MetricTargetSpanType) -> MetricTargetSpanType:
    """Create or update a registry-owned metric/span-type target atomically."""
    try:
      async with self._get_connection() as conn:
        query = """
                    INSERT INTO metric_target_span_type (id, metric, span_type, targeting_mode)
                    VALUES (%s, %s, %s, %s)
                    ON CONFLICT (metric, span_type) DO UPDATE
                    SET targeting_mode = EXCLUDED.targeting_mode
                    RETURNING *
                """

        async with conn.cursor(row_factory=class_row(MetricTargetSpanType)) as cur:
          await cur.execute(query, (entity.id, entity.metric, entity.span_type, entity.targeting_mode))
          result = await cur.fetchone()
          if result is None:
            raise RuntimeError(
              f'Upserted metric target mapping metric={entity.metric!r} span_type={entity.span_type!r} '
              'could not be reloaded'
            )
          return result

    except Exception as e:
      self._handle_db_error(e, f'upsert_from_registry({entity.metric}, {entity.span_type})')

  async def get_metrics_for_span_type(self, span_type: str) -> List[str]:
    """
    Get all metrics applicable to a given span type.

    Args:
        span_type: Span type name

    Returns:
        List of metric names
    """
    try:
      async with self._get_connection() as conn:
        query = """
                    SELECT metric FROM metric_target_span_type
                    WHERE span_type = %s
                    ORDER BY metric
                """

        async with conn.cursor() as cur:
          await cur.execute(query, (span_type,))
          results = await cur.fetchall()
          return [row[0] for row in results]

    except Exception as e:
      self._handle_db_error(e, f'get_metrics_for_span_type({span_type})')

  async def get_span_types_for_metric(self, metric: str) -> List[str]:
    """
    Get all span types that a metric can evaluate.

    Args:
        metric: Metric name

    Returns:
        List of span type names
    """
    try:
      async with self._get_connection() as conn:
        query = """
                    SELECT span_type FROM metric_target_span_type
                    WHERE metric = %s
                    ORDER BY span_type
                """

        async with conn.cursor() as cur:
          await cur.execute(query, (metric,))
          results = await cur.fetchall()
          return [row[0] for row in results]

    except Exception as e:
      self._handle_db_error(e, f'get_span_types_for_metric({metric})')

  async def exists_mapping(self, metric: str, span_type: str) -> bool:
    """
    Check if a metric-span_type mapping exists.

    Args:
        metric: Metric name
        span_type: Span type name

    Returns:
        True if mapping exists
    """
    try:
      async with self._get_connection() as conn:
        query = """
                    SELECT EXISTS(
                        SELECT 1 FROM metric_target_span_type
                        WHERE metric = %s AND span_type = %s
                    )
                """

        async with conn.cursor() as cur:
          await cur.execute(query, (metric, span_type))
          result = await cur.fetchone()
          return result[0] if result else False

    except Exception as e:
      self._handle_db_error(e, f'exists_mapping({metric}, {span_type})')

  async def get_by_metric_and_span_type(self, metric: str, span_type: str) -> MetricTargetSpanType | None:
    """Retrieve one mapping by metric name and span type."""
    try:
      async with self._get_connection() as conn:
        query = """
                    SELECT * FROM metric_target_span_type
                    WHERE metric = %s AND span_type = %s
                """

        async with conn.cursor(row_factory=class_row(MetricTargetSpanType)) as cur:
          await cur.execute(query, (metric, span_type))
          return await cur.fetchone()

    except Exception as e:
      self._handle_db_error(e, f'get_by_metric_and_span_type({metric}, {span_type})')

  async def delete_by_metric_and_span_type(self, metric: str, span_type: str) -> bool:
    """
    Delete a specific metric-span_type mapping.

    Args:
        metric: Metric name
        span_type: Span type name

    Returns:
        True if deleted, False if not found
    """
    try:
      async with self._get_connection() as conn:
        query = """
                    DELETE FROM metric_target_span_type
                    WHERE metric = %s AND span_type = %s
                """

        async with conn.cursor() as cur:
          await cur.execute(query, (metric, span_type))
          return cur.rowcount > 0

    except Exception as e:
      self._handle_db_error(e, f'delete_by_metric_and_span_type({metric}, {span_type})')
