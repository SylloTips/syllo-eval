"""Repository for metric computation entity operations."""

import json
from typing import Any, Dict, List
from uuid import UUID

from psycopg import AsyncConnection
from psycopg.rows import class_row

from syllo_eval.infrastructure.repositories.base import BaseRepository
from syllo_eval.model import MetricComputation, MetricComputationStatus, MetricTargetingMode

_MEMBERSHIP_CTE = """
WITH metric_computation_membership AS (
  SELECT
    ssmc.metric_computation_id,
    'SINGLE'::metric_targeting_mode AS targeting_mode,
    ssmc.span_id
  FROM single_span_metric_computation ssmc
  UNION ALL
  SELECT
    gsmcs.metric_computation_id,
    'GROUP'::metric_targeting_mode AS targeting_mode,
    gsmcs.span_id
  FROM group_span_metric_computation_span gsmcs
)
"""


class MetricComputationRepository(BaseRepository[MetricComputation]):
  """Repository for managing metric computations and their span memberships."""

  @property
  def table_name(self) -> str:
    return 'span_metric_computation'

  @property
  def model_class(self) -> type[MetricComputation]:
    return MetricComputation

  def _get_insert_fields(self, entity: MetricComputation) -> Dict[str, Any]:
    fields: Dict[str, Any] = {
      'evaluation_run_sample_id': entity.evaluation_run_sample_id,
      'metric': entity.metric,
      'ground_truth_id': entity.ground_truth_id,
      'targeting_mode': entity.targeting_mode,
      'target_span_type': entity.target_span_type,
      'score': entity.score,
      'status': entity.status,
    }

    if entity.id is not None:
      fields['id'] = entity.id
    if entity.reasoning is not None:
      fields['reasoning'] = entity.reasoning
    if entity.metadata is not None:
      fields['metadata'] = json.dumps(entity.metadata)
    if entity.error_message is not None:
      fields['error_message'] = entity.error_message
    if entity.raw_output is not None:
      fields['raw_output'] = json.dumps(entity.raw_output)

    return fields

  def _get_update_fields(self, entity: MetricComputation) -> Dict[str, Any]:
    fields: Dict[str, Any] = {
      'score': entity.score,
      'status': entity.status,
      'ground_truth_id': entity.ground_truth_id,
      'targeting_mode': entity.targeting_mode,
      'target_span_type': entity.target_span_type,
      'error_message': entity.error_message,
    }

    if entity.reasoning is not None:
      fields['reasoning'] = entity.reasoning
    if entity.metadata is not None:
      fields['metadata'] = json.dumps(entity.metadata)
    if entity.raw_output is not None:
      fields['raw_output'] = json.dumps(entity.raw_output)

    return fields

  async def create(self, entity: MetricComputation) -> MetricComputation:
    """Create one metric computation header plus the appropriate child rows."""
    try:
      async with self._get_connection() as conn:
        await self._create_with_connection(conn, entity)
        created = await self._get_by_id_with_connection(conn, entity.id)
        if created is None:
          raise ValueError(f'Created metric computation {entity.id} could not be reloaded')
        return created

    except Exception as e:
      self._handle_db_error(e, 'create')

  async def get_by_id(self, entity_id: UUID | str) -> MetricComputation | None:
    try:
      async with self._get_connection() as conn:
        return await self._get_by_id_with_connection(conn, entity_id)

    except Exception as e:
      self._handle_db_error(e, f'get_by_id({entity_id})')

  async def list_by_evaluation_run_sample(self, evaluation_run_sample_id: UUID) -> List[MetricComputation]:
    """List all computations for an evaluation run sample."""
    try:
      async with self._get_connection() as conn:
        query = (
          _MEMBERSHIP_CTE
          + """
            SELECT
              smc.id,
              smc.evaluation_run_sample_id,
              smc.metric,
              smc.ground_truth_id,
              smc.targeting_mode,
              smc.target_span_type,
              COALESCE(
                ARRAY_AGG(mcm.span_id ORDER BY s.start_time, s.external_id)
                  FILTER (WHERE mcm.span_id IS NOT NULL),
                '{}'::text[]
              ) AS span_ids,
              smc.score,
              smc.status,
              smc.reasoning,
              smc.metadata,
              smc.error_message,
              smc.raw_output
            FROM span_metric_computation smc
            LEFT JOIN metric_computation_membership mcm ON mcm.metric_computation_id = smc.id
            LEFT JOIN span s ON s.external_id = mcm.span_id
            WHERE smc.evaluation_run_sample_id = %s
            GROUP BY
              smc.id,
              smc.evaluation_run_sample_id,
              smc.metric,
              smc.ground_truth_id,
              smc.targeting_mode,
              smc.target_span_type,
              smc.score,
              smc.status,
              smc.reasoning,
              smc.metadata,
              smc.error_message,
              smc.raw_output
            ORDER BY smc.metric, smc.target_span_type, smc.id
          """
        )

        async with conn.cursor(row_factory=class_row(MetricComputation)) as cur:
          await cur.execute(query, (evaluation_run_sample_id,))
          results = await cur.fetchall()
          normalized_results: list[MetricComputation] = []
          for result in results:
            normalized = self._normalize_metric_computation(result)
            if normalized is not None:
              normalized_results.append(normalized)
          return normalized_results

    except Exception as e:
      self._handle_db_error(e, f'list_by_evaluation_run_sample({evaluation_run_sample_id})')

  async def list_by_evaluation_run(self, evaluation_run_id: UUID) -> List[MetricComputation]:
    """List all computations for an evaluation run."""
    try:
      async with self._get_connection() as conn:
        query = (
          _MEMBERSHIP_CTE
          + """
            SELECT
              smc.id,
              smc.evaluation_run_sample_id,
              smc.metric,
              smc.ground_truth_id,
              smc.targeting_mode,
              smc.target_span_type,
              COALESCE(
                ARRAY_AGG(mcm.span_id ORDER BY s.start_time, s.external_id)
                  FILTER (WHERE mcm.span_id IS NOT NULL),
                '{}'::text[]
              ) AS span_ids,
              smc.score,
              smc.status,
              smc.reasoning,
              smc.metadata,
              smc.error_message,
              smc.raw_output
            FROM span_metric_computation smc
            JOIN evaluation_run_sample ers ON ers.id = smc.evaluation_run_sample_id
            LEFT JOIN metric_computation_membership mcm ON mcm.metric_computation_id = smc.id
            LEFT JOIN span s ON s.external_id = mcm.span_id
            WHERE ers.evaluation_run_id = %s
            GROUP BY
              smc.id,
              smc.evaluation_run_sample_id,
              smc.metric,
              smc.ground_truth_id,
              smc.targeting_mode,
              smc.target_span_type,
              smc.score,
              smc.status,
              smc.reasoning,
              smc.metadata,
              smc.error_message,
              smc.raw_output
            ORDER BY MIN(ers.created_at), smc.metric, smc.target_span_type, smc.id
          """
        )

        async with conn.cursor(row_factory=class_row(MetricComputation)) as cur:
          await cur.execute(query, (evaluation_run_id,))
          results = await cur.fetchall()
          normalized_results: list[MetricComputation] = []
          for result in results:
            normalized = self._normalize_metric_computation(result)
            if normalized is not None:
              normalized_results.append(normalized)
          return normalized_results

    except Exception as e:
      self._handle_db_error(e, f'list_by_evaluation_run({evaluation_run_id})')

  async def list_by_span(self, span_id: str) -> List[MetricComputation]:
    """List all computations that include the given span."""
    try:
      async with self._get_connection() as conn:
        query = (
          _MEMBERSHIP_CTE
          + """
            SELECT
              smc.id,
              smc.evaluation_run_sample_id,
              smc.metric,
              smc.ground_truth_id,
              smc.targeting_mode,
              smc.target_span_type,
              ARRAY_AGG(mcm.span_id ORDER BY s.start_time, s.external_id) AS span_ids,
              smc.score,
              smc.status,
              smc.reasoning,
              smc.metadata,
              smc.error_message,
              smc.raw_output
            FROM span_metric_computation smc
            JOIN metric_computation_membership mcm ON mcm.metric_computation_id = smc.id
            JOIN span s ON s.external_id = mcm.span_id
            WHERE EXISTS (
              SELECT 1
              FROM metric_computation_membership filtered_mcm
              WHERE filtered_mcm.metric_computation_id = smc.id
                AND filtered_mcm.span_id = %s
            )
            GROUP BY
              smc.id,
              smc.evaluation_run_sample_id,
              smc.metric,
              smc.ground_truth_id,
              smc.targeting_mode,
              smc.target_span_type,
              smc.score,
              smc.status,
              smc.reasoning,
              smc.metadata,
              smc.error_message,
              smc.raw_output
            ORDER BY smc.metric, smc.id
          """
        )

        async with conn.cursor(row_factory=class_row(MetricComputation)) as cur:
          await cur.execute(query, (span_id,))
          results = await cur.fetchall()
          normalized_results: list[MetricComputation] = []
          for result in results:
            normalized = self._normalize_metric_computation(result)
            if normalized is not None:
              normalized_results.append(normalized)
          return normalized_results

    except Exception as e:
      self._handle_db_error(e, f'list_by_span({span_id})')

  async def get_aggregated_scores_by_evaluation_run(self, evaluation_run_id: UUID) -> List[Dict[str, Any]]:
    """Get aggregated metric scores for an evaluation run."""
    try:
      async with self._get_connection() as conn:
        query = """
                    SELECT
                        smc.metric,
                        AVG(smc.score) as avg_score,
                        MIN(smc.score) as min_score,
                        MAX(smc.score) as max_score,
                        STDDEV(smc.score) as stddev_score,
                        COUNT(*) as count
                    FROM span_metric_computation smc
                    JOIN evaluation_run_sample ers ON smc.evaluation_run_sample_id = ers.id
                    WHERE ers.evaluation_run_id = %s
                      AND smc.status = 'COMPLETED'
                      AND smc.score IS NOT NULL
                    GROUP BY smc.metric
                    ORDER BY smc.metric
                """

        async with conn.cursor() as cur:
          await cur.execute(query, (evaluation_run_id,))

          if cur.description is None:
            return []

          columns = [desc[0] for desc in cur.description]
          results = await cur.fetchall()

          return [dict(zip(columns, row)) for row in results]

    except Exception as e:
      self._handle_db_error(e, f'get_aggregated_scores_by_evaluation_run({evaluation_run_id})')

  async def get_scores_by_metric_and_evaluation_run(self, evaluation_run_id: UUID, metric: str) -> List[float]:
    """Get all scores for a specific metric in an evaluation run."""
    try:
      async with self._get_connection() as conn:
        query = """
                    SELECT smc.score
                    FROM span_metric_computation smc
                    JOIN evaluation_run_sample ers ON smc.evaluation_run_sample_id = ers.id
                    WHERE ers.evaluation_run_id = %s AND smc.metric = %s
                      AND smc.status = 'COMPLETED'
                      AND smc.score IS NOT NULL
                    ORDER BY smc.score DESC
                """

        async with conn.cursor() as cur:
          await cur.execute(query, (evaluation_run_id, metric))
          results = await cur.fetchall()
          return [row[0] for row in results]

    except Exception as e:
      self._handle_db_error(e, f'get_scores_by_metric_and_evaluation_run({evaluation_run_id}, {metric})')

  async def bulk_create(self, computations: List[MetricComputation]) -> List[MetricComputation]:
    """Create multiple computations in a single transaction."""
    if not computations:
      return []

    try:
      created_computations: list[MetricComputation] = []
      if self._connection is not None:
        for computation in computations:
          await self._create_with_connection(self._connection, computation)
        for computation in computations:
          created = await self._get_by_id_with_connection(self._connection, computation.id)
          if created is not None:
            created_computations.append(created)
        return created_computations

      async with self.db_manager.transaction() as conn:
        for computation in computations:
          await self._create_with_connection(conn, computation)
          created = await self._get_by_id_with_connection(conn, computation.id)
          if created is not None:
            created_computations.append(created)

      return created_computations

    except Exception as e:
      self._handle_db_error(e, 'bulk_create')

  async def count_by_evaluation_run(self, evaluation_run_id: UUID) -> int:
    """Count computations for an evaluation run."""
    try:
      async with self._get_connection() as conn:
        query = """
                    SELECT COUNT(*)
                    FROM span_metric_computation smc
                    JOIN evaluation_run_sample ers ON smc.evaluation_run_sample_id = ers.id
                    WHERE ers.evaluation_run_id = %s
                """

        async with conn.cursor() as cur:
          await cur.execute(query, (evaluation_run_id,))
          result = await cur.fetchone()
          return result[0] if result else 0

    except Exception as e:
      self._handle_db_error(e, f'count_by_evaluation_run({evaluation_run_id})')

  async def _create_with_connection(self, conn: AsyncConnection[Any], entity: MetricComputation) -> None:
    self._validate_targeting_mode(entity)

    fields = self._get_insert_fields(entity)
    columns = ', '.join(fields.keys())
    placeholders = ', '.join(['%s'] * len(fields))
    values = tuple(fields.values())

    query = f"""
                INSERT INTO {self.table_name} ({columns})
                VALUES ({placeholders})
            """

    async with conn.cursor() as cur:
      await cur.execute(query, values)

      if entity.targeting_mode == MetricTargetingMode.SINGLE:
        if not entity.span_ids:
          return
        await cur.execute(
          """
            INSERT INTO single_span_metric_computation (metric_computation_id, span_id)
            VALUES (%s, %s)
          """,
          (entity.id, entity.span_ids[0]),
        )
      else:
        if not entity.span_ids:
          return
        await cur.executemany(
          """
            INSERT INTO group_span_metric_computation_span (metric_computation_id, span_id)
            VALUES (%s, %s)
          """,
          [(entity.id, span_id) for span_id in entity.span_ids],
        )

  @staticmethod
  def _normalize_metric_computation(computation: MetricComputation | None) -> MetricComputation | None:
    if computation is None:
      return None

    if computation.metadata and isinstance(computation.metadata, str):
      computation.metadata = json.loads(computation.metadata)
    if computation.raw_output and isinstance(computation.raw_output, str):
      computation.raw_output = json.loads(computation.raw_output)

    return computation

  @staticmethod
  def _validate_targeting_mode(entity: MetricComputation) -> None:
    if not entity.span_ids:
      if entity.status == MetricComputationStatus.SKIPPED:
        return
      raise ValueError('Metric computation must reference at least one span.')

    if entity.targeting_mode == MetricTargetingMode.SINGLE and len(entity.span_ids) != 1:
      raise ValueError('Single-target metric computations must reference exactly one span.')

  async def _get_by_id_with_connection(
    self,
    conn: AsyncConnection[Any],
    entity_id: UUID | str,
  ) -> MetricComputation | None:
    query = (
      _MEMBERSHIP_CTE
      + """
        SELECT
          smc.id,
          smc.evaluation_run_sample_id,
          smc.metric,
          smc.ground_truth_id,
          smc.targeting_mode,
          smc.target_span_type,
          COALESCE(
            ARRAY_AGG(mcm.span_id ORDER BY s.start_time, s.external_id)
              FILTER (WHERE mcm.span_id IS NOT NULL),
            '{}'::text[]
          ) AS span_ids,
          smc.score,
          smc.status,
          smc.reasoning,
          smc.metadata,
          smc.error_message,
          smc.raw_output
        FROM span_metric_computation smc
        LEFT JOIN metric_computation_membership mcm ON mcm.metric_computation_id = smc.id
        LEFT JOIN span s ON s.external_id = mcm.span_id
        WHERE smc.id = %s
        GROUP BY
          smc.id,
          smc.evaluation_run_sample_id,
          smc.metric,
          smc.ground_truth_id,
          smc.targeting_mode,
          smc.target_span_type,
          smc.score,
          smc.status,
          smc.reasoning,
          smc.metadata,
          smc.error_message,
          smc.raw_output
      """
    )

    async with conn.cursor(row_factory=class_row(MetricComputation)) as cur:
      await cur.execute(query, (entity_id,))
      result = await cur.fetchone()
      return self._normalize_metric_computation(result)
