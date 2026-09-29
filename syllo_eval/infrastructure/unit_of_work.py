"""
Unit of Work pattern implementation.
"""

import logging
from typing import Optional, Any, AsyncContextManager

from psycopg import AsyncConnection, AsyncTransaction

from syllo_eval.infrastructure.database import DatabaseManager
from syllo_eval.infrastructure.repositories.agent_repository import AgentRepository
from syllo_eval.infrastructure.repositories.evaluation_run_repository import (
  EvaluationRunRepository,
)
from syllo_eval.infrastructure.repositories.evaluation_run_metric_repository import (
  EvaluationRunMetricRepository,
)
from syllo_eval.infrastructure.repositories.evaluation_run_plan_sample_repository import (
  EvaluationRunPlanSampleRepository,
)
from syllo_eval.infrastructure.repositories.ground_truth_repository import GroundTruthRepository
from syllo_eval.infrastructure.repositories.metric_target_span_type_repository import (
  MetricTargetSpanTypeRepository,
)
from syllo_eval.infrastructure.repositories.sample_repository import SampleRepository
from syllo_eval.infrastructure.repositories.dataset_repository import DatasetRepository
from syllo_eval.infrastructure.repositories.evaluation_run_sample_repository import (
  EvaluationRunSampleRepository,
)
from syllo_eval.infrastructure.repositories.metric_repository import MetricRepository
from syllo_eval.infrastructure.repositories.span_type_repository import SpanTypeRepository
from syllo_eval.infrastructure.repositories.trace_repository import TraceRepository
from syllo_eval.infrastructure.repositories.span_metric_computation_repository import (
  MetricComputationRepository,
)
from syllo_eval.infrastructure.repositories.span_repository import SpanRepository

logger = logging.getLogger(__name__)


class UnitOfWork:
  """
  Unit of Work pattern for coordinating repository operations.

  Provides a single entry point for all repository access and
  manages transaction boundaries across multiple operations.

  Usage:
      async with UnitOfWork(db_manager) as uow:
          agent = await uow.agents.get_by_id(agent_id)
          samples = await uow.samples.list_by_dataset(dataset_id)
          # All operations in same transaction
  """

  def __init__(self, db_manager: DatabaseManager):
    """
    Initialize Unit of Work.

    Args:
        db_manager: Database manager for connection handling
    """
    self.db_manager = db_manager
    self._connection: Optional[AsyncConnection[Any]] = None

    self._agents: Optional[AgentRepository] = None
    self._datasets: Optional[DatasetRepository] = None
    self._evaluation_runs: Optional[EvaluationRunRepository] = None
    self._evaluation_run_metrics: Optional[EvaluationRunMetricRepository] = None
    self._evaluation_run_plan_samples: Optional[EvaluationRunPlanSampleRepository] = None
    self._samples: Optional[SampleRepository] = None
    self._metrics: Optional[MetricRepository] = None
    self._span_types: Optional[SpanTypeRepository] = None
    self._spans: Optional[SpanRepository] = None
    self._traces: Optional[TraceRepository] = None
    self._evaluation_run_samples: Optional[EvaluationRunSampleRepository] = None
    self._metric_target_span_types: Optional[MetricTargetSpanTypeRepository] = None
    self._ground_truths: Optional[GroundTruthRepository] = None
    self._metric_computations: Optional[MetricComputationRepository] = None

  @property
  def agents(self) -> AgentRepository:
    """Get Agent repository."""
    if self._agents is None:
      self._agents = AgentRepository(self.db_manager, self._connection)
    return self._agents

  @property
  def datasets(self) -> DatasetRepository:
    """Get Dataset repository."""
    if self._datasets is None:
      self._datasets = DatasetRepository(self.db_manager, self._connection)
    return self._datasets

  @property
  def evaluation_runs(self) -> EvaluationRunRepository:
    """Get EvaluationRun repository."""
    if self._evaluation_runs is None:
      self._evaluation_runs = EvaluationRunRepository(self.db_manager, self._connection)
    return self._evaluation_runs

  @property
  def evaluation_run_metrics(self) -> EvaluationRunMetricRepository:
    """Get EvaluationRunMetric repository."""
    if self._evaluation_run_metrics is None:
      self._evaluation_run_metrics = EvaluationRunMetricRepository(self.db_manager, self._connection)
    return self._evaluation_run_metrics

  @property
  def evaluation_run_plan_samples(self) -> EvaluationRunPlanSampleRepository:
    """Get EvaluationRunPlanSample repository."""
    if self._evaluation_run_plan_samples is None:
      self._evaluation_run_plan_samples = EvaluationRunPlanSampleRepository(self.db_manager, self._connection)
    return self._evaluation_run_plan_samples

  @property
  def samples(self) -> SampleRepository:
    """Get Sample repository."""
    if self._samples is None:
      self._samples = SampleRepository(self.db_manager, self._connection)
    return self._samples

  @property
  def metrics(self) -> MetricRepository:
    """Get Metric repository."""
    if self._metrics is None:
      self._metrics = MetricRepository(self.db_manager, self._connection)
    return self._metrics

  @property
  def span_types(self) -> SpanTypeRepository:
    """Get SpanType repository."""
    if self._span_types is None:
      self._span_types = SpanTypeRepository(self.db_manager, self._connection)
    return self._span_types

  @property
  def spans(self) -> SpanRepository:
    """Get Span repository."""
    if self._spans is None:
      self._spans = SpanRepository(self.db_manager, self._connection)
    return self._spans

  @property
  def traces(self) -> TraceRepository:
    """Get Trace repository."""
    if self._traces is None:
      self._traces = TraceRepository(self.db_manager, self._connection)
    return self._traces

  @property
  def evaluation_run_samples(self) -> EvaluationRunSampleRepository:
    """Get EvaluationRunSample repository."""
    if self._evaluation_run_samples is None:
      self._evaluation_run_samples = EvaluationRunSampleRepository(self.db_manager, self._connection)
    return self._evaluation_run_samples

  @property
  def metric_target_span_types(self) -> MetricTargetSpanTypeRepository:
    """Get MetricTargetSpanType repository."""
    if self._metric_target_span_types is None:
      self._metric_target_span_types = MetricTargetSpanTypeRepository(self.db_manager, self._connection)
    return self._metric_target_span_types

  @property
  def ground_truths(self) -> GroundTruthRepository:
    """Get GroundTruth repository."""
    if self._ground_truths is None:
      self._ground_truths = GroundTruthRepository(self.db_manager, self._connection)
    return self._ground_truths

  @property
  def metric_computations(self) -> MetricComputationRepository:
    """Get MetricComputation repository."""
    if self._metric_computations is None:
      self._metric_computations = MetricComputationRepository(self.db_manager, self._connection)
    return self._metric_computations

  @property
  def span_metric_computations(self) -> MetricComputationRepository:
    """Get MetricComputation repository using the table-aligned name."""
    return self.metric_computations

  async def __aenter__(self):
    """Enter async context (no-op, repos manage their own connections)."""
    return self

  async def __aexit__(self, exc_type, exc_val, exc_tb):
    """Exit async context (no-op, repos manage their own connections)."""
    pass


class TransactionalUnitOfWork(UnitOfWork):
  """
  Transactional Unit of Work that wraps all operations in a single transaction.

  Usage:
      async with TransactionalUnitOfWork(db_manager) as uow:
          agent = await uow.agents.create(agent)
          run = await uow.evaluation_runs.create(run)
          # Both committed together or both rolled back
  """

  def __init__(self, db_manager: DatabaseManager):
    """
    Initialize Transactional Unit of Work.

    Args:
        db_manager: Database manager for connection handling
    """
    super().__init__(db_manager)
    self._connection_ctx: Optional[AsyncContextManager[AsyncConnection[Any]]] = None
    self._transaction_conn: Optional[AsyncConnection[Any]] = None
    self._transaction: Optional[AsyncContextManager[AsyncTransaction]] = None

  async def __aenter__(self):
    """
    Enter transactional context.

    Opens a transaction that all repository operations will use.
    """
    self._connection_ctx = self.db_manager.get_async_connection()
    self._transaction_conn = await self._connection_ctx.__aenter__()
    self._transaction = self._transaction_conn.transaction()
    await self._transaction.__aenter__()
    self._connection = self._transaction_conn

    logger.debug('Started transactional unit of work')
    return self

  async def __aexit__(self, exc_type, exc_val, exc_tb):
    """
    Exit transactional context.

    Commits transaction on success, rolls back on exception.
    """
    try:
      if exc_type is not None:
        logger.warning(f'Rolling back transaction due to {exc_type.__name__}: {exc_val}')
        if self._transaction is not None:
          await self._transaction.__aexit__(exc_type, exc_val, exc_tb)
      else:
        if self._transaction is not None:
          logger.debug('Committing transaction')
          await self._transaction.__aexit__(None, None, None)
    finally:
      self._connection = None
      if self._connection_ctx is not None:
        await self._connection_ctx.__aexit__(exc_type, exc_val, exc_tb)
      self._connection_ctx = None
      self._transaction_conn = None
      self._transaction = None
