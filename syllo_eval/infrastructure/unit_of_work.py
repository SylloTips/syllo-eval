"""
Unit of Work pattern implementation.
"""

import logging
from contextlib import AbstractAsyncContextManager
from functools import cached_property
from typing import Optional, Any

from psycopg import AsyncConnection

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
from syllo_eval.infrastructure.repositories.base import BaseRepository

logger = logging.getLogger(__name__)


class UnitOfWork:
  """
  Single entry point for repository access.

  Repositories acquire a pooled connection per operation, so a plain unit of work is not atomic. Use
  `TransactionalUnitOfWork` for changes that must commit or roll back together.

  Usage:
      async with UnitOfWork(db_manager) as uow:
          agent = await uow.agents.get_by_id(agent_id)
          samples = await uow.samples.list_by_dataset(dataset_id)
  """

  def __init__(self, db_manager: DatabaseManager):
    self.db_manager = db_manager

  def _active_connection(self) -> Optional[AsyncConnection[Any]]:
    return None

  @cached_property
  def agents(self) -> AgentRepository:
    return AgentRepository(self.db_manager, self._active_connection())

  @cached_property
  def datasets(self) -> DatasetRepository:
    return DatasetRepository(self.db_manager, self._active_connection())

  @cached_property
  def evaluation_runs(self) -> EvaluationRunRepository:
    return EvaluationRunRepository(self.db_manager, self._active_connection())

  @cached_property
  def evaluation_run_metrics(self) -> EvaluationRunMetricRepository:
    return EvaluationRunMetricRepository(self.db_manager, self._active_connection())

  @cached_property
  def evaluation_run_plan_samples(self) -> EvaluationRunPlanSampleRepository:
    return EvaluationRunPlanSampleRepository(self.db_manager, self._active_connection())

  @cached_property
  def samples(self) -> SampleRepository:
    return SampleRepository(self.db_manager, self._active_connection())

  @cached_property
  def metrics(self) -> MetricRepository:
    return MetricRepository(self.db_manager, self._active_connection())

  @cached_property
  def span_types(self) -> SpanTypeRepository:
    return SpanTypeRepository(self.db_manager, self._active_connection())

  @cached_property
  def spans(self) -> SpanRepository:
    return SpanRepository(self.db_manager, self._active_connection())

  @cached_property
  def traces(self) -> TraceRepository:
    return TraceRepository(self.db_manager, self._active_connection())

  @cached_property
  def evaluation_run_samples(self) -> EvaluationRunSampleRepository:
    return EvaluationRunSampleRepository(self.db_manager, self._active_connection())

  @cached_property
  def metric_target_span_types(self) -> MetricTargetSpanTypeRepository:
    return MetricTargetSpanTypeRepository(self.db_manager, self._active_connection())

  @cached_property
  def ground_truths(self) -> GroundTruthRepository:
    return GroundTruthRepository(self.db_manager, self._active_connection())

  @cached_property
  def metric_computations(self) -> MetricComputationRepository:
    return MetricComputationRepository(self.db_manager, self._active_connection())

  @property
  def span_metric_computations(self) -> MetricComputationRepository:
    """Table-aligned alias for `metric_computations`."""
    return self.metric_computations

  async def __aenter__(self):
    return self

  async def __aexit__(self, exc_type, exc_val, exc_tb):
    pass


class TransactionalUnitOfWork(UnitOfWork):
  """
  Unit of work whose repositories share one `DatabaseManager.transaction()`.

  Commits on success and rolls back on exception or cancellation. Instances are single-use: access repositories only
  inside the `async with` block and do not keep repository references after it exits.

  Usage:
      async with TransactionalUnitOfWork(db_manager) as uow:
          agent = await uow.agents.create(agent)
          run = await uow.evaluation_runs.create(run)
  """

  def __init__(self, db_manager: DatabaseManager):
    super().__init__(db_manager)
    self._transaction: AbstractAsyncContextManager[AsyncConnection[Any]] = db_manager.transaction()
    self._connection: Optional[AsyncConnection[Any]] = None
    self._entered = False

  def _active_connection(self) -> AsyncConnection[Any]:
    if self._connection is None:
      raise RuntimeError('TransactionalUnitOfWork repositories are only available inside its async with block')
    return self._connection

  async def __aenter__(self):
    if self._entered:
      raise RuntimeError('TransactionalUnitOfWork is single-use')
    self._entered = True
    self._connection = await self._transaction.__aenter__()
    return self

  async def __aexit__(self, exc_type, exc_val, exc_tb):
    self._connection = None
    for name in [name for name, value in vars(self).items() if isinstance(value, BaseRepository)]:
      del self.__dict__[name]
    if exc_type is not None:
      logger.warning('Rolling back transaction due to %s: %s', exc_type.__name__, exc_val)
    return await self._transaction.__aexit__(exc_type, exc_val, exc_tb)
