"""Evaluation services for experiment steps.

Each step opens its own service. The built-in metrics are disabled, so the service allocates no judge client: the step
passes the metric instances it needs, all bound to the shared judge (``judge.open_judge_client``). The database pool is
opened by the step and shared with the metrics that read stored traces.
"""

from collections.abc import AsyncIterator, Mapping, Sequence
from contextlib import asynccontextmanager

from syllo_eval.evaluation.metrics.contracts import EvaluationMetric
from syllo_eval.evaluation.trace_adapter import TraceAdapter
from syllo_eval.evaluation.trace_processor import TraceSourceClient
from syllo_eval.execution.agent_caller import AgentCaller
from syllo_eval.infrastructure import DatabaseManager
from syllo_eval.service import EvaluationService, build_db_manager
from syllo_eval.settings import Settings


@asynccontextmanager
async def open_database(settings: Settings) -> AsyncIterator[DatabaseManager]:
  db_manager = build_db_manager(settings)
  await db_manager.initialize_async()
  try:
    yield db_manager
  finally:
    await db_manager.close_async()


def build_service(
  settings: Settings,
  db_manager: DatabaseManager,
  *,
  metrics: Sequence[EvaluationMetric],
  callers_by_agent_name: Mapping[str, AgentCaller] | None = None,
  trace_client: TraceSourceClient | None = None,
  trace_adapters_by_agent_name: Mapping[str, TraceAdapter] | None = None,
  trace_adapters_by_name: Mapping[str, TraceAdapter] | None = None,
) -> EvaluationService:
  """A service that runs exactly ``metrics``; enter it with ``async with`` to health-check the shared pool."""
  return EvaluationService(
    settings,
    db_manager,
    custom_metrics=metrics,
    include_builtin_metrics=False,
    callers_by_agent_name=callers_by_agent_name,
    trace_client=trace_client,
    trace_adapters_by_agent_name=trace_adapters_by_agent_name,
    trace_adapters_by_name=trace_adapters_by_name,
  )
