import logging
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from contextlib import AsyncExitStack
from dataclasses import dataclass, field
from datetime import datetime, timezone
from uuid import UUID

from syllo_eval.evaluation.claim_extractor import ClaimExtractorClient
from syllo_eval.evaluation.judge import LlmJudgeClient
from syllo_eval.evaluation.evaluation_report import EvaluationReport, EvaluationReportAggregator
from syllo_eval.evaluation.metric_planner import MetricPlanner
from syllo_eval.evaluation.metric_registry import MetricRegistry
from syllo_eval.evaluation.metrics.available import (
  BUILTIN_METRICS,
  MetricClientRequirements,
  available_metric_names,
  build_available_metrics,
  required_clients,
)
from syllo_eval.evaluation.metrics.contracts import EvaluationMetric
from syllo_eval.evaluation.plan_executor import PlanExecutor
from syllo_eval.evaluation.trace_adapter import PhoenixTraceAdapter, TraceAdapter
from syllo_eval.evaluation.trace_import import (
  ImportedTrace,
  TraceImportError,
  match_traces_to_samples,
  normalize_imported_trace,
)
from syllo_eval.evaluation.trace_processor import TraceIntegration, TraceProcessor, TraceSourceClient, persist_trace
from syllo_eval.execution.agent_caller import AgentCallDispatcher, AgentCaller
from syllo_eval.execution.sample_executor import SampleExecutor
from syllo_eval.infrastructure import DatabaseConfig, DatabaseManager
from syllo_eval.infrastructure.arize import PhoenixClient
from syllo_eval.infrastructure.exceptions import ConfigurationError, NotFoundError
from syllo_eval.infrastructure.llm_judge import build_llm_judge_client
from syllo_eval.infrastructure.orbitals import OrbitalsClaimExtractorClient
from syllo_eval.infrastructure.unit_of_work import TransactionalUnitOfWork, UnitOfWork
from syllo_eval.model import EvaluationRun, EvaluationRunSample, EvaluationSampleStatus, EvaluationStatus
from syllo_eval.orchestration.evaluation_orchestrator import EvaluationConfig, EvaluationOrchestrator
from syllo_eval.settings import EvaluationSettings, Settings

logger = logging.getLogger(__name__)


class MetricSelectionError(ValueError):
  """Raised when requested metric names do not match the available metrics."""


class AgentCallerSelectionError(ValueError):
  """Raised when the requested agent has no registered caller."""


class RepeatNotAllowedError(ValueError):
  """Raised when a source evaluation run cannot be repeated."""


def build_db_manager(settings: Settings) -> DatabaseManager:
  """Build an independent pool; never reuse or replace process-global state."""
  db_settings = settings.database
  if not db_settings.password:
    raise ConfigurationError('database', 'password', 'set settings.database.password or DB_PASSWORD')
  return DatabaseManager(
    DatabaseConfig(
      host=db_settings.host,
      port=db_settings.port,
      database=db_settings.database,
      user=db_settings.user,
      password=db_settings.password,
      min_size=db_settings.min_size,
      max_size=db_settings.max_size,
      timeout=db_settings.timeout,
    )
  )


def settings_metric_names(settings: Settings) -> list[str]:
  """Built-in inventory enabled by ``settings``; needs no database and no provider client."""
  return available_metric_names(
    llm_judge_enabled=settings.llm_judge.provider is not None,
    claim_extractor_enabled=settings.orbitals.api_key is not None,
  )


def resolve_selected_metric_names(
  raw_metric_names: Sequence[str] | None,
  available_names: Sequence[str],
) -> list[str]:
  if not raw_metric_names:
    return list(available_names)

  metric_names_by_normalized = {_normalize_metric_name(metric_name): metric_name for metric_name in available_names}
  resolved_metric_names: list[str] = []
  seen_metric_names: set[str] = set()
  unknown_metric_names: list[str] = []

  for raw_metric_name in raw_metric_names:
    stripped_metric_name = raw_metric_name.strip()
    if not stripped_metric_name:
      raise MetricSelectionError(f'metrics cannot contain empty entries. Valid metrics: {", ".join(available_names)}')

    normalized_metric_name = _normalize_metric_name(stripped_metric_name)
    canonical_metric_name = metric_names_by_normalized.get(normalized_metric_name)
    if canonical_metric_name is None:
      unknown_metric_names.append(stripped_metric_name)
      continue

    if normalized_metric_name in seen_metric_names:
      continue

    seen_metric_names.add(normalized_metric_name)
    resolved_metric_names.append(canonical_metric_name)

  if unknown_metric_names:
    raise MetricSelectionError(
      f'Unknown metric name(s): {", ".join(unknown_metric_names)}. Valid metrics: {", ".join(available_names)}'
    )

  return resolved_metric_names


def resolve_selected_metric_names_csv(
  raw_metrics: str | None,
  available_names: Sequence[str],
) -> list[str]:
  if raw_metrics is None:
    return list(available_names)
  return resolve_selected_metric_names(raw_metrics.split(','), available_names)


def _resolve_rubric_additions(
  rubric_additions: Mapping[str, str] | None, selected_metric_names: Sequence[str]
) -> dict[str, str]:
  judge_metric_names = {
    metric_class.metric_name for metric_class in BUILTIN_METRICS if metric_class.requires_judge_client
  }
  names_by_normalized = {
    _normalize_metric_name(name): name for name in selected_metric_names if name in judge_metric_names
  }
  resolved: dict[str, str] = {}
  for raw_name, text in (rubric_additions or {}).items():
    name = names_by_normalized.get(_normalize_metric_name(raw_name))
    if name is None or name in resolved:
      raise MetricSelectionError(f'Rubric addition {raw_name!r} must target a distinct selected LLM-judge metric')
    if not text.strip():
      raise MetricSelectionError(f'Rubric addition for {name} cannot be blank')
    resolved[name] = text.strip()
  return resolved


def _normalize_metric_name(metric_name: str) -> str:
  return metric_name.strip().lower()


def _reject_bad_metric_names(names: Sequence[str]) -> None:
  seen: set[str] = set()
  for name in names:
    key = _normalize_metric_name(name)
    if not key or key in seen:
      raise MetricSelectionError(f'Empty or duplicate registered metric name: {name!r}')
    seen.add(key)


def _normalize_names[T](values: Mapping[str, T], kind: str) -> dict[str, T]:
  normalized = {name.strip().lower(): value for name, value in values.items()}
  if len(normalized) != len(values) or '' in normalized:
    raise ValueError(f'{kind} names must be non-empty and unique after normalization')
  return normalized


def _build_metric_registry(
  db_manager: DatabaseManager,
  judge_client: LlmJudgeClient | None = None,
  claim_extractor_client: ClaimExtractorClient | None = None,
  selected_metric_names: Sequence[str] | None = None,
  custom_metrics: Sequence[EvaluationMetric] = (),
  include_builtin_metrics: bool = True,
  rubric_additions: Mapping[str, str] | None = None,
  retrieval_span_types: Sequence[str] = ('agent_root',),
) -> MetricRegistry:
  registry = MetricRegistry(db_manager=db_manager)
  metrics = (
    build_available_metrics(
      judge_client=judge_client,
      claim_extractor_client=claim_extractor_client,
      rubric_additions=rubric_additions,
      retrieval_span_types=retrieval_span_types,
    )
    if include_builtin_metrics
    else []
  )
  registry.register_many([*metrics, *custom_metrics])
  if selected_metric_names is None:
    return registry
  try:
    selected = [registry.get(name) for name in selected_metric_names]
  except KeyError as err:
    raise MetricSelectionError(f'Selected metric has no available implementation: {err}') from err
  selected_registry = MetricRegistry(db_manager=db_manager)
  selected_registry.register_many(selected)
  return selected_registry


@dataclass(slots=True)
class EvaluationHandle:
  run: EvaluationRun
  _execution: '_PreparedEvaluationExecution'
  agent_name: str | None = None
  agent_version_tag: str | None = None

  async def execute(self) -> EvaluationRun:
    return await self._execution.execute_run(self.run)

  async def aclose(self) -> None:
    """Release prepared provider clients if execution is abandoned."""
    await self._execution.close()


@dataclass(slots=True)
class EvaluationSampleStatusCounts:
  total: int
  pending: int
  running: int
  completed: int
  failed: int
  unprocessed: int = 0


@dataclass(slots=True)
class EvaluationRunStatusSnapshot:
  run: EvaluationRun
  sample_counts: EvaluationSampleStatusCounts


@dataclass(slots=True)
class EvaluationRunList:
  runs: Sequence[EvaluationRun]
  total: int
  limit: int
  offset: int


@dataclass(slots=True)
class _PreparedEvaluationExecution:
  orchestrator: EvaluationOrchestrator
  config: EvaluationConfig
  resources: AsyncExitStack = field(default_factory=AsyncExitStack)

  async def create_run(self) -> EvaluationRun:
    return await self.orchestrator.create_run(self.config)

  async def execute_run(self, run: EvaluationRun) -> EvaluationRun:
    try:
      return await self.orchestrator.execute_run(run)
    finally:
      await self.close()

  async def close(self) -> None:
    await self.resources.aclose()


class EvaluationService:
  """Shared runtime. Injected dependencies and metric resources remain caller-owned."""

  def __init__(
    self,
    settings: Settings | None = None,
    db_manager: DatabaseManager | None = None,
    *,
    custom_metrics: Sequence[EvaluationMetric] = (),
    include_builtin_metrics: bool = True,
    callers_by_agent_name: Mapping[str, AgentCaller] | None = None,
    trace_client: TraceSourceClient | None = None,
    trace_adapters_by_agent_name: Mapping[str, TraceAdapter] | None = None,
    trace_integration: TraceIntegration | None = None,
    trace_adapters_by_name: Mapping[str, TraceAdapter] | None = None,
  ):
    if trace_client is not None and trace_integration is not None:
      raise ValueError('Supply either trace_client or trace_integration, not both')
    self._settings = settings if settings is not None else Settings()
    self._owns_db_manager = db_manager is None
    self._db_manager = db_manager if db_manager is not None else build_db_manager(self._settings)
    self._custom_metrics = tuple(custom_metrics)
    self._include_builtin_metrics = include_builtin_metrics
    self._callers = _normalize_names(callers_by_agent_name or {}, 'Agent caller')
    self._agent_call_dispatcher = AgentCallDispatcher(self._callers)
    self._trace_adapters = _normalize_names(trace_adapters_by_agent_name or {}, 'Trace adapter')
    self._import_adapters = {
      'phoenix': PhoenixTraceAdapter(),
      **_normalize_names(trace_adapters_by_name or {}, 'Import trace adapter'),
    }
    if trace_integration is not None and self._trace_adapters:
      raise ValueError('Supply trace adapters or a trace integration, not both')
    self._trace_client = trace_client
    self._trace_integration = trace_integration
    builtins = [metric_class.metric_name for metric_class in BUILTIN_METRICS] if include_builtin_metrics else []
    _reject_bad_metric_names([*builtins, *(metric.name for metric in self._custom_metrics)])

  @property
  def settings(self) -> Settings:
    return self._settings

  @property
  def db_manager(self) -> DatabaseManager:
    return self._db_manager

  def available_metric_names(self) -> list[str]:
    """List the combined inventory without constructing provider clients."""
    names = settings_metric_names(self._settings) if self._include_builtin_metrics else []
    names.extend(metric.name for metric in self._custom_metrics)
    _reject_bad_metric_names(names)
    return names

  async def __aenter__(self) -> 'EvaluationService':
    try:
      await self.initialize()
    except BaseException:
      await self.close()
      raise
    return self

  async def __aexit__(self, exc_type, exc, tb) -> None:
    await self.close()

  async def initialize(self) -> None:
    if self._owns_db_manager:
      await self._db_manager.initialize_async()
    if not await self._db_manager.health_check():
      raise RuntimeError('Database health check failed')

  async def close(self) -> None:
    if self._owns_db_manager:
      await self._db_manager.close_async()

  async def get_evaluation_run(self, run_id: UUID) -> EvaluationRun:
    async with UnitOfWork(self._db_manager) as uow:
      run = await uow.evaluation_runs.get_by_id(run_id)

    if run is None:
      raise NotFoundError('evaluation_run', run_id)

    return run

  async def list_evaluations(
    self,
    *,
    limit: int,
    offset: int,
    status: EvaluationStatus | None = None,
  ) -> EvaluationRunList:
    async with UnitOfWork(self._db_manager) as uow:
      runs = await uow.evaluation_runs.list_runs(limit=limit, offset=offset, status=status)
      total = await uow.evaluation_runs.count_runs(status=status)

    return EvaluationRunList(runs=runs, total=total, limit=limit, offset=offset)

  async def get_evaluation_status(self, run_id: UUID) -> EvaluationRunStatusSnapshot:
    async with UnitOfWork(self._db_manager) as uow:
      run = await uow.evaluation_runs.get_by_id(run_id)
      if run is None:
        raise NotFoundError('evaluation_run', run_id)

      planned_ids = await uow.evaluation_run_plan_samples.list_sample_ids(run.id)
      total_samples = (
        len(planned_ids)
        if planned_ids or (run.config or {}).get('plan_snapshot_version') == 1
        else await uow.samples.count_by_dataset(run.dataset_id)
      )
      run_samples = await uow.evaluation_run_samples.list_by_evaluation_run(run.id)

    return EvaluationRunStatusSnapshot(
      run=run,
      sample_counts=_build_sample_status_counts(total_samples, run_samples, run.status),
    )

  async def get_evaluation_report(self, run_id: UUID) -> EvaluationReport:
    return await EvaluationReportAggregator(self._db_manager).build(run_id)

  async def fail_evaluation_run(self, run_id: UUID) -> EvaluationRun:
    async with UnitOfWork(self._db_manager) as uow:
      updated_run = await uow.evaluation_runs.update_status_if_current(
        run_id=run_id,
        current_statuses=[EvaluationStatus.RUNNING],
        status=EvaluationStatus.FAILED,
        end_time=datetime.now(timezone.utc),
      )
      if updated_run is not None:
        return updated_run

      run = await uow.evaluation_runs.get_by_id(run_id)

    if run is None:
      raise NotFoundError('evaluation_run', run_id)

    return run

  async def fail_orphaned_running_evaluations(self) -> int:
    async with UnitOfWork(self._db_manager) as uow:
      failed_runs = await uow.evaluation_runs.fail_running(datetime.now(timezone.utc))

    return len(failed_runs)

  async def run_evaluation(
    self,
    *,
    agent_name: str,
    agent_version_tag: str,
    dataset_id: UUID,
    max_concurrent_samples: int | None = None,
    max_concurrent_tasks: int | None = None,
    sample_trace_timeout: float | None = None,
    sample_compute_timeout: float | None = None,
    selected_metric_names: Sequence[str] | None = None,
    rubric_additions: Mapping[str, str] | None = None,
  ) -> EvaluationRun:
    handle = await self.create_evaluation(
      agent_name=agent_name,
      agent_version_tag=agent_version_tag,
      dataset_id=dataset_id,
      max_concurrent_samples=max_concurrent_samples,
      max_concurrent_tasks=max_concurrent_tasks,
      sample_trace_timeout=sample_trace_timeout,
      sample_compute_timeout=sample_compute_timeout,
      selected_metric_names=selected_metric_names,
      rubric_additions=rubric_additions,
    )
    return await handle.execute()

  async def create_evaluation(
    self,
    *,
    agent_name: str,
    agent_version_tag: str,
    dataset_id: UUID,
    max_concurrent_samples: int | None = None,
    max_concurrent_tasks: int | None = None,
    sample_trace_timeout: float | None = None,
    sample_compute_timeout: float | None = None,
    selected_metric_names: Sequence[str] | None = None,
    rubric_additions: Mapping[str, str] | None = None,
  ) -> EvaluationHandle:
    execution = await self._build_execution(
      agent_name=agent_name,
      agent_version_tag=agent_version_tag,
      dataset_id=dataset_id,
      max_concurrent_samples=max_concurrent_samples,
      max_concurrent_tasks=max_concurrent_tasks,
      sample_trace_timeout=sample_trace_timeout,
      sample_compute_timeout=sample_compute_timeout,
      selected_metric_names=selected_metric_names,
      rubric_additions=rubric_additions,
    )
    try:
      run = await execution.create_run()
    except BaseException:
      await execution.close()
      raise

    return EvaluationHandle(run=run, _execution=execution, agent_name=agent_name, agent_version_tag=agent_version_tag)

  async def import_evaluation(
    self,
    *,
    agent_name: str,
    agent_version_tag: str,
    dataset_id: UUID,
    traces: Sequence[ImportedTrace],
    trace_adapter_name: str = 'phoenix',
    max_concurrent_samples: int | None = None,
    max_concurrent_tasks: int | None = None,
    sample_compute_timeout: float | None = None,
    selected_metric_names: Sequence[str] | None = None,
    rubric_additions: Mapping[str, str] | None = None,
  ) -> EvaluationHandle:
    """Evaluate exported traces without calling the agent; only samples with a trace are planned."""
    resolved_metric_names = resolve_selected_metric_names(selected_metric_names, self.available_metric_names())
    resolved_rubric_additions = _resolve_rubric_additions(rubric_additions, resolved_metric_names)
    adapter = self._import_adapters.get(trace_adapter_name.strip().lower())
    if adapter is None:
      raise TraceImportError(
        f'Unknown trace adapter {trace_adapter_name!r}. Valid adapters: {", ".join(self._import_adapters)}'
      )
    results = [normalize_imported_trace(adapter, trace) for trace in traces]
    async with UnitOfWork(self._db_manager) as uow:
      samples = await uow.samples.list_by_dataset(dataset_id)
    trace_ids_by_sample_id = match_traces_to_samples(results, samples)
    async with TransactionalUnitOfWork(self._db_manager) as uow:
      for result in results:
        try:
          await persist_trace(uow, result)
        except ValueError as err:
          raise TraceImportError(f'Trace {result.trace.external_id} could not be stored: {err}') from err
    async with UnitOfWork(self._db_manager) as uow:
      agent = await uow.agents.get_or_create_by_name_and_version(name=agent_name, version_tag=agent_version_tag)

    execution = await self._build_execution_from_plan(
      agent_id=agent.id,
      dataset_id=dataset_id,
      selected_metric_names=resolved_metric_names,
      planned_sample_ids=list(trace_ids_by_sample_id),
      max_concurrent_samples=max_concurrent_samples,
      max_concurrent_tasks=max_concurrent_tasks,
      sample_trace_timeout=None,
      sample_compute_timeout=sample_compute_timeout,
      imported_trace_ids=trace_ids_by_sample_id,
      rubric_additions=resolved_rubric_additions,
    )
    try:
      run = await execution.create_run()
    except BaseException:
      await execution.close()
      raise

    return EvaluationHandle(run=run, _execution=execution, agent_name=agent_name, agent_version_tag=agent_version_tag)

  async def repeat_evaluation(
    self,
    *,
    source_run_id: UUID,
    metrics: Sequence[str] | None = None,
    max_concurrent_samples: int | None = None,
    max_concurrent_tasks: int | None = None,
    sample_compute_timeout: float | None = None,
    rubric_additions: Mapping[str, str] | None = None,
  ) -> EvaluationHandle:
    """Recompute stored traces; rubric additions default to the source run's for metrics still selected."""
    async with UnitOfWork(self._db_manager) as uow:
      source_run = await uow.evaluation_runs.get_by_id(source_run_id)
      if source_run is None:
        raise NotFoundError('evaluation_run', source_run_id)
      if source_run.status == EvaluationStatus.RUNNING:
        raise RepeatNotAllowedError('Source run is still running.')

      source_metric_names = await uow.evaluation_run_metrics.list_metrics(source_run_id)
      source_run_samples = await uow.evaluation_run_samples.list_by_evaluation_run(source_run_id)
      traced_sample_ids = [run_sample.sample_id for run_sample in source_run_samples if run_sample.trace_id is not None]
      if not traced_sample_ids:
        raise RepeatNotAllowedError('Source run has no stored traces to recompute.')

      agent = await uow.agents.get_by_id_or_raise(source_run.agent_id)

    selected_metric_names = self._resolve_repeat_metric_names(source_metric_names, metrics)
    if rubric_additions is None:
      source_additions = (source_run.config or {}).get('rubric_additions') or {}
      rubric_additions = {name: text for name, text in source_additions.items() if name in selected_metric_names}
    resolved_rubric_additions = _resolve_rubric_additions(rubric_additions, selected_metric_names)

    execution = await self._build_execution_from_plan(
      agent_id=source_run.agent_id,
      dataset_id=source_run.dataset_id,
      selected_metric_names=selected_metric_names,
      planned_sample_ids=traced_sample_ids,
      max_concurrent_samples=max_concurrent_samples,
      max_concurrent_tasks=max_concurrent_tasks,
      sample_trace_timeout=None,
      sample_compute_timeout=sample_compute_timeout,
      source_run_id=source_run.id,
      rubric_additions=resolved_rubric_additions,
    )
    try:
      run = await execution.create_run()
    except BaseException:
      await execution.close()
      raise

    return EvaluationHandle(run=run, _execution=execution, agent_name=agent.name, agent_version_tag=agent.version_tag)

  async def _build_execution(
    self,
    *,
    agent_name: str,
    agent_version_tag: str,
    dataset_id: UUID,
    max_concurrent_samples: int | None,
    max_concurrent_tasks: int | None,
    sample_trace_timeout: float | None,
    sample_compute_timeout: float | None,
    selected_metric_names: Sequence[str] | None,
    rubric_additions: Mapping[str, str] | None,
  ) -> _PreparedEvaluationExecution:
    resolved_metric_names = resolve_selected_metric_names(selected_metric_names, self.available_metric_names())
    resolved_rubric_additions = _resolve_rubric_additions(rubric_additions, resolved_metric_names)
    dispatcher = self._build_agent_dispatcher(agent_name)
    async with UnitOfWork(self._db_manager) as uow:
      agent = await uow.agents.get_or_create_by_name_and_version(name=agent_name, version_tag=agent_version_tag)
      planned_samples = await uow.samples.list_by_dataset(dataset_id)

    return await self._build_execution_from_plan(
      agent_id=agent.id,
      dataset_id=dataset_id,
      selected_metric_names=resolved_metric_names,
      planned_sample_ids=[sample.id for sample in planned_samples],
      max_concurrent_samples=max_concurrent_samples,
      max_concurrent_tasks=max_concurrent_tasks,
      sample_trace_timeout=sample_trace_timeout,
      sample_compute_timeout=sample_compute_timeout,
      agent_call_dispatcher=dispatcher,
      trace_adapter=self._trace_adapters.get(agent_name.strip().lower()),
      rubric_additions=resolved_rubric_additions,
    )

  def _build_agent_dispatcher(self, agent_name: str) -> AgentCallDispatcher:
    if agent_name.strip().lower() not in self._callers:
      raise AgentCallerSelectionError(f'No caller registered for agent {agent_name!r}. Supply callers_by_agent_name.')
    return self._agent_call_dispatcher

  async def _build_execution_from_plan(
    self,
    *,
    agent_id: UUID,
    dataset_id: UUID,
    selected_metric_names: Sequence[str],
    planned_sample_ids: Sequence[UUID],
    max_concurrent_samples: int | None,
    max_concurrent_tasks: int | None,
    sample_trace_timeout: float | None,
    sample_compute_timeout: float | None,
    source_run_id: UUID | None = None,
    agent_call_dispatcher: AgentCallDispatcher | None = None,
    trace_adapter: TraceAdapter | None = None,
    imported_trace_ids: Mapping[UUID, str] | None = None,
    rubric_additions: Mapping[str, str] | None = None,
  ) -> _PreparedEvaluationExecution:
    overrides = {
      'max_concurrent_samples': max_concurrent_samples,
      'max_concurrent_tasks': max_concurrent_tasks,
      'sample_trace_timeout': sample_trace_timeout,
      'sample_compute_timeout': sample_compute_timeout,
    }
    options = EvaluationSettings.model_validate(
      {
        **self._settings.evaluation.model_dump(),
        **{name: value for name, value in overrides.items() if value is not None},
      }
    )
    max_concurrent_samples = options.max_concurrent_samples
    max_concurrent_tasks = options.max_concurrent_tasks
    uses_stored_traces = source_run_id is not None or imported_trace_ids is not None
    sample_trace_timeout = None if uses_stored_traces else options.sample_trace_timeout
    sample_compute_timeout = options.sample_compute_timeout
    async with AsyncExitStack() as resources:
      judge_client = None
      claim_extractor_client = None
      # Deterministic/custom-only runs need no built-in provider resources.
      requirements = (
        required_clients(selected_metric_names) if self._include_builtin_metrics else MetricClientRequirements()
      )
      if requirements.judge:
        judge_client = build_llm_judge_client(self._settings.llm_judge)
        if judge_client is not None:
          resources.push_async_callback(judge_client.aclose)
      if requirements.claim_extractor:
        claim_extractor_client = self._build_claim_extractor_client()
        if claim_extractor_client is not None:
          resources.push_async_callback(claim_extractor_client.aclose)
      config = EvaluationConfig(
        agent_id=agent_id,
        dataset_id=dataset_id,
        selected_metric_names=list(selected_metric_names),
        planned_sample_ids=list(planned_sample_ids),
        config=self._build_run_config_snapshot(
          selected_metric_names=selected_metric_names,
          rubric_additions=rubric_additions or {},
          max_concurrent_samples=max_concurrent_samples,
          max_concurrent_tasks=max_concurrent_tasks,
          sample_trace_timeout=sample_trace_timeout,
          sample_compute_timeout=sample_compute_timeout,
          retrieval_span_types=options.retrieval_span_types,
        ),
        source_run_id=source_run_id,
      )
      metric_registry = _build_metric_registry(
        db_manager=self._db_manager,
        judge_client=judge_client,
        claim_extractor_client=claim_extractor_client,
        selected_metric_names=selected_metric_names,
        custom_metrics=self._custom_metrics,
        include_builtin_metrics=self._include_builtin_metrics,
        rubric_additions=rubric_additions,
        retrieval_span_types=options.retrieval_span_types,
      )
      await metric_registry.sync_with_persistence()

      trace_processor = None
      if not uses_stored_traces:
        trace_processor = self._trace_integration
        if trace_processor is None:
          trace_client = self._trace_client
          if trace_client is None:
            trace_client = PhoenixClient(self._settings.phoenix)
          trace_processor = TraceProcessor(
            trace_client=trace_client, db_manager=self._db_manager, adapter=trace_adapter
          )
      metric_planner = MetricPlanner(metric_registry=metric_registry, db_manager=self._db_manager)
      plan_executor = PlanExecutor(
        db_manager=self._db_manager,
        max_concurrent_tasks=max_concurrent_tasks,
      )
      sample_executor = SampleExecutor(
        db_manager=self._db_manager,
        agent_call_dispatcher=agent_call_dispatcher,
        trace_processor=trace_processor,
        metric_planner=metric_planner,
        plan_executor=plan_executor,
      )

      orchestrator = EvaluationOrchestrator(
        db_manager=self._db_manager,
        sample_executor=sample_executor,
        max_concurrent_samples=max_concurrent_samples,
        sample_trace_timeout_seconds=sample_trace_timeout,
        sample_compute_timeout_seconds=sample_compute_timeout,
        imported_trace_ids=imported_trace_ids,
      )
      execution = _PreparedEvaluationExecution(orchestrator=orchestrator, config=config, resources=resources.pop_all())
    return execution

  def _resolve_repeat_metric_names(
    self,
    source_metric_names: Sequence[str],
    requested_metric_names: Sequence[str] | None,
  ) -> list[str]:
    if requested_metric_names:
      return resolve_selected_metric_names(
        requested_metric_names,
        self.available_metric_names(),
      )

    if not source_metric_names:
      raise RepeatNotAllowedError('Source run has no metric snapshot and cannot be repeated.')
    selected_metric_names = self._filter_current_metric_names(source_metric_names)
    if not selected_metric_names:
      raise RepeatNotAllowedError('No source metrics are available in the current runtime.')
    return selected_metric_names

  def _filter_current_metric_names(self, metric_names: Sequence[str]) -> list[str]:
    available_names = self.available_metric_names()
    available_normalized = {_normalize_metric_name(metric_name) for metric_name in available_names}
    filtered: list[str] = []
    for metric_name in metric_names:
      if _normalize_metric_name(metric_name) in available_normalized:
        filtered.append(metric_name)
      else:
        logger.warning('Dropping repeat metric not available in current runtime metric=%s', metric_name)
    return resolve_selected_metric_names(filtered, available_names) if filtered else []

  def _build_run_config_snapshot(
    self,
    *,
    selected_metric_names: Sequence[str],
    rubric_additions: Mapping[str, str],
    max_concurrent_samples: int,
    max_concurrent_tasks: int,
    sample_trace_timeout: float | None,
    sample_compute_timeout: float | None,
    retrieval_span_types: Sequence[str],
  ) -> dict[str, object | None]:
    provider = self._settings.llm_judge.provider
    model = None
    if provider == 'openai':
      model = self._settings.llm_judge.openai.model
    elif provider == 'gemini':
      model = self._settings.llm_judge.gemini.model

    return {
      'plan_snapshot_version': 1,
      'llm_judge_provider': provider,
      'llm_judge_model': model,
      'orbitals_claim_extractor_model': (
        self._settings.orbitals.claim_extractor_model if self._settings.orbitals.api_key is not None else None
      ),
      'max_concurrent_samples': max_concurrent_samples,
      'max_concurrent_tasks': max_concurrent_tasks,
      'sample_trace_timeout_seconds': sample_trace_timeout,
      'sample_compute_timeout_seconds': sample_compute_timeout,
      # The span types the built-in retrieval metrics scored; custom metrics declare their own.
      'retrieval_span_types': list(retrieval_span_types) if self._include_builtin_metrics else None,
      'prompt_versions': {
        metric_class.metric_name: version
        for metric_class in (BUILTIN_METRICS if self._include_builtin_metrics else ())
        if metric_class.metric_name in selected_metric_names
        and (version := getattr(metric_class, 'prompt_version', None)) is not None
      },
      'rubric_additions': dict(rubric_additions),
    }

  def _build_claim_extractor_client(self) -> ClaimExtractorClient | None:
    if self._settings.orbitals.api_key is None:
      return None
    return OrbitalsClaimExtractorClient(self._settings.orbitals)


ServiceFactory = Callable[[Settings], EvaluationService]


def _build_sample_status_counts(
  total_samples: int,
  run_samples: Sequence[EvaluationRunSample],
  run_status: EvaluationStatus,
) -> EvaluationSampleStatusCounts:
  status_counts = Counter(run_sample.status for run_sample in run_samples)
  missing = max(0, total_samples - len(run_samples))
  return EvaluationSampleStatusCounts(
    total=total_samples,
    pending=missing if run_status == EvaluationStatus.RUNNING else 0,
    unprocessed=missing if run_status != EvaluationStatus.RUNNING else 0,
    running=status_counts[EvaluationSampleStatus.RUNNING],
    completed=status_counts[EvaluationSampleStatus.COMPLETED],
    failed=status_counts[EvaluationSampleStatus.FAILED],
  )
