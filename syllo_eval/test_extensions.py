import asyncio
import unittest
from datetime import datetime, timezone
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

from syllo_eval.evaluation.metrics.contracts import MetricComputationResult, SpanEvaluationMetric
from syllo_eval.evaluation.metric_planner import MetricPlanner
from syllo_eval.evaluation.metric_registry import MetricRegistry
from syllo_eval.evaluation.plan_executor import PlanExecutor
from syllo_eval.evaluation.test_trace_import import phoenix_trace
from syllo_eval.evaluation.trace_adapter import PhoenixTraceAdapter
from syllo_eval.evaluation.trace_import import TraceImportError
from syllo_eval.evaluation.trace_processor import TraceProcessingResult
from syllo_eval.infrastructure.database import DatabaseManager
from syllo_eval.model import (
  Agent,
  EvaluationRun,
  EvaluationRunSample,
  EvaluationSampleStatus,
  EvaluationStatus,
  GroundTruth,
  Metric,
  MetricComputationStatus,
  MetricTargetSpanType,
  Sample,
  Span,
  SpanType,
  Trace,
)
from syllo_eval.service import (
  AgentCallerSelectionError,
  EvaluationService,
  MetricSelectionError,
  _build_metric_registry,
)
from syllo_eval.settings import EvaluationSettings, LlmJudgeSettings, Settings


class _CustomMetric(SpanEvaluationMetric):
  @property
  def name(self) -> str:
    return 'custom_metric'

  @property
  def requires_ground_truth(self) -> bool:
    return False

  @property
  def target_span_types(self) -> tuple[str, ...]:
    return ('agent_root',)

  async def compute(self, span: Span, ground_truth: GroundTruth | None) -> MetricComputationResult:
    del span, ground_truth
    return MetricComputationResult(score=1.0)


def _fake_repositories(*, agent: Agent | None = None, samples: list[Sample] | None = None) -> SimpleNamespace:
  return SimpleNamespace(
    agents=SimpleNamespace(
      get_or_create_by_name_and_version=AsyncMock(return_value=agent),
      get_by_id_or_raise=AsyncMock(return_value=agent),
    ),
    samples=SimpleNamespace(list_by_dataset=AsyncMock(return_value=samples or [])),
    evaluation_runs=SimpleNamespace(get_by_id=AsyncMock()),
    evaluation_run_metrics=SimpleNamespace(list_metrics=AsyncMock()),
    evaluation_run_samples=SimpleNamespace(list_by_evaluation_run=AsyncMock()),
  )


class _ExecutionStub:
  def __init__(self, run: EvaluationRun) -> None:
    self.create_run = AsyncMock(return_value=run)
    self.execute_run = AsyncMock(return_value=run)
    self.close = AsyncMock()


class _AsyncRepositoryContext:
  def __init__(self, repositories: Any) -> None:
    self.repositories = repositories

  async def __aenter__(self) -> Any:
    return self.repositories

  async def __aexit__(self, exc_type, exc, tb) -> None:
    del exc_type, exc, tb


class _RecordingTransaction(_AsyncRepositoryContext):
  exit_type: type[BaseException] | None = None

  async def __aexit__(self, exc_type, exc, tb) -> None:
    del exc, tb
    self.exit_type = exc_type


class _MemoryPersistence:
  """Small persistence boundary for the public-service extension test."""

  def __init__(self, agent: Agent, sample: Sample) -> None:
    self.agents_by_id = {agent.id: agent}
    self.samples_by_id = {sample.id: sample}
    self.runs_by_id: dict[Any, EvaluationRun] = {}
    self.run_metric_names: dict[Any, list[str]] = {}
    self.run_plan_sample_ids: dict[Any, list[Any]] = {}
    self.run_samples_by_run: dict[Any, list[EvaluationRunSample]] = {}
    self.traces_by_id: dict[str, Trace] = {}
    self.spans_by_id: dict[str, Span] = {}
    self.ground_truths_by_sample: dict[Any, list[GroundTruth]] = {}
    self.metrics_by_name: dict[str, Metric] = {}
    self.span_types_by_name: dict[str, SpanType] = {}
    self.metric_targets: dict[tuple[str, str], MetricTargetSpanType] = {}
    self.computations_by_run_sample: dict[Any, list[Any]] = {}


class _MemoryRepositoryContext:
  def __init__(self, persistence: _MemoryPersistence) -> None:
    self.persistence = persistence
    self.agents = SimpleNamespace(
      get_or_create_by_name_and_version=AsyncMock(side_effect=self._get_or_create_agent),
      get_by_id=AsyncMock(side_effect=lambda agent_id: persistence.agents_by_id.get(agent_id)),
      get_by_id_or_raise=AsyncMock(side_effect=lambda agent_id: persistence.agents_by_id[agent_id]),
    )
    self.samples = SimpleNamespace(
      list_by_dataset=AsyncMock(side_effect=self._list_samples_by_dataset),
      list_by_ids=AsyncMock(
        side_effect=lambda sample_ids: [persistence.samples_by_id[sample_id] for sample_id in sample_ids]
      ),
    )
    self.evaluation_runs = SimpleNamespace(
      create=AsyncMock(side_effect=self._create_run),
      get_by_id=AsyncMock(side_effect=lambda run_id: persistence.runs_by_id.get(run_id)),
      get_by_id_or_raise=AsyncMock(side_effect=lambda run_id: persistence.runs_by_id[run_id]),
      update_status_if_current=AsyncMock(side_effect=self._update_run_status),
    )
    self.evaluation_run_metrics = SimpleNamespace(
      create_many=AsyncMock(side_effect=self._create_run_metrics),
      list_metrics=AsyncMock(side_effect=lambda run_id: persistence.run_metric_names.get(run_id, [])),
    )
    self.evaluation_run_plan_samples = SimpleNamespace(
      create_many=AsyncMock(side_effect=self._create_plan_samples),
      list_sample_ids=AsyncMock(side_effect=lambda run_id: persistence.run_plan_sample_ids.get(run_id, [])),
    )
    self.evaluation_run_samples = SimpleNamespace(
      create=AsyncMock(side_effect=self._create_run_sample),
      update_status=AsyncMock(side_effect=self._update_run_sample),
      list_by_evaluation_run=AsyncMock(side_effect=lambda run_id: persistence.run_samples_by_run.get(run_id, [])),
    )
    self.traces = SimpleNamespace(
      get_by_id=AsyncMock(side_effect=lambda trace_id: persistence.traces_by_id.get(trace_id)),
      create=AsyncMock(side_effect=lambda trace: persistence.traces_by_id.setdefault(trace.external_id, trace)),
    )
    self.spans = SimpleNamespace(
      bulk_create=AsyncMock(
        side_effect=lambda spans: persistence.spans_by_id.update({span.external_id: span for span in spans})
      ),
      list_by_trace=AsyncMock(
        side_effect=lambda trace_id: [span for span in persistence.spans_by_id.values() if span.trace_id == trace_id]
      ),
    )
    self.ground_truths = SimpleNamespace(
      list_by_sample=AsyncMock(side_effect=lambda sample_id: persistence.ground_truths_by_sample.get(sample_id, [])),
    )
    self.metrics = SimpleNamespace(
      upsert_from_registry=AsyncMock(side_effect=self._upsert_metric),
    )
    self.span_types = SimpleNamespace(
      upsert_from_registry=AsyncMock(side_effect=self._upsert_span_type),
    )
    self.metric_target_span_types = SimpleNamespace(
      get_span_types_for_metric=AsyncMock(side_effect=self._get_target_span_types),
      upsert_from_registry=AsyncMock(side_effect=self._upsert_metric_target),
      list_all=AsyncMock(side_effect=lambda: list(persistence.metric_targets.values())),
    )
    self.metric_computations = SimpleNamespace(
      create=AsyncMock(side_effect=self._create_computation),
    )

  async def _get_or_create_agent(self, name: str, version_tag: str) -> Agent:
    for agent in self.persistence.agents_by_id.values():
      if agent.name == name and agent.version_tag == version_tag:
        return agent
    agent = Agent(id=uuid4(), name=name, version_tag=version_tag)
    self.persistence.agents_by_id[agent.id] = agent
    return agent

  async def _list_samples_by_dataset(self, dataset_id: Any) -> list[Sample]:
    return [sample for sample in self.persistence.samples_by_id.values() if sample.dataset_id == dataset_id]

  async def _create_run(self, run: EvaluationRun) -> EvaluationRun:
    self.persistence.runs_by_id[run.id] = run
    return run

  async def _update_run_status(self, **kwargs: Any) -> EvaluationRun | None:
    run = self.persistence.runs_by_id[kwargs['run_id']]
    if run.status not in kwargs['current_statuses']:
      return None
    run.status = kwargs['status']
    run.end_time = kwargs['end_time']
    return run

  async def _create_run_metrics(self, run_id: Any, metric_names: list[str]) -> None:
    self.persistence.run_metric_names[run_id] = list(metric_names)

  async def _create_plan_samples(self, run_id: Any, sample_ids: list[Any]) -> None:
    self.persistence.run_plan_sample_ids[run_id] = list(sample_ids)

  async def _create_run_sample(self, run_sample: EvaluationRunSample) -> EvaluationRunSample:
    self.persistence.run_samples_by_run.setdefault(run_sample.evaluation_run_id, []).append(run_sample)
    return run_sample

  async def _update_run_sample(self, **kwargs: Any) -> EvaluationRunSample:
    for run_samples in self.persistence.run_samples_by_run.values():
      for run_sample in run_samples:
        if run_sample.id == kwargs['run_sample_id']:
          run_sample.status = kwargs['status']
          if kwargs['trace_id'] is not None:
            run_sample.trace_id = kwargs['trace_id']
          run_sample.ended_at = kwargs['ended_at']
          run_sample.error_message = kwargs['error_message']
          run_sample.metadata = kwargs['metadata']
          return run_sample
    raise AssertionError(f'Unknown evaluation run sample {kwargs["run_sample_id"]}')

  async def _upsert_metric(self, metric: Metric) -> Metric:
    self.persistence.metrics_by_name[metric.name] = metric
    return metric

  async def _upsert_span_type(self, span_type: SpanType) -> SpanType:
    self.persistence.span_types_by_name[span_type.name] = span_type
    return span_type

  async def _get_target_span_types(self, metric_name: str) -> list[str]:
    return [span_type for (metric, span_type) in self.persistence.metric_targets if metric == metric_name]

  async def _upsert_metric_target(self, target: MetricTargetSpanType) -> MetricTargetSpanType:
    self.persistence.metric_targets[(target.metric, target.span_type)] = target
    return target

  async def _create_computation(self, computation: Any) -> Any:
    self.persistence.computations_by_run_sample.setdefault(computation.evaluation_run_sample_id, []).append(computation)
    return computation


class _CustomTraceIntegration:
  def __init__(self, persistence: _MemoryPersistence) -> None:
    self.persistence = persistence
    self.request_ids: list[str] = []

  async def process_trace(self, request_id: str) -> TraceProcessingResult:
    self.request_ids.append(request_id)
    trace = Trace(
      external_id=f'trace-{request_id}',
      start_time=datetime(2026, 1, 1, tzinfo=timezone.utc),
      end_time=datetime(2026, 1, 1, 0, 0, 1, tzinfo=timezone.utc),
    )
    span = Span(
      external_id=f'span-{request_id}',
      trace_id=trace.external_id,
      span_type='agent_root',
      name='custom-agent-root',
      start_time=trace.start_time,
      end_time=trace.end_time,
      input_data='question',
      output_data='answer',
    )
    self.persistence.traces_by_id[trace.external_id] = trace
    self.persistence.spans_by_id[span.external_id] = span
    return TraceProcessingResult(trace=trace, spans=[span])


class _EndToEndMetric(_CustomMetric):
  def __init__(self) -> None:
    self.span_ids: list[str] = []

  async def compute(self, span: Span, ground_truth: GroundTruth | None) -> MetricComputationResult:
    del ground_truth
    self.span_ids.append(span.external_id)
    return MetricComputationResult(score=0.75, reasoning='custom metric ran')


def _run(agent_id: Any | None = None, dataset_id: Any | None = None, **kwargs: Any) -> EvaluationRun:
  return EvaluationRun(
    id=uuid4(),
    agent_id=agent_id or uuid4(),
    dataset_id=dataset_id or uuid4(),
    status=EvaluationStatus.COMPLETED,
    start_time=kwargs.pop('start_time', datetime.now(timezone.utc)),
    **kwargs,
  )


class TestExtensionPoints(unittest.IsolatedAsyncioTestCase):
  def test_available_metric_names_include_custom_inventory_and_reject_duplicates(self) -> None:
    metric = _CustomMetric()
    service = EvaluationService(
      settings=Settings(), db_manager=cast(DatabaseManager, object()), custom_metrics=[metric]
    )

    self.assertIn(metric.name, service.available_metric_names())

    # A duplicate custom name is a configuration error, so construction fails rather than
    # every later start request returning 400.
    with self.assertRaisesRegex(MetricSelectionError, 'duplicate'):
      EvaluationService(
        settings=Settings(),
        db_manager=cast(DatabaseManager, object()),
        custom_metrics=[cast(Any, SimpleNamespace(name='LLM_CALLS'))],
      )

  async def test_custom_metric_is_registered_without_builtins(self) -> None:
    metric = _CustomMetric()
    registry = _build_metric_registry(
      db_manager=cast(DatabaseManager, object()),
      custom_metrics=[metric],
      include_builtin_metrics=False,
    )

    self.assertEqual(registry.list_registered(), [metric])
    self.assertIs(registry.get(metric.name), metric)

  async def test_fresh_run_uses_combined_inventory(self) -> None:
    dataset_id = uuid4()
    agent = Agent(id=uuid4(), name='custom-agent', version_tag='v1')
    sample = Sample(id=uuid4(), dataset_id=dataset_id, input_prompt='question')
    expected_run = _run(agent.id, dataset_id)
    caller = MagicMock()
    service = EvaluationService(
      settings=Settings(),
      db_manager=cast(DatabaseManager, object()),
      custom_metrics=[_CustomMetric()],
      include_builtin_metrics=False,
      callers_by_agent_name={'custom-agent': caller},
    )
    execution = _ExecutionStub(expected_run)
    repositories = _fake_repositories(agent=agent, samples=[sample])

    with (
      patch('syllo_eval.service.UnitOfWork', return_value=_AsyncRepositoryContext(repositories)),
      patch.object(service, '_build_execution_from_plan', AsyncMock(return_value=execution)) as build_execution,
    ):
      result = await service.run_evaluation(
        agent_name='custom-agent',
        agent_version_tag='v1',
        dataset_id=dataset_id,
        selected_metric_names=[' CUSTOM_METRIC '],
      )

    self.assertEqual(result, expected_run)
    build_args = cast(Any, build_execution.await_args)
    self.assertEqual(build_args.kwargs['selected_metric_names'], ['custom_metric'])

  async def test_public_service_runs_all_injected_extensions_and_repeats_without_trace_dependencies(self) -> None:
    dataset_id = uuid4()
    agent = Agent(id=uuid4(), name='custom-agent', version_tag='v1')
    sample = Sample(id=uuid4(), dataset_id=dataset_id, input_prompt='question')
    persistence = _MemoryPersistence(agent, sample)
    metric = _EndToEndMetric()
    caller = MagicMock()
    caller.call = AsyncMock(return_value='request-1')
    trace_integration = _CustomTraceIntegration(persistence)
    service = EvaluationService(
      settings=Settings(),
      db_manager=cast(DatabaseManager, object()),
      custom_metrics=[metric],
      include_builtin_metrics=False,
      callers_by_agent_name={'custom-agent': caller},
      trace_integration=trace_integration,
    )

    def memory_uow(_db_manager: Any) -> _AsyncRepositoryContext:
      return _AsyncRepositoryContext(_MemoryRepositoryContext(persistence))

    with (
      patch('syllo_eval.service.UnitOfWork', side_effect=memory_uow),
      patch('syllo_eval.service.PhoenixClient') as phoenix_factory,
      patch('syllo_eval.evaluation.metric_registry.UnitOfWork', side_effect=memory_uow),
      patch('syllo_eval.evaluation.metric_registry.TransactionalUnitOfWork', side_effect=memory_uow),
      patch('syllo_eval.evaluation.metric_planner.UnitOfWork', side_effect=memory_uow),
      patch('syllo_eval.evaluation.plan_executor.UnitOfWork', side_effect=memory_uow),
      patch('syllo_eval.execution.sample_executor.UnitOfWork', side_effect=memory_uow),
      patch('syllo_eval.orchestration.evaluation_orchestrator.UnitOfWork', side_effect=memory_uow),
      patch('syllo_eval.orchestration.evaluation_orchestrator.TransactionalUnitOfWork', side_effect=memory_uow),
    ):
      fresh_run = await service.run_evaluation(
        agent_name='custom-agent',
        agent_version_tag='v1',
        dataset_id=dataset_id,
        selected_metric_names=['custom_metric'],
      )

      self.assertEqual(fresh_run.status, EvaluationStatus.COMPLETED)
      caller.call.assert_awaited_once_with(sample)
      self.assertEqual(trace_integration.request_ids, ['request-1'])
      self.assertEqual(metric.span_ids, ['span-request-1'])
      phoenix_factory.assert_not_called()

      fresh_sample = persistence.run_samples_by_run[fresh_run.id][0]
      fresh_computations = persistence.computations_by_run_sample[fresh_sample.id]
      self.assertEqual(fresh_sample.status, EvaluationSampleStatus.COMPLETED)
      self.assertEqual(fresh_sample.trace_id, 'trace-request-1')
      self.assertEqual(len(fresh_computations), 1)
      self.assertEqual(fresh_computations[0].metric, 'custom_metric')
      self.assertEqual(fresh_computations[0].score, 0.75)

      repeated = await (
        await service.repeat_evaluation(
          source_run_id=fresh_run.id,
          metrics=['custom_metric'],
        )
      ).execute()
      phoenix_factory.assert_not_called()

    self.assertEqual(repeated.status, EvaluationStatus.COMPLETED)
    self.assertEqual(repeated.source_run_id, fresh_run.id)
    self.assertEqual(caller.call.await_count, 1)
    self.assertEqual(trace_integration.request_ids, ['request-1'])
    self.assertEqual(metric.span_ids, ['span-request-1', 'span-request-1'])
    repeated_sample = persistence.run_samples_by_run[repeated.id][0]
    repeated_computations = persistence.computations_by_run_sample[repeated_sample.id]
    self.assertEqual(repeated_sample.status, EvaluationSampleStatus.COMPLETED)
    self.assertEqual(len(repeated_computations), 1)
    self.assertEqual(repeated_computations[0].metric, 'custom_metric')
    self.assertEqual(repeated_computations[0].status, MetricComputationStatus.COMPLETED)
    self.assertEqual(repeated_sample.trace_id, 'trace-request-1')
    self.assertEqual(repeated_computations[0].score, 0.75)

  async def test_imported_traces_are_normalized_by_named_adapter_scored_and_repeatable(self) -> None:
    dataset_id = uuid4()
    agent = Agent(id=uuid4(), name='custom-agent', version_tag='v1')
    sample = Sample(id=uuid4(), dataset_id=dataset_id, input_prompt='question')
    persistence = _MemoryPersistence(agent, sample)
    metric = _EndToEndMetric()
    adapter = MagicMock(wraps=PhoenixTraceAdapter())
    service = EvaluationService(
      settings=Settings(),
      db_manager=cast(DatabaseManager, object()),
      custom_metrics=[metric],
      include_builtin_metrics=False,
      trace_adapters_by_name={'Custom-Source': adapter},
    )

    def memory_uow(_db_manager: Any) -> _AsyncRepositoryContext:
      return _AsyncRepositoryContext(_MemoryRepositoryContext(persistence))

    with (
      patch('syllo_eval.service.UnitOfWork', side_effect=memory_uow),
      patch('syllo_eval.service.PhoenixClient') as phoenix_factory,
      patch('syllo_eval.service.TransactionalUnitOfWork', side_effect=memory_uow),
      patch('syllo_eval.evaluation.metric_registry.UnitOfWork', side_effect=memory_uow),
      patch('syllo_eval.evaluation.metric_registry.TransactionalUnitOfWork', side_effect=memory_uow),
      patch('syllo_eval.evaluation.metric_planner.UnitOfWork', side_effect=memory_uow),
      patch('syllo_eval.evaluation.plan_executor.UnitOfWork', side_effect=memory_uow),
      patch('syllo_eval.execution.sample_executor.UnitOfWork', side_effect=memory_uow),
      patch('syllo_eval.orchestration.evaluation_orchestrator.UnitOfWork', side_effect=memory_uow),
      patch('syllo_eval.orchestration.evaluation_orchestrator.TransactionalUnitOfWork', side_effect=memory_uow),
    ):
      imported = await (
        await service.import_evaluation(
          agent_name='custom-agent',
          agent_version_tag='v1',
          dataset_id=dataset_id,
          traces=[phoenix_trace('trace-1', 'question')],
          trace_adapter_name='custom-source',
          selected_metric_names=['custom_metric'],
        )
      ).execute()
      repeated = await (await service.repeat_evaluation(source_run_id=imported.id)).execute()
      phoenix_factory.assert_not_called()

    adapter.normalize.assert_called_once()
    self.assertEqual(imported.status, EvaluationStatus.COMPLETED)
    self.assertEqual(persistence.run_plan_sample_ids[imported.id], [sample.id])
    self.assertIn('trace-1', persistence.traces_by_id)
    for run in (imported, repeated):
      run_sample = persistence.run_samples_by_run[run.id][0]
      self.assertEqual((run_sample.status, run_sample.trace_id), (EvaluationSampleStatus.COMPLETED, 'trace-1'))
    self.assertEqual(repeated.status, EvaluationStatus.COMPLETED)
    self.assertEqual(metric.span_ids, ['trace-1-root', 'trace-1-root'])

  async def test_import_stores_all_traces_in_one_transaction_that_a_conflict_rolls_back(self) -> None:
    dataset_id = uuid4()
    samples = [Sample(id=uuid4(), dataset_id=dataset_id, input_prompt=prompt) for prompt in ('first', 'second')]
    persistence = _MemoryPersistence(Agent(id=uuid4(), name='custom-agent', version_tag='v1'), samples[0])
    persistence.samples_by_id[samples[1].id] = samples[1]
    stored = PhoenixTraceAdapter().normalize('trace-2', phoenix_trace('trace-2', 'other').spans)
    persistence.traces_by_id['trace-2'] = stored.trace
    persistence.spans_by_id.update({span.external_id: span for span in stored.spans})
    transaction = _RecordingTransaction(_MemoryRepositoryContext(persistence))
    service = EvaluationService(
      settings=Settings(), db_manager=cast(DatabaseManager, object()), custom_metrics=[_CustomMetric()]
    )

    with (
      patch(
        'syllo_eval.service.UnitOfWork',
        side_effect=lambda _db: _AsyncRepositoryContext(_MemoryRepositoryContext(persistence)),
      ),
      patch('syllo_eval.service.TransactionalUnitOfWork', return_value=transaction) as transactional_uow,
      self.assertRaisesRegex(TraceImportError, 'Trace trace-2 could not be stored'),
    ):
      await service.import_evaluation(
        agent_name='custom-agent',
        agent_version_tag='v1',
        dataset_id=dataset_id,
        traces=[phoenix_trace('trace-1', 'first'), phoenix_trace('trace-2', 'second')],
        selected_metric_names=['custom_metric'],
      )

    transactional_uow.assert_called_once()
    self.assertIs(transaction.exit_type, TraceImportError)

  async def test_unexecuted_evaluation_handle_closes_prepared_resources(self) -> None:
    expected_run = _run()
    execution = _ExecutionStub(expected_run)
    service = EvaluationService(
      settings=Settings(),
      db_manager=cast(DatabaseManager, object()),
      custom_metrics=[_CustomMetric()],
      include_builtin_metrics=False,
      callers_by_agent_name={'custom-agent': MagicMock()},
    )

    with patch.object(service, '_build_execution', AsyncMock(return_value=execution)):
      handle = await service.create_evaluation(
        agent_name='custom-agent',
        agent_version_tag='v1',
        dataset_id=uuid4(),
        selected_metric_names=['custom_metric'],
      )
      await handle.aclose()

    execution.close.assert_awaited_once_with()

  async def test_repeat_uses_current_custom_inventory_without_phoenix(self) -> None:
    agent_id = uuid4()
    dataset_id = uuid4()
    source_run = _run(agent_id, dataset_id)
    source_sample = EvaluationRunSample(
      id=uuid4(),
      evaluation_run_id=source_run.id,
      sample_id=uuid4(),
      trace_id='trace-1',
      status=EvaluationSampleStatus.COMPLETED,
    )
    agent = Agent(id=agent_id, name='custom-agent', version_tag='v1')
    repositories = _fake_repositories(agent=agent)
    repositories.evaluation_runs.get_by_id.return_value = source_run
    repositories.evaluation_run_metrics.list_metrics.return_value = ['old_metric']
    repositories.evaluation_run_samples.list_by_evaluation_run.return_value = [source_sample]
    repeated_run = _run(agent_id, dataset_id, source_run_id=source_run.id)
    orchestrator = MagicMock()
    orchestrator.create_run = AsyncMock(return_value=repeated_run)
    orchestrator.execute_run = AsyncMock(return_value=repeated_run)
    phoenix = MagicMock()
    service = EvaluationService(
      settings=Settings(),
      db_manager=cast(DatabaseManager, object()),
      custom_metrics=[_CustomMetric()],
      include_builtin_metrics=False,
    )

    with (
      patch('syllo_eval.service.UnitOfWork', return_value=_AsyncRepositoryContext(repositories)),
      patch('syllo_eval.service.MetricRegistry.sync_with_persistence', new_callable=AsyncMock),
      patch('syllo_eval.service.EvaluationOrchestrator', return_value=orchestrator),
      patch('syllo_eval.service.PhoenixClient', phoenix),
    ):
      handle = await service.repeat_evaluation(source_run_id=source_run.id, metrics=['custom_metric'])

    self.assertEqual(handle.run, repeated_run)
    create_args = cast(Any, orchestrator.create_run.await_args)
    self.assertEqual(create_args.args[0].selected_metric_names, ['custom_metric'])
    phoenix.assert_not_called()

  async def test_explicit_caller_mapping_dispatches_by_agent_name(self) -> None:
    caller = MagicMock()
    caller.call = AsyncMock(return_value='request-1')
    service = EvaluationService(
      settings=Settings(),
      db_manager=cast(DatabaseManager, object()),
      callers_by_agent_name={'demo-agent': caller},
    )
    dispatcher = service._build_agent_dispatcher('demo-agent')
    agent = Agent(id=uuid4(), name='demo-agent', version_tag='v1')
    sample = Sample(id=uuid4(), dataset_id=uuid4(), input_prompt='question')

    self.assertEqual(await dispatcher.call(agent, sample), 'request-1')
    caller.call.assert_awaited_once_with(sample)

  def test_engine_registers_no_caller_by_default(self) -> None:
    service = EvaluationService(settings=Settings(), db_manager=cast(DatabaseManager, object()))

    with self.assertRaisesRegex(AgentCallerSelectionError, 'No caller registered for agent'):
      service._build_agent_dispatcher('demo-agent')

  def test_duplicate_caller_names_are_rejected_at_construction(self) -> None:
    with self.assertRaisesRegex(ValueError, 'non-empty and unique'):
      EvaluationService(
        settings=Settings(),
        db_manager=cast(DatabaseManager, object()),
        callers_by_agent_name={'demo-agent': MagicMock(), 'Demo-Agent ': MagicMock()},
      )

  def test_trace_dependencies_are_mutually_exclusive(self) -> None:
    with self.assertRaisesRegex(ValueError, 'either trace_client or trace_integration'):
      EvaluationService(
        settings=Settings(),
        db_manager=cast(DatabaseManager, object()),
        trace_client=cast(Any, object()),
        trace_integration=cast(Any, object()),
      )

  async def test_injected_trace_integration_is_used_for_fresh_execution(self) -> None:
    integration = MagicMock()
    orchestrator = MagicMock()
    service = EvaluationService(
      settings=Settings(),
      db_manager=cast(DatabaseManager, object()),
      include_builtin_metrics=False,
      trace_integration=integration,
    )

    with (
      patch('syllo_eval.service.MetricRegistry.sync_with_persistence', new_callable=AsyncMock),
      patch('syllo_eval.service.SampleExecutor') as sample_executor,
      patch('syllo_eval.service.EvaluationOrchestrator', return_value=orchestrator),
    ):
      await service._build_execution_from_plan(
        agent_id=uuid4(),
        dataset_id=uuid4(),
        selected_metric_names=[],
        planned_sample_ids=[],
        max_concurrent_samples=None,
        max_concurrent_tasks=None,
        sample_trace_timeout=None,
        sample_compute_timeout=None,
      )

    self.assertIs(sample_executor.call_args.kwargs['trace_processor'], integration)

  async def test_injected_trace_client_reaches_default_trace_processor(self) -> None:
    db_manager = cast(DatabaseManager, object())
    trace_client = MagicMock()
    trace_processor = MagicMock()
    orchestrator = MagicMock()
    service = EvaluationService(
      settings=Settings(),
      db_manager=db_manager,
      include_builtin_metrics=False,
      trace_client=trace_client,
    )

    with (
      patch('syllo_eval.service.MetricRegistry.sync_with_persistence', new_callable=AsyncMock),
      patch('syllo_eval.service.TraceProcessor', return_value=trace_processor) as trace_processor_factory,
      patch('syllo_eval.service.EvaluationOrchestrator', return_value=orchestrator),
    ):
      await service._build_execution_from_plan(
        agent_id=uuid4(),
        dataset_id=uuid4(),
        selected_metric_names=[],
        planned_sample_ids=[],
        max_concurrent_samples=None,
        max_concurrent_tasks=None,
        sample_trace_timeout=None,
        sample_compute_timeout=None,
      )

    trace_processor_factory.assert_called_once_with(trace_client=trace_client, db_manager=db_manager, adapter=None)

  async def test_evaluation_overrides_reach_orchestrator_and_run_snapshot(self) -> None:
    orchestrator = MagicMock()
    service = EvaluationService(
      settings=Settings(),
      db_manager=cast(DatabaseManager, object()),
      include_builtin_metrics=False,
      trace_integration=MagicMock(),
    )

    with (
      patch('syllo_eval.service.MetricRegistry.sync_with_persistence', new_callable=AsyncMock),
      patch('syllo_eval.service.SampleExecutor'),
      patch('syllo_eval.service.EvaluationOrchestrator', return_value=orchestrator) as orchestrator_factory,
    ):
      execution = await service._build_execution_from_plan(
        agent_id=uuid4(),
        dataset_id=uuid4(),
        selected_metric_names=[],
        planned_sample_ids=[],
        max_concurrent_samples=2,
        max_concurrent_tasks=3,
        sample_trace_timeout=4.0,
        sample_compute_timeout=5.0,
      )

    orchestrator_kwargs = cast(Any, orchestrator_factory.call_args).kwargs
    self.assertEqual(orchestrator_kwargs['max_concurrent_samples'], 2)
    self.assertEqual(orchestrator_kwargs['sample_trace_timeout_seconds'], 4.0)
    self.assertEqual(orchestrator_kwargs['sample_compute_timeout_seconds'], 5.0)
    snapshot = cast(dict[str, object], execution.config.config)
    self.assertEqual(snapshot['max_concurrent_samples'], 2)
    self.assertEqual(snapshot['max_concurrent_tasks'], 3)
    self.assertEqual(snapshot['sample_trace_timeout_seconds'], 4.0)
    self.assertEqual(snapshot['sample_compute_timeout_seconds'], 5.0)

  async def test_caller_owned_database_is_health_checked_but_not_closed(self) -> None:
    db_manager = MagicMock()
    db_manager.health_check = AsyncMock(return_value=True)
    db_manager.initialize_async = AsyncMock()
    db_manager.close_async = AsyncMock()
    service = EvaluationService(settings=Settings(), db_manager=db_manager)

    await service.initialize()
    await service.close()

    db_manager.health_check.assert_awaited_once_with()
    db_manager.initialize_async.assert_not_awaited()
    db_manager.close_async.assert_not_awaited()

  async def test_custom_metric_runs_through_planner_and_executor(self) -> None:
    db_manager = cast(DatabaseManager, object())
    sample_id = uuid4()
    span = Span(
      external_id='span-1',
      trace_id='trace-1',
      span_type='agent_root',
      name='agent',
      start_time=datetime.now(timezone.utc),
      end_time=datetime.now(timezone.utc),
      input_data='question',
      output_data='answer',
    )
    metric = _CustomMetric()
    registry = MetricRegistry(db_manager)
    registry.register(metric)
    planner = MetricPlanner(registry, db_manager)
    metric_computations = SimpleNamespace(create=AsyncMock(side_effect=lambda computation: computation))
    repositories = SimpleNamespace(
      spans=SimpleNamespace(list_by_trace=AsyncMock(return_value=[span])),
      ground_truths=SimpleNamespace(list_by_sample=AsyncMock(return_value=[])),
      metric_computations=metric_computations,
    )
    repository_context = _AsyncRepositoryContext(repositories)

    with (
      patch('syllo_eval.evaluation.metric_planner.UnitOfWork', return_value=repository_context),
      patch('syllo_eval.evaluation.plan_executor.UnitOfWork', return_value=repository_context),
    ):
      computations = await PlanExecutor(db_manager).execute(
        planner.iter_plan_items(trace_id='trace-1', sample_id=sample_id),
        evaluation_run_sample_id=uuid4(),
      )

    self.assertEqual(len(computations), 1)
    self.assertEqual(computations[0].metric, metric.name)
    self.assertEqual(computations[0].score, 1.0)
    metric_computations.create.assert_awaited_once()

  async def test_builtin_judge_resource_closes_after_success(self) -> None:
    judge_client, execution = await self._build_judge_execution()
    await execution.execute_run(_run())
    judge_client.aclose.assert_awaited_once_with()

  async def test_service_inherits_configured_options_without_creating_unused_clients(self) -> None:
    service = EvaluationService(
      settings=Settings(
        evaluation=EvaluationSettings(
          max_concurrent_samples=2,
          max_concurrent_tasks=3,
          sample_trace_timeout=4,
          sample_compute_timeout=5,
        ),
        llm_judge=LlmJudgeSettings(provider='openai'),
      ),
      db_manager=cast(DatabaseManager, object()),
      trace_integration=MagicMock(),
    )
    with (
      patch('syllo_eval.service.MetricRegistry.sync_with_persistence', new_callable=AsyncMock),
      patch('syllo_eval.service.build_llm_judge_client') as build_judge,
    ):
      execution = await service._build_execution_from_plan(
        agent_id=uuid4(),
        dataset_id=uuid4(),
        selected_metric_names=['llm_calls'],
        planned_sample_ids=[],
        max_concurrent_samples=None,
        max_concurrent_tasks=None,
        sample_trace_timeout=None,
        sample_compute_timeout=None,
      )
    snapshot = cast(dict[str, object], execution.config.config)
    self.assertEqual(snapshot['max_concurrent_samples'], 2)
    self.assertEqual(snapshot['max_concurrent_tasks'], 3)
    self.assertEqual(snapshot['sample_trace_timeout_seconds'], 4)
    self.assertEqual(snapshot['sample_compute_timeout_seconds'], 5)
    build_judge.assert_not_called()
    await execution.close()

  async def test_judge_closes_when_later_provider_construction_fails(self) -> None:
    judge = MagicMock(aclose=AsyncMock())
    service = EvaluationService(
      settings=Settings(llm_judge=LlmJudgeSettings(provider='openai')),
      db_manager=cast(DatabaseManager, object()),
    )
    with (
      patch('syllo_eval.service.build_llm_judge_client', return_value=judge),
      patch.object(service, '_build_claim_extractor_client', side_effect=RuntimeError('provider setup failed')),
    ):
      with self.assertRaisesRegex(RuntimeError, 'provider setup failed'):
        await service._build_execution_from_plan(
          agent_id=uuid4(),
          dataset_id=uuid4(),
          planned_sample_ids=[],
          selected_metric_names=['contextual_recall_document_claim_extractor'],
          max_concurrent_samples=None,
          max_concurrent_tasks=None,
          sample_trace_timeout=None,
          sample_compute_timeout=None,
        )
    judge.aclose.assert_awaited_once_with()

  async def test_builtin_judge_resource_closes_after_failure(self) -> None:
    judge_client, execution = await self._build_judge_execution()
    execution.orchestrator.execute_run.side_effect = RuntimeError('failed')

    with self.assertRaisesRegex(RuntimeError, 'failed'):
      await execution.execute_run(_run())
    judge_client.aclose.assert_awaited_once_with()

  async def test_builtin_judge_resource_closes_after_cancellation(self) -> None:
    judge_client, execution = await self._build_judge_execution()
    execution.orchestrator.execute_run.side_effect = asyncio.CancelledError()

    with self.assertRaises(asyncio.CancelledError):
      await execution.execute_run(_run())
    judge_client.aclose.assert_awaited_once_with()

  async def _build_judge_execution(self) -> tuple[MagicMock, Any]:
    judge_client = MagicMock()
    judge_client.aclose = AsyncMock()
    orchestrator = MagicMock()
    orchestrator.execute_run = AsyncMock(return_value=_run())
    service = EvaluationService(
      settings=Settings(llm_judge=LlmJudgeSettings(provider='openai')),
      db_manager=cast(DatabaseManager, object()),
    )

    with (
      patch('syllo_eval.service.build_llm_judge_client', return_value=judge_client),
      patch('syllo_eval.service.MetricRegistry.sync_with_persistence', new_callable=AsyncMock),
      patch('syllo_eval.service.EvaluationOrchestrator', return_value=orchestrator),
    ):
      execution = await service._build_execution_from_plan(
        agent_id=uuid4(),
        dataset_id=uuid4(),
        selected_metric_names=['answer_correctness_judge'],
        planned_sample_ids=[],
        max_concurrent_samples=None,
        max_concurrent_tasks=None,
        sample_trace_timeout=None,
        sample_compute_timeout=None,
      )
    return judge_client, execution


if __name__ == '__main__':
  unittest.main()
