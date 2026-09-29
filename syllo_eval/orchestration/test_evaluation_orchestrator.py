import asyncio
import unittest
from datetime import datetime, timezone
from typing import cast
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

from syllo_eval.execution.agent_caller import AgentCallDispatcher
from syllo_eval.execution.sample_executor import SampleExecutionResult, SampleExecutor
from syllo_eval.evaluation.metric_planner import MetricPlanner
from syllo_eval.evaluation.metric_registry import MetricRegistry
from syllo_eval.evaluation.metrics.contracts import EvaluationMetric, MetricComputationResult
from syllo_eval.evaluation.plan_executor import PlanExecutor
from syllo_eval.evaluation.trace_processor import TraceProcessor
from syllo_eval.infrastructure.database import DatabaseManager
from syllo_eval.infrastructure import UnitOfWork
from syllo_eval.model import (
  Agent,
  Dataset,
  EvaluationRun,
  EvaluationRunSample,
  EvaluationSampleStatus,
  EvaluationStatus,
  Sample,
  Span,
  SpanType,
  Trace,
)
from syllo_eval.orchestration.evaluation_orchestrator import EvaluationConfig, EvaluationOrchestrator
from syllo_eval.testing_database import setup_test_database as _setup_database


class _FailingTraceClient:
  async def get_trace_id_by_request_id(self, request_id: str) -> str | None:
    del request_id
    return 'trace-unavailable'

  async def get_trace_json(self, trace_id: str) -> list[dict[str, object]]:
    del trace_id
    raise RuntimeError('Trace source unavailable')


class _StaticTraceClient:
  def __init__(self, trace_ids_by_request_id: dict[str, str], records_by_trace_id: dict[str, list[dict[str, object]]]):
    self._trace_ids_by_request_id = trace_ids_by_request_id
    self._records_by_trace_id = records_by_trace_id

  async def get_trace_id_by_request_id(self, request_id: str) -> str | None:
    return self._trace_ids_by_request_id.get(request_id)

  async def get_trace_json(self, trace_id: str) -> list[dict[str, object]]:
    return self._records_by_trace_id.get(trace_id, [])


class _StaticAgentCallDispatcher:
  def __init__(self, request_ids_by_sample_id: dict[object, str]):
    self._request_ids_by_sample_id = request_ids_by_sample_id

  async def call(self, agent: Agent, sample: Sample) -> str:
    del agent
    return self._request_ids_by_sample_id[sample.id]


class _ConstantScoreMetric(EvaluationMetric):
  def __init__(self, metric_name: str, span_type_name: str):
    self._metric_name = metric_name
    self._span_type_name = span_type_name

  @property
  def name(self) -> str:
    return self._metric_name

  @property
  def description(self) -> str:
    return 'Returns a constant score for integration testing.'

  @property
  def requires_ground_truth(self) -> bool:
    return False

  @property
  def target_span_types(self) -> tuple[str, ...]:
    return (self._span_type_name,)

  async def compute(self, span: Span, ground_truth) -> MetricComputationResult:
    del span, ground_truth
    return MetricComputationResult(score=1.0, reasoning='constant')


class _AgentGroundTruthRequiredMetric(EvaluationMetric):
  def __init__(self, metric_name: str):
    self._metric_name = metric_name

  @property
  def name(self) -> str:
    return self._metric_name

  @property
  def description(self) -> str:
    return 'Test-only placeholder metric for preconfigured agent mappings.'

  @property
  def requires_ground_truth(self) -> bool:
    return True

  @property
  def target_span_types(self) -> tuple[str, ...]:
    return ('agent',)

  async def compute(self, span: Span, ground_truth) -> MetricComputationResult:
    del span, ground_truth
    return MetricComputationResult(score=0.0, reasoning='placeholder')


def _stub_orchestrator(
  *, sample_executor: SampleExecutor | None = None, max_concurrent_samples: int = 1
) -> EvaluationOrchestrator:
  return EvaluationOrchestrator(
    db_manager=cast(DatabaseManager, object()),
    sample_executor=sample_executor if sample_executor is not None else cast(SampleExecutor, object()),
    max_concurrent_samples=max_concurrent_samples,
  )


class TestEvaluationOrchestratorInit(unittest.TestCase):
  def test_rejects_non_positive_max_concurrent_samples(self) -> None:
    with self.assertRaisesRegex(ValueError, 'max_concurrent_samples must be >= 1'):
      _stub_orchestrator(max_concurrent_samples=0)

  def test_rejects_non_positive_sample_trace_timeout_seconds(self) -> None:
    with self.assertRaisesRegex(ValueError, 'sample_trace_timeout_seconds must be > 0'):
      EvaluationOrchestrator(
        db_manager=cast(DatabaseManager, object()),
        sample_executor=cast(SampleExecutor, object()),
        sample_trace_timeout_seconds=0,
      )

  def test_rejects_non_positive_sample_compute_timeout_seconds(self) -> None:
    with self.assertRaisesRegex(ValueError, 'sample_compute_timeout_seconds must be > 0'):
      EvaluationOrchestrator(
        db_manager=cast(DatabaseManager, object()),
        sample_executor=cast(SampleExecutor, object()),
        sample_compute_timeout_seconds=0,
      )


def _make_run(*, status: EvaluationStatus, run_id=None, agent_id=None, dataset_id=None) -> EvaluationRun:
  now = datetime.now(tz=timezone.utc)
  return EvaluationRun(
    id=run_id or uuid4(),
    agent_id=agent_id or uuid4(),
    dataset_id=dataset_id or uuid4(),
    status=status,
    start_time=now,
    end_time=now if status != EvaluationStatus.RUNNING else None,
  )


def _make_result(run_id, sample_id, trace_id, error: str | None = None) -> SampleExecutionResult:
  return SampleExecutionResult(
    evaluation_run_sample=EvaluationRunSample(
      id=uuid4(),
      evaluation_run_id=run_id,
      sample_id=sample_id,
      trace_id=trace_id,
    ),
    computations=[],
    error=error,
  )


def _patch_unit_of_work(repo: AsyncMock):
  """Patch UnitOfWork in the orchestrator module to yield a uow exposing the given repo."""
  mock_uow = MagicMock()
  mock_uow.evaluation_run_samples = repo
  mock_uow_cm = MagicMock()
  mock_uow_cm.__aenter__ = AsyncMock(return_value=mock_uow)
  mock_uow_cm.__aexit__ = AsyncMock(return_value=None)
  return patch('syllo_eval.orchestration.evaluation_orchestrator.UnitOfWork', return_value=mock_uow_cm)


def _patch_transactional_unit_of_work(mock_uow: MagicMock):
  mock_uow_cm = MagicMock()
  mock_uow_cm.__aenter__ = AsyncMock(return_value=mock_uow)
  mock_uow_cm.__aexit__ = AsyncMock(return_value=None)
  return patch('syllo_eval.orchestration.evaluation_orchestrator.TransactionalUnitOfWork', return_value=mock_uow_cm)


class TestEvaluationOrchestratorRun(unittest.IsolatedAsyncioTestCase):
  async def test_empty_saved_plan_does_not_load_current_dataset(self):
    orchestrator = _stub_orchestrator()
    run = _make_run(status=EvaluationStatus.RUNNING)
    run.config = {'plan_snapshot_version': 1}
    with (
      patch.object(orchestrator, '_fetch_run', AsyncMock(return_value=run)),
      patch.object(orchestrator, '_fetch_planned_sample_ids', AsyncMock(return_value=[])),
      patch.object(orchestrator, '_fetch_samples_by_ids', AsyncMock(return_value=[])),
      patch.object(orchestrator, '_fetch_samples', AsyncMock()) as fetch_dataset,
      patch.object(orchestrator, '_finalize_run', AsyncMock(return_value=run)) as finalize,
    ):
      await orchestrator.execute_run(run)
    fetch_dataset.assert_not_awaited()
    finalize.assert_awaited_once_with(run.id, EvaluationStatus.FAILED)

  async def test_create_run_creates_evaluation_run(self) -> None:
    orchestrator = _stub_orchestrator()
    created_run = _make_run(status=EvaluationStatus.RUNNING)
    config = EvaluationConfig(agent_id=created_run.agent_id, dataset_id=created_run.dataset_id)

    with patch.object(
      orchestrator, '_create_evaluation_run', AsyncMock(return_value=created_run)
    ) as create_evaluation_run:
      result = await orchestrator.create_run(config)

    self.assertEqual(result, created_run)
    create_evaluation_run.assert_awaited_once_with(config)

  async def test_create_evaluation_run_persists_plan_snapshot_atomically(self) -> None:
    orchestrator = _stub_orchestrator()
    sample_ids = [uuid4(), uuid4()]
    config = EvaluationConfig(
      agent_id=uuid4(),
      dataset_id=uuid4(),
      selected_metric_names=['llm_calls', 'plan_efficiency'],
      planned_sample_ids=sample_ids,
      config={'max_concurrent_samples': 2},
    )
    created_run = _make_run(status=EvaluationStatus.RUNNING, agent_id=config.agent_id, dataset_id=config.dataset_id)
    mock_uow = MagicMock()
    mock_uow.evaluation_runs.create = AsyncMock(return_value=created_run)
    mock_uow.evaluation_run_metrics.create_many = AsyncMock()
    mock_uow.evaluation_run_plan_samples.create_many = AsyncMock()

    with _patch_transactional_unit_of_work(mock_uow):
      result = await orchestrator._create_evaluation_run(config)

    self.assertEqual(result, created_run)
    mock_uow.evaluation_run_metrics.create_many.assert_awaited_once_with(
      created_run.id,
      ['llm_calls', 'plan_efficiency'],
    )
    mock_uow.evaluation_run_plan_samples.create_many.assert_awaited_once_with(created_run.id, sample_ids)

  async def test_run_starts_then_executes(self) -> None:
    orchestrator = _stub_orchestrator()
    created_run = _make_run(status=EvaluationStatus.RUNNING)
    finalized_run = _make_run(
      status=EvaluationStatus.COMPLETED,
      run_id=created_run.id,
      agent_id=created_run.agent_id,
      dataset_id=created_run.dataset_id,
    )
    config = EvaluationConfig(agent_id=created_run.agent_id, dataset_id=created_run.dataset_id)

    with (
      patch.object(orchestrator, 'create_run', AsyncMock(return_value=created_run)) as create_run,
      patch.object(orchestrator, 'execute_run', AsyncMock(return_value=finalized_run)) as execute_run,
    ):
      result = await orchestrator.run(config)

    self.assertEqual(result, finalized_run)
    create_run.assert_awaited_once_with(config)
    execute_run.assert_awaited_once_with(created_run)

  async def test_execute_run_finalizes_failed_without_executing_when_dataset_has_no_samples(self) -> None:
    orchestrator = _stub_orchestrator()
    created_run = _make_run(status=EvaluationStatus.RUNNING)
    finalized_run = _make_run(
      status=EvaluationStatus.FAILED,
      run_id=created_run.id,
      agent_id=created_run.agent_id,
      dataset_id=created_run.dataset_id,
    )
    agent = Agent(id=created_run.agent_id, name='demo-agent', version_tag='v-test')

    with (
      patch.object(orchestrator, '_fetch_run', AsyncMock(return_value=created_run)),
      patch.object(orchestrator, '_fetch_agent', AsyncMock(return_value=agent)),
      patch.object(orchestrator, '_fetch_planned_sample_ids', AsyncMock(return_value=[])),
      patch.object(orchestrator, '_fetch_samples', AsyncMock(return_value=[])),
      patch.object(orchestrator, '_execute_samples', AsyncMock(return_value=[])) as execute_samples,
      patch.object(orchestrator, '_finalize_run', AsyncMock(return_value=finalized_run)) as finalize_run,
    ):
      result = await orchestrator.execute_run(created_run)

    self.assertEqual(result.status, EvaluationStatus.FAILED)
    execute_samples.assert_not_awaited()
    finalize_run.assert_awaited_once_with(created_run.id, EvaluationStatus.FAILED)

  async def test_execute_run_returns_terminal_run_without_sample_work(self) -> None:
    orchestrator = _stub_orchestrator()
    terminal_run = _make_run(status=EvaluationStatus.FAILED)

    with (
      patch.object(orchestrator, '_fetch_run', AsyncMock(return_value=terminal_run)),
      patch.object(orchestrator, '_fetch_agent', AsyncMock()) as fetch_agent,
      patch.object(orchestrator, '_execute_samples', AsyncMock()) as execute_samples,
      patch.object(orchestrator, '_finalize_run', AsyncMock()) as finalize_run,
    ):
      result = await orchestrator.execute_run(terminal_run)

    self.assertEqual(result, terminal_run)
    fetch_agent.assert_not_awaited()
    execute_samples.assert_not_awaited()
    finalize_run.assert_not_awaited()

  async def test_execute_run_finalizes_partially_completed_for_mixed_outcomes(self) -> None:
    orchestrator = _stub_orchestrator()
    created_run = _make_run(status=EvaluationStatus.RUNNING)
    finalized_run = _make_run(
      status=EvaluationStatus.PARTIALLY_COMPLETED,
      run_id=created_run.id,
      agent_id=created_run.agent_id,
      dataset_id=created_run.dataset_id,
    )
    agent = Agent(id=created_run.agent_id, name='demo-agent', version_tag='v-test')
    samples = [
      Sample(id=uuid4(), dataset_id=created_run.dataset_id, input_prompt='ok'),
      Sample(id=uuid4(), dataset_id=created_run.dataset_id, input_prompt='fail'),
    ]
    outcomes = [
      _make_result(created_run.id, samples[0].id, 'trace-ok'),
      _make_result(created_run.id, samples[1].id, None, error='metric pipeline failure'),
    ]

    with (
      patch.object(orchestrator, '_fetch_run', AsyncMock(return_value=created_run)),
      patch.object(orchestrator, '_fetch_agent', AsyncMock(return_value=agent)),
      patch.object(
        orchestrator, '_fetch_planned_sample_ids', AsyncMock(return_value=[sample.id for sample in samples])
      ),
      patch.object(orchestrator, '_fetch_samples_by_ids', AsyncMock(return_value=samples)),
      patch.object(orchestrator, '_execute_samples', AsyncMock(return_value=outcomes)) as execute_samples,
      patch.object(orchestrator, '_finalize_run', AsyncMock(return_value=finalized_run)) as finalize_run,
    ):
      result = await orchestrator.execute_run(created_run)

    self.assertEqual(result.status, EvaluationStatus.PARTIALLY_COMPLETED)
    execute_samples.assert_awaited_once_with(created_run.id, agent, samples)
    finalize_run.assert_awaited_once_with(created_run.id, EvaluationStatus.PARTIALLY_COMPLETED)

  async def test_execute_run_recomputes_repeat_from_source_traces(self) -> None:
    orchestrator = _stub_orchestrator()
    source_run_id = uuid4()
    repeat_run = _make_run(status=EvaluationStatus.RUNNING)
    repeat_run.source_run_id = source_run_id
    sample_id = uuid4()
    source_sample = EvaluationRunSample(
      id=uuid4(),
      evaluation_run_id=source_run_id,
      sample_id=sample_id,
      trace_id='trace-source',
      status=EvaluationSampleStatus.COMPLETED,
    )
    outcome = _make_result(repeat_run.id, sample_id, 'trace-source')
    finalized_run = repeat_run.model_copy(update={'status': EvaluationStatus.COMPLETED})

    with (
      patch.object(orchestrator, '_fetch_run', AsyncMock(return_value=repeat_run)),
      patch.object(orchestrator, '_fetch_planned_sample_ids', AsyncMock(return_value=[sample_id])),
      patch.object(orchestrator, '_fetch_source_run_samples', AsyncMock(return_value=[source_sample])),
      patch.object(
        orchestrator, '_execute_recomputations', AsyncMock(return_value=[outcome])
      ) as execute_recomputations,
      patch.object(orchestrator, '_fetch_agent', AsyncMock()) as fetch_agent,
      patch.object(orchestrator, '_finalize_run', AsyncMock(return_value=finalized_run)),
    ):
      result = await orchestrator.execute_run(repeat_run)

    self.assertEqual(result.status, EvaluationStatus.COMPLETED)
    fetch_agent.assert_not_awaited()
    execute_recomputations.assert_awaited_once_with(repeat_run.id, [(sample_id, 'trace-source')])

  async def test_execute_samples_collects_exceptions_without_aborting(self) -> None:
    dataset_id = uuid4()
    evaluation_run_id = uuid4()
    agent = Agent(id=uuid4(), name='demo-agent', version_tag='v-test')
    sample_one = Sample(id=uuid4(), dataset_id=dataset_id, input_prompt='ok')
    sample_two = Sample(id=uuid4(), dataset_id=dataset_id, input_prompt='crash')

    class _CrashingExecutor:
      async def execute(self, evaluation_run_id, agent, sample, **_):
        del agent
        if sample.id == sample_two.id:
          raise ValueError('No trace found for request_id=request-crash')
        return _make_result(evaluation_run_id, sample.id, 'trace-ok')

    orchestrator = _stub_orchestrator(
      sample_executor=cast(SampleExecutor, _CrashingExecutor()), max_concurrent_samples=2
    )
    crashed_result = _make_result(evaluation_run_id, sample_two.id, None, error='crash persisted')

    with patch.object(orchestrator, '_create_crashed_sample_result', AsyncMock(return_value=crashed_result)) as patched:
      outcomes = await orchestrator._execute_samples(
        evaluation_run_id=evaluation_run_id, agent=agent, samples=[sample_one, sample_two]
      )

    self.assertEqual(len(outcomes), 2)
    self.assertEqual(sum(1 for o in outcomes if o.succeeded), 1)
    self.assertEqual(sum(1 for o in outcomes if not o.succeeded), 1)
    patched.assert_awaited_once()

  async def test_execute_samples_uses_sliding_window_concurrency(self) -> None:
    """A slow sample must not block later samples from starting once a permit frees."""
    dataset_id = uuid4()
    agent = Agent(id=uuid4(), name='demo-agent', version_tag='v-test')
    sample_slow = Sample(id=uuid4(), dataset_id=dataset_id, input_prompt='slow')
    sample_fast = Sample(id=uuid4(), dataset_id=dataset_id, input_prompt='fast')
    sample_after = Sample(id=uuid4(), dataset_id=dataset_id, input_prompt='after')

    release_slow = asyncio.Event()
    sample_after_started = asyncio.Event()

    class _SlidingExecutor:
      async def execute(self, evaluation_run_id, agent, sample, **_):
        del agent
        if sample.id == sample_slow.id:
          await release_slow.wait()
        elif sample.id == sample_after.id:
          sample_after_started.set()
        return _make_result(evaluation_run_id, sample.id, f'trace-{sample.id}')

    orchestrator = _stub_orchestrator(
      sample_executor=cast(SampleExecutor, _SlidingExecutor()), max_concurrent_samples=2
    )
    task = asyncio.create_task(
      orchestrator._execute_samples(
        evaluation_run_id=uuid4(), agent=agent, samples=[sample_slow, sample_fast, sample_after]
      )
    )

    try:
      await asyncio.wait_for(sample_after_started.wait(), timeout=1.0)
    finally:
      release_slow.set()

    outcomes = await task
    self.assertEqual(len(outcomes), 3)
    self.assertTrue(all(o.succeeded for o in outcomes))

  async def test_execute_samples_survives_crash_persistence_failure(self) -> None:
    """If persisting the crashed-sample row itself fails, the run must continue with an in-memory result."""
    dataset_id = uuid4()
    agent = Agent(id=uuid4(), name='demo-agent', version_tag='v-test')
    sample_good = Sample(id=uuid4(), dataset_id=dataset_id, input_prompt='good')
    sample_bad = Sample(id=uuid4(), dataset_id=dataset_id, input_prompt='bad')

    class _PartialCrashExecutor:
      async def execute(self, evaluation_run_id, agent, sample, **_):
        del agent
        if sample.id == sample_bad.id:
          raise ValueError('executor blew up')
        return _make_result(evaluation_run_id, sample.id, f'trace-{sample.id}')

    orchestrator = _stub_orchestrator(
      sample_executor=cast(SampleExecutor, _PartialCrashExecutor()), max_concurrent_samples=2
    )

    with patch.object(
      orchestrator,
      '_create_crashed_sample_result',
      AsyncMock(side_effect=RuntimeError('DB write failed during crash persistence')),
    ):
      outcomes = await orchestrator._execute_samples(
        evaluation_run_id=uuid4(), agent=agent, samples=[sample_good, sample_bad]
      )

    self.assertEqual(len(outcomes), 2)
    self.assertEqual(sum(1 for o in outcomes if o.succeeded), 1)
    self.assertEqual(sum(1 for o in outcomes if not o.succeeded), 1)

  async def test_create_crashed_sample_result_updates_existing_run_sample_row(self) -> None:
    """If a row already exists (created early by SampleExecutor), update it instead of inserting."""
    orchestrator = _stub_orchestrator()
    evaluation_run_id = uuid4()
    sample = Sample(id=uuid4(), dataset_id=uuid4(), input_prompt='x')
    existing_row = EvaluationRunSample(
      id=uuid4(),
      evaluation_run_id=evaluation_run_id,
      sample_id=sample.id,
      status=EvaluationSampleStatus.RUNNING,
      started_at=datetime.now(tz=timezone.utc),
    )

    repo = AsyncMock()
    repo.get_by_run_and_sample = AsyncMock(return_value=existing_row)
    repo.update_status = AsyncMock(
      return_value=existing_row.model_copy(update={'status': EvaluationSampleStatus.FAILED})
    )
    repo.create = AsyncMock()

    with _patch_unit_of_work(repo):
      result = await orchestrator._create_crashed_sample_result(
        evaluation_run_id=evaluation_run_id, sample=sample, error=ValueError('boom')
      )

    repo.update_status.assert_awaited_once()
    assert repo.update_status.await_args is not None
    self.assertEqual(repo.update_status.await_args.kwargs['run_sample_id'], existing_row.id)
    self.assertEqual(repo.update_status.await_args.kwargs['status'], EvaluationSampleStatus.FAILED)
    repo.create.assert_not_called()
    self.assertEqual(result.error, 'boom')

  async def test_create_crashed_sample_result_creates_when_no_row_exists(self) -> None:
    """When no prior row exists, fall back to creating a new FAILED row."""
    orchestrator = _stub_orchestrator()
    evaluation_run_id = uuid4()
    sample = Sample(id=uuid4(), dataset_id=uuid4(), input_prompt='x')
    created_row = EvaluationRunSample(
      id=uuid4(),
      evaluation_run_id=evaluation_run_id,
      sample_id=sample.id,
      status=EvaluationSampleStatus.FAILED,
      started_at=datetime.now(tz=timezone.utc),
      ended_at=datetime.now(tz=timezone.utc),
      error_message='boom',
    )

    repo = AsyncMock()
    repo.get_by_run_and_sample = AsyncMock(return_value=None)
    repo.update_status = AsyncMock()
    repo.create = AsyncMock(return_value=created_row)

    with _patch_unit_of_work(repo):
      result = await orchestrator._create_crashed_sample_result(
        evaluation_run_id=evaluation_run_id, sample=sample, error=ValueError('boom')
      )

    repo.create.assert_awaited_once()
    repo.update_status.assert_not_called()
    self.assertEqual(result.error, 'boom')

  async def test_execute_samples_propagates_cancelled_error(self) -> None:
    class _CancellingExecutor:
      async def execute(self, evaluation_run_id, agent, sample, **_):
        del evaluation_run_id, agent, sample
        raise asyncio.CancelledError()

    orchestrator = _stub_orchestrator(
      sample_executor=cast(SampleExecutor, _CancellingExecutor()), max_concurrent_samples=2
    )
    sample = Sample(id=uuid4(), dataset_id=uuid4(), input_prompt='cancel')

    with self.assertRaises(asyncio.CancelledError):
      await orchestrator._execute_samples(
        evaluation_run_id=uuid4(),
        agent=Agent(id=uuid4(), name='demo-agent', version_tag='v-test'),
        samples=[sample],
      )


class TestEvaluationOrchestratorLifecyclePersistence(unittest.IsolatedAsyncioTestCase):
  async def asyncSetUp(self) -> None:
    try:
      self.db_manager = await _setup_database()
    except RuntimeError as error:
      self.skipTest(str(error))

    suffix = uuid4().hex[:10]

    self.agent_id = uuid4()
    self.dataset_id = uuid4()
    self.sample_one_id = uuid4()
    self.sample_two_id = uuid4()

    self.trace_one_id = f'evaluation_orchestrator_trace_one_{suffix}'
    self.trace_two_id = f'evaluation_orchestrator_trace_two_{suffix}'
    self.request_one_id = f'evaluation_orchestrator_request_one_{suffix}'
    self.request_two_id = f'evaluation_orchestrator_request_two_{suffix}'
    self.span_one_id = f'evaluation_orchestrator_span_one_{suffix}'
    self.span_two_id = f'evaluation_orchestrator_span_two_{suffix}'
    self.agent_name = f'orchestrator_agent_{suffix}'
    self.dataset_name = f'orchestrator_dataset_{suffix}'

    self.span_type_name = f'evaluation_orchestrator_span_type_{suffix}'
    self.metric_name = f'evaluation_orchestrator_metric_{suffix}'
    self.sample_two_deleted = False
    self.agent_span_type_preexisting = False

    self.now = datetime.now(tz=timezone.utc)
    async with UnitOfWork(self.db_manager) as uow:
      await uow.agents.create(Agent(id=self.agent_id, name=self.agent_name, version_tag='v-test'))
      await uow.datasets.create(Dataset(id=self.dataset_id, name=self.dataset_name))
      await uow.samples.create(
        Sample(
          id=self.sample_one_id,
          dataset_id=self.dataset_id,
          input_prompt='orchestrator sample one',
          ground_truth_output='unused',
        )
      )
      await uow.samples.create(
        Sample(
          id=self.sample_two_id,
          dataset_id=self.dataset_id,
          input_prompt='orchestrator sample two',
          ground_truth_output='unused',
        )
      )

      await uow.span_types.create(
        SpanType(name=self.span_type_name, description='Span type for orchestrator lifecycle tests')
      )
      await uow.traces.create(Trace(external_id=self.trace_one_id, start_time=self.now, end_time=self.now))
      await uow.traces.create(Trace(external_id=self.trace_two_id, start_time=self.now, end_time=self.now))
      await uow.spans.create(
        Span(
          external_id=self.span_one_id,
          trace_id=self.trace_one_id,
          parent_span_id=None,
          span_type=self.span_type_name,
          name='orchestrator_span_one',
          start_time=self.now,
          end_time=self.now,
          input_data='{}',
          output_data='output-one',
          metadata=None,
        )
      )
      await uow.spans.create(
        Span(
          external_id=self.span_two_id,
          trace_id=self.trace_two_id,
          parent_span_id=None,
          span_type=self.span_type_name,
          name='orchestrator_span_two',
          start_time=self.now,
          end_time=self.now,
          input_data='{}',
          output_data='output-two',
          metadata=None,
        )
      )
      self.agent_span_type_preexisting = await uow.span_types.exists('agent')
      existing_agent_metrics = await uow.metric_target_span_types.get_metrics_for_span_type('agent')

    registry = MetricRegistry(db_manager=self.db_manager)
    registry.register_many([_ConstantScoreMetric(self.metric_name, self.span_type_name)])
    await registry.sync_with_persistence()
    registry.register_many([_AgentGroundTruthRequiredMetric(metric_name) for metric_name in existing_agent_metrics])
    self.registry = registry

  async def asyncTearDown(self) -> None:
    if not hasattr(self, 'db_manager'):
      return

    async with UnitOfWork(self.db_manager) as uow:
      runs = await uow.evaluation_runs.list_by_dataset(self.dataset_id)
      for run in runs:
        run_samples = await uow.evaluation_run_samples.list_by_evaluation_run(run.id)
        for run_sample in run_samples:
          computations = await uow.span_metric_computations.list_by_evaluation_run_sample(run_sample.id)
          for computation in computations:
            await uow.span_metric_computations.delete(computation.id)
          await uow.evaluation_run_samples.delete(run_sample.id)
        await uow.evaluation_runs.delete(run.id)

      await uow.spans.delete(self.span_two_id)
      await uow.spans.delete(self.span_one_id)
      await uow.traces.delete(self.trace_two_id)
      await uow.traces.delete(self.trace_one_id)
      if not self.sample_two_deleted:
        await uow.samples.delete(self.sample_two_id)
      await uow.samples.delete(self.sample_one_id)
      await uow.datasets.delete(self.dataset_id)
      await uow.agents.delete(self.agent_id)

      await uow.metric_target_span_types.delete_by_metric_and_span_type(self.metric_name, self.span_type_name)
      await uow.metrics.delete(self.metric_name)
      await uow.span_types.delete(self.span_type_name)
      if not self.agent_span_type_preexisting:
        await uow.span_types.delete('agent')

    await self.db_manager.close_async()

  async def test_run_completes_using_persisted_traces(self) -> None:
    orchestrator = self._build_orchestrator(
      trace_client=_StaticTraceClient(
        trace_ids_by_request_id={
          self.request_one_id: self.trace_one_id,
          self.request_two_id: self.trace_two_id,
        },
        records_by_trace_id={
          self.trace_one_id: self._make_trace_records(
            root_span_id=f'{self.span_one_id}_root',
            child_span_id=self.span_one_id,
            child_span_name='orchestrator_span_one',
            output_value='output-one',
          ),
          self.trace_two_id: self._make_trace_records(
            root_span_id=f'{self.span_two_id}_root',
            child_span_id=self.span_two_id,
            child_span_name='orchestrator_span_two',
            output_value='output-two',
          ),
        },
      )
    )
    config = EvaluationConfig(
      agent_id=self.agent_id,
      dataset_id=self.dataset_id,
    )

    started_run = await orchestrator.create_run(config)

    async with UnitOfWork(self.db_manager) as uow:
      persisted_run = await uow.evaluation_runs.get_by_id(started_run.id)
      self.assertEqual(persisted_run.status, EvaluationStatus.RUNNING)
      self.assertIsNone(persisted_run.end_time)
      self.assertEqual(await uow.evaluation_run_samples.list_by_evaluation_run(started_run.id), [])

    run = await orchestrator.execute_run(started_run)

    self.assertEqual(run.status, EvaluationStatus.COMPLETED)
    self.assertIsNotNone(run.end_time)

    async with UnitOfWork(self.db_manager) as uow:
      run_samples = await uow.evaluation_run_samples.list_by_evaluation_run(run.id)

      self.assertEqual(len(run_samples), 2)
      sample_to_trace = {entry.sample_id: entry.trace_id for entry in run_samples}
      self.assertEqual(
        sample_to_trace,
        {
          self.sample_one_id: self.trace_one_id,
          self.sample_two_id: self.trace_two_id,
        },
      )

      total_computations = 0
      for run_sample in run_samples:
        computations = await uow.span_metric_computations.list_by_evaluation_run_sample(run_sample.id)
        self.assertEqual(len(computations), 1)
        self.assertEqual(computations[0].metric, self.metric_name)
        total_computations += len(computations)

      self.assertEqual(total_computations, 2)

  async def test_run_completes_when_dataset_contains_one_sample(self) -> None:
    async with UnitOfWork(self.db_manager) as uow:
      await uow.samples.delete(self.sample_two_id)
    self.sample_two_deleted = True

    orchestrator = self._build_orchestrator(
      trace_client=_StaticTraceClient(
        trace_ids_by_request_id={
          self.request_one_id: self.trace_one_id,
        },
        records_by_trace_id={
          self.trace_one_id: self._make_trace_records(
            root_span_id=f'{self.span_one_id}_root',
            child_span_id=self.span_one_id,
            child_span_name='orchestrator_span_one',
            output_value='output-one',
          ),
        },
      )
    )
    config = EvaluationConfig(
      agent_id=self.agent_id,
      dataset_id=self.dataset_id,
    )

    run = await orchestrator.run(config)

    self.assertEqual(run.status, EvaluationStatus.COMPLETED)

    async with UnitOfWork(self.db_manager) as uow:
      run_samples = await uow.evaluation_run_samples.list_by_evaluation_run(run.id)
      self.assertEqual(len(run_samples), 1)
      self.assertEqual(run_samples[0].sample_id, self.sample_one_id)
      self.assertEqual(run_samples[0].trace_id, self.trace_one_id)

  async def test_run_fails_when_trace_source_is_unavailable(self) -> None:
    orchestrator = self._build_orchestrator(trace_client=_FailingTraceClient())
    config = EvaluationConfig(
      agent_id=self.agent_id,
      dataset_id=self.dataset_id,
    )

    run = await orchestrator.run(config)

    self.assertEqual(run.status, EvaluationStatus.FAILED)

    async with UnitOfWork(self.db_manager) as uow:
      run_samples = await uow.evaluation_run_samples.list_by_evaluation_run(run.id)
      self.assertEqual(len(run_samples), 2)
      self.assertEqual({run_sample.status for run_sample in run_samples}, {EvaluationSampleStatus.FAILED})
      self.assertEqual({run_sample.trace_id for run_sample in run_samples}, {None})
      self.assertTrue(all(run_sample.error_message == 'Trace source unavailable' for run_sample in run_samples))

      for run_sample in run_samples:
        computations = await uow.span_metric_computations.list_by_evaluation_run_sample(run_sample.id)
        self.assertEqual(computations, [])

  def _build_orchestrator(self, trace_client) -> EvaluationOrchestrator:
    agent_call_dispatcher = _StaticAgentCallDispatcher(
      {
        self.sample_one_id: self.request_one_id,
        self.sample_two_id: self.request_two_id,
      }
    )
    trace_processor = TraceProcessor(trace_client=trace_client, db_manager=self.db_manager)
    metric_planner = MetricPlanner(metric_registry=self.registry, db_manager=self.db_manager)
    plan_executor = PlanExecutor(db_manager=self.db_manager, max_concurrent_tasks=5)
    sample_executor = SampleExecutor(
      db_manager=self.db_manager,
      agent_call_dispatcher=cast(AgentCallDispatcher, agent_call_dispatcher),
      trace_processor=trace_processor,
      metric_planner=metric_planner,
      plan_executor=plan_executor,
    )
    return EvaluationOrchestrator(
      db_manager=self.db_manager,
      sample_executor=sample_executor,
      max_concurrent_samples=2,
    )

  def _make_trace_records(
    self, root_span_id: str, child_span_id: str, child_span_name: str, output_value: str
  ) -> list[dict[str, object]]:
    return [
      {
        'context.span_id': root_span_id,
        'parent_id': None,
        'name': 'orchestrator_agent',
        'start_time': self.now,
        'end_time': self.now,
        'attributes.openinference.span.kind': 'agent',
        'attributes.input.value': '{}',
        'attributes.output.value': '{}',
      },
      {
        'context.span_id': child_span_id,
        'parent_id': root_span_id,
        'name': child_span_name,
        'start_time': self.now,
        'end_time': self.now,
        'attributes.openinference.span.kind': self.span_type_name,
        'attributes.input.value': '{}',
        'attributes.output.value': output_value,
      },
    ]


if __name__ == '__main__':
  unittest.main()
