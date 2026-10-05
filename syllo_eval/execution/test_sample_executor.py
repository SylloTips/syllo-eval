import asyncio
import unittest
from collections.abc import Mapping
from datetime import datetime, timezone
from typing import Any, cast
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch
from uuid import UUID, uuid4

from syllo_eval.evaluation.metric_planner import MetricPlanner
from syllo_eval.evaluation.metric_registry import MetricRegistry
from syllo_eval.evaluation.metrics.contracts import (
  EvaluationMetric,
  MetricComputationResult,
  SpanGroupEvaluationMetric,
)
from syllo_eval.evaluation.plan_executor import PlanExecutor
from syllo_eval.evaluation.trace_processor import TraceProcessor
from syllo_eval.execution.agent_caller import AgentCallDispatcher
from syllo_eval.execution.sample_executor import SampleExecutionResult, SampleExecutor
from syllo_eval.infrastructure import UnitOfWork
from syllo_eval.model import (
  Agent,
  Dataset,
  EvaluationRun,
  EvaluationRunSample,
  EvaluationSampleStatus,
  EvaluationStatus,
  GroundTruth,
  MetricComputationStatus,
  Sample,
  Span,
  SpanType,
)
from syllo_eval.testing_database import setup_test_database as _setup_database


class _OutputLengthMetric(EvaluationMetric):
  def __init__(self, metric_name: str, span_type_name: str):
    self._metric_name = metric_name
    self._span_type_name = span_type_name

  @property
  def name(self) -> str:
    return self._metric_name

  @property
  def description(self) -> str:
    return 'Scores by output length.'

  @property
  def target_span_types(self) -> tuple[str, ...]:
    return (self._span_type_name,)

  async def compute(self, span: Span, ground_truths: Mapping[str, GroundTruth]) -> MetricComputationResult:
    del ground_truths
    return MetricComputationResult(
      score=float(len(str(span.output_data))),
      reasoning='Score is output length.',
      metadata={'computed_from': 'output_data_length'},
    )


class _GroundTruthMatchMetric(EvaluationMetric):
  def __init__(self, metric_name: str, span_type_name: str):
    self._metric_name = metric_name
    self._span_type_name = span_type_name

  @property
  def name(self) -> str:
    return self._metric_name

  @property
  def description(self) -> str:
    return 'Checks output equality against ground truth.'

  @property
  def target_span_types(self) -> tuple[str, ...]:
    return (self._span_type_name,)

  async def compute(self, span: Span, ground_truths: Mapping[str, GroundTruth]) -> MetricComputationResult:
    ground_truth = next(iter(ground_truths.values()), None)
    if ground_truth is None:
      return MetricComputationResult(score=0.0, reasoning='Missing ground truth.')

    expected_output = ground_truth.ground_truth_value.get('expected_output')
    is_match = span.output_data == expected_output
    return MetricComputationResult(
      score=1.0 if is_match else 0.0,
      reasoning='Output matches expected output.' if is_match else 'Output does not match expected output.',
      metadata={'expected_output': expected_output, 'actual_output': span.output_data},
    )


class _GroupedOutputLengthMetric(SpanGroupEvaluationMetric):
  def __init__(self, metric_name: str, span_type_name: str):
    self._metric_name = metric_name
    self._span_type_name = span_type_name

  @property
  def name(self) -> str:
    return self._metric_name

  @property
  def description(self) -> str:
    return 'Sums output lengths across a span group.'

  @property
  def target_span_types(self) -> tuple[str, ...]:
    return (self._span_type_name,)

  async def compute(self, spans, ground_truths: Mapping[str, GroundTruth]) -> MetricComputationResult:
    del ground_truths
    total_length = sum(len(str(span.output_data)) for span in spans)
    return MetricComputationResult(
      score=float(total_length),
      reasoning='Score is the summed output length for the span group.',
      metadata={'span_count': len(spans)},
    )


class _FailingMetric(EvaluationMetric):
  def __init__(self, metric_name: str, span_type_name: str):
    self._metric_name = metric_name
    self._span_type_name = span_type_name

  @property
  def name(self) -> str:
    return self._metric_name

  @property
  def description(self) -> str:
    return 'Always fails.'

  @property
  def target_span_types(self) -> tuple[str, ...]:
    return (self._span_type_name,)

  async def compute(self, span: Span, ground_truths: Mapping[str, GroundTruth]) -> MetricComputationResult:
    del span, ground_truths
    raise RuntimeError('metric service unavailable')


class _StaticTraceClient:
  def __init__(self, trace_id: str, records: list[dict[str, Any]]):
    self._trace_id = trace_id
    self._records = records

  async def get_trace_id_by_request_id(self, request_id: str) -> str | None:
    del request_id
    return self._trace_id

  async def get_trace_json(self, trace_id: str) -> list[dict[str, Any]]:
    assert trace_id == self._trace_id
    return self._records


class _SlowTraceProcessor:
  async def process_trace(self, request_id: str) -> None:
    del request_id
    await asyncio.sleep(0.05)


class _StaticAgentCallDispatcher:
  def __init__(self, request_id: str):
    self._request_id = request_id

  async def call(self, agent: Agent, sample: Sample) -> str:
    del agent, sample
    return self._request_id


class _NoOpMetricPlanner:
  async def _empty(self):
    if False:
      yield None

  def iter_plan_items(self, trace_id: str, sample_id: UUID):
    del trace_id, sample_id
    return self._empty()


class _NoOpPlanExecutor:
  async def execute(self, plan_items, evaluation_run_sample_id):
    del plan_items, evaluation_run_sample_id
    return []


class _SlowPlanExecutor:
  async def execute(self, plan_items, evaluation_run_sample_id):
    del plan_items, evaluation_run_sample_id
    await asyncio.sleep(0.05)
    return []


class TestSampleExecutionResult(unittest.TestCase):
  """Verify the `succeeded` property of SampleExecutionResult."""

  def test_succeeded_is_true_when_no_error(self) -> None:
    result = SampleExecutionResult(
      evaluation_run_sample=EvaluationRunSample(
        id=uuid4(),
        evaluation_run_id=uuid4(),
        sample_id=uuid4(),
        trace_id='trace-1',
      ),
      computations=[],
      error=None,
    )
    self.assertTrue(result.succeeded)

  def test_succeeded_is_false_when_error_is_set(self) -> None:
    result = SampleExecutionResult(
      evaluation_run_sample=EvaluationRunSample(
        id=uuid4(),
        evaluation_run_id=uuid4(),
        sample_id=uuid4(),
        trace_id='trace-1',
      ),
      computations=[],
      error='something went wrong',
    )
    self.assertFalse(result.succeeded)


class TestSampleExecutorTimeout(unittest.IsolatedAsyncioTestCase):
  async def test_execute_with_enabled_and_disabled_timeouts_and_existing_request(self) -> None:
    for timeout in (None, 1.0):
      for request_id in (None, 'existing-request'):
        with self.subTest(timeout=timeout, request_id=request_id):
          agent = Agent(id=uuid4(), name='agent', version_tag='v1')
          sample = Sample(id=uuid4(), dataset_id=uuid4(), input_prompt='question')
          row = EvaluationRunSample(id=uuid4(), evaluation_run_id=uuid4(), sample_id=sample.id)
          caller = AsyncMock()
          caller.call.return_value = 'new-request'
          trace = AsyncMock()
          trace.process_trace.return_value = SimpleNamespace(trace=SimpleNamespace(external_id='trace'))
          executor = SampleExecutor(
            cast(Any, object()), caller, trace, cast(Any, _NoOpMetricPlanner()), cast(Any, _NoOpPlanExecutor())
          )
          with (
            patch.object(executor, '_create_evaluation_run_sample', AsyncMock(return_value=row)),
            patch.object(executor, '_update_evaluation_run_sample', AsyncMock(return_value=row)) as update,
          ):
            result = await executor.execute(
              row.evaluation_run_id,
              agent,
              sample,
              request_id=request_id,
              trace_timeout_seconds=timeout,
              compute_timeout_seconds=timeout,
            )
          self.assertTrue(result.succeeded)
          self.assertEqual(caller.call.await_count, 1 if request_id is None else 0)
          trace.process_trace.assert_awaited_once_with(request_id or 'new-request')
          self.assertEqual(update.await_args_list[0].kwargs['trace_id'], 'trace')
          self.assertEqual(update.await_args_list[-1].kwargs['status'], EvaluationSampleStatus.COMPLETED)

  async def test_trace_errors_and_agent_timeout_record_the_active_phase(self) -> None:
    for phase, timed_out in [('agent_call', False), ('trace_fetch', False), ('agent_call', True)]:
      with self.subTest(phase=phase, timed_out=timed_out):
        agent = Agent(id=uuid4(), name='agent', version_tag='v1')
        sample = Sample(id=uuid4(), dataset_id=uuid4(), input_prompt='question')
        row = EvaluationRunSample(id=uuid4(), evaluation_run_id=uuid4(), sample_id=sample.id)
        caller = AsyncMock()
        caller.call.return_value = 'request'
        trace = AsyncMock()
        failing_call = caller.call if phase == 'agent_call' else trace.process_trace

        async def blocked(*args, **kwargs):
          await asyncio.Future()

        failing_call.side_effect = blocked if timed_out else ValueError('source failed')
        executor = SampleExecutor(
          cast(Any, object()), caller, trace, cast(Any, _NoOpMetricPlanner()), cast(Any, _NoOpPlanExecutor())
        )
        with (
          patch.object(executor, '_create_evaluation_run_sample', AsyncMock(return_value=row)),
          patch.object(executor, '_update_evaluation_run_sample', AsyncMock(return_value=row)) as update,
        ):
          result = await executor.execute(
            row.evaluation_run_id, agent, sample, trace_timeout_seconds=0.001 if timed_out else None
          )
        self.assertFalse(result.succeeded)
        assert update.await_args is not None
        self.assertEqual(update.await_args.kwargs['metadata'], {'failure_phase': phase})
        self.assertEqual(update.await_args.kwargs['status'], EvaluationSampleStatus.FAILED)
        self.assertEqual(result.error, 'Trace phase timed out after 0.001s' if timed_out else 'source failed')

  async def test_execute_marks_sample_failed_when_trace_phase_is_cancelled(self) -> None:
    sample = Sample(
      id=uuid4(),
      dataset_id=uuid4(),
      input_prompt='cancelled trace test',
    )
    agent = Agent(id=uuid4(), name='demo-agent', version_tag='v-test')
    evaluation_run_id = uuid4()
    run_sample = EvaluationRunSample(
      id=uuid4(),
      evaluation_run_id=evaluation_run_id,
      sample_id=sample.id,
      trace_id=None,
      status=EvaluationSampleStatus.RUNNING,
    )
    failed_run_sample = run_sample.model_copy(
      update={
        'status': EvaluationSampleStatus.FAILED,
        'ended_at': datetime.now(timezone.utc),
        'error_message': 'Sample execution cancelled during trace phase',
      }
    )

    executor = SampleExecutor(
      db_manager=cast(Any, object()),
      agent_call_dispatcher=cast(AgentCallDispatcher, _StaticAgentCallDispatcher('request-cancelled')),
      trace_processor=cast(TraceProcessor, object()),
      metric_planner=cast(MetricPlanner, _NoOpMetricPlanner()),
      plan_executor=cast(PlanExecutor, _NoOpPlanExecutor()),
    )

    with patch.object(
      executor, '_agent_call_dispatcher', AsyncMock(call=AsyncMock(side_effect=asyncio.CancelledError))
    ):
      with patch.object(executor, '_create_evaluation_run_sample', AsyncMock(return_value=run_sample)):
        with patch.object(
          executor,
          '_update_evaluation_run_sample',
          AsyncMock(return_value=failed_run_sample),
        ) as update_mock:
          with self.assertRaises(asyncio.CancelledError):
            await executor.execute(
              evaluation_run_id=evaluation_run_id,
              agent=agent,
              sample=sample,
              trace_timeout_seconds=1,
            )

    update_mock.assert_awaited_once()
    await_args = update_mock.await_args
    assert await_args is not None
    update_kwargs = await_args.kwargs
    self.assertEqual(update_kwargs['run_sample_id'], run_sample.id)
    self.assertEqual(update_kwargs['status'], EvaluationSampleStatus.FAILED)
    self.assertIsNotNone(update_kwargs['ended_at'])
    self.assertEqual(update_kwargs['error_message'], 'Sample execution cancelled during trace phase')
    self.assertEqual(update_kwargs['metadata'], {'failure_phase': 'agent_call'})

  async def test_execute_marks_sample_failed_when_compute_phase_is_cancelled(self) -> None:
    sample = Sample(
      id=uuid4(),
      dataset_id=uuid4(),
      input_prompt='cancelled compute test',
    )
    agent = Agent(id=uuid4(), name='demo-agent', version_tag='v-test')
    evaluation_run_id = uuid4()
    run_sample = EvaluationRunSample(
      id=uuid4(),
      evaluation_run_id=evaluation_run_id,
      sample_id=sample.id,
      trace_id=None,
      status=EvaluationSampleStatus.RUNNING,
    )
    traced_run_sample = run_sample.model_copy(
      update={'trace_id': 'trace-cancelled', 'status': EvaluationSampleStatus.RUNNING}
    )
    failed_run_sample = traced_run_sample.model_copy(
      update={
        'status': EvaluationSampleStatus.FAILED,
        'ended_at': datetime.now(timezone.utc),
        'error_message': 'Sample execution cancelled during compute phase',
      }
    )

    executor = SampleExecutor(
      db_manager=cast(Any, object()),
      agent_call_dispatcher=cast(AgentCallDispatcher, _StaticAgentCallDispatcher('request-cancelled')),
      trace_processor=cast(TraceProcessor, object()),
      metric_planner=cast(MetricPlanner, _NoOpMetricPlanner()),
      plan_executor=cast(PlanExecutor, _NoOpPlanExecutor()),
    )

    with patch.object(
      executor,
      '_trace_processor',
      AsyncMock(
        process_trace=AsyncMock(return_value=SimpleNamespace(trace=SimpleNamespace(external_id='trace-cancelled')))
      ),
    ):
      with patch.object(executor, '_execute_pipeline', AsyncMock(side_effect=asyncio.CancelledError)):
        with patch.object(executor, '_create_evaluation_run_sample', AsyncMock(return_value=run_sample)):
          with patch.object(
            executor,
            '_update_evaluation_run_sample',
            AsyncMock(side_effect=[traced_run_sample, failed_run_sample]),
          ) as update_mock:
            with self.assertRaises(asyncio.CancelledError):
              await executor.execute(
                evaluation_run_id=evaluation_run_id,
                agent=agent,
                sample=sample,
                compute_timeout_seconds=1,
              )

    self.assertEqual(update_mock.await_count, 2)
    failed_update_kwargs = update_mock.await_args_list[1].kwargs
    self.assertEqual(failed_update_kwargs['run_sample_id'], run_sample.id)
    self.assertEqual(failed_update_kwargs['status'], EvaluationSampleStatus.FAILED)
    self.assertIsNotNone(failed_update_kwargs['ended_at'])
    self.assertEqual(failed_update_kwargs['error_message'], 'Sample execution cancelled during compute phase')
    self.assertEqual(failed_update_kwargs['metadata'], {'failure_phase': 'metric_compute'})

  async def test_execute_returns_failed_result_when_trace_times_out_before_trace_creation(self) -> None:
    sample = Sample(
      id=uuid4(),
      dataset_id=uuid4(),
      input_prompt='timeout test',
    )
    agent = Agent(id=uuid4(), name='demo-agent', version_tag='v-test')
    evaluation_run_id = uuid4()
    run_sample = EvaluationRunSample(
      id=uuid4(),
      evaluation_run_id=evaluation_run_id,
      sample_id=sample.id,
      trace_id=None,
      status=EvaluationSampleStatus.RUNNING,
    )
    failed_run_sample = EvaluationRunSample(
      id=run_sample.id,
      evaluation_run_id=evaluation_run_id,
      sample_id=sample.id,
      trace_id=None,
      status=EvaluationSampleStatus.FAILED,
      error_message='Trace phase timed out after 0.001s',
    )

    executor = SampleExecutor(
      db_manager=cast(Any, object()),
      agent_call_dispatcher=cast(AgentCallDispatcher, _StaticAgentCallDispatcher('request-timeout')),
      trace_processor=cast(TraceProcessor, _SlowTraceProcessor()),
      metric_planner=cast(MetricPlanner, _NoOpMetricPlanner()),
      plan_executor=cast(PlanExecutor, _NoOpPlanExecutor()),
    )

    with patch.object(executor, '_create_evaluation_run_sample', AsyncMock(return_value=run_sample)) as create_mock:
      with patch.object(
        executor, '_update_evaluation_run_sample', AsyncMock(return_value=failed_run_sample)
      ) as update_mock:
        result = await executor.execute(
          evaluation_run_id=evaluation_run_id,
          agent=agent,
          sample=sample,
          trace_timeout_seconds=0.001,
        )

    create_mock.assert_awaited_once()
    update_mock.assert_awaited_once()
    await_args = update_mock.await_args
    assert await_args is not None
    self.assertEqual(await_args.kwargs['metadata'], {'failure_phase': 'trace_fetch'})
    self.assertFalse(result.succeeded)
    self.assertEqual(result.evaluation_run_sample.status, EvaluationSampleStatus.FAILED)
    self.assertIsNone(result.evaluation_run_sample.trace_id)
    self.assertEqual(result.error, 'Trace phase timed out after 0.001s')

  async def test_execute_returns_failed_result_when_compute_times_out_after_trace_creation(self) -> None:
    sample = Sample(
      id=uuid4(),
      dataset_id=uuid4(),
      input_prompt='compute timeout test',
    )
    agent = Agent(id=uuid4(), name='demo-agent', version_tag='v-test')
    evaluation_run_id = uuid4()
    run_sample = EvaluationRunSample(
      id=uuid4(),
      evaluation_run_id=evaluation_run_id,
      sample_id=sample.id,
      trace_id=None,
      status=EvaluationSampleStatus.RUNNING,
    )
    running_run_sample = EvaluationRunSample(
      id=run_sample.id,
      evaluation_run_id=evaluation_run_id,
      sample_id=sample.id,
      trace_id='trace-compute-timeout',
      status=EvaluationSampleStatus.RUNNING,
    )
    failed_run_sample = EvaluationRunSample(
      id=run_sample.id,
      evaluation_run_id=evaluation_run_id,
      sample_id=sample.id,
      trace_id='trace-compute-timeout',
      status=EvaluationSampleStatus.FAILED,
      error_message='Compute phase timed out after 0.001s',
    )

    executor = SampleExecutor(
      db_manager=cast(Any, object()),
      agent_call_dispatcher=cast(AgentCallDispatcher, _StaticAgentCallDispatcher('request-compute-timeout')),
      trace_processor=cast(TraceProcessor, object()),
      metric_planner=cast(MetricPlanner, _NoOpMetricPlanner()),
      plan_executor=cast(PlanExecutor, _SlowPlanExecutor()),
    )

    with patch.object(
      executor,
      '_trace_processor',
      AsyncMock(
        process_trace=AsyncMock(
          return_value=SimpleNamespace(trace=SimpleNamespace(external_id='trace-compute-timeout'))
        )
      ),
    ):
      with patch.object(executor, '_create_evaluation_run_sample', AsyncMock(return_value=run_sample)):
        with patch.object(
          executor,
          '_update_evaluation_run_sample',
          AsyncMock(side_effect=[running_run_sample, failed_run_sample]),
        ) as update_mock:
          result = await executor.execute(
            evaluation_run_id=evaluation_run_id,
            agent=agent,
            sample=sample,
            compute_timeout_seconds=0.001,
          )

    self.assertEqual(update_mock.await_count, 2)
    self.assertEqual(update_mock.await_args_list[1].kwargs['metadata'], {'failure_phase': 'metric_compute'})
    self.assertFalse(result.succeeded)
    self.assertEqual(result.evaluation_run_sample.status, EvaluationSampleStatus.FAILED)
    self.assertEqual(result.evaluation_run_sample.trace_id, 'trace-compute-timeout')
    self.assertEqual(result.error, 'Compute phase timed out after 0.001s')

  async def test_recompute_runs_compute_phase_without_preparing_trace(self) -> None:
    sample_id = uuid4()
    evaluation_run_id = uuid4()
    run_sample = EvaluationRunSample(
      id=uuid4(),
      evaluation_run_id=evaluation_run_id,
      sample_id=sample_id,
      trace_id='trace-repeat',
      status=EvaluationSampleStatus.RUNNING,
    )
    completed_run_sample = run_sample.model_copy(
      update={'status': EvaluationSampleStatus.COMPLETED, 'ended_at': datetime.now(timezone.utc)}
    )

    executor = SampleExecutor(
      db_manager=cast(Any, object()),
      agent_call_dispatcher=cast(AgentCallDispatcher, object()),
      trace_processor=cast(TraceProcessor, object()),
      metric_planner=cast(MetricPlanner, _NoOpMetricPlanner()),
      plan_executor=cast(PlanExecutor, _NoOpPlanExecutor()),
    )

    with patch.object(executor, '_trace_processor', AsyncMock()) as trace_processor:
      with patch.object(executor, '_create_evaluation_run_sample', AsyncMock(return_value=run_sample)):
        with patch.object(executor, '_update_evaluation_run_sample', AsyncMock(return_value=completed_run_sample)):
          result = await executor.recompute(
            evaluation_run_id=evaluation_run_id,
            sample_id=sample_id,
            trace_id='trace-repeat',
          )

    trace_processor.process_trace.assert_not_awaited()
    self.assertTrue(result.succeeded)
    self.assertEqual(result.evaluation_run_sample.status, EvaluationSampleStatus.COMPLETED)
    self.assertEqual(result.evaluation_run_sample.trace_id, 'trace-repeat')


class TestSampleExecutorLifecyclePersistence(unittest.IsolatedAsyncioTestCase):
  async def asyncSetUp(self) -> None:
    try:
      self.db_manager = await _setup_database()
    except RuntimeError as error:
      self.skipTest(str(error))

    suffix = uuid4().hex[:10]
    self.agent_name = f'sample_executor_agent_{suffix}'
    self.dataset_name = f'sample_executor_dataset_{suffix}'
    self.span_type_name = f'sample_executor_span_type_{suffix}'
    self.root_span_type_name = f'{self.span_type_name}_root'
    self.length_metric_name = f'sample_executor_length_metric_{suffix}'
    self.group_metric_name = f'sample_executor_group_metric_{suffix}'
    self.failing_metric_name = f'sample_executor_failing_metric_{suffix}'
    self.trace_id = f'sample_executor_trace_{suffix}'
    self.root_span_id = f'sample_executor_root_span_{suffix}'
    self.span_id = f'sample_executor_span_{suffix}'
    self.second_span_id = f'sample_executor_span_two_{suffix}'

    self.agent_id = uuid4()
    self.dataset_id = uuid4()
    self.sample_id = uuid4()
    self.ground_truth_id = uuid4()
    self.evaluation_run_id = uuid4()
    self.request_id = f'sample_executor_request_{suffix}'

    self.created_computation_ids: list[UUID] = []
    self.created_evaluation_run_sample_id: UUID | None = None

  async def asyncTearDown(self) -> None:
    if not hasattr(self, 'db_manager'):
      return

    async with UnitOfWork(self.db_manager) as uow:
      computation_ids = set(self.created_computation_ids)
      if self.created_evaluation_run_sample_id is not None:
        computations = await uow.metric_computations.list_by_evaluation_run_sample(
          self.created_evaluation_run_sample_id
        )
        computation_ids.update(computation.id for computation in computations)
      for computation_id in computation_ids:
        await uow.metric_computations.delete(computation_id)

      if self.created_evaluation_run_sample_id is not None:
        await uow.evaluation_run_samples.delete(self.created_evaluation_run_sample_id)

      await uow.ground_truths.delete(self.ground_truth_id)
      await uow.evaluation_runs.delete(self.evaluation_run_id)
      await uow.spans.delete(self.second_span_id)
      await uow.spans.delete(self.span_id)
      await uow.spans.delete(self.root_span_id)
      await uow.traces.delete(self.trace_id)
      await uow.samples.delete(self.sample_id)
      await uow.datasets.delete(self.dataset_id)
      await uow.agents.delete(self.agent_id)

      await uow.metric_target_span_types.delete_by_metric_and_span_type(self.length_metric_name, self.span_type_name)
      await uow.metric_target_span_types.delete_by_metric_and_span_type(self.group_metric_name, self.span_type_name)
      await uow.metric_target_span_types.delete_by_metric_and_span_type(self.failing_metric_name, self.span_type_name)
      await uow.metrics.delete(self.length_metric_name)
      await uow.metrics.delete(self.group_metric_name)
      await uow.metrics.delete(self.failing_metric_name)
      await uow.span_types.delete(self.span_type_name)
      await uow.span_types.delete(self.root_span_type_name)

    await self.db_manager.close_async()

  async def test_execute_runs_full_pipeline_and_persists_computations(self) -> None:
    now = datetime.now(tz=timezone.utc)

    async with UnitOfWork(self.db_manager) as uow:
      await uow.agents.create(Agent(id=self.agent_id, name=self.agent_name, version_tag='v-test'))
      await uow.datasets.create(Dataset(id=self.dataset_id, name=self.dataset_name))
      sample = await uow.samples.create(
        Sample(
          id=self.sample_id,
          dataset_id=self.dataset_id,
          input_prompt='Sample executor lifecycle prompt',
          ground_truth_output='expected-response',
        )
      )
      await uow.evaluation_runs.create(
        EvaluationRun(
          id=self.evaluation_run_id,
          agent_id=self.agent_id,
          dataset_id=self.dataset_id,
          status=EvaluationStatus.RUNNING,
          start_time=now,
        )
      )
      await uow.span_types.create(self._build_span_type())

    registry = MetricRegistry(db_manager=self.db_manager)
    registry.register_many(
      [
        _OutputLengthMetric(self.length_metric_name, self.span_type_name),
        _GroupedOutputLengthMetric(self.group_metric_name, self.span_type_name),
        _FailingMetric(self.failing_metric_name, self.span_type_name),
      ]
    )
    await registry.sync_with_persistence()

    trace_records: list[dict[str, Any]] = [
      {
        'context.span_id': self.root_span_id,
        'parent_id': None,
        'name': None,
        'start_time': now,
        'end_time': now,
        'attributes.openinference.span.kind': self.root_span_type_name,
        'attributes.input.value': '{"prompt":"test"}',
        'attributes.output.value': 'root-output',
      },
      {
        'context.span_id': self.span_id,
        'parent_id': self.root_span_id,
        'name': 'metric_span',
        'start_time': now,
        'end_time': now,
        'attributes.openinference.span.kind': self.span_type_name,
        'attributes.input.value': '{"prompt":"test"}',
        'attributes.output.value': 'expected-response',
      },
      {
        'context.span_id': self.second_span_id,
        'parent_id': self.root_span_id,
        'name': 'metric_span_two',
        'start_time': now,
        'end_time': now,
        'attributes.openinference.span.kind': self.span_type_name,
        'attributes.input.value': '{"prompt":"test"}',
        'attributes.output.value': 'secondary-response',
      },
    ]
    trace_processor = TraceProcessor(
      trace_client=_StaticTraceClient(trace_id=self.trace_id, records=trace_records),
      db_manager=self.db_manager,
    )
    planner = MetricPlanner(metric_registry=registry, db_manager=self.db_manager)
    plan_executor = PlanExecutor(db_manager=self.db_manager, max_concurrent_tasks=5)
    executor = SampleExecutor(
      db_manager=self.db_manager,
      agent_call_dispatcher=cast(AgentCallDispatcher, _StaticAgentCallDispatcher(self.request_id)),
      trace_processor=trace_processor,
      metric_planner=planner,
      plan_executor=plan_executor,
    )

    agent = Agent(id=self.agent_id, name='demo-agent', version_tag='v-test')
    result = await executor.execute(self.evaluation_run_id, agent, sample)

    self.assertTrue(result.succeeded)
    self.assertIsNone(result.error)
    self.assertEqual(result.evaluation_run_sample.evaluation_run_id, self.evaluation_run_id)
    self.assertEqual(result.evaluation_run_sample.sample_id, self.sample_id)
    self.assertEqual(result.evaluation_run_sample.trace_id, self.trace_id)
    self.assertEqual(result.evaluation_run_sample.status, EvaluationSampleStatus.COMPLETED)
    self.assertIsNotNone(result.evaluation_run_sample.started_at)
    self.assertIsNotNone(result.evaluation_run_sample.ended_at)
    self.assertEqual(len(result.computations), 5)

    self.created_evaluation_run_sample_id = result.evaluation_run_sample.id
    self.created_computation_ids = [computation.id for computation in result.computations]

    metric_names = [computation.metric for computation in result.computations]
    self.assertEqual(metric_names.count(self.length_metric_name), 2)
    self.assertEqual(metric_names.count(self.group_metric_name), 1)
    self.assertEqual(metric_names.count(self.failing_metric_name), 2)

    length_computations = [comp for comp in result.computations if comp.metric == self.length_metric_name]
    self.assertEqual([comp.span_ids for comp in length_computations], [[self.span_id], [self.second_span_id]])
    self.assertEqual(
      {comp.score for comp in length_computations},
      {float(len('expected-response')), float(len('secondary-response'))},
    )

    group_computation = next(comp for comp in result.computations if comp.metric == self.group_metric_name)
    self.assertEqual(group_computation.span_ids, [self.span_id, self.second_span_id])
    self.assertEqual(group_computation.targeting_mode.value, 'GROUP')
    self.assertEqual(group_computation.score, float(len('expected-response') + len('secondary-response')))

    failed_computations = [comp for comp in result.computations if comp.metric == self.failing_metric_name]
    self.assertEqual({comp.status for comp in failed_computations}, {MetricComputationStatus.FAILED})
    self.assertEqual({comp.error_message for comp in failed_computations}, {'metric service unavailable'})

    async with UnitOfWork(self.db_manager) as uow:
      persisted = await uow.metric_computations.list_by_evaluation_run_sample(result.evaluation_run_sample.id)

    self.assertEqual(len(persisted), 5)
    self.assertEqual(
      {comp.metric for comp in persisted},
      {self.length_metric_name, self.group_metric_name, self.failing_metric_name},
    )

  def _build_span_type(self):
    return SpanType(name=self.span_type_name, description='Span type for sample executor lifecycle test')
