import unittest
from datetime import datetime, timezone
from typing import Any, cast
from unittest.mock import AsyncMock, patch
from uuid import UUID, uuid4

from syllo_eval.evaluation.metric_planner import MetricPlanItem, MetricPlanner, SpanTarget
from syllo_eval.evaluation.metric_registry import MetricRegistry
from syllo_eval.evaluation.metrics.contracts import (
  EvaluationMetric,
  MetricComputationResult,
  SpanGroupEvaluationMetric,
)
from syllo_eval.evaluation.plan_executor import PlanExecutor
from syllo_eval.infrastructure import UnitOfWork
from syllo_eval.model import (
  Agent,
  Dataset,
  EvaluationRun,
  EvaluationRunSample,
  EvaluationStatus,
  GroundTruth,
  MetricComputation,
  MetricComputationStatus,
  Sample,
  Span,
  SpanType,
  Trace,
)
from syllo_eval.testing_database import setup_test_database as _setup_database


class _MetricComputationRepositoryStub:
  def __init__(self, error: Exception):
    self.create = AsyncMock(side_effect=error)


class _UnitOfWorkStub:
  def __init__(self, error: Exception):
    self.metric_computations = _MetricComputationRepositoryStub(error)

  async def __aenter__(self):
    return self

  async def __aexit__(self, exc_type, exc, tb):
    del exc_type, exc, tb


class _PlanItemStubMetric(EvaluationMetric):
  @property
  def name(self) -> str:
    return 'persistence_failure_metric'

  @property
  def requires_ground_truth(self) -> bool:
    return False

  async def compute(self, span: Span, ground_truth: GroundTruth | None) -> MetricComputationResult:
    del span, ground_truth
    return MetricComputationResult(score=1.0)


async def _single_plan_item():
  now = datetime.now(tz=timezone.utc)
  span = Span(
    external_id='span-persistence-failure',
    trace_id='trace-persistence-failure',
    parent_span_id=None,
    span_type='agent',
    name='span-persistence-failure',
    start_time=now,
    end_time=now,
    input_data='{}',
    output_data='{}',
  )
  yield MetricPlanItem(
    metric=_PlanItemStubMetric(),
    target=SpanTarget(target_span_type='agent', span=span),
    ground_truth=None,
  )


def _metric_metadata(metadata: dict[str, Any] | None) -> dict[str, Any]:
  """Drop the latency the executor adds to every computed item, failing when it is missing."""
  if metadata is None or not isinstance(metadata.get('latency_seconds'), float):
    raise AssertionError(f'Computation metadata has no latency: {metadata!r}')
  return {key: value for key, value in metadata.items() if key != 'latency_seconds'}


class _RecordingMetricComputationRepository:
  def __init__(self):
    self.created: list[MetricComputation] = []

  async def create(self, computation: MetricComputation) -> MetricComputation:
    self.created.append(computation)
    return computation


class _RecordingUnitOfWork:
  def __init__(self, repository: _RecordingMetricComputationRepository):
    self.metric_computations = repository

  async def __aenter__(self):
    return self

  async def __aexit__(self, exc_type, exc, tb):
    del exc_type, exc, tb


class _RaisingStubMetric(_PlanItemStubMetric):
  async def compute(self, span: Span, ground_truth: GroundTruth | None) -> MetricComputationResult:
    del span, ground_truth
    raise RuntimeError('judge unavailable')


async def _latency_plan_items():
  async for item in _single_plan_item():
    yield item
    yield MetricPlanItem(metric=_RaisingStubMetric(), target=item.target, ground_truth=None)
    yield MetricPlanItem(metric=_PlanItemStubMetric(), target=item.target, ground_truth=None, skip_reason='missing')


class TestPlanExecutorLatency(unittest.IsolatedAsyncioTestCase):
  async def test_computed_and_failed_items_record_latency_but_skipped_items_do_not(self) -> None:
    repository = _RecordingMetricComputationRepository()
    executor = PlanExecutor(db_manager=cast(Any, object()))

    with patch('syllo_eval.evaluation.plan_executor.UnitOfWork', return_value=_RecordingUnitOfWork(repository)):
      completed, failed, skipped = await executor.execute(_latency_plan_items(), uuid4())

    self.assertEqual(completed.status, MetricComputationStatus.COMPLETED)
    self.assertEqual(_metric_metadata(completed.metadata), {})
    self.assertEqual(failed.status, MetricComputationStatus.FAILED)
    self.assertEqual(_metric_metadata(failed.metadata), {})
    self.assertEqual(skipped.status, MetricComputationStatus.SKIPPED)
    self.assertIsNone(skipped.metadata)


class TestPlanExecutorPersistenceErrors(unittest.IsolatedAsyncioTestCase):
  async def test_execute_raises_when_metric_computation_persistence_fails(self) -> None:
    executor = PlanExecutor(db_manager=cast(Any, object()))
    error = RuntimeError('database unavailable')

    with patch('syllo_eval.evaluation.plan_executor.UnitOfWork', return_value=_UnitOfWorkStub(error)):
      with self.assertRaisesRegex(RuntimeError, 'database unavailable'):
        await executor.execute(_single_plan_item(), uuid4())


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
  def requires_ground_truth(self) -> bool:
    return False

  @property
  def target_span_types(self) -> tuple[str, ...]:
    return (self._span_type_name,)

  async def compute(self, span: Span, ground_truth: GroundTruth | None) -> MetricComputationResult:
    del ground_truth
    score = float(len(str(span.output_data)))
    return MetricComputationResult(
      score=score,
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

  @property
  def requires_ground_truth(self) -> bool:
    return True

  @property
  def ground_truth_key(self) -> str | None:
    return self._metric_name

  async def compute(self, span: Span, ground_truth: GroundTruth | None) -> MetricComputationResult:
    if ground_truth is None:
      return MetricComputationResult(score=0.0, reasoning='Missing ground truth.')

    expected_output = ground_truth.ground_truth_value.get('expected_output')
    matches = span.output_data == expected_output
    return MetricComputationResult(
      score=1.0 if matches else 0.0,
      reasoning='Output matches expected output.' if matches else 'Output does not match expected output.',
      metadata={
        'expected_output': expected_output,
        'actual_output': span.output_data,
      },
    )


class _ContainsResponseMetric(EvaluationMetric):
  def __init__(self, metric_name: str, span_type_name: str):
    self._metric_name = metric_name
    self._span_type_name = span_type_name

  @property
  def name(self) -> str:
    return self._metric_name

  @property
  def description(self) -> str:
    return "Checks whether output contains 'response'."

  @property
  def requires_ground_truth(self) -> bool:
    return False

  @property
  def target_span_types(self) -> tuple[str, ...]:
    return (self._span_type_name,)

  async def compute(self, span: Span, ground_truth: GroundTruth | None) -> MetricComputationResult:
    del ground_truth
    contains_response = 'response' in str(span.output_data)
    return MetricComputationResult(
      score=1.0 if contains_response else 0.0,
      reasoning="Output contains 'response'." if contains_response else "Output does not contain 'response'.",
      metadata={'needle': 'response'},
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
  def requires_ground_truth(self) -> bool:
    return False

  @property
  def target_span_types(self) -> tuple[str, ...]:
    return (self._span_type_name,)

  async def compute(self, span: Span, ground_truth: GroundTruth | None) -> MetricComputationResult:
    del span, ground_truth
    raise RuntimeError('judge service unavailable')


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
  def requires_ground_truth(self) -> bool:
    return False

  @property
  def target_span_types(self) -> tuple[str, ...]:
    return (self._span_type_name,)

  async def compute(self, spans, ground_truth: GroundTruth | None) -> MetricComputationResult:
    del ground_truth
    total_length = sum(len(str(span.output_data)) for span in spans)
    return MetricComputationResult(
      score=float(total_length),
      reasoning='Score is the summed output length for the span group.',
      metadata={'span_count': len(spans)},
    )


class TestPlanExecutorLifecyclePersistence(unittest.IsolatedAsyncioTestCase):
  async def asyncSetUp(self) -> None:
    try:
      self.db_manager = await _setup_database()
    except RuntimeError as error:
      self.skipTest(str(error))

    suffix = uuid4().hex[:10]

    self.span_type_name = f'plan_executor_span_type_{suffix}'
    self.length_metric_name = f'plan_executor_length_metric_{suffix}'
    self.match_metric_name = f'plan_executor_match_metric_{suffix}'
    self.contains_metric_name = f'plan_executor_contains_metric_{suffix}'
    self.group_metric_name = f'plan_executor_group_metric_{suffix}'
    self.failing_metric_name = f'plan_executor_failing_metric_{suffix}'

    self.agent_id = uuid4()
    self.dataset_id = uuid4()
    self.sample_id = uuid4()
    self.sample_two_id = uuid4()
    self.ground_truth_ids = [uuid4(), uuid4()]
    self.evaluation_run_id = uuid4()
    self.evaluation_run_sample_id = uuid4()
    self.evaluation_run_sample_two_id = uuid4()

    self.trace_id = f'plan_executor_trace_{suffix}'
    self.span_id = f'plan_executor_span_{suffix}'
    self.trace_two_id = f'plan_executor_trace_two_{suffix}'
    self.span_two_id = f'plan_executor_span_two_{suffix}'

    self.created_computation_ids: list[UUID] = []

  async def asyncTearDown(self) -> None:
    if not hasattr(self, 'db_manager'):
      return

    async with UnitOfWork(self.db_manager) as uow:
      computation_ids = set(self.created_computation_ids)
      for run_sample_id in [self.evaluation_run_sample_id, self.evaluation_run_sample_two_id]:
        computations = await uow.metric_computations.list_by_evaluation_run_sample(run_sample_id)
        computation_ids.update(computation.id for computation in computations)
      for computation_id in computation_ids:
        await uow.metric_computations.delete(computation_id)

      for ground_truth_id in self.ground_truth_ids:
        await uow.ground_truths.delete(ground_truth_id)

      await uow.evaluation_run_samples.delete(self.evaluation_run_sample_two_id)
      await uow.evaluation_run_samples.delete(self.evaluation_run_sample_id)
      await uow.evaluation_runs.delete(self.evaluation_run_id)
      await uow.spans.delete(self.span_two_id)
      await uow.spans.delete(self.span_id)
      await uow.traces.delete(self.trace_two_id)
      await uow.traces.delete(self.trace_id)
      await uow.samples.delete(self.sample_two_id)
      await uow.samples.delete(self.sample_id)
      await uow.datasets.delete(self.dataset_id)
      await uow.agents.delete(self.agent_id)

      await uow.metric_target_span_types.delete_by_metric_and_span_type(self.length_metric_name, self.span_type_name)
      await uow.metric_target_span_types.delete_by_metric_and_span_type(self.match_metric_name, self.span_type_name)
      await uow.metric_target_span_types.delete_by_metric_and_span_type(self.contains_metric_name, self.span_type_name)
      await uow.metric_target_span_types.delete_by_metric_and_span_type(self.group_metric_name, self.span_type_name)
      await uow.metric_target_span_types.delete_by_metric_and_span_type(self.failing_metric_name, self.span_type_name)
      await uow.metrics.delete(self.length_metric_name)
      await uow.metrics.delete(self.match_metric_name)
      await uow.metrics.delete(self.contains_metric_name)
      await uow.metrics.delete(self.group_metric_name)
      await uow.metrics.delete(self.failing_metric_name)
      await uow.span_types.delete(self.span_type_name)

    await self.db_manager.close_async()

  async def test_execute_plan_and_persist_metric_computations(self) -> None:
    now = datetime.now(tz=timezone.utc)

    async with UnitOfWork(self.db_manager) as uow:
      await uow.span_types.create(
        SpanType(name=self.span_type_name, description='Span type for plan executor lifecycle test')
      )

    length_metric = _OutputLengthMetric(self.length_metric_name, self.span_type_name)
    match_metric = _GroundTruthMatchMetric(self.match_metric_name, self.span_type_name)
    contains_metric = _ContainsResponseMetric(self.contains_metric_name, self.span_type_name)

    registry = MetricRegistry(db_manager=self.db_manager)
    registry.register_many([length_metric, match_metric, contains_metric])
    await registry.sync_with_persistence()

    first_span = Span(
      external_id=self.span_id,
      trace_id=self.trace_id,
      parent_span_id=None,
      span_type=self.span_type_name,
      name='plan_executor_span',
      start_time=now,
      end_time=now,
      input_data='{}',
      output_data='expected-response',
      metadata=None,
    )
    second_span = Span(
      external_id=self.span_two_id,
      trace_id=self.trace_two_id,
      parent_span_id=None,
      span_type=self.span_type_name,
      name='plan_executor_span_two',
      start_time=now,
      end_time=now,
      input_data='{}',
      output_data='second-expected-response',
      metadata=None,
    )

    async with UnitOfWork(self.db_manager) as uow:
      await uow.agents.create(
        Agent(
          id=self.agent_id,
          name=f'plan_executor_agent_{self.agent_id.hex[:8]}',
          version_tag='v-test',
        )
      )
      await uow.datasets.create(Dataset(id=self.dataset_id, name=f'plan_executor_dataset_{self.dataset_id.hex[:8]}'))
      await uow.samples.create(
        Sample(
          id=self.sample_id,
          dataset_id=self.dataset_id,
          input_prompt='Plan executor lifecycle prompt',
          ground_truth_output='expected-response',
        )
      )
      await uow.samples.create(
        Sample(
          id=self.sample_two_id,
          dataset_id=self.dataset_id,
          input_prompt='Plan executor lifecycle prompt 2',
          ground_truth_output='second-expected-response',
        )
      )
      await uow.ground_truths.create(
        GroundTruth(
          id=self.ground_truth_ids[0],
          sample_id=self.sample_id,
          key=self.match_metric_name,
          ground_truth_value={'expected_output': 'expected-response'},
        )
      )
      await uow.ground_truths.create(
        GroundTruth(
          id=self.ground_truth_ids[1],
          sample_id=self.sample_two_id,
          key=self.match_metric_name,
          ground_truth_value={'expected_output': 'second-expected-response'},
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
      await uow.traces.create(Trace(external_id=self.trace_id, start_time=now, end_time=now))
      await uow.traces.create(Trace(external_id=self.trace_two_id, start_time=now, end_time=now))
      await uow.spans.create(first_span)
      await uow.spans.create(second_span)
      await uow.evaluation_run_samples.create(
        EvaluationRunSample(
          id=self.evaluation_run_sample_id,
          evaluation_run_id=self.evaluation_run_id,
          sample_id=self.sample_id,
          trace_id=self.trace_id,
        )
      )
      await uow.evaluation_run_samples.create(
        EvaluationRunSample(
          id=self.evaluation_run_sample_two_id,
          evaluation_run_id=self.evaluation_run_id,
          sample_id=self.sample_two_id,
          trace_id=self.trace_two_id,
        )
      )

    planner = MetricPlanner(metric_registry=registry, db_manager=self.db_manager)
    executor = PlanExecutor(db_manager=self.db_manager, max_concurrent_tasks=10)
    first_sample_computations = await executor.execute(
      plan_items=planner.iter_plan_items(trace_id=self.trace_id, sample_id=self.sample_id),
      evaluation_run_sample_id=self.evaluation_run_sample_id,
    )

    second_sample_computations = await executor.execute(
      plan_items=planner.iter_plan_items(trace_id=self.trace_two_id, sample_id=self.sample_two_id),
      evaluation_run_sample_id=self.evaluation_run_sample_two_id,
    )
    computations = first_sample_computations + second_sample_computations

    self.assertEqual(len(computations), 6)
    self.created_computation_ids = [computation.id for computation in computations]

    computations_by_sample_and_metric = {
      (computation.evaluation_run_sample_id, computation.metric): computation for computation in computations
    }
    expected_keys = {
      (self.evaluation_run_sample_id, self.length_metric_name),
      (self.evaluation_run_sample_id, self.match_metric_name),
      (self.evaluation_run_sample_id, self.contains_metric_name),
      (self.evaluation_run_sample_two_id, self.length_metric_name),
      (self.evaluation_run_sample_two_id, self.match_metric_name),
      (self.evaluation_run_sample_two_id, self.contains_metric_name),
    }
    self.assertEqual(set(computations_by_sample_and_metric.keys()), expected_keys)

    length_computation = computations_by_sample_and_metric[(self.evaluation_run_sample_id, self.length_metric_name)]
    self.assertEqual(length_computation.evaluation_run_sample_id, self.evaluation_run_sample_id)
    self.assertIsNone(length_computation.ground_truth_id)
    self.assertEqual(length_computation.span_ids, [self.span_id])
    self.assertEqual(length_computation.score, float(len(str(first_span.output_data))))
    self.assertEqual(length_computation.reasoning, 'Score is output length.')
    self.assertEqual(_metric_metadata(length_computation.metadata), {'computed_from': 'output_data_length'})

    match_computation = computations_by_sample_and_metric[(self.evaluation_run_sample_id, self.match_metric_name)]
    self.assertEqual(match_computation.evaluation_run_sample_id, self.evaluation_run_sample_id)
    self.assertEqual(match_computation.ground_truth_id, self.ground_truth_ids[0])
    self.assertEqual(match_computation.span_ids, [self.span_id])
    self.assertEqual(match_computation.score, 1.0)
    self.assertEqual(match_computation.reasoning, 'Output matches expected output.')
    self.assertEqual(
      _metric_metadata(match_computation.metadata),
      {'expected_output': 'expected-response', 'actual_output': 'expected-response'},
    )

    contains_computation = computations_by_sample_and_metric[(self.evaluation_run_sample_id, self.contains_metric_name)]
    self.assertEqual(contains_computation.evaluation_run_sample_id, self.evaluation_run_sample_id)
    self.assertIsNone(contains_computation.ground_truth_id)
    self.assertEqual(contains_computation.span_ids, [self.span_id])
    self.assertEqual(contains_computation.score, 1.0)
    self.assertEqual(contains_computation.reasoning, "Output contains 'response'.")
    self.assertEqual(_metric_metadata(contains_computation.metadata), {'needle': 'response'})

    async with UnitOfWork(self.db_manager) as uow:
      persisted_first_sample_computations = await uow.metric_computations.list_by_evaluation_run_sample(
        self.evaluation_run_sample_id
      )
      persisted_second_sample_computations = await uow.metric_computations.list_by_evaluation_run_sample(
        self.evaluation_run_sample_two_id
      )

    persisted_computations = persisted_first_sample_computations + persisted_second_sample_computations
    self.assertEqual(len(persisted_computations), 6)

    persisted_by_sample_and_metric = {
      (computation.evaluation_run_sample_id, computation.metric): computation for computation in persisted_computations
    }
    self.assertEqual(set(persisted_by_sample_and_metric.keys()), expected_keys)

    self.assertEqual(
      persisted_by_sample_and_metric[(self.evaluation_run_sample_id, self.length_metric_name)].score,
      float(len(str(first_span.output_data))),
    )
    self.assertEqual(
      persisted_by_sample_and_metric[(self.evaluation_run_sample_two_id, self.length_metric_name)].score,
      float(len(str(second_span.output_data))),
    )
    self.assertEqual(
      persisted_by_sample_and_metric[(self.evaluation_run_sample_two_id, self.match_metric_name)].score,
      1.0,
    )
    self.assertEqual(
      persisted_by_sample_and_metric[(self.evaluation_run_sample_two_id, self.match_metric_name)].ground_truth_id,
      self.ground_truth_ids[1],
    )
    self.assertEqual(
      persisted_by_sample_and_metric[(self.evaluation_run_sample_two_id, self.contains_metric_name)].score,
      1.0,
    )

  async def test_execute_group_metric_and_persist_span_membership(self) -> None:
    now = datetime.now(tz=timezone.utc)

    async with UnitOfWork(self.db_manager) as uow:
      await uow.span_types.create(
        SpanType(name=self.span_type_name, description='Span type for group plan executor test')
      )

    group_metric = _GroupedOutputLengthMetric(self.group_metric_name, self.span_type_name)
    registry = MetricRegistry(db_manager=self.db_manager)
    registry.register(group_metric)
    await registry.sync_with_persistence()

    first_span = Span(
      external_id=self.span_id,
      trace_id=self.trace_id,
      parent_span_id=None,
      span_type=self.span_type_name,
      name='plan_executor_group_span_one',
      start_time=now,
      end_time=now,
      input_data='{}',
      output_data='alpha',
      metadata=None,
    )
    second_span = Span(
      external_id=self.span_two_id,
      trace_id=self.trace_id,
      parent_span_id=None,
      span_type=self.span_type_name,
      name='plan_executor_group_span_two',
      start_time=now,
      end_time=now,
      input_data='{}',
      output_data='beta',
      metadata=None,
    )

    async with UnitOfWork(self.db_manager) as uow:
      await uow.agents.create(
        Agent(
          id=self.agent_id,
          name=f'plan_executor_group_agent_{self.agent_id.hex[:8]}',
          version_tag='v-test',
        )
      )
      await uow.datasets.create(
        Dataset(id=self.dataset_id, name=f'plan_executor_group_dataset_{self.dataset_id.hex[:8]}')
      )
      await uow.samples.create(
        Sample(
          id=self.sample_id,
          dataset_id=self.dataset_id,
          input_prompt='Plan executor group metric prompt',
          ground_truth_output=None,
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
      await uow.traces.create(Trace(external_id=self.trace_id, start_time=now, end_time=now))
      await uow.spans.create(first_span)
      await uow.spans.create(second_span)
      await uow.evaluation_run_samples.create(
        EvaluationRunSample(
          id=self.evaluation_run_sample_id,
          evaluation_run_id=self.evaluation_run_id,
          sample_id=self.sample_id,
          trace_id=self.trace_id,
        )
      )

    planner = MetricPlanner(metric_registry=registry, db_manager=self.db_manager)
    executor = PlanExecutor(db_manager=self.db_manager, max_concurrent_tasks=10)
    computations = await executor.execute(
      plan_items=planner.iter_plan_items(trace_id=self.trace_id, sample_id=self.sample_id),
      evaluation_run_sample_id=self.evaluation_run_sample_id,
    )

    self.assertEqual(len(computations), 1)
    self.created_computation_ids.extend(computation.id for computation in computations)

    group_computation = computations[0]
    self.assertEqual(group_computation.metric, self.group_metric_name)
    self.assertEqual(group_computation.targeting_mode.value, 'GROUP')
    self.assertEqual(group_computation.target_span_type, self.span_type_name)
    self.assertEqual(group_computation.span_ids, [self.span_id, self.span_two_id])
    self.assertEqual(group_computation.score, float(len('alpha') + len('beta')))
    self.assertEqual(_metric_metadata(group_computation.metadata), {'span_count': 2})

    async with UnitOfWork(self.db_manager) as uow:
      persisted = await uow.metric_computations.list_by_evaluation_run_sample(self.evaluation_run_sample_id)
      persisted_for_span = await uow.metric_computations.list_by_span(self.span_two_id)

    self.assertEqual(len(persisted), 1)
    self.assertEqual(persisted[0].span_ids, [self.span_id, self.span_two_id])
    self.assertEqual(len(persisted_for_span), 1)
    self.assertEqual(persisted_for_span[0].id, group_computation.id)

  async def test_execute_persists_failed_metric_computation_when_metric_raises(self) -> None:
    now = datetime.now(tz=timezone.utc)

    async with UnitOfWork(self.db_manager) as uow:
      await uow.span_types.create(
        SpanType(name=self.span_type_name, description='Span type for failed plan executor test')
      )

    failing_metric = _FailingMetric(self.failing_metric_name, self.span_type_name)
    registry = MetricRegistry(db_manager=self.db_manager)
    registry.register(failing_metric)
    await registry.sync_with_persistence()

    async with UnitOfWork(self.db_manager) as uow:
      await uow.agents.create(
        Agent(
          id=self.agent_id,
          name=f'plan_executor_failed_agent_{self.agent_id.hex[:8]}',
          version_tag='v-test',
        )
      )
      await uow.datasets.create(
        Dataset(id=self.dataset_id, name=f'plan_executor_failed_dataset_{self.dataset_id.hex[:8]}')
      )
      await uow.samples.create(
        Sample(id=self.sample_id, dataset_id=self.dataset_id, input_prompt='Plan executor failed metric prompt')
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
      await uow.traces.create(Trace(external_id=self.trace_id, start_time=now, end_time=now))
      await uow.spans.create(
        Span(
          external_id=self.span_id,
          trace_id=self.trace_id,
          parent_span_id=None,
          span_type=self.span_type_name,
          name='plan_executor_failed_span',
          start_time=now,
          end_time=now,
          input_data='{}',
          output_data='answer',
        )
      )
      await uow.evaluation_run_samples.create(
        EvaluationRunSample(
          id=self.evaluation_run_sample_id,
          evaluation_run_id=self.evaluation_run_id,
          sample_id=self.sample_id,
          trace_id=self.trace_id,
        )
      )

    planner = MetricPlanner(metric_registry=registry, db_manager=self.db_manager)
    executor = PlanExecutor(db_manager=self.db_manager, max_concurrent_tasks=10)
    computations = await executor.execute(
      plan_items=planner.iter_plan_items(trace_id=self.trace_id, sample_id=self.sample_id),
      evaluation_run_sample_id=self.evaluation_run_sample_id,
    )

    self.assertEqual(len(computations), 1)
    self.created_computation_ids.extend(computation.id for computation in computations)
    self.assertEqual(computations[0].status, MetricComputationStatus.FAILED)
    self.assertIsNone(computations[0].score)
    self.assertEqual(computations[0].error_message, 'judge service unavailable')
    self.assertEqual(computations[0].span_ids, [self.span_id])

  async def test_execute_persists_skipped_metric_computation_when_ground_truth_is_missing(self) -> None:
    now = datetime.now(tz=timezone.utc)

    async with UnitOfWork(self.db_manager) as uow:
      await uow.span_types.create(
        SpanType(name=self.span_type_name, description='Span type for skipped plan executor test')
      )

    match_metric = _GroundTruthMatchMetric(self.match_metric_name, self.span_type_name)
    registry = MetricRegistry(db_manager=self.db_manager)
    registry.register(match_metric)
    await registry.sync_with_persistence()

    async with UnitOfWork(self.db_manager) as uow:
      await uow.agents.create(
        Agent(
          id=self.agent_id,
          name=f'plan_executor_skipped_agent_{self.agent_id.hex[:8]}',
          version_tag='v-test',
        )
      )
      await uow.datasets.create(
        Dataset(id=self.dataset_id, name=f'plan_executor_skipped_dataset_{self.dataset_id.hex[:8]}')
      )
      await uow.samples.create(
        Sample(id=self.sample_id, dataset_id=self.dataset_id, input_prompt='Plan executor skipped metric prompt')
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
      await uow.traces.create(Trace(external_id=self.trace_id, start_time=now, end_time=now))
      await uow.spans.create(
        Span(
          external_id=self.span_id,
          trace_id=self.trace_id,
          parent_span_id=None,
          span_type=self.span_type_name,
          name='plan_executor_skipped_span',
          start_time=now,
          end_time=now,
          input_data='{}',
          output_data='answer',
        )
      )
      await uow.evaluation_run_samples.create(
        EvaluationRunSample(
          id=self.evaluation_run_sample_id,
          evaluation_run_id=self.evaluation_run_id,
          sample_id=self.sample_id,
          trace_id=self.trace_id,
        )
      )

    planner = MetricPlanner(metric_registry=registry, db_manager=self.db_manager)
    executor = PlanExecutor(db_manager=self.db_manager, max_concurrent_tasks=10)
    computations = await executor.execute(
      plan_items=planner.iter_plan_items(trace_id=self.trace_id, sample_id=self.sample_id),
      evaluation_run_sample_id=self.evaluation_run_sample_id,
    )

    self.assertEqual(len(computations), 1)
    self.created_computation_ids.extend(computation.id for computation in computations)
    self.assertEqual(computations[0].status, MetricComputationStatus.SKIPPED)
    self.assertIsNone(computations[0].score)
    self.assertEqual(computations[0].error_message, 'Missing required ground truth.')
    self.assertEqual(computations[0].span_ids, [self.span_id])

  async def test_execute_persists_skipped_metric_computation_when_target_span_is_missing(self) -> None:
    now = datetime.now(tz=timezone.utc)

    async with UnitOfWork(self.db_manager) as uow:
      await uow.span_types.create(
        SpanType(name=self.span_type_name, description='Span type for missing-target plan executor test')
      )

    length_metric = _OutputLengthMetric(self.length_metric_name, self.span_type_name)
    registry = MetricRegistry(db_manager=self.db_manager)
    registry.register(length_metric)
    await registry.sync_with_persistence()

    async with UnitOfWork(self.db_manager) as uow:
      await uow.agents.create(
        Agent(
          id=self.agent_id,
          name=f'plan_executor_missing_target_agent_{self.agent_id.hex[:8]}',
          version_tag='v-test',
        )
      )
      await uow.datasets.create(
        Dataset(id=self.dataset_id, name=f'plan_executor_missing_target_dataset_{self.dataset_id.hex[:8]}')
      )
      await uow.samples.create(
        Sample(id=self.sample_id, dataset_id=self.dataset_id, input_prompt='Plan executor missing target prompt')
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
      await uow.traces.create(Trace(external_id=self.trace_id, start_time=now, end_time=now))
      await uow.evaluation_run_samples.create(
        EvaluationRunSample(
          id=self.evaluation_run_sample_id,
          evaluation_run_id=self.evaluation_run_id,
          sample_id=self.sample_id,
          trace_id=self.trace_id,
        )
      )

    planner = MetricPlanner(metric_registry=registry, db_manager=self.db_manager)
    executor = PlanExecutor(db_manager=self.db_manager, max_concurrent_tasks=10)
    computations = await executor.execute(
      plan_items=planner.iter_plan_items(trace_id=self.trace_id, sample_id=self.sample_id),
      evaluation_run_sample_id=self.evaluation_run_sample_id,
    )

    self.assertEqual(len(computations), 1)
    self.created_computation_ids.extend(computation.id for computation in computations)
    self.assertEqual(computations[0].metric, self.length_metric_name)
    self.assertEqual(computations[0].status, MetricComputationStatus.SKIPPED)
    self.assertIsNone(computations[0].score)
    self.assertEqual(computations[0].target_span_type, self.span_type_name)
    self.assertEqual(computations[0].span_ids, [])
    self.assertEqual(
      computations[0].error_message,
      f'No spans found for target span type "{self.span_type_name}".',
    )

    async with UnitOfWork(self.db_manager) as uow:
      persisted = await uow.metric_computations.list_by_evaluation_run_sample(self.evaluation_run_sample_id)

    self.assertEqual(len(persisted), 1)
    self.assertEqual(persisted[0].span_ids, [])
    self.assertEqual(persisted[0].status, MetricComputationStatus.SKIPPED)


if __name__ == '__main__':
  unittest.main()
