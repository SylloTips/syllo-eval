import unittest
from datetime import datetime, timedelta, timezone
from unittest.mock import patch
from uuid import UUID, uuid4

from syllo_eval.evaluation.metric_planner import (
  MetricPlanItem,
  MetricPlanner,
  MissingTarget,
  SpanGroupTarget,
  SpanTarget,
)
from syllo_eval.evaluation.metric_registry import MetricRegistry
from syllo_eval.evaluation.metrics.contracts import (
  EvaluationMetric,
  MetricComputationResult,
  SpanGroupEvaluationMetric,
)
from syllo_eval.infrastructure import UnitOfWork
from syllo_eval.model import Dataset, GroundTruth, Metric, MetricTargetSpanType, Sample, Span, SpanType, Trace
from syllo_eval.testing_database import setup_test_database as _setup_database


class _DummyMetric(EvaluationMetric):
  def __init__(
    self,
    name: str,
    target_span_types: tuple[str, ...],
    requires_ground_truth: bool = True,
    ground_truth_key: str | None = None,
  ):
    self._name = name
    self._target_span_types = target_span_types
    self._requires_ground_truth = requires_ground_truth
    self._ground_truth_key = ground_truth_key or name

  @property
  def name(self) -> str:
    return self._name

  @property
  def target_span_types(self) -> tuple[str, ...]:
    return self._target_span_types

  @property
  def requires_ground_truth(self) -> bool:
    return self._requires_ground_truth

  @property
  def ground_truth_key(self) -> str | None:
    return self._ground_truth_key

  async def compute(self, span: Span, ground_truth: GroundTruth | None) -> MetricComputationResult:
    del span, ground_truth
    return MetricComputationResult(score=1.0)


class _GroupDummyMetric(SpanGroupEvaluationMetric):
  def __init__(
    self,
    name: str,
    target_span_types: tuple[str, ...],
    requires_ground_truth: bool = True,
    ground_truth_key: str | None = None,
  ):
    self._name = name
    self._target_span_types = target_span_types
    self._requires_ground_truth = requires_ground_truth
    self._ground_truth_key = ground_truth_key or name

  @property
  def name(self) -> str:
    return self._name

  @property
  def target_span_types(self) -> tuple[str, ...]:
    return self._target_span_types

  @property
  def requires_ground_truth(self) -> bool:
    return self._requires_ground_truth

  @property
  def ground_truth_key(self) -> str | None:
    return self._ground_truth_key

  async def compute(self, spans, ground_truth: GroundTruth | None) -> MetricComputationResult:
    del ground_truth
    return MetricComputationResult(score=float(len(spans)))


class _MetricPlannerIntegrationBase(unittest.IsolatedAsyncioTestCase):
  async def asyncSetUp(self) -> None:
    try:
      self.db_manager = await _setup_database()
    except RuntimeError as error:
      self.skipTest(str(error))

    self._suffix = uuid4().hex[:10]

    self.dataset_id = uuid4()
    self.sample_id = uuid4()
    self.trace_id = f'metric_planner_trace_{self._suffix}'

    self._created_span_ids: list[str] = []
    self._created_ground_truth_ids: list[UUID] = []
    self._created_span_types: set[str] = set()
    self._created_metrics: set[str] = set()
    self._created_metric_mappings: set[tuple[str, str]] = set()

    now = datetime.now(tz=timezone.utc)
    async with UnitOfWork(self.db_manager) as uow:
      await uow.datasets.create(Dataset(id=self.dataset_id, name=f'metric_planner_dataset_{self._suffix}'))
      await uow.samples.create(
        Sample(
          id=self.sample_id,
          dataset_id=self.dataset_id,
          input_prompt='Metric planner prompt',
          ground_truth_output='Metric planner expected output',
        )
      )
      await uow.traces.create(Trace(external_id=self.trace_id, start_time=now, end_time=now))

  async def asyncTearDown(self) -> None:
    if not hasattr(self, 'db_manager'):
      return

    async with UnitOfWork(self.db_manager) as uow:
      for ground_truth_id in self._created_ground_truth_ids:
        await uow.ground_truths.delete(ground_truth_id)

      for span_id in self._created_span_ids:
        await uow.spans.delete(span_id)

      await uow.traces.delete(self.trace_id)
      await uow.samples.delete(self.sample_id)
      await uow.datasets.delete(self.dataset_id)

      for metric_name, span_type_name in sorted(self._created_metric_mappings):
        await uow.metric_target_span_types.delete_by_metric_and_span_type(metric_name, span_type_name)

      for metric_name in sorted(self._created_metrics):
        await uow.metrics.delete(metric_name)

      for span_type_name in sorted(self._created_span_types):
        await uow.span_types.delete(span_type_name)

    await self.db_manager.close_async()

  def _new_name(self, prefix: str) -> str:
    return f'{prefix}_{self._suffix}_{uuid4().hex[:6]}'

  async def _create_span_type(self, span_type_name: str) -> None:
    if span_type_name in self._created_span_types:
      return

    async with UnitOfWork(self.db_manager) as uow:
      await uow.span_types.create(SpanType(name=span_type_name, description=f'Test span type {span_type_name}'))
    self._created_span_types.add(span_type_name)

  async def _create_span(
    self,
    span_id: str,
    span_type_name: str,
    *,
    start_time: datetime | None = None,
    end_time: datetime | None = None,
  ) -> None:
    effective_start_time = start_time or datetime.now(tz=timezone.utc)
    effective_end_time = end_time or effective_start_time
    async with UnitOfWork(self.db_manager) as uow:
      await uow.spans.create(
        Span(
          external_id=span_id,
          trace_id=self.trace_id,
          parent_span_id=None,
          span_type=span_type_name,
          name=f'{span_type_name}_span',
          start_time=effective_start_time,
          end_time=effective_end_time,
          input_data='{}',
          output_data='{}',
          metadata=None,
        )
      )
    self._created_span_ids.append(span_id)

  async def _create_ground_truth(self, key: str) -> None:
    ground_truth_id = uuid4()
    async with UnitOfWork(self.db_manager) as uow:
      await uow.ground_truths.create(
        GroundTruth(
          id=ground_truth_id,
          sample_id=self.sample_id,
          key=key,
          ground_truth_value={'expected': key},
        )
      )
    self._created_ground_truth_ids.append(ground_truth_id)

  async def _create_registry(self, metrics: list[EvaluationMetric]) -> MetricRegistry:
    span_types = {span_type for metric in metrics for span_type in metric.target_span_types}
    for span_type_name in span_types:
      if span_type_name not in self._created_span_types:
        await self._create_span_type(span_type_name)

    registry = MetricRegistry(db_manager=self.db_manager)
    registry.register_many(metrics)
    await registry.sync_with_persistence()

    for metric in metrics:
      self._created_metrics.add(metric.name)
      for span_type_name in metric.target_span_types:
        self._created_metric_mappings.add((metric.name, span_type_name))

    return registry

  async def _create_persisted_metric_without_implementation(self, metric_name: str, span_type_name: str) -> None:
    await self._create_span_type(span_type_name)

    async with UnitOfWork(self.db_manager) as uow:
      await uow.metrics.create(
        Metric(name=metric_name, description='Persisted metric without in-memory implementation')
      )
      await uow.metric_target_span_types.create(
        MetricTargetSpanType(id=uuid4(), metric=metric_name, span_type=span_type_name)
      )

    self._created_metrics.add(metric_name)
    self._created_metric_mappings.add((metric_name, span_type_name))


class _RequiredLifecycleMetric(EvaluationMetric):
  def __init__(self, metric_name: str, span_type_name: str):
    self._metric_name = metric_name
    self._span_type_name = span_type_name

  @property
  def name(self) -> str:
    return self._metric_name

  @property
  def description(self) -> str:
    return 'Lifecycle test metric requiring ground truth.'

  @property
  def target_span_types(self) -> tuple[str, ...]:
    return (self._span_type_name,)

  @property
  def ground_truth_key(self) -> str | None:
    return self._metric_name

  async def compute(self, span: Span, ground_truth: GroundTruth | None) -> MetricComputationResult:
    del span
    if ground_truth is None:
      return MetricComputationResult(score=0.0, reasoning='Missing ground truth.')
    return MetricComputationResult(score=1.0, reasoning='Ground truth provided.')


class _OptionalLifecycleMetric(EvaluationMetric):
  def __init__(self, metric_name: str, span_type_name: str):
    self._metric_name = metric_name
    self._span_type_name = span_type_name

  @property
  def name(self) -> str:
    return self._metric_name

  @property
  def description(self) -> str:
    return 'Lifecycle test metric with optional ground truth.'

  @property
  def requires_ground_truth(self) -> bool:
    return False

  @property
  def target_span_types(self) -> tuple[str, ...]:
    return (self._span_type_name,)

  @property
  def ground_truth_key(self) -> str | None:
    return self._metric_name

  async def compute(self, span: Span, ground_truth: GroundTruth | None) -> MetricComputationResult:
    del span
    if ground_truth is None:
      return MetricComputationResult(score=1.0, reasoning='No ground truth required.')
    return MetricComputationResult(score=0.5, reasoning='Ground truth provided.')


class TestMetricPlanner(_MetricPlannerIntegrationBase):
  async def test_yields_expected_triplets_when_ground_truth_exists(self) -> None:
    span_type = self._new_name('agent_span_type')
    metric_a_name = self._new_name('metric_a')
    metric_b_name = self._new_name('metric_b')
    span_id = self._new_name('span')

    metrics: list[EvaluationMetric] = [
      _DummyMetric(metric_a_name, target_span_types=(span_type,)),
      _DummyMetric(metric_b_name, target_span_types=(span_type,)),
    ]
    registry = await self._create_registry(metrics)
    await self._create_span(span_id, span_type)
    await self._create_ground_truth(metric_a_name)
    await self._create_ground_truth(metric_b_name)

    planner = MetricPlanner(metric_registry=registry, db_manager=self.db_manager)
    items = [item async for item in planner.iter_plan_items(self.trace_id, self.sample_id)]

    self.assertEqual(len(items), 2)
    self.assertCountEqual([item.metric_name for item in items], [metric_a_name, metric_b_name])
    self.assertTrue(all(item.ground_truth is not None for item in items))

  async def test_required_metric_missing_ground_truth_yields_skipped_item(self) -> None:
    span_type = self._new_name('agent_span_type')
    metric_name = self._new_name('required_metric')
    span_id = self._new_name('span')

    metric = _DummyMetric(metric_name, target_span_types=(span_type,), requires_ground_truth=True)
    registry = await self._create_registry([metric])
    await self._create_span(span_id, span_type)

    planner = MetricPlanner(metric_registry=registry, db_manager=self.db_manager)
    with patch('syllo_eval.evaluation.metric_planner.logger.warning'):
      items = [item async for item in planner.iter_plan_items(self.trace_id, self.sample_id)]

    self.assertEqual(len(items), 1)
    self.assertIsInstance(items[0], MetricPlanItem)
    self.assertEqual(items[0].metric_name, metric_name)
    self.assertEqual(items[0].span_ids, [span_id])
    self.assertEqual(items[0].skip_reason, 'Missing required ground truth.')

  async def test_optional_metric_missing_ground_truth_yields_none(self) -> None:
    span_type = self._new_name('agent_span_type')
    metric_name = self._new_name('optional_metric')
    span_id = self._new_name('span')

    metric = _DummyMetric(metric_name, target_span_types=(span_type,), requires_ground_truth=False)
    registry = await self._create_registry([metric])
    await self._create_span(span_id, span_type)

    planner = MetricPlanner(metric_registry=registry, db_manager=self.db_manager)
    items = [item async for item in planner.iter_plan_items(self.trace_id, self.sample_id)]

    self.assertEqual(len(items), 1)
    self.assertEqual(items[0].metric_name, metric_name)
    self.assertIsNone(items[0].ground_truth)

  async def test_missing_ground_truth_warning_emitted_once_per_metric_per_sample(self) -> None:
    span_type = self._new_name('agent_span_type')
    metric_name = self._new_name('required_metric')
    span_id_1 = self._new_name('span')
    span_id_2 = self._new_name('span')
    metric = _DummyMetric(metric_name, target_span_types=(span_type,), requires_ground_truth=True)

    registry = await self._create_registry([metric])
    await self._create_span(span_id_1, span_type)
    await self._create_span(span_id_2, span_type)

    planner = MetricPlanner(metric_registry=registry, db_manager=self.db_manager)

    with (
      patch('syllo_eval.evaluation.metric_planner.logger.warning') as warning_mock,
    ):
      items = [item async for item in planner.iter_plan_items(self.trace_id, self.sample_id)]

    self.assertEqual(len(items), 2)
    self.assertTrue(all(isinstance(item, MetricPlanItem) for item in items))
    self.assertTrue(all(item.skip_reason == 'Missing required ground truth.' for item in items))
    self.assertCountEqual([item.span_ids[0] for item in items], [span_id_1, span_id_2])
    warning_mock.assert_called_once()
    self.assertIn('Missing required ground truth', warning_mock.call_args.args[0])

  async def test_registered_metrics_are_planned_for_present_target_spans(self) -> None:
    agent_span_type = self._new_name('agent_span_type')
    planner_span_type = self._new_name('planner_span_type')

    agent_metric = _DummyMetric(
      self._new_name('metric_agent'),
      target_span_types=(agent_span_type,),
      requires_ground_truth=False,
    )
    planner_metric = _DummyMetric(
      self._new_name('metric_planner'),
      target_span_types=(planner_span_type,),
      requires_ground_truth=False,
    )

    registry = await self._create_registry([agent_metric, planner_metric])
    await self._create_span(self._new_name('span'), agent_span_type)
    await self._create_span(self._new_name('span'), agent_span_type)
    await self._create_span(self._new_name('span'), planner_span_type)

    planner = MetricPlanner(metric_registry=registry, db_manager=self.db_manager)
    items = [item async for item in planner.iter_plan_items(self.trace_id, self.sample_id)]

    self.assertEqual(len(items), 3)
    metric_names = [item.metric_name for item in items]
    self.assertEqual(metric_names.count(agent_metric.name), 2)
    self.assertEqual(metric_names.count(planner_metric.name), 1)

  async def test_group_metric_yields_one_item_with_ordered_spans(self) -> None:
    span_type = self._new_name('retrieval_span_type')
    metric_name = self._new_name('group_metric')
    late_span_id = self._new_name('span')
    early_span_id = self._new_name('span')
    middle_span_id = self._new_name('span')

    metric = _GroupDummyMetric(metric_name, target_span_types=(span_type,), requires_ground_truth=False)
    registry = await self._create_registry([metric])

    base_time = datetime.now(tz=timezone.utc)
    await self._create_span(late_span_id, span_type, start_time=base_time + timedelta(seconds=2))
    await self._create_span(early_span_id, span_type, start_time=base_time)
    await self._create_span(middle_span_id, span_type, start_time=base_time + timedelta(seconds=1))

    planner = MetricPlanner(metric_registry=registry, db_manager=self.db_manager)
    items = [item async for item in planner.iter_plan_items(self.trace_id, self.sample_id)]

    self.assertEqual(len(items), 1)
    assert isinstance(items[0].target, SpanGroupTarget)
    self.assertEqual(items[0].metric_name, metric_name)
    self.assertEqual(items[0].span_ids, [early_span_id, middle_span_id, late_span_id])

  async def test_missing_target_span_yields_skipped_item(self) -> None:
    span_type = self._new_name('missing_span_type')
    metric_name = self._new_name('missing_target_metric')

    metric = _DummyMetric(metric_name, target_span_types=(span_type,), requires_ground_truth=False)
    registry = await self._create_registry([metric])

    planner = MetricPlanner(metric_registry=registry, db_manager=self.db_manager)
    items = [item async for item in planner.iter_plan_items(self.trace_id, self.sample_id)]

    self.assertEqual(len(items), 1)
    self.assertIsInstance(items[0].target, MissingTarget)
    self.assertEqual(items[0].metric_name, metric_name)
    self.assertEqual(items[0].target_span_type, span_type)
    self.assertEqual(items[0].span_ids, [])
    self.assertEqual(items[0].skip_reason, f'No spans found for target span type "{span_type}".')

  async def test_generator_is_lazy(self) -> None:
    span_type = self._new_name('agent_span_type')
    metric_name = self._new_name('metric_a')
    span_id = self._new_name('span')

    metric = _DummyMetric(metric_name, target_span_types=(span_type,), requires_ground_truth=False)
    registry = await self._create_registry([metric])
    planner = MetricPlanner(metric_registry=registry, db_manager=self.db_manager)

    generator = planner.iter_plan_items(self.trace_id, self.sample_id)

    await self._create_span(span_id, span_type)

    first_item = await anext(generator)
    self.assertIsInstance(first_item, MetricPlanItem)
    self.assertIsInstance(first_item.target, SpanTarget)
    self.assertEqual(first_item.metric_name, metric_name)
    self.assertEqual(first_item.span_ids, [span_id])

  async def test_missing_registered_metric_implementation_yields_no_items(self) -> None:
    span_type = self._new_name('agent_span_type')
    metric_name = self._new_name('missing_metric_impl')
    span_id = self._new_name('span')

    await self._create_persisted_metric_without_implementation(metric_name, span_type)
    await self._create_span(span_id, span_type)

    registry = MetricRegistry(db_manager=self.db_manager)
    planner = MetricPlanner(metric_registry=registry, db_manager=self.db_manager)

    items = [item async for item in planner.iter_plan_items(self.trace_id, self.sample_id)]
    self.assertEqual(items, [])


class TestMetricPlannerLifecyclePersistence(unittest.IsolatedAsyncioTestCase):
  async def asyncSetUp(self) -> None:
    try:
      self.db_manager = await _setup_database()
    except RuntimeError as error:
      self.skipTest(str(error))

    suffix = uuid4().hex[:10]

    self.span_type_name = f'lifecycle_span_type_{suffix}'
    self.required_metric_name = f'lifecycle_required_metric_{suffix}'
    self.optional_metric_name = f'lifecycle_optional_metric_{suffix}'
    self.missing_required_metric_name = f'lifecycle_missing_required_metric_{suffix}'

    self.dataset_id = uuid4()
    self.sample_id = uuid4()
    self.ground_truth_id = uuid4()
    self.trace_id = f'lifecycle_trace_{suffix}'
    self.span_id = f'lifecycle_span_{suffix}'

  async def asyncTearDown(self) -> None:
    if not hasattr(self, 'db_manager'):
      return

    async with UnitOfWork(self.db_manager) as uow:
      await uow.ground_truths.delete(self.ground_truth_id)
      await uow.spans.delete(self.span_id)
      await uow.traces.delete(self.trace_id)
      await uow.samples.delete(self.sample_id)
      await uow.datasets.delete(self.dataset_id)

      await uow.metric_target_span_types.delete_by_metric_and_span_type(self.required_metric_name, self.span_type_name)
      await uow.metric_target_span_types.delete_by_metric_and_span_type(self.optional_metric_name, self.span_type_name)
      await uow.metric_target_span_types.delete_by_metric_and_span_type(
        self.missing_required_metric_name, self.span_type_name
      )

      await uow.metrics.delete(self.required_metric_name)
      await uow.metrics.delete(self.optional_metric_name)
      await uow.metrics.delete(self.missing_required_metric_name)

      await uow.span_types.delete(self.span_type_name)

    await self.db_manager.close_async()

  async def test_metric_planner_lifecycle_with_optional_ground_truth(self) -> None:
    required_metric = _RequiredLifecycleMetric(self.required_metric_name, self.span_type_name)
    optional_metric = _OptionalLifecycleMetric(self.optional_metric_name, self.span_type_name)
    missing_required_metric = _RequiredLifecycleMetric(self.missing_required_metric_name, self.span_type_name)

    async with UnitOfWork(self.db_manager) as uow:
      await uow.span_types.create(
        SpanType(name=self.span_type_name, description='Span type for metric planner lifecycle test')
      )

    registry = MetricRegistry(db_manager=self.db_manager)
    registry.register_many([required_metric, optional_metric, missing_required_metric])
    await registry.sync_with_persistence()

    now = datetime.now(tz=timezone.utc)
    async with UnitOfWork(self.db_manager) as uow:
      await uow.datasets.create(Dataset(id=self.dataset_id, name=f'lifecycle_dataset_{self.dataset_id.hex[:8]}'))
      await uow.samples.create(
        Sample(
          id=self.sample_id,
          dataset_id=self.dataset_id,
          input_prompt='Lifecycle test prompt',
          ground_truth_output='Lifecycle expected output',
        )
      )
      await uow.traces.create(Trace(external_id=self.trace_id, start_time=now, end_time=now))
      await uow.spans.create(
        Span(
          external_id=self.span_id,
          trace_id=self.trace_id,
          parent_span_id=None,
          span_type=self.span_type_name,
          name='lifecycle_span',
          start_time=now,
          end_time=now,
          input_data='{}',
          output_data='{}',
          metadata=None,
        )
      )
      await uow.ground_truths.create(
        GroundTruth(
          id=self.ground_truth_id,
          sample_id=self.sample_id,
          key=self.required_metric_name,
          ground_truth_value={'expected': 'value'},
        )
      )

    planner = MetricPlanner(metric_registry=registry, db_manager=self.db_manager)
    plan_items = [item async for item in planner.iter_plan_items(self.trace_id, self.sample_id)]

    self.assertEqual(len(plan_items), 3)

    items_by_metric = {item.metric_name: item for item in plan_items}
    self.assertIn(self.required_metric_name, items_by_metric)
    self.assertIn(self.optional_metric_name, items_by_metric)
    self.assertIn(self.missing_required_metric_name, items_by_metric)

    required_item = items_by_metric[self.required_metric_name]
    optional_item = items_by_metric[self.optional_metric_name]
    missing_required_item = items_by_metric[self.missing_required_metric_name]

    assert isinstance(required_item.target, SpanTarget)
    assert isinstance(optional_item.target, SpanTarget)
    assert isinstance(missing_required_item, MetricPlanItem)
    self.assertEqual(required_item.span_ids, [self.span_id])
    self.assertIsNotNone(required_item.ground_truth)
    required_ground_truth = required_item.ground_truth
    assert required_ground_truth is not None
    self.assertEqual(required_ground_truth.sample_id, self.sample_id)

    self.assertEqual(optional_item.span_ids, [self.span_id])
    self.assertIsNone(optional_item.ground_truth)
    self.assertEqual(missing_required_item.span_ids, [self.span_id])
    self.assertEqual(missing_required_item.skip_reason, 'Missing required ground truth.')

    required_result = await required_metric.compute(required_item.target.compute_input, required_item.ground_truth)
    optional_result = await optional_metric.compute(optional_item.target.compute_input, optional_item.ground_truth)
    self.assertEqual(required_result.score, 1.0)
    self.assertEqual(optional_result.score, 1.0)


if __name__ == '__main__':
  unittest.main()
