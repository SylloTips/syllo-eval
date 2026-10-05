import asyncio
import random
import unittest
from datetime import datetime, timezone
from uuid import uuid4
from typing import Any, cast
from unittest.mock import AsyncMock, patch

from syllo_eval.evaluation.metric_registry import MetricRegistry

from syllo_eval.evaluation.metrics.contracts import (
  EvaluationMetric,
  MetricComputationResult,
  SpanGroupEvaluationMetric,
)
from syllo_eval.infrastructure import UnitOfWork
from syllo_eval.model import GroundTruth, MetricTargetingMode, Span, SpanType
from syllo_eval.testing_database import setup_test_database as _setup_database


async def _cleanup_test_metric(uow: UnitOfWork, metric_name: str) -> None:
  """Clean up test metric and its mappings from database."""
  mapped_span_types = await uow.metric_target_span_types.get_span_types_for_metric(metric_name)
  for span_type in mapped_span_types:
    await uow.metric_target_span_types.delete_by_metric_and_span_type(metric_name, span_type)
  await uow.metrics.delete(metric_name)


async def _ensure_span_types_exist(uow: UnitOfWork, span_type_names: list[str]) -> None:
  """Ensure span types exist in database."""
  for span_type_name in span_type_names:
    existing = await uow.span_types.get_by_id(span_type_name)
    if existing is None:
      await uow.span_types.create(SpanType(name=span_type_name, description=f'Test span type: {span_type_name}'))


async def _cleanup_span_types(uow: UnitOfWork, span_type_names: list[str]) -> None:
  """Clean up test span types from database."""
  for span_type_name in span_type_names:
    await uow.span_types.delete(span_type_name)


class DummyMetric(EvaluationMetric):
  """Minimal test metric for testing purposes."""

  def __init__(
    self,
    name: str,
    description: str | None = 'Test metric',
    span_types: tuple[str, ...] = ('test_span',),
    ground_truth_key: str | None = None,
  ):
    self._name = name
    self._description = description
    self._span_types = tuple(span_types) if span_types else ('test_span',)
    self._ground_truth_key = ground_truth_key

  @property
  def name(self) -> str:
    return self._name

  @property
  def description(self) -> str | None:
    return self._description

  @property
  def target_span_types(self) -> tuple[str, ...]:
    return tuple(self._span_types)

  @property
  def ground_truth_key(self) -> str | None:
    return self._ground_truth_key

  async def compute(self, span: Span, ground_truth: GroundTruth | None) -> MetricComputationResult:
    return MetricComputationResult(score=1.0, reasoning='Test reasoning', metadata={'source': 'test_metric'})


class TestMetric(EvaluationMetric):
  """Test metric implementation that returns a random score."""

  @property
  def name(self) -> str:
    return 'test_metric'

  @property
  def description(self) -> str:
    return 'Test metric used for registry testing.'

  @property
  def target_span_types(self) -> tuple[str, ...]:
    return ('planner', 'replanner', 'checker')

  async def compute(self, span: Span, ground_truth: GroundTruth | None) -> MetricComputationResult:
    del span, ground_truth  # Inputs are intentionally unused for this test metric.
    score = random.random()
    return MetricComputationResult(
      score=score,
      reasoning='Random score generated for testing.',
      metadata={'source': 'test_metric'},
    )


class GroupMetric(SpanGroupEvaluationMetric):
  """Test group metric implementation."""

  def __init__(self, name: str, span_types: tuple[str, ...]):
    self._name = name
    self._span_types = span_types

  @property
  def name(self) -> str:
    return self._name

  @property
  def description(self) -> str:
    return 'Group metric used for registry testing.'

  @property
  def target_span_types(self) -> tuple[str, ...]:
    return self._span_types

  async def compute(self, spans, ground_truth: GroundTruth | None) -> MetricComputationResult:
    del ground_truth
    return MetricComputationResult(score=float(len(spans)), reasoning='Group metric score.')


class TestMetricRegistryRegistration(unittest.IsolatedAsyncioTestCase):
  """Tests for in-memory metric registration."""

  async def asyncSetUp(self) -> None:
    self.db_manager = cast(Any, object())
    self.MetricRegistry = MetricRegistry
    self.test_metric = TestMetric()

  async def test_sync_persists_definitions_and_targets_without_loading_a_cache(self) -> None:
    registry = self.MetricRegistry(self.db_manager)
    metric = GroupMetric('custom_bla', ('bla',))
    registry.register(metric)
    uow = AsyncMock()
    uow.__aenter__.return_value = uow
    uow.metric_target_span_types.get_span_types_for_metric.return_value = ['obsolete']
    with patch('syllo_eval.evaluation.metric_registry.TransactionalUnitOfWork', return_value=uow):
      await registry.sync_with_persistence()
    self.assertIs(registry.get('CUSTOM_BLA'), metric)
    self.assertEqual(registry.list_registered(), [metric])
    self.assertEqual(uow.metrics.upsert_from_registry.call_args.args[0].name, 'custom_bla')
    self.assertEqual(uow.span_types.upsert_from_registry.call_args.args[0].name, 'bla')
    mapping = uow.metric_target_span_types.upsert_from_registry.call_args.args[0]
    self.assertEqual(
      (mapping.metric, mapping.span_type, mapping.targeting_mode), ('custom_bla', 'bla', MetricTargetingMode.GROUP)
    )
    uow.metric_target_span_types.delete_by_metric_and_span_type.assert_awaited_once_with('custom_bla', 'obsolete')
    uow.metric_target_span_types.list_all.assert_not_awaited()

  async def test_register_metric(self) -> None:
    """Test basic metric registration."""
    registry = self.MetricRegistry(db_manager=self.db_manager)
    registry.register(self.test_metric)

    retrieved = registry.get(self.test_metric.name)
    self.assertEqual(retrieved.name, self.test_metric.name)
    self.assertIs(retrieved, self.test_metric)

  async def test_register_duplicate_instance_is_idempotent(self) -> None:
    """Test registering same instance multiple times is allowed (idempotent)."""
    registry = self.MetricRegistry(db_manager=self.db_manager)
    registry.register(self.test_metric)
    registry.register(self.test_metric)

    retrieved = registry.get(self.test_metric.name)
    self.assertIs(retrieved, self.test_metric)

  async def test_register_duplicate_name_different_instance_raises(self) -> None:
    """Test registering different instance with same name raises ValueError."""
    registry = self.MetricRegistry(db_manager=self.db_manager)
    registry.register(self.test_metric)

    duplicate_metric = TestMetric()
    with self.assertRaises(ValueError) as ctx:
      registry.register(duplicate_metric)

    self.assertIn(self.test_metric.name, str(ctx.exception))
    self.assertIn('already registered', str(ctx.exception).lower())

  async def test_get_nonexistent_metric_raises(self) -> None:
    """Test getting unregistered metric raises KeyError."""
    registry = self.MetricRegistry(db_manager=self.db_manager)

    with self.assertRaises(KeyError) as ctx:
      registry.get('nonexistent_metric')

    self.assertIn('nonexistent_metric', str(ctx.exception))

  async def test_case_insensitive_metric_names(self) -> None:
    """Test metric names are case-insensitive."""
    registry = self.MetricRegistry(db_manager=self.db_manager)
    registry.register(self.test_metric)

    retrieved_upper = registry.get(self.test_metric.name.upper())
    retrieved_lower = registry.get(self.test_metric.name.lower())
    retrieved_mixed = registry.get(self.test_metric.name.title())

    self.assertIs(retrieved_upper, self.test_metric)
    self.assertIs(retrieved_lower, self.test_metric)
    self.assertIs(retrieved_mixed, self.test_metric)

  async def test_register_many(self) -> None:
    """Test registering multiple metrics at once."""
    metric1 = DummyMetric(name='metric_1', span_types=('span_a',))
    metric2 = DummyMetric(name='metric_2', span_types=('span_b',))
    metric3 = DummyMetric(name='metric_3', span_types=('span_c',))

    registry = self.MetricRegistry(db_manager=self.db_manager)
    registry.register_many([metric1, metric2, metric3])

    self.assertIs(registry.get('metric_1'), metric1)
    self.assertIs(registry.get('metric_2'), metric2)
    self.assertIs(registry.get('metric_3'), metric3)

  async def test_whitespace_in_metric_names_normalized(self) -> None:
    """Test metric names with whitespace are normalized."""
    metric = DummyMetric(name='  test_metric  ')

    registry = self.MetricRegistry(db_manager=self.db_manager)
    registry.register(metric)

    retrieved = registry.get('test_metric')
    self.assertIs(retrieved, metric)


class TestMetricRegistryPersistence(unittest.IsolatedAsyncioTestCase):
  """Tests for database persistence and sync operations."""

  async def asyncSetUp(self) -> None:
    try:
      self.db_manager = await _setup_database()
    except RuntimeError as error:
      self.skipTest(str(error))

    try:
      from syllo_eval.evaluation.metric_registry import MetricRegistry

      self.MetricRegistry = MetricRegistry
    except ModuleNotFoundError as error:
      self.skipTest(f'MetricRegistry dependencies are missing: {error}')

    self.test_metric = TestMetric()
    self.metric_name = self.test_metric.name
    self.expected_span_types = self.test_metric.target_span_types

    async with UnitOfWork(self.db_manager) as uow:
      await _cleanup_test_metric(uow, self.metric_name)

  async def asyncTearDown(self) -> None:
    if hasattr(self, 'db_manager'):
      async with UnitOfWork(self.db_manager) as uow:
        await _cleanup_test_metric(uow, self.metric_name)
        for name in [
          'group_metric',
          'metric_with_auto_span_type',
          'metric_with_desc',
          'metric_no_desc',
          'updated_desc_metric',
        ]:
          await _cleanup_test_metric(uow, name)
        await _cleanup_span_types(
          uow,
          ['auto_created_span_type', 'group_span', 'test_span', 'span_type_1', 'span_type_2'],
        )

      await self.db_manager.close_async()

  async def test_sync_creates_metric_in_db(self) -> None:
    """Test sync persists metric to database."""
    registry = self.MetricRegistry(db_manager=self.db_manager)
    registry.register(self.test_metric)
    await registry.sync_with_persistence()

    async with UnitOfWork(self.db_manager) as uow:
      persisted = await uow.metrics.get_by_id(self.metric_name)

    self.assertIsNotNone(persisted)
    self.assertEqual(persisted.name, self.test_metric.name)
    if self.test_metric.description:
      self.assertEqual(persisted.description, self.test_metric.description)

  async def test_sync_creates_span_type_mappings(self) -> None:
    """Test sync creates metric-span_type mappings in database."""
    registry = self.MetricRegistry(db_manager=self.db_manager)
    registry.register(self.test_metric)
    await registry.sync_with_persistence()

    async with UnitOfWork(self.db_manager) as uow:
      for span_type in self.expected_span_types:
        mapping = await uow.metric_target_span_types.get_by_metric_and_span_type(self.metric_name, span_type)
        self.assertIsNotNone(mapping, f'Mapping missing for {span_type}')
        assert mapping is not None
        self.assertEqual(mapping.targeting_mode, MetricTargetingMode.SINGLE)

  async def test_sync_creates_missing_span_types_for_metric_targets(self) -> None:
    metric_name = 'metric_with_auto_span_type'
    metric = DummyMetric(name=metric_name, span_types=('auto_created_span_type',))

    async with UnitOfWork(self.db_manager) as uow:
      await _cleanup_test_metric(uow, metric_name)
      await _cleanup_span_types(uow, ['auto_created_span_type'])

    registry = self.MetricRegistry(db_manager=self.db_manager)
    registry.register(metric)
    await registry.sync_with_persistence()

    async with UnitOfWork(self.db_manager) as uow:
      span_type = await uow.span_types.get_by_id('auto_created_span_type')
      mapping = await uow.metric_target_span_types.get_by_metric_and_span_type(metric_name, 'auto_created_span_type')
      await _cleanup_test_metric(uow, metric_name)

    self.assertIsNotNone(span_type)
    self.assertIsNotNone(mapping)

  async def test_sync_persists_group_targeting_mode(self) -> None:
    async with UnitOfWork(self.db_manager) as uow:
      await _ensure_span_types_exist(uow, ['group_span'])

    metric_name = 'group_metric'
    group_metric = GroupMetric(name=metric_name, span_types=('group_span',))

    registry = self.MetricRegistry(db_manager=self.db_manager)
    registry.register(group_metric)
    await registry.sync_with_persistence()

    async with UnitOfWork(self.db_manager) as uow:
      mapping = await uow.metric_target_span_types.get_by_metric_and_span_type(metric_name, 'group_span')
      self.assertIsNotNone(mapping)
      assert mapping is not None
      self.assertEqual(mapping.targeting_mode, MetricTargetingMode.GROUP)

      await _cleanup_test_metric(uow, metric_name)

  async def test_sync_is_idempotent(self) -> None:
    """Test multiple syncs don't create duplicates."""
    registry = self.MetricRegistry(db_manager=self.db_manager)
    registry.register(self.test_metric)

    await registry.sync_with_persistence()
    await registry.sync_with_persistence()
    await registry.sync_with_persistence()

    async with UnitOfWork(self.db_manager) as uow:
      for span_type in self.expected_span_types:
        all_mappings = await uow.metric_target_span_types.list_all()
        matching = [m for m in all_mappings if m.metric == self.metric_name and m.span_type == span_type]
        self.assertEqual(len(matching), 1, f'Duplicate mappings for {span_type}')

  async def test_concurrent_sync_is_idempotent(self) -> None:
    """Test concurrent registry syncs do not race on shared metric metadata."""
    metric_name = f'concurrent_sync_metric_{uuid4().hex[:8]}'
    span_type = f'concurrent_sync_span_type_{uuid4().hex[:8]}'
    metric = DummyMetric(name=metric_name, span_types=(span_type,))

    registry1 = self.MetricRegistry(db_manager=self.db_manager)
    registry2 = self.MetricRegistry(db_manager=self.db_manager)
    registry1.register(metric)
    registry2.register(DummyMetric(name=metric_name, span_types=(span_type,)))

    try:
      await asyncio.gather(registry1.sync_with_persistence(), registry2.sync_with_persistence())

      async with UnitOfWork(self.db_manager) as uow:
        persisted_metric = await uow.metrics.get_by_id(metric_name)
        persisted_span_type = await uow.span_types.get_by_id(span_type)
        all_mappings = await uow.metric_target_span_types.list_all()

      matching_mappings = [
        mapping for mapping in all_mappings if mapping.metric == metric_name and mapping.span_type == span_type
      ]

      self.assertIsNotNone(persisted_metric)
      self.assertIsNotNone(persisted_span_type)
      self.assertEqual(len(matching_mappings), 1)
    finally:
      async with UnitOfWork(self.db_manager) as uow:
        await _cleanup_test_metric(uow, metric_name)
        await _cleanup_span_types(uow, [span_type])

  async def test_sync_updates_description(self) -> None:
    """Test sync updates metric description when changed."""
    async with UnitOfWork(self.db_manager) as uow:
      await _ensure_span_types_exist(uow, ['test_span'])

    metric_with_desc = DummyMetric(
      name='metric_with_desc', description='Initial description', span_types=('test_span',)
    )

    registry1 = self.MetricRegistry(db_manager=self.db_manager)
    registry1.register(metric_with_desc)
    await registry1.sync_with_persistence()

    async with UnitOfWork(self.db_manager) as uow:
      persisted = await uow.metrics.get_by_id('metric_with_desc')
      self.assertEqual(persisted.description, 'Initial description')

    metric_updated = DummyMetric(name='metric_with_desc', description='Updated description', span_types=('test_span',))

    registry2 = self.MetricRegistry(db_manager=self.db_manager)
    registry2.register(metric_updated)
    await registry2.sync_with_persistence()

    async with UnitOfWork(self.db_manager) as uow:
      persisted = await uow.metrics.get_by_id('metric_with_desc')
      self.assertEqual(persisted.description, 'Updated description')

  async def test_sync_skips_description_update_if_none(self) -> None:
    """Test sync doesn't update description if new value is None."""
    async with UnitOfWork(self.db_manager) as uow:
      await _ensure_span_types_exist(uow, ['test_span'])

    metric_with_desc = DummyMetric(name='metric_no_desc', description='Initial description', span_types=('test_span',))

    registry1 = self.MetricRegistry(db_manager=self.db_manager)
    registry1.register(metric_with_desc)
    await registry1.sync_with_persistence()

    metric_no_desc = DummyMetric(name='metric_no_desc', description=None, span_types=('test_span',))

    registry2 = self.MetricRegistry(db_manager=self.db_manager)
    registry2.register(metric_no_desc)
    await registry2.sync_with_persistence()

    async with UnitOfWork(self.db_manager) as uow:
      persisted = await uow.metrics.get_by_id('metric_no_desc')
      self.assertEqual(persisted.description, 'Initial description')

  async def test_sync_adds_new_span_type_mappings(self) -> None:
    """Test sync adds new span type mappings without removing old ones."""
    async with UnitOfWork(self.db_manager) as uow:
      await _ensure_span_types_exist(uow, ['span_type_1', 'span_type_2'])

    metric_v1 = DummyMetric(name='updated_desc_metric', span_types=('span_type_1',))

    registry1 = self.MetricRegistry(db_manager=self.db_manager)
    registry1.register(metric_v1)
    await registry1.sync_with_persistence()

    async with UnitOfWork(self.db_manager) as uow:
      exists = await uow.metric_target_span_types.exists_mapping('updated_desc_metric', 'span_type_1')
      self.assertTrue(exists)

    metric_v2 = DummyMetric(name='updated_desc_metric', span_types=('span_type_1', 'span_type_2'))

    registry2 = self.MetricRegistry(db_manager=self.db_manager)
    registry2.register(metric_v2)
    await registry2.sync_with_persistence()

    async with UnitOfWork(self.db_manager) as uow:
      exists_1 = await uow.metric_target_span_types.exists_mapping('updated_desc_metric', 'span_type_1')
      exists_2 = await uow.metric_target_span_types.exists_mapping('updated_desc_metric', 'span_type_2')
      self.assertTrue(exists_1)
      self.assertTrue(exists_2)

  async def test_sync_removes_span_type_mappings_no_longer_targeted(self) -> None:
    """Test sync removes stale mappings when a metric no longer targets a span type."""
    async with UnitOfWork(self.db_manager) as uow:
      await _ensure_span_types_exist(uow, ['span_type_1', 'span_type_2'])

    metric_v1 = DummyMetric(name='updated_desc_metric', span_types=('span_type_1', 'span_type_2'))

    registry1 = self.MetricRegistry(db_manager=self.db_manager)
    registry1.register(metric_v1)
    await registry1.sync_with_persistence()

    metric_v2 = DummyMetric(name='updated_desc_metric', span_types=('span_type_2',))

    registry2 = self.MetricRegistry(db_manager=self.db_manager)
    registry2.register(metric_v2)
    await registry2.sync_with_persistence()

    async with UnitOfWork(self.db_manager) as uow:
      exists_1 = await uow.metric_target_span_types.exists_mapping('updated_desc_metric', 'span_type_1')
      exists_2 = await uow.metric_target_span_types.exists_mapping('updated_desc_metric', 'span_type_2')
      self.assertFalse(exists_1)
      self.assertTrue(exists_2)

  async def test_sync_stores_metric_ground_truth_key(self) -> None:
    """Test sync stores the ground-truth key declared by a metric implementation."""
    metric_name = 'metric_with_ground_truth_key'
    metric = DummyMetric(name=metric_name, span_types=('test_span',), ground_truth_key='expected_output')

    registry = self.MetricRegistry(db_manager=self.db_manager)
    registry.register(metric)
    await registry.sync_with_persistence()

    async with UnitOfWork(self.db_manager) as uow:
      persisted_metric = await uow.metrics.get_by_id(metric_name)

    self.assertIsNotNone(persisted_metric)
    assert persisted_metric is not None
    self.assertEqual(persisted_metric.ground_truth_keys, ['expected_output'])

    async with UnitOfWork(self.db_manager) as uow:
      await _cleanup_test_metric(uow, metric_name)


class TestMetricRegistryIntegration(unittest.IsolatedAsyncioTestCase):
  """End-to-end integration tests."""

  async def asyncSetUp(self) -> None:
    try:
      self.db_manager = await _setup_database()
    except RuntimeError as error:
      self.skipTest(str(error))

    try:
      from syllo_eval.evaluation.metric_registry import MetricRegistry

      self.MetricRegistry = MetricRegistry
    except ModuleNotFoundError as error:
      self.skipTest(f'MetricRegistry dependencies are missing: {error}')

    self.test_metric = TestMetric()
    self.metric_name = self.test_metric.name
    self.expected_span_types = self.test_metric.target_span_types

    async with UnitOfWork(self.db_manager) as uow:
      await _cleanup_test_metric(uow, self.metric_name)

  async def asyncTearDown(self) -> None:
    if hasattr(self, 'db_manager'):
      async with UnitOfWork(self.db_manager) as uow:
        await _cleanup_test_metric(uow, self.metric_name)
      await self.db_manager.close_async()

  async def test_complete_workflow(self) -> None:
    """Test complete workflow: register -> sync -> retrieve -> compute."""
    registry = self.MetricRegistry(db_manager=self.db_manager)
    registry.register(self.test_metric)

    await registry.sync_with_persistence()

    async with UnitOfWork(self.db_manager) as uow:
      persisted_metric = await uow.metrics.get_by_id(self.metric_name)
      self.assertIsNotNone(persisted_metric)

      for span_type in self.expected_span_types:
        exists = await uow.metric_target_span_types.exists_mapping(self.metric_name, span_type)
        self.assertTrue(exists)

    for index, span_type in enumerate(self.expected_span_types):
      metric = registry.get(self.metric_name)
      self.assertIn(span_type, metric.target_span_types)

      span = Span(
        external_id=f'span_test_{index}',
        trace_id='trace_test',
        parent_span_id=None,
        span_type=span_type,
        name='test_span',
        start_time=datetime.now(tz=timezone.utc),
        end_time=datetime.now(tz=timezone.utc),
        input_data='{}',
        output_data='{}',
      )
      ground_truth = GroundTruth(
        id=uuid4(),
        sample_id=uuid4(),
        key='expected_output',
        ground_truth_value={},
      )

      result = await metric.compute(span, ground_truth)
      score = result.score
      if score is None:
        self.fail('Expected successful metric to return a score.')
      self.assertGreaterEqual(score, 0.0)
      self.assertLessEqual(score, 1.0)


if __name__ == '__main__':
  unittest.main()
