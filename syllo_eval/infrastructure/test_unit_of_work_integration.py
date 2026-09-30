import unittest
from datetime import datetime, timedelta, timezone
from uuid import UUID, uuid4

from syllo_eval.infrastructure import TransactionalUnitOfWork, UnitOfWork
from syllo_eval.model import (
  Agent,
  Dataset,
  EvaluationRun,
  EvaluationRunSample,
  EvaluationSampleStatus,
  EvaluationStatus,
  GroundTruth,
  Metric,
  MetricComputation,
  MetricComputationStatus,
  MetricTargetSpanType,
  MetricTargetingMode,
  Sample,
  Span,
  SpanType,
  Trace,
)
from syllo_eval.testing_database import setup_test_database as _setup_database


class _UnitOfWorkIntegrationBase(unittest.IsolatedAsyncioTestCase):
  async def asyncSetUp(self) -> None:
    try:
      self.db_manager = await _setup_database()
    except RuntimeError as error:
      self.skipTest(str(error))

    self._suffix = uuid4().hex[:10]
    self._created_agents: list[UUID] = []
    self._created_datasets: list[UUID] = []
    self._created_samples: list[UUID] = []
    self._created_ground_truths: list[UUID] = []
    self._created_traces: list[str] = []
    self._created_spans: list[str] = []
    self._created_evaluation_runs: list[UUID] = []
    self._created_evaluation_run_samples: list[UUID] = []
    self._created_metric_computations: list[UUID] = []
    self._created_metrics: set[str] = set()
    self._created_span_types: set[str] = set()
    self._created_metric_target_pairs: set[tuple[str, str]] = set()

  async def asyncTearDown(self) -> None:
    if not hasattr(self, 'db_manager'):
      return

    async with UnitOfWork(self.db_manager) as uow:
      for computation_id in reversed(self._created_metric_computations):
        await uow.metric_computations.delete(computation_id)

      for evaluation_run_sample_id in reversed(self._created_evaluation_run_samples):
        await uow.evaluation_run_samples.delete(evaluation_run_sample_id)

      for span_id in reversed(self._created_spans):
        await uow.spans.delete(span_id)

      for trace_id in reversed(self._created_traces):
        await uow.traces.delete(trace_id)

      for ground_truth_id in reversed(self._created_ground_truths):
        await uow.ground_truths.delete(ground_truth_id)

      for evaluation_run_id in reversed(self._created_evaluation_runs):
        await uow.evaluation_runs.delete(evaluation_run_id)

      for sample_id in reversed(self._created_samples):
        await uow.samples.delete(sample_id)

      for dataset_id in reversed(self._created_datasets):
        await uow.datasets.delete(dataset_id)

      for agent_id in reversed(self._created_agents):
        await uow.agents.delete(agent_id)

      for metric_name, span_type_name in sorted(self._created_metric_target_pairs):
        await uow.metric_target_span_types.delete_by_metric_and_span_type(metric_name, span_type_name)

      for metric_name in sorted(self._created_metrics):
        await uow.metrics.delete(metric_name)

      for span_type_name in sorted(self._created_span_types):
        await uow.span_types.delete(span_type_name)

    await self.db_manager.close_async()

  def _new_name(self, prefix: str) -> str:
    return f'{prefix}_{self._suffix}'


class TestUnitOfWorkIntegration(_UnitOfWorkIntegrationBase):
  async def test_unit_of_work_persists_and_queries_related_entities(self) -> None:
    now = datetime.now(tz=timezone.utc)
    agent_name = self._new_name('agent')
    other_agent_name = self._new_name('other_agent')
    dataset_name = self._new_name('dataset')
    accuracy_metric_name = self._new_name('accuracy')
    relevance_metric_name = self._new_name('relevance')
    failed_metric_name = self._new_name('failed_metric')
    agent_span_type_name = self._new_name('agent_span')
    llm_span_type_name = self._new_name('llm_span')
    retrieval_span_type_name = self._new_name('retrieval_span')
    trace_id = self._new_name('trace')

    agent_v1 = Agent(id=uuid4(), name=agent_name, version_tag='v1.0.0')
    agent_v2 = Agent(id=uuid4(), name=agent_name, version_tag='v2.0.0')
    other_agent = Agent(id=uuid4(), name=other_agent_name, version_tag='v1.0.0')
    dataset = Dataset(id=uuid4(), name=dataset_name)
    samples = [
      Sample(
        id=uuid4(),
        dataset_id=dataset.id,
        input_prompt=f'Question {index}',
        ground_truth_output=f'Answer {index}',
      )
      for index in range(3)
    ]
    accuracy_metric = Metric(name=accuracy_metric_name, description='Accuracy metric for unit of work tests')
    relevance_metric = Metric(name=relevance_metric_name, description='Relevance metric for unit of work tests')
    failed_metric = Metric(name=failed_metric_name, description='Failed metric for unit of work tests')
    agent_span_type = SpanType(name=agent_span_type_name, description='Agent span type for unit of work tests')
    llm_span_type = SpanType(name=llm_span_type_name, description='LLM span type for unit of work tests')
    retrieval_span_type = SpanType(
      name=retrieval_span_type_name,
      description='Retrieval span type for unit of work tests',
    )
    metric_targets = [
      MetricTargetSpanType(id=uuid4(), metric=accuracy_metric_name, span_type=agent_span_type_name),
      MetricTargetSpanType(id=uuid4(), metric=accuracy_metric_name, span_type=llm_span_type_name),
      MetricTargetSpanType(id=uuid4(), metric=relevance_metric_name, span_type=retrieval_span_type_name),
      MetricTargetSpanType(id=uuid4(), metric=failed_metric_name, span_type=llm_span_type_name),
    ]
    ground_truths = [
      GroundTruth(
        id=uuid4(),
        sample_id=samples[0].id,
        key=accuracy_metric_name,
        ground_truth_value={'expected': 'Answer 0'},
      ),
      GroundTruth(
        id=uuid4(),
        sample_id=samples[0].id,
        key=relevance_metric_name,
        ground_truth_value={'expected_keywords': ['Answer', '0']},
      ),
    ]
    evaluation_run = EvaluationRun(
      id=uuid4(),
      agent_id=agent_v1.id,
      dataset_id=dataset.id,
      status=EvaluationStatus.RUNNING,
      start_time=now,
    )
    trace = Trace(external_id=trace_id, start_time=now, end_time=now + timedelta(seconds=3))
    evaluation_run_sample = EvaluationRunSample(
      id=uuid4(),
      evaluation_run_id=evaluation_run.id,
      sample_id=samples[0].id,
      trace_id=trace_id,
      status=EvaluationSampleStatus.RUNNING,
      started_at=now,
      metadata={'batch': 'unit_test'},
    )
    root_span = Span(
      external_id=self._new_name('span_root'),
      trace_id=trace_id,
      parent_span_id=None,
      span_type=agent_span_type_name,
      name='agent_execution',
      start_time=now,
      end_time=now + timedelta(seconds=3),
      input_data=samples[0].input_prompt,
      output_data='Agent response',
      metadata={'role': 'root_agent', 'source': 'unit_test'},
    )
    retrieval_span = Span(
      external_id=self._new_name('span_retrieval'),
      trace_id=trace_id,
      parent_span_id=root_span.external_id,
      span_type=retrieval_span_type_name,
      name='document_retrieval',
      start_time=now + timedelta(milliseconds=500),
      end_time=now + timedelta(milliseconds=900),
      input_data='question',
      output_data='retrieved documents',
    )
    llm_span = Span(
      external_id=self._new_name('span_llm'),
      trace_id=trace_id,
      parent_span_id=root_span.external_id,
      span_type=llm_span_type_name,
      name='llm_call',
      start_time=now + timedelta(seconds=1),
      end_time=now + timedelta(seconds=2),
      input_data='prompt',
      output_data='model output',
      metadata={'provider': 'openai', 'token_usage': {'prompt_tokens': 12, 'completion_tokens': 8}},
    )
    computations = [
      MetricComputation(
        id=uuid4(),
        evaluation_run_sample_id=evaluation_run_sample.id,
        metric=accuracy_metric_name,
        targeting_mode=MetricTargetingMode.SINGLE,
        target_span_type=agent_span_type_name,
        span_ids=[root_span.external_id],
        score=0.88,
        reasoning='Agent span is mostly correct.',
        metadata={'source': 'unit_test'},
      ),
      MetricComputation(
        id=uuid4(),
        evaluation_run_sample_id=evaluation_run_sample.id,
        metric=accuracy_metric_name,
        targeting_mode=MetricTargetingMode.SINGLE,
        target_span_type=llm_span_type_name,
        span_ids=[llm_span.external_id],
        score=0.91,
        reasoning='LLM span is correct.',
        metadata={'source': 'unit_test'},
      ),
      MetricComputation(
        id=uuid4(),
        evaluation_run_sample_id=evaluation_run_sample.id,
        metric=relevance_metric_name,
        targeting_mode=MetricTargetingMode.SINGLE,
        target_span_type=retrieval_span_type_name,
        span_ids=[retrieval_span.external_id],
        score=0.84,
        reasoning='Retrieved context is relevant.',
        metadata={'source': 'unit_test'},
      ),
      MetricComputation(
        id=uuid4(),
        evaluation_run_sample_id=evaluation_run_sample.id,
        metric=failed_metric_name,
        targeting_mode=MetricTargetingMode.SINGLE,
        target_span_type=llm_span_type_name,
        span_ids=[llm_span.external_id],
        score=None,
        status=MetricComputationStatus.FAILED,
        error_message='judge service unavailable',
        raw_output={'provider': 'openai'},
      ),
      MetricComputation(
        id=uuid4(),
        evaluation_run_sample_id=evaluation_run_sample.id,
        metric=failed_metric_name,
        targeting_mode=MetricTargetingMode.SINGLE,
        target_span_type=agent_span_type_name,
        span_ids=[root_span.external_id],
        score=None,
        status=MetricComputationStatus.SKIPPED,
        error_message='Missing required ground truth.',
      ),
    ]

    self._created_agents.extend([agent_v1.id, agent_v2.id, other_agent.id])
    self._created_datasets.append(dataset.id)
    self._created_samples.extend(sample.id for sample in samples)
    self._created_metrics.update([accuracy_metric_name, relevance_metric_name, failed_metric_name])
    self._created_span_types.update([agent_span_type_name, llm_span_type_name, retrieval_span_type_name])
    self._created_metric_target_pairs.update((item.metric, item.span_type) for item in metric_targets)
    self._created_ground_truths.extend(ground_truth.id for ground_truth in ground_truths)
    self._created_evaluation_runs.append(evaluation_run.id)
    self._created_traces.append(trace_id)
    self._created_evaluation_run_samples.append(evaluation_run_sample.id)
    self._created_spans.extend([root_span.external_id, retrieval_span.external_id, llm_span.external_id])
    self._created_metric_computations.extend(computation.id for computation in computations)

    async with UnitOfWork(self.db_manager) as uow:
      await uow.agents.create(agent_v1)
      await uow.agents.create(agent_v2)
      await uow.agents.create(other_agent)
      await uow.datasets.create(dataset)
      await uow.samples.bulk_create(samples)
      await uow.metrics.create(accuracy_metric)
      await uow.metrics.create(relevance_metric)
      await uow.metrics.create(failed_metric)
      await uow.span_types.create(agent_span_type)
      await uow.span_types.create(llm_span_type)
      await uow.span_types.create(retrieval_span_type)
      for metric_target in metric_targets:
        await uow.metric_target_span_types.create(metric_target)
      await uow.ground_truths.bulk_create(ground_truths)
      await uow.evaluation_runs.create(evaluation_run)
      await uow.traces.create(trace)
      await uow.evaluation_run_samples.create(evaluation_run_sample)
      await uow.spans.bulk_create([root_span, retrieval_span, llm_span])
      await uow.metric_computations.bulk_create(computations)

    async with UnitOfWork(self.db_manager) as uow:
      retrieved_agent = await uow.agents.get_by_id(agent_v1.id)
      self.assertEqual(retrieved_agent, agent_v1)

      retrieved_by_name_and_version = await uow.agents.get_by_name_and_version(agent_name, 'v1.0.0')
      self.assertEqual(retrieved_by_name_and_version, agent_v1)

      versions = await uow.agents.list_by_name(agent_name)
      self.assertEqual({agent.id for agent in versions}, {agent_v1.id, agent_v2.id})

      distinct_names = await uow.agents.list_distinct_names()
      self.assertIn(agent_name, distinct_names)
      self.assertIn(other_agent_name, distinct_names)

      retrieved_dataset = await uow.datasets.get_by_name(dataset_name)
      self.assertEqual(retrieved_dataset, dataset)

      dataset_samples = await uow.samples.list_by_dataset(dataset.id)
      self.assertEqual({sample.id for sample in dataset_samples}, {sample.id for sample in samples})

      first_page = await uow.samples.list_by_dataset(dataset.id, limit=2, offset=0)
      second_page = await uow.samples.list_by_dataset(dataset.id, limit=2, offset=2)
      self.assertEqual(len(first_page), 2)
      self.assertEqual(len(second_page), 1)
      self.assertEqual(await uow.samples.count_by_dataset(dataset.id), 3)

      retrieved_ground_truth = await uow.ground_truths.get_by_sample_and_key(samples[0].id, accuracy_metric_name)
      self.assertIsNotNone(retrieved_ground_truth)
      self.assertEqual(retrieved_ground_truth.ground_truth_value, {'expected': 'Answer 0'})

      sample_ground_truths = await uow.ground_truths.list_by_sample(samples[0].id)
      self.assertEqual(
        {ground_truth.key for ground_truth in sample_ground_truths},
        {
          accuracy_metric_name,
          relevance_metric_name,
        },
      )

      key_ground_truths = await uow.ground_truths.list_by_key(accuracy_metric_name)
      self.assertEqual([ground_truth.id for ground_truth in key_ground_truths], [ground_truths[0].id])

      metrics_for_llm_span = await uow.metric_target_span_types.get_metrics_for_span_type(llm_span_type_name)
      self.assertEqual(metrics_for_llm_span, [accuracy_metric_name, failed_metric_name])

      span_types_for_accuracy = await uow.metric_target_span_types.get_span_types_for_metric(accuracy_metric_name)
      self.assertEqual(set(span_types_for_accuracy), {agent_span_type_name, llm_span_type_name})
      self.assertTrue(await uow.metric_target_span_types.exists_mapping(accuracy_metric_name, llm_span_type_name))
      self.assertFalse(await uow.metric_target_span_types.exists_mapping(relevance_metric_name, llm_span_type_name))

      run_samples = await uow.evaluation_run_samples.list_by_evaluation_run(evaluation_run.id)
      self.assertEqual([run_sample.id for run_sample in run_samples], [evaluation_run_sample.id])

      retrieved_run_sample = await uow.evaluation_run_samples.get_by_run_and_sample(
        evaluation_run.id,
        samples[0].id,
      )
      self.assertEqual(retrieved_run_sample, evaluation_run_sample)
      self.assertIsNotNone(retrieved_run_sample)
      self.assertEqual(retrieved_run_sample.status, EvaluationSampleStatus.RUNNING)
      self.assertEqual(retrieved_run_sample.metadata, {'batch': 'unit_test'})

      completed_run_sample = await uow.evaluation_run_samples.update_status(
        run_sample_id=evaluation_run_sample.id,
        status=EvaluationSampleStatus.COMPLETED,
        ended_at=now + timedelta(minutes=1),
      )
      self.assertEqual(completed_run_sample.status, EvaluationSampleStatus.COMPLETED)
      self.assertEqual(completed_run_sample.ended_at, now + timedelta(minutes=1))

      trace_spans = await uow.spans.list_by_trace(trace_id)
      self.assertEqual(
        [span.external_id for span in trace_spans],
        [
          root_span.external_id,
          retrieval_span.external_id,
          llm_span.external_id,
        ],
      )
      self.assertEqual(trace_spans[0].metadata, {'role': 'root_agent', 'source': 'unit_test'})
      self.assertIsNone(trace_spans[1].metadata)
      self.assertEqual(
        trace_spans[2].metadata,
        {
          'provider': 'openai',
          'token_usage': {'prompt_tokens': 12, 'completion_tokens': 8},
        },
      )

      llm_spans = await uow.spans.list_by_span_type(trace_id, llm_span_type_name)
      self.assertEqual(llm_spans, [llm_span])

      root_spans = await uow.spans.get_root_spans(trace_id)
      self.assertEqual(root_spans, [root_span])

      child_spans = await uow.spans.get_child_spans(root_span.external_id)
      self.assertEqual([span.external_id for span in child_spans], [retrieval_span.external_id, llm_span.external_id])
      self.assertEqual(await uow.spans.count_by_trace(trace_id), 3)

      run_computations = await uow.metric_computations.list_by_evaluation_run_sample(evaluation_run_sample.id)
      self.assertEqual(len(run_computations), 5)
      self.assertEqual({computation.id for computation in run_computations}, {item.id for item in computations})
      run_level_computations = await uow.metric_computations.list_by_evaluation_run(evaluation_run.id)
      self.assertEqual({computation.id for computation in run_level_computations}, {item.id for item in computations})
      failed_computations = [
        computation for computation in run_computations if computation.status == MetricComputationStatus.FAILED
      ]
      skipped_computations = [
        computation for computation in run_computations if computation.status == MetricComputationStatus.SKIPPED
      ]
      self.assertEqual(len(failed_computations), 1)
      self.assertIsNone(failed_computations[0].score)
      self.assertEqual(failed_computations[0].error_message, 'judge service unavailable')
      self.assertEqual(failed_computations[0].raw_output, {'provider': 'openai'})
      self.assertEqual(len(skipped_computations), 1)
      self.assertEqual(skipped_computations[0].error_message, 'Missing required ground truth.')

      llm_computations = await uow.metric_computations.list_by_span(llm_span.external_id)
      self.assertEqual({computation.id for computation in llm_computations}, {computations[1].id, computations[3].id})

      aggregated_scores = await uow.metric_computations.get_aggregated_scores_by_evaluation_run(evaluation_run.id)
      aggregated_scores_by_metric = {item['metric']: item for item in aggregated_scores}
      self.assertNotIn(failed_metric_name, aggregated_scores_by_metric)
      self.assertEqual(aggregated_scores_by_metric[accuracy_metric_name]['count'], 2)
      self.assertAlmostEqual(float(aggregated_scores_by_metric[accuracy_metric_name]['avg_score']), 0.895)
      self.assertEqual(aggregated_scores_by_metric[relevance_metric_name]['count'], 1)
      self.assertAlmostEqual(float(aggregated_scores_by_metric[relevance_metric_name]['avg_score']), 0.84)

      accuracy_scores = await uow.metric_computations.get_scores_by_metric_and_evaluation_run(
        evaluation_run.id,
        accuracy_metric_name,
      )
      self.assertEqual(accuracy_scores, [0.91, 0.88])
      self.assertEqual(await uow.metric_computations.count_by_evaluation_run(evaluation_run.id), 5)

      updated_run = await uow.evaluation_runs.update_status(
        evaluation_run.id,
        EvaluationStatus.COMPLETED,
        now + timedelta(minutes=5),
      )
      self.assertEqual(updated_run.status, EvaluationStatus.COMPLETED)
      self.assertEqual(updated_run.end_time, now + timedelta(minutes=5))

      agent_runs = await uow.evaluation_runs.list_by_agent(agent_v1.id)
      self.assertEqual([run.id for run in agent_runs], [evaluation_run.id])

      dataset_runs = await uow.evaluation_runs.list_by_dataset(dataset.id)
      self.assertEqual([run.id for run in dataset_runs], [evaluation_run.id])

      completed_runs = await uow.evaluation_runs.list_by_status(EvaluationStatus.COMPLETED)
      self.assertIn(evaluation_run.id, [run.id for run in completed_runs])

      running_runs = await uow.evaluation_runs.get_running_runs()
      self.assertNotIn(evaluation_run.id, [run.id for run in running_runs])
      self.assertGreaterEqual(await uow.evaluation_runs.count_by_status(EvaluationStatus.COMPLETED), 1)


class TestTransactionalUnitOfWorkIntegration(_UnitOfWorkIntegrationBase):
  async def test_transactional_unit_of_work_commits_related_changes(self) -> None:
    agent = Agent(id=uuid4(), name=self._new_name('transactional_agent'), version_tag='v1.0.0')
    dataset = Dataset(id=uuid4(), name=self._new_name('transactional_dataset'))
    evaluation_run = EvaluationRun(
      id=uuid4(),
      agent_id=agent.id,
      dataset_id=dataset.id,
      status=EvaluationStatus.RUNNING,
      start_time=datetime.now(tz=timezone.utc),
    )

    self._created_agents.append(agent.id)
    self._created_datasets.append(dataset.id)
    self._created_evaluation_runs.append(evaluation_run.id)

    async with TransactionalUnitOfWork(self.db_manager) as uow:
      await uow.agents.create(agent)
      await uow.datasets.create(dataset)
      await uow.evaluation_runs.create(evaluation_run)

    async with UnitOfWork(self.db_manager) as uow:
      self.assertEqual(await uow.agents.get_by_id(agent.id), agent)
      self.assertEqual(await uow.datasets.get_by_id(dataset.id), dataset)
      self.assertEqual(await uow.evaluation_runs.get_by_id(evaluation_run.id), evaluation_run)

  async def test_transactional_unit_of_work_rolls_back_related_changes(self) -> None:
    agent = Agent(id=uuid4(), name=self._new_name('rollback_agent'), version_tag='v1.0.0')
    dataset = Dataset(id=uuid4(), name=self._new_name('rollback_dataset'))

    self._created_agents.append(agent.id)
    self._created_datasets.append(dataset.id)

    with self.assertRaisesRegex(ValueError, 'rollback'):
      async with TransactionalUnitOfWork(self.db_manager) as uow:
        await uow.agents.create(agent)
        await uow.datasets.create(dataset)
        raise ValueError('rollback')

    async with UnitOfWork(self.db_manager) as uow:
      self.assertIsNone(await uow.agents.get_by_id(agent.id))
      self.assertIsNone(await uow.datasets.get_by_id(dataset.id))


if __name__ == '__main__':
  unittest.main()
