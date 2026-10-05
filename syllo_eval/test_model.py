import unittest
from datetime import datetime, timezone
from uuid import uuid4

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


class TestModel(unittest.TestCase):
  def test_evaluation_sample_status_only_contains_used_states(self) -> None:
    self.assertEqual(
      [status.value for status in EvaluationSampleStatus],
      ['RUNNING', 'COMPLETED', 'FAILED'],
    )

  def test_metric_computation_status_only_contains_terminal_states(self) -> None:
    self.assertEqual(
      [status.value for status in MetricComputationStatus],
      ['COMPLETED', 'FAILED', 'SKIPPED'],
    )

  def test_domain_models_can_be_instantiated_and_related(self) -> None:
    now = datetime.now(tz=timezone.utc)

    agent = Agent(id=uuid4(), name='Test Agent', version_tag='v1')
    dataset = Dataset(id=uuid4(), name='Test Dataset')
    sample = Sample(
      id=uuid4(),
      dataset_id=dataset.id,
      input_prompt='Hello',
      ground_truth_output='World',
    )
    metric = Metric(name='Accuracy', description=None)
    ground_truth = GroundTruth(
      id=uuid4(),
      sample_id=sample.id,
      key='expected_output',
      ground_truth_value={'expected': 'World'},
    )
    evaluation_run = EvaluationRun(
      id=uuid4(),
      agent_id=agent.id,
      dataset_id=dataset.id,
      status=EvaluationStatus.COMPLETED,
      start_time=now,
      end_time=now,
    )
    evaluation_run_sample = EvaluationRunSample(
      id=uuid4(),
      evaluation_run_id=evaluation_run.id,
      sample_id=sample.id,
      trace_id='trace_001',
      status=EvaluationSampleStatus.COMPLETED,
      started_at=now,
      ended_at=now,
      metadata={'source': 'unit_test'},
    )
    trace = Trace(external_id='trace_001', start_time=now, end_time=now)
    span_type = SpanType(name='LLM', description='LLM call')
    span = Span(
      external_id='span_001',
      trace_id=trace.external_id,
      parent_span_id=None,
      span_type=span_type.name,
      name='Generate',
      start_time=now,
      end_time=now,
      input_data='Hello',
      output_data='World',
      metadata={'model': 'gpt-4o-mini', 'token_usage': {'prompt': 12, 'completion': 6}},
    )
    metric_target_span_type = MetricTargetSpanType(
      id=uuid4(),
      metric=metric.name,
      span_type=span_type.name,
      targeting_mode=MetricTargetingMode.SINGLE,
    )
    span_metric_computation = MetricComputation(
      id=uuid4(),
      evaluation_run_sample_id=evaluation_run_sample.id,
      metric=metric.name,
      targeting_mode=MetricTargetingMode.SINGLE,
      target_span_type=span_type.name,
      span_ids=[span.external_id],
      score=0.99,
      status=MetricComputationStatus.COMPLETED,
      reasoning='Matches expected output',
      metadata={'model': 'test'},
      raw_output={'score': 0.99},
    )

    self.assertEqual(sample.dataset_id, dataset.id)
    self.assertEqual(ground_truth.sample_id, sample.id)
    self.assertEqual(ground_truth.key, 'expected_output')
    self.assertEqual(evaluation_run.agent_id, agent.id)
    self.assertEqual(evaluation_run.dataset_id, dataset.id)
    self.assertEqual(evaluation_run_sample.evaluation_run_id, evaluation_run.id)
    self.assertEqual(trace.external_id, span.trace_id)
    self.assertEqual(metric_target_span_type.span_type, span_type.name)
    self.assertEqual(metric_target_span_type.targeting_mode, MetricTargetingMode.SINGLE)
    self.assertEqual(metric.ground_truth_keys, [])
    self.assertIsNotNone(span.metadata)
    self.assertEqual(span.metadata, {'model': 'gpt-4o-mini', 'token_usage': {'prompt': 12, 'completion': 6}})
    self.assertEqual(span_metric_computation.span_ids, [span.external_id])
    self.assertEqual(span_metric_computation.metric, metric.name)
    self.assertEqual(evaluation_run_sample.status, EvaluationSampleStatus.COMPLETED)
    self.assertEqual(evaluation_run_sample.metadata, {'source': 'unit_test'})
    self.assertEqual(span_metric_computation.status, MetricComputationStatus.COMPLETED)
    self.assertEqual(span_metric_computation.raw_output, {'score': 0.99})

  def test_failed_sample_and_metric_can_be_represented_without_scores_or_traces(self) -> None:
    now = datetime.now(tz=timezone.utc)
    failed_sample = EvaluationRunSample(
      id=uuid4(),
      evaluation_run_id=uuid4(),
      sample_id=uuid4(),
      trace_id=None,
      status=EvaluationSampleStatus.FAILED,
      started_at=now,
      ended_at=now,
      error_message='Trace phase timed out after 1.0s',
    )
    failed_computation = MetricComputation(
      id=uuid4(),
      evaluation_run_sample_id=failed_sample.id,
      metric='AnswerCorrectness',
      targeting_mode=MetricTargetingMode.SINGLE,
      target_span_type='llm',
      span_ids=['span_001'],
      score=None,
      status=MetricComputationStatus.FAILED,
      error_message='judge service unavailable',
      raw_output={'provider': 'openai'},
    )

    self.assertIsNone(failed_sample.trace_id)
    self.assertEqual(failed_sample.status, EvaluationSampleStatus.FAILED)
    self.assertIsNone(failed_computation.score)
    self.assertEqual(failed_computation.status, MetricComputationStatus.FAILED)


if __name__ == '__main__':
  unittest.main()
