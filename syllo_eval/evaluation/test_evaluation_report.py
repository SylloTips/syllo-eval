from syllo_eval.trace_semantics import SpanSemantics, LlmUsage
import unittest
from unittest.mock import AsyncMock, patch
from typing import Any, cast
from syllo_eval.evaluation.metrics.test_support import span
from datetime import datetime, timedelta, timezone
from uuid import uuid4

from syllo_eval.evaluation.evaluation_report import (
  EvaluationReportAggregator,
  _compose_report,
  _score_stats,
  _build_token_usage_section,
)
from syllo_eval.model import (
  Agent,
  Dataset,
  EvaluationRun,
  EvaluationRunSample,
  EvaluationSampleStatus,
  EvaluationStatus,
  Metric,
  MetricComputation,
  MetricComputationStatus,
  MetricTargetingMode,
  Span,
)


class EvaluationReportCompositionTest(unittest.TestCase):
  def test_usage_on_custom_spans_is_deduplicated_per_call_and_trace(self):
    first = span(usage=LlmUsage(call_id='call', input_tokens=10, output_tokens=2, total_tokens=12))
    first.span_type = 'bla'
    duplicate = first.model_copy(update={'external_id': 'another-span', 'span_type': 'agent_root'})
    empty = first.model_copy(deep=True)
    empty.external_id = 'empty'
    empty.semantics.usage = LlmUsage(call_id='call')
    another_trace = first.model_copy(update={'trace_id': 'another-trace'})
    legacy = span()
    legacy.external_id = 'legacy'
    legacy.metadata = {'token_usage': {'input_tokens': 3, 'output_tokens': 1, 'total_tokens': 4}}
    missing = span()
    missing.external_id = 'missing'
    missing.span_type = 'llm'
    result = _build_token_usage_section([empty, first, duplicate, another_trace, legacy, missing], []).agent
    self.assertEqual(result.input_tokens, 23)
    self.assertEqual(result.output_tokens, 5)
    self.assertEqual(result.total_tokens, 28)
    self.assertEqual(result.sources_with_data, 3)
    self.assertEqual(result.sources_total, 4)

  def test_compose_report_aggregates_samples_metrics_failures_and_latency(self) -> None:
    generated_at = datetime(2026, 5, 22, 12, 0, tzinfo=timezone.utc)
    run_started_at = generated_at - timedelta(minutes=5)
    run = EvaluationRun(
      id=uuid4(),
      agent_id=uuid4(),
      dataset_id=uuid4(),
      status=EvaluationStatus.PARTIALLY_COMPLETED,
      start_time=run_started_at,
      end_time=generated_at,
    )
    agent = Agent(id=run.agent_id, name='demo-agent', version_tag='v-test')
    dataset = Dataset(id=run.dataset_id, name='rag_eval_v1')
    completed_sample = EvaluationRunSample(
      id=uuid4(),
      evaluation_run_id=run.id,
      sample_id=uuid4(),
      trace_id='trace-completed',
      status=EvaluationSampleStatus.COMPLETED,
      started_at=generated_at - timedelta(seconds=20),
      ended_at=generated_at - timedelta(seconds=10),
    )
    metric_failed_sample = EvaluationRunSample(
      id=uuid4(),
      evaluation_run_id=run.id,
      sample_id=uuid4(),
      trace_id='trace-metric-failed',
      status=EvaluationSampleStatus.FAILED,
      started_at=generated_at - timedelta(seconds=30),
      ended_at=generated_at - timedelta(seconds=5),
      error_message='Compute phase timed out after 10s',
      metadata={'failure_phase': 'metric_compute'},
    )
    trace_failed_sample = EvaluationRunSample(
      id=uuid4(),
      evaluation_run_id=run.id,
      sample_id=uuid4(),
      status=EvaluationSampleStatus.FAILED,
      started_at=generated_at - timedelta(seconds=25),
      ended_at=generated_at - timedelta(seconds=15),
      error_message='Phoenix trace missing',
      metadata={'failure_phase': 'trace_fetch'},
    )
    computations = [
      MetricComputation(
        id=uuid4(),
        evaluation_run_sample_id=completed_sample.id,
        metric='answer_correctness_judge',
        targeting_mode=MetricTargetingMode.SINGLE,
        target_span_type='agent',
        span_ids=['agent-span-final'],
        score=0.8,
      ),
      MetricComputation(
        id=uuid4(),
        evaluation_run_sample_id=completed_sample.id,
        metric='answer_correctness_judge',
        targeting_mode=MetricTargetingMode.SINGLE,
        target_span_type='agent',
        span_ids=['agent-span-retry'],
        score=0.6,
      ),
      MetricComputation(
        id=uuid4(),
        evaluation_run_sample_id=completed_sample.id,
        metric='plan_efficiency',
        targeting_mode=MetricTargetingMode.SINGLE,
        target_span_type='agent',
        span_ids=['agent-span-final'],
        score=None,
        status=MetricComputationStatus.SKIPPED,
        error_message='Missing required ground truth.',
      ),
      MetricComputation(
        id=uuid4(),
        evaluation_run_sample_id=metric_failed_sample.id,
        metric='contextual_recall_document_judge',
        targeting_mode=MetricTargetingMode.SINGLE,
        target_span_type='agent',
        span_ids=['agent-span-failed'],
        score=None,
        status=MetricComputationStatus.FAILED,
        error_message='judge unavailable',
      ),
    ]
    metrics = [
      Metric(
        name='answer_correctness_judge',
        description='Answer judge',
        requires_ground_truth=True,
      ),
      Metric(
        name='contextual_recall_document_judge',
        description='Recall judge',
        requires_ground_truth=True,
      ),
      Metric(name='plan_efficiency', description='Plan efficiency', requires_ground_truth=True),
    ]

    report = _compose_report(
      run=run,
      agent=agent,
      dataset=dataset,
      total_samples=5,
      run_samples=[completed_sample, metric_failed_sample, trace_failed_sample],
      computations=computations,
      metrics=metrics,
      generated_at=generated_at,
    )

    self.assertEqual(report.report_version, '1.3')
    self.assertEqual(report.generated_at, generated_at)
    self.assertEqual(report.run.samples_processed, 3)
    self.assertEqual(report.run.duration_seconds, 300.0)
    self.assertEqual(report.dataset.total_samples, 5)
    self.assertEqual(
      report.observed_metrics,
      ['answer_correctness_judge', 'contextual_recall_document_judge', 'plan_efficiency'],
    )
    self.assertEqual(report.summary.samples.total, 5)
    self.assertEqual(report.summary.samples.succeeded, 1)
    self.assertEqual(report.summary.samples.failed, 2)
    self.assertAlmostEqual(report.summary.samples.success_rate, 1 / 5)
    self.assertEqual(report.summary.latency_seconds.mean, 10.0)
    self.assertEqual(report.summary.latency_seconds.p95, 10.0)
    self.assertEqual(report.summary.latency_seconds.total_wall_time, 300.0)
    self.assertEqual(report.summary.metric_computations.total, 4)
    self.assertEqual(report.summary.metric_computations.completed, 2)
    self.assertEqual(report.summary.metric_computations.failed, 1)
    self.assertEqual(report.summary.metric_computations.skipped, 1)

    metric_by_name = {metric.name: metric for metric in report.metrics}
    answer_metric = metric_by_name['answer_correctness_judge']
    self.assertEqual(answer_metric.coverage.samples_total, 5)
    self.assertEqual(answer_metric.coverage.computations_total, 2)
    self.assertEqual(answer_metric.coverage.completed, 2)
    self.assertAlmostEqual(answer_metric.coverage.coverage_rate, 1 / 5)
    self.assertEqual(answer_metric.scores.count, 2)
    self.assertIsNotNone(answer_metric.scores.mean)
    self.assertAlmostEqual(answer_metric.scores.mean or 0, 0.7)
    self.assertAlmostEqual(answer_metric.scores.stddev or 0, 0.14142135623730953)
    self.assertEqual(answer_metric.scores.min, 0.6)
    self.assertEqual(answer_metric.scores.max, 0.8)

    completed_sample_report = report.samples[0]
    self.assertEqual(completed_sample_report.evaluation_run_sample_id, completed_sample.id)
    sample_answer_metric = completed_sample_report.metrics['answer_correctness_judge']
    self.assertAlmostEqual(sample_answer_metric.value or 0, 0.7)
    self.assertEqual(sample_answer_metric.computations_total, 2)
    self.assertEqual(sample_answer_metric.completed, 2)
    self.assertEqual(completed_sample_report.metrics['plan_efficiency'].skipped, 1)

    self.assertEqual(report.failures.samples.by_phase['metric_compute'], 1)
    self.assertEqual(report.failures.samples.by_phase['trace_fetch'], 1)
    self.assertEqual(report.failures.metric_computations.by_metric['contextual_recall_document_judge'].failed, 1)
    self.assertEqual(
      report.failures.metric_computations.by_metric['contextual_recall_document_judge'].failed_reasons,
      {'judge unavailable': 1},
    )
    self.assertEqual(
      report.failures.metric_computations.by_metric['contextual_recall_document_judge'].skipped_reasons,
      {},
    )
    self.assertEqual(
      report.failures.metric_computations.by_metric['plan_efficiency'].skipped_reasons,
      {'Missing required ground truth.': 1},
    )
    self.assertEqual(report.failures.metric_computations.failed_reasons, {'judge unavailable': 1})
    self.assertEqual(report.failures.metric_computations.skipped_reasons, {'Missing required ground truth.': 1})

    plan_metric = metric_by_name['plan_efficiency']
    self.assertEqual(plan_metric.coverage.skipped, 1)
    self.assertEqual(plan_metric.coverage.skipped_reasons, {'Missing required ground truth.': 1})
    self.assertEqual(answer_metric.coverage.skipped_reasons, {})

  def test_compose_report_aggregates_token_usage_from_spans_and_judge_metadata(self) -> None:
    generated_at = datetime(2026, 5, 22, 12, 0, tzinfo=timezone.utc)
    run = EvaluationRun(
      id=uuid4(),
      agent_id=uuid4(),
      dataset_id=uuid4(),
      status=EvaluationStatus.COMPLETED,
      start_time=generated_at,
      end_time=generated_at,
    )
    sample_with_tokens = EvaluationRunSample(
      id=uuid4(),
      evaluation_run_id=run.id,
      sample_id=uuid4(),
      trace_id='trace-tokens',
      status=EvaluationSampleStatus.COMPLETED,
    )
    sample_without_tokens = EvaluationRunSample(
      id=uuid4(),
      evaluation_run_id=run.id,
      sample_id=uuid4(),
      trace_id='trace-no-tokens',
      status=EvaluationSampleStatus.COMPLETED,
    )
    spans = [
      Span(
        external_id='span-tokens-1',
        trace_id='trace-tokens',
        span_type='llm',
        name='llm-call-1',
        start_time=generated_at,
        end_time=generated_at,
        input_data='',
        output_data='',
        semantics=SpanSemantics(usage=LlmUsage(input_tokens=10, output_tokens=4, total_tokens=14)),
      ),
      Span(
        external_id='span-tokens-2',
        trace_id='trace-tokens',
        span_type='llm',
        name='llm-call-2',
        start_time=generated_at,
        end_time=generated_at,
        input_data='',
        output_data='',
        semantics=SpanSemantics(usage=LlmUsage(input_tokens=6, output_tokens=2, total_tokens=8)),
      ),
      Span(
        external_id='span-no-tokens',
        trace_id='trace-no-tokens',
        span_type='llm',
        name='llm-call-3',
        start_time=generated_at,
        end_time=generated_at,
        input_data='',
        output_data='',
        metadata=None,
      ),
    ]
    judge_computation = MetricComputation(
      id=uuid4(),
      evaluation_run_sample_id=sample_with_tokens.id,
      metric='answer_correctness_judge',
      targeting_mode=MetricTargetingMode.SINGLE,
      target_span_type='agent',
      span_ids=['agent-span'],
      score=0.9,
      metadata={'judge_usage': {'input_tokens': 20, 'output_tokens': 5, 'total_tokens': 25}},
    )
    deterministic_computation = MetricComputation(
      id=uuid4(),
      evaluation_run_sample_id=sample_without_tokens.id,
      metric='plan_efficiency',
      targeting_mode=MetricTargetingMode.SINGLE,
      target_span_type='agent',
      span_ids=['agent-span'],
      score=0.5,
    )

    report = _compose_report(
      run=run,
      agent=Agent(id=run.agent_id, name='demo-agent', version_tag='v-test'),
      dataset=Dataset(id=run.dataset_id, name='dataset'),
      total_samples=2,
      run_samples=[sample_with_tokens, sample_without_tokens],
      computations=[judge_computation, deterministic_computation],
      metrics=[
        Metric(name='answer_correctness_judge', description=None, requires_ground_truth=True),
        Metric(name='plan_efficiency', description=None, requires_ground_truth=True),
      ],
      usage_spans=spans,
      generated_at=generated_at,
    )

    run_tokens = report.summary.token_usage
    self.assertEqual(run_tokens.agent.input_tokens, 16)
    self.assertEqual(run_tokens.agent.output_tokens, 6)
    self.assertEqual(run_tokens.agent.total_tokens, 22)
    self.assertEqual(run_tokens.agent.sources_with_data, 2)
    self.assertEqual(run_tokens.agent.sources_total, 3)
    self.assertEqual(run_tokens.judge.input_tokens, 20)
    self.assertEqual(run_tokens.judge.output_tokens, 5)
    self.assertEqual(run_tokens.judge.total_tokens, 25)
    self.assertEqual(run_tokens.judge.sources_with_data, 1)
    self.assertEqual(run_tokens.judge.sources_total, 2)

    sample_reports = {report_sample.evaluation_run_sample_id: report_sample for report_sample in report.samples}
    with_tokens = sample_reports[sample_with_tokens.id].token_usage
    self.assertEqual(with_tokens.agent.total_tokens, 22)
    self.assertEqual(with_tokens.agent.sources_with_data, 2)
    self.assertEqual(with_tokens.agent.sources_total, 2)
    self.assertEqual(with_tokens.judge.total_tokens, 25)
    self.assertEqual(with_tokens.judge.sources_with_data, 1)
    self.assertEqual(with_tokens.judge.sources_total, 1)

    without_tokens = sample_reports[sample_without_tokens.id].token_usage
    self.assertEqual(without_tokens.agent.total_tokens, 0)
    self.assertEqual(without_tokens.agent.sources_with_data, 0)
    self.assertEqual(without_tokens.agent.sources_total, 1)
    self.assertEqual(without_tokens.judge.total_tokens, 0)
    self.assertEqual(without_tokens.judge.sources_with_data, 0)
    self.assertEqual(without_tokens.judge.sources_total, 1)

  def test_compose_report_uses_unknown_phase_for_legacy_failed_samples(self) -> None:
    generated_at = datetime(2026, 5, 22, 12, 0, tzinfo=timezone.utc)
    run = EvaluationRun(
      id=uuid4(),
      agent_id=uuid4(),
      dataset_id=uuid4(),
      status=EvaluationStatus.FAILED,
      start_time=generated_at,
      end_time=generated_at,
    )
    failed_sample = EvaluationRunSample(
      id=uuid4(),
      evaluation_run_id=run.id,
      sample_id=uuid4(),
      status=EvaluationSampleStatus.FAILED,
      error_message='legacy failure',
    )

    report = _compose_report(
      run=run,
      agent=Agent(id=run.agent_id, name='demo-agent', version_tag='v-test'),
      dataset=Dataset(id=run.dataset_id, name='dataset'),
      total_samples=1,
      run_samples=[failed_sample],
      computations=[],
      metrics=[],
      generated_at=generated_at,
    )

    self.assertEqual(report.failures.samples.by_phase['unknown'], 1)
    self.assertEqual(report.failures.samples.items[0].phase, 'unknown')

  def test_compose_report_falls_back_to_reasoning_when_skip_has_no_error_message(self) -> None:
    generated_at = datetime(2026, 5, 22, 12, 0, tzinfo=timezone.utc)
    run = EvaluationRun(
      id=uuid4(),
      agent_id=uuid4(),
      dataset_id=uuid4(),
      status=EvaluationStatus.COMPLETED,
      start_time=generated_at,
      end_time=generated_at,
    )
    sample = EvaluationRunSample(
      id=uuid4(),
      evaluation_run_id=run.id,
      sample_id=uuid4(),
      status=EvaluationSampleStatus.COMPLETED,
    )
    legacy_skip = MetricComputation(
      id=uuid4(),
      evaluation_run_sample_id=sample.id,
      metric='contextual_precision_document_judge',
      targeting_mode=MetricTargetingMode.SINGLE,
      target_span_type='agent',
      span_ids=['agent-span'],
      score=None,
      status=MetricComputationStatus.SKIPPED,
      reasoning='Skipped contextual precision: no retrieved documents.',
      error_message=None,
    )

    report = _compose_report(
      run=run,
      agent=Agent(id=run.agent_id, name='demo-agent', version_tag='v-test'),
      dataset=Dataset(id=run.dataset_id, name='dataset'),
      total_samples=1,
      run_samples=[sample],
      computations=[legacy_skip],
      metrics=[Metric(name='contextual_precision_document_judge', description=None, requires_ground_truth=True)],
      generated_at=generated_at,
    )

    expected_reason = 'Skipped contextual precision: no retrieved documents.'
    self.assertEqual(report.metrics[0].coverage.skipped_reasons, {expected_reason: 1})
    self.assertEqual(
      report.failures.metric_computations.by_metric['contextual_precision_document_judge'].skipped_reasons,
      {expected_reason: 1},
    )
    self.assertEqual(report.failures.metric_computations.skipped_reasons, {expected_reason: 1})

  def test_score_stats_are_null_for_empty_scores(self) -> None:
    stats = _score_stats([])

    self.assertEqual(stats.count, 0)
    self.assertIsNone(stats.mean)
    self.assertIsNone(stats.stddev)
    self.assertIsNone(stats.min)
    self.assertIsNone(stats.p25)
    self.assertIsNone(stats.median)
    self.assertIsNone(stats.p75)
    self.assertIsNone(stats.max)

  def test_score_stats_define_single_sample_distribution(self) -> None:
    stats = _score_stats([0.9])

    self.assertEqual(stats.count, 1)
    self.assertEqual(stats.mean, 0.9)
    self.assertIsNone(stats.stddev)
    self.assertEqual(stats.min, 0.9)
    self.assertEqual(stats.p25, 0.9)
    self.assertEqual(stats.median, 0.9)
    self.assertEqual(stats.p75, 0.9)
    self.assertEqual(stats.max, 0.9)


if __name__ == '__main__':
  unittest.main()


class EvaluationReportSnapshotTest(unittest.IsolatedAsyncioTestCase):
  async def test_report_loads_plan_for_subset_and_metrics_with_no_computations(self):
    run = EvaluationRun(
      id=uuid4(),
      agent_id=uuid4(),
      dataset_id=uuid4(),
      status=EvaluationStatus.FAILED,
      start_time=datetime.now(timezone.utc),
      config={'plan_snapshot_version': 1},
    )
    row = EvaluationRunSample(
      id=uuid4(), evaluation_run_id=run.id, sample_id=uuid4(), status=EvaluationSampleStatus.FAILED
    )
    uow = AsyncMock()
    uow.__aenter__.return_value = uow
    uow.evaluation_runs.get_by_id.return_value = run
    uow.agents.get_by_id_or_raise.return_value = Agent(id=run.agent_id, name='agent', version_tag='v1')
    uow.datasets.get_by_id_or_raise.return_value = Dataset(id=run.dataset_id, name='dataset')
    uow.samples.count_by_dataset.return_value = 100
    uow.evaluation_run_samples.list_by_evaluation_run.return_value = [row]
    uow.metric_computations.list_by_evaluation_run.return_value = []
    uow.metrics.list_all.return_value = [Metric(name='custom_bla')]
    uow.spans.list_usage_spans_by_evaluation_run.return_value = []
    uow.evaluation_run_plan_samples.list_sample_ids.return_value = [row.sample_id, uuid4()]
    uow.evaluation_run_metrics.list_metrics.return_value = ['custom_bla']
    with patch('syllo_eval.evaluation.evaluation_report.UnitOfWork', return_value=uow):
      report = await EvaluationReportAggregator(cast(Any, object())).build(run.id)
      self.assertEqual(report.dataset.total_samples, 100)
      self.assertEqual(report.summary.samples.total, 2)
      self.assertEqual(report.summary.samples.unprocessed, 1)
      self.assertEqual(report.observed_metrics, [])
      self.assertEqual(report.selected_metrics, ['custom_bla'])
      self.assertEqual(report.metrics[0].coverage.samples_total, 2)
      self.assertEqual(report.metrics[0].coverage.computations_total, 0)
      self.assertEqual(report.metrics[0].coverage.coverage_rate, 0)
      self.assertIsNone(report.metrics[0].scores.mean)
      uow.evaluation_run_plan_samples.list_sample_ids.return_value = []
      uow.evaluation_run_samples.list_by_evaluation_run.return_value = []
      report = await EvaluationReportAggregator(cast(Any, object())).build(run.id)
      self.assertEqual(report.summary.samples.total, 0)
