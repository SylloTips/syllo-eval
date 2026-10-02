import asyncio
import statistics
from collections import Counter, defaultdict
from collections.abc import Callable, Iterable, Mapping, Sequence
from datetime import datetime, timezone
from math import ceil, floor
from typing import Any
from uuid import UUID

from pydantic import BaseModel, Field

from syllo_eval.infrastructure.database import DatabaseManager
from syllo_eval.infrastructure.exceptions import NotFoundError
from syllo_eval.infrastructure.unit_of_work import UnitOfWork
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
  Span,
)

_REPORT_VERSION = '1.5'
_TOKEN_USAGE_KEYS = ('input_tokens', 'output_tokens', 'total_tokens', 'cached_input_tokens')
_FAILURE_PHASES = ('agent_call', 'trace_fetch', 'metric_compute', 'unknown')
_FAILURE_PHASE_SET = set(_FAILURE_PHASES)


class EvaluationReportNotAvailableError(Exception):
  """Raised when a report is requested before the run reaches a terminal status."""


class EvaluationReportRun(BaseModel):
  id: UUID
  status: EvaluationStatus
  start_time: datetime
  end_time: datetime | None
  duration_seconds: float | None
  samples_processed: int
  config: dict[str, Any] | None = None


class EvaluationReportAgent(BaseModel):
  id: UUID
  name: str
  version_tag: str


class EvaluationReportDataset(BaseModel):
  id: UUID
  name: str
  total_samples: int


class EvaluationReportSampleCounts(BaseModel):
  unprocessed: int = 0
  total: int
  succeeded: int
  failed: int
  success_rate: float


class EvaluationReportLatencyStats(BaseModel):
  mean: float | None
  median: float | None
  p95: float | None
  min: float | None
  max: float | None
  total_wall_time: float | None


class EvaluationReportMetricComputationCounts(BaseModel):
  total: int
  completed: int
  failed: int
  skipped: int


class EvaluationReportTokenUsage(BaseModel):
  input_tokens: int
  output_tokens: int
  total_tokens: int
  cached_input_tokens: int
  sources_with_data: int
  sources_total: int


class EvaluationReportJudgeTokenUsage(EvaluationReportTokenUsage):
  calls: int


class EvaluationReportTokenUsageSection(BaseModel):
  agent: EvaluationReportTokenUsage
  judge: EvaluationReportJudgeTokenUsage


class EvaluationReportSummary(BaseModel):
  samples: EvaluationReportSampleCounts
  latency_seconds: EvaluationReportLatencyStats
  metric_computations: EvaluationReportMetricComputationCounts
  token_usage: EvaluationReportTokenUsageSection


class EvaluationReportMetricCoverage(BaseModel):
  samples_total: int
  computations_total: int
  completed: int
  failed: int
  skipped: int
  coverage_rate: float
  skipped_reasons: dict[str, int]


class EvaluationReportScoreStats(BaseModel):
  count: int
  mean: float | None
  stddev: float | None
  min: float | None
  p25: float | None
  median: float | None
  p75: float | None
  max: float | None


class EvaluationReportMetric(BaseModel):
  name: str
  description: str | None
  requires_ground_truth: bool
  coverage: EvaluationReportMetricCoverage
  scores: EvaluationReportScoreStats


class EvaluationReportSampleFailureItem(BaseModel):
  evaluation_run_sample_id: UUID
  sample_id: UUID
  phase: str
  error_message: str | None


class EvaluationReportSampleFailures(BaseModel):
  by_phase: dict[str, int]
  items: list[EvaluationReportSampleFailureItem]


class EvaluationReportMetricFailureCounts(BaseModel):
  failed: int
  skipped: int
  failed_reasons: dict[str, int]
  skipped_reasons: dict[str, int]


class EvaluationReportMetricComputationFailures(BaseModel):
  by_metric: dict[str, EvaluationReportMetricFailureCounts]
  failed_reasons: dict[str, int]
  skipped_reasons: dict[str, int]


class EvaluationReportFailures(BaseModel):
  samples: EvaluationReportSampleFailures
  metric_computations: EvaluationReportMetricComputationFailures


class EvaluationReportSampleMetric(BaseModel):
  value: float | None
  computations_total: int
  completed: int
  failed: int
  skipped: int


class EvaluationReportSample(BaseModel):
  evaluation_run_sample_id: UUID
  sample_id: UUID
  trace_id: str | None
  status: EvaluationSampleStatus
  started_at: datetime | None
  ended_at: datetime | None
  duration_seconds: float | None
  error_message: str | None
  metrics: dict[str, EvaluationReportSampleMetric]
  token_usage: EvaluationReportTokenUsageSection


class EvaluationReport(BaseModel):
  report_version: str
  generated_at: datetime
  run: EvaluationReportRun
  agent: EvaluationReportAgent
  dataset: EvaluationReportDataset
  observed_metrics: list[str]
  selected_metrics: list[str] = Field(default_factory=list)
  summary: EvaluationReportSummary
  metrics: list[EvaluationReportMetric]
  failures: EvaluationReportFailures
  samples: list[EvaluationReportSample]


class EvaluationReportAggregator:
  def __init__(
    self,
    db_manager: DatabaseManager,
    clock: Callable[[], datetime] | None = None,
  ):
    self._db_manager = db_manager
    self._clock = clock or _utc_now

  async def build(self, run_id: UUID) -> EvaluationReport:
    async with UnitOfWork(self._db_manager) as uow:
      run = await uow.evaluation_runs.get_by_id(run_id)

    if run is None:
      raise NotFoundError('evaluation_run', run_id)

    if run.status == EvaluationStatus.RUNNING:
      raise EvaluationReportNotAvailableError('Evaluation run is still running.')

    usage_spans, plan = await asyncio.gather(self._list_usage_spans(run.id), self._get_run_plan(run.id))
    agent, dataset, total_samples, run_samples, computations, metrics = await asyncio.gather(
      self._get_agent(run.agent_id),
      self._get_dataset(run.dataset_id),
      self._count_dataset_samples(run.dataset_id),
      self._list_run_samples(run.id),
      self._list_metric_computations(run.id),
      self._list_metrics(),
    )
    planned_ids, selected_metrics = plan
    planned_sample_count = (
      len(planned_ids) if planned_ids or (run.config or {}).get('plan_snapshot_version') == 1 else total_samples
    )

    return _compose_report(
      run=run,
      agent=agent,
      dataset=dataset,
      total_samples=total_samples,
      run_samples=run_samples,
      computations=computations,
      metrics=metrics,
      usage_spans=usage_spans,
      generated_at=self._clock(),
      planned_sample_count=planned_sample_count,
      selected_metrics=selected_metrics,
    )

  async def _get_run_plan(self, run_id: UUID) -> tuple[list[UUID], list[str]]:
    async with UnitOfWork(self._db_manager) as uow:
      return (
        await uow.evaluation_run_plan_samples.list_sample_ids(run_id),
        await uow.evaluation_run_metrics.list_metrics(run_id),
      )

  async def _get_agent(self, agent_id: UUID) -> Agent:
    async with UnitOfWork(self._db_manager) as uow:
      return await uow.agents.get_by_id_or_raise(agent_id)

  async def _get_dataset(self, dataset_id: UUID) -> Dataset:
    async with UnitOfWork(self._db_manager) as uow:
      return await uow.datasets.get_by_id_or_raise(dataset_id)

  async def _count_dataset_samples(self, dataset_id: UUID) -> int:
    async with UnitOfWork(self._db_manager) as uow:
      return await uow.samples.count_by_dataset(dataset_id)

  async def _list_run_samples(self, run_id: UUID) -> list[EvaluationRunSample]:
    async with UnitOfWork(self._db_manager) as uow:
      return await uow.evaluation_run_samples.list_by_evaluation_run(run_id)

  async def _list_metric_computations(self, run_id: UUID) -> list[MetricComputation]:
    async with UnitOfWork(self._db_manager) as uow:
      return await uow.metric_computations.list_by_evaluation_run(run_id)

  async def _list_metrics(self) -> list[Metric]:
    async with UnitOfWork(self._db_manager) as uow:
      return await uow.metrics.list_all()

  async def _list_usage_spans(self, run_id: UUID) -> list[Span]:
    async with UnitOfWork(self._db_manager) as uow:
      return await uow.spans.list_usage_spans_by_evaluation_run(run_id)


def _compose_report(
  *,
  run: EvaluationRun,
  agent: Agent,
  dataset: Dataset,
  total_samples: int,
  run_samples: Sequence[EvaluationRunSample],
  computations: Sequence[MetricComputation],
  metrics: Sequence[Metric],
  usage_spans: Sequence[Span] = (),
  generated_at: datetime,
  planned_sample_count: int | None = None,
  selected_metrics: Sequence[str] = (),
) -> EvaluationReport:
  expected_samples = planned_sample_count if planned_sample_count is not None else total_samples
  computations_by_sample = _computations_by_sample(run_samples, computations)
  spans_by_trace = _usage_spans_by_trace(usage_spans)
  observed_metrics = sorted({computation.metric for computation in computations})
  metric_metadata = {metric.name: metric for metric in metrics}
  run_duration = _duration_seconds(run.start_time, run.end_time)

  return EvaluationReport(
    report_version=_REPORT_VERSION,
    generated_at=generated_at,
    run=EvaluationReportRun(
      id=run.id,
      status=run.status,
      start_time=run.start_time,
      end_time=run.end_time,
      duration_seconds=run_duration,
      samples_processed=len(run_samples),
      config=run.config,
    ),
    agent=EvaluationReportAgent(id=agent.id, name=agent.name, version_tag=agent.version_tag),
    dataset=EvaluationReportDataset(id=dataset.id, name=dataset.name, total_samples=total_samples),
    observed_metrics=observed_metrics,
    selected_metrics=list(selected_metrics),
    summary=_build_summary(run_samples, computations, usage_spans, run_duration, expected_samples),
    metrics=[
      _build_metric_report(metric_name, metric_metadata.get(metric_name), expected_samples, computations)
      for metric_name in sorted(set(selected_metrics) | set(observed_metrics))
    ],
    failures=_build_failures(run_samples, computations),
    samples=[
      _build_sample_report(
        run_sample,
        computations_by_sample[run_sample.id],
        spans_by_trace.get(run_sample.trace_id or '', ()),
      )
      for run_sample in run_samples
    ],
  )


def _build_summary(
  run_samples: Sequence[EvaluationRunSample],
  computations: Sequence[MetricComputation],
  usage_spans: Sequence[Span],
  run_duration: float | None,
  expected_samples: int,
) -> EvaluationReportSummary:
  succeeded = sum(1 for run_sample in run_samples if run_sample.status == EvaluationSampleStatus.COMPLETED)
  failed = sum(1 for run_sample in run_samples if run_sample.status == EvaluationSampleStatus.FAILED)
  sample_durations = [
    duration
    for run_sample in run_samples
    if run_sample.status == EvaluationSampleStatus.COMPLETED
    for duration in [_duration_seconds(run_sample.started_at, run_sample.ended_at)]
    if duration is not None
  ]
  computation_counts = _status_counts(computations)

  return EvaluationReportSummary(
    samples=EvaluationReportSampleCounts(
      total=expected_samples,
      unprocessed=max(0, expected_samples - len(run_samples)),
      succeeded=succeeded,
      failed=failed,
      success_rate=_rate(succeeded, expected_samples),
    ),
    latency_seconds=_latency_stats(sample_durations, run_duration),
    metric_computations=EvaluationReportMetricComputationCounts(
      total=len(computations),
      completed=computation_counts[MetricComputationStatus.COMPLETED],
      failed=computation_counts[MetricComputationStatus.FAILED],
      skipped=computation_counts[MetricComputationStatus.SKIPPED],
    ),
    token_usage=_build_token_usage_section(usage_spans, computations),
  )


def _build_metric_report(
  metric_name: str,
  metric: Metric | None,
  expected_samples: int,
  computations: Sequence[MetricComputation],
) -> EvaluationReportMetric:
  metric_computations = [computation for computation in computations if computation.metric == metric_name]
  counts = _status_counts(metric_computations)
  completed_sample_ids = {
    computation.evaluation_run_sample_id
    for computation in metric_computations
    if computation.status == MetricComputationStatus.COMPLETED and computation.score is not None
  }

  return EvaluationReportMetric(
    name=metric_name,
    description=metric.description if metric is not None else None,
    requires_ground_truth=metric.requires_ground_truth if metric is not None else True,
    coverage=EvaluationReportMetricCoverage(
      samples_total=expected_samples,
      computations_total=len(metric_computations),
      completed=counts[MetricComputationStatus.COMPLETED],
      failed=counts[MetricComputationStatus.FAILED],
      skipped=counts[MetricComputationStatus.SKIPPED],
      coverage_rate=_rate(len(completed_sample_ids), expected_samples),
      skipped_reasons=_skipped_reasons(metric_computations),
    ),
    scores=_score_stats(
      [
        computation.score
        for computation in metric_computations
        if computation.status == MetricComputationStatus.COMPLETED and computation.score is not None
      ]
    ),
  )


def _build_failures(
  run_samples: Sequence[EvaluationRunSample],
  computations: Sequence[MetricComputation],
) -> EvaluationReportFailures:
  sample_failure_items: list[EvaluationReportSampleFailureItem] = []
  sample_failures_by_phase = {phase: 0 for phase in _FAILURE_PHASES}
  for run_sample in run_samples:
    if run_sample.status != EvaluationSampleStatus.FAILED:
      continue

    phase = _sample_failure_phase(run_sample)
    sample_failures_by_phase[phase] += 1
    sample_failure_items.append(
      EvaluationReportSampleFailureItem(
        evaluation_run_sample_id=run_sample.id,
        sample_id=run_sample.sample_id,
        phase=phase,
        error_message=run_sample.error_message,
      )
    )

  metric_failures_by_metric: dict[str, EvaluationReportMetricFailureCounts] = {}
  for metric_name in sorted({computation.metric for computation in computations}):
    metric_computations = [computation for computation in computations if computation.metric == metric_name]
    failed = sum(1 for computation in metric_computations if computation.status == MetricComputationStatus.FAILED)
    skipped = sum(1 for computation in metric_computations if computation.status == MetricComputationStatus.SKIPPED)
    if failed or skipped:
      metric_failures_by_metric[metric_name] = EvaluationReportMetricFailureCounts(
        failed=failed,
        skipped=skipped,
        failed_reasons=_failed_reasons(metric_computations),
        skipped_reasons=_skipped_reasons(metric_computations),
      )

  return EvaluationReportFailures(
    samples=EvaluationReportSampleFailures(
      by_phase=sample_failures_by_phase,
      items=sample_failure_items,
    ),
    metric_computations=EvaluationReportMetricComputationFailures(
      by_metric=metric_failures_by_metric,
      failed_reasons=_failed_reasons(computations),
      skipped_reasons=_skipped_reasons(computations),
    ),
  )


def _build_sample_report(
  run_sample: EvaluationRunSample,
  computations: Sequence[MetricComputation],
  usage_spans: Sequence[Span],
) -> EvaluationReportSample:
  return EvaluationReportSample(
    evaluation_run_sample_id=run_sample.id,
    sample_id=run_sample.sample_id,
    trace_id=run_sample.trace_id,
    status=run_sample.status,
    started_at=run_sample.started_at,
    ended_at=run_sample.ended_at,
    duration_seconds=_duration_seconds(run_sample.started_at, run_sample.ended_at),
    error_message=run_sample.error_message,
    metrics={
      metric_name: _build_sample_metric(metric_computations)
      for metric_name, metric_computations in _computations_by_metric(computations).items()
    },
    token_usage=_build_token_usage_section(usage_spans, computations),
  )


def _build_sample_metric(computations: Sequence[MetricComputation]) -> EvaluationReportSampleMetric:
  counts = _status_counts(computations)
  completed_scores = [
    computation.score
    for computation in computations
    if computation.status == MetricComputationStatus.COMPLETED and computation.score is not None
  ]

  return EvaluationReportSampleMetric(
    value=statistics.mean(completed_scores) if completed_scores else None,
    computations_total=len(computations),
    completed=counts[MetricComputationStatus.COMPLETED],
    failed=counts[MetricComputationStatus.FAILED],
    skipped=counts[MetricComputationStatus.SKIPPED],
  )


def _score_stats(scores: Sequence[float]) -> EvaluationReportScoreStats:
  if not scores:
    return EvaluationReportScoreStats(
      count=0,
      mean=None,
      stddev=None,
      min=None,
      p25=None,
      median=None,
      p75=None,
      max=None,
    )

  sorted_scores = sorted(scores)
  if len(sorted_scores) == 1:
    score = sorted_scores[0]
    return EvaluationReportScoreStats(
      count=1,
      mean=score,
      stddev=None,
      min=score,
      p25=score,
      median=score,
      p75=score,
      max=score,
    )

  return EvaluationReportScoreStats(
    count=len(sorted_scores),
    mean=statistics.mean(sorted_scores),
    stddev=statistics.stdev(sorted_scores),
    min=sorted_scores[0],
    p25=_percentile(sorted_scores, 0.25),
    median=statistics.median(sorted_scores),
    p75=_percentile(sorted_scores, 0.75),
    max=sorted_scores[-1],
  )


def _latency_stats(durations: Sequence[float], total_wall_time: float | None) -> EvaluationReportLatencyStats:
  stats = _score_stats(durations)
  return EvaluationReportLatencyStats(
    mean=stats.mean,
    median=stats.median,
    p95=_percentile(sorted(durations), 0.95) if durations else None,
    min=stats.min,
    max=stats.max,
    total_wall_time=total_wall_time,
  )


def _percentile(sorted_values: Sequence[float], percentile: float) -> float:
  if len(sorted_values) == 1:
    return sorted_values[0]

  rank = (len(sorted_values) - 1) * percentile
  lower_index = floor(rank)
  upper_index = ceil(rank)
  if lower_index == upper_index:
    return sorted_values[lower_index]

  lower_value = sorted_values[lower_index]
  upper_value = sorted_values[upper_index]
  return lower_value + (upper_value - lower_value) * (rank - lower_index)


def _computations_by_sample(
  run_samples: Sequence[EvaluationRunSample],
  computations: Sequence[MetricComputation],
) -> dict[UUID, list[MetricComputation]]:
  grouped: dict[UUID, list[MetricComputation]] = {run_sample.id: [] for run_sample in run_samples}
  for computation in computations:
    grouped[computation.evaluation_run_sample_id].append(computation)
  return grouped


def _computations_by_metric(
  computations: Sequence[MetricComputation],
) -> dict[str, list[MetricComputation]]:
  grouped: dict[str, list[MetricComputation]] = defaultdict(list)
  for computation in computations:
    grouped[computation.metric].append(computation)
  return dict(sorted(grouped.items()))


def _build_token_usage_section(
  usage_spans: Sequence[Span],
  computations: Sequence[MetricComputation],
) -> EvaluationReportTokenUsageSection:
  return EvaluationReportTokenUsageSection(
    agent=_sum_token_usage(_span_token_usages(usage_spans), len({_span_usage_key(span) for span in usage_spans})),
    judge=EvaluationReportJudgeTokenUsage(
      **_sum_token_usage(_judge_token_usages(computations), len(computations)).model_dump(),
      calls=sum(
        calls
        for computation in computations
        if isinstance(calls := (computation.metadata or {}).get('judge_calls'), int)
      ),
    ),
  )


def _sum_token_usage(usages: Iterable[Mapping[str, int]], sources_total: int) -> EvaluationReportTokenUsage:
  totals = {key: 0 for key in _TOKEN_USAGE_KEYS}
  sources_with_data = 0
  for usage in usages:
    sources_with_data += 1
    for key in _TOKEN_USAGE_KEYS:
      value = usage.get(key)
      if isinstance(value, int):
        totals[key] += value
  return EvaluationReportTokenUsage(
    **totals,
    sources_with_data=sources_with_data,
    sources_total=sources_total,
  )


def _span_usage_key(span: Span) -> tuple[str, str]:
  usage = span.semantics.usage
  return span.trace_id, (usage.call_id if usage else None) or span.external_id


def _span_token_usages(spans: Sequence[Span]) -> Iterable[Mapping[str, int]]:
  seen: set[tuple[str, str]] = set()
  for span in spans:
    usage = span.semantics.usage
    call_id = _span_usage_key(span)
    if call_id in seen:
      continue
    if usage is not None:
      counts = {key: value for key in _TOKEN_USAGE_KEYS if (value := getattr(usage, key)) is not None}
    else:
      legacy = (span.metadata or {}).get('token_usage')
      if not isinstance(legacy, Mapping):
        continue
      counts = {key: value for key, value in legacy.items() if key in _TOKEN_USAGE_KEYS and isinstance(value, int)}
    if counts:
      seen.add(call_id)
      yield counts


def _judge_token_usages(computations: Sequence[MetricComputation]) -> Iterable[Mapping[str, int]]:
  for computation in computations:
    usage = (computation.metadata or {}).get('judge_usage') if computation.metadata else None
    if isinstance(usage, Mapping) and usage:
      yield usage


def _usage_spans_by_trace(spans: Sequence[Span]) -> dict[str, list[Span]]:
  grouped: dict[str, list[Span]] = defaultdict(list)
  for span in spans:
    grouped[span.trace_id].append(span)
  return dict(grouped)


def _status_counts(computations: Sequence[MetricComputation]) -> Counter[MetricComputationStatus]:
  return Counter(computation.status for computation in computations)


def _skipped_reasons(computations: Sequence[MetricComputation]) -> dict[str, int]:
  counter: Counter[str] = Counter(
    _skip_reason(computation) for computation in computations if computation.status == MetricComputationStatus.SKIPPED
  )
  return dict(sorted(counter.items()))


def _failed_reasons(computations: Sequence[MetricComputation]) -> dict[str, int]:
  counter: Counter[str] = Counter(
    _failure_reason(computation) for computation in computations if computation.status == MetricComputationStatus.FAILED
  )
  return dict(sorted(counter.items()))


def _failure_reason(computation: MetricComputation) -> str:
  return computation.error_message or computation.reasoning or 'unknown'


def _skip_reason(computation: MetricComputation) -> str:
  return computation.error_message or computation.reasoning or 'unknown'


def _duration_seconds(start_time: datetime | None, end_time: datetime | None) -> float | None:
  if start_time is None or end_time is None:
    return None
  return (end_time - start_time).total_seconds()


def _rate(numerator: int, denominator: int) -> float:
  if denominator == 0:
    return 0.0
  return numerator / denominator


def _sample_failure_phase(run_sample: EvaluationRunSample) -> str:
  phase = (run_sample.metadata or {}).get('failure_phase', 'unknown')
  if phase in _FAILURE_PHASE_SET:
    return str(phase)
  return 'unknown'


def _utc_now() -> datetime:
  return datetime.now(timezone.utc)
