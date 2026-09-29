import unittest
from datetime import datetime, timezone
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import AsyncMock, patch
from uuid import uuid4

from syllo_eval.service import (
  EvaluationService,
  MetricSelectionError,
  RepeatNotAllowedError,
  available_metric_names,
  resolve_selected_metric_names,
  resolve_selected_metric_names_csv,
)
from syllo_eval.evaluation.metrics import (
  AnswerCorrectnessJudgeMetric,
  ContextualPrecisionDocumentJudgeMetric,
  ContextualPrecisionSnippetJudgeMetric,
  ContextualRecallDocumentJudgeMetric,
  ContextualRecallSnippetJudgeMetric,
  NdcgAt10DocumentMetric,
  NdcgAt10SnippetMetric,
  PlanCorrectnessJudgeMetric,
  PlanEfficiencyMetric,
)
from syllo_eval.infrastructure.database import DatabaseManager
from syllo_eval.infrastructure.exceptions import NotFoundError
from syllo_eval.model import EvaluationRun, EvaluationRunSample, EvaluationSampleStatus, EvaluationStatus
from syllo_eval.model import Agent


_JUDGE_METRIC_NAMES = [
  AnswerCorrectnessJudgeMetric.metric_name,
  PlanCorrectnessJudgeMetric.metric_name,
  ContextualPrecisionDocumentJudgeMetric.metric_name,
  ContextualPrecisionSnippetJudgeMetric.metric_name,
  ContextualRecallDocumentJudgeMetric.metric_name,
  ContextualRecallSnippetJudgeMetric.metric_name,
]

_DETERMINISTIC_METRIC_NAMES = [
  NdcgAt10DocumentMetric.metric_name,
  NdcgAt10SnippetMetric.metric_name,
  PlanEfficiencyMetric.metric_name,
]


class TestMetricSelectionHelpers(unittest.TestCase):
  def test_available_metric_names_excludes_judge_metric_when_unavailable(self) -> None:
    metric_names = available_metric_names(llm_judge_enabled=False)

    for metric_name in _JUDGE_METRIC_NAMES:
      self.assertNotIn(metric_name, metric_names)
    for metric_name in _DETERMINISTIC_METRIC_NAMES:
      self.assertIn(metric_name, metric_names)

  def test_available_metric_names_includes_judge_metric_when_available(self) -> None:
    metric_names = available_metric_names(llm_judge_enabled=True)

    for metric_name in [*_JUDGE_METRIC_NAMES, *_DETERMINISTIC_METRIC_NAMES]:
      self.assertIn(metric_name, metric_names)

  def test_resolve_selected_metric_names_returns_all_metrics_when_omitted_or_empty(self) -> None:
    available_names = available_metric_names(llm_judge_enabled=False)
    raw_metric_names_cases: list[list[str] | None] = [None, []]

    for raw_metric_names in raw_metric_names_cases:
      with self.subTest(raw_metric_names=raw_metric_names):
        self.assertEqual(resolve_selected_metric_names(raw_metric_names, available_names), available_names)

  def test_resolve_selected_metric_names_parses_list_normalizes_case_and_deduplicates(self) -> None:
    available_names = available_metric_names(llm_judge_enabled=True)

    resolved = resolve_selected_metric_names(
      ['  LLM_CALLS ', 'set_precision_document', 'llm_calls', 'answer_correctness_judge'],
      available_names,
    )

    self.assertEqual(
      resolved,
      [
        'llm_calls',
        'set_precision_document',
        AnswerCorrectnessJudgeMetric.metric_name,
      ],
    )

  def test_resolve_selected_metric_names_csv_parses_csv_values(self) -> None:
    available_names = available_metric_names(llm_judge_enabled=False)

    resolved = resolve_selected_metric_names_csv('llm_calls, set_precision_document', available_names)

    self.assertEqual(resolved, ['llm_calls', 'set_precision_document'])

  def test_resolve_selected_metric_names_rejects_empty_entries(self) -> None:
    available_names = available_metric_names(llm_judge_enabled=False)

    with self.assertRaisesRegex(MetricSelectionError, 'cannot contain empty entries'):
      resolve_selected_metric_names(['llm_calls', '   '], available_names)

  def test_resolve_selected_metric_names_rejects_unknown_metrics(self) -> None:
    available_names = available_metric_names(llm_judge_enabled=False)

    with self.assertRaisesRegex(MetricSelectionError, 'Valid metrics:'):
      resolve_selected_metric_names(['llm_calls', 'unknown_metric'], available_names)

  def test_resolve_selected_metric_names_rejects_judge_metric_when_unavailable(self) -> None:
    available_names = available_metric_names(llm_judge_enabled=False)

    with self.assertRaisesRegex(MetricSelectionError, AnswerCorrectnessJudgeMetric.metric_name):
      resolve_selected_metric_names([AnswerCorrectnessJudgeMetric.metric_name], available_names)


class _PreparedExecutionStub:
  def __init__(self, run: EvaluationRun):
    self.create_run = AsyncMock(return_value=run)
    self.execute_run = AsyncMock(return_value=run)
    self.close = AsyncMock()


class _EvaluationRunsRepositoryStub:
  def __init__(
    self,
    *,
    get_by_id_result: EvaluationRun | None = None,
    list_result: list[EvaluationRun] | None = None,
    count_result: int = 0,
    fail_running_result: list[EvaluationRun] | None = None,
    update_status_if_current_result: EvaluationRun | None = None,
  ) -> None:
    self.get_by_id = AsyncMock(return_value=get_by_id_result)
    self.list_runs = AsyncMock(return_value=list_result or [])
    self.count_runs = AsyncMock(return_value=count_result)
    self.fail_running = AsyncMock(return_value=fail_running_result or [])
    self.update_status_if_current = AsyncMock(return_value=update_status_if_current_result)


class _AgentsRepositoryStub:
  def __init__(self, *, get_by_id_or_raise_result: Agent | None = None) -> None:
    self.get_by_id_or_raise = AsyncMock(return_value=get_by_id_or_raise_result)


class _SamplesRepositoryStub:
  def __init__(self, *, count_by_dataset_result: int = 0) -> None:
    self.count_by_dataset = AsyncMock(return_value=count_by_dataset_result)


class _EvaluationRunSamplesRepositoryStub:
  def __init__(self, *, list_by_evaluation_run_result: list[EvaluationRunSample] | None = None) -> None:
    self.list_by_evaluation_run = AsyncMock(return_value=list_by_evaluation_run_result or [])


class _EvaluationRunMetricsRepositoryStub:
  def __init__(self, *, list_metrics_result: list[str] | None = None) -> None:
    self.list_metrics = AsyncMock(return_value=list_metrics_result or [])


class _UnitOfWorkStub:
  def __init__(
    self,
    evaluation_runs: _EvaluationRunsRepositoryStub,
    samples: _SamplesRepositoryStub | None = None,
    evaluation_run_samples: _EvaluationRunSamplesRepositoryStub | None = None,
    evaluation_run_metrics: _EvaluationRunMetricsRepositoryStub | None = None,
    agents: _AgentsRepositoryStub | None = None,
  ) -> None:
    self.evaluation_runs = evaluation_runs
    self.agents = agents or _AgentsRepositoryStub()
    self.samples = samples or _SamplesRepositoryStub()
    self.evaluation_run_samples = evaluation_run_samples or _EvaluationRunSamplesRepositoryStub()
    self.evaluation_run_metrics = evaluation_run_metrics or _EvaluationRunMetricsRepositoryStub()
    self.evaluation_run_plan_samples = SimpleNamespace(list_sample_ids=AsyncMock(return_value=[]))

  async def __aenter__(self):
    return self

  async def __aexit__(self, exc_type, exc, tb):
    del exc_type, exc, tb


class TestEvaluationService(unittest.IsolatedAsyncioTestCase):
  async def test_status_uses_saved_plan_and_terminal_missing_samples_are_unprocessed(self):
    run = EvaluationRun(
      id=uuid4(),
      agent_id=uuid4(),
      dataset_id=uuid4(),
      status=EvaluationStatus.COMPLETED,
      start_time=datetime.now(timezone.utc),
    )
    row = EvaluationRunSample(
      id=uuid4(), evaluation_run_id=run.id, sample_id=uuid4(), status=EvaluationSampleStatus.COMPLETED
    )
    uow = _UnitOfWorkStub(_EvaluationRunsRepositoryStub(get_by_id_result=run))
    uow.evaluation_run_samples.list_by_evaluation_run.return_value = [row]
    uow.evaluation_run_plan_samples.list_sample_ids.return_value = [row.sample_id, uuid4()]
    uow.samples.count_by_dataset.return_value = 100
    service = EvaluationService(db_manager=cast(DatabaseManager, object()))
    with patch('syllo_eval.service.UnitOfWork', return_value=uow):
      result = await service.get_evaluation_status(run.id)
      self.assertEqual(
        (result.sample_counts.total, result.sample_counts.pending, result.sample_counts.unprocessed), (2, 0, 1)
      )
      run.status = EvaluationStatus.RUNNING
      result = await service.get_evaluation_status(run.id)
      self.assertEqual((result.sample_counts.pending, result.sample_counts.unprocessed), (1, 0))
      run.config = {'plan_snapshot_version': 1}
      uow.evaluation_run_plan_samples.list_sample_ids.return_value = []
      uow.evaluation_run_samples.list_by_evaluation_run.return_value = []
      result = await service.get_evaluation_status(run.id)
      self.assertEqual(result.sample_counts.total, 0)
    uow.samples.count_by_dataset.assert_not_awaited()

  async def test_run_evaluation_executes_blocking_run(self) -> None:
    dataset_id = uuid4()
    run = EvaluationRun(
      id=uuid4(),
      agent_id=uuid4(),
      dataset_id=dataset_id,
      status=EvaluationStatus.COMPLETED,
      start_time=datetime.now(tz=timezone.utc),
      end_time=datetime.now(tz=timezone.utc),
    )
    service = EvaluationService(
      settings=cast(
        Any,
        SimpleNamespace(llm_judge=SimpleNamespace(provider=None), orbitals=SimpleNamespace(api_key=None)),
      ),
      db_manager=cast(DatabaseManager, object()),
    )
    execution = _PreparedExecutionStub(run)

    with patch.object(service, '_build_execution', AsyncMock(return_value=execution)):
      result = await service.run_evaluation(
        agent_name='demo-agent',
        agent_version_tag='v-test',
        dataset_id=dataset_id,
      )

    self.assertEqual(result, run)
    execution.create_run.assert_awaited_once_with()
    execution.execute_run.assert_awaited_once_with(run)

  async def test_get_evaluation_run_returns_persisted_run(self) -> None:
    run = EvaluationRun(
      id=uuid4(),
      agent_id=uuid4(),
      dataset_id=uuid4(),
      status=EvaluationStatus.RUNNING,
      start_time=datetime.now(tz=timezone.utc),
    )
    service = EvaluationService(
      settings=cast(Any, SimpleNamespace(llm_judge=SimpleNamespace(provider=None))),
      db_manager=cast(DatabaseManager, object()),
    )
    evaluation_runs = _EvaluationRunsRepositoryStub(get_by_id_result=run)

    with patch('syllo_eval.service.UnitOfWork', return_value=_UnitOfWorkStub(evaluation_runs)):
      result = await service.get_evaluation_run(run.id)

    self.assertEqual(result, run)
    evaluation_runs.get_by_id.assert_awaited_once_with(run.id)

  async def test_get_evaluation_run_raises_not_found_for_missing_run(self) -> None:
    run_id = uuid4()
    service = EvaluationService(
      settings=cast(Any, SimpleNamespace(llm_judge=SimpleNamespace(provider=None))),
      db_manager=cast(DatabaseManager, object()),
    )
    evaluation_runs = _EvaluationRunsRepositoryStub(get_by_id_result=None)

    with patch('syllo_eval.service.UnitOfWork', return_value=_UnitOfWorkStub(evaluation_runs)):
      with self.assertRaises(NotFoundError):
        await service.get_evaluation_run(run_id)

    evaluation_runs.get_by_id.assert_awaited_once_with(run_id)

  async def test_get_evaluation_status_returns_sample_counts(self) -> None:
    run = EvaluationRun(
      id=uuid4(),
      agent_id=uuid4(),
      dataset_id=uuid4(),
      status=EvaluationStatus.RUNNING,
      start_time=datetime.now(tz=timezone.utc),
    )
    run_samples = [
      EvaluationRunSample(
        id=uuid4(),
        evaluation_run_id=run.id,
        sample_id=uuid4(),
        status=EvaluationSampleStatus.RUNNING,
      ),
      EvaluationRunSample(
        id=uuid4(),
        evaluation_run_id=run.id,
        sample_id=uuid4(),
        status=EvaluationSampleStatus.COMPLETED,
      ),
      EvaluationRunSample(
        id=uuid4(),
        evaluation_run_id=run.id,
        sample_id=uuid4(),
        status=EvaluationSampleStatus.FAILED,
      ),
    ]
    service = EvaluationService(
      settings=cast(Any, SimpleNamespace(llm_judge=SimpleNamespace(provider=None))),
      db_manager=cast(DatabaseManager, object()),
    )
    evaluation_runs = _EvaluationRunsRepositoryStub(get_by_id_result=run)
    samples = _SamplesRepositoryStub(count_by_dataset_result=5)
    evaluation_run_samples = _EvaluationRunSamplesRepositoryStub(list_by_evaluation_run_result=run_samples)

    with patch(
      'syllo_eval.service.UnitOfWork',
      return_value=_UnitOfWorkStub(evaluation_runs, samples, evaluation_run_samples),
    ):
      result = await service.get_evaluation_status(run.id)

    self.assertEqual(result.run, run)
    self.assertEqual(result.sample_counts.total, 5)
    self.assertEqual(result.sample_counts.pending, 2)
    self.assertEqual(result.sample_counts.running, 1)
    self.assertEqual(result.sample_counts.completed, 1)
    self.assertEqual(result.sample_counts.failed, 1)
    evaluation_runs.get_by_id.assert_awaited_once_with(run.id)
    samples.count_by_dataset.assert_awaited_once_with(run.dataset_id)
    evaluation_run_samples.list_by_evaluation_run.assert_awaited_once_with(run.id)

  async def test_list_evaluations_returns_paginated_runs(self) -> None:
    run = EvaluationRun(
      id=uuid4(),
      agent_id=uuid4(),
      dataset_id=uuid4(),
      status=EvaluationStatus.COMPLETED,
      start_time=datetime.now(tz=timezone.utc),
      end_time=datetime.now(tz=timezone.utc),
    )
    service = EvaluationService(
      settings=cast(Any, SimpleNamespace(llm_judge=SimpleNamespace(provider=None))),
      db_manager=cast(DatabaseManager, object()),
    )
    evaluation_runs = _EvaluationRunsRepositoryStub(list_result=[run], count_result=7)

    with patch('syllo_eval.service.UnitOfWork', return_value=_UnitOfWorkStub(evaluation_runs)):
      result = await service.list_evaluations(limit=25, offset=50, status=EvaluationStatus.COMPLETED)

    self.assertEqual(result.runs, [run])
    self.assertEqual(result.total, 7)
    self.assertEqual(result.limit, 25)
    self.assertEqual(result.offset, 50)
    evaluation_runs.list_runs.assert_awaited_once_with(limit=25, offset=50, status=EvaluationStatus.COMPLETED)
    evaluation_runs.count_runs.assert_awaited_once_with(status=EvaluationStatus.COMPLETED)

  async def test_fail_orphaned_running_evaluations_marks_running_runs_failed(self) -> None:
    run = EvaluationRun(
      id=uuid4(),
      agent_id=uuid4(),
      dataset_id=uuid4(),
      status=EvaluationStatus.FAILED,
      start_time=datetime.now(tz=timezone.utc),
      end_time=datetime.now(tz=timezone.utc),
    )
    service = EvaluationService(
      settings=cast(Any, SimpleNamespace(llm_judge=SimpleNamespace(provider=None))),
      db_manager=cast(DatabaseManager, object()),
    )
    evaluation_runs = _EvaluationRunsRepositoryStub(fail_running_result=[run])

    with patch('syllo_eval.service.UnitOfWork', return_value=_UnitOfWorkStub(evaluation_runs)):
      result = await service.fail_orphaned_running_evaluations()

    self.assertEqual(result, 1)
    evaluation_runs.fail_running.assert_awaited_once()

  async def test_fail_evaluation_run_marks_running_run_failed(self) -> None:
    run_id = uuid4()
    failed_run = EvaluationRun(
      id=run_id,
      agent_id=uuid4(),
      dataset_id=uuid4(),
      status=EvaluationStatus.FAILED,
      start_time=datetime.now(tz=timezone.utc),
      end_time=datetime.now(tz=timezone.utc),
    )
    service = EvaluationService(
      settings=cast(Any, SimpleNamespace(llm_judge=SimpleNamespace(provider=None))),
      db_manager=cast(DatabaseManager, object()),
    )
    evaluation_runs = _EvaluationRunsRepositoryStub(update_status_if_current_result=failed_run)

    with patch('syllo_eval.service.UnitOfWork', return_value=_UnitOfWorkStub(evaluation_runs)):
      result = await service.fail_evaluation_run(run_id)

    self.assertEqual(result, failed_run)
    evaluation_runs.update_status_if_current.assert_awaited_once()
    evaluation_runs.get_by_id.assert_not_awaited()

  async def test_repeat_evaluation_rejects_running_source_run(self) -> None:
    source_run = EvaluationRun(
      id=uuid4(),
      agent_id=uuid4(),
      dataset_id=uuid4(),
      status=EvaluationStatus.RUNNING,
      start_time=datetime.now(tz=timezone.utc),
    )
    service = EvaluationService(
      settings=cast(Any, SimpleNamespace(llm_judge=SimpleNamespace(provider=None))),
      db_manager=cast(DatabaseManager, object()),
    )
    evaluation_runs = _EvaluationRunsRepositoryStub(get_by_id_result=source_run)

    with patch('syllo_eval.service.UnitOfWork', return_value=_UnitOfWorkStub(evaluation_runs)):
      with self.assertRaises(RepeatNotAllowedError):
        await service.repeat_evaluation(source_run_id=source_run.id)

  async def test_repeat_evaluation_allows_metric_outside_source_snapshot(self) -> None:
    source_run = EvaluationRun(
      id=uuid4(),
      agent_id=uuid4(),
      dataset_id=uuid4(),
      status=EvaluationStatus.COMPLETED,
      start_time=datetime.now(tz=timezone.utc),
      end_time=datetime.now(tz=timezone.utc),
    )
    traced_sample = EvaluationRunSample(
      id=uuid4(),
      evaluation_run_id=source_run.id,
      sample_id=uuid4(),
      trace_id='trace-repeat',
      status=EvaluationSampleStatus.COMPLETED,
    )
    repeated_run = source_run.model_copy(update={'id': uuid4(), 'source_run_id': source_run.id})
    service = EvaluationService(
      settings=cast(
        Any,
        SimpleNamespace(llm_judge=SimpleNamespace(provider=None), orbitals=SimpleNamespace(api_key=None)),
      ),
      db_manager=cast(DatabaseManager, object()),
    )
    evaluation_runs = _EvaluationRunsRepositoryStub(get_by_id_result=source_run)
    evaluation_run_metrics = _EvaluationRunMetricsRepositoryStub(list_metrics_result=['plan_efficiency'])
    evaluation_run_samples = _EvaluationRunSamplesRepositoryStub(list_by_evaluation_run_result=[traced_sample])
    agents = _AgentsRepositoryStub(
      get_by_id_or_raise_result=Agent(id=source_run.agent_id, name='demo-agent', version_tag='v-test')
    )
    execution = _PreparedExecutionStub(repeated_run)

    with (
      patch(
        'syllo_eval.service.UnitOfWork',
        return_value=_UnitOfWorkStub(
          evaluation_runs,
          evaluation_run_samples=evaluation_run_samples,
          evaluation_run_metrics=evaluation_run_metrics,
          agents=agents,
        ),
      ),
      patch.object(service, '_build_execution_from_plan', AsyncMock(return_value=execution)) as build_execution,
    ):
      started = await service.repeat_evaluation(source_run_id=source_run.id, metrics=['llm_calls'])

    self.assertEqual(started.run, repeated_run)
    build_execution.assert_awaited_once()
    await_args = build_execution.await_args
    assert await_args is not None
    build_kwargs = await_args.kwargs
    self.assertEqual(build_kwargs['agent_id'], source_run.agent_id)
    self.assertEqual(build_kwargs['dataset_id'], source_run.dataset_id)
    self.assertEqual(build_kwargs['selected_metric_names'], ['llm_calls'])
    self.assertEqual(build_kwargs['planned_sample_ids'], [traced_sample.sample_id])
    self.assertEqual(build_kwargs['source_run_id'], source_run.id)


if __name__ == '__main__':
  unittest.main()
