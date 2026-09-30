import asyncio
import unittest
from datetime import datetime, timezone
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import patch
from uuid import uuid4

from fastapi.testclient import TestClient

from syllo_eval.API.app import _drain_evaluation_tasks, create_app
from syllo_eval.service import AgentCallerSelectionError, MetricSelectionError, RepeatNotAllowedError
from syllo_eval.datasets import DatasetAlreadyExistsError, DatasetImportSummary, DatasetListPage
from syllo_eval.evaluation.evaluation_report import EvaluationReportNotAvailableError
from syllo_eval.evaluation.trace_import import TraceImportError
from syllo_eval.infrastructure.exceptions import NotFoundError
from syllo_eval.model import Dataset, DatasetSummary, EvaluationRun, EvaluationStatus


class _EvaluationHandleStub:
  def __init__(
    self, run: EvaluationRun, agent_name: str | None = 'demo-agent', agent_version_tag: str | None = 'v-test'
  ):
    self.run = run
    self.agent_name = agent_name
    self.agent_version_tag = agent_version_tag

  async def execute(self) -> EvaluationRun:
    return self.run


class _ServiceStub:
  def __init__(
    self,
    started_evaluation: _EvaluationHandleStub | None = None,
    run: EvaluationRun | None = None,
    runs: list[EvaluationRun] | None = None,
    status_snapshot: object | None = None,
    failed_run: EvaluationRun | None = None,
    report: dict[str, Any] | None = None,
    error: Exception | None = None,
  ):
    self._started_evaluation = started_evaluation
    self._run = run or (started_evaluation.run if started_evaluation is not None else None)
    self._runs = runs or []
    self._status_snapshot = status_snapshot
    self._failed_run = failed_run or self._run
    self._report = report
    self._error = error
    self.db_manager = object()
    self.calls: list[dict[str, object]] = []
    self.list_calls: list[dict[str, object]] = []
    self.get_calls: list[object] = []
    self.cancel_calls: list[object] = []
    self.report_calls: list[object] = []
    self.repeat_calls: list[dict[str, object]] = []
    self.orphan_cleanup_calls = 0

  async def create_evaluation(
    self,
    *,
    agent_name: str,
    agent_version_tag: str,
    dataset_id,
    selected_metric_names,
  ):
    self.calls.append(
      {
        'agent_name': agent_name,
        'agent_version_tag': agent_version_tag,
        'dataset_id': dataset_id,
        'selected_metric_names': selected_metric_names,
      }
    )
    if self._error is not None:
      raise self._error
    return self._started_evaluation

  async def import_evaluation(self, **kwargs):
    self.calls.append(kwargs)
    if self._error is not None:
      raise self._error
    return self._started_evaluation

  async def repeat_evaluation(self, *, source_run_id, metrics, **kwargs):
    del kwargs
    self.repeat_calls.append({'source_run_id': source_run_id, 'metrics': metrics})
    if self._error is not None:
      raise self._error
    return self._started_evaluation

  async def get_evaluation_run(self, run_id):
    self.get_calls.append(run_id)
    if self._error is not None:
      raise self._error
    return self._run

  async def list_evaluations(self, *, limit: int, offset: int, status=None):
    self.list_calls.append({'limit': limit, 'offset': offset, 'status': status})
    return SimpleNamespace(runs=self._runs, total=len(self._runs), limit=limit, offset=offset)

  async def get_evaluation_status(self, run_id):
    self.get_calls.append(run_id)
    if self._error is not None:
      raise self._error
    return self._status_snapshot

  async def fail_evaluation_run(self, run_id):
    self.cancel_calls.append(run_id)
    if self._error is not None:
      raise self._error
    return self._failed_run

  async def get_evaluation_report(self, run_id):
    self.report_calls.append(run_id)
    if self._error is not None:
      raise self._error
    return self._report

  async def fail_orphaned_running_evaluations(self):
    self.orphan_cleanup_calls += 1
    return 0


class _DatasetServiceStub:
  def __init__(
    self,
    import_summary: DatasetImportSummary | None = None,
    list_page: DatasetListPage | None = None,
    error: Exception | None = None,
  ):
    self._import_summary = import_summary
    self._list_page = list_page
    self._error = error
    self.import_calls: list[dict[str, object]] = []
    self.list_calls: list[dict[str, int]] = []

  async def import_dataset(self, *, name, payload):
    self.import_calls.append({'name': name, 'payload': payload})
    if self._error is not None:
      raise self._error
    return self._import_summary

  async def list_datasets(self, *, limit: int, offset: int):
    self.list_calls.append({'limit': limit, 'offset': offset})
    if self._error is not None:
      raise self._error
    return self._list_page


class _FakeTask:
  def __init__(self):
    self.callbacks = []
    self.cancel_calls = 0
    self.done_value = False

  def add_done_callback(self, callback):
    self.callbacks.append(callback)

  def cancel(self):
    self.cancel_calls += 1

  def done(self):
    return self.done_value

  def __await__(self):
    return iter(())


class _RuntimeServiceStub:
  def __init__(self):
    self.initialize_calls = 0
    self.close_calls = 0

  async def initialize(self) -> None:
    self.initialize_calls += 1

  async def close(self) -> None:
    self.close_calls += 1


class TestEvaluationApi(unittest.TestCase):
  @staticmethod
  def _build_run() -> EvaluationRun:
    return EvaluationRun(
      id=uuid4(),
      agent_id=uuid4(),
      dataset_id=uuid4(),
      status=EvaluationStatus.RUNNING,
      start_time=datetime.now(tz=timezone.utc),
    )

  def _build_task_factory(self, scheduled_tasks: list[_FakeTask]):
    def _fake_create_task(started):
      del started
      task = _FakeTask()
      scheduled_tasks.append(task)
      return task

    return _fake_create_task

  def test_post_evaluations_returns_202_with_run_identifier(self) -> None:
    run = self._build_run()
    service = _ServiceStub(started_evaluation=_EvaluationHandleStub(run))
    app = create_app(service=cast(Any, service))
    scheduled_tasks: list[_FakeTask] = []

    with (
      patch('syllo_eval.API.app._create_evaluation_task', side_effect=self._build_task_factory(scheduled_tasks)),
      TestClient(app) as client,
    ):
      response = client.post(
        '/evaluations',
        json={
          'agent_name': 'demo-agent',
          'agent_version_tag': 'v-test',
          'dataset_id': str(run.dataset_id),
        },
      )

    self.assertEqual(response.status_code, 202)
    self.assertEqual(
      response.json(),
      {
        'evaluation_run_id': str(run.id),
        'status': EvaluationStatus.RUNNING.value,
        'agent_name': 'demo-agent',
        'agent_version_tag': 'v-test',
        'dataset_id': str(run.dataset_id),
      },
    )
    self.assertEqual(len(scheduled_tasks), 1)
    self.assertEqual(len(app.state.evaluation_tasks), 1)
    self.assertIs(app.state.evaluation_tasks[run.id], scheduled_tasks[0])

  def test_post_evaluations_passes_metric_selection(self) -> None:
    cases = [
      (None, None),
      ([], []),
      (['llm_calls', 'set_precision_document'], ['llm_calls', 'set_precision_document']),
    ]

    for request_metrics, expected_metric_names in cases:
      with self.subTest(request_metrics=request_metrics):
        run = self._build_run()
        service = _ServiceStub(started_evaluation=_EvaluationHandleStub(run))
        app = create_app(service=cast(Any, service))
        request_body: dict[str, object] = {
          'agent_name': 'demo-agent',
          'agent_version_tag': 'v-test',
          'dataset_id': str(run.dataset_id),
        }
        if request_metrics is not None:
          request_body['metrics'] = request_metrics

        with (
          patch('syllo_eval.API.app._create_evaluation_task', side_effect=self._build_task_factory([])),
          TestClient(app) as client,
        ):
          client.post('/evaluations', json=request_body)

        self.assertEqual(service.calls[0]['selected_metric_names'], expected_metric_names)

  def test_post_evaluations_returns_400_for_invalid_metrics(self) -> None:
    service = _ServiceStub(
      error=MetricSelectionError('Unknown metric name(s): unknown_metric. Valid metrics: llm_calls')
    )
    app = create_app(service=cast(Any, service))

    with TestClient(app) as client:
      response = client.post(
        '/evaluations',
        json={
          'agent_name': 'demo-agent',
          'agent_version_tag': 'v-test',
          'dataset_id': str(uuid4()),
          'metrics': ['unknown_metric'],
        },
      )

    self.assertEqual(response.status_code, 400)
    self.assertEqual(
      response.json(),
      {'detail': 'Unknown metric name(s): unknown_metric. Valid metrics: llm_calls'},
    )

  def test_post_evaluations_returns_400_for_unregistered_caller(self) -> None:
    message = "No caller registered for agent 'unknown'. Supply callers_by_agent_name."
    app = create_app(service=cast(Any, _ServiceStub(error=AgentCallerSelectionError(message))))

    with TestClient(app) as client, patch('syllo_eval.API.app._create_evaluation_task') as create_task:
      response = client.post(
        '/evaluations',
        json={'agent_name': 'unknown', 'agent_version_tag': 'v-test', 'dataset_id': str(uuid4())},
      )
      create_task.assert_not_called()

    self.assertEqual(response.status_code, 400)
    self.assertEqual(response.json(), {'detail': message})

  def test_post_evaluations_does_not_translate_unrelated_value_errors(self) -> None:
    app = create_app(service=cast(Any, _ServiceStub(error=ValueError('unexpected failure'))))

    with TestClient(app) as client:
      with self.assertRaisesRegex(ValueError, 'unexpected failure'):
        client.post(
          '/evaluations',
          json={'agent_name': 'demo-agent', 'agent_version_tag': 'v-test', 'dataset_id': str(uuid4())},
        )

  def test_post_import_passes_traces_and_adapter_and_maps_import_errors_to_400(self) -> None:
    run = self._build_run()
    body = {
      'agent_name': 'demo-agent',
      'agent_version_tag': 'v-test',
      'dataset_id': str(run.dataset_id),
      'trace_adapter': 'custom-source',
      'traces': [{'trace_id': 'trace-1', 'spans': [{'name': 'root'}]}],
    }
    service = _ServiceStub(started_evaluation=_EvaluationHandleStub(run))
    scheduled_tasks: list[_FakeTask] = []

    with (
      patch('syllo_eval.API.app._create_evaluation_task', side_effect=self._build_task_factory(scheduled_tasks)),
      TestClient(create_app(service=cast(Any, service))) as client,
    ):
      response = client.post('/evaluations/import', json=body)

    self.assertEqual(response.status_code, 202)
    self.assertEqual(response.json()['evaluation_run_id'], str(run.id))
    self.assertEqual(len(scheduled_tasks), 1)
    call = service.calls[0]
    self.assertEqual((call['trace_adapter_name'], call['dataset_id']), ('custom-source', run.dataset_id))
    self.assertEqual([trace.trace_id for trace in cast(Any, call['traces'])], ['trace-1'])

    error_service = _ServiceStub(error=TraceImportError('Trace trace-1 request matches 0 dataset samples'))
    with TestClient(create_app(service=cast(Any, error_service))) as client:
      response = client.post('/evaluations/import', json=body)
      empty_response = client.post('/evaluations/import', json={**body, 'traces': []})

    self.assertEqual(response.status_code, 400)
    self.assertEqual(response.json(), {'detail': 'Trace trace-1 request matches 0 dataset samples'})
    self.assertEqual(empty_response.status_code, 422)

  def test_post_repeat_returns_202_with_source_run_identifier(self) -> None:
    source_run_id = uuid4()
    run = self._build_run()
    run.source_run_id = source_run_id
    service = _ServiceStub(started_evaluation=_EvaluationHandleStub(run))
    app = create_app(service=cast(Any, service))
    scheduled_tasks: list[_FakeTask] = []

    with (
      patch('syllo_eval.API.app._create_evaluation_task', side_effect=self._build_task_factory(scheduled_tasks)),
      TestClient(app) as client,
    ):
      response = client.post(
        f'/evaluations/{source_run_id}/repeat',
        json={'metrics': ['llm_calls']},
      )

    self.assertEqual(response.status_code, 202)
    self.assertEqual(response.json()['evaluation_run_id'], str(run.id))
    self.assertEqual(response.json()['source_run_id'], str(source_run_id))
    self.assertEqual(service.repeat_calls, [{'source_run_id': source_run_id, 'metrics': ['llm_calls']}])
    self.assertEqual(len(scheduled_tasks), 1)

  def test_post_repeat_returns_409_when_repeat_is_not_allowed(self) -> None:
    source_run_id = uuid4()
    service = _ServiceStub(error=RepeatNotAllowedError('Source run is still running.'))
    app = create_app(service=cast(Any, service))

    with TestClient(app) as client:
      response = client.post(f'/evaluations/{source_run_id}/repeat', json={})

    self.assertEqual(response.status_code, 409)
    self.assertEqual(response.json(), {'detail': 'Source run is still running.'})

  def test_get_evaluations_returns_paginated_runs(self) -> None:
    completed_run = EvaluationRun(
      id=uuid4(),
      agent_id=uuid4(),
      dataset_id=uuid4(),
      status=EvaluationStatus.COMPLETED,
      start_time=datetime.now(tz=timezone.utc),
      end_time=datetime.now(tz=timezone.utc),
    )
    running_run = self._build_run()
    service = _ServiceStub(runs=[completed_run, running_run])
    app = create_app(service=cast(Any, service))

    with TestClient(app) as client:
      response = client.get('/evaluations?limit=2&offset=10')

    self.assertEqual(response.status_code, 200)
    self.assertEqual(response.json()['total'], 2)
    self.assertEqual(response.json()['limit'], 2)
    self.assertEqual(response.json()['offset'], 10)
    self.assertEqual(
      [
        {
          'evaluation_run_id': str(completed_run.id),
          'status': EvaluationStatus.COMPLETED.value,
        },
        {
          'evaluation_run_id': str(running_run.id),
          'status': EvaluationStatus.RUNNING.value,
        },
      ],
      [
        {
          'evaluation_run_id': item['evaluation_run_id'],
          'status': item['status'],
        }
        for item in response.json()['items']
      ],
    )
    self.assertEqual(service.list_calls, [{'limit': 2, 'offset': 10, 'status': None}])

  def test_get_evaluations_filters_by_status(self) -> None:
    run = self._build_run()
    service = _ServiceStub(runs=[run])
    app = create_app(service=cast(Any, service))

    with TestClient(app) as client:
      response = client.get('/evaluations?status=RUNNING')

    self.assertEqual(response.status_code, 200)
    self.assertEqual(service.list_calls, [{'limit': 50, 'offset': 0, 'status': EvaluationStatus.RUNNING}])

  def test_openapi_spec_version_is_1_3(self) -> None:
    app = create_app(service=cast(Any, _ServiceStub()))

    with TestClient(app) as client:
      response = client.get('/openapi.json')

    self.assertEqual(response.status_code, 200)
    self.assertEqual(response.json()['info']['version'], '1.3.0')

  def test_startup_fails_orphaned_running_evaluations(self) -> None:
    service = _ServiceStub()
    app = create_app(service=cast(Any, service))

    with TestClient(app):
      pass

    self.assertEqual(service.orphan_cleanup_calls, 1)

  def test_injected_service_provides_dataset_manager(self) -> None:
    service = _ServiceStub()
    app = create_app(service=cast(Any, service))

    with TestClient(app):
      self.assertIs(app.state.dataset_service._db_manager, service.db_manager)

  def test_get_evaluation_status_returns_running_run(self) -> None:
    run = self._build_run()
    service = _ServiceStub(
      run=run,
      status_snapshot=SimpleNamespace(
        run=run,
        sample_counts=SimpleNamespace(total=5, pending=2, running=1, completed=1, failed=1, unprocessed=0),
      ),
    )
    app = create_app(service=cast(Any, service))

    with TestClient(app) as client:
      response = client.get(f'/evaluations/{run.id}')

    self.assertEqual(response.status_code, 200)
    self.assertEqual(
      response.json(),
      {
        'evaluation_run_id': str(run.id),
        'status': EvaluationStatus.RUNNING.value,
        'sample_counts': {
          'unprocessed': 0,
          'total': 5,
          'pending': 2,
          'running': 1,
          'completed': 1,
          'failed': 1,
        },
      },
    )
    self.assertEqual(service.get_calls, [run.id])

  def test_get_evaluation_status_returns_404_when_run_does_not_exist(self) -> None:
    missing_run_id = uuid4()
    service = _ServiceStub(error=NotFoundError('evaluation_run', missing_run_id))
    app = create_app(service=cast(Any, service))

    with TestClient(app) as client:
      response = client.get(f'/evaluations/{missing_run_id}')

    self.assertEqual(response.status_code, 404)
    self.assertEqual(response.json(), {'detail': f'Evaluation run {missing_run_id} not found'})

  def test_get_evaluation_report_returns_report(self) -> None:
    run = self._build_run()
    report = self._build_report(run)
    service = _ServiceStub(run=run, report=report)
    app = create_app(service=cast(Any, service))

    with TestClient(app) as client:
      response = client.get(f'/evaluations/{run.id}/report')

    self.assertEqual(response.status_code, 200)
    self.assertEqual(response.json()['run']['id'], str(run.id))
    self.assertEqual(response.json()['observed_metrics'], ['answer_correctness_judge'])
    self.assertEqual(service.report_calls, [run.id])

  def test_get_evaluation_report_returns_404_when_run_does_not_exist(self) -> None:
    missing_run_id = uuid4()
    service = _ServiceStub(error=NotFoundError('evaluation_run', missing_run_id))
    app = create_app(service=cast(Any, service))

    with TestClient(app) as client:
      response = client.get(f'/evaluations/{missing_run_id}/report')

    self.assertEqual(response.status_code, 404)
    self.assertEqual(response.json(), {'detail': f'Evaluation run {missing_run_id} not found'})

  def test_get_evaluation_report_returns_409_when_run_is_still_running(self) -> None:
    run = self._build_run()
    service = _ServiceStub(
      run=run,
      error=EvaluationReportNotAvailableError('Evaluation run is still running'),
    )
    app = create_app(service=cast(Any, service))

    with TestClient(app) as client:
      response = client.get(f'/evaluations/{run.id}/report')

    self.assertEqual(response.status_code, 409)
    self.assertEqual(
      response.json(),
      {'detail': 'Evaluation run is still running; report is only available when it reaches a terminal status.'},
    )

  def test_post_cancel_marks_running_evaluation_failed_and_cancels_task(self) -> None:
    running_run = EvaluationRun(
      id=uuid4(),
      agent_id=uuid4(),
      dataset_id=uuid4(),
      status=EvaluationStatus.RUNNING,
      start_time=datetime.now(tz=timezone.utc),
    )
    failed_run = EvaluationRun(
      id=uuid4(),
      agent_id=uuid4(),
      dataset_id=uuid4(),
      status=EvaluationStatus.FAILED,
      start_time=datetime.now(tz=timezone.utc),
      end_time=datetime.now(tz=timezone.utc),
    )
    failed_run.id = running_run.id
    failed_run.agent_id = running_run.agent_id
    failed_run.dataset_id = running_run.dataset_id
    failed_run.start_time = running_run.start_time
    service = _ServiceStub(run=running_run, failed_run=failed_run)
    app = create_app(service=cast(Any, service))
    fake_task = _FakeTask()

    with TestClient(app) as client:
      app.state.evaluation_tasks[running_run.id] = fake_task
      response = client.post(f'/evaluations/{running_run.id}/cancel')
      self.assertEqual(fake_task.cancel_calls, 1)

    self.assertEqual(response.status_code, 200)
    self.assertEqual(
      response.json(),
      {
        'evaluation_run_id': str(running_run.id),
        'status': EvaluationStatus.FAILED.value,
      },
    )
    self.assertEqual(service.cancel_calls, [running_run.id])

  def test_post_cancel_does_not_cancel_completed_task(self) -> None:
    run = EvaluationRun(
      id=uuid4(),
      agent_id=uuid4(),
      dataset_id=uuid4(),
      status=EvaluationStatus.COMPLETED,
      start_time=datetime.now(tz=timezone.utc),
      end_time=datetime.now(tz=timezone.utc),
    )
    service = _ServiceStub(run=run)
    app = create_app(service=cast(Any, service))
    fake_task = _FakeTask()

    with TestClient(app) as client:
      app.state.evaluation_tasks[run.id] = fake_task
      response = client.post(f'/evaluations/{run.id}/cancel')
      self.assertEqual(fake_task.cancel_calls, 0)

    self.assertEqual(response.status_code, 200)
    self.assertEqual(
      response.json(),
      {
        'evaluation_run_id': str(run.id),
        'status': EvaluationStatus.COMPLETED.value,
      },
    )

  def test_post_datasets_returns_201_with_import_summary(self) -> None:
    dataset = Dataset(id=uuid4(), name='demo')
    dataset_service = _DatasetServiceStub(
      import_summary=DatasetImportSummary(dataset=dataset, sample_count=2, ground_truth_count=13)
    )
    app = create_app(service=cast(Any, _ServiceStub()), dataset_service=cast(Any, dataset_service))

    with TestClient(app) as client:
      response = client.post(
        '/datasets',
        json={
          'name': 'demo',
          'samples': [
            {'input_prompt': 'q1', 'ground_truth_output': 'a1'},
            {'input_prompt': 'q2', 'snippet_ids': ['s1'], 'document_ids': ['d1']},
          ],
        },
      )

    self.assertEqual(response.status_code, 201)
    self.assertEqual(
      response.json(),
      {
        'dataset_id': str(dataset.id),
        'name': 'demo',
        'sample_count': 2,
        'ground_truth_count': 13,
      },
    )
    self.assertEqual(len(dataset_service.import_calls), 1)
    self.assertEqual(dataset_service.import_calls[0]['name'], 'demo')
    payload = dataset_service.import_calls[0]['payload']
    self.assertEqual([sample.input_prompt for sample in payload.samples], ['q1', 'q2'])  # type: ignore[attr-defined]

  def test_post_datasets_returns_409_for_duplicate_name(self) -> None:
    dataset_service = _DatasetServiceStub(error=DatasetAlreadyExistsError("Dataset 'demo' already exists."))
    app = create_app(service=cast(Any, _ServiceStub()), dataset_service=cast(Any, dataset_service))

    with TestClient(app) as client:
      response = client.post('/datasets', json={'name': 'demo', 'samples': [{'input_prompt': 'q1'}]})

    self.assertEqual(response.status_code, 409)
    self.assertEqual(response.json(), {'detail': "Dataset 'demo' already exists."})

  def test_post_datasets_returns_422_for_empty_samples(self) -> None:
    dataset_service = _DatasetServiceStub()
    app = create_app(service=cast(Any, _ServiceStub()), dataset_service=cast(Any, dataset_service))

    with TestClient(app) as client:
      response = client.post('/datasets', json={'name': 'demo', 'samples': []})

    self.assertEqual(response.status_code, 422)
    self.assertEqual(dataset_service.import_calls, [])

  def test_get_datasets_returns_paginated_summaries(self) -> None:
    created_at = datetime.now(tz=timezone.utc)
    summary = DatasetSummary(id=uuid4(), name='demo', created_at=created_at, sample_count=3)
    dataset_service = _DatasetServiceStub(list_page=DatasetListPage(items=[summary], total=4, limit=2, offset=1))
    app = create_app(service=cast(Any, _ServiceStub()), dataset_service=cast(Any, dataset_service))

    with TestClient(app) as client:
      response = client.get('/datasets?limit=2&offset=1')

    self.assertEqual(response.status_code, 200)
    body = response.json()
    self.assertEqual(body['total'], 4)
    self.assertEqual(body['limit'], 2)
    self.assertEqual(body['offset'], 1)
    self.assertEqual(
      body['items'],
      [
        {
          'dataset_id': str(summary.id),
          'name': 'demo',
          'created_at': created_at.isoformat().replace('+00:00', 'Z'),
          'sample_count': 3,
        }
      ],
    )
    self.assertEqual(dataset_service.list_calls, [{'limit': 2, 'offset': 1}])

  @staticmethod
  def _build_report(run: EvaluationRun) -> dict[str, Any]:
    generated_at = datetime.now(tz=timezone.utc)
    empty_token_usage = {
      'agent': {
        'input_tokens': 0,
        'output_tokens': 0,
        'total_tokens': 0,
        'sources_with_data': 0,
        'sources_total': 0,
      },
      'judge': {
        'input_tokens': 0,
        'output_tokens': 0,
        'total_tokens': 0,
        'sources_with_data': 0,
        'sources_total': 0,
      },
    }
    return {
      'report_version': '1.2',
      'generated_at': generated_at,
      'run': {
        'id': run.id,
        'status': EvaluationStatus.COMPLETED,
        'start_time': run.start_time,
        'end_time': generated_at,
        'duration_seconds': 1.0,
        'samples_processed': 1,
      },
      'agent': {'id': run.agent_id, 'name': 'demo-agent', 'version_tag': 'v-test'},
      'dataset': {'id': run.dataset_id, 'name': 'dataset', 'total_samples': 1},
      'observed_metrics': ['answer_correctness_judge'],
      'summary': {
        'samples': {'total': 1, 'succeeded': 1, 'failed': 0, 'success_rate': 1.0},
        'latency_seconds': {
          'mean': 1.0,
          'median': 1.0,
          'p95': 1.0,
          'min': 1.0,
          'max': 1.0,
          'total_wall_time': 1.0,
        },
        'metric_computations': {'total': 1, 'completed': 1, 'failed': 0, 'skipped': 0},
        'token_usage': empty_token_usage,
      },
      'metrics': [
        {
          'name': 'answer_correctness_judge',
          'description': 'Answer judge',
          'requires_ground_truth': True,
          'coverage': {
            'samples_total': 1,
            'computations_total': 1,
            'completed': 1,
            'failed': 0,
            'skipped': 0,
            'coverage_rate': 1.0,
            'skipped_reasons': {},
          },
          'scores': {
            'count': 1,
            'mean': 0.9,
            'stddev': None,
            'min': 0.9,
            'p25': 0.9,
            'median': 0.9,
            'p75': 0.9,
            'max': 0.9,
          },
        }
      ],
      'failures': {
        'samples': {
          'by_phase': {'agent_call': 0, 'trace_fetch': 0, 'metric_compute': 0, 'unknown': 0},
          'items': [],
        },
        'metric_computations': {'by_metric': {}, 'failed_reasons': {}, 'skipped_reasons': {}},
      },
      'samples': [
        {
          'evaluation_run_sample_id': uuid4(),
          'sample_id': uuid4(),
          'trace_id': 'trace-1',
          'status': 'COMPLETED',
          'started_at': generated_at,
          'ended_at': generated_at,
          'duration_seconds': 1.0,
          'error_message': None,
          'metrics': {
            'answer_correctness_judge': {
              'value': 0.9,
              'computations_total': 1,
              'completed': 1,
              'failed': 0,
              'skipped': 0,
            }
          },
          'token_usage': empty_token_usage,
        }
      ],
    }


class TestEvaluationTaskLifecycle(unittest.IsolatedAsyncioTestCase):
  async def test_drain_cancels_and_awaits_in_flight_tasks(self) -> None:
    started = asyncio.Event()

    async def _never_finishes() -> None:
      started.set()
      await asyncio.Event().wait()

    task = asyncio.create_task(_never_finishes())
    await started.wait()
    app = cast(Any, SimpleNamespace(state=SimpleNamespace(evaluation_tasks={uuid4(): task})))

    await _drain_evaluation_tasks(app)

    self.assertTrue(task.cancelled())


if __name__ == '__main__':
  unittest.main()
