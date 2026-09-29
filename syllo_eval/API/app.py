import asyncio
import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from datetime import datetime
from enum import Enum
from typing import Annotated, Any
from uuid import UUID

from fastapi import FastAPI, HTTPException, Query, status
from pydantic import BaseModel, Field, StringConstraints

from syllo_eval.service import (
  AgentCallerSelectionError,
  EvaluationHandle,
  EvaluationService,
  MetricSelectionError,
  RepeatNotAllowedError,
  ServiceFactory,
)
from syllo_eval.datasets import (
  DatasetAlreadyExistsError,
  DatasetJsonPayload,
  DatasetJsonSample,
  DatasetService,
)
from syllo_eval.evaluation.evaluation_report import EvaluationReport, EvaluationReportNotAvailableError
from syllo_eval.infrastructure.exceptions import NotFoundError
from syllo_eval.logging_utils import LoggingSettings, configure_logging
from syllo_eval.model import EvaluationStatus
from syllo_eval.settings import Settings, load_settings_env

logger = logging.getLogger(__name__)

NonEmptyStr = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]


class StartEvaluationRequest(BaseModel):
  agent_name: NonEmptyStr
  agent_version_tag: NonEmptyStr
  dataset_id: UUID
  metrics: list[str] | None = None


class RepeatEvaluationRequest(BaseModel):
  metrics: list[str] | None = None


class StartEvaluationResponse(BaseModel):
  evaluation_run_id: UUID
  status: 'ApiEvaluationStatus'
  agent_name: str
  agent_version_tag: str
  dataset_id: UUID


class RepeatEvaluationResponse(StartEvaluationResponse):
  source_run_id: UUID


class ApiEvaluationStatus(str, Enum):
  RUNNING = 'RUNNING'
  COMPLETED = 'COMPLETED'
  PARTIALLY_COMPLETED = 'PARTIALLY_COMPLETED'
  FAILED = 'FAILED'


class EvaluationSampleCountsResponse(BaseModel):
  unprocessed: int = 0
  total: int
  pending: int
  running: int
  completed: int
  failed: int


class EvaluationStatusResponse(BaseModel):
  evaluation_run_id: UUID
  status: ApiEvaluationStatus


class EvaluationListItemResponse(EvaluationStatusResponse):
  agent_id: UUID
  dataset_id: UUID
  source_run_id: UUID | None
  start_time: datetime
  end_time: datetime | None


class EvaluationListResponse(BaseModel):
  items: list[EvaluationListItemResponse]
  total: int
  limit: int
  offset: int


class EvaluationRunStatusResponse(EvaluationStatusResponse):
  sample_counts: EvaluationSampleCountsResponse


class CreateDatasetRequest(BaseModel):
  name: NonEmptyStr
  samples: list[DatasetJsonSample] = Field(min_length=1)


class CreateDatasetResponse(BaseModel):
  dataset_id: UUID
  name: str
  sample_count: int
  ground_truth_count: int


class DatasetListItemResponse(BaseModel):
  dataset_id: UUID
  name: str
  created_at: datetime
  sample_count: int


class DatasetListResponse(BaseModel):
  items: list[DatasetListItemResponse]
  total: int
  limit: int
  offset: int


class ErrorResponse(BaseModel):
  detail: str


NOT_FOUND_RESPONSE: dict[int | str, dict[str, Any]] = {
  status.HTTP_404_NOT_FOUND: {
    'model': ErrorResponse,
    'description': 'Evaluation run not found.',
  }
}

METRIC_SELECTION_ERROR_RESPONSE: dict[int | str, dict[str, Any]] = {
  status.HTTP_400_BAD_REQUEST: {
    'model': ErrorResponse,
    'description': 'Unknown or invalid metric name(s).',
  }
}

REPEAT_ERROR_RESPONSES: dict[int | str, dict[str, Any]] = {
  **NOT_FOUND_RESPONSE,
  **METRIC_SELECTION_ERROR_RESPONSE,
  status.HTTP_409_CONFLICT: {
    'model': ErrorResponse,
    'description': 'The source run cannot be repeated.',
  },
}

REPORT_ERROR_RESPONSES: dict[int | str, dict[str, Any]] = {
  **NOT_FOUND_RESPONSE,
  status.HTTP_409_CONFLICT: {
    'model': ErrorResponse,
    'description': 'Evaluation run is still running.',
  },
}

DATASET_CONFLICT_RESPONSE: dict[int | str, dict[str, Any]] = {
  status.HTTP_409_CONFLICT: {
    'model': ErrorResponse,
    'description': 'A dataset with the same name already exists.',
  }
}


def create_app(
  service: EvaluationService | None = None,
  dataset_service: DatasetService | None = None,
  service_factory: ServiceFactory = EvaluationService,
) -> FastAPI:
  @asynccontextmanager
  async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    app.state.evaluation_tasks = {}

    if service is not None:
      app.state.evaluation_service = service
      app.state.dataset_service = dataset_service or DatasetService(db_manager=service.db_manager)
      await _fail_orphaned_running_evaluations(service)
      try:
        yield
      finally:
        await _drain_evaluation_tasks(app)
      return

    load_settings_env(override=False)
    configure_logging(LoggingSettings.from_env())
    runtime_service = service_factory(Settings())
    try:
      await runtime_service.initialize()
      app.state.evaluation_service = runtime_service
      app.state.dataset_service = dataset_service or DatasetService(db_manager=runtime_service.db_manager)
      await _fail_orphaned_running_evaluations(runtime_service)
      yield
    finally:
      await _drain_evaluation_tasks(app)
      await runtime_service.close()

  app = FastAPI(
    title='syllo-eval',
    description='Manage datasets, start evaluation runs, list their status, poll reports, and cancel running runs.',
    version='1.2.0',
    lifespan=lifespan,
    openapi_tags=[
      {
        'name': 'evaluations',
        'description': 'Operations for listing, starting, checking, repeating, and cancelling evaluation runs.',
      },
      {
        'name': 'datasets',
        'description': 'Operations for importing and listing datasets.',
      },
    ],
  )

  @app.get(
    '/evaluations',
    response_model=EvaluationListResponse,
    tags=['evaluations'],
    summary='List Evaluations',
    description='Returns evaluation runs in reverse chronological order.',
  )
  async def list_evaluations(
    limit: Annotated[int, Query(ge=1, le=200)] = 50,
    offset: Annotated[int, Query(ge=0)] = 0,
    evaluation_status: Annotated[ApiEvaluationStatus | None, Query(alias='status')] = None,
  ) -> EvaluationListResponse:
    requested_status = EvaluationStatus(evaluation_status.value) if evaluation_status is not None else None
    page = await app.state.evaluation_service.list_evaluations(
      limit=limit,
      offset=offset,
      status=requested_status,
    )

    return EvaluationListResponse(
      items=[
        EvaluationListItemResponse(
          evaluation_run_id=run.id,
          status=_project_status(run.status),
          agent_id=run.agent_id,
          dataset_id=run.dataset_id,
          source_run_id=run.source_run_id,
          start_time=run.start_time,
          end_time=run.end_time,
        )
        for run in page.runs
      ],
      total=page.total,
      limit=page.limit,
      offset=page.offset,
    )

  @app.post(
    '/evaluations',
    response_model=StartEvaluationResponse,
    status_code=status.HTTP_202_ACCEPTED,
    responses=METRIC_SELECTION_ERROR_RESPONSE,
    tags=['evaluations'],
    summary='Start Evaluation',
    description='Creates an evaluation run and executes it asynchronously in the background.',
  )
  async def start_evaluation(request: StartEvaluationRequest) -> StartEvaluationResponse:
    evaluation_service = app.state.evaluation_service

    try:
      started = await evaluation_service.create_evaluation(
        agent_name=request.agent_name,
        agent_version_tag=request.agent_version_tag,
        dataset_id=request.dataset_id,
        selected_metric_names=request.metrics,
      )
    except (MetricSelectionError, AgentCallerSelectionError) as err:
      raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(err)) from err

    _register_evaluation_task(app, started.run.id, _create_evaluation_task(started))

    return StartEvaluationResponse(
      evaluation_run_id=started.run.id,
      status=_project_status(started.run.status),
      agent_name=request.agent_name,
      agent_version_tag=request.agent_version_tag,
      dataset_id=request.dataset_id,
    )

  @app.post(
    '/evaluations/{evaluation_run_id}/repeat',
    response_model=RepeatEvaluationResponse,
    status_code=status.HTTP_202_ACCEPTED,
    responses=REPEAT_ERROR_RESPONSES,
    tags=['evaluations'],
    summary='Repeat Evaluation',
    description='Creates a new run that recomputes metrics over stored traces from a terminal source run.',
  )
  async def repeat_evaluation(evaluation_run_id: UUID, request: RepeatEvaluationRequest) -> RepeatEvaluationResponse:
    evaluation_service = app.state.evaluation_service

    try:
      started = await evaluation_service.repeat_evaluation(
        source_run_id=evaluation_run_id,
        metrics=request.metrics,
      )
    except NotFoundError as err:
      raise _build_not_found_http_exception(evaluation_run_id) from err
    except MetricSelectionError as err:
      raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(err)) from err
    except RepeatNotAllowedError as err:
      raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(err)) from err

    _register_evaluation_task(app, started.run.id, _create_evaluation_task(started))

    return RepeatEvaluationResponse(
      evaluation_run_id=started.run.id,
      status=_project_status(started.run.status),
      source_run_id=evaluation_run_id,
      agent_name=started.agent_name or '',
      agent_version_tag=started.agent_version_tag or '',
      dataset_id=started.run.dataset_id,
    )

  @app.get(
    '/evaluations/{evaluation_run_id}',
    response_model=EvaluationRunStatusResponse,
    responses=NOT_FOUND_RESPONSE,
    tags=['evaluations'],
    summary='Get Evaluation Status',
    description='Returns the current status for an evaluation run.',
  )
  async def get_evaluation_status(evaluation_run_id: UUID) -> EvaluationRunStatusResponse:
    try:
      snapshot = await app.state.evaluation_service.get_evaluation_status(evaluation_run_id)
    except NotFoundError as err:
      raise _build_not_found_http_exception(evaluation_run_id) from err

    return EvaluationRunStatusResponse(
      evaluation_run_id=snapshot.run.id,
      status=_project_status(snapshot.run.status),
      sample_counts=EvaluationSampleCountsResponse(
        total=snapshot.sample_counts.total,
        unprocessed=snapshot.sample_counts.unprocessed,
        pending=snapshot.sample_counts.pending,
        running=snapshot.sample_counts.running,
        completed=snapshot.sample_counts.completed,
        failed=snapshot.sample_counts.failed,
      ),
    )

  @app.get(
    '/evaluations/{evaluation_run_id}/report',
    response_model=EvaluationReport,
    responses=REPORT_ERROR_RESPONSES,
    tags=['evaluations'],
    summary='Get Evaluation Report',
    description='Returns an aggregated report for a terminal evaluation run.',
  )
  async def get_evaluation_report(evaluation_run_id: UUID) -> EvaluationReport:
    try:
      return await app.state.evaluation_service.get_evaluation_report(evaluation_run_id)
    except NotFoundError as err:
      raise _build_not_found_http_exception(evaluation_run_id) from err
    except EvaluationReportNotAvailableError as err:
      raise HTTPException(
        status_code=status.HTTP_409_CONFLICT,
        detail='Evaluation run is still running; report is only available when it reaches a terminal status.',
      ) from err

  @app.post(
    '/datasets',
    response_model=CreateDatasetResponse,
    status_code=status.HTTP_201_CREATED,
    responses=DATASET_CONFLICT_RESPONSE,
    tags=['datasets'],
    summary='Create Dataset',
    description='Imports a dataset payload as a new dataset with its samples and derived ground truths.',
  )
  async def create_dataset(request: CreateDatasetRequest) -> CreateDatasetResponse:
    try:
      summary = await app.state.dataset_service.import_dataset(
        name=request.name,
        payload=DatasetJsonPayload(samples=request.samples),
      )
    except DatasetAlreadyExistsError as err:
      raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(err)) from err

    return CreateDatasetResponse(
      dataset_id=summary.dataset.id,
      name=summary.dataset.name,
      sample_count=summary.sample_count,
      ground_truth_count=summary.ground_truth_count,
    )

  @app.get(
    '/datasets',
    response_model=DatasetListResponse,
    tags=['datasets'],
    summary='List Datasets',
    description='Returns registered datasets in reverse chronological order with their sample counts.',
  )
  async def list_datasets(
    limit: Annotated[int, Query(ge=1, le=200)] = 50,
    offset: Annotated[int, Query(ge=0)] = 0,
  ) -> DatasetListResponse:
    page = await app.state.dataset_service.list_datasets(limit=limit, offset=offset)

    return DatasetListResponse(
      items=[
        DatasetListItemResponse(
          dataset_id=item.id,
          name=item.name,
          created_at=item.created_at,
          sample_count=item.sample_count,
        )
        for item in page.items
      ],
      total=page.total,
      limit=page.limit,
      offset=page.offset,
    )

  @app.post(
    '/evaluations/{evaluation_run_id}/cancel',
    response_model=EvaluationStatusResponse,
    responses=NOT_FOUND_RESPONSE,
    tags=['evaluations'],
    summary='Cancel Evaluation',
    description='Best-effort cancellation for a running evaluation. Cancelled runs are surfaced as FAILED.',
  )
  async def cancel_evaluation(evaluation_run_id: UUID) -> EvaluationStatusResponse:
    evaluation_service = app.state.evaluation_service

    try:
      run = await evaluation_service.get_evaluation_run(evaluation_run_id)
    except NotFoundError as err:
      raise _build_not_found_http_exception(evaluation_run_id) from err

    task = app.state.evaluation_tasks.get(evaluation_run_id)
    if run.status == EvaluationStatus.RUNNING and task is not None and not task.done():
      task.cancel()

    try:
      run = await evaluation_service.fail_evaluation_run(evaluation_run_id)
    except NotFoundError as err:
      raise _build_not_found_http_exception(evaluation_run_id) from err

    return EvaluationStatusResponse(
      evaluation_run_id=run.id,
      status=_project_status(run.status),
    )

  return app


def _build_not_found_http_exception(evaluation_run_id: UUID) -> HTTPException:
  return HTTPException(
    status_code=status.HTTP_404_NOT_FOUND,
    detail=f'Evaluation run {evaluation_run_id} not found',
  )


def _project_status(status_value: EvaluationStatus) -> ApiEvaluationStatus:
  return ApiEvaluationStatus(status_value.value)


def _register_evaluation_task(app: FastAPI, evaluation_run_id: UUID, task: asyncio.Task) -> None:
  app.state.evaluation_tasks[evaluation_run_id] = task

  def _cleanup(_task: asyncio.Task) -> None:
    app.state.evaluation_tasks.pop(evaluation_run_id, None)

  task.add_done_callback(_cleanup)
  task.add_done_callback(_log_background_task_failure)


def _log_background_task_failure(task: asyncio.Task) -> None:
  try:
    task.result()
  except asyncio.CancelledError:
    logger.warning('Background evaluation task was cancelled')
  except Exception:
    logger.exception('Background evaluation task failed')


def _create_evaluation_task(started: EvaluationHandle) -> asyncio.Task:
  return asyncio.create_task(started.execute())


async def _drain_evaluation_tasks(app: FastAPI) -> None:
  """Stop background evaluation tasks before the pool they hold connections from is closed."""
  tasks = [task for task in app.state.evaluation_tasks.values() if not task.done()]
  for task in tasks:
    task.cancel()
  if tasks:
    await asyncio.gather(*tasks, return_exceptions=True)
    logger.warning('Cancelled %s in-flight evaluation task(s) during shutdown', len(tasks))


async def _fail_orphaned_running_evaluations(service: EvaluationService) -> None:
  failed_count = await service.fail_orphaned_running_evaluations()
  if failed_count:
    logger.warning('Failed %s orphaned running evaluation(s) during startup', failed_count)


app = create_app()
