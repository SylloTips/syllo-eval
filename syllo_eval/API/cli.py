import argparse
import asyncio
import json
import logging
import sys
from collections.abc import Sequence
from pathlib import Path
from uuid import UUID

from syllo_eval.service import (
  EvaluationService,
  MetricSelectionError,
  ServiceFactory,
  build_db_manager,
  resolve_selected_metric_names_csv,
  settings_metric_names,
)
from syllo_eval.datasets import (
  DatasetAlreadyExistsError,
  DatasetImportSummary,
  DatasetJsonPayload,
  DatasetListPage,
  DatasetService,
  load_dataset_json,
)
from syllo_eval.logging_utils import LoggingSettings, configure_logging
from syllo_eval.model import EvaluationRun, EvaluationStatus
from syllo_eval.settings import Settings, load_settings_env

logger = logging.getLogger(__name__)

EXIT_RUN_FAILED = 4
EXIT_UNEXPECTED_STATUS = 5


def _exit_code_for_status(status: EvaluationStatus) -> int:
  """Map a terminal run status to a process exit code.

  COMPLETED and PARTIALLY_COMPLETED both succeed (0); a partial run only warns.
  """
  if status in (EvaluationStatus.COMPLETED, EvaluationStatus.PARTIALLY_COMPLETED):
    if status is EvaluationStatus.PARTIALLY_COMPLETED:
      logger.warning('Evaluation completed partially; some samples failed. See the report for details.')
    return 0
  if status is EvaluationStatus.FAILED:
    return EXIT_RUN_FAILED
  logger.warning('Evaluation finished with unexpected status %s', status.value)
  return EXIT_UNEXPECTED_STATUS


def _parse_uuid(value: str) -> UUID:
  try:
    return UUID(value)
  except ValueError as err:
    raise argparse.ArgumentTypeError(f'Invalid UUID value: {value}') from err


def _parse_non_empty(value: str) -> str:
  stripped = value.strip()
  if not stripped:
    raise argparse.ArgumentTypeError('Value cannot be empty')
  return stripped


def _parse_existing_file(value: str) -> Path:
  path = Path(value).expanduser().resolve()
  if not path.is_file():
    raise argparse.ArgumentTypeError(f'File not found: {value}')
  return path


def build_parser() -> argparse.ArgumentParser:
  parser = argparse.ArgumentParser(
    prog='syllo-eval',
    description='Run an evaluation for one agent/dataset by calling the agent for each sample.',
  )
  parser.add_argument(
    '--agent-name',
    required=True,
    type=_parse_non_empty,
    help='Agent name to evaluate.',
  )
  parser.add_argument(
    '--agent-version-tag',
    required=True,
    type=_parse_non_empty,
    help='Agent version tag to evaluate.',
  )
  parser.add_argument(
    '--dataset-id',
    required=True,
    type=_parse_uuid,
    help='Dataset UUID to evaluate.',
  )
  parser.add_argument(
    '--sample-trace-timeout-seconds',
    type=float,
    default=None,
    help='Optional timeout in seconds for trace ingestion per sample.',
  )
  parser.add_argument(
    '--sample-compute-timeout-seconds',
    type=float,
    default=None,
    help='Optional timeout in seconds for metric computation per sample.',
  )
  parser.add_argument(
    '--max-concurrent-tasks',
    type=int,
    default=None,
    help='Maximum number of concurrent metric computations per sample.',
  )
  parser.add_argument(
    '--log-level',
    choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
    default=None,
    help='Logging level.',
  )
  parser.add_argument(
    '--metrics',
    help=(
      'Optional comma-separated list of metric names to run. '
      'Defaults to all metrics available in the current environment.'
    ),
  )
  parser.add_argument(
    '--report-path',
    type=str,
    default=None,
    help='If set, write the JSON evaluation report to this path after the run finishes.',
  )
  return parser


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
  return build_parser().parse_args(argv)


async def _run(
  service: EvaluationService,
  agent_name: str,
  agent_version_tag: str,
  dataset_id: UUID,
  max_concurrent_tasks: int | None,
  sample_trace_timeout: float | None,
  sample_compute_timeout: float | None,
  selected_metric_names: Sequence[str],
  report_path: Path | None,
) -> EvaluationRun:
  try:
    await service.initialize()
    run = await service.run_evaluation(
      agent_name=agent_name,
      agent_version_tag=agent_version_tag,
      dataset_id=dataset_id,
      max_concurrent_tasks=max_concurrent_tasks,
      sample_trace_timeout=sample_trace_timeout,
      sample_compute_timeout=sample_compute_timeout,
      selected_metric_names=selected_metric_names,
    )
    if report_path is not None:
      report = await service.get_evaluation_report(run.id)
      report_path.parent.mkdir(parents=True, exist_ok=True)
      report_path.write_text(report.model_dump_json(indent=2))
      logger.info('Evaluation report written to %s', report_path)
    return run
  finally:
    await service.close()


async def _repeat(
  service: EvaluationService,
  source_run_id: UUID,
  max_concurrent_tasks: int | None,
  sample_compute_timeout: float | None,
  selected_metric_names: Sequence[str] | None,
) -> EvaluationRun:
  try:
    await service.initialize()
    started = await service.repeat_evaluation(
      source_run_id=source_run_id,
      metrics=selected_metric_names,
      max_concurrent_tasks=max_concurrent_tasks,
      sample_compute_timeout=sample_compute_timeout,
    )
    return await started.execute()
  finally:
    await service.close()


async def _import_dataset(service: DatasetService, *, name: str, payload: DatasetJsonPayload) -> DatasetImportSummary:
  try:
    await service.initialize()
    return await service.import_dataset(name=name, payload=payload)
  finally:
    await service.close()


async def _list_datasets(service: DatasetService, *, limit: int, offset: int) -> DatasetListPage:
  try:
    await service.initialize()
    return await service.list_datasets(limit=limit, offset=offset)
  finally:
    await service.close()


def build_dataset_parser() -> argparse.ArgumentParser:
  parser = argparse.ArgumentParser(prog='syllo-eval dataset', description='Manage evaluation datasets.')
  subparsers = parser.add_subparsers(dest='command', required=True)

  import_parser = subparsers.add_parser('import', help='Import a dataset JSON file as a new dataset.')
  import_parser.add_argument('dataset_path', type=_parse_existing_file, help='Path to the dataset JSON file.')
  import_parser.add_argument(
    '--name',
    type=_parse_non_empty,
    default=None,
    help='Optional dataset name. Defaults to the JSON filename stem.',
  )
  import_parser.add_argument(
    '--log-level',
    choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
    default=None,
    help='Logging level.',
  )

  list_parser = subparsers.add_parser('list', help='List registered datasets.')
  list_parser.add_argument('--limit', type=int, default=50, help='Maximum number of datasets to return.')
  list_parser.add_argument('--offset', type=int, default=0, help='Number of datasets to skip.')
  list_parser.add_argument(
    '--log-level',
    choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
    default=None,
    help='Logging level.',
  )
  return parser


def build_repeat_parser() -> argparse.ArgumentParser:
  parser = argparse.ArgumentParser(
    prog='syllo-eval repeat',
    description='Repeat an evaluation by recomputing metrics over stored traces.',
  )
  parser.add_argument('source_run_id', type=_parse_uuid, help='Source evaluation run UUID.')
  parser.add_argument(
    '--metrics',
    help='Optional comma-separated metrics to recompute (any current-runtime metric). '
    'Defaults to the source run metric snapshot.',
  )
  parser.add_argument(
    '--sample-compute-timeout-seconds',
    type=float,
    default=None,
    help='Optional timeout in seconds for metric recomputation per sample.',
  )
  parser.add_argument(
    '--max-concurrent-tasks',
    type=int,
    default=None,
    help='Maximum number of concurrent metric computations per sample.',
  )
  parser.add_argument(
    '--log-level',
    choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
    default=None,
    help='Logging level.',
  )
  return parser


def main(argv: Sequence[str] | None = None, service_factory: ServiceFactory = EvaluationService) -> int:
  load_settings_env(override=False)
  raw_argv = list(argv) if argv is not None else sys.argv[1:]
  if raw_argv[:1] == ['repeat']:
    return _main_repeat(raw_argv[1:], service_factory)
  if raw_argv[:1] == ['dataset']:
    return _main_dataset(raw_argv[1:])

  parser = build_parser()
  args = parser.parse_args(raw_argv)
  configure_logging(LoggingSettings.from_env(level_override=args.log_level))

  try:
    settings = Settings()
    selected_metric_names = resolve_selected_metric_names_csv(
      args.metrics,
      settings_metric_names(settings),
    )
    service = service_factory(settings)
  except MetricSelectionError as err:
    parser.error(str(err))
  except Exception:
    logger.exception('Evaluation run failed')
    return 1

  report_path = Path(args.report_path) if args.report_path else None

  try:
    run = asyncio.run(
      _run(
        service=service,
        agent_name=args.agent_name,
        agent_version_tag=args.agent_version_tag,
        dataset_id=args.dataset_id,
        max_concurrent_tasks=args.max_concurrent_tasks,
        sample_trace_timeout=args.sample_trace_timeout_seconds,
        sample_compute_timeout=args.sample_compute_timeout_seconds,
        selected_metric_names=selected_metric_names,
        report_path=report_path,
      )
    )
  except Exception:
    logger.exception('Evaluation run failed')
    return 1

  print(
    json.dumps(
      {
        'evaluation_run_id': str(run.id),
        'status': run.status.value,
        'agent_name': args.agent_name,
        'agent_version_tag': args.agent_version_tag,
        'dataset_id': str(run.dataset_id),
        'start_time': run.start_time.isoformat(),
        'end_time': run.end_time.isoformat() if run.end_time else None,
      }
    )
  )
  return _exit_code_for_status(run.status)


def _main_dataset(argv: Sequence[str]) -> int:
  parser = build_dataset_parser()
  args = parser.parse_args(argv)
  configure_logging(LoggingSettings.from_env(level_override=args.log_level))

  try:
    settings = Settings()
    service = DatasetService(db_manager=build_db_manager(settings))

    if args.command == 'import':
      payload = load_dataset_json(args.dataset_path)
      dataset_name = args.name or args.dataset_path.stem
      summary = asyncio.run(_import_dataset(service, name=dataset_name, payload=payload))
      print(
        json.dumps(
          {
            'dataset_id': str(summary.dataset.id),
            'dataset_name': summary.dataset.name,
            'sample_count': summary.sample_count,
            'ground_truth_count': summary.ground_truth_count,
          }
        )
      )
      return 0

    page = asyncio.run(_list_datasets(service, limit=args.limit, offset=args.offset))
    print(
      json.dumps(
        {
          'items': [
            {
              'dataset_id': str(item.id),
              'dataset_name': item.name,
              'created_at': item.created_at.isoformat(),
              'sample_count': item.sample_count,
            }
            for item in page.items
          ],
          'total': page.total,
          'limit': page.limit,
          'offset': page.offset,
        }
      )
    )
    return 0
  except DatasetAlreadyExistsError as err:
    logger.error('%s', err)
    return 1
  except Exception:
    logger.exception('Dataset command failed')
    return 1


def _main_repeat(argv: Sequence[str], service_factory: ServiceFactory) -> int:
  parser = build_repeat_parser()
  args = parser.parse_args(argv)
  configure_logging(LoggingSettings.from_env(level_override=args.log_level))

  try:
    settings = Settings()
    selected_metric_names = None if args.metrics is None else [metric.strip() for metric in args.metrics.split(',')]
  except Exception:
    logger.exception('Evaluation repeat failed')
    return 1

  service = service_factory(settings)
  try:
    run = asyncio.run(
      _repeat(
        service=service,
        source_run_id=args.source_run_id,
        max_concurrent_tasks=args.max_concurrent_tasks,
        sample_compute_timeout=args.sample_compute_timeout_seconds,
        selected_metric_names=selected_metric_names,
      )
    )
  except Exception:
    logger.exception('Evaluation repeat failed')
    return 1

  print(
    json.dumps(
      {
        'evaluation_run_id': str(run.id),
        'source_run_id': str(run.source_run_id) if run.source_run_id else None,
        'status': run.status.value,
        'dataset_id': str(run.dataset_id),
        'start_time': run.start_time.isoformat(),
        'end_time': run.end_time.isoformat() if run.end_time else None,
      }
    )
  )
  return _exit_code_for_status(run.status)


if __name__ == '__main__':
  raise SystemExit(main())
