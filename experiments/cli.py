"""Command-line entry point: ``poetry run syllo-exp <command>``. Run it from the ``experiments/`` folder."""

import argparse
import asyncio
import sys
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any
from uuid import UUID

import httpx
import psycopg

from syllo_eval.evaluation.judge.failures import JudgeFailureKind
from syllo_eval.infrastructure import DatabaseManager
from syllo_eval.infrastructure.exceptions import ConnectionError as DatabaseConnectionError
from syllo_eval.infrastructure.exceptions import PersistenceError
from syllo_eval.infrastructure.unit_of_work import UnitOfWork
from syllo_eval.model import EvaluationStatus, MetricComputationStatus
from syllo_eval.settings import GeminiJudgeSettings, Settings

from benchmarks import steps as benchmark_steps
from benchmarks.download import FetchInProgressError, PinnedFileMismatchError
from config import CONFIG_DIR, DATA_DIR, BenchmarkSource, load_config, load_environment
from deepeval_baseline import DEEPEVAL_VERSION, METRIC_KEYS, deepeval_metrics, open_deepeval_judge
from manifest import Manifest, StepStatus
from service import build_service, open_database

DEFAULT_MANIFEST = Path(__file__).resolve().parent / 'outputs' / 'manifest.jsonl'
# Failures that are outcomes of the judge: the unit counts as wrong (METHODOLOGY.md, Failures). Any other failed unit,
# such as one with a provider error, is unscored and must run again, so a judge step that has one is not complete.
_JUDGE_OUTCOMES = frozenset(
  kind.value
  for kind in (
    JudgeFailureKind.MISALIGNED,
    JudgeFailureKind.TRUNCATED,
    JudgeFailureKind.INVALID_OUTPUT,
    JudgeFailureKind.CONTEXT_OVERFLOW,
    JudgeFailureKind.TIMEOUT,
  )
)
_BENCHMARK_STEPS = {
  'fetch': 'Download the pinned benchmark files, verifying their size and SHA-256.',
  'convert': 'Convert the pinned files into syllo-eval datasets and check them.',
  'import': 'Import the converted datasets and their claim ground truths into the database.',
}


def build_parser() -> argparse.ArgumentParser:
  parser = argparse.ArgumentParser(prog='syllo-exp', description='Reproduce the experiments of the paper.')
  commands = parser.add_subparsers(dest='command', required=True)

  configurations = commands.add_parser('configurations', help='Validate the configs and list agent configurations.')
  configurations.add_argument('--config-dir', type=Path, default=CONFIG_DIR, help='Folder with the YAML configs.')

  steps = commands.add_parser('steps', help='Show the current state of every recorded experiment step.')
  steps.add_argument('--manifest', type=Path, default=DEFAULT_MANIFEST, help='Manifest JSONL file.')

  benchmarks = commands.add_parser('benchmarks', help='Fetch, convert and import the pinned benchmarks.')
  benchmark_commands = benchmarks.add_subparsers(dest='step', required=True)
  for step, description in _BENCHMARK_STEPS.items():
    command = benchmark_commands.add_parser(step, help=description, description=description)
    command.add_argument(
      '--only', nargs='+', choices=sorted(benchmark_steps.BUILDERS), help='Benchmarks to process (default: all).'
    )
    command.add_argument('--config-dir', type=Path, default=CONFIG_DIR, help='Folder with the YAML configs.')
    command.add_argument('--data-dir', type=Path, default=DATA_DIR, help='Folder for downloads and converted data.')
    command.add_argument('--manifest', type=Path, default=DEFAULT_MANIFEST, help='Manifest JSONL file.')

  deepeval = commands.add_parser(
    'deepeval',
    help='Score the stored traces of a run with the DeepEval baseline.',
    description='Repeat a run on its stored traces with DeepEval metrics, judged by the shared judge model.',
  )
  deepeval.add_argument('--source-run', type=UUID, required=True, help='Run whose stored traces DeepEval scores.')
  deepeval.add_argument(
    '--metrics', nargs='+', choices=METRIC_KEYS, default=list(METRIC_KEYS), help='Metrics to run (default: all).'
  )
  deepeval.add_argument(
    '--max-concurrent-samples',
    type=_positive_int,
    help='Samples scored at once (default: EVALUATION_MAX_CONCURRENT_SAMPLES, or 1); cost runs use 1.',
  )
  deepeval.add_argument('--config-dir', type=Path, default=CONFIG_DIR, help='Folder with the YAML configs.')
  deepeval.add_argument('--manifest', type=Path, default=DEFAULT_MANIFEST, help='Manifest JSONL file.')
  return parser


def _positive_int(value: str) -> int:
  number = int(value)
  if number < 1:
    raise argparse.ArgumentTypeError(f'must be at least 1, got {number}')
  return number


def main(argv: Sequence[str] | None = None) -> int:
  args = build_parser().parse_args(argv)
  if args.command == 'configurations':
    config = load_config(args.config_dir)
    for configuration in config.configurations:
      model = config.models.agents[configuration.model]
      print(
        f'{configuration.id:<34} agent={configuration.agent:<15} version={configuration.version_tag:<16} {model.label}'
      )
    print(f'{len(config.configurations)} configurations; judge: {config.models.judge.label}')
  elif args.command == 'steps':
    for step, record in sorted(Manifest(args.manifest).latest().items()):
      runs = ', '.join(str(run_id) for run_id in record.run_ids) or '-'
      print(f'{step:<48} {record.status.value:<10} {record.recorded_at:%Y-%m-%d %H:%M} runs: {runs}')
  elif args.command == 'benchmarks':
    return _run_benchmark_step(args)
  elif args.command == 'deepeval':
    return asyncio.run(_score_with_deepeval(args))
  return 0


def _run_benchmark_step(args: argparse.Namespace) -> int:
  """Run one step for each selected benchmark; a failing benchmark is recorded and the others still run."""
  config = load_config(args.config_dir)
  unpinned = sorted(set(args.only or ()) - set(config.benchmarks))
  if unpinned:
    print(f'Not pinned in {args.config_dir / "benchmarks.yaml"}: {", ".join(unpinned)}', file=sys.stderr)
    return 2
  selected = [(name, source) for name, source in config.benchmarks.items() if not args.only or name in args.only]
  manifest = Manifest(args.manifest)
  if args.step == 'fetch':
    with httpx.Client(timeout=httpx.Timeout(60.0)) as client:
      results = [_fetch(name, source, args.data_dir, client, manifest) for name, source in selected]
  elif args.step == 'convert':
    results = [_convert(name, source, args.data_dir, manifest) for name, source in selected]
  else:
    results = asyncio.run(_import_benchmarks(selected, args.data_dir, manifest))
  return 0 if all(results) else 1


def _fetch(name: str, source: BenchmarkSource, data_dir: Path, client: httpx.Client, manifest: Manifest) -> bool:
  step = f'benchmarks:fetch:{name}'
  try:
    paths = benchmark_steps.fetch(name, source, data_dir, client)
  except (httpx.HTTPError, PinnedFileMismatchError, FetchInProgressError) as error:
    return _record_failure(manifest, step, name, error)
  manifest.append(step, StepStatus.COMPLETED, details={'base_url': source.base_url})
  print(f'{name}: {len(paths)} files verified under {data_dir / name / "raw"}')
  return True


def _convert(name: str, source: BenchmarkSource, data_dir: Path, manifest: Manifest) -> bool:
  step = f'benchmarks:convert:{name}'
  try:
    report = benchmark_steps.convert(name, source, data_dir)
  except (PinnedFileMismatchError, ValueError) as error:
    return _record_failure(manifest, step, name, error)
  status = StepStatus.COMPLETED if report.passed else StepStatus.FAILED
  manifest.append(step, status, details={'errors': report.errors, 'warnings': report.warnings})
  print(f'{name}: {status.value}, written to {data_dir / name}')
  for message in report.errors:
    print(f'  error: {message}')
  for message in report.warnings:
    print(f'  warning: {message}')
  return report.passed


async def _import_benchmarks(
  selected: Sequence[tuple[str, BenchmarkSource]], data_dir: Path, manifest: Manifest
) -> list[bool]:
  load_environment()
  results = []
  async with open_database(Settings()) as db_manager:
    for name, source in selected:
      step = f'benchmarks:import:{name}'
      # A crash past this point leaves the step started, never an earlier completion.
      manifest.append(step, StepStatus.STARTED)
      try:
        outcome = await benchmark_steps.import_converted(name, source, data_dir, db_manager)
      except (FileNotFoundError, ValueError, PersistenceError, psycopg.Error) as error:
        results.append(_record_failure(manifest, step, name, error))
        continue
      manifest.append(
        step,
        StepStatus.COMPLETED,
        details={
          'dataset_id': str(outcome.dataset.id),
          'dataset_name': source.dataset_name,
          'created': outcome.created,
          'claims_created': outcome.claims_created,
        },
      )
      action = 'imported' if outcome.created else 'already present'
      print(
        f'{name}: dataset {source.dataset_name} {action} ({outcome.dataset.id}); claims added: {outcome.claims_created}'
      )
      results.append(True)
  return results


async def _score_with_deepeval(args: argparse.Namespace) -> int:
  """Repeat the source run with the selected DeepEval metrics; the manifest records the step and the run it made."""
  judge_config = load_config(args.config_dir).models.judge
  load_environment()
  settings, judge_settings = Settings(), GeminiJudgeSettings()
  if judge_settings.api_key is None:
    print('The judge needs GOOGLE_API_KEY.', file=sys.stderr)
    return 2
  keys = [key for key in METRIC_KEYS if key in args.metrics]
  step = f'deepeval:{args.source_run}:{"+".join(keys)}'
  manifest = Manifest(args.manifest)
  details: dict[str, Any] = {'source_run_id': str(args.source_run), 'deepeval_version': DEEPEVAL_VERSION}
  run_ids: tuple[UUID, ...] = ()
  try:
    async with open_database(settings) as db_manager, open_deepeval_judge(judge_config, judge_settings) as judge:
      # The service's own check raises a bare RuntimeError; this one is recorded like any other database error.
      if not await db_manager.health_check():
        raise DatabaseConnectionError('Database health check failed')
      metrics = [deepeval_metrics(judge, output_token_limit=judge_config.output_token_limit)[key] for key in keys]
      details['metrics'] = [metric.name for metric in metrics]
      async with build_service(settings, db_manager, metrics=metrics) as service:
        handle = await service.repeat_evaluation(
          source_run_id=args.source_run,
          metrics=[metric.name for metric in metrics],
          max_concurrent_samples=args.max_concurrent_samples,
        )
        run_ids = (handle.run.id,)
        # A crash past this point leaves the step started, with the run it made.
        manifest.append(step, StepStatus.STARTED, run_ids=run_ids, details=details)
        run = await handle.execute()
      failed_units = await _failed_units(db_manager, run.id)
  except (ValueError, PersistenceError, psycopg.Error) as error:
    _record_failure(manifest, step, 'deepeval', error, run_ids=run_ids, details=details)
    return 1
  unscored = sum(
    count for kinds in failed_units.values() for kind, count in kinds.items() if kind not in _JUDGE_OUTCOMES
  )
  complete = run.status is EvaluationStatus.COMPLETED and not unscored
  details.update(run_status=run.status.value, failed_units=failed_units)
  manifest.append(step, StepStatus.COMPLETED if complete else StepStatus.FAILED, run_ids=run_ids, details=details)
  print(f'deepeval: run {run.id} {run.status.value}, scoring {", ".join(keys)} on the traces of run {args.source_run}')
  for metric, kinds in failed_units.items():
    print(f'  {metric} failed units: ' + ', '.join(f'{count} {kind}' for kind, count in kinds.items()))
  if unscored:
    print(f'deepeval: failed: {unscored} units failed without a judge outcome and must run again')
  return 0 if complete else 1


async def _failed_units(db_manager: DatabaseManager, run_id: UUID) -> dict[str, dict[str, int]]:
  """The run's failed computations per metric and failure class, ``unclassified`` when the metric recorded none."""
  async with UnitOfWork(db_manager) as uow:
    computations = await uow.metric_computations.list_by_evaluation_run(run_id)
  counts: defaultdict[str, Counter[str]] = defaultdict(Counter)
  for computation in computations:
    if computation.status is MetricComputationStatus.FAILED:
      counts[computation.metric][(computation.metadata or {}).get('failure', 'unclassified')] += 1
  return {metric: dict(sorted(kinds.items())) for metric, kinds in sorted(counts.items())}


def _record_failure(
  manifest: Manifest,
  step: str,
  name: str,
  error: Exception,
  *,
  run_ids: tuple[UUID, ...] = (),
  details: Mapping[str, Any] | None = None,
) -> bool:
  error_text = f'{type(error).__name__}: {error}'
  manifest.append(step, StepStatus.FAILED, run_ids=run_ids, details={**(details or {}), 'error': error_text})
  print(f'{name}: failed: {error_text}')
  return False
