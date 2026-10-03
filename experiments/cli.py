"""Command-line entry point: ``poetry run syllo-exp <command>``. Run it from the ``experiments/`` folder."""

import argparse
import asyncio
import sys
from collections.abc import Sequence
from pathlib import Path

import httpx
import psycopg

from syllo_eval.infrastructure.exceptions import PersistenceError
from syllo_eval.settings import Settings

from benchmarks import steps as benchmark_steps
from benchmarks.download import FetchInProgressError, PinnedFileMismatchError
from config import CONFIG_DIR, DATA_DIR, BenchmarkSource, load_config, load_environment
from manifest import Manifest, StepStatus
from service import open_database

DEFAULT_MANIFEST = Path(__file__).resolve().parent / 'outputs' / 'manifest.jsonl'
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
  return parser


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


def _record_failure(manifest: Manifest, step: str, name: str, error: Exception) -> bool:
  manifest.append(step, StepStatus.FAILED, details={'error': f'{type(error).__name__}: {error}'})
  print(f'{name}: failed: {type(error).__name__}: {error}')
  return False
