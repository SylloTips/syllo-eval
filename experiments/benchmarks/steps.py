"""The steps every benchmark goes through: fetch its pinned files, convert them, then import the result.

Files live under ``<data dir>/<benchmark>/``: the pinned downloads in ``raw/`` (at their repository paths), the
converted dataset, side records, claims and check report next to it.
"""

import json
from collections.abc import Callable, Mapping
from pathlib import Path

import httpx

from syllo_eval.infrastructure import DatabaseManager

from benchmarks import erb, tau2, wixqa
from benchmarks.common import (
  REPORT_FILE,
  CheckReport,
  ConvertedBenchmark,
  ImportOutcome,
  check_benchmark,
  import_benchmark,
  read_benchmark,
  write_benchmark,
)
from benchmarks.download import fetch_pinned_files, verify_pinned_files
from config import BenchmarkSource

BUILDERS: dict[str, Callable[[Mapping[str, Path]], ConvertedBenchmark]] = {
  'erb': erb.build,
  'wixqa': wixqa.build,
  'tau2': tau2.build,
}


def fetch(name: str, source: BenchmarkSource, data_dir: Path, client: httpx.Client) -> dict[str, Path]:
  return fetch_pinned_files(source, data_dir / name / 'raw', client)


def convert(name: str, source: BenchmarkSource, data_dir: Path) -> CheckReport:
  """Convert the verified pinned files and write the result with its check report, which the import requires clean.

  The previous report is removed first, so a conversion that fails or stops halfway cannot leave an importable one.
  """
  (data_dir / name / REPORT_FILE).unlink(missing_ok=True)
  benchmark = BUILDERS[name](verify_pinned_files(source, data_dir / name / 'raw'))
  report = check_benchmark(benchmark, expected_samples=source.expected_samples)
  write_benchmark(benchmark, report, data_dir / name, source.model_dump(mode='json'))
  return report


async def import_converted(
  name: str, source: BenchmarkSource, data_dir: Path, db_manager: DatabaseManager
) -> ImportOutcome:
  """Import a conversion only if it passed its checks and was made from ``source``'s current pins."""
  directory = data_dir / name
  if not (directory / REPORT_FILE).exists():
    raise FileNotFoundError(f'{name} has no completed conversion; run `syllo-exp benchmarks convert --only {name}`')
  report = json.loads((directory / REPORT_FILE).read_text(encoding='utf-8'))
  if report['errors']:
    raise ValueError(f'{name} failed its checks, so it cannot be imported: {report["errors"]}')
  if report.get('source') != source.model_dump(mode='json'):
    raise ValueError(f'{name} was converted from other pins; run `syllo-exp benchmarks convert --only {name}` again')
  payload, claims = read_benchmark(directory)
  return await import_benchmark(db_manager, name=source.dataset_name, payload=payload, claims=claims)
