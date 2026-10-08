"""The steps every benchmark goes through: fetch its pinned files, convert them, then import the result.

Files live under ``<data dir>/<benchmark>/``: the pinned downloads in ``raw/`` (at their repository paths), the
converted dataset, side records, claims and check report next to it. A pilot dataset, a few samples of the conversion
that a pilot runs on, goes to ``pilot/``.
"""

import json
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path

import httpx

from syllo_eval.infrastructure import DatabaseManager

from benchmarks import erb, tau2, wixqa
from syllo_eval.datasets.model import DatasetJsonPayload

from benchmarks.common import (
  DATASET_FILE,
  REPORT_FILE,
  CheckReport,
  ClaimsByKey,
  ConvertedBenchmark,
  ImportOutcome,
  check_benchmark,
  import_benchmark,
  read_benchmark,
  read_records,
  write_benchmark,
  write_claims,
  write_records,
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
  payload, claims = _read_checked(name, source, data_dir)
  return await import_benchmark(db_manager, name=source.dataset_name, payload=payload, claims=claims)


def pilot_dataset_name(source: BenchmarkSource) -> str:
  return f'{source.dataset_name}-pilot'


async def import_pilot(
  name: str, source: BenchmarkSource, data_dir: Path, db_manager: DatabaseManager, sample_keys: Sequence[str]
) -> ImportOutcome:
  """Import the samples of a checked conversion with the given ``sample_key``s as the benchmark's pilot dataset.

  The pilot dataset is also written to ``pilot/``, in the conversion's format. Its samples and ground truths are the
  conversion's, so a pilot run scores them as the full run will. A pilot dataset holds one selection: importing other
  samples under its name is refused, as for any dataset.
  """
  payload, claims = _read_checked(name, source, data_dir)
  records = read_records(data_dir / name)
  unknown = sorted(set(sample_keys) - {str(record['sample_key']) for record in records})
  if unknown:
    raise ValueError(f'{name} has no samples with keys {unknown}')
  chosen = [index for index, record in enumerate(records) if str(record['sample_key']) in set(sample_keys)]
  pilot = DatasetJsonPayload(samples=[payload.samples[index] for index in chosen])
  prompts = {sample.input_prompt for sample in pilot.samples}
  pilot_claims: ClaimsByKey = {
    key: {prompt: claims for prompt, claims in by_prompt.items() if prompt in prompts}
    for key, by_prompt in claims.items()
  }
  directory = data_dir / name / 'pilot'
  directory.mkdir(exist_ok=True)
  (directory / DATASET_FILE).write_text(pilot.model_dump_json(indent=2), encoding='utf-8')
  write_records(directory, [records[index] for index in chosen])
  write_claims(directory, pilot_claims)
  return await import_benchmark(db_manager, name=pilot_dataset_name(source), payload=pilot, claims=pilot_claims)


def _read_checked(name: str, source: BenchmarkSource, data_dir: Path) -> tuple[DatasetJsonPayload, ClaimsByKey]:
  """A conversion that passed its checks and was made from ``source``'s current pins."""
  directory = data_dir / name
  if not (directory / REPORT_FILE).exists():
    raise FileNotFoundError(f'{name} has no completed conversion; run `syllo-exp benchmarks convert --only {name}`')
  report = json.loads((directory / REPORT_FILE).read_text(encoding='utf-8'))
  if report['errors']:
    raise ValueError(f'{name} failed its checks, so it cannot be imported: {report["errors"]}')
  if report.get('source') != source.model_dump(mode='json'):
    raise ValueError(f'{name} was converted from other pins; run `syllo-exp benchmarks convert --only {name}` again')
  return read_benchmark(directory)
