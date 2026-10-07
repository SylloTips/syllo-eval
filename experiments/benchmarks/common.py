"""What every benchmark converter produces, the checks it must pass, and how it reaches the database.

A converter turns a pinned benchmark release into a syllo-eval dataset payload plus one side record per sample with
the benchmark's own fields. Records are keyed by input prompt, because the database generates sample IDs at import.
Claim ground truths, which the dataset format does not cover, are attached after the import.
"""

import hashlib
import json
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from uuid import UUID, uuid4

from pydantic import JsonValue

from syllo_eval.datasets.ground_truths import build_sample_ground_truths
from syllo_eval.datasets.model import DatasetJsonPayload, DatasetJsonSample
from syllo_eval.datasets.service import DatasetService
from syllo_eval.infrastructure import DatabaseManager
from syllo_eval.infrastructure.unit_of_work import TransactionalUnitOfWork, UnitOfWork
from syllo_eval.model import Dataset, GroundTruth, GroundTruthKey, Sample

DATASET_FILE = 'dataset.json'
RECORDS_FILE = 'samples.jsonl'
CLAIMS_FILE = 'claims.json'
REPORT_FILE = 'report.json'


@dataclass(frozen=True, slots=True)
class Claim:
  id: str
  text: str


# Ground-truth key -> input prompt -> claims of that sample.
ClaimsByKey = dict[str, dict[str, list[Claim]]]


@dataclass(frozen=True, slots=True)
class KnowledgeDocument:
  """A knowledge-base document, as the search index stores it."""

  # The id the benchmark's gold lists use; the id in the benchmark file differs only for renamed ERB duplicates.
  document_id: str
  source_document_id: str
  title: str
  # The body without NUL characters: PostgreSQL JSONB rejects them once a document reaches a trace.
  text: str
  # Benchmark fields kept with the document, such as an ERB document's source type.
  metadata: dict[str, str] = field(default_factory=dict)


@dataclass(slots=True)
class ConvertedBenchmark:
  """A benchmark converted to syllo-eval samples; ``records[i]`` holds the benchmark fields of ``samples[i]``."""

  samples: list[DatasetJsonSample]
  records: list[dict[str, JsonValue]]
  claims: ClaimsByKey = field(default_factory=dict)
  # Document ids of the benchmark's knowledge base; None when it has none.
  knowledge_base_ids: frozenset[str] | None = None
  stats: dict[str, JsonValue] = field(default_factory=dict)

  @property
  def payload(self) -> DatasetJsonPayload:
    return DatasetJsonPayload(samples=self.samples)


@dataclass(slots=True)
class CheckReport:
  errors: list[str] = field(default_factory=list)
  warnings: list[str] = field(default_factory=list)

  @property
  def passed(self) -> bool:
    return not self.errors


def check_benchmark(benchmark: ConvertedBenchmark, *, expected_samples: int) -> CheckReport:
  """Integrity checks every converted benchmark must pass before it is written or imported."""
  report = CheckReport()
  samples = benchmark.samples
  if len(samples) != expected_samples:
    report.errors.append(f'Expected {expected_samples} samples, converted {len(samples)}')
  if len(benchmark.records) != len(samples):
    report.errors.append(f'{len(benchmark.records)} records for {len(samples)} samples')

  unstripped = [index for index, sample in enumerate(samples) if sample.input_prompt != sample.input_prompt.strip()]
  if unstripped or any(not sample.input_prompt for sample in samples):
    report.errors.append(f'Input prompts must be non-empty and stripped; offending samples: {unstripped[:10]}')
  # Imports bind each trace to the one sample whose prompt equals its request.
  duplicated = sorted(
    prompt for prompt, count in Counter(sample.input_prompt for sample in samples).items() if count > 1
  )
  if duplicated:
    report.errors.append(f'{len(duplicated)} input prompts are shared by several samples: {_preview(duplicated)}')

  if benchmark.knowledge_base_ids is not None:
    missing = sorted({doc for sample in samples for doc in sample.document_ids} - benchmark.knowledge_base_ids)
    if missing:
      report.errors.append(f'{len(missing)} gold documents are not in the knowledge base: {_preview(missing)}')
    without_gold = [
      str(record['sample_key']) for sample, record in zip(samples, benchmark.records) if not sample.document_ids
    ]
    if without_gold:
      report.warnings.append(
        f'{len(without_gold)} samples have no gold documents; set recall scores 1.0 on empty labels, so label-based '
        f'analyses must exclude them: {_preview(without_gold)}'
      )

  prompts = {sample.input_prompt for sample in samples}
  for key, claims_by_prompt in benchmark.claims.items():
    unknown = sorted(set(claims_by_prompt) - prompts)
    if unknown:
      report.errors.append(f'{key}: claims for {len(unknown)} prompts that are not samples: {_preview(unknown)}')
    empty = sorted(prompt for prompt, claims in claims_by_prompt.items() if not claims)
    if empty:
      report.errors.append(f'{key}: {len(empty)} samples have an empty claim list: {_preview(empty)}')
    claim_ids = Counter(claim.id for claims in claims_by_prompt.values() for claim in claims)
    repeated = sorted(claim_id for claim_id, count in claim_ids.items() if count > 1)
    if repeated:
      report.errors.append(f'{key}: claim ids are not unique: {_preview(repeated)}')
  return report


def write_benchmark(
  benchmark: ConvertedBenchmark, report: CheckReport, directory: Path, source: Mapping[str, JsonValue]
) -> None:
  """Write the import payload, the side records, the claims and the check report of one benchmark.

  ``source`` (the pins the files were converted from) goes into the report, which is written last: the import trusts
  a conversion only if its report exists, has no errors and names the current source.
  """
  directory.mkdir(parents=True, exist_ok=True)
  (directory / DATASET_FILE).write_text(benchmark.payload.model_dump_json(indent=2), encoding='utf-8')
  with (directory / RECORDS_FILE).open('w', encoding='utf-8') as file:
    for record in benchmark.records:
      file.write(json.dumps(record, ensure_ascii=False) + '\n')
  claims = {
    key: {prompt: [{'id': claim.id, 'text': claim.text} for claim in claims] for prompt, claims in by_prompt.items()}
    for key, by_prompt in benchmark.claims.items()
  }
  (directory / CLAIMS_FILE).write_text(json.dumps(claims, ensure_ascii=False, indent=2), encoding='utf-8')
  summary = {
    'source': source,
    'errors': report.errors,
    'warnings': report.warnings,
    'samples': len(benchmark.samples),
    **benchmark.stats,
  }
  (directory / REPORT_FILE).write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')


def read_benchmark(directory: Path) -> tuple[DatasetJsonPayload, ClaimsByKey]:
  """Read back what ``write_benchmark`` wrote, as the import needs it."""
  payload = DatasetJsonPayload.model_validate_json((directory / DATASET_FILE).read_text(encoding='utf-8'))
  raw_claims = json.loads((directory / CLAIMS_FILE).read_text(encoding='utf-8'))
  claims = {
    key: {prompt: [Claim(**claim) for claim in claims] for prompt, claims in by_prompt.items()}
    for key, by_prompt in raw_claims.items()
  }
  return payload, claims


@dataclass(frozen=True, slots=True)
class ImportOutcome:
  dataset: Dataset
  created: bool
  claims_created: dict[str, int]


async def import_benchmark(
  db_manager: DatabaseManager, *, name: str, payload: DatasetJsonPayload, claims: ClaimsByKey
) -> ImportOutcome:
  """Import the dataset unless it exists, then attach any claim ground truths it still lacks.

  Re-running is safe and never rewrites anything. An existing dataset must hold exactly this payload: its prompts, its
  answers and the ground truths the dataset import derives from them. Each claim key must hold exactly these claims.
  """
  prompts = {sample.input_prompt for sample in payload.samples}
  for key, claims_by_prompt in claims.items():
    unknown = sorted(set(claims_by_prompt) - prompts)
    if unknown:
      raise ValueError(f'{key}: claims for {len(unknown)} prompts that are not samples: {_preview(unknown)}')

  async with UnitOfWork(db_manager) as uow:
    dataset = await uow.datasets.get_by_name(name)
  created = dataset is None
  if dataset is None:
    dataset = (await DatasetService(db_manager).import_dataset(name=name, payload=payload)).dataset

  async with UnitOfWork(db_manager) as uow:
    samples = await uow.samples.list_by_dataset(dataset.id)
    sample_ids = _sample_ids_by_prompt(name, samples, payload.samples)
    if not created:
      await _check_stored_samples(uow, name, samples, payload.samples)

  claims_created: dict[str, int] = {}
  for key, claims_by_prompt in claims.items():
    async with TransactionalUnitOfWork(db_manager) as uow:
      stored = {
        ground_truth.sample_id: ground_truth.ground_truth_value
        for ground_truth in await uow.ground_truths.list_by_key(key)
        if ground_truth.sample_id in sample_ids.values()
      }
      claimed_ids = {sample_ids[prompt] for prompt in claims_by_prompt}
      if set(stored) - claimed_ids:
        raise ValueError(
          f'{len(set(stored) - claimed_ids)} samples of {name!r} have {key!r} ground truth this conversion does not'
        )
      new: list[GroundTruth] = []
      for prompt, prompt_claims in claims_by_prompt.items():
        sample_id = sample_ids[prompt]
        value = claims_value(prompt_claims)
        if sample_id not in stored:
          new.append(GroundTruth(id=uuid4(), sample_id=sample_id, key=key, ground_truth_value=value))
        elif stored[sample_id] != value:
          raise ValueError(f'Sample {sample_id} of {name!r} already has different {key!r} ground truth')
      if new:
        await uow.ground_truths.bulk_create(new)
    claims_created[key] = len(new)
  return ImportOutcome(dataset=dataset, created=created, claims_created=claims_created)


def claims_value(claims: Sequence[Claim]) -> dict[str, JsonValue]:
  """The ground-truth value of a claim list: ``{"claims": [{"id", "text"}, ...]}``."""
  return {'claims': [{'id': claim.id, 'text': claim.text} for claim in claims]}


def file_sha256(path: Path) -> str:
  digest = hashlib.sha256()
  with path.open('rb') as file:
    for chunk in iter(lambda: file.read(1 << 20), b''):
      digest.update(chunk)
  return digest.hexdigest()


# Lengths, in characters, at which a per-document cap in the search tool would start cutting gold text.
LENGTH_THRESHOLDS = (4_000, 8_000, 16_000)


def length_summary(lengths: Sequence[int]) -> dict[str, JsonValue]:
  """Percentiles, plus how many lengths exceed each threshold a per-document cap could be set at."""
  return {
    **percentiles(lengths),
    **{f'over_{t}': sum(1 for length in lengths if length > t) for t in LENGTH_THRESHOLDS},
  }


def longest_gold_lengths(samples: Sequence[DatasetJsonSample], length_by_id: Mapping[str, int]) -> list[int]:
  """Each sample's longest gold document, for samples that have one.

  Scores are means over questions, so a cap's cost is the share of questions whose longest gold document it cuts.
  """
  return [max(length_by_id.get(doc, 0) for doc in sample.document_ids) for sample in samples if sample.document_ids]


def percentiles(values: Sequence[int], points: Sequence[int] = (50, 90, 99)) -> dict[str, JsonValue]:
  """Nearest-rank percentiles plus min and max, for reporting length distributions."""
  if not values:
    return {}
  ordered = sorted(values)
  summary: dict[str, JsonValue] = {'min': ordered[0], 'max': ordered[-1]}
  for point in points:
    summary[f'p{point}'] = ordered[max(0, -(-point * len(ordered) // 100) - 1)]
  return summary


def _sample_ids_by_prompt(
  name: str, stored: Sequence[Sample], expected: Sequence[DatasetJsonSample]
) -> Mapping[str, UUID]:
  if sorted(sample.input_prompt for sample in stored) != sorted(sample.input_prompt for sample in expected):
    raise _different_samples(name)
  return {sample.input_prompt: sample.id for sample in stored}


async def _check_stored_samples(
  uow: UnitOfWork, name: str, stored: Sequence[Sample], expected: Sequence[DatasetJsonSample]
) -> None:
  """Compare an existing dataset with what importing ``expected`` would store: answers and derived ground truths."""
  expected_by_prompt = {sample.input_prompt: sample for sample in expected}
  expected_rows: dict[tuple[UUID, str], dict[str, Any]] = {}
  for sample in stored:
    payload_sample = expected_by_prompt[sample.input_prompt]
    if sample.ground_truth_output != payload_sample.ground_truth_output:
      raise _different_samples(name)
    for ground_truth in build_sample_ground_truths(sample.id, payload_sample):
      expected_rows[(sample.id, ground_truth.key)] = ground_truth.ground_truth_value
  sample_ids = {sample.id for sample in stored}
  stored_rows = {
    (ground_truth.sample_id, key.value): ground_truth.ground_truth_value
    for key in GroundTruthKey
    for ground_truth in await uow.ground_truths.list_by_key(key.value)
    if ground_truth.sample_id in sample_ids
  }
  if stored_rows != expected_rows:
    raise _different_samples(name)


def _different_samples(name: str) -> ValueError:
  return ValueError(f'Dataset {name!r} exists with different samples; import the new conversion under another name')


def _preview(values: Sequence[str], limit: int = 5) -> str:
  shown = ', '.join(repr(value[:80]) for value in values[:limit])
  return shown + (f' and {len(values) - limit} more' if len(values) > limit else '')
