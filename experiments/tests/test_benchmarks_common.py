import json
import tempfile
import unittest
from pathlib import Path
from typing import Any, cast
from uuid import uuid4

from syllo_eval.datasets.model import DatasetJsonPlanStep, DatasetJsonSample
from syllo_eval.infrastructure.unit_of_work import UnitOfWork
from syllo_eval.testing_database import setup_test_database

from benchmarks.common import (
  Claim,
  ConvertedBenchmark,
  check_benchmark,
  claims_from_value,
  claims_value,
  import_benchmark,
  percentiles,
  read_benchmark,
  write_benchmark,
)
from config import load_environment


def _benchmark(**overrides) -> ConvertedBenchmark:
  fields = {
    'samples': [
      DatasetJsonSample(input_prompt='Who approved the budget?', ground_truth_output='Dana.', document_ids=['d1']),
      DatasetJsonSample(input_prompt='When is the offsite?', ground_truth_output='In May.', document_ids=['d2', 'd3']),
    ],
    'records': [{'sample_key': 'q1'}, {'sample_key': 'q2'}],
    'claims': {'expected_claims_gold': {'Who approved the budget?': [Claim(id='q1#1', text='Dana approved it.')]}},
    'knowledge_base_ids': frozenset({'d1', 'd2', 'd3'}),
  }
  return ConvertedBenchmark(**{**fields, **overrides})


class CheckBenchmarkTest(unittest.TestCase):
  def test_clean_benchmark_passes(self) -> None:
    report = check_benchmark(_benchmark(), expected_samples=2)

    self.assertTrue(report.passed, report.errors)
    self.assertEqual(report.warnings, [])

  def test_reports_every_integrity_error(self) -> None:
    samples = [
      DatasetJsonSample(input_prompt='Same question?', document_ids=['d1']),
      DatasetJsonSample(input_prompt='Same question?', document_ids=['missing-doc']),
      DatasetJsonSample(input_prompt=' padded ', document_ids=['d1']),
    ]
    claims = {
      'expected_claims_gold': {
        'Same question?': [Claim(id='c1', text='A'), Claim(id='c1', text='B')],
        'Not a sample?': [],
      }
    }
    benchmark = _benchmark(samples=samples, records=[{'sample_key': 'q1'}], claims=claims)

    errors = '\n'.join(check_benchmark(benchmark, expected_samples=2).errors)

    for expected in [
      'Expected 2 samples, converted 3',
      '1 records for 3 samples',
      'non-empty and stripped',
      'shared by several samples',
      'not in the knowledge base',
      'prompts that are not samples',
      'empty claim list',
      'claim ids are not unique',
    ]:
      self.assertIn(expected, errors)

  def test_samples_without_gold_documents_are_a_warning(self) -> None:
    samples = [DatasetJsonSample(input_prompt='Who approved the budget?', document_ids=[])]
    benchmark = _benchmark(samples=samples, records=[{'sample_key': 'q1'}])

    report = check_benchmark(benchmark, expected_samples=1)

    self.assertTrue(report.passed, report.errors)
    self.assertIn('no gold documents', report.warnings[0])
    self.assertIn('q1', report.warnings[0])

  def test_benchmarks_without_a_knowledge_base_skip_document_checks(self) -> None:
    samples = [DatasetJsonSample(input_prompt='tau2/retail/0')]
    benchmark = _benchmark(samples=samples, records=[{'sample_key': '0'}], claims={}, knowledge_base_ids=None)

    report = check_benchmark(benchmark, expected_samples=1)

    self.assertTrue(report.passed, report.errors)
    self.assertEqual(report.warnings, [])


class WriteReadBenchmarkTest(unittest.TestCase):
  def test_round_trip_keeps_payload_and_claims(self) -> None:
    benchmark = _benchmark(stats={'knowledge_base_documents': 3})
    with tempfile.TemporaryDirectory() as directory:
      write_benchmark(benchmark, check_benchmark(benchmark, expected_samples=2), Path(directory), {'pin': 'abc'})
      payload, claims = read_benchmark(Path(directory))
      records = (Path(directory) / 'samples.jsonl').read_text(encoding='utf-8').splitlines()
      report = json.loads((Path(directory) / 'report.json').read_text(encoding='utf-8'))

    self.assertEqual(payload, benchmark.payload)
    self.assertEqual(claims, benchmark.claims)
    self.assertEqual(len(records), 2)
    self.assertEqual(report['source'], {'pin': 'abc'})
    self.assertEqual(report['knowledge_base_documents'], 3)


class ClaimsValueTest(unittest.TestCase):
  def test_claims_round_trip_through_the_ground_truth_value(self) -> None:
    claims = [Claim(id='q1-f01', text='Dana approved it.'), Claim(id='q1-f02', text='It was in May.')]

    self.assertEqual(claims_from_value(claims_value(claims)), claims)

  def test_values_without_well_formed_claims_are_refused(self) -> None:
    values: list[dict[str, Any]] = [
      {},
      {'claims': []},
      {'claims': 'Dana approved it.'},
      {'claims': [{'id': 'q1-f01'}]},
      {'claims': [{'id': '', 'text': 'Dana approved it.'}]},
      {'claims': [{'id': 'q1-f01', 'text': '  '}]},
    ]
    for value in values:
      with self.subTest(value=value), self.assertRaises(ValueError):
        claims_from_value(value)


class ImportBenchmarkValidationTest(unittest.IsolatedAsyncioTestCase):
  async def test_claims_for_prompts_that_are_not_samples_are_refused_before_any_database_access(self) -> None:
    claims = {'expected_claims_gold': {'Not a sample?': [Claim(id='x#1', text='X.')]}}

    with self.assertRaisesRegex(ValueError, 'not samples'):
      await import_benchmark(cast(Any, object()), name='unused', payload=_benchmark().payload, claims=claims)


class PercentilesTest(unittest.TestCase):
  def test_nearest_rank_percentiles(self) -> None:
    self.assertEqual(percentiles(list(range(1, 101))), {'min': 1, 'max': 100, 'p50': 50, 'p90': 90, 'p99': 99})
    self.assertEqual(percentiles([7]), {'min': 7, 'max': 7, 'p50': 7, 'p90': 7, 'p99': 7})
    self.assertEqual(percentiles([]), {})


class ImportBenchmarkTest(unittest.IsolatedAsyncioTestCase):
  async def asyncSetUp(self) -> None:
    load_environment()
    try:
      self.db_manager = await setup_test_database()
    except RuntimeError as error:
      self.skipTest(f'Database unavailable: {error}')
    self.name = f'test-benchmark-{uuid4().hex[:10]}'

  async def asyncTearDown(self) -> None:
    async with UnitOfWork(self.db_manager) as uow:
      dataset = await uow.datasets.get_by_name(self.name)
      if dataset is not None:
        await uow.datasets.delete(dataset.id)
    await self.db_manager.close_async()

  async def test_import_is_idempotent_and_attaches_claims_once(self) -> None:
    benchmark = _benchmark()

    first = await import_benchmark(self.db_manager, name=self.name, payload=benchmark.payload, claims=benchmark.claims)
    again = await import_benchmark(self.db_manager, name=self.name, payload=benchmark.payload, claims=benchmark.claims)

    self.assertTrue(first.created)
    self.assertEqual(first.claims_created, {'expected_claims_gold': 1})
    self.assertFalse(again.created)
    self.assertEqual(again.dataset.id, first.dataset.id)
    self.assertEqual(again.claims_created, {'expected_claims_gold': 0})
    async with UnitOfWork(self.db_manager) as uow:
      samples = {sample.input_prompt: sample for sample in await uow.samples.list_by_dataset(first.dataset.id)}
      stored = await uow.ground_truths.get_by_sample_and_key(
        samples['Who approved the budget?'].id, 'expected_claims_gold'
      )
      documents = await uow.ground_truths.get_by_sample_and_key(
        samples['When is the offsite?'].id, 'relevant_document_ids'
      )
    self.assertIsNotNone(stored)
    assert stored is not None and documents is not None
    self.assertEqual(stored.ground_truth_value, claims_value([Claim(id='q1#1', text='Dana approved it.')]))
    self.assertEqual(documents.ground_truth_value, {'relevant_ids': ['d2', 'd3']})

  async def test_conflicting_claims_and_changed_samples_are_rejected(self) -> None:
    benchmark = _benchmark()
    await import_benchmark(self.db_manager, name=self.name, payload=benchmark.payload, claims=benchmark.claims)

    changed_claims = {'expected_claims_gold': {'Who approved the budget?': [Claim(id='q1#1', text='Someone else.')]}}
    with self.assertRaisesRegex(ValueError, 'different'):
      await import_benchmark(self.db_manager, name=self.name, payload=benchmark.payload, claims=changed_claims)

    changed_samples = _benchmark(samples=[DatasetJsonSample(input_prompt='A new question?')]).payload
    with self.assertRaisesRegex(ValueError, 'different samples'):
      await import_benchmark(self.db_manager, name=self.name, payload=changed_samples, claims={})

  async def test_a_reimport_compares_answers_and_derived_ground_truths_not_only_prompts(self) -> None:
    benchmark = _benchmark()
    await import_benchmark(self.db_manager, name=self.name, payload=benchmark.payload, claims=benchmark.claims)
    first, second = benchmark.samples
    changes = {
      'answer': first.model_copy(update={'ground_truth_output': 'Dana and Lee.'}),
      'gold documents': first.model_copy(update={'document_ids': ['d1', 'd3']}),
      'added plan': first.model_copy(update={'plan': [DatasetJsonPlanStep(operation='search', instruction='{}')]}),
    }
    for change, sample in changes.items():
      payload = _benchmark(samples=[sample, second]).payload
      with self.subTest(change=change), self.assertRaisesRegex(ValueError, 'different samples'):
        await import_benchmark(self.db_manager, name=self.name, payload=payload, claims=benchmark.claims)

  async def test_a_reimport_refuses_claims_the_conversion_no_longer_has(self) -> None:
    benchmark = _benchmark()
    await import_benchmark(self.db_manager, name=self.name, payload=benchmark.payload, claims=benchmark.claims)

    with self.assertRaisesRegex(ValueError, 'this conversion does not'):
      await import_benchmark(
        self.db_manager, name=self.name, payload=benchmark.payload, claims={'expected_claims_gold': {}}
      )

  async def test_claims_are_attached_when_an_interrupted_import_is_run_again(self) -> None:
    benchmark = _benchmark()
    # A first run that stopped after the dataset import, before its claims were attached.
    await import_benchmark(self.db_manager, name=self.name, payload=benchmark.payload, claims={})

    resumed = await import_benchmark(
      self.db_manager, name=self.name, payload=benchmark.payload, claims=benchmark.claims
    )

    self.assertFalse(resumed.created)
    self.assertEqual(resumed.claims_created, {'expected_claims_gold': 1})
    async with UnitOfWork(self.db_manager) as uow:
      samples = {sample.input_prompt: sample for sample in await uow.samples.list_by_dataset(resumed.dataset.id)}
      stored = await uow.ground_truths.get_by_sample_and_key(
        samples['Who approved the budget?'].id, 'expected_claims_gold'
      )
    assert stored is not None
    self.assertEqual(stored.ground_truth_value, claims_value([Claim(id='q1#1', text='Dana approved it.')]))


if __name__ == '__main__':
  unittest.main()
