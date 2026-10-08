import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from typing import Any, cast
from unittest.mock import AsyncMock, patch

from syllo_eval.datasets.model import DatasetJsonSample

from benchmarks import steps
from benchmarks.common import REPORT_FILE, CheckReport, Claim, ConvertedBenchmark, read_benchmark, write_benchmark
from benchmarks.download import PinnedFileMismatchError
from config import BenchmarkSource

TASK = {
  'id': '0',
  'user_scenario': {
    'persona': None,
    'instructions': {
      'domain': 'retail',
      'reason_for_call': 'Return a lamp.',
      'known_info': 'You are sam.',
      'unknown_info': None,
      'task_instructions': '.',
    },
  },
  'evaluation_criteria': {'actions': [], 'communicate_info': [], 'nl_assertions': None, 'reward_basis': ['DB']},
}
FILES = {
  'tasks': ('tasks.json', json.dumps([TASK]).encode()),
  'splits': ('split_tasks.json', json.dumps({'base': ['0'], 'train': ['0'], 'test': []}).encode()),
}


def _source(**overrides: Any) -> BenchmarkSource:
  fields: dict[str, Any] = {
    'dataset_name': 'tau2-test',
    'expected_samples': 1,
    'base_url': 'https://example.org/repo/' + 'a' * 40,
    'files': {
      role: {'path': path, 'size': len(content), 'sha256': hashlib.sha256(content).hexdigest()}
      for role, (path, content) in FILES.items()
    },
  }
  return BenchmarkSource.model_validate({**fields, **overrides})


class StepsTest(unittest.IsolatedAsyncioTestCase):
  def setUp(self) -> None:
    self._directory = tempfile.TemporaryDirectory()
    self.data_dir = Path(self._directory.name)
    raw = self.data_dir / 'tau2' / 'raw'
    raw.mkdir(parents=True)
    for path, content in FILES.values():
      (raw / path).write_bytes(content)
    self.report = self.data_dir / 'tau2' / REPORT_FILE

  def tearDown(self) -> None:
    self._directory.cleanup()

  async def _import(self, source: BenchmarkSource) -> Any:
    # The checks below run before any database access, so the database manager is never used.
    return await steps.import_converted('tau2', source, self.data_dir, cast(Any, object()))

  def test_convert_records_the_source_it_was_made_from(self) -> None:
    source = _source()

    self.assertTrue(steps.convert('tau2', source, self.data_dir).passed)

    self.assertEqual(json.loads(self.report.read_text(encoding='utf-8'))['source'], source.model_dump(mode='json'))

  def test_a_failed_conversion_leaves_no_report_behind(self) -> None:
    steps.convert('tau2', _source(), self.data_dir)
    bumped = _source(files={'tasks': {'path': 'tasks.json', 'size': 1, 'sha256': 'f' * 64}})

    with self.assertRaises(PinnedFileMismatchError):
      steps.convert('tau2', bumped, self.data_dir)

    self.assertFalse(self.report.exists())

  async def test_import_refuses_a_missing_report(self) -> None:
    with self.assertRaisesRegex(FileNotFoundError, 'no completed conversion'):
      await self._import(_source())

  async def test_import_refuses_a_conversion_that_failed_its_checks(self) -> None:
    steps.convert('tau2', _source(expected_samples=2), self.data_dir)

    with self.assertRaisesRegex(ValueError, 'failed its checks'):
      await self._import(_source(expected_samples=2))

  async def test_import_refuses_a_conversion_made_from_other_pins(self) -> None:
    steps.convert('tau2', _source(), self.data_dir)

    with self.assertRaisesRegex(ValueError, 'other pins'):
      await self._import(_source(dataset_name='tau2-renamed'))


class PilotTest(unittest.IsolatedAsyncioTestCase):
  """A pilot dataset made from a conversion of three samples, one of them with claims."""

  def setUp(self) -> None:
    self._directory = tempfile.TemporaryDirectory()
    self.data_dir = Path(self._directory.name)
    self.source = _source(dataset_name='kb-test', expected_samples=3)
    benchmark = ConvertedBenchmark(
      samples=[
        DatasetJsonSample(input_prompt=f'Question {key}?', ground_truth_output=f'Answer {key}.') for key in 'abc'
      ],
      records=[{'sample_key': key, 'kind': 'basic'} for key in 'abc'],
      claims={'claims_gold': {'Question c?': [Claim(id='c1', text='C holds.')], 'Question a?': [Claim('a1', 'A.')]}},
    )
    write_benchmark(benchmark, CheckReport(), self.data_dir / 'kb', self.source.model_dump(mode='json'))
    self.import_benchmark = AsyncMock(return_value='outcome')

  def tearDown(self) -> None:
    self._directory.cleanup()

  async def _pilot(self, sample_keys: list[str], source: BenchmarkSource | None = None) -> Any:
    with patch('benchmarks.steps.import_benchmark', self.import_benchmark):
      return await steps.import_pilot('kb', source or self.source, self.data_dir, cast(Any, object()), sample_keys)

  async def test_imports_and_writes_the_chosen_samples_with_their_claims_as_the_pilot_dataset(self) -> None:
    outcome = await self._pilot(['c', 'b'])

    self.assertEqual(outcome, 'outcome')
    call = self.import_benchmark.await_args
    assert call is not None
    self.assertEqual(call.kwargs['name'], 'kb-test-pilot')
    self.assertEqual([sample.input_prompt for sample in call.kwargs['payload'].samples], ['Question b?', 'Question c?'])
    self.assertEqual(call.kwargs['claims'], {'claims_gold': {'Question c?': [Claim(id='c1', text='C holds.')]}})
    pilot = self.data_dir / 'kb' / 'pilot'
    self.assertEqual(read_benchmark(pilot), (call.kwargs['payload'], call.kwargs['claims']))
    records = [json.loads(line) for line in (pilot / 'samples.jsonl').read_text(encoding='utf-8').splitlines()]
    self.assertEqual([record['sample_key'] for record in records], ['b', 'c'])

  async def test_unknown_samples_and_a_stale_conversion_are_refused(self) -> None:
    with self.assertRaisesRegex(ValueError, r"kb has no samples with keys \['z'\]"):
      await self._pilot(['a', 'z'])
    with self.assertRaisesRegex(ValueError, 'other pins'):
      await self._pilot(['a'], _source(dataset_name='kb-renamed', expected_samples=3))
    self.import_benchmark.assert_not_awaited()


if __name__ == '__main__':
  unittest.main()
