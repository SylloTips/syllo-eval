import hashlib
import io
import json
import shutil
import tempfile
import unittest
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager, redirect_stderr, redirect_stdout
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, patch
from uuid import uuid4

import httpx
import psycopg
import yaml

from syllo_eval.model import Dataset

from benchmarks.common import ImportOutcome
from cli import main
from config import CONFIG_DIR
from manifest import Manifest, StepStatus


# Patching cli.httpx.Client patches the shared httpx module, so keep the real class for the mock client.
_REAL_CLIENT = httpx.Client


def _run(argv: list[str]) -> tuple[int, str]:
  output = io.StringIO()
  with redirect_stdout(output):
    status = main(argv)
  return status, output.getvalue()


def _task(task_id: str) -> dict:
  return {
    'id': task_id,
    'user_scenario': {
      'persona': None,
      'instructions': {
        'domain': 'retail',
        'reason_for_call': f'Reason {task_id}.',
        'known_info': 'You are sam.',
        'unknown_info': None,
        'task_instructions': '.',
      },
    },
    'evaluation_criteria': {
      'actions': [{'action_id': f'{task_id}_0', 'name': 'get_user_details', 'arguments': {'user_id': 'sam'}}],
      'communicate_info': [],
      'nl_assertions': None,
      'reward_basis': ['DB'],
    },
  }


class CliTest(unittest.TestCase):
  def test_configurations_lists_every_agent_configuration(self) -> None:
    status, output = _run(['configurations'])

    self.assertEqual(status, 0)
    self.assertIn('erb/react/sonnet/f0.25', output)
    self.assertIn('version=deepseek-trial4', output)
    self.assertIn('22 configurations', output)

  def test_steps_shows_the_latest_state_of_each_step(self) -> None:
    with tempfile.TemporaryDirectory() as directory:
      manifest_path = Path(directory) / 'manifest.jsonl'
      manifest = Manifest(manifest_path)
      manifest.append('collect:erb/react/sonnet', StepStatus.STARTED)
      manifest.append('collect:erb/react/sonnet', StepStatus.COMPLETED)

      status, output = _run(['steps', '--manifest', str(manifest_path)])

    self.assertEqual(status, 0)
    self.assertEqual(output.count('collect:erb/react/sonnet'), 1)
    self.assertIn('completed', output)


class BenchmarksConvertCliTest(unittest.TestCase):
  """Runs `benchmarks convert` on synthetic tau2 files pinned by a temporary config."""

  def setUp(self) -> None:
    self._directory = tempfile.TemporaryDirectory()
    root = Path(self._directory.name)
    self.config_dir, self.data_dir, self.manifest = root / 'configs', root / 'data', root / 'manifest.jsonl'
    shutil.copytree(CONFIG_DIR, self.config_dir)
    files = {
      'tasks': ('tasks.json', json.dumps([_task('0'), _task('1')]).encode()),
      'splits': ('split_tasks.json', json.dumps({'base': ['0', '1'], 'train': ['0'], 'test': ['1']}).encode()),
    }
    for path, content in files.values():
      (self.data_dir / 'tau2' / 'raw').mkdir(parents=True, exist_ok=True)
      (self.data_dir / 'tau2' / 'raw' / path).write_bytes(content)
    self.pins = {
      role: {'path': path, 'size': len(content), 'sha256': hashlib.sha256(content).hexdigest()}
      for role, (path, content) in files.items()
    }

  def tearDown(self) -> None:
    self._directory.cleanup()

  def _write_tau2_source(self, expected_samples: int) -> None:
    path = self.config_dir / 'benchmarks.yaml'
    benchmarks = yaml.safe_load(path.read_text(encoding='utf-8'))
    benchmarks['tau2'] = {**benchmarks['tau2'], 'expected_samples': expected_samples, 'files': self.pins}
    path.write_text(yaml.safe_dump(benchmarks), encoding='utf-8')

  def _convert(self) -> tuple[int, str]:
    return _run(
      [
        'benchmarks',
        'convert',
        '--only',
        'tau2',
        '--config-dir',
        str(self.config_dir),
        '--data-dir',
        str(self.data_dir),
        '--manifest',
        str(self.manifest),
      ]
    )

  def test_writes_the_converted_dataset_and_records_the_step(self) -> None:
    self._write_tau2_source(expected_samples=2)

    status, output = self._convert()

    self.assertEqual(status, 0, output)
    dataset = json.loads((self.data_dir / 'tau2' / 'dataset.json').read_text(encoding='utf-8'))
    self.assertEqual(len(dataset['samples']), 2)
    self.assertTrue(Manifest(self.manifest).is_completed('benchmarks:convert:tau2'))

  def test_failed_checks_exit_non_zero_and_are_recorded(self) -> None:
    self._write_tau2_source(expected_samples=3)

    status, output = self._convert()

    self.assertEqual(status, 1)
    self.assertIn('Expected 3 samples, converted 2', output)
    record = Manifest(self.manifest).latest()['benchmarks:convert:tau2']
    self.assertEqual(record.status, StepStatus.FAILED)


class _ConfigDirTestCase(unittest.TestCase):
  """A copy of the committed configs, with every benchmark pinned to small synthetic files."""

  def setUp(self) -> None:
    self._directory = tempfile.TemporaryDirectory()
    root = Path(self._directory.name)
    self.config_dir, self.data_dir, self.manifest = root / 'configs', root / 'data', root / 'manifest.jsonl'
    shutil.copytree(CONFIG_DIR, self.config_dir)
    self.contents = {name: f'{name} bytes'.encode() for name in ('erb', 'wixqa', 'tau2')}
    benchmarks = {
      name: {
        'dataset_name': f'{name}-test',
        'expected_samples': 1,
        'base_url': f'https://example.org/{name}/{"a" * 40}',
        'files': {'data': {'path': 'data.bin', 'size': len(content), 'sha256': hashlib.sha256(content).hexdigest()}},
      }
      for name, content in self.contents.items()
    }
    (self.config_dir / 'benchmarks.yaml').write_text(yaml.safe_dump(benchmarks), encoding='utf-8')

  def tearDown(self) -> None:
    self._directory.cleanup()

  def _run_step(self, *args: str) -> tuple[int, str]:
    return _run(
      [
        'benchmarks',
        *args,
        '--config-dir',
        str(self.config_dir),
        '--data-dir',
        str(self.data_dir),
        '--manifest',
        str(self.manifest),
      ]
    )

  def _status(self, step: str) -> StepStatus:
    return Manifest(self.manifest).latest()[step].status


class BenchmarksFetchCliTest(_ConfigDirTestCase):
  def test_a_failing_benchmark_is_recorded_and_the_others_still_run(self) -> None:
    def serve(request: httpx.Request) -> httpx.Response:
      name = request.url.path.split('/')[1]
      return httpx.Response(503) if name == 'erb' else httpx.Response(200, content=self.contents[name])

    def mock_client(**kwargs: Any) -> httpx.Client:
      return _REAL_CLIENT(transport=httpx.MockTransport(serve), **kwargs)

    with patch('cli.httpx.Client', side_effect=mock_client), patch('benchmarks.download.time.sleep'):
      status, output = self._run_step('fetch')

    self.assertEqual(status, 1)
    self.assertIn('erb: failed', output)
    self.assertEqual(self._status('benchmarks:fetch:erb'), StepStatus.FAILED)
    self.assertEqual(self._status('benchmarks:fetch:wixqa'), StepStatus.COMPLETED)
    self.assertEqual(self._status('benchmarks:fetch:tau2'), StepStatus.COMPLETED)


class BenchmarksImportCliTest(_ConfigDirTestCase):
  def test_a_database_error_is_recorded_and_the_next_benchmark_still_imports(self) -> None:
    @asynccontextmanager
    async def fake_database(settings: Any) -> AsyncIterator[object]:
      yield object()

    outcome = ImportOutcome(dataset=Dataset(id=uuid4(), name='wixqa-test'), created=True, claims_created={})
    import_converted = AsyncMock(side_effect=[psycopg.OperationalError('connection refused'), outcome])
    with patch('cli.open_database', fake_database), patch('cli.benchmark_steps.import_converted', import_converted):
      status, output = self._run_step('import', '--only', 'erb', 'wixqa')

    self.assertEqual(status, 1)
    self.assertIn('erb: failed: OperationalError', output)
    self.assertEqual(self._status('benchmarks:import:erb'), StepStatus.FAILED)
    self.assertEqual(self._status('benchmarks:import:wixqa'), StepStatus.COMPLETED)


class BenchmarksSelectionCliTest(_ConfigDirTestCase):
  def test_selecting_a_benchmark_the_config_does_not_pin_is_an_error(self) -> None:
    (self.config_dir / 'benchmarks.yaml').write_text(
      yaml.safe_dump({'erb': yaml.safe_load((self.config_dir / 'benchmarks.yaml').read_text())['erb']}),
      encoding='utf-8',
    )
    configurations = yaml.safe_load((self.config_dir / 'configurations.yaml').read_text())
    configurations['configurations'] = [c for c in configurations['configurations'] if c['benchmark'] == 'erb']
    (self.config_dir / 'configurations.yaml').write_text(yaml.safe_dump(configurations), encoding='utf-8')

    errors = io.StringIO()
    with redirect_stderr(errors):
      status, _ = self._run_step('convert', '--only', 'tau2')

    self.assertEqual(status, 2)
    self.assertIn('Not pinned', errors.getvalue())
    self.assertFalse(self.manifest.exists())


if __name__ == '__main__':
  unittest.main()
