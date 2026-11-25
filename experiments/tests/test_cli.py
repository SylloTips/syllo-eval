import hashlib
import io
import json
import os
import shutil
import tempfile
import unittest
from collections.abc import AsyncIterator, Sequence
from contextlib import ExitStack, asynccontextmanager, redirect_stderr, redirect_stdout
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import UUID, uuid4

import httpx
import psycopg
import yaml

from syllo_eval.infrastructure.exceptions import NotFoundError
from syllo_eval.model import Dataset, EvaluationStatus, MetricComputationStatus

from agentevals_baseline import AGENTEVALS_VERSION
from benchmarks.common import ImportOutcome
from cli import DEFAULT_MANIFEST, main
from config import CONFIG_DIR
from deepeval_baseline import DEEPEVAL_VERSION, METRIC_KEYS
from indexing.embedding import EmbeddingError, Embeddings
from indexing.pipeline import EmbedOutcome, LoadOutcome
from indexing.vector_store import VectorStoreError
from manifest import Manifest, StepStatus
from openevals_baseline import OPENEVALS_VERSION, openevals_metrics


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
    self.assertIn('erb/react/gpt/f0.25', output)
    self.assertIn('version=deepseek-trial4', output)
    self.assertIn('26 configurations', output)

  def test_steps_shows_the_latest_state_of_each_step(self) -> None:
    with tempfile.TemporaryDirectory() as directory:
      manifest_path = Path(directory) / 'manifest.jsonl'
      manifest = Manifest(manifest_path)
      manifest.append('collect:erb/react/gpt', StepStatus.STARTED)
      manifest.append('collect:erb/react/gpt', StepStatus.COMPLETED)

      status, output = _run(['steps', '--manifest', str(manifest_path)])

    self.assertEqual(status, 0)
    self.assertEqual(output.count('collect:erb/react/gpt'), 1)
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


class BenchmarksPilotCliTest(_ConfigDirTestCase):
  def _pilot(self, import_pilot: AsyncMock) -> tuple[int, str]:
    @asynccontextmanager
    async def fake_database(settings: Any) -> AsyncIterator[object]:
      yield object()

    with patch('cli.open_database', fake_database), patch('cli.benchmark_steps.import_pilot', import_pilot):
      return self._run_step('pilot', '--benchmark', 'tau2', '--samples', '0', '24', '40')

  def test_imports_the_chosen_samples_as_the_pilot_dataset(self) -> None:
    outcome = ImportOutcome(dataset=Dataset(id=uuid4(), name='tau2-pilot'), created=True, claims_created={})
    import_pilot = AsyncMock(return_value=outcome)

    status, output = self._pilot(import_pilot)

    self.assertEqual(status, 0, output)
    call = import_pilot.await_args
    assert call is not None
    self.assertEqual((call.args[0], call.args[4]), ('tau2', ['0', '24', '40']))
    record = Manifest(self.manifest).latest()['benchmarks:pilot:tau2']
    self.assertEqual(record.status, StepStatus.COMPLETED)
    self.assertTrue(str(record.details['dataset_name']).endswith('-pilot'))
    self.assertEqual(record.details['samples'], ['0', '24', '40'])
    self.assertIn(f'imported ({outcome.dataset.id}) with samples 0, 24, 40', output)

  def test_a_refused_selection_is_recorded(self) -> None:
    status, output = self._pilot(AsyncMock(side_effect=ValueError('tau2 has no samples with keys')))

    self.assertEqual(status, 1)
    self.assertIn('tau2: failed: ValueError', output)
    self.assertEqual(self._status('benchmarks:pilot:tau2'), StepStatus.FAILED)


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


class IndexCliTest(_ConfigDirTestCase):
  """Runs `index` with the embedding model and the vector store replaced by fakes."""

  def _index(self, *args: str, environment: dict[str, str] | None = None) -> tuple[int, str]:
    if environment is None:
      environment = {'AZURE_FOUNDRY_BASE_URL': 'https://foundry.example', 'AZURE_FOUNDRY_API_KEY': 'test-key'}
    with patch.dict(os.environ, environment), patch('cli.load_environment'):
      for name in ('AZURE_FOUNDRY_BASE_URL', 'AZURE_FOUNDRY_API_KEY'):
        if name not in environment:
          os.environ.pop(name, None)
      return _run(
        [
          'index',
          *args,
          '--config-dir',
          str(self.config_dir),
          '--data-dir',
          str(self.data_dir),
          '--manifest',
          str(self.manifest),
        ]
      )

  def _pin_wixqa_knowledge_base(self, articles: list[dict[str, str]]) -> None:
    content = ''.join(json.dumps(article) + '\n' for article in articles).encode()
    (self.data_dir / 'wixqa' / 'raw').mkdir(parents=True)
    (self.data_dir / 'wixqa' / 'raw' / 'kb.jsonl').write_bytes(content)
    path = self.config_dir / 'benchmarks.yaml'
    benchmarks = yaml.safe_load(path.read_text(encoding='utf-8'))
    pin = {'path': 'kb.jsonl', 'size': len(content), 'sha256': hashlib.sha256(content).hexdigest()}
    benchmarks['wixqa']['files'] = {'knowledge_base': pin}
    path.write_text(yaml.safe_dump(benchmarks), encoding='utf-8')

  def test_embed_writes_the_index_reports_progress_and_records_the_step(self) -> None:
    self._pin_wixqa_knowledge_base(
      [
        {'id': f'k{index}', 'url': f'https://e.x/{index}', 'title': 'T', 'contents': 'T\nB', 'article_type': 'article'}
        for index in range(3)
      ]
    )

    class Embedder:
      async def embed(self, texts: list[str], input_type: str) -> Embeddings:
        return Embeddings(vectors=[[0.25] * 8 for _ in texts], input_tokens=2 * len(texts))

    @asynccontextmanager
    async def fake_client(config: Any, settings: Any) -> AsyncIterator[object]:
      self.assertEqual(settings.api_key, 'test-key')
      yield Embedder()

    with patch('cli.open_embedding_client', fake_client):
      status, output = self._index('embed', '--only', 'wixqa')

    self.assertEqual(status, 0, output)
    self.assertIn('wixqa: shard 0 embedded; 3 documents done, 6 tokens billed by this run', output)
    self.assertTrue((self.data_dir / 'wixqa' / 'index' / 'report.json').exists())
    started, completed = Manifest(self.manifest).records()
    self.assertEqual(
      (started.step, started.status, completed.status), ('index:embed:wixqa', StepStatus.STARTED, StepStatus.COMPLETED)
    )
    self.assertEqual(
      completed.details,
      {
        'model': 'Cohere-Embed-V5-Fast',
        'output_dimension': 2048,
        'documents': 3,
        'shards': 1,
        'embedded_shards': 1,
        'input_tokens': 6,
        'run_input_tokens': 6,
      },
    )

  def test_a_failing_knowledge_base_is_recorded_and_the_next_one_still_embeds(self) -> None:
    @asynccontextmanager
    async def fake_client(config: Any, settings: Any) -> AsyncIterator[object]:
      yield object()

    outcome = EmbedOutcome(documents=5, shards=3, embedded_shards=2, input_tokens=50, run_input_tokens=30)
    embed = AsyncMock(side_effect=[EmbeddingError('The embedding request was rejected: 401 denied'), outcome])
    with patch('cli.open_embedding_client', fake_client), patch('cli.index_pipeline.embed_knowledge_base', embed):
      status, output = self._index('embed')

    self.assertEqual(status, 1)
    self.assertEqual([call.args[0] for call in embed.call_args_list], ['erb', 'wixqa'])
    self.assertIn('erb: failed: EmbeddingError: The embedding request was rejected: 401 denied', output)
    self.assertEqual(self._status('index:embed:erb'), StepStatus.FAILED)
    self.assertEqual(self._status('index:embed:wixqa'), StepStatus.COMPLETED)

  def test_embed_without_foundry_credentials_runs_nothing(self) -> None:
    errors = io.StringIO()

    with redirect_stderr(errors):
      status, _ = self._index('embed', environment={})

    self.assertEqual(status, 2)
    self.assertIn('AZURE_FOUNDRY_BASE_URL and AZURE_FOUNDRY_API_KEY', errors.getvalue())
    self.assertFalse(self.manifest.exists())

  def test_load_records_the_collection_it_filled(self) -> None:
    @asynccontextmanager
    async def fake_store(settings: Any) -> AsyncIterator[object]:
      yield object()

    load = AsyncMock(return_value=LoadOutcome(collection='wixqa-test', points=5))
    with patch('cli.open_vector_store', fake_store), patch('cli.index_pipeline.load_knowledge_base', load):
      status, output = self._index('load', '--only', 'wixqa')

    self.assertEqual(status, 0, output)
    self.assertIn('wixqa: 5 documents loaded into collection wixqa-test', output)
    record = Manifest(self.manifest).latest()['index:load:wixqa']
    self.assertEqual((record.status, record.details), (StepStatus.COMPLETED, {'collection': 'wixqa-test', 'points': 5}))

  def test_a_vector_store_failure_is_recorded_and_the_next_knowledge_base_still_loads(self) -> None:
    @asynccontextmanager
    async def fake_store(settings: Any) -> AsyncIterator[object]:
      yield object()

    outcome = LoadOutcome(collection='wixqa-test', points=5)
    load = AsyncMock(side_effect=[VectorStoreError('Qdrant could not count the points of erb-test: refused'), outcome])
    with patch('cli.open_vector_store', fake_store), patch('cli.index_pipeline.load_knowledge_base', load):
      status, output = self._index('load')

    self.assertEqual(status, 1)
    self.assertIn('erb: failed: VectorStoreError: Qdrant could not count the points of erb-test: refused', output)
    self.assertEqual(self._status('index:load:erb'), StepStatus.FAILED)
    self.assertEqual(self._status('index:load:wixqa'), StepStatus.COMPLETED)

  def test_only_benchmarks_with_a_knowledge_base_can_be_indexed(self) -> None:
    with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
      self._index('embed', '--only', 'tau2')


class SearchServerCliTest(unittest.TestCase):
  """Runs `search-server` with the server itself replaced by a fake."""

  def _serve(self, *args: str, environment: dict[str, str] | None = None) -> tuple[int, str, AsyncMock]:
    if environment is None:
      environment = {'AZURE_FOUNDRY_BASE_URL': 'https://foundry.example', 'AZURE_FOUNDRY_API_KEY': 'test-key'}
    errors = io.StringIO()
    with (
      patch.dict(os.environ, environment),
      patch('cli.load_environment'),
      patch('cli.logging.basicConfig'),
      patch('search_tool.server.serve', AsyncMock(side_effect=self.serve_error)) as serve,
      redirect_stderr(errors),
    ):
      for name in ('AZURE_FOUNDRY_BASE_URL', 'AZURE_FOUNDRY_API_KEY'):
        if name not in environment:
          os.environ.pop(name, None)
      status, _ = _run(['search-server', *args])
    return status, errors.getvalue(), serve

  def setUp(self) -> None:
    self.serve_error: Exception | None = None

  def test_serves_the_collection_with_the_given_options(self) -> None:
    status, _, serve = self._serve(
      '--collection', 'erb-69916e3', '--swap-fraction', '0.25', '--seed', '3', '--port', '8101'
    )

    self.assertEqual(status, 0)
    [call] = serve.await_args_list
    self.assertEqual(call.args[0], 'erb-69916e3')
    self.assertEqual(call.args[1].model, 'Cohere-Embed-V5-Fast')
    self.assertEqual(
      call.kwargs,
      {
        'swap_fraction': 0.25,
        'seed': 3,
        'max_document_chars': None,
        'call_log': DEFAULT_MANIFEST.parent / 'search_calls' / 'erb-69916e3-8101.jsonl',
        'host': '127.0.0.1',
        'port': 8101,
      },
    )

  def test_a_collection_it_cannot_serve_is_reported_before_serving(self) -> None:
    self.serve_error = ValueError('Collection erb-69916e3 was not loaded for hybrid search')

    status, errors, _ = self._serve('--collection', 'erb-69916e3')

    self.assertEqual(status, 1)
    self.assertIn('search-server: Collection erb-69916e3 was not loaded for hybrid search', errors)

  def test_without_foundry_credentials_nothing_is_served(self) -> None:
    status, errors, serve = self._serve('--collection', 'erb-69916e3', environment={})

    self.assertEqual(status, 2)
    self.assertIn('AZURE_FOUNDRY_BASE_URL and AZURE_FOUNDRY_API_KEY', errors)
    serve.assert_not_awaited()

  def test_rejects_a_swap_fraction_outside_zero_to_one(self) -> None:
    for fraction in ('1', '-0.1'):
      with self.subTest(fraction=fraction), self.assertRaises(SystemExit):
        self._serve('--collection', 'erb-69916e3', '--swap-fraction', fraction)


class DeepEvalCliTest(unittest.TestCase):
  """Runs `deepeval` with the database, judge and service replaced by fakes."""

  def setUp(self) -> None:
    self._directory = tempfile.TemporaryDirectory()
    self.manifest = Path(self._directory.name) / 'manifest.jsonl'
    self.source_run, self.run_id = uuid4(), uuid4()
    self.metrics = {key: SimpleNamespace(name=f'deepeval_{key}') for key in METRIC_KEYS}
    self.service = MagicMock()
    self.service.__aenter__.return_value = self.service
    self.service.repeat_evaluation = AsyncMock()
    self.build_service = MagicMock(return_value=self.service)
    self.db_manager = MagicMock()
    self.db_manager.health_check = AsyncMock(return_value=True)
    # The computations the finished run persisted, as the CLI reads them back.
    self.computations: list[SimpleNamespace] = []
    self.environment = {'GOOGLE_API_KEY': 'test-key'}
    self.configuration = SimpleNamespace(id='wixqa/react/gpt')

  def tearDown(self) -> None:
    self._directory.cleanup()

  def _finishing(self, status: EvaluationStatus) -> None:
    run = SimpleNamespace(id=self.run_id, status=status)
    self.service.repeat_evaluation.return_value = SimpleNamespace(run=run, execute=AsyncMock(return_value=run))

  def _computation(self, metric: str, status: MetricComputationStatus, failure: str | None = None) -> None:
    metadata: dict[str, Any] = {'failure': failure} if failure else {'judge_calls': 1}
    self.computations.append(SimpleNamespace(metric=metric, status=status, metadata=metadata))

  def _deepeval(self, *args: str) -> tuple[int, str]:
    @asynccontextmanager
    async def fake_database(settings: Any) -> AsyncIterator[object]:
      yield self.db_manager

    @asynccontextmanager
    async def fake_judge(config: Any, settings: Any) -> AsyncIterator[object]:
      yield object()

    uow = MagicMock()
    uow.__aenter__.return_value = uow
    uow.metric_computations.list_by_evaluation_run = AsyncMock(return_value=self.computations)
    with (
      patch.dict(os.environ, self.environment),
      patch('cli.load_environment'),
      patch('cli.open_database', fake_database),
      patch('cli.open_deepeval_judge', fake_judge),
      patch('cli.deepeval_metrics', return_value=self.metrics) as deepeval_metrics,
      patch('cli.build_service', self.build_service),
      patch('cli.UnitOfWork', return_value=uow),
      patch('cli._collection_run', AsyncMock(return_value=SimpleNamespace(configuration=self.configuration))),
    ):
      # GeminiJudgeSettings reads the key under either name.
      for name in ('GOOGLE_API_KEY', 'GEMINI_API_KEY'):
        if name not in self.environment:
          os.environ.pop(name, None)
      status, output = _run(['deepeval', '--source-run', str(self.source_run), '--manifest', str(self.manifest), *args])
    if deepeval_metrics.called:
      self.assertEqual(deepeval_metrics.call_args.kwargs, {'output_token_limit': 65536})
    return status, output

  def test_repeats_the_source_run_with_the_selected_metrics_and_records_the_run(self) -> None:
    self._finishing(EvaluationStatus.COMPLETED)
    self._computation('deepeval_precision', MetricComputationStatus.COMPLETED)
    self._computation('deepeval_precision', MetricComputationStatus.FAILED, 'misaligned')
    self._computation('deepeval_answer', MetricComputationStatus.FAILED, 'invalid_output')

    status, output = self._deepeval('--metrics', 'answer', 'precision', '--max-concurrent-samples', '1')

    self.assertEqual(status, 0, output)
    self.service.repeat_evaluation.assert_awaited_once_with(
      source_run_id=self.source_run,
      metrics=['deepeval_precision', 'deepeval_answer'],
      max_concurrent_samples=1,
      metadata={
        'step': f'deepeval:{self.source_run}:precision+answer',
        'configuration': 'wixqa/react/gpt',
        'arm': 'deepeval',
        'repetition': 1,
      },
    )
    self.assertEqual(
      self.build_service.call_args.kwargs['metrics'], [self.metrics['precision'], self.metrics['answer']]
    )
    records = Manifest(self.manifest).records()
    self.assertEqual([record.step for record in records], [f'deepeval:{self.source_run}:precision+answer'] * 2)
    self.assertEqual([record.status for record in records], [StepStatus.STARTED, StepStatus.COMPLETED])
    self.assertEqual([record.run_ids for record in records], [(self.run_id,)] * 2)
    # Judge outcomes count as wrong decisions, so failed units of these classes leave the step complete.
    self.assertEqual(
      records[-1].details,
      {
        'source_run_id': str(self.source_run),
        'deepeval_version': DEEPEVAL_VERSION,
        'metrics': ['deepeval_precision', 'deepeval_answer'],
        'run_status': 'COMPLETED',
        'failed_units': {'deepeval_answer': {'invalid_output': 1}, 'deepeval_precision': {'misaligned': 1}},
      },
    )
    self.assertIn(str(self.run_id), output)
    self.assertIn('deepeval_precision failed units: 1 misaligned', output)

  def test_units_that_failed_without_a_judge_outcome_fail_the_step(self) -> None:
    self._finishing(EvaluationStatus.COMPLETED)
    # A metric that raised records no failure class.
    for failure in ('provider', 'trace_load', None):
      with self.subTest(failure=failure):
        self.manifest = Path(self._directory.name) / f'manifest-{failure}.jsonl'
        self.computations = []
        self._computation('deepeval_recall', MetricComputationStatus.FAILED, 'timeout')
        self._computation('deepeval_recall', MetricComputationStatus.FAILED, failure)

        status, output = self._deepeval('--metrics', 'recall')

        self.assertEqual(status, 1)
        self.assertIn('1 units failed without a judge outcome', output)
        record = Manifest(self.manifest).latest()[f'deepeval:{self.source_run}:recall']
        self.assertEqual((record.status, record.run_ids), (StepStatus.FAILED, (self.run_id,)))
        self.assertEqual(
          record.details['failed_units'], {'deepeval_recall': {'timeout': 1, failure or 'unclassified': 1}}
        )

  def test_a_run_that_does_not_complete_fails_the_step(self) -> None:
    self._finishing(EvaluationStatus.PARTIALLY_COMPLETED)

    status, _ = self._deepeval()

    self.assertEqual(status, 1)
    record = Manifest(self.manifest).latest()[f'deepeval:{self.source_run}:precision+recall+answer']
    self.assertEqual((record.status, record.details['run_status']), (StepStatus.FAILED, 'PARTIALLY_COMPLETED'))

  def test_a_source_run_that_cannot_be_repeated_is_recorded_as_failed(self) -> None:
    self.service.repeat_evaluation.side_effect = NotFoundError('evaluation_run', self.source_run)

    status, output = self._deepeval('--metrics', 'recall')

    self.assertEqual(status, 1)
    self.assertIn('deepeval: failed: NotFoundError', output)
    [record] = Manifest(self.manifest).records()
    self.assertEqual((record.step, record.status), (f'deepeval:{self.source_run}:recall', StepStatus.FAILED))

  def test_an_unreachable_database_is_recorded_as_failed(self) -> None:
    self.db_manager.health_check.return_value = False

    status, output = self._deepeval()

    self.assertEqual(status, 1)
    self.assertIn('deepeval: failed: ConnectionError: Database health check failed', output)
    self.build_service.assert_not_called()
    [record] = Manifest(self.manifest).records()
    self.assertEqual(record.status, StepStatus.FAILED)

  def test_a_failure_after_the_run_started_keeps_the_run_and_details(self) -> None:
    self._finishing(EvaluationStatus.COMPLETED)
    handle = self.service.repeat_evaluation.return_value
    handle.execute.side_effect = psycopg.OperationalError('server closed the connection')

    status, _ = self._deepeval('--metrics', 'precision')

    self.assertEqual(status, 1)
    started, failed = Manifest(self.manifest).records()
    self.assertEqual((started.status, failed.status), (StepStatus.STARTED, StepStatus.FAILED))
    self.assertEqual(failed.run_ids, (self.run_id,))
    self.assertEqual(failed.details['metrics'], ['deepeval_precision'])
    self.assertEqual(failed.details['error'], 'OperationalError: server closed the connection')

  def test_without_a_judge_key_nothing_runs(self) -> None:
    self.environment = {}
    errors = io.StringIO()

    with redirect_stderr(errors):
      status, _ = self._deepeval()

    self.assertEqual(status, 2)
    self.assertIn('GOOGLE_API_KEY', errors.getvalue())
    self.db_manager.health_check.assert_not_awaited()
    self.assertFalse(self.manifest.exists())

  def test_rejects_scoring_fewer_than_one_sample_at_a_time(self) -> None:
    with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
      self._deepeval('--max-concurrent-samples', '0')


class _LangChainCliTestCase(unittest.TestCase):
  """Runs a LangChain baseline command with the database, judge, source run and service replaced by fakes."""

  COMMAND = ''
  METRIC = ''
  JUDGE = {'model': 'gemini-3.8-flash', 'temperature': 1.0, 'thinking_level': None}
  # The source run's configuration, as _collection_run reads it.
  configuration: SimpleNamespace

  def setUp(self) -> None:
    self._directory = tempfile.TemporaryDirectory()
    self.manifest = Path(self._directory.name) / 'manifest.jsonl'
    self.source_run, self.run_id = uuid4(), uuid4()
    self.judge = SimpleNamespace(verdicts=None)
    self.service = MagicMock()
    self.service.__aenter__.return_value = self.service
    self.service.repeat_evaluation = AsyncMock()
    self.build_service = MagicMock(return_value=self.service)
    self.db_manager = MagicMock()
    self.db_manager.health_check = AsyncMock(return_value=True)
    # What the new run persisted, as the CLI reads it back: its computations and its unresolved samples.
    self.computations: list[SimpleNamespace] = []
    self.unresolved: list[UUID] = []
    # The samples that a retry of the step's latest run scores again.
    self.retried = [uuid4()]
    self.unresolved_samples = AsyncMock()
    self.retry_samples = AsyncMock()
    self.environment = {'GOOGLE_API_KEY': 'test-key'}

  def tearDown(self) -> None:
    self._directory.cleanup()

  def _finishing(self, status: EvaluationStatus) -> None:
    run = SimpleNamespace(id=self.run_id, status=status)
    self.service.repeat_evaluation.return_value = SimpleNamespace(run=run, execute=AsyncMock(return_value=run))

  def _computation(self, status: MetricComputationStatus, failure: str | None = None) -> None:
    metadata: dict[str, Any] = {'failure': failure} if failure else {'judge_calls': 1}
    self.computations.append(SimpleNamespace(metric=self.METRIC, status=status, metadata=metadata))

  def _command(self, *args: str, patches: Sequence[Any] = ()) -> tuple[int, str]:
    @asynccontextmanager
    async def fake_database(settings: Any) -> AsyncIterator[object]:
      yield self.db_manager

    @asynccontextmanager
    async def fake_judge(config: Any, settings: Any, verdict_store: Path | None = None) -> AsyncIterator[object]:
      yield self.judge

    uow = MagicMock()
    uow.__aenter__.return_value = uow
    # Only the new run holds the baseline's computations.
    uow.metric_computations.list_by_evaluation_run = AsyncMock(
      side_effect=lambda run_id: self.computations if run_id == self.run_id else []
    )
    self.unresolved_samples = AsyncMock(return_value=self.unresolved)
    self.retry_samples = AsyncMock(return_value=self.retried)
    collection_run = AsyncMock(return_value=SimpleNamespace(configuration=self.configuration))
    with ExitStack() as stack:
      stack.enter_context(patch.dict(os.environ, self.environment))
      for patcher in (
        patch('cli.load_environment'),
        patch('cli.open_database', fake_database),
        patch('cli.open_openevals_judge', fake_judge),
        patch('cli._collection_run', collection_run),
        patch('cli._unresolved_samples', self.unresolved_samples),
        patch('cli._retry_samples', self.retry_samples),
        patch('cli.build_service', self.build_service),
        patch('cli.UnitOfWork', return_value=uow),
        *patches,
      ):
        stack.enter_context(patcher)
      # GeminiJudgeSettings reads the key under either name.
      for name in ('GOOGLE_API_KEY', 'GEMINI_API_KEY'):
        if name not in self.environment:
          os.environ.pop(name, None)
      status, output = _run(
        [self.COMMAND, '--source-run', str(self.source_run), '--manifest', str(self.manifest), *args]
      )
    if collection_run.called:
      self.assertEqual(collection_run.call_args.args[2], self.source_run)
    return status, output


class AgentEvalsCliTest(_LangChainCliTestCase):
  COMMAND = 'agentevals'
  METRIC = 'agentevals_trajectory_accuracy'

  def setUp(self) -> None:
    super().setUp()
    self.metric = SimpleNamespace(name=self.METRIC)
    self.configuration = SimpleNamespace(id='tau2/llm-agent/gpt/trial-1', benchmark='tau2')
    self.step = f'agentevals:{self.source_run}'

  def _agentevals(self, *args: str) -> tuple[int, str]:
    metric_class = MagicMock(return_value=self.metric)
    status, output = self._command(*args, patches=[patch('cli.AgentEvalsTrajectoryAccuracy', metric_class)])
    if metric_class.called:
      self.assertEqual(metric_class.call_args.kwargs, {'judge': self.judge, 'output_token_limit': 65536})
    return status, output

  def test_repeats_the_source_run_with_trajectory_accuracy_and_records_the_run(self) -> None:
    self._finishing(EvaluationStatus.COMPLETED)
    self._computation(MetricComputationStatus.COMPLETED)
    self._computation(MetricComputationStatus.FAILED, 'invalid_output')

    status, output = self._agentevals('--max-concurrent-samples', '1')

    self.assertEqual(status, 0, output)
    self.service.repeat_evaluation.assert_awaited_once_with(
      source_run_id=self.source_run,
      metrics=[self.METRIC],
      max_concurrent_samples=1,
      sample_ids=None,
      metadata={
        'step': self.step,
        'configuration': self.configuration.id,
        'arm': 'agentevals',
        'repetition': 1,
        'judge': self.JUDGE,
      },
    )
    self.assertEqual(self.build_service.call_args.kwargs['metrics'], [self.metric])
    self.unresolved_samples.assert_awaited_once_with(self.db_manager, self.run_id, [self.METRIC])
    records = Manifest(self.manifest).records()
    self.assertEqual([record.step for record in records], [self.step] * 2)
    self.assertEqual([record.status for record in records], [StepStatus.STARTED, StepStatus.COMPLETED])
    self.assertEqual([record.run_ids for record in records], [(self.run_id,)] * 2)
    # A judge outcome counts as a wrong decision, so a unit failed with one leaves its sample resolved.
    self.assertEqual(
      records[-1].details,
      {
        'source_run_id': str(self.source_run),
        'agentevals_version': AGENTEVALS_VERSION,
        'openevals_version': OPENEVALS_VERSION,
        'judge': self.JUDGE,
        'configuration': self.configuration.id,
        'metrics': [self.METRIC],
        'run_status': 'COMPLETED',
        'failed_units': {self.METRIC: {'invalid_output': 1}},
        'unresolved': [],
      },
    )
    self.assertIn(str(self.run_id), output)
    self.assertIn(f'{self.METRIC} failed units: 1 invalid_output', output)

  def test_a_run_of_another_benchmark_is_refused_before_scoring(self) -> None:
    self.configuration = SimpleNamespace(id='erb/react/gpt', benchmark='erb')

    status, output = self._agentevals()

    self.assertEqual(status, 1)
    self.assertIn('agentevals: failed: ValueError', output)
    self.build_service.assert_not_called()
    [record] = Manifest(self.manifest).records()
    self.assertEqual((record.step, record.status), (self.step, StepStatus.FAILED))

  def test_unresolved_samples_fail_the_step_and_name_the_run_to_retry(self) -> None:
    self._finishing(EvaluationStatus.PARTIALLY_COMPLETED)
    self._computation(MetricComputationStatus.FAILED, 'timeout')
    self._computation(MetricComputationStatus.FAILED, 'provider')
    self.unresolved = [uuid4()]

    status, output = self._agentevals()

    self.assertEqual(status, 1)
    self.assertIn(f'1 samples are unresolved; score them again with --retry {self.run_id}', output)
    record = Manifest(self.manifest).latest()[self.step]
    self.assertEqual((record.status, record.run_ids), (StepStatus.FAILED, (self.run_id,)))
    self.assertEqual(record.details['run_status'], 'PARTIALLY_COMPLETED')
    self.assertEqual(record.details['unresolved'], [str(self.unresolved[0])])
    self.assertEqual(record.details['failed_units'], {self.METRIC: {'provider': 1, 'timeout': 1}})

  def test_a_retry_scores_the_unresolved_samples_again_and_completes_the_step(self) -> None:
    first = uuid4()
    Manifest(self.manifest).append(self.step, StepStatus.FAILED, run_ids=(first,), details={'unresolved': ['s']})
    self._finishing(EvaluationStatus.COMPLETED)

    status, output = self._agentevals('--retry', str(first))

    self.assertEqual(status, 0, output)
    self.retry_samples.assert_awaited_once_with(
      self.db_manager, first, self.source_run, [self.METRIC], judge=self.JUDGE
    )
    retry_step = f'{self.step}:retry-1'
    kwargs = self.service.repeat_evaluation.call_args.kwargs
    self.assertEqual((kwargs['sample_ids'], kwargs['metadata']['step']), (self.retried, retry_step))
    latest = Manifest(self.manifest).latest()
    self.assertEqual((latest[retry_step].status, latest[retry_step].run_ids), (StepStatus.COMPLETED, (self.run_id,)))
    # Its runs are every attempt in order: for each sample, the latest run that scored it holds its results.
    self.assertEqual(
      (latest[self.step].status, latest[self.step].run_ids), (StepStatus.COMPLETED, (first, self.run_id))
    )
    self.assertEqual(latest[self.step].details['unresolved'], [])

  def test_a_retry_must_name_the_latest_attempt_of_the_step(self) -> None:
    for earlier in ((), (uuid4(),)):
      with self.subTest(earlier=earlier):
        self.manifest = Path(self._directory.name) / f'manifest-{len(earlier)}.jsonl'
        if earlier:
          Manifest(self.manifest).append(self.step, StepStatus.FAILED, run_ids=earlier)
        errors = io.StringIO()

        with redirect_stderr(errors):
          status, _ = self._agentevals('--retry', str(uuid4()))

        self.assertEqual(status, 2)
        self.assertIn(f'--retry must name the latest attempt of {self.step}', errors.getvalue())
        self.db_manager.health_check.assert_not_awaited()

  def test_a_failure_after_the_run_started_keeps_the_run_and_details(self) -> None:
    self._finishing(EvaluationStatus.COMPLETED)
    handle = self.service.repeat_evaluation.return_value
    handle.execute.side_effect = psycopg.OperationalError('server closed the connection')

    status, _ = self._agentevals()

    self.assertEqual(status, 1)
    started, failed = Manifest(self.manifest).records()
    self.assertEqual((started.status, failed.status), (StepStatus.STARTED, StepStatus.FAILED))
    self.assertEqual(failed.run_ids, (self.run_id,))
    self.assertEqual(failed.details['metrics'], [self.METRIC])
    self.assertEqual(failed.details['error'], 'OperationalError: server closed the connection')

  def test_an_unreachable_database_is_recorded_as_failed(self) -> None:
    self.db_manager.health_check.return_value = False

    status, output = self._agentevals()

    self.assertEqual(status, 1)
    self.assertIn('agentevals: failed: ConnectionError: Database health check failed', output)
    self.build_service.assert_not_called()
    [record] = Manifest(self.manifest).records()
    self.assertEqual(record.status, StepStatus.FAILED)

  def test_without_a_judge_key_nothing_runs(self) -> None:
    self.environment = {}
    errors = io.StringIO()

    with redirect_stderr(errors):
      status, _ = self._agentevals()

    self.assertEqual(status, 2)
    self.assertIn('GOOGLE_API_KEY', errors.getvalue())
    self.db_manager.health_check.assert_not_awaited()
    self.assertFalse(self.manifest.exists())

  def test_rejects_scoring_fewer_than_one_sample_at_a_time(self) -> None:
    with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
      self._agentevals('--max-concurrent-samples', '0')


class OpenEvalsCliTest(_LangChainCliTestCase):
  COMMAND = 'openevals'
  METRIC = 'openevals_correctness'
  ERB_METRICS = ['openevals_retrieval_relevance', 'openevals_groundedness_gold_claims', 'openevals_correctness']

  def setUp(self) -> None:
    super().setUp()
    self.configuration = SimpleNamespace(id='erb/react/gpt', benchmark='erb')

  def _openevals(self, *args: str) -> tuple[int, str]:
    factory = MagicMock(wraps=openevals_metrics)
    status, output = self._command(*args, patches=[patch('cli.openevals_metrics', factory)])
    if factory.called:
      self.assertEqual(factory.call_args.args, (self.configuration.benchmark, self.judge))
      self.assertEqual(factory.call_args.kwargs, {'output_token_limit': 65536})
    return status, output

  def _scored_metrics(self) -> list[str]:
    return [metric.name for metric in self.build_service.call_args.kwargs['metrics']]

  def test_scores_every_metric_of_the_run_benchmark_by_default_and_records_the_run(self) -> None:
    self._finishing(EvaluationStatus.COMPLETED)
    self._computation(MetricComputationStatus.FAILED, 'invalid_output')

    status, output = self._openevals('--max-concurrent-samples', '1')

    self.assertEqual(status, 0, output)
    step = f'openevals:{self.source_run}'
    self.service.repeat_evaluation.assert_awaited_once_with(
      source_run_id=self.source_run,
      metrics=self.ERB_METRICS,
      max_concurrent_samples=1,
      sample_ids=None,
      metadata={
        'step': step,
        'configuration': self.configuration.id,
        'arm': 'openevals',
        'repetition': 1,
        'judge': self.JUDGE,
      },
    )
    self.assertEqual(self._scored_metrics(), self.ERB_METRICS)
    records = Manifest(self.manifest).records()
    self.assertEqual(
      [(record.step, record.status) for record in records], [(step, StepStatus.STARTED), (step, StepStatus.COMPLETED)]
    )
    self.assertEqual(
      records[-1].details,
      {
        'source_run_id': str(self.source_run),
        'openevals_version': OPENEVALS_VERSION,
        'judge': self.JUDGE,
        'configuration': self.configuration.id,
        'metrics': self.ERB_METRICS,
        'run_status': 'COMPLETED',
        'failed_units': {self.METRIC: {'invalid_output': 1}},
        'unresolved': [],
      },
    )

  def test_each_benchmark_gets_a_metric_for_each_judge_metric_of_its_main_arm(self) -> None:
    self._finishing(EvaluationStatus.COMPLETED)
    expected = {
      'wixqa': ['openevals_retrieval_relevance', 'openevals_groundedness_orbitals_claims', 'openevals_correctness'],
      'tau2': ['openevals_plan_adherence'],
    }
    for benchmark, metrics in expected.items():
      with self.subTest(benchmark=benchmark):
        self.configuration = SimpleNamespace(id=f'{benchmark}/agent/gpt', benchmark=benchmark)

        status, output = self._openevals()

        self.assertEqual(status, 0, output)
        self.assertEqual(self._scored_metrics(), metrics)

  def test_selected_metrics_name_the_step(self) -> None:
    self._finishing(EvaluationStatus.COMPLETED)

    status, output = self._openevals('--metrics', 'answer', 'relevance')

    self.assertEqual(status, 0, output)
    self.assertEqual(self._scored_metrics(), ['openevals_retrieval_relevance', 'openevals_correctness'])
    [step] = Manifest(self.manifest).latest()
    self.assertEqual(step, f'openevals:{self.source_run}:relevance+answer')

  def test_a_metric_the_run_benchmark_lacks_is_refused_before_scoring(self) -> None:
    status, output = self._openevals('--metrics', 'plan')

    self.assertEqual(status, 1)
    self.assertIn('openevals: failed: ValueError: OpenEvals has no plan metric for erb', output)
    self.build_service.assert_not_called()
    [record] = Manifest(self.manifest).records()
    self.assertEqual((record.step, record.status), (f'openevals:{self.source_run}:plan', StepStatus.FAILED))

  def test_unresolved_samples_fail_the_step(self) -> None:
    self._finishing(EvaluationStatus.COMPLETED)
    self._computation(MetricComputationStatus.FAILED, 'provider')
    self.unresolved = [uuid4()]

    status, output = self._openevals()

    self.assertEqual(status, 1)
    self.assertIn(f'score them again with --retry {self.run_id}', output)
    self.unresolved_samples.assert_awaited_once_with(self.db_manager, self.run_id, self.ERB_METRICS)
    record = Manifest(self.manifest).latest()[f'openevals:{self.source_run}']
    self.assertEqual((record.status, record.details['unresolved']), (StepStatus.FAILED, [str(self.unresolved[0])]))

  def test_without_a_judge_key_nothing_runs(self) -> None:
    self.environment = {}

    with redirect_stderr(io.StringIO()):
      status, _ = self._openevals()

    self.assertEqual(status, 2)
    self.db_manager.health_check.assert_not_awaited()
    self.assertFalse(self.manifest.exists())


if __name__ == '__main__':
  unittest.main()
