"""Command-line entry point: ``poetry run syllo-exp <command>``. Run it from the ``experiments/`` folder."""

import argparse
import asyncio
import faulthandler
import logging
import signal
import sys
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, NamedTuple
from uuid import UUID

import httpx
import psycopg

from syllo_eval.evaluation.judge.failures import JudgeFailureKind
from syllo_eval.infrastructure import DatabaseManager
from syllo_eval.infrastructure.arize import PhoenixClient
from syllo_eval.infrastructure.exceptions import ConnectionError as DatabaseConnectionError
from syllo_eval.infrastructure.exceptions import PersistenceError
from syllo_eval.infrastructure.unit_of_work import UnitOfWork
from syllo_eval.model import (
  EvaluationRunSample,
  EvaluationSampleStatus,
  EvaluationStatus,
  MetricComputation,
  MetricComputationStatus,
)
from syllo_eval.settings import GeminiJudgeSettings, OrbitalsSettings, Settings

import collect
from benchmarks import orbitals_claims
from benchmarks import steps as benchmark_steps
from benchmarks.download import FetchInProgressError, PinnedFileMismatchError
from config import (
  CONFIG_DIR,
  DATA_DIR,
  BenchmarkSource,
  ClaimExtractorConfig,
  Configuration,
  EmbeddingConfig,
  ExperimentConfig,
  load_config,
  load_environment,
)
from deepeval_baseline import DEEPEVAL_VERSION, METRIC_KEYS, deepeval_metrics, open_deepeval_judge
from indexing import pipeline as index_pipeline
from indexing.embedding import AzureFoundrySettings, Embedder, EmbeddingError, open_embedding_client
from indexing.vector_store import QdrantSettings, VectorStore, VectorStoreError, open_vector_store
from judge import ARMS, judge_metrics, open_judge_client
from manifest import Manifest, StepStatus
from run_outputs import RunOutputs
from service import build_service, open_database

DEFAULT_OUTPUTS = Path(__file__).resolve().parent / 'outputs'
DEFAULT_MANIFEST = DEFAULT_OUTPUTS / 'manifest.jsonl'
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
  'claims': 'Extract the claims of the expected answers with Orbitals, once, for the benchmarks that use them.',
}
_INDEX_STEPS = {
  'embed': 'Embed every knowledge-base document into data/<benchmark>/index, resuming an interrupted run.',
  'load': 'Load the embedded documents into the vector store, one collection per knowledge base.',
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
    _add_benchmark_step_arguments(command, benchmark_steps.BUILDERS)
  pilot = benchmark_commands.add_parser(
    'pilot',
    help='Import a few samples of a converted benchmark as its pilot dataset.',
    description=(
      'Import the given samples of a converted benchmark as <dataset>-pilot, which `collect --pilot` runs on; the '
      'dataset is also written to data/<benchmark>/pilot/.'
    ),
  )
  pilot.add_argument('--benchmark', required=True, choices=sorted(benchmark_steps.BUILDERS), help='Benchmark to pilot.')
  pilot.add_argument(
    '--samples', nargs='+', required=True, help='Sample keys, as samples.jsonl has them; for tau2, task ids.'
  )
  pilot.add_argument('--config-dir', type=Path, default=CONFIG_DIR, help='Folder with the YAML configs.')
  pilot.add_argument('--data-dir', type=Path, default=DATA_DIR, help='Folder for downloads and converted data.')
  pilot.add_argument('--manifest', type=Path, default=DEFAULT_MANIFEST, help='Manifest JSONL file.')

  index = commands.add_parser('index', help='Embed the knowledge bases and load them into the vector store.')
  index_commands = index.add_subparsers(dest='step', required=True)
  for step, description in _INDEX_STEPS.items():
    command = index_commands.add_parser(step, help=description, description=description)
    _add_benchmark_step_arguments(command, index_pipeline.KNOWLEDGE_BASES)

  collect = commands.add_parser(
    'collect',
    help='Run one agent configuration on its benchmark and store its traces.',
    description=(
      'Call the agent of one configuration on every sample of its benchmark, store and archive its traces, and '
      'compute the metrics that need no judge.'
    ),
  )
  collect.add_argument('--configuration', required=True, help='Configuration id, such as tau2/llm-agent/gpt/trial-1.')
  collect.add_argument(
    '--pilot',
    action='store_true',
    help='Run on the pilot dataset of the benchmark (`syllo-exp benchmarks pilot`) instead of the full one.',
  )
  collect.add_argument(
    '--phoenix-project',
    help="Phoenix project of the agent's spans and trace lookups (default: the benchmark's dataset name).",
  )
  collect.add_argument(
    '--max-concurrent-samples',
    type=_positive_int,
    help='Samples run at once (default: EVALUATION_MAX_CONCURRENT_SAMPLES, or 1).',
  )
  collect.add_argument('--config-dir', type=Path, default=CONFIG_DIR, help='Folder with the YAML configs.')
  collect.add_argument('--data-dir', type=Path, default=DATA_DIR, help='Folder of the fetched benchmark files.')
  collect.add_argument(
    '--outputs-dir', type=Path, default=DEFAULT_OUTPUTS, help='Folder for raw traces and agent outputs.'
  )
  collect.add_argument('--manifest', type=Path, default=DEFAULT_MANIFEST, help='Manifest JSONL file.')

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

  judge = commands.add_parser(
    'judge',
    help='Score the stored traces of a collection run with the metrics of one experiment arm.',
    description=(
      'Repeat a finished collection run on its stored traces with the judge metrics of one arm: main (Syllo-eval), '
      'sc (single-call ablation) or wt (whole-trace ablation).'
    ),
  )
  judge.add_argument('--source-run', type=UUID, required=True, help='Collection run whose stored traces are scored.')
  judge.add_argument('--arm', choices=ARMS, required=True, help='Experiment arm.')
  judge.add_argument(
    '--repetition', type=int, choices=(1, 2, 3), default=1, help='Pass of the main arm: 2 and 3 are the retests.'
  )
  scope = judge.add_mutually_exclusive_group()
  scope.add_argument('--pilot', action='store_true', help="Score only the benchmark's pilot samples (judge.yaml).")
  scope.add_argument(
    '--retry', type=UUID, metavar='JUDGE_RUN', help='Score again the unresolved samples of a run of the same step.'
  )
  judge.add_argument(
    '--max-concurrent-samples',
    type=_positive_int,
    help='Samples scored at once (default: EVALUATION_MAX_CONCURRENT_SAMPLES, or 1); cost runs use 1.',
  )
  judge.add_argument('--config-dir', type=Path, default=CONFIG_DIR, help='Folder with the YAML configs.')
  judge.add_argument('--data-dir', type=Path, default=DATA_DIR, help='Folder of the converted benchmarks.')
  judge.add_argument('--manifest', type=Path, default=DEFAULT_MANIFEST, help='Manifest JSONL file.')

  search = commands.add_parser(
    'search-server',
    help='Serve the search tool of one knowledge base over MCP.',
    description="Serve the agents' search tool over MCP at http://<host>:<port>/mcp.",
  )
  search.add_argument('--collection', required=True, help='Qdrant collection to search, such as erb-69916e3.')
  search.add_argument(
    '--swap-fraction',
    type=_fraction,
    default=0.0,
    help='Share of the results replaced by random documents of the same knowledge base (default: 0).',
  )
  search.add_argument('--seed', type=int, default=0, help='Seed of the random replacements (default: 0).')
  search.add_argument(
    '--max-document-chars', type=_positive_int, help='Cut each returned document at this length (default: no cut).'
  )
  search.add_argument(
    '--call-log',
    type=Path,
    help='JSON Lines file of every search (default: outputs/search_calls/<collection>-<port>.jsonl).',
  )
  search.add_argument('--host', default='127.0.0.1', help='Interface to listen on (default: 127.0.0.1).')
  search.add_argument('--port', type=_positive_int, default=8000, help='Port to listen on (default: 8000).')
  search.add_argument('--config-dir', type=Path, default=CONFIG_DIR, help='Folder with the YAML configs.')
  return parser


def _add_benchmark_step_arguments(command: argparse.ArgumentParser, benchmarks: Mapping[str, Any]) -> None:
  command.add_argument('--only', nargs='+', choices=sorted(benchmarks), help='Benchmarks to process (default: all).')
  command.add_argument('--config-dir', type=Path, default=CONFIG_DIR, help='Folder with the YAML configs.')
  command.add_argument('--data-dir', type=Path, default=DATA_DIR, help='Folder for downloads and converted data.')
  command.add_argument('--manifest', type=Path, default=DEFAULT_MANIFEST, help='Manifest JSONL file.')


def _positive_int(value: str) -> int:
  number = int(value)
  if number < 1:
    raise argparse.ArgumentTypeError(f'must be at least 1, got {number}')
  return number


def _fraction(value: str) -> float:
  number = float(value)
  if not 0 <= number < 1:
    raise argparse.ArgumentTypeError(f'must be at least 0 and below 1, got {number}')
  return number


def main(argv: Sequence[str] | None = None) -> int:
  faulthandler.register(signal.SIGUSR1, file=2)
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
  elif args.command == 'index':
    return asyncio.run(_run_index_step(args))
  elif args.command == 'search-server':
    return _serve_search(args)
  elif args.command == 'collect':
    return asyncio.run(_collect(args))
  elif args.command == 'deepeval':
    return asyncio.run(_score_with_deepeval(args))
  elif args.command == 'judge':
    return asyncio.run(_judge(args))
  return 0


def _run_benchmark_step(args: argparse.Namespace) -> int:
  """Run one step for each selected benchmark; a failing benchmark is recorded and the others still run."""
  if args.step == 'pilot':
    return asyncio.run(_import_pilot(args))
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
  elif args.step == 'claims':
    results = asyncio.run(_extract_claims(selected, config.models.claim_extractor, args.data_dir, manifest))
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


async def _extract_claims(
  selected: Sequence[tuple[str, BenchmarkSource]], config: ClaimExtractorConfig, data_dir: Path, manifest: Manifest
) -> list[bool]:
  load_environment()
  settings = OrbitalsSettings()
  if settings.api_key is None:
    print('Claim extraction needs ORBITALS_API_KEY.', file=sys.stderr)
    return [False]
  results = []
  async with httpx.AsyncClient() as client:
    for name, source in selected:
      description = config.ai_service_descriptions.get(name)
      if description is None:
        print(f'{name}: no Orbitals claims, its claims come from the benchmark')
        continue
      step = f'benchmarks:claims:{name}'
      manifest.append(step, StepStatus.STARTED)
      extract_batch = orbitals_claims.batch_extractor(client, settings, config, description)
      try:
        counts = await benchmark_steps.extract_claims(name, source, data_dir, extract_batch, config.batch_size)
      except (FileNotFoundError, ValueError, httpx.HTTPError) as error:
        results.append(_record_failure(manifest, step, name, error))
        continue
      details = {'model': config.model, 'ai_service_description': description, **counts}
      manifest.append(step, StepStatus.COMPLETED, details=details)
      print(f'{name}: {counts["claims"]} claims of {counts["samples"]} samples in {data_dir / name}')
      results.append(True)
  return results


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


async def _import_pilot(args: argparse.Namespace) -> int:
  source = load_config(args.config_dir).benchmarks.get(args.benchmark)
  if source is None:
    print(f'Not pinned in {args.config_dir / "benchmarks.yaml"}: {args.benchmark}', file=sys.stderr)
    return 2
  step = f'benchmarks:pilot:{args.benchmark}'
  manifest = Manifest(args.manifest)
  load_environment()
  async with open_database(Settings()) as db_manager:
    manifest.append(step, StepStatus.STARTED)
    try:
      outcome = await benchmark_steps.import_pilot(args.benchmark, source, args.data_dir, db_manager, args.samples)
    except (FileNotFoundError, ValueError, PersistenceError, psycopg.Error) as error:
      _record_failure(manifest, step, args.benchmark, error)
      return 1
  name = benchmark_steps.pilot_dataset_name(source)
  manifest.append(
    step,
    StepStatus.COMPLETED,
    details={'dataset_id': str(outcome.dataset.id), 'dataset_name': name, 'samples': args.samples},
  )
  action = 'imported' if outcome.created else 'already present'
  print(
    f'{args.benchmark}: pilot dataset {name} {action} ({outcome.dataset.id}) with samples {", ".join(args.samples)}'
  )
  return 0


async def _run_index_step(args: argparse.Namespace) -> int:
  """Embed or load each selected knowledge base; a failing one is recorded and the others still run."""
  config = load_config(args.config_dir)
  unpinned = sorted(set(args.only or ()) - set(config.benchmarks))
  if unpinned:
    print(f'Not pinned in {args.config_dir / "benchmarks.yaml"}: {", ".join(unpinned)}', file=sys.stderr)
    return 2
  selected = [
    (name, source)
    for name, source in config.benchmarks.items()
    if name in index_pipeline.KNOWLEDGE_BASES and (not args.only or name in args.only)
  ]
  embedding = config.models.embedding
  manifest = Manifest(args.manifest)
  load_environment()
  if args.step == 'embed':
    settings = AzureFoundrySettings()
    if settings.base_url is None or settings.api_key is None:
      print('Embedding needs AZURE_FOUNDRY_BASE_URL and AZURE_FOUNDRY_API_KEY.', file=sys.stderr)
      return 2
    async with open_embedding_client(embedding, settings) as client:
      results = [await _embed(name, source, args.data_dir, client, embedding, manifest) for name, source in selected]
  else:
    async with open_vector_store(QdrantSettings()) as store:
      results = [await _load(name, source, args.data_dir, store, embedding, manifest) for name, source in selected]
  return 0 if all(results) else 1


async def _embed(
  name: str, source: BenchmarkSource, data_dir: Path, embedder: Embedder, config: EmbeddingConfig, manifest: Manifest
) -> bool:
  step = f'index:embed:{name}'
  details: dict[str, Any] = {'model': config.model, 'output_dimension': config.output_dimension}
  # An interrupted run leaves the step started; running it again resumes at the first missing shard.
  manifest.append(step, StepStatus.STARTED, details=details)

  def report(progress: index_pipeline.ShardProgress) -> None:
    print(
      f'{name}: shard {progress.shard} embedded; {progress.documents:,} documents done, '
      f'{progress.input_tokens:,} tokens billed by this run',
      flush=True,
    )

  try:
    outcome = await index_pipeline.embed_knowledge_base(name, source, data_dir, embedder, config, progress=report)
  except (EmbeddingError, PinnedFileMismatchError, index_pipeline.IndexInProgressError, ValueError) as error:
    return _record_failure(manifest, step, name, error, details=details)
  details.update(
    documents=outcome.documents,
    shards=outcome.shards,
    embedded_shards=outcome.embedded_shards,
    input_tokens=outcome.input_tokens,
    run_input_tokens=outcome.run_input_tokens,
  )
  manifest.append(step, StepStatus.COMPLETED, details=details)
  print(
    f'{name}: {outcome.documents:,} documents embedded in {outcome.shards} shards, {outcome.embedded_shards} by this '
    f'run; {outcome.input_tokens:,} tokens billed in all; written to {data_dir / name / index_pipeline.INDEX_DIR}'
  )
  return True


async def _load(
  name: str, source: BenchmarkSource, data_dir: Path, store: VectorStore, config: EmbeddingConfig, manifest: Manifest
) -> bool:
  step = f'index:load:{name}'
  manifest.append(step, StepStatus.STARTED)
  try:
    outcome = await index_pipeline.load_knowledge_base(name, source, data_dir, store, config)
  except (FileNotFoundError, PinnedFileMismatchError, VectorStoreError, ValueError) as error:
    return _record_failure(manifest, step, name, error)
  manifest.append(step, StepStatus.COMPLETED, details={'collection': outcome.collection, 'points': outcome.points})
  print(f'{name}: {outcome.points:,} documents loaded into collection {outcome.collection}')
  return True


def _serve_search(args: argparse.Namespace) -> int:
  """Serve the search tool until interrupted; a collection or setting that cannot serve fails before it starts."""
  embedding = load_config(args.config_dir).models.embedding
  load_environment()
  foundry = AzureFoundrySettings()
  if foundry.base_url is None or foundry.api_key is None:
    print(
      'The search tool embeds queries, so it needs AZURE_FOUNDRY_BASE_URL and AZURE_FOUNDRY_API_KEY.', file=sys.stderr
    )
    return 2
  logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(name)s: %(message)s')
  # Each search would otherwise log every HTTP request and MCP message it makes, and FastMCP every failed search with
  # a full traceback; the search logs its own failures in one line.
  for noisy in ('httpx', 'mcp'):
    logging.getLogger(noisy).setLevel(logging.WARNING)
  logging.getLogger('fastmcp.fastmcp.tools.tool_manager').setLevel(logging.CRITICAL)
  call_log = args.call_log or DEFAULT_MANIFEST.parent / 'search_calls' / f'{args.collection}-{args.port}.jsonl'
  # Imported here: the MCP framework is slow to import, and only this command needs it.
  from search_tool import server as search_server

  try:
    asyncio.run(
      search_server.serve(
        args.collection,
        embedding,
        foundry,
        QdrantSettings(),
        swap_fraction=args.swap_fraction,
        seed=args.seed,
        max_document_chars=args.max_document_chars,
        call_log=call_log,
        host=args.host,
        port=args.port,
      )
    )
  except (ValueError, VectorStoreError) as error:
    print(f'search-server: {error}', file=sys.stderr)
    return 1
  return 0


async def _collect(args: argparse.Namespace) -> int:
  """Run one configuration on its whole benchmark; the manifest records the step and the run it made."""
  config = load_config(args.config_dir)
  try:
    configuration = config.configuration(args.configuration)
  except KeyError as error:
    print(f'collect: {error.args[0]}', file=sys.stderr)
    return 2
  source = config.benchmarks[configuration.benchmark]
  dataset_name = benchmark_steps.pilot_dataset_name(source) if args.pilot else source.dataset_name
  load_environment()
  logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(name)s: %(message)s')
  logging.getLogger('httpx').setLevel(logging.WARNING)
  settings = Settings()
  phoenix = collect.phoenix_settings(settings.phoenix, args.phoenix_project or source.dataset_name, configuration.agent)
  outputs = RunOutputs(args.outputs_dir)
  step = f'collect:{configuration.id}' + (':pilot' if args.pilot else '')
  manifest = Manifest(args.manifest)
  details: dict[str, Any] = {
    'agent': configuration.agent,
    'version_tag': configuration.version_tag,
    'dataset_name': dataset_name,
    'phoenix_project': phoenix.project_id,
  }
  try:
    integration = collect.build_integration(
      configuration, config, data_dir=args.data_dir, phoenix=phoenix, outputs=outputs
    )
    if integration.search_url is not None:
      await collect.check_search_server(configuration, config, integration.search_url)
  except (FileNotFoundError, PinnedFileMismatchError, ValueError) as error:
    _record_failure(manifest, step, 'collect', error, details=details)
    return 1
  metrics = collect.deterministic_metrics(configuration.benchmark)
  details['metrics'] = [metric.name for metric in metrics]
  run_ids: tuple[UUID, ...] = ()
  try:
    async with open_database(settings) as db_manager:
      if not await db_manager.health_check():
        raise DatabaseConnectionError('Database health check failed')
      async with UnitOfWork(db_manager) as uow:
        dataset = await uow.datasets.get_by_name(dataset_name)
      if dataset is None:
        step_to_run = 'pilot --benchmark' if args.pilot else 'import --only'
        raise ValueError(
          f'Dataset {dataset_name} is not imported: run `syllo-exp benchmarks {step_to_run} {configuration.benchmark}`'
        )
      service = build_service(
        settings,
        db_manager,
        metrics=metrics,
        callers_by_agent_name={configuration.agent: integration.caller},
        trace_client=collect.TraceArchive(PhoenixClient(phoenix), outputs),
        trace_adapters_by_agent_name={configuration.agent: integration.adapter},
      )
      async with service:
        handle = await service.create_evaluation(
          agent_name=configuration.agent,
          agent_version_tag=configuration.version_tag,
          dataset_id=dataset.id,
          max_concurrent_samples=args.max_concurrent_samples,
          selected_metric_names=details['metrics'],
        )
        outputs.bind(handle.run.id)
        run_ids = (handle.run.id,)
        # A crash past this point leaves the step started, with the run it made.
        manifest.append(step, StepStatus.STARTED, run_ids=run_ids, details=details)
        run = await handle.execute()
        counts = (await service.get_evaluation_status(run.id)).sample_counts
  except (ValueError, PersistenceError, psycopg.Error) as error:
    _record_failure(manifest, step, 'collect', error, run_ids=run_ids, details=details)
    return 1
  complete = run.status is EvaluationStatus.COMPLETED and counts.completed == counts.total
  details.update(
    run_status=run.status.value,
    samples={'total': counts.total, 'completed': counts.completed, 'failed': counts.failed},
    outputs=str(outputs.root),
  )
  manifest.append(step, StepStatus.COMPLETED if complete else StepStatus.FAILED, run_ids=run_ids, details=details)
  print(
    f'collect: run {run.id} {run.status.value}: {counts.completed} of {counts.total} samples completed, '
    f'{counts.failed} failed; raw traces in {outputs.root / "traces" / str(run.id)}'
  )
  if not complete:
    print('collect: failed: every sample must complete; the failed ones must run again')
  return 0 if complete else 1


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


async def _judge(args: argparse.Namespace) -> int:
  """Repeat a collection run with one arm's metrics; the manifest records the step, its run and unresolved samples."""
  if args.repetition != 1 and args.arm != 'main':
    print('judge: only the main arm has repetitions', file=sys.stderr)
    return 2
  config = load_config(args.config_dir)
  judge_config = config.models.judge
  load_environment()
  settings, judge_settings = Settings(), GeminiJudgeSettings()
  if judge_settings.api_key is None:
    print('The judge needs GOOGLE_API_KEY.', file=sys.stderr)
    return 2
  manifest = Manifest(args.manifest)
  base_step = f'judge:{args.source_run}:{args.arm}:{args.repetition}'
  chain = _step_chain(manifest, base_step)
  step = base_step
  if args.pilot:
    step = f'{base_step}:pilot'
  elif args.retry:
    # Retrying only the latest attempt keeps every other sample's latest attempt resolved, so the step is complete
    # exactly when a retry leaves nothing unresolved.
    if not chain or args.retry != chain[-1]:
      latest = f'its latest attempt is {chain[-1]}' if chain else 'it has no run yet'
      print(f'judge: --retry must name the latest attempt of {base_step}; {latest}', file=sys.stderr)
      return 2
    step = f'{base_step}:retry-{sum(name.startswith(f"{base_step}:retry-") for name in manifest.latest()) + 1}'
  details: dict[str, Any] = {
    'source_run_id': str(args.source_run),
    'judge': judge_config.model_dump(include={'model', 'temperature', 'thinking_level'}),
  }
  run_ids: tuple[UUID, ...] = ()
  try:
    async with open_database(settings) as db_manager, open_judge_client(judge_config, judge_settings) as judge:
      if not await db_manager.health_check():
        raise DatabaseConnectionError('Database health check failed')
      source = await _collection_run(db_manager, config, args.source_run)
      configuration = source.configuration
      metrics = judge_metrics(
        configuration.benchmark,
        args.arm,
        judge=judge,
        db_manager=db_manager,
        output_token_limit=judge_config.output_token_limit,
      )
      if not metrics:
        raise ValueError(f'the {args.arm} arm does not apply to {configuration.benchmark}')
      metric_names = [metric.name for metric in metrics]
      if args.retry:
        sample_ids: list[UUID] | None = await _retry_samples(db_manager, args.retry, args.source_run, metric_names)
      else:
        sample_ids = await _scoped_samples(db_manager, config, source, args)
      details.update(
        configuration=configuration.id,
        metrics=metric_names,
        collection_failures=source.collection_failures,
        samples=len(source.traced) if sample_ids is None else len(sample_ids),
      )
      async with build_service(settings, db_manager, metrics=metrics) as service:
        handle = await service.repeat_evaluation(
          source_run_id=args.source_run,
          metrics=metric_names,
          max_concurrent_samples=args.max_concurrent_samples,
          sample_ids=sample_ids,
        )
        run_ids = (handle.run.id,)
        # A crash past this point leaves the step started, with the run it made.
        manifest.append(step, StepStatus.STARTED, run_ids=run_ids, details=details)
        run = await handle.execute()
      failed_units = await _failed_units(db_manager, run.id)
      unresolved = await _unresolved_samples(db_manager, run.id, metric_names)
  except (ValueError, PersistenceError, psycopg.Error) as error:
    _record_failure(manifest, step, 'judge', error, run_ids=run_ids, details=details)
    return 1
  details.update(run_status=run.status.value, failed_units=failed_units, unresolved=[str(s) for s in unresolved])
  manifest.append(step, StepStatus.FAILED if unresolved else StepStatus.COMPLETED, run_ids=run_ids, details=details)
  if args.retry and not unresolved:
    # The real step completes once a retry resolves what its earlier runs left. Its runs are every attempt in order:
    # for each sample, the latest run that scored it holds its results.
    base = manifest.latest().get(base_step)
    manifest.append(
      base_step,
      StepStatus.COMPLETED,
      run_ids=tuple(dict.fromkeys((*chain, *run_ids))),
      details={**(base.details if base else {}), 'unresolved': []},
    )
  print(f'judge: run {run.id} {run.status.value}, {configuration.id} {args.arm} arm on the traces of {args.source_run}')
  for metric, kinds in failed_units.items():
    print(f'  {metric} failed units: ' + ', '.join(f'{count} {kind}' for kind, count in kinds.items()))
  if unresolved:
    print(f'judge: failed: {len(unresolved)} samples are unresolved; score them again with --retry {run.id}')
  return 1 if unresolved else 0


def _step_chain(manifest: Manifest, base_step: str) -> list[UUID]:
  """The runs of a judge step and of its retries, in the order they were made."""
  records = manifest.records()
  steps = [record for record in records if record.step == base_step or record.step.startswith(f'{base_step}:retry-')]
  return list(dict.fromkeys(run_id for record in steps for run_id in record.run_ids))


class _CollectionRun(NamedTuple):
  configuration: Configuration
  dataset_id: UUID
  traced: frozenset[UUID]
  collection_failures: int


async def _collection_run(db_manager: DatabaseManager, config: ExperimentConfig, run_id: UUID) -> _CollectionRun:
  """A finished collection run: its configuration, dataset, traced samples and samples without a trace."""
  async with UnitOfWork(db_manager) as uow:
    run = await uow.evaluation_runs.get_by_id_or_raise(run_id)
    dataset = await uow.datasets.get_by_id_or_raise(run.dataset_id)
    agent = await uow.agents.get_by_id_or_raise(run.agent_id)
    run_samples = await uow.evaluation_run_samples.list_by_evaluation_run(run_id)
  if run.source_run_id is not None:
    raise ValueError(f'run {run_id} repeats run {run.source_run_id}; pass the collection run')
  if run.status not in (EvaluationStatus.COMPLETED, EvaluationStatus.PARTIALLY_COMPLETED):
    raise ValueError(f'collection run {run_id} is {run.status.value}, not finished')
  benchmarks = {
    name
    for name, source in config.benchmarks.items()
    if dataset.name in (source.dataset_name, benchmark_steps.pilot_dataset_name(source))
  }
  agent_version = (agent.name, agent.version_tag)
  matches = [
    c for c in config.configurations if c.benchmark in benchmarks and (c.agent, c.version_tag) == agent_version
  ]
  if len(matches) != 1:
    raise ValueError(f'run {run_id} ({dataset.name}, {agent.name}/{agent.version_tag}) matches no configuration')
  traced = frozenset(run_sample.sample_id for run_sample in run_samples if run_sample.trace_id is not None)
  return _CollectionRun(matches[0], run.dataset_id, traced, len(run_samples) - len(traced))


async def _scoped_samples(
  db_manager: DatabaseManager, config: ExperimentConfig, source: _CollectionRun, args: argparse.Namespace
) -> list[UUID] | None:
  """The traced samples of a pilot or of the ERB whole-trace arm, by sample key; ``None`` for every traced one."""
  benchmark = source.configuration.benchmark
  keys: tuple[str, ...] | None = None
  if benchmark == 'erb' and args.arm == 'wt':
    keys = config.judge_samples.annotated_samples.get(benchmark, ())
  if args.pilot:
    pilot = config.judge_samples.pilot_samples.get(benchmark, ())
    keys = pilot if keys is None else tuple(key for key in pilot if key in keys)
  if keys is None:
    return None
  if not keys:
    raise ValueError(f'configs/judge.yaml lists no samples of {benchmark} for this step')
  prompts = benchmark_steps.sample_prompts(benchmark, config.benchmarks[benchmark], args.data_dir, keys)
  async with UnitOfWork(db_manager) as uow:
    samples = await uow.samples.list_by_dataset(source.dataset_id)
  ids = {sample.input_prompt: sample.id for sample in samples}
  missing = [key for key, prompt in zip(keys, prompts) if prompt not in ids]
  if missing:
    raise ValueError(f"the source run's dataset has no samples with keys {missing}")
  # Collection failures have no trace to judge; the manifest counts them with the run's.
  scoped = [ids[prompt] for prompt in prompts if ids[prompt] in source.traced]
  if not scoped:
    raise ValueError(f'none of the samples {list(keys)} has a trace in the source run')
  return scoped


async def _retry_samples(
  db_manager: DatabaseManager, judge_run_id: UUID, source_run_id: UUID, metric_names: Sequence[str]
) -> list[UUID]:
  async with UnitOfWork(db_manager) as uow:
    judge_run = await uow.evaluation_runs.get_by_id_or_raise(judge_run_id)
    judge_metric_names = await uow.evaluation_run_metrics.list_metrics(judge_run_id)
  if judge_run.source_run_id != source_run_id or sorted(judge_metric_names) != sorted(metric_names):
    raise ValueError(f'run {judge_run_id} is not a run of this step')
  unresolved = await _unresolved_samples(db_manager, judge_run_id, metric_names)
  if not unresolved:
    raise ValueError(f'run {judge_run_id} has no unresolved samples')
  return unresolved


async def _unresolved_samples(db_manager: DatabaseManager, run_id: UUID, metric_names: Sequence[str]) -> list[UUID]:
  async with UnitOfWork(db_manager) as uow:
    planned = await uow.evaluation_run_plan_samples.list_sample_ids(run_id)
    run_samples = await uow.evaluation_run_samples.list_by_evaluation_run(run_id)
    computations = await uow.metric_computations.list_by_evaluation_run(run_id)
  return unresolved_samples(planned, run_samples, computations, metric_names)


def unresolved_samples(
  planned: Sequence[UUID],
  run_samples: Sequence[EvaluationRunSample],
  computations: Sequence[MetricComputation],
  metric_names: Sequence[str],
) -> list[UUID]:
  """Planned samples that did not complete, or with a metric that has no outcome.

  Completed computations, persisted skips and judge outcomes are outcomes. A metric without any computation, or with
  one failed for another reason (provider, trace load, unclassified), is unresolved. A sample whose compute phase did
  not complete is unresolved even with computations, since an interruption can leave only some of them persisted.
  """
  completed = {
    run_sample.sample_id for run_sample in run_samples if run_sample.status is EvaluationSampleStatus.COMPLETED
  }
  sample_ids = {run_sample.id: run_sample.sample_id for run_sample in run_samples}
  resolved: defaultdict[tuple[UUID, str], list[bool]] = defaultdict(list)
  for computation in computations:
    resolved[(sample_ids[computation.evaluation_run_sample_id], computation.metric)].append(
      computation.status is not MetricComputationStatus.FAILED
      or (computation.metadata or {}).get('failure') in _JUDGE_OUTCOMES
    )
  return [
    sample_id
    for sample_id in planned
    if sample_id not in completed
    or not all(resolved[(sample_id, metric)] and all(resolved[(sample_id, metric)]) for metric in metric_names)
  ]


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
