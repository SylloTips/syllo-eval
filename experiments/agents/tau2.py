"""τ2-bench's own agent: a caller that runs one traced simulation per sample, and the adapter of its traces.

The caller runs ``tau2_runtime/run_simulation.py`` with the interpreter of the tau2 environment, one process per
simulation, because tau2's dependencies conflict with syllo-eval's. The process sends the simulation's spans to Phoenix
and writes the simulation, with its reward, to ``<outputs>/tau2/<run id>/simulations/``. The caller adds one line per
simulation to ``rewards.jsonl`` next to it, keyed by run, sample and request id.

The adapter makes tau2's agent span the agent root. Its request is the task's user scenario, the sample's prompt; its
executed steps are the agent's tool calls, with their arguments rendered as the dataset renders the expected plan.
"""

import asyncio
import json
import logging
import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Literal
from uuid import uuid4

from pydantic import Field
from pydantic_settings import SettingsConfigDict

from syllo_eval.evaluation.trace_adapter import PhoenixTraceAdapter, TraceProcessingResult
from syllo_eval.model import Sample, Span
from syllo_eval.settings import EnvSettings, PhoenixSettings
from syllo_eval.trace_semantics import Answer, ExecutionStep, PlanningData, SpanSemantics

from benchmarks.download import verify_pinned_files
from benchmarks.tau2 import render_arguments, render_user_scenario
from config import EXPERIMENTS_DIR, Configuration, ExperimentConfig, ModelRef
from indexing.embedding import AzureFoundrySettings
from run_outputs import RunOutputs

ROOT_SPAN_NAME = 'tau2.simulation'
RUNNER = EXPERIMENTS_DIR / 'tau2_runtime' / 'run_simulation.py'
# run_simulation.py's exit code for a failure that a new attempt can avoid.
_EXIT_TRANSIENT = 75

logger = logging.getLogger(__name__)


class Tau2Settings(EnvSettings):
  """How the tau2 runtime runs here; what it runs is in ``configs/tau2.yaml``."""

  model_config = SettingsConfigDict(env_prefix='TAU2_')

  python: Path = EXPERIMENTS_DIR / '.venv-tau2' / 'bin' / 'python'
  # Attempts per sample at a transient failure, each a new simulation with a new request id.
  max_attempts: int = Field(default=3, ge=1)
  retry_delay_seconds: float = Field(default=60.0, ge=0)
  # For litellm's azure/ models, served by the Azure AI Foundry resource of AZURE_FOUNDRY_BASE_URL.
  azure_api_version: str = Field(default='2025-04-01-preview', min_length=1)


class Tau2SimulationError(RuntimeError):
  """A simulation that did not finish: the sample fails."""


class Tau2Caller:
  """Runs tau2's agent on the task of a sample and returns the simulation's request id."""

  def __init__(
    self,
    *,
    python: Path,
    spec: Mapping[str, Any],
    task_ids: Mapping[str, str],
    outputs: RunOutputs,
    environment: Mapping[str, str],
    max_attempts: int = 3,
    retry_delay_seconds: float = 60.0,
    runner: Path = RUNNER,
  ):
    self._python = python
    self._runner = runner
    # Everything run_simulation.py needs except the request and the task.
    self._spec = dict(spec)
    self._task_ids = dict(task_ids)
    self._outputs = outputs
    self._environment = dict(environment)
    self._max_attempts = max_attempts
    self._retry_delay_seconds = retry_delay_seconds

  async def call(self, sample: Sample) -> str:
    task_id = self._task_ids.get(sample.input_prompt)
    if task_id is None:
      raise ValueError(f'Sample {sample.id} is not the user scenario of a tau2 retail task')
    attempt = 1
    while True:
      request_id = uuid4().hex
      code, log = await self._simulate(sample, task_id, request_id, attempt)
      if code == 0:
        self._record_reward(sample, task_id, request_id, attempt)
        return request_id
      if code != _EXIT_TRANSIENT or attempt == self._max_attempts:
        raise Tau2SimulationError(f'tau2 task {task_id} failed in attempt {attempt} with exit code {code}; see {log}')
      logger.warning(
        'tau2 task %s: attempt %d failed transiently (see %s); retrying in %gs',
        task_id,
        attempt,
        log,
        self._retry_delay_seconds,
      )
      await asyncio.sleep(self._retry_delay_seconds)
      attempt += 1

  async def _simulate(self, sample: Sample, task_id: str, request_id: str, attempt: int) -> tuple[int, Path]:
    directory = self._outputs.directory('tau2') / 'simulations'
    directory.mkdir(exist_ok=True)
    spec = {
      **self._spec,
      'request_id': request_id,
      'task_id': task_id,
      'metadata': {**self._spec.get('metadata', {}), 'sample_id': str(sample.id), 'attempt': attempt},
    }
    spec_path, log_path = directory / f'{request_id}.spec.json', directory / f'{request_id}.log'
    spec_path.write_text(json.dumps(spec, indent=2), encoding='utf-8')
    with log_path.open('wb') as log:
      process = await asyncio.create_subprocess_exec(
        str(self._python),
        str(self._runner),
        '--spec',
        str(spec_path),
        '--result',
        str(directory / f'{request_id}.json'),
        stdin=asyncio.subprocess.DEVNULL,
        stdout=log,
        stderr=asyncio.subprocess.STDOUT,
        env=self._environment,
      )
      try:
        return await process.wait(), log_path
      except BaseException:
        await _stop(process)
        raise

  def _record_reward(self, sample: Sample, task_id: str, request_id: str, attempt: int) -> None:
    directory = self._outputs.directory('tau2')
    simulation = json.loads((directory / 'simulations' / f'{request_id}.json').read_text(encoding='utf-8'))
    reward_info = simulation['reward_info']
    record = {
      'run_id': str(self._outputs.run_id),
      'sample_id': str(sample.id),
      'request_id': request_id,
      'task_id': task_id,
      **self._spec.get('metadata', {}),
      'seed': self._spec['seed'],
      'attempt': attempt,
      'termination_reason': simulation['termination_reason'],
      'reward': reward_info['reward'],
      'reward_info': reward_info,
      'agent_cost': simulation.get('agent_cost'),
      'user_cost': simulation.get('user_cost'),
      'duration': simulation.get('duration'),
    }
    with (directory / 'rewards.jsonl').open('a', encoding='utf-8') as file:
      file.write(json.dumps(record) + '\n')


async def _stop(process: asyncio.subprocess.Process, grace_seconds: float = 30.0) -> None:
  """Terminate the simulation, which then exports the spans it has; kill it if it does not stop in time."""
  if process.returncode is not None:
    return
  process.terminate()
  try:
    await asyncio.wait_for(process.wait(), grace_seconds)
  except TimeoutError:
    process.kill()
    await process.wait()


def build_caller(
  configuration: Configuration,
  config: ExperimentConfig,
  *,
  data_dir: Path,
  phoenix: PhoenixSettings,
  outputs: RunOutputs,
  settings: Tau2Settings | None = None,
  foundry: AzureFoundrySettings | None = None,
) -> Tau2Caller:
  """The caller of a tau2 configuration, on the verified pinned files; spans go to ``phoenix``'s project."""
  if configuration.trial is None or phoenix.project_id is None:
    raise ValueError(f'{configuration.id}: a tau2 run needs a trial and a Phoenix project')
  settings = settings or Tau2Settings()
  if not settings.python.exists():
    raise FileNotFoundError(f'No tau2 runtime at {settings.python}: run scripts/setup-tau2.sh')
  raw = data_dir / 'tau2' / 'raw'
  tasks = json.loads(verify_pinned_files(config.benchmarks['tau2'], raw)['tasks'].read_text(encoding='utf-8'))
  agent = {
    'model': _route(config.models.agents[configuration.model]),
    'llm_args': config.tau2.agent_llm_args[configuration.model],
  }
  spec = {
    'seed': config.tau2.trial_seed(configuration.trial),
    'agent_name': configuration.agent,
    'agent': agent,
    'customer': {'model': _route(config.models.customer_simulator), 'llm_args': config.tau2.customer_llm_args},
    # tau2 grades its assertions with the agent model.
    'grader': agent,
    'max_steps': config.tau2.max_steps,
    'max_errors': config.tau2.max_errors,
    'otlp_endpoint': f'{phoenix.base_url.rstrip("/")}/v1/traces',
    'project': phoenix.project_id,
    'cost_map_url': config.tau2.cost_map_url,
    'metadata': {'configuration': configuration.id, 'trial': configuration.trial},
  }
  # The pinned files keep their repository paths, so tau2's data folder is raw/data.
  environment = {**os.environ, 'TAU2_DATA_DIR': str((raw / 'data').resolve()), 'PYTHONUNBUFFERED': '1'}
  foundry = foundry or AzureFoundrySettings()
  if foundry.base_url and foundry.api_key:
    # litellm's azure/ models are called at the resource, its azure_ai/ models at its model inference API.
    environment.update(
      AZURE_API_BASE=foundry.base_url,
      AZURE_API_KEY=foundry.api_key,
      AZURE_API_VERSION=settings.azure_api_version,
      AZURE_AI_API_BASE=f'{foundry.base_url.rstrip("/")}/models',
      AZURE_AI_API_KEY=foundry.api_key,
    )
  return Tau2Caller(
    python=settings.python,
    spec=spec,
    task_ids={render_user_scenario(task['user_scenario']): task['id'] for task in tasks},
    outputs=outputs,
    environment=environment,
    max_attempts=settings.max_attempts,
    retry_delay_seconds=settings.retry_delay_seconds,
  )


def _route(model: ModelRef) -> str:
  """The litellm route of a model, such as anthropic/claude-sonnet-5-5."""
  return f'{model.provider}/{model.model}'


class Tau2TraceAdapter:
  """The canonical trace of one simulation, whose agent root is tau2's agent span.

  The simulation's root span also holds the customer simulator and the reward. Their LLM calls get span types of their
  own and no usage, so that a run's report counts the agent's calls only. So does an LLM span nested in another, such
  as litellm's retry, which repeats its parent's usage.
  """

  def normalize(self, trace_id: str, records: list[dict[str, Any]]) -> TraceProcessingResult:
    base = PhoenixTraceAdapter().normalize(trace_id, records)
    records_by_id = {str(record['context.span_id']): record for record in records}
    spans = {span.external_id: span for span in base.spans}
    roots = [span for span in base.spans if span.parent_span_id is None]
    if len(roots) != 1 or roots[0].name != ROOT_SPAN_NAME:
      raise ValueError(f'Trace {trace_id} must have one root span, named {ROOT_SPAN_NAME}')
    root = roots[0]
    agents = [span for span in base.spans if span.span_type == 'agent' and span.parent_span_id == root.external_id]
    if len(agents) != 1:
      raise ValueError(f'Trace {trace_id} has {len(agents)} agent spans under its root; expected one')
    agent = agents[0]
    if not isinstance(root.input_data, str) or not root.input_data.strip():
      raise ValueError(f'Trace {trace_id} has no user scenario on its root span')

    def top(span: Span) -> Span:
      """The child of the root that holds ``span``."""
      while span.parent_span_id != root.external_id:
        span = spans[str(span.parent_span_id)]
      return span

    steps = [
      _executed_step(span, records_by_id[span.external_id])
      for span in sorted(base.spans, key=lambda span: (span.start_time, span.external_id))
      if span.span_type == 'tool' and span is not root and top(span) is agent
    ]
    answer = agent.output_data if isinstance(agent.output_data, str) and agent.output_data.strip() else None
    normalized = []
    for span in base.spans:
      if span is root:
        span = span.model_copy(update={'span_type': 'simulation'})
      elif span is agent:
        semantics = SpanSemantics(
          request=root.input_data,
          answer=Answer(text=answer) if answer else None,
          planning=PlanningData(executed_steps=steps),
        )
        span = span.model_copy(update={'span_type': 'agent_root', 'semantics': semantics})
      elif span.span_type == 'llm':
        span = _llm_span(span, spans, outside=_outside_agent(top(span), agent))
      normalized.append(span)
    return TraceProcessingResult(trace=base.trace.model_copy(update={'adapter': 'tau2:1'}), spans=normalized)


def _outside_agent(top: Span, agent: Span) -> str | None:
  """The span type of an LLM call outside the agent, or None for the agent's own."""
  if top is agent:
    return None
  return {'user_turn': 'customer_llm', 'evaluate_simulation': 'grader_llm'}.get(top.name, 'external_llm')


def _llm_span(span: Span, spans: Mapping[str, Span], *, outside: str | None) -> Span:
  parent = spans.get(str(span.parent_span_id))
  span_type: str | None = 'llm_internal' if parent is not None and parent.span_type == 'llm' else outside
  if span_type is not None:
    return span.model_copy(update={'span_type': span_type, 'semantics': SpanSemantics()})
  usage = span.semantics.usage
  # litellm reports 0 for a model it cannot price.
  if usage is not None and usage.cost == 0:
    semantics = span.semantics.model_copy(update={'usage': usage.model_copy(update={'cost': None, 'currency': None})})
    span = span.model_copy(update={'semantics': semantics})
  return span


def _executed_step(span: Span, record: Mapping[str, Any]) -> ExecutionStep:
  arguments = span.input_data
  status: Literal['completed', 'error'] = 'error' if span.status == 'error' else 'completed'
  return ExecutionStep(
    id=span.external_id,
    operation=str(record.get('attributes.tool.name') or span.name),
    instruction=render_arguments(arguments) if isinstance(arguments, dict) else str(arguments),
    status=status,
    input=arguments,
    # The tool's result as the agent read it.
    output=record.get('attributes.output.value'),
    start_time=span.start_time,
    end_time=span.end_time,
    span_ids=[span.external_id],
    attributes={'tool_call_id': record.get('attributes.tool.id')},
  )
