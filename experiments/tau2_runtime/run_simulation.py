"""Run one traced tau2 retail simulation; run with the tau2 environment's interpreter (``scripts/setup-tau2.sh``).

  .venv-tau2/bin/python tau2_runtime/run_simulation.py --spec <spec.json> --result <simulation.json>

The spec names the task, the seed, the models and where the spans go. On success the simulation, with its reward, is
written to ``--result``. Exit codes: 0 done; 75 a transient failure (a provider's rate limit or outage, an unreachable
cost map, spans that could not be exported), worth retrying with a new request id; 1 any other failure. tau2 reads its
data from ``TAU2_DATA_DIR``, and the models' credentials from the environment.
"""

import argparse
import os
import signal
import sys
import traceback
from pathlib import Path
from types import FrameType
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

DOMAIN = 'retail'
EXIT_FAILED = 1
EXIT_TRANSIENT = 75
_FLUSH_TIMEOUT_MILLIS = 120_000


class ModelSpec(BaseModel):
  model_config = ConfigDict(extra='forbid')

  # A litellm model route, such as anthropic/claude-sonnet-5-5.
  model: str = Field(min_length=1)
  llm_args: dict[str, Any] = Field(default_factory=dict)


class SimulationSpec(BaseModel):
  model_config = ConfigDict(extra='forbid')

  request_id: str = Field(min_length=1)
  task_id: str = Field(min_length=1)
  seed: int
  agent_name: str = Field(min_length=1)
  agent: ModelSpec
  customer: ModelSpec
  # Grades the task's natural-language assertions.
  grader: ModelSpec
  max_steps: int = Field(gt=0)
  max_errors: int = Field(gt=0)
  otlp_endpoint: str = Field(min_length=1)
  project: str = Field(min_length=1)
  # litellm's model cost map at a fixed commit: it also decides which parameters litellm passes to each model.
  cost_map_url: str = Field(pattern=r'^https://\S+/[0-9a-f]{40}/\S+\.json$')
  # Added to the root span's metadata.
  metadata: dict[str, Any] = Field(default_factory=dict)


class TransientError(Exception):
  """A failure that a new attempt can avoid."""


def main(argv: list[str] | None = None) -> int:
  parser = argparse.ArgumentParser(description='Run one traced tau2 retail simulation.')
  parser.add_argument('--spec', type=Path, required=True, help='JSON simulation spec.')
  parser.add_argument('--result', type=Path, required=True, help='Where to write the simulation JSON.')
  args = parser.parse_args(argv)
  spec = SimulationSpec.model_validate_json(args.spec.read_text(encoding='utf-8'))

  # litellm loads its cost map when imported, and the instrumentor imports it.
  os.environ['LITELLM_MODEL_COST_MAP_URL'] = spec.cost_map_url
  os.environ.pop('LITELLM_LOCAL_MODEL_COST_MAP', None)
  from telemetry import configure_tracing, otlp_exporter

  headers = {'authorization': f'Bearer {key}'} if (key := os.environ.get('PHOENIX_API_KEY')) else {}
  exporter = otlp_exporter(spec.otlp_endpoint, headers)
  provider = configure_tracing(spec.project, exporter)
  # A terminated run still exports the spans it has.
  signal.signal(signal.SIGTERM, _exit_on_sigterm)
  code = EXIT_FAILED
  try:
    _check_cost_map(spec.cost_map_url)
    simulation = simulate(spec, provider.get_tracer('syllo-exp.tau2'))
    partial = args.result.with_name(args.result.name + '.part')
    partial.write_text(simulation.model_dump_json(), encoding='utf-8')
    partial.replace(args.result)
    code = 0
  except Exception as error:
    traceback.print_exc()
    code = EXIT_TRANSIENT if _is_transient(error) else EXIT_FAILED
  finally:
    flushed = provider.force_flush(timeout_millis=_FLUSH_TIMEOUT_MILLIS)
    provider.shutdown()
  if not flushed or exporter.failed:
    # An incomplete trace would be scored as if the agent had not made the missing calls.
    print('The spans could not all be exported', file=sys.stderr)
    return EXIT_TRANSIENT
  return code


def simulate(spec: SimulationSpec, tracer: Any) -> Any:
  """tau2's run of one task, as its CLI runs it, with the spec's models; tau2 is imported only now."""
  from tau2.data_model.simulation import TextRunConfig
  from tau2.runner.build import build_text_orchestrator
  from tau2.runner.helpers import get_tasks

  from traced_simulation import TracedOrchestrator, configure_grader, run_traced_simulation

  configure_grader(spec.grader.model, spec.grader.llm_args)
  (task,) = get_tasks(DOMAIN, task_split_name='base', task_ids=[spec.task_id])
  config = TextRunConfig(
    domain=DOMAIN,
    agent='llm_agent',
    llm_agent=spec.agent.model,
    llm_args_agent=dict(spec.agent.llm_args),
    user='user_simulator',
    llm_user=spec.customer.model,
    llm_args_user=dict(spec.customer.llm_args),
    max_steps=spec.max_steps,
    max_errors=spec.max_errors,
    seed=spec.seed,
  )
  built = build_text_orchestrator(config, task, seed=spec.seed, simulation_id=spec.request_id)
  orchestrator = TracedOrchestrator.wrap(built, tracer=tracer, agent_name=spec.agent_name)
  return run_traced_simulation(orchestrator, tracer=tracer, request_id=spec.request_id, metadata=spec.metadata)


def _check_cost_map(url: str) -> None:
  """litellm falls back to its bundled cost map when the pinned one cannot be fetched; never run on that one."""
  from litellm.litellm_core_utils.get_model_cost_map import get_model_cost_map_source_info

  info = get_model_cost_map_source_info()
  if info.get('source') != 'remote' or info.get('url') != url:
    raise TransientError(f'litellm did not load the pinned cost map {url}: {info}')


def _is_transient(error: BaseException) -> bool:
  if isinstance(error, TransientError):
    return True
  # litellm raises subclasses of the OpenAI SDK's errors, for every provider.
  import openai

  if isinstance(error, openai.APIConnectionError):  # timeouts too
    return True
  status = getattr(error, 'status_code', None)
  return isinstance(error, openai.APIError) and isinstance(status, int) and (status in (408, 429) or status >= 500)


def _exit_on_sigterm(signum: int, frame: FrameType | None) -> None:
  raise SystemExit(128 + signum)


if __name__ == '__main__':
  sys.exit(main())
