"""Collection runs: a fresh evaluation that calls one agent configuration on every sample of its benchmark.

It stores each sample's trace, keeps a raw copy of it in ``<outputs>/traces/<run id>/`` in the format that
``syllo-eval`` imports, and computes the benchmark's deterministic metrics only. Judge metrics come later, as repeats
of the run on its stored traces.

Each benchmark has one Phoenix project, named after its dataset, to which every agent of the benchmark sends its spans.
The collection of an agent that searches starts only once the search server that the agent calls serves the
configuration's collection and swap fraction.
"""

import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, assert_never
from urllib.parse import urlsplit

from syllo_eval.evaluation.metrics.contracts import EvaluationMetric
from syllo_eval.evaluation.metrics.implementations.plan.efficiency import PlanEfficiencyMetric
from syllo_eval.evaluation.metrics.implementations.rag.ndcg_at_10 import NdcgAt10DocumentMetric
from syllo_eval.evaluation.metrics.implementations.rag.set_precision import SetPrecisionDocumentMetric
from syllo_eval.evaluation.metrics.implementations.rag.set_recall import SetRecallDocumentMetric
from syllo_eval.evaluation.trace_adapter import TraceAdapter
from syllo_eval.evaluation.trace_import import ImportedTrace
from syllo_eval.evaluation.trace_processor import TraceSourceClient
from syllo_eval.execution.agent_caller import AgentCaller
from syllo_eval.settings import PhoenixSettings

from agents import dify, tau2
from ablation_metrics import SEARCH_SPAN_TYPE
from config import Benchmark, Configuration, ExperimentConfig
from run_outputs import RunOutputs

# A trace is looked up among the root spans that started this long before: a single agent call, such as a tau2
# simulation, can outlast the library's default of ten minutes.
REQUEST_ID_LOOKUP_WINDOW_SECONDS = 24 * 3600.0
# Set per stack, not in the environment that every stack's collection shares.
_REQUEST_ID_ATTRIBUTES = {'react': dify.REQUEST_ID_ATTRIBUTE}


class SearchServerError(ValueError):
  """A search server that did not report its settings, or that serves another collection or swap fraction."""


@dataclass(frozen=True, slots=True)
class AgentIntegration:
  caller: AgentCaller
  # None for the library's Phoenix adapter.
  adapter: TraceAdapter | None
  # The MCP endpoint of the search server that the agent calls; None for an agent that does not search.
  search_url: str | None


def phoenix_settings(base: PhoenixSettings, project: str, agent: str) -> PhoenixSettings:
  """``base`` for the Phoenix project of one benchmark, looking traces up by the request-ID attribute of ``agent``."""
  window = max(base.request_id_lookup_time_window_seconds, REQUEST_ID_LOOKUP_WINDOW_SECONDS)
  return base.model_copy(
    update={
      'project_id': project,
      'request_id_lookup_time_window_seconds': window,
      'request_id_attribute': _REQUEST_ID_ATTRIBUTES.get(agent, 'request_id'),
    }
  )


def build_integration(
  configuration: Configuration,
  config: ExperimentConfig,
  *,
  data_dir: Path,
  phoenix: PhoenixSettings,
  outputs: RunOutputs,
) -> AgentIntegration:
  agent = configuration.agent
  if agent == 'tau2-llm-agent':
    caller = tau2.build_caller(configuration, config, data_dir=data_dir, phoenix=phoenix, outputs=outputs)
    return AgentIntegration(caller=caller, adapter=tau2.Tau2TraceAdapter(), search_url=None)
  if agent == 'odr':
    # ODR's graph and MCP client are slow to import, and only its collections need them.
    from agents import odr

    odr_settings = odr.OdrSettings()
    return AgentIntegration(
      caller=odr.build_caller(configuration, config, phoenix=phoenix, settings=odr_settings),
      adapter=odr.OdrTraceAdapter(config.models.agents[configuration.model].model),
      search_url=odr_settings.mcp_url,
    )
  if agent == 'react':
    dify_settings = dify.DifySettings()
    # No Dify adapter yet: its searches are not retrieval spans.
    return AgentIntegration(caller=dify.DifyCaller(dify_settings), adapter=None, search_url=dify_settings.search_url)
  if agent == 'smolagents':
    # smolagents and the MCP framework are slow to import, and only the CodeAgent's collections need them.
    from agents import code_agent

    return AgentIntegration(
      caller=code_agent.build_caller(configuration, config, phoenix=phoenix),
      adapter=code_agent.CodeAgentTraceAdapter(),
      search_url=code_agent.CodeAgentSettings().search_url,
    )
  # Type checking fails here for a stack without a caller.
  assert_never(agent)


async def check_search_server(configuration: Configuration, config: ExperimentConfig, url: str) -> None:
  """Refuse to collect unless the search server at ``url`` serves the configuration's collection and swap fraction.

  The agents keep one URL, and the server is relaunched for each configuration: a server left on another one would
  answer every search from the wrong knowledge base, or with the wrong share of random documents.
  """
  collection = config.benchmarks[configuration.benchmark].dataset_name
  port = urlsplit(url).port
  command = f'syllo-exp search-server --collection {collection}' + (f' --port {port}' if port else '')
  if configuration.degradation:
    command += f' --swap-fraction {configuration.degradation:g}'
  try:
    served = await _search_server_settings(url)
  except Exception as error:
    raise SearchServerError(
      f'{configuration.id}: the search server at {url} did not report its settings ({error}); start it with `{command}`'
    ) from error
  if (served.get('collection'), served.get('swap_fraction')) != (collection, configuration.degradation):
    raise SearchServerError(
      f'{configuration.id}: the search server at {url} serves {served.get("collection")} with swap fraction '
      f'{served.get("swap_fraction")}, but the configuration needs {collection} with swap fraction '
      f'{configuration.degradation}; relaunch it with `{command}`'
    )


async def _search_server_settings(url: str) -> dict[str, Any]:
  # Imported here: the MCP framework is slow to import, and only the collections of agents that search need it.
  from fastmcp import Client

  from search_tool.server import SETTINGS_URI

  async with Client(url, timeout=30) as client:
    [content] = await client.read_resource(SETTINGS_URI)
  return json.loads(getattr(content, 'text'))


def deterministic_metrics(benchmark: Benchmark) -> Sequence[EvaluationMetric]:
  """The metrics a collection run computes: those that need no judge."""
  if benchmark == 'tau2':
    return [PlanEfficiencyMetric()]
  targets = (SEARCH_SPAN_TYPE,)
  return [
    SetPrecisionDocumentMetric(target_span_types=targets),
    SetRecallDocumentMetric(target_span_types=targets),
    NdcgAt10DocumentMetric(target_span_types=targets),
  ]


class TraceArchive:
  """A trace client that writes each trace it fetches to ``<outputs>/traces/<run id>/<trace id>.json``."""

  def __init__(self, client: TraceSourceClient, outputs: RunOutputs):
    self._client = client
    self._outputs = outputs

  async def get_trace_id_by_request_id(self, request_id: str) -> str | None:
    return await self._client.get_trace_id_by_request_id(request_id)

  async def get_trace_json(self, trace_id: str) -> list[dict[str, Any]]:
    records = await self._client.get_trace_json(trace_id)
    if records:
      path = self._outputs.directory('traces') / f'{trace_id}.json'
      path.write_text(ImportedTrace(trace_id=trace_id, spans=records).model_dump_json(), encoding='utf-8')
    return records
