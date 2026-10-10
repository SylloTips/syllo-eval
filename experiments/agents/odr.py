"""Open Deep Research (ODR), run in this process, and the adapter of its traces.

ODR's LangGraph graph runs unmodified, at the commit pyproject.toml pins, with the settings of ``configs/odr.yaml``: no
web search, the search tool over MCP as its only source, and the configuration's model in every phase. Each call runs
the graph under a root span with the request id, the question and the answer, and OpenInference's LangChain
instrumentation traces the graph to Phoenix. Spans are flushed before the call returns.

ODR turns many failures into text or silence, so a run that failed can look complete. A question runs again, with a new
request id, when the search tool was unreachable at any point, when a provider's rate limit, outage or connection error
outlasted ODR's retries, a researcher's included, when ODR wrote an error in place of its report, when spans did not
reach Phoenix, or when the run took too long.
"""

import asyncio
import json
import logging
import os
import threading
from collections.abc import Callable, Mapping, Sequence
from typing import Any
from uuid import UUID, uuid4

import anthropic
import open_deep_research.deep_researcher as deep_research
import openai
from langchain.chat_models import init_chat_model
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.messages import BaseMessage, HumanMessage
from langchain_core.tools import ToolException
from open_deep_research.configuration import Configuration as OdrSettingNames
from open_deep_research.utils import get_all_tools
from openinference.instrumentation.langchain import LangChainInstrumentor
from openinference.semconv.resource import ResourceAttributes
from opentelemetry import trace
from opentelemetry.context import Context
from opentelemetry.exporter.otlp.proto.http import Compression
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import ReadableSpan, SpanLimits, TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor, SpanExporter, SpanExportResult
from pydantic import Field, ValidationError, field_validator
from pydantic_settings import SettingsConfigDict

from syllo_eval.evaluation.trace_adapter import PhoenixTraceAdapter, TraceProcessingResult
from syllo_eval.model import Sample
from syllo_eval.settings import EnvSettings, PhoenixSettings
from syllo_eval.trace_semantics import Answer, SpanSemantics

from ablation_metrics import SEARCH_SPAN_TYPE
from config import Configuration, ExperimentConfig, ModelRef
from indexing.embedding import AzureFoundrySettings
from search_tool.search import SearchResults
from search_tool.server import PATH, TOOL_NAME

ROOT_SPAN_NAME = 'odr.request'
# What ODR writes in place of its report, and in place of a researcher's findings, when it could not produce them.
REPORT_ERROR = 'Error generating final report'
FINDINGS_ERROR = 'Error synthesizing research report'
# ODR reads these before its run config: each of its settings, and the switch to take API keys from the config.
_OVERRIDING_VARIABLES = tuple(name.upper() for name in OdrSettingNames.model_fields) + ('GET_API_KEYS_FROM_CONFIG',)
# The settings that name a model: ODR picks each phase's model from one of them.
_MODEL_SETTINGS = ('research_model', 'compression_model', 'final_report_model', 'summarization_model')
_FLUSH_TIMEOUT_MILLIS = 120_000
# Spans wait in this queue to be exported, and past its size the SDK drops them with only a log line.
_SPAN_QUEUE_SIZE = 1_000_000

logger = logging.getLogger(__name__)


class OdrSettings(EnvSettings):
  """How ODR runs here; what it runs is in ``configs/odr.yaml``."""

  model_config = SettingsConfigDict(env_prefix='ODR_')

  # The search server's base URL: ODR appends /mcp, and drops a server whose URL already ends with it.
  search_url: str = Field(default='http://127.0.0.1:8101', min_length=1)
  # Attempts per sample at a transient failure, each a new run with a new request id.
  max_attempts: int = Field(default=3, ge=1)
  retry_delay_seconds: float = Field(default=60.0, ge=0)
  # A provider can hold a request open for good, and neither ODR nor Anthropic's client sets a timeout.
  timeout_seconds: float = Field(default=3600.0, gt=0)
  # DeepSeek is a deployment of the Foundry resource at AZURE_FOUNDRY_BASE_URL, called through its Azure OpenAI API.
  azure_api_version: str = Field(default='2025-04-01-preview', min_length=1)

  @field_validator('search_url')
  @classmethod
  def _check_base_url(cls, url: str) -> str:
    if url.rstrip('/').endswith('/mcp'):
      raise ValueError('give the search server without /mcp, which ODR appends')
    return url

  @property
  def mcp_url(self) -> str:
    """The search server's MCP endpoint, as ODR builds it."""
    return self.search_url.rstrip('/') + PATH


class OdrRunError(RuntimeError):
  """A question ODR could not answer in any attempt: the sample fails."""


class _TransientError(Exception):
  """A failure that a new run can avoid."""


class RunWatch(BaseCallbackHandler):
  """Watches one ODR run, through the LangChain callbacks of its graph, for the failures that ODR hides.

  - The search tool was unreachable when a researcher's LLM call lacked it, since ODR drops a server it cannot reach
    without an error; when a researcher's tool step could not find it, since it vanished after the call; or when a
    search got no answer from the server. A search the server rejected, such as an empty query, is the agent's failed
    call.
  - A researcher failed for a provider's rate limit, outage or connection error: ODR then ends its research and writes
    the report from what the earlier rounds found. A researcher's other failures, such as a tool the model made up, are
    the agent's.
  """

  # Quick bookkeeping, so the callbacks run in the event loop, in order.
  run_inline = True

  def __init__(self) -> None:
    self.problems: list[str] = []
    self._nodes: dict[UUID, str] = {}
    self._searches: set[UUID] = set()

  def on_chat_model_start(
    self,
    serialized: dict[str, Any],
    messages: list[list[BaseMessage]],
    *,
    run_id: UUID,
    metadata: dict[str, Any] | None = None,
    **kwargs: Any,
  ) -> None:
    tools = (kwargs.get('invocation_params') or {}).get('tools') or []
    if (metadata or {}).get('langgraph_node') == 'researcher' and TOOL_NAME not in _tool_names(tools):
      self.problems.append('a researcher ran without the search tool')

  def on_chain_start(
    self,
    serialized: dict[str, Any],
    inputs: dict[str, Any],
    *,
    run_id: UUID,
    metadata: dict[str, Any] | None = None,
    **kwargs: Any,
  ) -> None:
    node = (metadata or {}).get('langgraph_node')
    # The node's own run, which is named after it: the runs inside it carry its name in their metadata too.
    if node in ('researcher', 'researcher_tools') and kwargs.get('name') == node:
      self._nodes[run_id] = node

  def on_chain_end(self, outputs: dict[str, Any], *, run_id: UUID, **kwargs: Any) -> None:
    self._nodes.pop(run_id, None)

  def on_chain_error(self, error: BaseException, *, run_id: UUID, **kwargs: Any) -> None:
    node = self._nodes.pop(run_id, None)
    if node == 'researcher' and isinstance(error, Exception) and _is_transient(error):
      self.problems.append(f'a researcher failed with {type(error).__name__}')
    elif node == 'researcher_tools' and isinstance(error, KeyError) and error.args == (TOOL_NAME,):
      self.problems.append("the search tool vanished before a researcher's searches")

  def on_tool_start(self, serialized: dict[str, Any], input_str: str, *, run_id: UUID, **kwargs: Any) -> None:
    if serialized.get('name') == TOOL_NAME:
      self._searches.add(run_id)

  def on_tool_end(self, output: Any, *, run_id: UUID, **kwargs: Any) -> None:
    self._searches.discard(run_id)

  def on_tool_error(self, error: BaseException, *, run_id: UUID, **kwargs: Any) -> None:
    # The MCP client raises a ToolException for the server's own answer, and other errors when it got none.
    if run_id in self._searches and not isinstance(error, ToolException):
      self.problems.append('a search got no answer from the search tool')
    self._searches.discard(run_id)


def _tool_names(tools: Sequence[Any]) -> set[str]:
  """The names of the tools an LLM call is given, in Anthropic's or OpenAI's schema."""
  return {tool.get('name') or tool.get('function', {}).get('name') for tool in tools if isinstance(tool, dict)}


class _CheckedExporter(SpanExporter):
  """Remembers the traces whose spans failed to export: a flush only says whether the export finished in time.

  One exporter serves every sample of the process, so the failures are kept per trace.
  """

  def __init__(self, exporter: SpanExporter):
    self._exporter = exporter
    self._lock = threading.Lock()
    self._failed: set[int] = set()

  def export(self, spans: Sequence[ReadableSpan]) -> SpanExportResult:
    result = self._exporter.export(spans)
    if result is not SpanExportResult.SUCCESS:
      with self._lock:
        self._failed.update(span.context.trace_id for span in spans if span.context is not None)
    return result

  def lost(self, trace_id: int) -> bool:
    with self._lock:
      return trace_id in self._failed

  def shutdown(self) -> None:
    self._exporter.shutdown()

  def force_flush(self, timeout_millis: int = 30_000) -> bool:
    return self._exporter.force_flush(timeout_millis)


class OdrCaller:
  """Runs ODR on the question of a sample and returns the request id of the run that answered it."""

  def __init__(
    self,
    *,
    configurable: Mapping[str, Any],
    tracer: trace.Tracer,
    flush: Callable[[int], bool],
    metadata: Mapping[str, Any],
    max_attempts: int = 3,
    retry_delay_seconds: float = 60.0,
    timeout_seconds: float = 3600.0,
    graph: Any = deep_research.deep_researcher,
  ):
    self._config = {'configurable': dict(configurable)}
    self._tracer = tracer
    # Whether the spans of a trace reached Phoenix.
    self._flush = flush
    # Added to the root span.
    self._metadata = json.dumps(dict(metadata))
    self._max_attempts = max_attempts
    self._retry_delay_seconds = retry_delay_seconds
    self._timeout_seconds = timeout_seconds
    self._graph = graph

  async def call(self, sample: Sample) -> str:
    attempt = 1
    while True:
      request_id = uuid4().hex
      try:
        await self._run(sample.input_prompt, request_id)
        return request_id
      except _TransientError as error:
        if attempt == self._max_attempts:
          raise OdrRunError(f'ODR failed on sample {sample.id} in all {attempt} attempts; the last: {error}') from error
        logger.warning(
          'ODR on sample %s: attempt %d (request %s) failed: %s; retrying in %gs',
          sample.id,
          attempt,
          request_id,
          error,
          self._retry_delay_seconds,
        )
        await asyncio.sleep(self._retry_delay_seconds)
        attempt += 1

  async def _run(self, question: str, request_id: str) -> None:
    # ODR loads the tools at every researcher step and drops an unreachable server without an error.
    if TOOL_NAME not in {tool.name for tool in await get_all_tools(self._config)}:
      raise _TransientError(f'the search tool is unreachable at {self._config["configurable"]["mcp_config"]["url"]}')
    watch = RunWatch()
    attributes = {
      'request_id': request_id,
      'openinference.span.kind': 'AGENT',
      'input.value': question,
      'metadata': self._metadata,
    }
    # An empty parent context makes this a root span, which the request_id lookup requires.
    with self._tracer.start_as_current_span(ROOT_SPAN_NAME, context=Context(), attributes=attributes) as span:
      trace_id = span.get_span_context().trace_id
      failure: Exception | None = None
      try:
        state = await self._invoke(question, watch)
      except Exception as error:
        failure = error
      finally:
        problems = sorted(set(watch.problems))
        await _cancel_leftovers(trace_id)
      # Whatever the run did after losing its search tool or a researcher's provider, it did without them.
      if problems:
        raise _TransientError('; '.join(problems)) from failure
      if failure is not None:
        raise failure
      report = state['final_report']
      if isinstance(report, str) and report.startswith(REPORT_ERROR):
        raise _TransientError(report[:500])
      lost = [
        message.content
        for message in state.get('supervisor_messages', [])
        if isinstance(message.content, str) and message.content.startswith(FINDINGS_ERROR)
      ]
      if lost:
        # ODR reports what the other researchers found.
        logger.warning('ODR request %s: %d researchers lost their findings: %s', request_id, len(lost), lost[0][:300])
      # The report as text: Sonnet's reply also holds its thinking.
      span.set_attribute('output.value', state['messages'][-1].text)
    if not await asyncio.to_thread(self._flush, trace_id):
      raise _TransientError('spans of the run did not reach Phoenix')

  async def _invoke(self, question: str, watch: RunWatch) -> dict[str, Any]:
    config = {**self._config, 'callbacks': [watch]}
    try:
      async with asyncio.timeout(self._timeout_seconds):
        return await self._graph.ainvoke({'messages': [HumanMessage(content=question)]}, config)
    except TimeoutError as error:
      raise _TransientError(f'the run took longer than {self._timeout_seconds:g}s') from error
    except Exception as error:
      if _is_transient(error):
        raise _TransientError(f'{type(error).__name__}: {error}') from error
      raise


async def _cancel_leftovers(trace_id: int) -> None:
  """Cancel the tasks that a run left running, and wait for them to stop.

  When a researcher fails, ODR's supervisor neither waits for nor cancels the others, which would go on adding spans to
  the trace after the call returns. A run's tasks are those whose current span is in its trace.
  """
  current = asyncio.current_task()
  leftovers = [
    task
    for task in asyncio.all_tasks()
    if task is not current and task.get_context().run(trace.get_current_span).get_span_context().trace_id == trace_id
  ]
  for task in leftovers:
    task.cancel()
  await asyncio.gather(*leftovers, return_exceptions=True)


def _is_transient(error: Exception) -> bool:
  """A provider's rate limit, outage or connection error."""
  status = getattr(error, 'status_code', None)
  return (
    isinstance(error, (anthropic.APIConnectionError, openai.APIConnectionError))
    or status == 429
    or (isinstance(status, int) and status >= 500)
  )


def build_caller(
  configuration: Configuration,
  config: ExperimentConfig,
  *,
  phoenix: PhoenixSettings,
  settings: OdrSettings | None = None,
  foundry: AzureFoundrySettings | None = None,
) -> OdrCaller:
  """The caller of an ODR configuration, whose spans go to ``phoenix``'s project.

  ODR's model and the instrumentation are process-wide: one collection per process, as `syllo-exp collect` runs.
  """
  if phoenix.project_id is None:
    raise ValueError(f'{configuration.id}: an ODR run needs a Phoenix project')
  # Even empty: ODR would read an empty MCP_PROMPT as the prompt.
  overriding = [name for name in _OVERRIDING_VARIABLES if name in os.environ]
  if overriding:
    raise ValueError(f'{configuration.id}: unset {", ".join(overriding)}: ODR would read them instead of its settings')
  settings = settings or OdrSettings()
  model, connection = _model(configuration, config.models.agents[configuration.model], settings, foundry)
  # Every phase names the model through these settings; the module-level model ODR configures per phase also
  # defaults to it and carries its connection.
  deep_research.configurable_model = init_chat_model(
    model, configurable_fields=('model', 'max_tokens', 'api_key'), **connection
  )
  configurable = {
    **config.odr.model_dump(),
    **dict.fromkeys(_MODEL_SETTINGS, model),
    'mcp_config': {'url': settings.search_url, 'tools': [TOOL_NAME], 'auth_required': False},
  }
  provider = TracerProvider(
    resource=Resource.create({ResourceAttributes.PROJECT_NAME: phoenix.project_id}),
    # The SDK keeps 128 attributes per span by default and drops the oldest: a researcher's LLM span would lose its
    # input, its output and its first messages.
    span_limits=SpanLimits(max_span_attributes=SpanLimits.UNSET),
  )
  # OTLP over HTTP with gzip: Phoenix's gRPC receiver keeps gRPC's 4 MiB message limit, which long traces exceed.
  exporter = _CheckedExporter(
    OTLPSpanExporter(
      endpoint=f'{phoenix.base_url.rstrip("/")}/v1/traces',
      headers={'authorization': f'Bearer {phoenix.api_key}'} if phoenix.api_key else None,
      compression=Compression.Gzip,
      timeout=30,
    )
  )
  provider.add_span_processor(BatchSpanProcessor(exporter, max_queue_size=_SPAN_QUEUE_SIZE))
  LangChainInstrumentor().instrument(tracer_provider=provider)
  return OdrCaller(
    configurable=configurable,
    tracer=provider.get_tracer(__name__),
    flush=lambda trace_id: provider.force_flush(_FLUSH_TIMEOUT_MILLIS) and not exporter.lost(trace_id),
    metadata={'configuration': configuration.id, 'model': model},
    max_attempts=settings.max_attempts,
    retry_delay_seconds=settings.retry_delay_seconds,
    timeout_seconds=settings.timeout_seconds,
  )


def _model(
  configuration: Configuration, ref: ModelRef, settings: OdrSettings, foundry: AzureFoundrySettings | None
) -> tuple[str, dict[str, Any]]:
  """The model as ODR names it, and the connection its client needs beyond what ODR passes."""
  if ref.provider == 'anthropic':
    # ODR passes ANTHROPIC_API_KEY itself.
    return f'anthropic:{ref.model}', {}
  if ref.provider == 'azure_ai':
    foundry = foundry or AzureFoundrySettings()
    if not (foundry.base_url and foundry.api_key):
      raise ValueError(f'{configuration.id}: AZURE_FOUNDRY_BASE_URL and AZURE_FOUNDRY_API_KEY must be set')
    # ODR passes no key for azure_openai models, so the client takes this one, never OPENAI_API_KEY.
    connection = {
      'azure_endpoint': foundry.base_url,
      'api_key': foundry.api_key,
      'api_version': settings.azure_api_version,
    }
    return f'azure_openai:{ref.model}', connection
  raise ValueError(f'{configuration.id}: ODR has no route for {ref.provider} models')


class OdrTraceAdapter:
  """The canonical trace of one ODR run: its root answers the question, and each search is a retrieval for it.

  A search becomes a retrieval span when its output is the search tool's JSON, which LangChain's MCP client returns as
  text blocks. A failed search, whose output is an error message, stays a tool span: it is not a search unit. A trace
  with an LLM call to any model but ``model`` is refused, so that every phase is known to use it.
  """

  def __init__(self, model: str):
    self._model = model

  def normalize(self, trace_id: str, records: list[dict[str, Any]]) -> TraceProcessingResult:
    base = PhoenixTraceAdapter().normalize(trace_id, records)
    roots = [record for record in records if record.get('parent_id') is None]
    if len(roots) != 1 or roots[0].get('name') != ROOT_SPAN_NAME:
      raise ValueError(f'Trace {trace_id} must have one root span, named {ROOT_SPAN_NAME}')
    # Read raw: the decoded values would turn an answer such as "42" into a number.
    question, answer = roots[0].get('attributes.input.value'), roots[0].get('attributes.output.value')
    if not isinstance(question, str) or not question.strip():
      raise ValueError(f'Trace {trace_id} has no question on its root span')
    spans = []
    for span in base.spans:
      usage = span.semantics.usage
      if usage is not None and usage.model != self._model:
        raise ValueError(f'Trace {trace_id} has an LLM call to {usage.model}, not {self._model}')
      if span.parent_span_id is None:
        semantics = SpanSemantics(request=question, answer=Answer(text=answer) if answer and answer.strip() else None)
        span = span.model_copy(update={'semantics': semantics})
      elif (span.metadata or {}).get('attributes.tool.name') == TOOL_NAME:
        found = _search_results(span.output_data)
        if found is not None:
          semantics = SpanSemantics(request=question, retrieval=[found.retrieval()])
          span = span.model_copy(update={'span_type': SEARCH_SPAN_TYPE, 'semantics': semantics})
      spans.append(span)
    return TraceProcessingResult(trace=base.trace.model_copy(update={'adapter': 'odr:1'}), spans=spans)


def _search_results(output: Any) -> SearchResults | None:
  """The search tool's results in a search span's output, or None for a failed search."""
  if not isinstance(output, list):
    return None
  text = ''.join(block['text'] for block in output if isinstance(block, dict) and block.get('type') == 'text')
  try:
    return SearchResults.model_validate_json(text)
  except ValidationError:
    return None
