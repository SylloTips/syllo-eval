"""What the agents that run a LangChain graph in this process share: Open Deep Research and the deep agent.

- Tracing: OpenInference's LangChain instrumentation sends their spans to Phoenix, and a flush says whether the spans
  of a trace got there.
- Retries: a question runs again, with a new request id, after a failure that a new run can avoid.
- Cleanup: the tasks that a run leaves running are cancelled before the call returns.
- Search: the search tool's MCP client fails fast when the server is down.
- Traces: the same adapter reads their searches and checks their model.
"""

import asyncio
import logging
import re
import threading
from collections.abc import Awaitable, Callable, Sequence
from typing import Any
from uuid import uuid4

import anthropic
import httpx
import openai
from langchain_core.tools import BaseTool
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_mcp_adapters.sessions import StreamableHttpConnection
from openinference.instrumentation.langchain import LangChainInstrumentor
from openinference.semconv.resource import ResourceAttributes
from opentelemetry import trace
from opentelemetry.exporter.otlp.proto.http import Compression
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import ReadableSpan, SpanLimits, TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor, SpanExporter, SpanExportResult

from syllo_eval.evaluation.trace_adapter import PhoenixTraceAdapter, TraceProcessingResult
from syllo_eval.model import Sample
from syllo_eval.settings import PhoenixSettings
from syllo_eval.trace_semantics import Answer, SpanSemantics

from ablation_metrics import SEARCH_SPAN_TYPE
from search_tool.search import search_span_retrieval, with_retrieved_evidence
from search_tool.server import TOOL_NAME

_FLUSH_TIMEOUT_MILLIS = 120_000
# Spans wait in this queue to be exported, and past its size the SDK drops them with only a log line.
_SPAN_QUEUE_SIZE = 1_000_000
_CONNECT_TIMEOUT_SECONDS = 2.0
_CONNECT_RETRIES = 2
# The coroutines of the tasks in which LangChain's callback managers call the handlers of a run's end or error.
_CALLBACK_TASK = re.compile(r'(?:AsyncCallbackManagerFor\w+Run\.on_\w+|_ahandle_event_for_handler)$')

logger = logging.getLogger(__name__)


class TransientError(Exception):
  """A failure that a new run can avoid."""


def is_transient(error: Exception) -> bool:
  """A provider's rate limit, outage or connection error."""
  status = getattr(error, 'status_code', None)
  return (
    isinstance(error, (anthropic.APIConnectionError, openai.APIConnectionError))
    or status == 429
    or (isinstance(status, int) and status >= 500)
  )


async def call_with_retries(
  sample: Sample,
  run: Callable[[str], Awaitable[None]],
  *,
  agent: str,
  failure: type[Exception],
  max_attempts: int,
  retry_delay_seconds: float,
) -> str:
  """Run ``sample``'s question under a new request id until a run ends without a transient failure.

  Returns the request id of that run. Raises ``failure`` when every attempt failed for a transient cause.
  """
  attempt = 1
  while True:
    request_id = uuid4().hex
    try:
      await run(request_id)
      return request_id
    except TransientError as error:
      if attempt == max_attempts:
        raise failure(f'{agent} failed on sample {sample.id} in all {attempt} attempts; the last: {error}') from error
      logger.warning(
        '%s on sample %s: attempt %d (request %s) failed: %s; retrying in %gs',
        agent,
        sample.id,
        attempt,
        request_id,
        error,
        retry_delay_seconds,
      )
      await asyncio.sleep(retry_delay_seconds)
      attempt += 1


class CheckedExporter(SpanExporter):
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


def trace_to_phoenix(phoenix: PhoenixSettings, name: str) -> tuple[trace.Tracer, Callable[[int], bool]]:
  """Instrument LangChain, process-wide, to send its spans to ``phoenix``'s project.

  Returns a tracer for the root spans, and the flush that says whether the spans of a trace reached Phoenix.
  """
  provider = TracerProvider(
    resource=Resource.create({ResourceAttributes.PROJECT_NAME: phoenix.project_id}),
    # The SDK keeps 128 attributes per span by default and drops the oldest: a long LLM call's span would lose its
    # input, its output and its first messages.
    span_limits=SpanLimits(max_span_attributes=SpanLimits.UNSET),
  )
  # OTLP over HTTP with gzip: Phoenix's gRPC receiver keeps gRPC's 4 MiB message limit, which long traces exceed.
  exporter = CheckedExporter(
    OTLPSpanExporter(
      endpoint=f'{phoenix.base_url.rstrip("/")}/v1/traces',
      headers={'authorization': f'Bearer {phoenix.api_key}'} if phoenix.api_key else None,
      compression=Compression.Gzip,
      timeout=30,
    )
  )
  provider.add_span_processor(BatchSpanProcessor(exporter, max_queue_size=_SPAN_QUEUE_SIZE))
  LangChainInstrumentor().instrument(tracer_provider=provider)

  def flush(trace_id: int) -> bool:
    return provider.force_flush(_FLUSH_TIMEOUT_MILLIS) and not exporter.lost(trace_id)

  return provider.get_tracer(name), flush


async def cancel_leftovers(trace_id: int) -> None:
  """Cancel the tasks that a run left running, and wait for them to stop.

  A graph that fails in one branch can leave the others running, which would go on adding spans to the trace after the
  call returns: ODR's supervisor neither waits for nor cancels the other researchers. A run's tasks are those whose
  current span is in its trace.

  The tasks in which LangChain ends or fails a run, which it shields from the cancellation of the run, finish: cancelled
  directly, they would leave the run's span open, and the trace without the parent of spans it already exported.
  """
  current = asyncio.current_task()
  leftovers = [
    task
    for task in asyncio.all_tasks()
    if task is not current and task.get_context().run(trace.get_current_span).get_span_context().trace_id == trace_id
  ]
  for task in leftovers:
    if not _CALLBACK_TASK.search(getattr(task.get_coro(), '__qualname__', '')):
      task.cancel()
  await asyncio.gather(*leftovers, return_exceptions=True)


def search_http_client(
  headers: dict[str, str] | None = None, timeout: httpx.Timeout | None = None, auth: httpx.Auth | None = None
) -> httpx.AsyncClient:
  """The MCP client's HTTP client, which gives up connecting to a server that is down within seconds."""
  timeout = timeout or httpx.Timeout(30.0)
  return httpx.AsyncClient(
    headers=headers,
    auth=auth,
    follow_redirects=True,
    timeout=httpx.Timeout(timeout.read, connect=_CONNECT_TIMEOUT_SECONDS, write=timeout.write, pool=timeout.pool),
    transport=httpx.AsyncHTTPTransport(retries=_CONNECT_RETRIES),
  )


async def mcp_tools(url: str) -> list[BaseTool]:
  """The tools of the MCP server at ``url``; each call of a tool opens its own session."""
  connection = StreamableHttpConnection(url=url, transport='streamable_http', httpx_client_factory=search_http_client)
  return await MultiServerMCPClient({'search': connection}).get_tools()


class LangChainTraceAdapter:
  """The canonical trace of one run: its root answers the question, and each search is a retrieval for it.

  A search becomes a retrieval span when its output is the search tool's JSON, which LangChain's MCP client returns as
  text blocks. A search that returned no documents, as the server rejected it or it was cancelled under way, stays a
  tool span; any other output becomes a retrieval span whose results are unavailable. A trace with an LLM call to any
  model but ``model``, or a dated version of it such as Azure reports, is refused, so that every call is known to use
  it.
  """

  def __init__(self, model: str, *, root_span_name: str, version: str):
    self._model = model
    self._models = re.compile(rf'{re.escape(model)}(?:-\d{{4}}-\d{{2}}-\d{{2}})?')
    self._root_span_name = root_span_name
    self._version = version

  def normalize(self, trace_id: str, records: list[dict[str, Any]]) -> TraceProcessingResult:
    base = PhoenixTraceAdapter().normalize(trace_id, records)
    roots = [record for record in records if record.get('parent_id') is None]
    if len(roots) != 1 or roots[0].get('name') != self._root_span_name:
      raise ValueError(f'Trace {trace_id} must have one root span, named {self._root_span_name}')
    # Read raw: the decoded values would turn an answer such as "42" into a number.
    question, answer = roots[0].get('attributes.input.value'), roots[0].get('attributes.output.value')
    if not isinstance(question, str) or not question.strip():
      raise ValueError(f'Trace {trace_id} has no question on its root span')
    spans = []
    for span in base.spans:
      usage = span.semantics.usage
      if usage is not None and not self._models.fullmatch(usage.model or ''):
        raise ValueError(f'Trace {trace_id} has an LLM call to {usage.model}, not {self._model}')
      if span.parent_span_id is None:
        semantics = SpanSemantics(request=question, answer=Answer(text=answer) if answer and answer.strip() else None)
        span = span.model_copy(update={'semantics': semantics})
      elif (span.metadata or {}).get('attributes.tool.name') == TOOL_NAME:
        retrieval = search_span_retrieval(span)
        if retrieval is not None:
          semantics = SpanSemantics(request=question, retrieval=[retrieval])
          span = span.model_copy(update={'span_type': SEARCH_SPAN_TYPE, 'semantics': semantics})
      spans.append(span)
    return TraceProcessingResult(
      trace=base.trace.model_copy(update={'adapter': self._version}), spans=with_retrieved_evidence(spans)
    )
