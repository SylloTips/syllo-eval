"""The smolagents CodeAgent, run in this process, and the adapter of its traces.

Each call runs in a worker thread, as smolagents is synchronous, under a root span with the request id, the question
and the answer. Spans are exported as they end, so the trace is in Phoenix when the call returns.
"""

import asyncio
from typing import Any
from uuid import uuid4

from openinference.instrumentation.smolagents import SmolagentsInstrumentor
from openinference.semconv.resource import ResourceAttributes
from opentelemetry import trace
from opentelemetry.context import Context
from opentelemetry.exporter.otlp.proto.http import Compression
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import SpanLimits, TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from pydantic import Field, ValidationError
from pydantic_settings import SettingsConfigDict
from smolagents import CodeAgent, LiteLLMModel, LogLevel, MCPClient

from syllo_eval.evaluation.trace_adapter import PhoenixTraceAdapter, TraceProcessingResult
from syllo_eval.model import Sample
from syllo_eval.settings import EnvSettings, PhoenixSettings
from syllo_eval.trace_semantics import Answer, SpanSemantics

from ablation_metrics import SEARCH_SPAN_TYPE
from config import Configuration, ExperimentConfig
from indexing.embedding import AzureFoundrySettings
from search_tool.search import SearchResults
from search_tool.server import TOOL_NAME

ROOT_SPAN_NAME = 'smolagents.request'
PLANNING_INTERVAL = 3


class CodeAgentSettings(EnvSettings):
  model_config = SettingsConfigDict(env_prefix='SMOLAGENTS_')

  search_url: str = Field(default='http://127.0.0.1:8101/mcp', min_length=1)


class CodeAgentCaller:
  def __init__(self, *, model: dict[str, Any], search_url: str, tracer: trace.Tracer):
    self._model = model
    self._search_url = search_url
    self._tracer = tracer

  async def call(self, sample: Sample) -> str:
    request_id = str(uuid4())
    await asyncio.to_thread(self._run, sample.input_prompt, request_id)
    return request_id

  def _run(self, question: str, request_id: str) -> None:
    attributes = {'request_id': request_id, 'openinference.span.kind': 'AGENT', 'input.value': question}
    # An empty parent context makes this a root span, which the request_id lookup requires.
    with (
      self._tracer.start_as_current_span(ROOT_SPAN_NAME, context=Context(), attributes=attributes) as span,
      # Text output, so that the agent reads the same JSON as the other agents.
      MCPClient({'url': self._search_url, 'transport': 'streamable-http'}, structured_output=False) as tools,
    ):
      agent = CodeAgent(
        tools=tools,
        model=LiteLLMModel(**self._model),
        planning_interval=PLANNING_INTERVAL,
        additional_authorized_imports=['json'],
        verbosity_level=LogLevel.OFF,
      )
      span.set_attribute('output.value', str(agent.run(question)))


def build_caller(
  configuration: Configuration,
  config: ExperimentConfig,
  *,
  phoenix: PhoenixSettings,
) -> CodeAgentCaller:
  ref = config.models.agents[configuration.model]
  # Provider defaults for every model parameter, as in the Dify agent.
  model: dict[str, Any] = {'model_id': f'{ref.provider}/{ref.model}'}
  if ref.provider == 'azure_ai':
    foundry = AzureFoundrySettings()
    if not (foundry.base_url and foundry.api_key):
      raise ValueError(f'{configuration.id}: AZURE_FOUNDRY_BASE_URL and AZURE_FOUNDRY_API_KEY must be set')
    model.update(api_base=f'{foundry.base_url.rstrip("/")}/models', api_key=foundry.api_key)
  provider = TracerProvider(
    resource=Resource.create({ResourceAttributes.PROJECT_NAME: str(phoenix.project_id)}),
    # The SDK keeps 128 attributes per span by default: a long run's LLM spans would lose their last messages.
    span_limits=SpanLimits(max_span_attributes=SpanLimits.UNSET),
  )
  # OTLP over HTTP with gzip: Phoenix's gRPC receiver keeps gRPC's 4 MiB message limit, which long traces exceed.
  provider.add_span_processor(
    SimpleSpanProcessor(
      OTLPSpanExporter(
        endpoint=f'{phoenix.base_url.rstrip("/")}/v1/traces',
        headers={'authorization': f'Bearer {phoenix.api_key}'} if phoenix.api_key else None,
        compression=Compression.Gzip,
      )
    )
  )
  # Instrumentation is process-wide: one collection per process, as `syllo-exp collect` runs.
  SmolagentsInstrumentor().instrument(tracer_provider=provider)
  return CodeAgentCaller(
    model=model,
    search_url=CodeAgentSettings().search_url,
    tracer=provider.get_tracer(__name__),
  )


class CodeAgentTraceAdapter:
  """Searches that returned the search tool's JSON become retrieval spans, with the question as their request.

  A failed search, whose output is an error message, stays a tool span: it is not a search unit.
  """

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
      if span.parent_span_id is None:
        semantics = SpanSemantics(request=question, answer=Answer(text=answer) if answer and answer.strip() else None)
        span = span.model_copy(update={'semantics': semantics})
      elif (span.metadata or {}).get('attributes.tool.name') == TOOL_NAME:
        try:
          retrieval = SearchResults.model_validate(span.output_data).retrieval()
        except ValidationError:
          pass
        else:
          semantics = SpanSemantics(request=question, retrieval=[retrieval])
          span = span.model_copy(update={'span_type': SEARCH_SPAN_TYPE, 'semantics': semantics})
      spans.append(span)
    return TraceProcessingResult(trace=base.trace.model_copy(update={'adapter': 'smolagents:1'}), spans=spans)
