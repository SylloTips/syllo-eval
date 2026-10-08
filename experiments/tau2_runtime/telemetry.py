"""The tracer provider of one simulation process. Configure it before anything imports tau2.

tau2's ``llm_utils`` binds ``litellm.completion`` when it is first imported, so litellm must already be instrumented by
then, or tau2's calls produce no LLM spans.
"""

from collections.abc import Mapping, Sequence

from openinference.instrumentation.litellm import LiteLLMInstrumentor
from openinference.semconv.resource import ResourceAttributes
from opentelemetry import trace
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import ReadableSpan, SpanLimits, TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor, SpanExporter, SpanExportResult


def configure_tracing(project: str, exporter: SpanExporter) -> TracerProvider:
  """Spans go to the Phoenix ``project``; LLM calls through litellm are traced from now on."""
  provider = TracerProvider(
    resource=Resource.create({ResourceAttributes.PROJECT_NAME: project}),
    # The SDK keeps 128 attributes per span by default and drops the rest: a long conversation's LLM span would lose
    # its model name and its first messages.
    span_limits=SpanLimits(max_span_attributes=SpanLimits.UNSET),
  )
  provider.add_span_processor(BatchSpanProcessor(exporter))
  trace.set_tracer_provider(provider)
  LiteLLMInstrumentor().instrument(tracer_provider=provider)
  return provider


class CheckedExporter(SpanExporter):
  """Remembers whether any batch failed to export: a flush reports only whether the export finished in time."""

  def __init__(self, exporter: SpanExporter):
    self._exporter = exporter
    self.failed = False

  def export(self, spans: Sequence[ReadableSpan]) -> SpanExportResult:
    result = self._exporter.export(spans)
    if result is not SpanExportResult.SUCCESS:
      self.failed = True
    return result

  def shutdown(self) -> None:
    self._exporter.shutdown()

  def force_flush(self, timeout_millis: int = 30_000) -> bool:
    return self._exporter.force_flush(timeout_millis)


def otlp_exporter(endpoint: str, headers: Mapping[str, str]) -> CheckedExporter:
  """OTLP over HTTP with gzip: Phoenix's gRPC receiver keeps gRPC's 4 MiB message limit, which long traces exceed."""
  from opentelemetry.exporter.otlp.proto.http import Compression
  from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter

  return CheckedExporter(
    OTLPSpanExporter(endpoint=endpoint, headers=dict(headers), compression=Compression.Gzip, timeout=30)
  )
