"""The Dify agent: a chat app built in Dify's UI, called through Dify's service API.

Each sample starts a new conversation under a request ID the caller draws and sends as Dify's ``trace_id``. Dify's
Phoenix tracing writes it as ``dify_trace_id`` on the root of the workflow trace, which holds the agent node and its
tool calls, so collection looks traces up by that attribute. The message ID would find Dify's separate message trace
instead, which holds only the question and the answer.
"""

import json
from datetime import timedelta
from typing import Any
from uuid import uuid4

import httpx
from pydantic import Field
from pydantic_settings import SettingsConfigDict

from syllo_eval.evaluation.trace_adapter import PhoenixTraceAdapter, TraceProcessingResult
from syllo_eval.model import Sample, Span
from syllo_eval.settings import EnvSettings
from syllo_eval.trace_semantics import Answer, LlmUsage, SpanSemantics

from ablation_metrics import SEARCH_SPAN_TYPE
from search_tool.search import search_retrieval, with_retrieved_evidence
from search_tool.server import TOOL_NAME

REQUEST_ID_ATTRIBUTE = 'dify_trace_id'
# Dify writes a tool's output in its agent log after this prefix, and a period after the search tool's JSON.
TOOL_OUTPUT_PREFIX = 'tool response: '


class DifySettings(EnvSettings):
  model_config = SettingsConfigDict(env_prefix='DIFY_')

  # Dify's service API, behind its nginx.
  base_url: str = 'http://localhost/v1'
  api_key: str | None = None
  # The end-user identifier Dify records for every message.
  user: str = Field(default='syllo-eval', min_length=1)
  timeout_seconds: float = Field(default=600.0, gt=0)
  # The search server that the app calls, as this host reaches it: collect checks what it serves.
  search_url: str = Field(default='http://127.0.0.1:8101/mcp', min_length=1)


class DifyError(RuntimeError):
  """A message Dify rejected, or a stream that failed or ended without an answer."""


class DifyCaller:
  """Sends a prompt as a new conversation and reads the stream to its end: agent apps do not answer in blocking mode.

  Each call opens its own client, so the caller holds nothing to close.
  """

  def __init__(self, settings: DifySettings, transport: httpx.AsyncBaseTransport | None = None) -> None:
    if settings.api_key is None:
      raise ValueError('DIFY_API_KEY must be set to the API key of the app under test')
    self._settings = settings
    self._transport = transport
    self._headers = {'Authorization': f'Bearer {settings.api_key}'}

  async def call(self, sample: Sample) -> str:
    request_id = str(uuid4())
    body = {
      'query': sample.input_prompt,
      'inputs': {},
      'user': self._settings.user,
      'response_mode': 'streaming',
      'trace_id': request_id,
    }
    async with (
      httpx.AsyncClient(
        base_url=self._settings.base_url, timeout=self._settings.timeout_seconds, transport=self._transport
      ) as client,
      client.stream('POST', '/chat-messages', headers=self._headers, json=body) as response,
    ):
      if response.status_code != 200:
        await response.aread()
        raise DifyError(f'Dify returned {response.status_code}: {response.text[:500]}')
      async for line in response.aiter_lines():
        if not line.startswith('data:'):
          continue
        event = json.loads(line.removeprefix('data:'))
        if event.get('event') == 'error':
          raise DifyError(f'Dify failed the message: {event.get("message")}')
        if event.get('event') == 'message_end':
          return request_id
    raise DifyError('Dify ended the stream before the end of the message')


class DifyTraceAdapter:
  """The workflow root answers the question, and an agent node's ReAct log becomes spans under the node.

  See dify/README.md, What the trace holds.
  """

  def normalize(self, trace_id: str, records: list[dict[str, Any]]) -> TraceProcessingResult:
    base = PhoenixTraceAdapter().normalize(trace_id, records)
    roots = [span for span in base.spans if span.parent_span_id is None]
    if len(roots) != 1 or f'attributes.{REQUEST_ID_ATTRIBUTE}' not in (roots[0].metadata or {}):
      raise ValueError(f'Trace {trace_id} must have one root span, with {REQUEST_ID_ATTRIBUTE}')
    root = roots[0]
    inputs = root.input_data if isinstance(root.input_data, dict) else {}
    outputs = root.output_data if isinstance(root.output_data, dict) else {}
    question, answer = inputs.get('sys.query'), outputs.get('answer')
    if not isinstance(question, str) or not question.strip():
      raise ValueError(f'Trace {trace_id} has no question on its root span')
    spans = []
    unreadable = []
    for span in base.spans:
      if span is root:
        semantics = SpanSemantics(
          request=question, answer=Answer(text=answer) if isinstance(answer, str) and answer.strip() else None
        )
        span = span.model_copy(update={'semantics': semantics})
      elif (span.metadata or {}).get('attributes.metadata.node_type') == 'agent':
        log = span.output_data.get('json') if isinstance(span.output_data, dict) else None
        log_spans, untimed = _log_spans(span, log, question)
        spans += log_spans
        if not isinstance(log, list) or any(
          (entry.get('data') or {}).get('tool_name') == TOOL_NAME for entry in untimed
        ):
          unreadable.append(span.external_id)
        if isinstance(span.output_data, dict):
          output = {k: v for k, v in span.output_data.items() if k != 'json'}
          metadata = {**(span.metadata or {}), 'dify.agent_log.untimed': untimed}
          span = span.model_copy(update={'output_data': output, 'metadata': metadata})
      spans.append(span)
    return TraceProcessingResult(
      trace=base.trace.model_copy(update={'adapter': 'dify:3'}), spans=with_retrieved_evidence(spans, unreadable)
    )


def _log_spans(agent: Span, log: Any, question: str) -> tuple[list[Span], list[dict[str, Any]]]:
  entries: list[dict[str, Any]] = []
  untimed: list[dict[str, Any]] = []
  for entry in log or []:
    if isinstance(entry, dict) and entry.get('label'):
      timed = {'started_at', 'finished_at'} <= (entry.get('metadata') or {}).keys()
      (entries if timed else untimed).append(entry)
  # The log's first entry is taken to start with its node; the log fits in the node with ~0.2 s to spare.
  origin = min((entry['metadata']['started_at'] for entry in entries), default=0.0)
  spans = []
  for entry in entries:
    data, clock = entry.get('data') or {}, entry['metadata']
    start = agent.start_time + timedelta(seconds=clock['started_at'] - origin)
    is_tool = 'tool_name' in data
    usage = None if is_tool else data.get('usage')
    span = Span(
      external_id=str(entry['id']),
      trace_id=agent.trace_id,
      parent_span_id=entry.get('parent_id') or agent.external_id,
      span_type='tool' if is_tool else 'llm' if 'provider' in clock else 'chain',
      name=str(entry['label']),
      start_time=start,
      end_time=start + timedelta(seconds=clock['finished_at'] - clock['started_at']),
      input_data=data.get('tool_call_args') if is_tool else None,
      output_data=data.get('output') if is_tool else data,
      status='success' if entry.get('status') == 'success' else 'error' if entry.get('status') == 'error' else 'unset',
      metadata={'dify.agent_log.metadata': clock, 'dify.agent_log.error': entry.get('error')},
      semantics=SpanSemantics(
        usage=LlmUsage(
          call_id=str(entry['id']),
          provider=clock.get('provider'),
          input_tokens=usage.get('prompt_tokens'),
          output_tokens=usage.get('completion_tokens'),
          total_tokens=clock.get('total_tokens'),
        )
        if isinstance(usage, dict)
        else None
      ),
    )
    if (
      data.get('tool_name') == TOOL_NAME
      and (retrieval := search_retrieval(_tool_output(data.get('output')))) is not None
    ):
      semantics = SpanSemantics(request=question, retrieval=[retrieval])
      span = span.model_copy(update={'span_type': SEARCH_SPAN_TYPE, 'semantics': semantics})
    spans.append(span)
  return spans, untimed


def _tool_output(output: Any) -> Any:
  if not isinstance(output, str) or not output.startswith(TOOL_OUTPUT_PREFIX):
    return output
  try:
    return json.JSONDecoder().raw_decode(output, len(TOOL_OUTPUT_PREFIX))[0]
  except ValueError:
    return output.removeprefix(TOOL_OUTPUT_PREFIX)
