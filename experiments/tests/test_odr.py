import asyncio
import builtins
import json
import os
import socket
import unittest
import warnings
from collections.abc import Awaitable, Callable, Sequence
from datetime import datetime, timedelta, timezone
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import anthropic
import httpx
import httpx2
import open_deep_research.deep_researcher as deep_research
import uvicorn
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.tools import ToolException
from langchain_core.utils.function_calling import convert_to_openai_tool
from openinference.instrumentation.langchain import LangChainInstrumentor
from opentelemetry.sdk.trace import ReadableSpan, SpanLimits, TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from opentelemetry.trace import StatusCode
from pydantic import Field, ValidationError
from sse_starlette.sse import AppStatus

from syllo_eval.model import Sample
from syllo_eval.settings import PhoenixSettings

from agents.odr import ROOT_SPAN_NAME, OdrCaller, OdrRunError, OdrSettings, OdrTraceAdapter, RunWatch, build_caller
from config import load_config
from indexing.embedding import AzureFoundrySettings, Embeddings, InputType
from indexing.vector_store import StoredPoint
from search_tool.search import KnowledgeBaseSearch
from search_tool.server import PATH, TOOL_NAME, build_server

MODEL = 'claude-sonnet-5-5'
QUESTION = 'How long do refunds take?'
TRACE_ID = 'd' * 32
T0 = datetime(2026, 10, 1, 12, 0, tzinfo=timezone.utc)
# Long texts with characters that JSON escapes, so that a cut or a re-encoding would show.
DOCUMENTS = [
  {
    'id': f'doc-{rank:02d}/"ü"',
    'title': f'Title {rank}: "quoted" — ünïcode',
    'text': f'Document {rank}.\n\tTabs, "quotes", back\\slashes, {{braces}} and ✓ marks. ' * 300,
  }
  for rank in range(10, 0, -1)
]
# How a search failed in the live runs when the MCP client got no answer from the server.
NO_ANSWER = builtins.ExceptionGroup('unhandled errors in a TaskGroup', [httpx.ConnectTimeout('')])


def _search_output(results: Sequence[dict[str, Any]], query: str = 'refund time') -> str:
  """A search span's output as LangChain's MCP client returns the tool's JSON: a list of text blocks."""
  ranked = [{'rank': rank, **result} for rank, result in enumerate(results, 1)]
  payload = json.dumps({'call_id': 'call-1', 'query': query, 'results': ranked})
  return json.dumps([{'type': 'text', 'text': payload, 'id': 'lc_1'}])


def _record(
  span_id: str, name: str, kind: str, parent: str | None, status: str = 'OK', **attributes: Any
) -> dict[str, Any]:
  """A span as the Phoenix client returns it: attributes flattened under ``attributes.``."""
  second = len(span_id)
  return {
    'name': name,
    'status_code': status,
    'parent_id': parent,
    'start_time': T0 + timedelta(seconds=second),
    'end_time': T0 + timedelta(seconds=second + 1),
    'context.span_id': span_id,
    'attributes.openinference.span.kind': kind,
    **{f'attributes.{key}': value for key, value in attributes.items()},
  }


def _trace(answer: str = '42', model: str = MODEL) -> list[dict[str, Any]]:
  search = {'tool.name': TOOL_NAME, 'input.value': 'refund time'}
  llm: dict[str, Any] = {'llm.model_name': model, 'llm.token_count.prompt': 100}
  failed = json.dumps([{'type': 'text', 'text': 'The search failed: the query is empty', 'id': 'lc_2'}])
  return [
    _record('r', ROOT_SPAN_NAME, 'AGENT', None, **{'input.value': QUESTION, 'output.value': answer}),
    _record('graph', 'LangGraph', 'CHAIN', 'r'),
    _record('llm', 'ChatAnthropic', 'LLM', 'graph', **llm),
    _record('tools', 'researcher_tools', 'CHAIN', 'graph'),
    _record('ok', TOOL_NAME, 'TOOL', 'tools', **search, **{'output.value': _search_output(DOCUMENTS)}),
    _record('fail', TOOL_NAME, 'TOOL', 'tools', **search, **{'output.value': failed}),
    _record('down', TOOL_NAME, 'TOOL', 'tools', status='ERROR', **search),
    _record('think', 'think_tool', 'TOOL', 'tools', **{'tool.name': 'think_tool', 'output.value': 'Reflection'}),
  ]


class OdrTraceAdapterTest(unittest.TestCase):
  def test_the_root_answers_and_each_search_is_a_ranked_retrieval_of_the_question(self) -> None:
    result = OdrTraceAdapter(MODEL).normalize(TRACE_ID, _trace())
    spans = {span.external_id: span for span in result.spans}

    root = spans['r']
    self.assertEqual(root.span_type, 'agent_root')
    self.assertEqual(root.semantics.request, QUESTION)
    assert root.semantics.answer is not None
    self.assertEqual(root.semantics.answer.text, '42')

    search = spans['ok']
    self.assertEqual(search.span_type, 'retrieval')
    self.assertEqual(search.semantics.request, QUESTION)
    [retrieval] = search.semantics.retrieval
    self.assertEqual((retrieval.kind, retrieval.stage, retrieval.ranked), ('document', 'selected', True))
    self.assertEqual(retrieval.query, 'refund time')
    self.assertEqual([item.id for item in retrieval.items], [document['id'] for document in DOCUMENTS])
    self.assertEqual([item.content for item in retrieval.items], [document['text'] for document in DOCUMENTS])

    self.assertEqual([spans[span_id].span_type for span_id in ('fail', 'down', 'think')], ['tool'] * 3)
    self.assertEqual(spans['llm'].semantics.usage.input_tokens, 100)  # type: ignore[union-attr]
    self.assertEqual(result.trace.adapter, 'odr:1')

  def test_a_call_to_another_model_is_refused(self) -> None:
    with self.assertRaisesRegex(ValueError, f'LLM call to gpt-4.1, not {MODEL}'):
      OdrTraceAdapter(MODEL).normalize(TRACE_ID, _trace(model='gpt-4.1'))

  def test_a_trace_without_the_callers_root_is_refused(self) -> None:
    records = _trace()[1:]
    records[0]['parent_id'] = None
    with self.assertRaisesRegex(ValueError, ROOT_SPAN_NAME):
      OdrTraceAdapter(MODEL).normalize(TRACE_ID, records)


def _llm_call(watch: RunWatch, tools: Sequence[str], node: str = 'researcher', style: str = 'anthropic') -> None:
  """An LLM call of an ODR node, given its tools in the provider's schema."""
  schemas: list[dict[str, Any]] = [
    {'name': name} if style == 'anthropic' else {'type': 'function', 'function': {'name': name}} for name in tools
  ]
  watch.on_chat_model_start(
    {}, [[]], run_id=uuid4(), metadata={'langgraph_node': node}, invocation_params={'tools': schemas}
  )


def _failed_run(watch: RunWatch, node: str, error: BaseException, name: str | None = None) -> None:
  """A failed run in an ODR node: by default the node's own run, which is named after it."""
  run_id = uuid4()
  watch.on_chain_start({}, {}, run_id=run_id, metadata={'langgraph_node': node}, name=name or node)
  watch.on_chain_error(error, run_id=run_id)


def _failed_search(watch: RunWatch, error: BaseException) -> None:
  run_id = uuid4()
  watch.on_tool_start({'name': TOOL_NAME}, 'refund time', run_id=run_id)
  watch.on_tool_error(error, run_id=run_id)


class ProviderError(Exception):
  def __init__(self, status_code: int):
    super().__init__(f'status {status_code}')
    self.status_code = status_code


class RunWatchTest(unittest.TestCase):
  def test_it_reports_each_sign_that_the_search_tool_was_unreachable(self) -> None:
    watch = RunWatch()
    _llm_call(watch, ['ResearchComplete', 'think_tool', TOOL_NAME])
    _llm_call(watch, ['ResearchComplete', 'think_tool', TOOL_NAME], style='openai')
    # The supervisor never searches.
    _llm_call(watch, ['ConductResearch', 'ResearchComplete', 'think_tool'], node='supervisor')
    self.assertEqual(watch.problems, [])

    _llm_call(watch, ['ResearchComplete', 'think_tool'], style='openai')
    _failed_run(watch, 'researcher_tools', KeyError(TOOL_NAME))
    _failed_search(watch, NO_ANSWER)

    self.assertEqual(
      watch.problems,
      [
        'a researcher ran without the search tool',
        "the search tool vanished before a researcher's searches",
        'a search got no answer from the search tool',
      ],
    )

  def test_a_researcher_that_failed_for_its_provider_is_reported(self) -> None:
    watch = RunWatch()
    request = httpx2.Request('POST', 'https://api.anthropic.com/v1/messages')

    _failed_run(watch, 'researcher', ProviderError(429))
    _failed_run(watch, 'researcher', anthropic.APIConnectionError(request=request))

    self.assertEqual(
      watch.problems, ['a researcher failed with ProviderError', 'a researcher failed with APIConnectionError']
    )

  def test_the_agents_own_failures_are_not_reported(self) -> None:
    watch = RunWatch()
    # The server's answer to a search it rejected.
    _failed_search(watch, ToolException('The search failed: the query is empty'))
    # A tool the model made up.
    _failed_run(watch, 'researcher_tools', KeyError('tavily_search'))
    # A request the provider refused, such as one over the context window.
    _failed_run(watch, 'researcher', ProviderError(400))
    # A failure inside the node, which ODR retries: only the node's own failure counts.
    _failed_run(watch, 'researcher', ProviderError(429), name='RunnableRetry')

    self.assertEqual(watch.problems, [])


def _sample() -> Sample:
  return Sample(id=uuid4(), dataset_id=uuid4(), input_prompt=QUESTION)


def _state(report: Any = 'The report.', **extra: Any) -> dict[str, Any]:
  return {'final_report': report, 'messages': [AIMessage(content=report)], **extra}


Step = Callable[[dict[str, Any]], Awaitable[dict[str, Any]]]


class FakeGraph:
  """Runs one scripted step per call; a step reaches the run's watch through the config, as ODR's callbacks do."""

  def __init__(self, *steps: Step):
    self._steps = list(steps)
    self.calls = 0

  async def ainvoke(self, graph_input: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    self.calls += 1
    return await self._steps.pop(0)(config)


def _watch(config: dict[str, Any]) -> RunWatch:
  [watch] = config['callbacks']
  return watch


def _returning(state: dict[str, Any]) -> Step:
  async def step(config: dict[str, Any]) -> dict[str, Any]:
    return state

  return step


def _raising(error: Exception) -> Step:
  async def step(config: dict[str, Any]) -> dict[str, Any]:
    raise error

  return step


def _tools(*names: str) -> list[MagicMock]:
  tools = []
  for name in names:
    tool = MagicMock()
    tool.name = name
    tools.append(tool)
  return tools


class OdrCallerTest(unittest.IsolatedAsyncioTestCase):
  def setUp(self) -> None:
    self.exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(self.exporter))
    self.tracer = provider.get_tracer('test')
    self.flushed: list[int] = []
    self.flush_results: list[bool] = []
    loading = patch('agents.odr.get_all_tools', AsyncMock(return_value=_tools('think_tool', TOOL_NAME)))
    self.get_all_tools = loading.start()
    self.addCleanup(loading.stop)

  def _caller(self, graph: FakeGraph, **options: Any) -> OdrCaller:
    def flush(trace_id: int) -> bool:
      self.flushed.append(trace_id)
      return self.flush_results.pop(0) if self.flush_results else True

    configurable = {'mcp_config': {'url': 'http://127.0.0.1:8101', 'tools': [TOOL_NAME], 'auth_required': False}}
    options = {'max_attempts': 3, 'retry_delay_seconds': 0, **options}
    return OdrCaller(
      configurable=configurable,
      tracer=self.tracer,
      flush=flush,
      metadata={'configuration': 'wixqa/odr/sonnet'},
      graph=graph,
      **options,
    )

  def _roots(self) -> list[ReadableSpan]:
    return [span for span in self.exporter.get_finished_spans() if span.name == ROOT_SPAN_NAME]

  async def test_the_root_span_holds_the_request_id_the_question_and_the_report_as_text(self) -> None:
    thinking = [{'type': 'thinking', 'thinking': '', 'signature': 'sig'}, {'type': 'text', 'text': 'The report.'}]
    caller = self._caller(FakeGraph(_returning(_state(thinking))))

    request_id = await caller.call(_sample())

    [root] = self._roots()
    attributes = root.attributes or {}
    self.assertEqual(attributes['request_id'], request_id)
    self.assertEqual((attributes['input.value'], attributes['output.value']), (QUESTION, 'The report.'))
    self.assertEqual(json.loads(str(attributes['metadata'])), {'configuration': 'wixqa/odr/sonnet'})
    self.assertIsNone(root.parent)
    assert root.context is not None
    self.assertEqual(self.flushed, [root.context.trace_id])

  async def test_an_unreachable_search_tool_runs_the_question_again(self) -> None:
    self.get_all_tools.side_effect = [_tools('think_tool'), _tools('think_tool', TOOL_NAME)]
    graph = FakeGraph(_returning(_state()))

    request_id = await self._caller(graph).call(_sample())

    self.assertEqual(graph.calls, 1)
    self.assertEqual([(span.attributes or {})['request_id'] for span in self._roots()], [request_id])

  async def test_a_researcher_without_the_search_tool_runs_the_question_again_under_a_new_request_id(self) -> None:
    async def lost_tool(config: dict[str, Any]) -> dict[str, Any]:
      _llm_call(_watch(config), ['ResearchComplete', 'think_tool'])
      return _state()

    graph = FakeGraph(lost_tool, _returning(_state()))

    request_id = await self._caller(graph).call(_sample())

    failed, answered = self._roots()
    self.assertEqual(failed.status.status_code, StatusCode.ERROR)
    self.assertIn('a researcher ran without the search tool', str(failed.status.description))
    self.assertEqual((answered.attributes or {})['request_id'], request_id)
    self.assertNotEqual((failed.attributes or {})['request_id'], request_id)

  async def test_a_researcher_that_failed_for_its_provider_runs_the_question_again(self) -> None:
    async def cut_short(config: dict[str, Any]) -> dict[str, Any]:
      # ODR ends its research when a researcher fails, and writes the report all the same.
      _failed_run(_watch(config), 'researcher', ProviderError(429))
      return _state()

    graph = FakeGraph(cut_short, _returning(_state()))

    await self._caller(graph).call(_sample())
    self.assertEqual(graph.calls, 2)

  async def test_a_search_that_the_server_rejected_is_the_agents_and_the_run_counts(self) -> None:
    async def empty_query(config: dict[str, Any]) -> dict[str, Any]:
      _failed_search(_watch(config), ToolException('The search failed: the query is empty'))
      return _state()

    graph = FakeGraph(empty_query)

    await self._caller(graph).call(_sample())
    self.assertEqual(graph.calls, 1)

  async def test_transient_failures_are_retried_until_the_attempts_run_out(self) -> None:
    graph = FakeGraph(
      _raising(ProviderError(429)),
      _raising(ProviderError(529)),
      _returning(_state('Error generating final report: Maximum retries exceeded')),
    )

    with self.assertRaisesRegex(OdrRunError, 'all 3 attempts; the last: Error generating final report'):
      await self._caller(graph).call(_sample())
    self.assertEqual(graph.calls, 3)
    self.assertEqual(self.flushed, [])

  async def test_a_run_that_takes_too_long_is_retried(self) -> None:
    async def hang(config: dict[str, Any]) -> dict[str, Any]:
      await asyncio.sleep(60)
      return _state()

    graph = FakeGraph(hang, _returning(_state()))

    await self._caller(graph, timeout_seconds=0.05).call(_sample())
    self.assertEqual(graph.calls, 2)

  async def test_other_failures_fail_the_sample_at_once(self) -> None:
    graph = FakeGraph(_raising(ProviderError(400)))

    with self.assertRaisesRegex(ProviderError, 'status 400'):
      await self._caller(graph).call(_sample())
    self.assertEqual(graph.calls, 1)

  async def test_a_run_that_lost_the_search_tool_runs_again_whatever_failed_next(self) -> None:
    async def lost_then_crashed(config: dict[str, Any]) -> dict[str, Any]:
      _failed_search(_watch(config), NO_ANSWER)
      raise KeyError('reflection')

    graph = FakeGraph(lost_then_crashed, _returning(_state()))

    await self._caller(graph).call(_sample())
    self.assertEqual(graph.calls, 2)
    self.assertIn('a search got no answer from the search tool', str(self._roots()[0].status.description))

  async def test_spans_that_did_not_reach_phoenix_run_the_question_again(self) -> None:
    self.flush_results = [False]
    graph = FakeGraph(_returning(_state()), _returning(_state()))

    request_id = await self._caller(graph).call(_sample())

    self.assertEqual(graph.calls, 2)
    self.assertEqual((self._roots()[1].attributes or {})['request_id'], request_id)

  async def test_the_tasks_a_run_left_running_are_cancelled_before_it_returns(self) -> None:
    left: list[asyncio.Task[None]] = []

    async def leave_a_task(config: dict[str, Any]) -> dict[str, Any]:
      left.append(asyncio.create_task(asyncio.sleep(60)))
      return _state()

    unrelated = asyncio.create_task(asyncio.sleep(60))
    self.addCleanup(unrelated.cancel)

    await self._caller(FakeGraph(leave_a_task)).call(_sample())

    self.assertTrue(left[0].cancelled())
    self.assertFalse(unrelated.done())

  async def test_lost_findings_are_logged_and_the_report_stands(self) -> None:
    lost = ToolMessage(content='Error synthesizing research report: Maximum retries exceeded', tool_call_id='c1')
    graph = FakeGraph(_returning(_state(supervisor_messages=[lost])))

    with self.assertLogs('agents.odr', 'WARNING') as logs:
      await self._caller(graph).call(_sample())
    self.assertIn('1 researchers lost their findings', logs.output[0])


class BuildCallerTest(unittest.TestCase):
  def setUp(self) -> None:
    self.config = load_config()
    self.phoenix = PhoenixSettings(project_id='wixqa-d662dc4')
    self.settings = OdrSettings(search_url='http://127.0.0.1:8101', azure_api_version='2025-04-01-preview')
    original = deep_research.configurable_model
    self.addCleanup(setattr, deep_research, 'configurable_model', original)
    instrumentor, model = patch('agents.odr.LangChainInstrumentor'), patch('agents.odr.init_chat_model')
    self.instrumentor, self.init_chat_model = instrumentor.start(), model.start()
    self.addCleanup(instrumentor.stop)
    self.addCleanup(model.stop)

  def _build(self, configuration_id: str, **options: Any) -> OdrCaller:
    configuration = self.config.configuration(configuration_id)
    return build_caller(configuration, self.config, phoenix=self.phoenix, settings=self.settings, **options)

  def test_sonnet_runs_every_phase_through_anthropic_without_web_search(self) -> None:
    caller = self._build('wixqa/odr/sonnet')

    configurable = caller._config['configurable']
    models = {configurable[setting] for setting in ('research_model', 'compression_model', 'final_report_model')}
    self.assertEqual(models | {configurable['summarization_model']}, {'anthropic:claude-sonnet-5-5'})
    self.assertEqual((configurable['search_api'], configurable['allow_clarification']), ('none', False))
    self.assertEqual(
      configurable['mcp_config'], {'url': 'http://127.0.0.1:8101', 'tools': [TOOL_NAME], 'auth_required': False}
    )
    self.assertEqual(configurable['mcp_prompt'], self.config.odr.mcp_prompt)
    self.init_chat_model.assert_called_once_with(
      'anthropic:claude-sonnet-5-5', configurable_fields=('model', 'max_tokens', 'api_key')
    )
    self.assertIs(deep_research.configurable_model, self.init_chat_model.return_value)
    self.instrumentor.return_value.instrument.assert_called_once()

  def test_deepseek_runs_on_the_foundry_deployment_with_the_foundry_key(self) -> None:
    foundry = AzureFoundrySettings(base_url='https://resource.services.ai.azure.com', api_key='foundry-key')

    caller = self._build('wixqa/odr/deepseek', foundry=foundry)

    self.assertEqual(caller._config['configurable']['research_model'], 'azure_openai:DeepSeek-V4.1-Flash')
    self.init_chat_model.assert_called_once_with(
      'azure_openai:DeepSeek-V4.1-Flash',
      configurable_fields=('model', 'max_tokens', 'api_key'),
      azure_endpoint='https://resource.services.ai.azure.com',
      api_key='foundry-key',
      api_version='2025-04-01-preview',
    )
    with self.assertRaisesRegex(ValueError, 'AZURE_FOUNDRY_BASE_URL and AZURE_FOUNDRY_API_KEY must be set'):
      self._build('wixqa/odr/deepseek', foundry=AzureFoundrySettings(base_url=None))

  def test_settings_that_odr_would_read_from_the_environment_are_refused(self) -> None:
    for name, value in (('SEARCH_API', 'tavily'), ('MCP_PROMPT', '')):
      with (
        self.subTest(name=name),
        patch.dict(os.environ, {name: value}),
        self.assertRaisesRegex(ValueError, f'unset {name}: ODR would read them'),
      ):
        self._build('wixqa/odr/sonnet')
    self.init_chat_model.assert_not_called()

  def test_the_search_url_is_the_servers_base_url(self) -> None:
    with self.assertRaisesRegex(ValidationError, 'without /mcp'):
      OdrSettings(search_url='http://127.0.0.1:8101/mcp')


class ScriptedModel(BaseChatModel):
  """Plays ODR's phases by the tools each is given: a brief, the topics of ``research``, then findings and a report.

  Each topic's researcher makes the calls that ``research`` gives it, then completes its research. The first
  ``researcher_failures`` researcher calls fail with a rate limit.
  """

  model: str = 'scripted-model'
  research: dict[str, list[tuple[str, dict[str, Any]]]] = Field(
    default_factory=lambda: {'Refund times': [(TOOL_NAME, {'query': f'refund {n}'}) for n in (1, 2)]}
  )
  researcher_failures: int = 0

  @property
  def _llm_type(self) -> str:
    return 'scripted'

  def bind_tools(self, tools: Sequence[Any], **kwargs: Any) -> Any:
    return self.bind(tools=[convert_to_openai_tool(tool) for tool in tools], **kwargs)

  def _generate(
    self, messages: list[BaseMessage], stop: Any = None, run_manager: Any = None, **kwargs: Any
  ) -> ChatResult:
    tools = {tool['function']['name'] for tool in kwargs.get('tools', [])}
    answered = any(isinstance(message, ToolMessage) for message in messages)
    if 'ResearchQuestion' in tools:
      calls = [('ResearchQuestion', {'research_brief': QUESTION})]
    elif 'ConductResearch' in tools:
      topics = [('ConductResearch', {'research_topic': topic}) for topic in self.research]
      calls = [('ResearchComplete', {})] if answered else topics
    elif TOOL_NAME in tools:
      if self.researcher_failures:
        self.researcher_failures -= 1
        raise ProviderError(429)
      topic = next(str(message.content) for message in messages if isinstance(message, HumanMessage))
      calls = [('ResearchComplete', {})] if answered else self.research[topic]
    else:
      return ChatResult(generations=[ChatGeneration(message=AIMessage(content='The report.'))])
    tool_calls = [
      {'name': name, 'args': args, 'id': f'call-{n}', 'type': 'tool_call'} for n, (name, args) in enumerate(calls)
    ]
    return ChatResult(generations=[ChatGeneration(message=AIMessage(content='', tool_calls=tool_calls))])


class LongIndex:
  """Ranks the documents in their order; a search for something slow waits until ``release`` is set."""

  def __init__(self, release: asyncio.Event):
    self._release = release

  async def search(self, vector: Sequence[float], query: str, limit: int) -> list[StoredPoint]:
    if 'slow' in query:
      await self._release.wait()
    return [
      StoredPoint(
        id=f'p{n}', payload={'document_id': document['id'], 'title': document['title'], 'text': document['text']}
      )
      for n, document in enumerate(DOCUMENTS[:limit])
    ]

  async def retrieve(self, ids: Sequence[str]) -> list[StoredPoint]:
    raise AssertionError('no swaps')


class FakeEmbedder:
  async def embed(self, texts: Sequence[str], input_type: InputType) -> Embeddings:
    return Embeddings(vectors=[[0.25, 0.75]], input_tokens=3)


def _phoenix_records(spans: Sequence[ReadableSpan]) -> list[dict[str, Any]]:
  """Exported spans as the Phoenix client returns them: attributes flattened under ``attributes.``."""
  records = []
  for span in spans:
    assert span.context is not None and span.start_time is not None and span.end_time is not None
    records.append(
      {
        'name': span.name,
        'status_code': span.status.status_code.name,
        'parent_id': format(span.parent.span_id, '016x') if span.parent else None,
        'start_time': datetime.fromtimestamp(span.start_time / 1e9, timezone.utc),
        'end_time': datetime.fromtimestamp(span.end_time / 1e9, timezone.utc),
        'context.span_id': format(span.context.span_id, '016x'),
        **{f'attributes.{key}': value for key, value in (span.attributes or {}).items()},
      }
    )
  return records


class OdrGraphTest(unittest.IsolatedAsyncioTestCase):
  """ODR's own graph, with a scripted model, searching through the real search server."""

  async def asyncSetUp(self) -> None:
    # The MCP server leaves a stream of each request to the garbage collector (mcp 1.12.4).
    self.enterContext(warnings.catch_warnings())
    warnings.filterwarnings('ignore', 'Unclosed <MemoryObject', ResourceWarning)
    # The graph's first run is slow to start, which the test loop's debug mode would report.
    asyncio.get_running_loop().slow_callback_duration = 5
    # sse-starlette 2.1 keeps one exit event per process, bound to the first event loop that waits on it: a server in
    # a later test's loop would fail every streamed response, and its MCP clients would wait for good.
    AppStatus.should_exit_event = None
    with socket.socket() as probe:
      probe.bind(('127.0.0.1', 0))
      port = probe.getsockname()[1]
    release = asyncio.Event()
    search = KnowledgeBaseSearch(LongIndex(release), FakeEmbedder())
    app = build_server(search, {'collection': 'kb-test'}).http_app(path=PATH, stateless_http=True)
    config = uvicorn.Config(app, host='127.0.0.1', port=port, ws='none', log_level='warning', access_log=False)
    server = uvicorn.Server(config)
    serving = asyncio.create_task(server.serve())
    self.addAsyncCleanup(self._stop, server, serving)
    # Before the server stops, which waits for the searches it is serving.
    self.addCleanup(release.set)
    while not server.started:
      if serving.done():
        serving.result()
      await asyncio.sleep(0.01)
    self.url = f'http://127.0.0.1:{port}'
    original = deep_research.configurable_model
    self.addCleanup(setattr, deep_research, 'configurable_model', original)

  @staticmethod
  async def _stop(server: uvicorn.Server, serving: asyncio.Task[None]) -> None:
    server.should_exit = True
    await serving

  def _caller(self, model: ScriptedModel, max_attempts: int = 1) -> tuple[OdrCaller, InMemorySpanExporter]:
    deep_research.configurable_model = model
    exporter = InMemorySpanExporter()
    provider = TracerProvider(span_limits=SpanLimits(max_span_attributes=SpanLimits.UNSET))
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    LangChainInstrumentor().instrument(tracer_provider=provider)
    self.addCleanup(LangChainInstrumentor().uninstrument)
    configurable = {
      **load_config().odr.model_dump(),
      'research_model': 'scripted',
      'compression_model': 'scripted',
      'final_report_model': 'scripted',
      'mcp_config': {'url': self.url, 'tools': [TOOL_NAME], 'auth_required': False},
    }
    caller = OdrCaller(
      configurable=configurable,
      tracer=provider.get_tracer('test'),
      flush=lambda trace_id: True,
      metadata={},
      max_attempts=max_attempts,
      retry_delay_seconds=0,
    )
    return caller, exporter

  async def test_every_search_reaches_the_trace_with_its_ten_documents_whole_and_in_rank_order(self) -> None:
    caller, exporter = self._caller(ScriptedModel())

    request_id = await caller.call(_sample())

    spans = exporter.get_finished_spans()
    [root] = [span for span in spans if span.parent is None]
    self.assertEqual((root.attributes or {})['request_id'], request_id)
    assert root.context is not None
    result = OdrTraceAdapter('scripted-model').normalize(format(root.context.trace_id, '032x'), _phoenix_records(spans))
    searches = [span for span in result.spans if span.span_type == 'retrieval']
    self.assertEqual(len(searches), 2)
    for search in searches:
      [retrieval] = search.semantics.retrieval
      self.assertEqual(search.semantics.request, QUESTION)
      self.assertEqual([item.id for item in retrieval.items], [document['id'] for document in DOCUMENTS])
      self.assertEqual([item.title for item in retrieval.items], [document['title'] for document in DOCUMENTS])
      self.assertEqual([item.content for item in retrieval.items], [document['text'] for document in DOCUMENTS])
    [answer] = [span.semantics.answer for span in result.spans if span.span_type == 'agent_root']
    assert answer is not None
    self.assertEqual(answer.text, 'The report.')

  async def test_when_a_researcher_fails_its_siblings_stop_with_the_call_and_the_trace_stays_whole(self) -> None:
    # A tool the model made up fails its researcher, which ends ODR's research while the other is still searching.
    research = {
      'Made-up tool': [('tavily_search', {'query': 'refunds'})],
      'Slow search': [(TOOL_NAME, {'query': 'slow refunds'})],
    }
    caller, exporter = self._caller(ScriptedModel(research=research))

    request_id = await caller.call(_sample())
    returned = len(exporter.get_finished_spans())
    await asyncio.sleep(0.5)

    spans = exporter.get_finished_spans()
    self.assertEqual(len(spans), returned)
    [root] = [span for span in spans if span.parent is None]
    self.assertEqual((root.attributes or {})['request_id'], request_id)
    assert root.context is not None
    result = OdrTraceAdapter('scripted-model').normalize(format(root.context.trace_id, '032x'), _phoenix_records(spans))
    self.assertEqual([span for span in result.spans if span.span_type == 'retrieval'], [])
    self.assertIn('researcher_tools', {span.name for span in result.spans if span.status == 'error'})

  async def test_a_researcher_that_failed_for_its_provider_runs_the_question_again(self) -> None:
    # Every attempt ODR makes at the first researcher call: the researcher fails, and ODR ends its research.
    caller, exporter = self._caller(ScriptedModel(researcher_failures=3), max_attempts=2)

    request_id = await caller.call(_sample())

    failed, answered = [span for span in exporter.get_finished_spans() if span.parent is None]
    self.assertIn('a researcher failed with ProviderError', str(failed.status.description))
    self.assertEqual((answered.attributes or {})['request_id'], request_id)


if __name__ == '__main__':
  unittest.main()
