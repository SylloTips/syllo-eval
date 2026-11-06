import asyncio
import builtins
import hashlib
import json
import re
import unittest
from collections.abc import Awaitable, Callable, Sequence
from datetime import datetime, timedelta, timezone
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import httpx
from langchain.chat_models import init_chat_model
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage, message_to_dict
from langchain_core.outputs import ChatGeneration, ChatResult, LLMResult
from langchain_core.utils.function_calling import convert_to_openai_tool
from openinference.instrumentation.langchain import LangChainInstrumentor
from opentelemetry.sdk.trace import ReadableSpan, SpanLimits, TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from pydantic import Field

from syllo_eval.model import Sample, Span
from syllo_eval.settings import PhoenixSettings

from agents import deep_research
from agents.deep_agent import (
  _model,
  ROOT_SPAN_NAME,
  DeepAgentCaller,
  DeepAgentRunError,
  DeepAgentSettings,
  DeepAgentTraceAdapter,
  RunWatch,
  build_agent,
  build_caller,
  orchestrator_prompt,
  researcher_prompt,
)
from agents.deep_research.tools import think_tool
from config import EXPERIMENTS_DIR, load_config
from indexing.embedding import AzureFoundrySettings
from search_fakes import DOCUMENTS, SearchServerTestCase, phoenix_records
from search_tool.server import PATH, TOOL_NAME

MODEL = 'gpt-6.1-sol'
QUESTION = 'How long do refunds take?'
TRACE_ID = 'e' * 32
T0 = datetime(2026, 11, 6, 12, 0, tzinfo=timezone.utc)
DATE = '2026-11-06'
# How a search failed when the MCP client got no answer from the server.
NO_ANSWER = builtins.ExceptionGroup('unhandled errors in a TaskGroup', [httpx.ConnectTimeout('')])


class ExampleTest(unittest.TestCase):
  def test_the_prompts_are_the_examples_at_the_pinned_commit(self) -> None:
    prompts = (EXPERIMENTS_DIR / 'agents' / 'deep_research' / 'prompts.py').read_bytes()
    self.assertEqual(
      hashlib.sha256(prompts).hexdigest(), '3c828da7d23a1c25181370549a881154293845b9be9b047d47ae24d59d336ce4'
    )
    self.assertEqual(deep_research.COMMIT, '9ea3f1912abd8b7eecb504d78ef99fcf279bf561')

  def test_the_prompts_name_the_search_tool_cite_document_ids_and_ask_for_the_answer_as_the_last_message(self) -> None:
    orchestrator = orchestrator_prompt(load_config().deepagents)
    researcher = researcher_prompt(DATE)

    for prompt in (orchestrator, researcher):
      for web in ('URL', 'tavily', 'example.com', '/final_report.md'):
        self.assertNotIn(web, prompt)
    self.assertIn(f'**{TOOL_NAME}**: For searching the knowledge base', researcher)
    self.assertIn("today's date is 2026-11-06", researcher)
    self.assertIn('listing each numbered source with title and document id', researcher)
    self.assertIn('[1] Document Title: document id', orchestrator)
    # The request is checked before the report, which is the last message.
    self.assertLess(orchestrator.index('**Verify**'), orchestrator.index('**Write Report**'))
    self.assertIn('Do not write it to a file: your final message is the answer', orchestrator)
    self.assertIn('Research every request as asked: do not ask clarifying questions', orchestrator)
    self.assertIn('Use at most 3 parallel sub-agents per iteration', orchestrator)
    self.assertIn('Stop after 3 delegation rounds', orchestrator)

  def test_a_passage_that_an_edit_no_longer_finds_fails_loudly(self) -> None:
    with (
      patch('agents.deep_agent.RESEARCHER_INSTRUCTIONS', 'Another prompt, for {date}.'),
      self.assertRaisesRegex(RuntimeError, 'no longer has'),
    ):
      researcher_prompt(DATE)


def _record(span_id: str, name: str, kind: str, parent: str | None, second: int, **attributes: Any) -> dict[str, Any]:
  """A span as the Phoenix client returns it: attributes flattened under ``attributes.``."""
  return {
    'name': name,
    'status_code': 'OK',
    'status_message': '',
    'parent_id': parent,
    'start_time': T0 + timedelta(seconds=second),
    'end_time': T0 + timedelta(seconds=second + 1),
    'context.span_id': span_id,
    'attributes.openinference.span.kind': kind,
    **{f'attributes.{key}': value for key, value in attributes.items()},
  }


def _search_message(query: str, results: Sequence[dict[str, Any]] = DOCUMENTS) -> str:
  """A search span's output as OpenInference records it under create_agent: the whole tool message."""
  ranked = [{'rank': rank, **result} for rank, result in enumerate(results, 1)]
  payload = {'call_id': uuid4().hex, 'query': query, 'results': ranked}
  message = ToolMessage(
    content=[{'type': 'text', 'text': json.dumps(payload), 'id': 'lc_1'}],
    artifact={'structured_content': payload},
    tool_call_id='call-1',
    name=TOOL_NAME,
  )
  return json.dumps(message_to_dict(message))


def _trace(model: str = f'{MODEL}-2026-09-29') -> list[dict[str, Any]]:
  rejected = ToolMessage(
    content='The search failed: the query is empty', tool_call_id='call-2', name=TOOL_NAME, status='error'
  )
  # The MCP server's own answer to arguments the tool does not take.
  invalid = ToolMessage(
    content="Input validation error: 'query' is a required property",
    tool_call_id='call-3',
    name=TOOL_NAME,
    status='error',
  )
  evicted = json.dumps(_search_message('refund 1'))
  return [
    _record('r', ROOT_SPAN_NAME, 'AGENT', None, 0, **{'input.value': QUESTION, 'output.value': 'The report.'}),
    _record('graph', 'LangGraph', 'CHAIN', 'r', 1),
    _record('llm', 'AzureChatOpenAI', 'LLM', 'graph', 2, **{'llm.model_name': model, 'llm.token_count.prompt': 100}),
    _record('search', TOOL_NAME, 'TOOL', 'graph', 3, **{'tool.name': TOOL_NAME, 'output.value': _search_message('a')}),
    _record('task', 'task', 'TOOL', 'graph', 4, **{'tool.name': 'task', 'output.value': 'Findings.'}),
    _record('sub-llm', 'AzureChatOpenAI', 'LLM', 'task', 5, **{'llm.model_name': model}),
    _record(
      'sub-search', TOOL_NAME, 'TOOL', 'task', 6, **{'tool.name': TOOL_NAME, 'output.value': _search_message('b')}
    ),
    _record(
      'rejected',
      TOOL_NAME,
      'TOOL',
      'task',
      7,
      **{'tool.name': TOOL_NAME, 'output.value': json.dumps(message_to_dict(rejected))},
    ),
    _record(
      'invalid',
      TOOL_NAME,
      'TOOL',
      'task',
      7,
      **{'tool.name': TOOL_NAME, 'output.value': json.dumps(message_to_dict(invalid))},
    ),
    # The agent reads again a search result that was moved to its files.
    _record('read', 'read_file', 'TOOL', 'task', 8, **{'tool.name': 'read_file', 'output.value': evicted}),
    _record('think', 'think_tool', 'TOOL', 'task', 9, **{'tool.name': 'think_tool', 'output.value': 'Reflection'}),
  ]


class DeepAgentTraceAdapterTest(unittest.TestCase):
  def test_every_search_of_the_orchestrator_and_its_sub_agents_is_a_ranked_retrieval_of_the_question(self) -> None:
    result = DeepAgentTraceAdapter(MODEL).normalize(TRACE_ID, _trace())
    spans = {span.external_id: span for span in result.spans}

    root = spans['r']
    assert root.semantics.answer is not None
    self.assertEqual((root.semantics.request, root.semantics.answer.text), (QUESTION, 'The report.'))
    for span_id, query in (('search', 'a'), ('sub-search', 'b')):
      search = spans[span_id]
      self.assertEqual((search.span_type, search.semantics.request), ('retrieval', QUESTION))
      [retrieval] = search.semantics.retrieval
      self.assertEqual((retrieval.stage, retrieval.ranked, retrieval.query), ('selected', True, query))
      self.assertEqual([item.id for item in retrieval.items], [document['id'] for document in DOCUMENTS])
      self.assertEqual([item.title for item in retrieval.items], [document['title'] for document in DOCUMENTS])
      self.assertEqual([item.content for item in retrieval.items], [document['text'] for document in DOCUMENTS])
    # A rejected search is a failed call, and reading a search result again is no new search.
    calls = ('task', 'rejected', 'invalid', 'read', 'think')
    self.assertEqual([spans[span_id].span_type for span_id in calls], ['tool'] * 5)
    [evidence] = root.semantics.retrieval
    self.assertEqual((evidence.stage, evidence.availability), ('retrieved', 'available'))
    self.assertEqual([item.id for item in evidence.items], [document['id'] for document in DOCUMENTS])
    self.assertEqual(result.trace.adapter, 'deepagents:1')

  def test_a_call_to_another_model_is_refused(self) -> None:
    for model in ('gpt-6-luna', f'{MODEL}-mini', 'DeepSeek-V4.1-Flash'):
      with self.subTest(model=model), self.assertRaisesRegex(ValueError, f'LLM call to {model}, not {MODEL}'):
        DeepAgentTraceAdapter(MODEL).normalize(TRACE_ID, _trace(model=model))

  def test_a_trace_without_the_callers_root_is_refused(self) -> None:
    records = _trace()[1:]
    records[0]['parent_id'] = None
    with self.assertRaisesRegex(ValueError, ROOT_SPAN_NAME):
      DeepAgentTraceAdapter(MODEL).normalize(TRACE_ID, records)


def _search_error(watch: RunWatch, error: BaseException, name: str = TOOL_NAME) -> None:
  run_id = uuid4()
  watch.on_tool_start({'name': name}, '{}', run_id=run_id)
  watch.on_tool_error(error, run_id=run_id)


def _llm_end(watch: RunWatch, **metadata: Any) -> None:
  message = AIMessage(content='', response_metadata=metadata)
  watch.on_llm_end(LLMResult(generations=[[ChatGeneration(message=message)]]), run_id=uuid4())


class RunWatchTest(unittest.TestCase):
  def test_a_search_without_an_answer_is_reported_but_a_cancelled_search_or_another_tool_is_not(self) -> None:
    watch = RunWatch()
    _search_error(watch, asyncio.CancelledError())
    _search_error(watch, ValueError('made-up path'), name='read_file')
    self.assertEqual(watch.problems, [])

    _search_error(watch, NO_ANSWER)
    self.assertEqual(watch.problems, ['a search got no answer from the search tool'])

  def test_a_cut_off_model_call_is_reported(self) -> None:
    watch = RunWatch()
    _llm_end(watch, status='completed')
    _llm_end(watch, finish_reason='stop')
    _llm_end(watch, finish_reason='tool_calls')
    self.assertEqual(watch.problems, [])

    _llm_end(watch, status='incomplete', incomplete_details={'reason': 'max_output_tokens'})
    _llm_end(watch, finish_reason='length')
    _llm_end(watch, status='incomplete', incomplete_details={'reason': 'content_filter'})
    # What LangChain's Azure client raises when a content filter stopped a Chat Completions response.
    filtered = ValueError('Azure has not provided the response due to a content filter being triggered')
    watch.on_llm_error(filtered, run_id=uuid4())
    watch.on_llm_error(ValueError('another failure'), run_id=uuid4())
    self.assertEqual(
      watch.problems,
      [
        'a model call was cut off (max_output_tokens)',
        'a model call was cut off (length)',
        'a model call was cut off (content_filter)',
        'a model call was cut off (content_filter)',
      ],
    )


class ProviderError(Exception):
  def __init__(self, status_code: int):
    super().__init__(f'status {status_code}')
    self.status_code = status_code


def _sample() -> Sample:
  return Sample(id=uuid4(), dataset_id=uuid4(), input_prompt=QUESTION)


def _state(answer: Any = 'The report.', **extra: Any) -> dict[str, Any]:
  return {'messages': [HumanMessage(content=QUESTION), AIMessage(content=answer)], **extra}


Step = Callable[[dict[str, Any]], Awaitable[dict[str, Any]]]


class FakeAgent:
  """Runs one scripted step per call; a step reaches the run's watch through the config, as the callbacks do."""

  def __init__(self, *steps: Step):
    self._steps = list(steps)
    self.calls = 0

  async def ainvoke(self, graph_input: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    self.calls += 1
    return await self._steps.pop(0)(config)


def _watch(config: dict[str, Any]) -> RunWatch:
  [watch] = config['callbacks']
  return watch


def _returning(state: dict[str, Any], problem: Callable[[RunWatch], None] | None = None) -> Step:
  async def step(config: dict[str, Any]) -> dict[str, Any]:
    if problem is not None:
      problem(_watch(config))
    return state

  return step


def _raising(error: Exception) -> Step:
  async def step(config: dict[str, Any]) -> dict[str, Any]:
    raise error

  return step


def _tool(name: str) -> MagicMock:
  tool = MagicMock()
  tool.name = name
  return tool


class DeepAgentCallerTest(unittest.IsolatedAsyncioTestCase):
  def setUp(self) -> None:
    self.exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(self.exporter))
    self.tracer = provider.get_tracer('test')
    self.flush_results: list[bool] = []
    self.search = _tool(TOOL_NAME)
    self.search.handle_tool_error = False
    loading = patch('agents.deep_agent.mcp_tools', AsyncMock(return_value=[self.search, _tool('other')]))
    self.mcp_tools = loading.start()
    self.addCleanup(loading.stop)
    self.built: list[list[Any]] = []

  def _caller(self, agent: FakeAgent, **options: Any) -> DeepAgentCaller:
    def build(tools: Sequence[Any]) -> FakeAgent:
      self.built.append(list(tools))
      return agent

    def flush(trace_id: int) -> bool:
      return self.flush_results.pop(0) if self.flush_results else True

    options = {'max_attempts': 3, 'retry_delay_seconds': 0, **options}
    return DeepAgentCaller(
      build=build,
      search_url='http://127.0.0.1:8101/mcp',
      tracer=self.tracer,
      flush=flush,
      metadata={'configuration': 'wixqa/deepagents/gpt'},
      **options,
    )

  def _roots(self) -> list[ReadableSpan]:
    return [span for span in self.exporter.get_finished_spans() if span.name == ROOT_SPAN_NAME]

  async def test_the_root_span_holds_the_request_id_the_question_and_the_answer_as_text(self) -> None:
    # The Responses API's reply: its reasoning, then its text.
    reply = [{'type': 'reasoning', 'id': 'rs_1', 'summary': []}, {'type': 'text', 'text': 'The report.'}]
    agent = FakeAgent(_returning(_state(reply)), _returning(_state()))
    caller = self._caller(agent)

    request_id = await caller.call(_sample())
    await caller.call(_sample())

    first, _ = self._roots()
    attributes = first.attributes or {}
    self.assertEqual(attributes['request_id'], request_id)
    self.assertEqual((attributes['input.value'], attributes['output.value']), (QUESTION, 'The report.'))
    self.assertEqual(json.loads(str(attributes['metadata'])), {'configuration': 'wixqa/deepagents/gpt'})
    # Built once, with the server's search tool, which hands the server's errors back to the agent.
    [tools] = self.built
    self.assertEqual([tool.name for tool in tools], [TOOL_NAME, 'think_tool'])
    self.assertIs(self.search.handle_tool_error, True)
    self.mcp_tools.assert_awaited_once_with('http://127.0.0.1:8101/mcp')

  async def test_an_unreachable_search_tool_runs_the_question_again(self) -> None:
    self.mcp_tools.side_effect = [httpx.ConnectError('refused'), [self.search]]
    agent = FakeAgent(_returning(_state()))

    await self._caller(agent).call(_sample())

    self.assertEqual((self.mcp_tools.await_count, len(self.built), agent.calls), (2, 1, 1))

  async def test_a_search_without_an_answer_runs_the_question_again_under_a_new_request_id(self) -> None:
    agent = FakeAgent(
      _returning(_state(), lambda watch: _search_error(watch, NO_ANSWER)),
      _returning(_state(), lambda watch: _llm_end(watch, finish_reason='length')),
      _returning(_state()),
    )

    request_id = await self._caller(agent).call(_sample())

    first, second, answered = self._roots()
    self.assertIn('a search got no answer from the search tool', str(first.status.description))
    self.assertIn('a model call was cut off (length)', str(second.status.description))
    self.assertEqual((answered.attributes or {})['request_id'], request_id)
    self.assertNotEqual((first.attributes or {})['request_id'], request_id)

  async def test_transient_failures_are_retried_until_the_attempts_run_out(self) -> None:
    agent = FakeAgent(*(_raising(ProviderError(status)) for status in (429, 503, 500)))

    with self.assertRaisesRegex(DeepAgentRunError, 'in all 3 attempts; the last: ProviderError: status 500'):
      await self._caller(agent).call(_sample())
    self.assertEqual(agent.calls, 3)

  async def test_an_empty_answer_a_slow_run_and_lost_spans_run_the_question_again(self) -> None:
    async def slow(config: dict[str, Any]) -> dict[str, Any]:
      await asyncio.sleep(1)
      return _state()

    self.flush_results = [False, True]
    agent = FakeAgent(_returning(_state('  ')), slow, _returning(_state()), _returning(_state()))

    await self._caller(agent, max_attempts=4, timeout_seconds=0.1).call(_sample())

    reasons = [str(root.status.description) for root in self._roots()[:2]]
    self.assertIn('the run ended without an answer', reasons[0])
    self.assertIn('the run took longer than 0.1s', reasons[1])
    self.assertEqual(agent.calls, 4)

  async def test_other_failures_fail_the_sample_at_once(self) -> None:
    agent = FakeAgent(_raising(ValueError('a made-up sub-agent')))

    with self.assertRaisesRegex(ValueError, 'a made-up sub-agent'):
      await self._caller(agent).call(_sample())
    self.assertEqual(agent.calls, 1)

  async def test_a_report_file_and_tool_results_moved_to_files_are_logged(self) -> None:
    files: dict[str, Any] = {'/final_report.md': {}, '/large_tool_results/call-1': {}, '/research_request.md': {}}
    agent = FakeAgent(_returning(_state(files=files)))

    with self.assertLogs('agents.deep_agent', 'WARNING') as logs:
      await self._caller(agent).call(_sample())

    self.assertIn('wrote /final_report.md', logs.output[0])
    self.assertIn("moved 1 tool results to its files: ['/large_tool_results/call-1']", logs.output[1])


class BuildCallerTest(unittest.TestCase):
  def setUp(self) -> None:
    self.config = load_config()
    self.phoenix = PhoenixSettings(project_id='wixqa-d662dc4')
    self.settings = DeepAgentSettings(search_url='http://127.0.0.1:8101/mcp', azure_api_version='2025-04-01-preview')
    self.foundry = AzureFoundrySettings(base_url='https://resource.services.ai.azure.com', api_key='foundry-key')
    instrumentor, model = (
      patch('agents.langchain_runs.LangChainInstrumentor'),
      patch('agents.deep_agent.init_chat_model'),
    )
    self.instrumentor, self.init_chat_model = instrumentor.start(), model.start()
    self.addCleanup(instrumentor.stop)
    self.addCleanup(model.stop)

  def _build(self, configuration_id: str, **options: Any) -> DeepAgentCaller:
    configuration = self.config.configuration(configuration_id)
    options = {'foundry': self.foundry, **options}
    return build_caller(configuration, self.config, phoenix=self.phoenix, settings=self.settings, **options)

  def test_each_model_runs_on_its_foundry_deployment_at_the_providers_defaults(self) -> None:
    for configuration_id, model in (
      ('wixqa/deepagents/gpt', 'gpt-6.1-sol'),
      ('erb/deepagents/deepseek', 'DeepSeek-V4.1-Flash'),
    ):
      with self.subTest(model=model):
        self.init_chat_model.reset_mock()
        caller = self._build(configuration_id)

        # Only the connection, the client's patience, and the window deepagents summarizes on.
        self.init_chat_model.assert_called_once_with(
          f'azure_openai:{model}',
          azure_endpoint='https://resource.services.ai.azure.com',
          api_key='foundry-key',
          api_version='2025-04-01-preview',
          timeout=600.0,
          max_retries=5,
          profile={'max_input_tokens': 272000},
        )
        self.assertEqual(json.loads(caller._metadata)['model'], f'azure_openai:{model}')
        self.assertEqual(caller._search_url, 'http://127.0.0.1:8101/mcp')
    self.assertEqual(self.instrumentor.return_value.instrument.call_count, 2)

  def test_the_requests_carry_no_model_parameter_and_gpts_calls_with_tools_go_to_the_responses_api(self) -> None:
    search = convert_to_openai_tool(think_tool)
    for model in ('gpt', 'deepseek'):
      with self.subTest(model=model), patch('agents.deep_agent.init_chat_model', init_chat_model):
        configuration = self.config.configuration(f'wixqa/deepagents/{model}')
        _, client = _model(configuration, self.config, self.settings, self.foundry)
        payload = client._get_request_payload([HumanMessage(content=QUESTION)], tools=[search])  # type: ignore[attr-defined]

        parameters = set(payload) - {'model', 'stream', 'input', 'messages', 'tools'}
        self.assertEqual(parameters, set())
        self.assertEqual(client._use_responses_api(payload), model == 'gpt')  # type: ignore[attr-defined]

  def test_a_run_needs_the_foundry_deployment_and_a_phoenix_project(self) -> None:
    with self.assertRaisesRegex(ValueError, 'AZURE_FOUNDRY_BASE_URL and AZURE_FOUNDRY_API_KEY must be set'):
      self._build('wixqa/deepagents/gpt', foundry=AzureFoundrySettings(base_url=None))
    self.phoenix = PhoenixSettings(project_id=None)
    with self.assertRaisesRegex(ValueError, 'needs a Phoenix project'):
      self._build('wixqa/deepagents/gpt')
    self.init_chat_model.assert_not_called()


class ScriptedModel(BaseChatModel):
  """Plays the example's agents, which it tells apart by their system prompts.

  The orchestrator plans, searches once itself, delegates each topic of ``research`` to a research sub-agent, then
  answers. A research sub-agent makes its topic's calls, reads back a search result moved to its files, reflects, and
  reports its findings. ``received`` and ``orchestrator_received`` collect the search results that the sub-agents' and
  the orchestrator's calls receive.
  """

  model: str = 'scripted-model'
  research: dict[str, list[tuple[str, dict[str, Any]]]] = Field(default_factory=dict)
  received: list[str] = Field(default_factory=list)
  orchestrator_received: list[str] = Field(default_factory=list)

  @property
  def _llm_type(self) -> str:
    return 'scripted'

  def bind_tools(self, tools: Sequence[Any], **kwargs: Any) -> Any:
    return self.bind(tools=[convert_to_openai_tool(tool) for tool in tools], **kwargs)

  def _generate(
    self, messages: list[BaseMessage], stop: Any = None, run_manager: Any = None, **kwargs: Any
  ) -> ChatResult:
    system = next((message.text for message in messages if isinstance(message, SystemMessage)), '')
    results = [message for message in messages if isinstance(message, ToolMessage)]
    if '# Research Workflow' in system:
      calls = self._orchestrator(results)
    elif 'You are a research assistant' in system:
      calls = self._researcher(messages, results)
    else:
      raise AssertionError(f'an unexpected agent: {system[:200]}')
    if not calls:
      text = 'The report.' if '# Research Workflow' in system else 'Findings.'
      return ChatResult(generations=[ChatGeneration(message=AIMessage(content=text))])
    tool_calls = [
      {'name': name, 'args': args, 'id': f'{name}-{uuid4().hex[:8]}', 'type': 'tool_call'} for name, args in calls
    ]
    return ChatResult(generations=[ChatGeneration(message=AIMessage(content='', tool_calls=tool_calls))])

  def _orchestrator(self, results: list[ToolMessage]) -> list[tuple[str, dict[str, Any]]]:
    if results:
      self.orchestrator_received.extend(result.text for result in results if result.tool_call_id.startswith(TOOL_NAME))
      return []
    todos = [{'content': topic, 'status': 'in_progress'} for topic in self.research]
    tasks = [('task', {'description': topic, 'subagent_type': 'research-agent'}) for topic in self.research]
    return [('write_todos', {'todos': todos}), (TOOL_NAME, {'query': 'refund policy'}), *tasks]

  def _researcher(self, messages: list[BaseMessage], results: list[ToolMessage]) -> list[tuple[str, dict[str, Any]]]:
    if not results:
      topic = next(message.text for message in messages if isinstance(message, HumanMessage))
      return self.research[topic]
    called = {result.tool_call_id.split('-')[0] for result in results}
    searches = [result.text for result in results if result.tool_call_id.startswith(TOOL_NAME)]
    # The first call after the searches.
    if not called & {'read_file', 'think_tool'}:
      self.received.extend(searches)
    moved = [re.search(r'at this path: (\S+)', text) for text in searches]
    paths = [match.group(1) for match in moved if match]
    if paths and 'read_file' not in called:
      return [('read_file', {'file_path': paths[0], 'offset': 0, 'limit': 100})]
    if 'think_tool' not in called:
      return [('think_tool', {'reflection': 'Enough is known.'})]
    return []


class DeepAgentGraphTest(SearchServerTestCase):
  """The example's agent, with a scripted model, searching through the real search server."""

  def _caller(
    self, model: ScriptedModel, eviction_tokens: int | None = None
  ) -> tuple[DeepAgentCaller, InMemorySpanExporter]:
    exporter = InMemorySpanExporter()
    provider = TracerProvider(span_limits=SpanLimits(max_span_attributes=SpanLimits.UNSET))
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    LangChainInstrumentor().instrument(tracer_provider=provider)
    self.addCleanup(LangChainInstrumentor().uninstrument)
    settings = load_config().deepagents
    if eviction_tokens is not None:
      settings = settings.model_copy(update={'tool_result_eviction_tokens': eviction_tokens})
    caller = DeepAgentCaller(
      build=lambda tools: build_agent(model, tools, settings, DATE),
      search_url=self.url + PATH,
      tracer=provider.get_tracer('test'),
      flush=lambda trace_id: True,
      metadata={},
      max_attempts=1,
    )
    return caller, exporter

  def _normalize(self, exporter: InMemorySpanExporter, request_id: str) -> list[Span]:
    spans = exporter.get_finished_spans()
    [root] = [span for span in spans if span.parent is None]
    self.assertEqual((root.attributes or {})['request_id'], request_id)
    assert root.context is not None
    trace_id = format(root.context.trace_id, '032x')
    return DeepAgentTraceAdapter('scripted-model').normalize(trace_id, phoenix_records(spans)).spans

  def _assert_whole(self, search: Span) -> None:
    [retrieval] = search.semantics.retrieval
    self.assertEqual(search.semantics.request, QUESTION)
    self.assertEqual([item.id for item in retrieval.items], [document['id'] for document in DOCUMENTS])
    self.assertEqual([item.title for item in retrieval.items], [document['title'] for document in DOCUMENTS])
    self.assertEqual([item.content for item in retrieval.items], [document['text'] for document in DOCUMENTS])

  async def test_every_search_of_the_orchestrator_and_its_sub_agents_reaches_the_trace_whole_and_in_rank_order(
    self,
  ) -> None:
    # An empty query, which the tool rejects, and an argument it does not take, which the MCP server rejects.
    calls = [(TOOL_NAME, {'query': 'refund 1'}), (TOOL_NAME, {'query': ' '}), (TOOL_NAME, {'q': 'refunds'})]
    model = ScriptedModel(research={'Refund times': calls})
    caller, exporter = self._caller(model)

    spans = self._normalize(exporter, await caller.call(_sample()))

    by_id = {span.external_id: span for span in spans}
    searches = [span for span in spans if span.span_type == 'retrieval']
    self.assertEqual(
      sorted(search.semantics.retrieval[0].query or '' for search in searches), ['refund 1', 'refund policy']
    )
    for search in searches:
      self._assert_whole(search)
    # The sub-agent's search is under the task tool's span.
    [task] = [span for span in spans if span.name == 'task']
    [delegated] = [search for search in searches if search.semantics.retrieval[0].query == 'refund 1']
    ancestors = []
    parent = delegated.parent_span_id
    while parent is not None:
      ancestors.append(parent)
      parent = by_id[parent].parent_span_id
    self.assertIn(task.external_id, ancestors)
    # The server's errors go back to the agent, as its failed calls.
    rejected = [json.dumps(span.output_data) for span in spans if span.name == TOOL_NAME and span.span_type == 'tool']
    self.assertEqual(len(rejected), 2)
    self.assertTrue(any('The search failed: the query is empty' in output for output in rejected))
    self.assertTrue(any('Input validation error' in output for output in rejected))
    [root] = [span for span in spans if span.parent_span_id is None]
    assert root.semantics.answer is not None
    self.assertEqual(root.semantics.answer.text, 'The report.')
    self.assertEqual([evidence.availability for evidence in root.semantics.retrieval], ['available'])
    # About 63,000 tokens: above deepagents' default of 20,000 but below the configured limit, which both the
    # orchestrator and the research sub-agent keep.
    results = [json.loads(text) for text in (*model.received, *model.orchestrator_received) if text.startswith('{')]
    self.assertEqual(len(results), 2)
    for result in results:
      self.assertEqual([document['id'] for document in result['results']], [d['id'] for d in DOCUMENTS])

  async def test_a_search_that_gets_no_answer_runs_the_question_again(self) -> None:
    caller, exporter = self._caller(ScriptedModel(research={'Refund times': [(TOOL_NAME, {'query': 'refund 1'})]}))
    # Built with the server's tool, which then goes down.
    await caller._built()
    await self.stop_search_server()

    with self.assertRaisesRegex(DeepAgentRunError, 'in all 1 attempts; the last: a search got no answer'):
      await caller.call(_sample())

  async def test_a_search_result_moved_to_the_agents_files_stays_whole_in_its_span(self) -> None:
    model = ScriptedModel(research={'Refund times': [(TOOL_NAME, {'query': 'refund 1'})]})
    # Ten documents are about 63,000 tokens at 4 characters a token.
    caller, exporter = self._caller(model, eviction_tokens=10_000)

    with self.assertLogs('agents.deep_agent', 'WARNING') as logs:
      request_id = await caller.call(_sample())
    spans = self._normalize(exporter, request_id)

    # The sub-agent received the head and tail of the result, with the path of the whole.
    [received] = model.received
    self.assertIn('Tool result too large', received)
    self.assertLess(len(received), 10_000)
    self.assertIn('moved 2 tool results to its files', logs.output[0])
    searches = {span.semantics.retrieval[0].query: span for span in spans if span.span_type == 'retrieval'}
    self.assertEqual(set(searches), {'refund 1', 'refund policy'})
    for search in searches.values():
      self._assert_whole(search)
    [read] = [span for span in spans if span.name == 'read_file']
    self.assertEqual(read.span_type, 'tool')


if __name__ == '__main__':
  unittest.main()
