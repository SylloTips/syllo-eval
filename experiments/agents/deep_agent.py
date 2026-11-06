"""LangChain's deep agent, run in this process, and the adapter of its traces.

The agent is the deep_research example of deepagents (``agents/deep_research``), wired as the example's ``agent.py``
does, with the settings of ``configs/deepagents.yaml``. An orchestrator plans with ``write_todos`` and delegates the
research to sub-agents through deepagents' ``task`` tool; it and its sub-agents have the search tool over MCP, in place
of the example's web search, and the example's ``think_tool``. Every call, the sub-agents' and the summaries' included,
uses the configuration's model. The example's prompts are edited for documents with ids instead of web pages with URLs,
and for an answer in the orchestrator's last message instead of a report file.

Long contexts are deepagents' own: it summarizes the older history once a call's input nears the window of
``configs/deepagents.yaml``, and moves a tool result above its size to the agent's files, where the agent reads it with
the file tools. A search's span holds its whole result either way: the result is moved once the search has returned.

Each call runs the agent under a root span with the request id, the question and the answer, and OpenInference's
LangChain instrumentation traces it to Phoenix. Spans are flushed before the call returns. A question runs again, with a
new request id, when a search got no answer from the search tool, when a provider's rate limit, outage or connection
error outlasted the client's retries, when a model call was cut off, when the run ended without an answer, when spans
did not reach Phoenix, or when the run took too long. deepagents stops at a model's refusal: a sub-agent's reaches the
orchestrator as a notice, and the orchestrator's fails the sample.
"""

import asyncio
import json
import logging
from collections.abc import Callable, Mapping, Sequence
from datetime import datetime
from typing import Any
from uuid import UUID

from deepagents import FilesystemMiddleware, SubAgent, create_deep_agent
from deepagents.backends import StateBackend
from langchain.agents.middleware import AgentMiddleware, TodoListMiddleware
from langchain.chat_models import init_chat_model
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage
from langchain_core.outputs import LLMResult
from langchain_core.tools import BaseTool
from opentelemetry import trace
from opentelemetry.context import Context
from pydantic import Field
from pydantic_settings import SettingsConfigDict

from syllo_eval.model import Sample
from syllo_eval.settings import EnvSettings, PhoenixSettings

from agents.deep_research.prompts import (
  RESEARCH_WORKFLOW_INSTRUCTIONS,
  RESEARCHER_INSTRUCTIONS,
  SUBAGENT_DELEGATION_INSTRUCTIONS,
)
from agents.deep_research.tools import think_tool
from agents.langchain_runs import (
  LangChainTraceAdapter,
  TransientError,
  call_with_retries,
  cancel_leftovers,
  is_transient,
  mcp_tools,
  trace_to_phoenix,
)
from config import Configuration, DeepAgentsConfig, ExperimentConfig
from indexing.embedding import AzureFoundrySettings
from search_tool.server import TOOL_NAME

ROOT_SPAN_NAME = 'deepagents.request'
# The example's research sub-agent, as its agent.py describes it to the orchestrator.
RESEARCH_AGENT = 'research-agent'
RESEARCH_AGENT_DESCRIPTION = (
  'Delegate research to the sub-agent researcher. Only give this researcher one topic at a time.'
)
# What LangChain's Azure client raises for a Chat Completions response that a content filter stopped.
_AZURE_CONTENT_FILTER = 'Azure has not provided the response due to a content filter being triggered'
# Where deepagents writes a tool result that it moves to the agent's files, and the report file of the example.
EVICTED_PREFIX = '/large_tool_results/'
REPORT_FILE = '/final_report.md'

# The edits of the example's prompts, each an exact replacement of a passage that must occur once.
_WORKFLOW_EDITS = (
  # A benchmark question is researched as asked, as ODR's are: no one answers a question back.
  (
    'Follow this workflow for all research requests:',
    'Research every request as asked: do not ask clarifying questions, which no one would answer. Follow this '
    'workflow for all research requests:',
  ),
  # Documents have ids, not URLs.
  (
    '(each unique URL gets one number across all findings)',
    '(each unique document gets one number across all findings)',
  ),
  # The answer is the last message, so the request is checked before the report is written, not after.
  (
    '5. **Write Report**: Write a comprehensive final report to `/final_report.md` (see Report Writing Guidelines '
    "below)\n6. **Verify**: Read `/research_request.md` and confirm you've addressed all aspects with proper citations "
    'and structure',
    '5. **Verify**: Read `/research_request.md` and confirm that the findings cover all aspects of the request\n'
    '6. **Write Report**: Write a comprehensive final report, with proper citations and structure, as your final '
    'message (see Report Writing Guidelines below). Do not write it to a file: your final message is the answer',
  ),
  ('When writing the final report to `/final_report.md`, follow', 'When writing the final report, follow'),
  ('- Assign each unique URL a single citation number', '- Assign each unique document a single citation number'),
  ('- Format: [1] Source Title: URL (each', '- Format: [1] Document Title: document id (each'),
  (
    '[1] AI Research Paper: https://example.com/paper\n  [2] Industry Analysis: https://example.com/analysis',
    '[1] AI Research Paper: doc-1042\n  [2] Industry Analysis: doc-0318',
  ),
)
_RESEARCHER_EDITS = (
  (
    '**tavily_search**: For conducting web searches to gather information',
    f'**{TOOL_NAME}**: For searching the knowledge base, which returns its 10 most relevant documents, best first, '
    'each with its id, title and text',
  ),
  ('listing each numbered source with title and URL', 'listing each numbered source with title and document id'),
  (
    '[1] Context Engineering Guide: https://example.com/context-guide\n'
    '[2] AI Performance Study: https://example.com/study',
    '[1] Context Engineering Guide: doc-1042\n[2] AI Performance Study: doc-0318',
  ),
)

logger = logging.getLogger(__name__)


class DeepAgentSettings(EnvSettings):
  """How the deep agent runs here; what it runs is in ``configs/deepagents.yaml``."""

  model_config = SettingsConfigDict(env_prefix='DEEPAGENTS_')

  # The search server's MCP endpoint.
  search_url: str = Field(default='http://127.0.0.1:8101/mcp', min_length=1)
  # Attempts per sample at a transient failure, each a new run with a new request id.
  max_attempts: int = Field(default=3, ge=1)
  retry_delay_seconds: float = Field(default=60.0, ge=0)
  timeout_seconds: float = Field(default=3600.0, gt=0)
  # Each model call: how long the client waits for its response, and how many times it retries a rate limit, an outage
  # or a connection error, after waiting as long as the provider asks.
  request_timeout_seconds: float = Field(default=600.0, gt=0)
  max_retries: int = Field(default=5, ge=0)
  # GPT and DeepSeek are Foundry deployments at AZURE_FOUNDRY_BASE_URL, called through its Azure OpenAI API.
  azure_api_version: str = Field(default='2025-04-01-preview', min_length=1)


class DeepAgentRunError(RuntimeError):
  """A question the deep agent could not answer in any attempt: the sample fails."""


class RunWatch(BaseCallbackHandler):
  """Watches one run, through the LangChain callbacks of the orchestrator and its sub-agents, for failures it hides.

  - A search got no answer from the search tool, which the agent would take as a failed call. The tool returns the
    server's own errors, such as a rejected empty query, to the agent: those are the agent's failed calls.
  - A model call was cut off, at its output limit or by a content filter. GPT's Responses API then returns no output at
    all, which the agent would take as its answer, or as a sub-agent's findings.
  """

  # Quick bookkeeping, so the callbacks run in the event loop, in order.
  run_inline = True

  def __init__(self) -> None:
    self.problems: list[str] = []
    self._searches: set[UUID] = set()

  def on_tool_start(self, serialized: dict[str, Any], input_str: str, *, run_id: UUID, **kwargs: Any) -> None:
    if serialized.get('name') == TOOL_NAME:
      self._searches.add(run_id)

  def on_tool_end(self, output: Any, *, run_id: UUID, **kwargs: Any) -> None:
    self._searches.discard(run_id)

  def on_tool_error(self, error: BaseException, *, run_id: UUID, **kwargs: Any) -> None:
    # A search cancelled with its run already failed for another reason.
    if run_id in self._searches and not isinstance(error, asyncio.CancelledError):
      self.problems.append('a search got no answer from the search tool')
    self._searches.discard(run_id)

  def on_llm_end(self, response: LLMResult, *, run_id: UUID, **kwargs: Any) -> None:
    for generations in response.generations:
      for generation in generations:
        reason = _cut_off(generation)
        if reason is not None:
          self.problems.append(f'a model call was cut off ({reason})')

  def on_llm_error(self, error: BaseException, *, run_id: UUID, **kwargs: Any) -> None:
    # LangChain's Azure client raises it for a Chat Completions response that a content filter stopped.
    if isinstance(error, ValueError) and str(error) == _AZURE_CONTENT_FILTER:
      self.problems.append('a model call was cut off (content_filter)')


def _cut_off(generation: Any) -> str | None:
  """Why a model call stopped before its end, if it did."""
  message = getattr(generation, 'message', None)
  metadata = {**(generation.generation_info or {}), **(getattr(message, 'response_metadata', None) or {})}
  # The Responses API.
  if metadata.get('status') == 'incomplete':
    return (metadata.get('incomplete_details') or {}).get('reason') or 'incomplete'
  # Chat Completions.
  if metadata.get('finish_reason') in ('length', 'content_filter'):
    return metadata['finish_reason']
  return None


class DeepAgentCaller:
  """Runs the deep agent on the question of a sample and returns the request id of the run that answered it.

  The agent is built at the first call, with the search tool as the server describes it.
  """

  def __init__(
    self,
    *,
    build: Callable[[Sequence[BaseTool]], Any],
    search_url: str,
    tracer: trace.Tracer,
    flush: Callable[[int], bool],
    metadata: Mapping[str, Any],
    max_attempts: int = 3,
    retry_delay_seconds: float = 60.0,
    timeout_seconds: float = 3600.0,
  ):
    self._build = build
    self._search_url = search_url
    self._tracer = tracer
    # Whether the spans of a trace reached Phoenix.
    self._flush = flush
    # Added to the root span.
    self._metadata = json.dumps(dict(metadata))
    self._max_attempts = max_attempts
    self._retry_delay_seconds = retry_delay_seconds
    self._timeout_seconds = timeout_seconds
    self._agent: Any = None
    self._building = asyncio.Lock()

  async def call(self, sample: Sample) -> str:
    return await call_with_retries(
      sample,
      lambda request_id: self._run(sample.input_prompt, request_id),
      agent='The deep agent',
      failure=DeepAgentRunError,
      max_attempts=self._max_attempts,
      retry_delay_seconds=self._retry_delay_seconds,
    )

  async def _built(self) -> Any:
    async with self._building:
      if self._agent is None:
        try:
          tools = await mcp_tools(self._search_url)
        except Exception as error:
          raise TransientError(f'the search tool is unreachable at {self._search_url}: {error!r}') from error
        searches = [tool for tool in tools if tool.name == TOOL_NAME]
        if not searches:
          raise TransientError(f'the server at {self._search_url} has no {TOOL_NAME} tool')
        [search] = searches
        # The server's own errors, such as an empty query, go back to the agent as the tool's result: unhandled, they
        # would end the run.
        search.handle_tool_error = True
        self._agent = self._build([search, think_tool])
      return self._agent

  async def _run(self, question: str, request_id: str) -> None:
    agent = await self._built()
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
        state = await self._invoke(agent, question, watch)
      except Exception as error:
        failure = error
      finally:
        problems = sorted(set(watch.problems))
        await cancel_leftovers(trace_id)
      # Whatever the run did after a search without an answer or a cut-off call, it did without them.
      if problems:
        raise TransientError('; '.join(problems)) from failure
      if failure is not None:
        raise failure
      answer = state['messages'][-1].text
      if not answer.strip():
        raise TransientError('the run ended without an answer')
      files = state.get('files') or {}
      if REPORT_FILE in files:
        logger.warning('Deep agent request %s wrote %s; its last message is the answer', request_id, REPORT_FILE)
      evicted = sorted(path for path in files if path.startswith(EVICTED_PREFIX))
      if evicted:
        logger.warning(
          'Deep agent request %s moved %d tool results to its files: %s', request_id, len(evicted), evicted
        )
      span.set_attribute('output.value', answer)
    if not await asyncio.to_thread(self._flush, trace_id):
      raise TransientError('spans of the run did not reach Phoenix')

  async def _invoke(self, agent: Any, question: str, watch: RunWatch) -> dict[str, Any]:
    try:
      async with asyncio.timeout(self._timeout_seconds):
        return await agent.ainvoke({'messages': [HumanMessage(content=question)]}, {'callbacks': [watch]})
    except TimeoutError as error:
      raise TransientError(f'the run took longer than {self._timeout_seconds:g}s') from error
    except Exception as error:
      if is_transient(error):
        raise TransientError(f'{type(error).__name__}: {error}') from error
      raise


def orchestrator_prompt(config: DeepAgentsConfig) -> str:
  """The example's orchestrator instructions, edited for the knowledge base and for an answer in the last message."""
  delegation = SUBAGENT_DELEGATION_INSTRUCTIONS.format(
    max_concurrent_research_units=config.max_concurrent_research_units,
    max_researcher_iterations=config.max_researcher_iterations,
  )
  return _edited(RESEARCH_WORKFLOW_INSTRUCTIONS, _WORKFLOW_EDITS) + '\n\n' + '=' * 80 + '\n\n' + delegation


def researcher_prompt(date: str) -> str:
  """The example's research sub-agent instructions, edited for the knowledge base."""
  return _edited(RESEARCHER_INSTRUCTIONS, _RESEARCHER_EDITS).format(date=date)


def _edited(prompt: str, edits: Sequence[tuple[str, str]]) -> str:
  for old, new in edits:
    if prompt.count(old) != 1:
      raise RuntimeError(f'The example prompt no longer has {old!r} once: review the edits of agents/deep_agent.py')
    prompt = prompt.replace(old, new)
  return prompt


def build_agent(model: BaseChatModel, tools: Sequence[BaseTool], config: DeepAgentsConfig, date: str) -> Any:
  """The example's agent: the orchestrator and its research sub-agent, with deepagents' defaults but ``config``'s.

  deepagents 0.7 no longer gives the orchestrator ``write_todos``, which the example's workflow starts with. Every
  agent, deepagents' general-purpose sub-agent included, moves a tool result above the configured size to its files.
  """
  backend = StateBackend()

  def files() -> AgentMiddleware[Any, Any, Any]:
    return FilesystemMiddleware(backend=backend, tool_token_limit_before_evict=config.tool_result_eviction_tokens)

  middleware: list[AgentMiddleware[Any, Any, Any]] = [TodoListMiddleware(), files()]

  research_agent: SubAgent = {
    'name': RESEARCH_AGENT,
    'description': RESEARCH_AGENT_DESCRIPTION,
    'system_prompt': researcher_prompt(date),
    'tools': list(tools),
    'middleware': [files()],
  }
  return create_deep_agent(
    model=model,
    tools=list(tools),
    system_prompt=orchestrator_prompt(config),
    middleware=middleware,
    subagents=[research_agent],
    backend=backend,
  )


def build_caller(
  configuration: Configuration,
  config: ExperimentConfig,
  *,
  phoenix: PhoenixSettings,
  settings: DeepAgentSettings | None = None,
  foundry: AzureFoundrySettings | None = None,
) -> DeepAgentCaller:
  """The caller of a deep agent configuration, whose spans go to ``phoenix``'s project.

  The instrumentation is process-wide: one collection per process, as `syllo-exp collect` runs.
  """
  if phoenix.project_id is None:
    raise ValueError(f'{configuration.id}: a deep agent run needs a Phoenix project')
  settings = settings or DeepAgentSettings()
  model_name, model = _model(configuration, config, settings, foundry)
  # The example gives the research sub-agent the date of the day it starts.
  date = datetime.now().strftime('%Y-%m-%d')
  tracer, flush = trace_to_phoenix(phoenix, __name__)
  return DeepAgentCaller(
    build=lambda tools: build_agent(model, tools, config.deepagents, date),
    search_url=settings.search_url,
    tracer=tracer,
    flush=flush,
    metadata={'configuration': configuration.id, 'model': model_name},
    max_attempts=settings.max_attempts,
    retry_delay_seconds=settings.retry_delay_seconds,
    timeout_seconds=settings.timeout_seconds,
  )


def _model(
  configuration: Configuration,
  config: ExperimentConfig,
  settings: DeepAgentSettings,
  foundry: AzureFoundrySettings | None,
) -> tuple[str, BaseChatModel]:
  """The configuration's model, as init_chat_model names it, and its client.

  Every model parameter is the provider's default, as for the other agents: GPT 6.1 Sol reasons at its default effort,
  and LangChain sends its calls with tools to the Responses API, the only one on which it takes them, as for ODR.
  """
  ref = config.models.agents[configuration.model]
  if ref.provider not in ('azure', 'azure_ai'):
    raise ValueError(f'{configuration.id}: the deep agent has no route for {ref.provider} models')
  foundry = foundry or AzureFoundrySettings()
  if not (foundry.base_url and foundry.api_key):
    raise ValueError(f'{configuration.id}: AZURE_FOUNDRY_BASE_URL and AZURE_FOUNDRY_API_KEY must be set')
  name = f'azure_openai:{ref.model}'
  model = init_chat_model(
    name,
    azure_endpoint=foundry.base_url,
    api_key=foundry.api_key,
    api_version=settings.azure_api_version,
    timeout=settings.request_timeout_seconds,
    max_retries=settings.max_retries,
    # deepagents summarizes once a call's input nears the window it reads from the model's profile.
    profile={'max_input_tokens': config.deepagents.context_window_tokens},
  )
  return name, model


class DeepAgentTraceAdapter(LangChainTraceAdapter):
  """The canonical trace of one deep agent run, whose every call must be to ``model``; see ``LangChainTraceAdapter``.

  Searches run in the orchestrator and in its sub-agents, whose spans are under the ``task`` tool's. The agent's file
  tools, which may read again a search result moved to its files, stay tool spans: the search's span holds the result.
  """

  def __init__(self, model: str):
    super().__init__(model, root_span_name=ROOT_SPAN_NAME, version='deepagents:1')
