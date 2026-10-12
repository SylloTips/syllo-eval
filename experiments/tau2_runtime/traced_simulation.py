"""OpenInference spans for one tau2 half-duplex simulation. Import only after ``telemetry.configure_tracing``.

One simulation is one trace:

  tau2.simulation          CHAIN      root: request_id, the user scenario in, the reward out
  |- <agent name>          AGENT      the agent: the first customer message in, the last agent message out
  |  |- agent_turn         CHAIN      a customer message or tool results in, the agent's message out
  |  |  |- completion      LLM        from the litellm instrumentor: messages, tools, tool calls, tokens, cost
  |  |  |- <tool name>     TOOL       one per tool call the environment executes: arguments in, result out
  |- user_turn             CHAIN      the customer simulator, outside the agent
  |  |- completion         LLM
  |- evaluate_simulation   EVALUATOR  tau2's reward, outside the agent; the assertion grader's LLM spans nest here

The orchestrator is tau2's own, subclassed only to open spans around its steps: the conversation is tau2's. The one
change to tau2's run is how its assertion grader reads a reply that is not JSON (``configure_grader``).
"""

import json
from collections.abc import Callable, Mapping
from contextlib import nullcontext
from typing import Any, Literal

import litellm
import tau2.evaluator.evaluator_nl_assertions as nl_assertions
import tau2.utils.llm_utils as tau2_llm_utils
from openinference.instrumentation import (
  get_input_attributes,
  get_metadata_attributes,
  get_output_attributes,
  get_span_kind_attributes,
  get_tool_attributes,
)
from openinference.semconv.trace import SpanAttributes
from opentelemetry import context as otel_context
from opentelemetry import trace
from opentelemetry.trace import Span, Status, StatusCode
from tau2.data_model.message import AssistantMessage, Message, MultiToolMessage, ToolCall, ToolMessage, UserMessage
from tau2.data_model.simulation import SimulationRun
from tau2.environment.toolkit import get_tool_types
from tau2.evaluator.evaluator import EvaluationType, evaluate_simulation
from tau2.orchestrator.modes import CommunicationMode
from tau2.orchestrator.orchestrator import Orchestrator, Role

ROOT_SPAN_NAME = 'tau2.simulation'

if not getattr(litellm.completion, 'is_wrapper', False) or tau2_llm_utils.completion is not litellm.completion:
  raise RuntimeError('litellm must be instrumented before tau2 is imported: tau2 bound litellm.completion unwrapped')


def configure_grader(model: str, llm_args: Mapping[str, Any]) -> None:
  """Grade the natural-language assertions with ``model``, reading a reply in a Markdown fence as the JSON inside it.

  tau2's grader reads its model from module names when it runs, and parses the reply with ``json.loads``, so a fenced
  reply would fail the simulation. A reply that is not JSON is replaced by the JSON that tau2's own
  ``extract_json_from_llm_response`` finds in it, as tau2's other LLM judges read their replies. A JSON reply stays as
  it is, so every reply that tau2 parses gets the reward it would get from tau2.
  """
  nl_assertions.DEFAULT_LLM_NL_ASSERTIONS = model
  nl_assertions.DEFAULT_LLM_NL_ASSERTIONS_ARGS = dict(llm_args)
  nl_assertions.generate = _with_json_content(tau2_llm_utils.generate)


def json_content(content: str | None) -> str | None:
  """``content`` if it is JSON, otherwise the JSON that tau2 extracts from it."""
  if content is None:
    return None
  try:
    json.loads(content)
  except ValueError:
    return tau2_llm_utils.extract_json_from_llm_response(content)
  return content


def _with_json_content(generate: Callable[..., AssistantMessage]) -> Callable[..., AssistantMessage]:
  def generate_json(*args: Any, **kwargs: Any) -> AssistantMessage:
    message = generate(*args, **kwargs)
    return message.model_copy(update={'content': json_content(message.content)})

  return generate_json


class TracedOrchestrator(Orchestrator):
  """tau2's half-duplex orchestrator with a span around each step."""

  def __init__(self, *args: Any, tracer: trace.Tracer, agent_name: str, **kwargs: Any):
    super().__init__(*args, **kwargs)
    self._tracer = tracer
    self._agent_name = agent_name
    toolkits = [toolkit for toolkit in (self.environment.tools, self.environment.user_tools) if toolkit is not None]
    self._tool_types = {name: kind.value for toolkit in toolkits for name, kind in get_tool_types(toolkit).items()}
    self._tool_schemas = {
      name: tool.openai_schema['function'] for toolkit in toolkits for name, tool in toolkit.get_tools().items()
    }
    self._simulation_context: otel_context.Context | None = None
    self._agent_context: otel_context.Context | None = None
    self._open_turn: Span | None = None

  @classmethod
  def wrap(cls, built: Orchestrator, *, tracer: trace.Tracer, agent_name: str) -> 'TracedOrchestrator':
    """The parts and settings of an orchestrator made by tau2's ``build_text_orchestrator``."""
    return cls(
      domain=built.domain,
      agent=built.agent,
      user=built.user,
      environment=built.environment,
      task=built.task,
      max_steps=built.max_steps,
      max_errors=built.max_errors,
      seed=built.seed,
      solo_mode=built.solo_mode,
      simulation_id=built.simulation_id,
      validate_communication=built.validate_communication,
      timeout=built.timeout,
      tracer=tracer,
      agent_name=agent_name,
    )

  def run(self) -> SimulationRun:
    self._simulation_context = otel_context.get_current()
    agent_span = self._tracer.start_span(
      self._agent_name,
      context=self._simulation_context,
      attributes={**get_span_kind_attributes('agent'), SpanAttributes.AGENT_NAME: self._agent_name},
    )
    self._agent_context = trace.set_span_in_context(agent_span, self._simulation_context)
    try:
      result = super().run()
    except BaseException as error:
      self._close_turn(error)
      _fail(agent_span, error)
      agent_span.end()
      raise
    self._close_turn(None)
    texts = [m for m in result.messages if isinstance(m, (UserMessage, AssistantMessage)) and m.has_text_content()]
    first_customer = next((m for m in texts if isinstance(m, UserMessage)), None)
    last_agent = next((m for m in reversed(texts) if isinstance(m, AssistantMessage)), None)
    agent_span.set_attributes({**_io_attributes(first_customer, 'input'), **_io_attributes(last_agent, 'output')})
    agent_span.set_status(Status(StatusCode.OK))
    agent_span.end()
    return result

  def step(self) -> None:
    if self.to_role == Role.ENV:
      # The turn that requested the tool calls is still open, so their spans become its children.
      with trace.use_span(self._open_turn, end_on_exit=False) if self._open_turn else nullcontext():
        super().step()
      self._close_turn(None)
      return
    is_agent = self.to_role == Role.AGENT
    span = self._tracer.start_span(
      'agent_turn' if is_agent else 'user_turn',
      context=self._agent_context if is_agent else self._simulation_context,
      attributes={
        **get_span_kind_attributes('chain'),
        **_io_attributes(self.message, 'input'),
        **get_metadata_attributes(metadata={'step': self.step_count}),
      },
    )
    try:
      with trace.use_span(span, end_on_exit=False, record_exception=False, set_status_on_exception=False):
        super().step()
    except BaseException as error:
      _fail(span, error)
      span.end()
      raise
    span.set_attributes(_io_attributes(self.message, 'output'))
    span.set_status(Status(StatusCode.OK))
    if self.to_role == Role.ENV:
      self._open_turn = span
    else:
      span.end()

  def _close_turn(self, error: BaseException | None) -> None:
    if self._open_turn is None:
      return
    if error is not None:
      _fail(self._open_turn, error)
    self._open_turn.end()
    self._open_turn = None

  def _execute_tool_calls(self, tool_calls: list[ToolCall]) -> list[ToolMessage]:
    results = []
    for call in tool_calls:
      attributes = {
        **get_span_kind_attributes('tool'),
        **get_input_attributes(call.arguments, mime_type='application/json'),
        SpanAttributes.TOOL_NAME: call.name,
        SpanAttributes.TOOL_ID: call.id,
        **get_metadata_attributes(metadata={'requestor': call.requestor, 'tool_type': self._tool_types.get(call.name)}),
      }
      if (schema := self._tool_schemas.get(call.name)) is not None:
        attributes.update(
          get_tool_attributes(name=call.name, description=schema.get('description'), parameters=schema['parameters'])
        )
      with self._tracer.start_as_current_span(call.name, attributes=attributes) as span:
        # One call at a time through the parent, which counts the errors that end a simulation.
        [result] = super()._execute_tool_calls([call])
        span.set_attributes(get_output_attributes(result.content, mime_type=_mime_type(result.content)))
        span.set_status(Status(StatusCode.ERROR, result.content) if result.error else Status(StatusCode.OK))
      results.append(result)
    return results


def run_traced_simulation(
  orchestrator: TracedOrchestrator, *, tracer: trace.Tracer, request_id: str, metadata: dict[str, Any]
) -> SimulationRun:
  """tau2's ``run_simulation``, its conversation then its evaluation, inside the root and evaluation spans."""
  task = orchestrator.task
  root_attributes = {
    **get_span_kind_attributes('chain'),
    'request_id': request_id,
    **get_input_attributes(str(task.user_scenario), mime_type='text/plain'),
    **get_metadata_attributes(
      metadata={
        'task_id': task.id,
        'simulation_id': orchestrator.simulation_id,
        'domain': orchestrator.domain,
        'seed': orchestrator.seed,
        **metadata,
      }
    ),
  }
  # An empty parent context makes this a root span, which the request_id lookup requires.
  with tracer.start_as_current_span(ROOT_SPAN_NAME, context=otel_context.Context(), attributes=root_attributes) as root:
    simulation = orchestrator.run()
    simulation.policy = orchestrator.environment.get_policy()
    evaluation_type = EvaluationType.ALL
    evaluation_input = {
      'task_id': task.id,
      'evaluation_type': evaluation_type.value,
      'termination_reason': simulation.termination_reason,
    }
    with tracer.start_as_current_span(
      'evaluate_simulation',
      attributes={
        **get_span_kind_attributes('evaluator'),
        **get_input_attributes(evaluation_input, mime_type='application/json'),
      },
    ) as evaluation:
      reward_info = evaluate_simulation(
        simulation=simulation,
        task=task,
        evaluation_type=evaluation_type,
        solo_mode=orchestrator.solo_mode,
        domain=orchestrator.environment.get_domain_name(),
        mode=CommunicationMode.HALF_DUPLEX,
      )
      evaluation.set_attributes(
        get_output_attributes(reward_info.model_dump(mode='json'), mime_type='application/json')
      )
      evaluation.set_status(Status(StatusCode.OK))
    simulation.reward_info = reward_info
    outcome = {'termination_reason': simulation.termination_reason, 'reward': reward_info.reward}
    root.set_attributes(get_output_attributes(outcome, mime_type='application/json'))
    root.set_status(Status(StatusCode.OK))
  return simulation


def _fail(span: Span, error: BaseException) -> None:
  span.record_exception(error)
  span.set_status(Status(StatusCode.ERROR, f'{type(error).__name__}: {error}'))


def _io_attributes(message: Message | None, direction: str) -> dict[str, Any]:
  """Plain text for text messages, JSON for tool calls and tool results."""
  if message is None:
    return {}
  attributes = get_input_attributes if direction == 'input' else get_output_attributes
  if isinstance(message, (UserMessage, AssistantMessage)) and not message.tool_calls:
    return attributes(message.content or '', mime_type='text/plain')
  return attributes(_message_payload(message), mime_type='application/json')


def _message_payload(message: Message) -> Any:
  if isinstance(message, MultiToolMessage):
    return [_message_payload(tool_message) for tool_message in message.tool_messages]
  if isinstance(message, ToolMessage):
    return {'role': 'tool', 'tool_call_id': message.id, 'content': message.content, 'error': message.error}
  payload: dict[str, Any] = {'role': message.role, 'content': message.content}
  if message.tool_calls:
    payload['tool_calls'] = [
      {'id': call.id, 'name': call.name, 'arguments': call.arguments} for call in message.tool_calls
    ]
  return payload


def _mime_type(content: str | None) -> Literal['application/json', 'text/plain']:
  try:
    json.loads(content or '')
  except ValueError:
    return 'text/plain'
  return 'application/json'
