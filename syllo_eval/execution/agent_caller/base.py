from typing import Protocol

from syllo_eval.model import Agent, Sample


class AgentCaller(Protocol):
  async def call(self, sample: Sample) -> str:
    """Execute the agent for a sample and return the request ID."""


class AgentCallDispatcher:
  """Routes a sample execution to the appropriate agent caller."""

  def __init__(self, callers_by_agent_name: dict[str, AgentCaller]):
    self._callers_by_agent_name = {
      self._normalize_agent_name(agent_name): caller for agent_name, caller in callers_by_agent_name.items()
    }

  async def call(self, agent: Agent, sample: Sample) -> str:
    caller = self._callers_by_agent_name.get(self._normalize_agent_name(agent.name))
    if caller is None:
      supported_agents = ', '.join(sorted(self._callers_by_agent_name)) or '<none>'
      raise ValueError(
        f'No agent caller registered for agent name="{agent.name}" version_tag="{agent.version_tag}". '
        f'Supported agents: {supported_agents}'
      )
    return await caller.call(sample)

  @staticmethod
  def _normalize_agent_name(agent_name: str) -> str:
    return agent_name.strip().lower()
