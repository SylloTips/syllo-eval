import unittest
from unittest.mock import AsyncMock
from uuid import uuid4

from syllo_eval.execution.agent_caller import AgentCallDispatcher
from syllo_eval.model import Agent, Sample


class TestAgentCallDispatcher(unittest.IsolatedAsyncioTestCase):
  async def test_dispatches_by_normalized_agent_name(self) -> None:
    sample = Sample(id=uuid4(), dataset_id=uuid4(), input_prompt='hello')
    agent = Agent(id=uuid4(), name='Demo-Agent', version_tag='v-test')
    caller = _AsyncCaller('request-123')
    dispatcher = AgentCallDispatcher(callers_by_agent_name={'demo-agent': caller})

    request_id = await dispatcher.call(agent=agent, sample=sample)

    self.assertEqual(request_id, 'request-123')
    caller.call.assert_awaited_once_with(sample)

  async def test_raises_for_unregistered_agent(self) -> None:
    sample = Sample(id=uuid4(), dataset_id=uuid4(), input_prompt='hello')
    agent = Agent(id=uuid4(), name='Unknown', version_tag='v-test')
    dispatcher = AgentCallDispatcher(callers_by_agent_name={})

    with self.assertRaisesRegex(ValueError, 'No agent caller registered'):
      await dispatcher.call(agent=agent, sample=sample)


class _AsyncCaller:
  def __init__(self, request_id: str):
    self.call = AsyncMock(return_value=request_id)
