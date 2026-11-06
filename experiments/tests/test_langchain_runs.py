import asyncio
import unittest
from typing import Any
from uuid import UUID

from langchain_core.callbacks import AsyncCallbackHandler, AsyncCallbackManager
from opentelemetry.context import Context
from opentelemetry.sdk.trace import TracerProvider

from agents.langchain_runs import cancel_leftovers


class CancelLeftoversTest(unittest.IsolatedAsyncioTestCase):
  async def test_a_leftover_is_cancelled_but_the_callbacks_of_a_run_that_is_ending_finish(self) -> None:
    release = asyncio.Event()
    ended: list[UUID] = []

    class SlowHandler(AsyncCallbackHandler):
      async def on_chain_end(self, outputs: Any, *, run_id: UUID, **kwargs: Any) -> None:
        await release.wait()
        ended.append(run_id)

    tracer = TracerProvider().get_tracer('test')
    with tracer.start_as_current_span('root', context=Context()) as root:
      run = await AsyncCallbackManager(handlers=[SlowHandler()]).on_chain_start({'name': 'leftover'}, {})
      leftover = asyncio.create_task(asyncio.sleep(3600))
      # LangChain ends the run in tasks of their own, shielded from the run's cancellation: one per handler.
      ending = asyncio.create_task(run.on_chain_end({}))
      await asyncio.sleep(0.01)
      asyncio.get_running_loop().call_later(0.05, release.set)

      await cancel_leftovers(root.get_span_context().trace_id)

    self.assertTrue(leftover.cancelled())
    self.assertTrue(ending.done())
    self.assertEqual(ended, [run.run_id])


if __name__ == '__main__':
  unittest.main()
