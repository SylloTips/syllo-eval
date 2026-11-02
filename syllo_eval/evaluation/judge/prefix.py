import asyncio
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import Any


@dataclass
class SharedPromptPrefix:
  system_prompt: str
  user_prefix: str
  requests: int
  cache_lock: asyncio.Lock = field(default_factory=asyncio.Lock)
  cache_attempted: bool = False
  cache_name: str | None = None
  cache_creation_input_tokens: int | None = None
  cache_seconds: float = 0.0
  cache_close: Callable[[], Awaitable[None]] | None = None
  users: set[asyncio.Task[Any]] = field(default_factory=set)


_active_prefix: ContextVar[SharedPromptPrefix | None] = ContextVar('shared_prompt_prefix', default=None)


def shared_prefix_of(system_prompt: str, user_prompt: str) -> SharedPromptPrefix | None:
  prefix = _active_prefix.get()
  if prefix is None or system_prompt != prefix.system_prompt or not user_prompt.startswith(prefix.user_prefix):
    return None
  return prefix


@asynccontextmanager
async def shared_prompt_prefix(
  system_prompt: str, user_prefix: str, requests: int
) -> AsyncIterator[SharedPromptPrefix]:
  prefix = SharedPromptPrefix(system_prompt, user_prefix, requests)
  token = _active_prefix.set(prefix)
  try:
    yield prefix
  finally:
    _active_prefix.reset(token)
    await asyncio.shield(_close(prefix))


async def _close(prefix: SharedPromptPrefix) -> None:
  users = list(prefix.users)
  for task in users:
    task.cancel()
  await asyncio.gather(*users, return_exceptions=True)
  if prefix.cache_close is not None:
    await prefix.cache_close()
