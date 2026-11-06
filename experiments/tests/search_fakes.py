"""The search server that the graph tests of the in-process agents search, and the Phoenix records of their spans."""

import asyncio
import logging
import socket
import unittest
import warnings
from collections.abc import Sequence
from datetime import datetime, timezone
from typing import Any

import uvicorn
from opentelemetry.sdk.trace import ReadableSpan
from sse_starlette.sse import AppStatus

from indexing.embedding import Embeddings, InputType
from indexing.vector_store import StoredPoint
from search_tool.search import KnowledgeBaseSearch
from search_tool.server import PATH, build_server

# Long texts with characters that JSON escapes, so that a cut or a re-encoding would show.
DOCUMENTS = [
  {
    'id': f'doc-{rank:02d}/"ü"',
    'title': f'Title {rank}: "quoted" — ünïcode',
    'text': f'Document {rank}.\n\tTabs, "quotes", back\\slashes, {{braces}} and ✓ marks. ' * 300,
  }
  for rank in range(10, 0, -1)
]


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


class SearchServerTestCase(unittest.IsolatedAsyncioTestCase):
  """Serves the search tool over MCP at ``self.url``, as the search server does, on ``DOCUMENTS``."""

  async def asyncSetUp(self) -> None:
    # The MCP server leaves a stream of each request to the garbage collector (mcp 1.12.4).
    self.enterContext(warnings.catch_warnings())
    warnings.filterwarnings('ignore', 'Unclosed <MemoryObject', ResourceWarning)
    # The graph's first run is slow to start, which the test loop's debug mode would report.
    asyncio.get_running_loop().slow_callback_duration = 5
    # The traceback of each rejected search, which the search server's command silences too.
    tools_log = logging.getLogger('fastmcp.fastmcp.tools.tool_manager')
    self.addCleanup(tools_log.setLevel, tools_log.level)
    tools_log.setLevel(logging.CRITICAL)
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
    self._serving = (server, serving)
    self.addAsyncCleanup(self.stop_search_server)
    # Before the server stops, which waits for the searches it is serving.
    self.addCleanup(release.set)
    while not server.started:
      if serving.done():
        serving.result()
      await asyncio.sleep(0.01)
    # The server's base URL; its MCP endpoint is at PATH.
    self.url = f'http://127.0.0.1:{port}'

  async def stop_search_server(self) -> None:
    server, serving = self._serving
    server.should_exit = True
    await serving


def phoenix_records(spans: Sequence[ReadableSpan]) -> list[dict[str, Any]]:
  """Exported spans as the Phoenix client returns them: attributes flattened under ``attributes.``."""
  records = []
  for span in spans:
    assert span.context is not None and span.start_time is not None and span.end_time is not None
    records.append(
      {
        'name': span.name,
        'status_code': span.status.status_code.name,
        'status_message': span.status.description or '',
        'parent_id': format(span.parent.span_id, '016x') if span.parent else None,
        'start_time': datetime.fromtimestamp(span.start_time / 1e9, timezone.utc),
        'end_time': datetime.fromtimestamp(span.end_time / 1e9, timezone.utc),
        'context.span_id': format(span.context.span_id, '016x'),
        **{f'attributes.{key}': value for key, value in (span.attributes or {}).items()},
      }
    )
  return records
