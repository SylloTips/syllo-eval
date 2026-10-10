"""The search tool as an MCP server: one tool, served over streamable HTTP at ``/mcp``.

One server searches one collection. Its clients and the random swaps are set up before it starts serving, in the event
loop that serves, and closed when it stops; the HTTP transport is stateless, so concurrent agents share them. The
server also publishes its settings as a resource, which the agents never list, so that a collection can check them.
"""

import logging
from collections.abc import Mapping
from pathlib import Path
from typing import Annotated

from fastmcp import FastMCP
from fastmcp.exceptions import ToolError
from mcp.types import ToolAnnotations
from pydantic import Field, JsonValue

from config import EmbeddingConfig
from indexing.embedding import AzureFoundrySettings, open_embedding_client
from indexing.vector_store import QdrantSettings, open_search_index
from search_tool.search import RESULTS, CallLog, KnowledgeBaseSearch, RandomSwap, SearchError, SearchResults

TOOL_NAME = 'search_knowledge_base'
PATH = '/mcp'
# The server's settings: collection, swap fraction, seed and document cap.
SETTINGS_URI = 'search://settings'

logger = logging.getLogger(__name__)


def build_server(search: KnowledgeBaseSearch, settings: Mapping[str, JsonValue]) -> FastMCP:
  server = FastMCP('Knowledge base search')

  @server.resource(SETTINGS_URI, name='settings', mime_type='application/json')
  def search_settings() -> dict[str, JsonValue]:
    """What this server searches: its collection, the share of random results, their seed and the document cap."""
    return dict(settings)

  @server.tool(
    name=TOOL_NAME,
    description=(
      f'Search the knowledge base. Returns the {RESULTS} most relevant documents, best first, each with its id, '
      'title and text.'
    ),
    annotations=ToolAnnotations(readOnlyHint=True, idempotentHint=True, openWorldHint=False),
  )
  async def search_knowledge_base(
    query: Annotated[str, Field(description='What to look for: a question or keywords.')],
  ) -> SearchResults:
    try:
      return await search.search(query)
    except SearchError as error:
      raise ToolError(f'The search failed: {error}') from error
    except Exception:
      # FastMCP turns it into a tool error; its own traceback is silenced (cli.py), so it is logged here.
      logger.exception('Search for %r failed unexpectedly', query)
      raise

  return server


async def serve(
  collection: str,
  config: EmbeddingConfig,
  foundry: AzureFoundrySettings,
  qdrant: QdrantSettings,
  *,
  swap_fraction: float,
  seed: int,
  max_document_chars: int | None,
  call_log: Path,
  host: str,
  port: int,
) -> None:
  """Serve the search tool of ``collection`` until interrupted, appending every search to ``call_log``."""
  async with open_search_index(qdrant, collection) as index, open_embedding_client(config, foundry) as embedder:
    built_with = (index.metadata.get('model'), index.metadata.get('output_dimension'))
    if built_with != (config.model, config.output_dimension):
      raise ValueError(
        f'Collection {collection} was embedded with {built_with[0]} at {built_with[1]} dimensions, but queries '
        f'would be embedded with {config.model} at {config.output_dimension}'
      )
    if index.status != 'green':
      logger.warning(
        'Qdrant is still optimizing %s (status %s); wait for it to finish before agents search',
        collection,
        index.status,
      )
    swap = RandomSwap(swap_fraction, await index.point_ids(), seed) if swap_fraction else None
    settings: dict[str, JsonValue] = {
      'collection': collection,
      'swap_fraction': swap_fraction,
      'seed': seed,
      'max_document_chars': max_document_chars,
    }
    search = KnowledgeBaseSearch(
      index, embedder, swap=swap, max_document_chars=max_document_chars, call_log=CallLog(call_log, settings)
    )
    logger.info(
      'Starting to serve %s at http://%s:%d%s, logging searches to %s; random documents replace %g of the results%s',
      collection,
      host,
      port,
      PATH,
      call_log,
      swap_fraction,
      f'; documents are cut at {max_document_chars} characters' if max_document_chars else '',
    )
    await build_server(search, settings).run_http_async(
      host=host, port=port, path=PATH, stateless_http=True, uvicorn_config={'access_log': False}
    )
