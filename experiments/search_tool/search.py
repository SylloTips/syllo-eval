"""Knowledge-base search: the ten best documents for a query, of which a share can be replaced by random ones.

The configurations of known lower quality (Sec. 5.3) replace a fraction f of the results with documents drawn at random
from the same knowledge base:
- f x 10 positions are chosen at random. A count that is not whole is rounded up with a probability equal to its
  fractional part, so that on average exactly f of the results are replaced: for f = 0.25, 2 or 3 of the 10, each half
  of the time.
- Each chosen position gets a document drawn uniformly from those not already among the results; the other documents
  keep their ranks.
- The draws depend only on the seed and the query. The search itself is deterministic, so a query always gets the
  same results.

Every search can be appended to a call log, which records what the agent received and which ranks were random, or why
the search failed, so that the traces can be checked against it.
"""

import json
import logging
import random
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Protocol
from uuid import uuid4

from pydantic import BaseModel, JsonValue

from syllo_eval.trace_semantics import RetrievalItem, RetrievalResult

from indexing.embedding import QUERY_INPUT_TYPE, Embedder, EmbeddingError
from indexing.vector_store import StoredPoint, VectorStoreError

RESULTS = 10
TRUNCATION_MARK = ' [truncated]'

logger = logging.getLogger(__name__)


class SearchError(RuntimeError):
  """A search that returned no documents: an empty query, or a model or index that failed."""


class SearchIndex(Protocol):
  async def search(self, vector: Sequence[float], query: str, limit: int) -> list[StoredPoint]: ...

  async def retrieve(self, ids: Sequence[str]) -> list[StoredPoint]: ...


class SearchResult(BaseModel):
  rank: int
  # The id the benchmark's gold lists use.
  id: str
  title: str
  text: str


class SearchResults(BaseModel):
  # Identifies the search in the call log.
  call_id: str
  query: str
  results: list[SearchResult]

  def retrieval(self) -> RetrievalResult:
    items = [RetrievalItem(id=result.id, title=result.title, content=result.text) for result in self.results]
    return RetrievalResult(kind='document', stage='selected', query=self.query, items=items)


class CallLog:
  """A JSON Lines file with one record per search, each with the server's settings."""

  def __init__(self, path: Path, settings: Mapping[str, JsonValue]):
    path.parent.mkdir(parents=True, exist_ok=True)
    self.path = path
    self._settings = dict(settings)

  def write(self, record: Mapping[str, Any]) -> None:
    with self.path.open('a', encoding='utf-8') as file:
      file.write(json.dumps({**self._settings, **record}, ensure_ascii=False) + '\n')


class RandomSwap:
  """Chooses which results a search replaces with random documents, and which documents replace them."""

  def __init__(self, fraction: float, point_ids: Sequence[str], seed: int = 0):
    if not 0 < fraction < 1:
      raise ValueError(f'The swapped fraction must be above 0 and below 1, got {fraction}')
    if len(point_ids) < 2 * RESULTS:
      raise ValueError(f'Random swaps need at least {2 * RESULTS} documents, the knowledge base has {len(point_ids)}')
    self.fraction = fraction
    self._point_ids = point_ids
    self._seed = seed

  def replacements(self, query: str, result_ids: Sequence[str]) -> dict[int, str]:
    """The positions of ``result_ids`` to replace, each with the id of the random point that replaces it."""
    rng = random.Random(f'{self._seed}:{query}')
    expected = self.fraction * len(result_ids)
    count = int(expected) + (rng.random() < expected - int(expected))
    positions = sorted(rng.sample(range(len(result_ids)), count))
    taken = set(result_ids)
    chosen: list[str] = []
    while len(chosen) < count:
      point_id = self._point_ids[rng.randrange(len(self._point_ids))]
      if point_id not in taken:
        taken.add(point_id)
        chosen.append(point_id)
    return dict(zip(positions, chosen))


class KnowledgeBaseSearch:
  def __init__(
    self,
    index: SearchIndex,
    embedder: Embedder,
    *,
    swap: RandomSwap | None = None,
    max_document_chars: int | None = None,
    call_log: CallLog | None = None,
  ):
    self._index = index
    self._embedder = embedder
    self._swap = swap
    self._max_document_chars = max_document_chars
    self._call_log = call_log

  async def search(self, query: str) -> SearchResults:
    """The results for ``query``; a search that fails raises SearchError, after being logged with its error."""
    call_id = uuid4().hex
    try:
      if not query.strip():
        raise SearchError('the query is empty')
      found, record = await self._search(call_id, query)
    except SearchError as error:
      self._failed(call_id, query, error)
      raise
    except (EmbeddingError, VectorStoreError) as error:
      self._failed(call_id, query, error)
      raise SearchError(str(error)) from error
    logger.info('Search %s for %r; random documents at ranks %s', call_id, query, record['random_ranks'])
    self._log(record)
    return found

  async def _search(self, call_id: str, query: str) -> tuple[SearchResults, dict[str, Any]]:
    embedding = await self._embedder.embed([query], QUERY_INPUT_TYPE)
    retrieved = await self._index.search(embedding.vectors[0], query, RESULTS)
    points = list(retrieved)
    swapped: dict[int, str] = {}
    if self._swap is not None:
      swapped = self._swap.replacements(query, [point.id for point in points])
      for position, point in zip(swapped, await self._index.retrieve(list(swapped.values()))):
        points[position] = point
    found = SearchResults(
      call_id=call_id, query=query, results=[self._result(rank, point) for rank, point in enumerate(points, 1)]
    )
    record = {
      'call_id': call_id,
      'query': query,
      'ids': [result.id for result in found.results],
      'random_ranks': [position + 1 for position in swapped],
      'replaced_ids': [retrieved[position].payload['document_id'] for position in swapped],
    }
    return found, record

  def _failed(self, call_id: str, query: str, error: Exception) -> None:
    logger.warning('Search %s for %r failed: %s', call_id, query, error)
    self._log({'call_id': call_id, 'query': query, 'error': str(error)})

  def _log(self, record: Mapping[str, Any]) -> None:
    if self._call_log is not None:
      self._call_log.write({'at': datetime.now(timezone.utc).isoformat(), **record})

  def _result(self, rank: int, point: StoredPoint) -> SearchResult:
    text = point.payload['text']
    if self._max_document_chars is not None and len(text) > self._max_document_chars:
      text = text[: self._max_document_chars] + TRUNCATION_MARK
    return SearchResult(rank=rank, id=point.payload['document_id'], title=point.payload['title'], text=text)
