"""The vector store that holds the search indexes: a Qdrant server, reached through its REST API.

A collection holds one point per document with two vectors: the document's embedding (``dense``, compared by cosine)
and its BM25 term weights (``bm25``), which Qdrant computes from the text and weighs by inverse document frequency at
query time. A search ranks the documents both ways and fuses the two rankings by reciprocal rank, here rather than in
Qdrant: Qdrant's fusion orders tied documents at random, so the same query could return different results.
"""

import math
import re
from collections import defaultdict
from collections.abc import AsyncIterator, Iterable, Iterator, Mapping, Sequence
from contextlib import asynccontextmanager, contextmanager
from dataclasses import dataclass
from typing import Any, Protocol
from uuid import UUID

from pydantic import JsonValue
from pydantic_settings import SettingsConfigDict
from qdrant_client import AsyncQdrantClient, models
from qdrant_client.common.client_exceptions import ResourceExhaustedResponse
from qdrant_client.http.exceptions import ResponseHandlingException, UnexpectedResponse

from syllo_eval.settings import EnvSettings

# Upserts wait until the points are stored, often beyond the client's 5-second default.
_TIMEOUT_SECONDS = 120
DENSE_VECTOR = 'dense'
BM25_VECTOR = 'bm25'
# Qdrant tokenizes, removes English stopwords and stems the text itself.
BM25_MODEL = 'Qdrant/bm25'
# Documents each ranking contributes to the fusion, and the constant of reciprocal rank fusion (Cormack et al., 2009).
RANKING_LIMIT = 50
RRF_K = 60
_SCROLL_PAGE = 10_000
# Letters and digits: Qdrant's tokenizer splits words at any other character, underscores included.
_WORD = re.compile(r'[^\W_]+')
# NLTK's English stopwords, close to the list Qdrant drops before it counts a document's BM25 tokens.
_STOPWORDS = frozenset(
  """a about above after again against ain all am an and any are aren as at be because been before being below between
  both but by can couldn d did didn do does doesn doing don down during each few for from further had hadn has hasn
  have haven having he her here hers herself him himself his how i if in into is isn it its itself just ll m ma me
  mightn more most mustn my myself needn no nor not now o of off on once only or other our ours ourselves out over own
  re s same shan she should shouldn so some such t than that the their theirs them themselves then there these they
  this those through to too under until up ve very was wasn we were weren what when where which while who whom why
  will with won wouldn y you your yours yourself yourselves""".split()
)


class QdrantSettings(EnvSettings):
  model_config = SettingsConfigDict(env_prefix='QDRANT_')

  url: str = 'http://localhost:6333'
  api_key: str | None = None


class VectorStoreError(RuntimeError):
  """A request the vector store rejected or never answered."""


@dataclass(frozen=True, slots=True)
class CollectionSpec:
  name: str
  dimension: int
  # The mean length of the documents in BM25 tokens, by which BM25 normalizes their term frequencies.
  bm25_average_length: float
  # What the collection was built from, stored with it so that a search can check it embeds queries the same way.
  metadata: Mapping[str, JsonValue]


@dataclass(frozen=True, slots=True)
class IndexPoint:
  id: UUID
  vector: Sequence[float]
  # The text BM25 indexes.
  text: str
  payload: Mapping[str, JsonValue]


@dataclass(frozen=True, slots=True)
class StoredPoint:
  id: str
  payload: Mapping[str, Any]


class VectorStore(Protocol):
  async def ensure_collection(self, spec: CollectionSpec) -> None:
    """Create the collection, or check that an existing one has the same vectors and metadata."""

  async def upsert(self, spec: CollectionSpec, points: Sequence[IndexPoint]) -> None:
    """Store the points, replacing any stored under the same ids."""

  async def count(self, collection: str) -> int:
    """The exact number of points in the collection."""


def bm25_length(text: str) -> int:
  """An estimate of the number of tokens Qdrant's BM25 counts in ``text``: its words that are not stopwords."""
  return sum(1 for word in _WORD.findall(text.lower()) if word not in _STOPWORDS)


def bm25_average_length(texts: Iterable[str]) -> float:
  """The mean BM25 length of ``texts``, and at least 1, which Qdrant requires."""
  lengths = [bm25_length(text) for text in texts]
  return max(1.0, round(sum(lengths) / len(lengths), 1)) if lengths else 1.0


def bm25_document(text: str, average_length: float) -> models.Document:
  return models.Document(text=text, model=BM25_MODEL, options=models.Bm25Config(avg_len=average_length))


def reciprocal_rank_fusion(rankings: Sequence[Sequence[str]], k: int = RRF_K) -> list[str]:
  """Fuse rankings of ids: an id scores the sum of 1 / (k + its rank) over the rankings that hold it, ranks from 1.

  Ties go to the better best rank, then to the better ranks in ranking order, then to the smaller id.
  """
  scores: defaultdict[str, float] = defaultdict(float)
  ranks: dict[str, list[float]] = {}
  for position, ranking in enumerate(rankings):
    for rank, point_id in enumerate(ranking, 1):
      scores[point_id] += 1 / (k + rank)
      ranks.setdefault(point_id, [math.inf] * len(rankings))[position] = rank
  return sorted(scores, key=lambda point_id: (-scores[point_id], min(ranks[point_id]), ranks[point_id], point_id))


class QdrantVectorStore:
  """Collections of documents with a dense and a BM25 vector each, kept in memory."""

  def __init__(self, client: AsyncQdrantClient) -> None:
    self._client = client

  async def ensure_collection(self, spec: CollectionSpec) -> None:
    with _requests(f'prepare collection {spec.name}'):
      if not await self._client.collection_exists(spec.name):
        await self._client.create_collection(
          spec.name,
          vectors_config={DENSE_VECTOR: models.VectorParams(size=spec.dimension, distance=models.Distance.COSINE)},
          sparse_vectors_config={BM25_VECTOR: models.SparseVectorParams(modifier=models.Modifier.IDF)},
          metadata=dict(spec.metadata),
        )
        return
      config = (await self._client.get_collection(spec.name)).config
    if not _has_search_vectors(config.params, spec.dimension) or config.metadata != dict(spec.metadata):
      raise ValueError(
        f'Collection {spec.name} exists with other vectors or metadata than {dict(spec.metadata)}; '
        'delete it to load again'
      )

  async def upsert(self, spec: CollectionSpec, points: Sequence[IndexPoint]) -> None:
    structs = [
      models.PointStruct(
        id=str(point.id),
        vector={
          DENSE_VECTOR: list(point.vector),
          BM25_VECTOR: bm25_document(point.text, spec.bm25_average_length),
        },
        payload=dict(point.payload),
      )
      for point in points
    ]
    with _requests(f'upsert {len(points)} points into {spec.name}'):
      await self._client.upsert(spec.name, points=structs, wait=True)

  async def count(self, collection: str) -> int:
    with _requests(f'count the points of {collection}'):
      return (await self._client.count(collection, exact=True)).count


class QdrantSearchIndex:
  """Hybrid search in one collection: the exact nearest embeddings and the best BM25 matches, fused by reciprocal rank.

  The dense ranking is exact rather than approximate, so it does not depend on how Qdrant built its index. Each
  ranking orders documents with equal scores by id, including at the cut: Qdrant would choose the documents that tie
  at its limit by how it stores them.
  """

  def __init__(
    self, client: AsyncQdrantClient, collection: str, metadata: Mapping[str, Any], status: str = 'green'
  ) -> None:
    self._client = client
    self.collection = collection
    # What the collection was built from, as the load step stored it.
    self.metadata = metadata
    # Qdrant's status of the collection when it was opened: 'green' once its optimizers are done.
    self.status = status

  async def search(self, vector: Sequence[float], query: str, limit: int) -> list[StoredPoint]:
    """The ``limit`` best points for a query, given its embedding."""
    rankings = await self._rankings(
      [
        models.QueryRequest(query=list(vector), using=DENSE_VECTOR, params=models.SearchParams(exact=True)),
        models.QueryRequest(query=bm25_document(query, self.metadata['bm25_average_length']), using=BM25_VECTOR),
      ]
    )
    return await self.retrieve(reciprocal_rank_fusion(rankings)[:limit])

  async def _rankings(self, requests: Sequence[models.QueryRequest]) -> list[list[str]]:
    """The first ``RANKING_LIMIT`` ids of each ranking, with equal scores ordered by id.

    Each ranking is fetched past its cut, and further while the documents that tie at the cut run past what was
    fetched, so that the id decides which of them make the cut.
    """
    fetched = 2 * RANKING_LIMIT
    while True:
      with _requests(f'search {self.collection}'):
        responses = await self._client.query_batch_points(
          self.collection, requests=[request.model_copy(update={'limit': fetched}) for request in requests]
        )
      rankings = [sorted(response.points, key=lambda point: (-point.score, str(point.id))) for response in responses]
      if not any(_tie_runs_past(ranking, fetched) for ranking in rankings):
        return [[str(point.id) for point in ranking[:RANKING_LIMIT]] for ranking in rankings]
      fetched *= 2

  async def retrieve(self, ids: Sequence[str]) -> list[StoredPoint]:
    """The points with these ids, in this order."""
    with _requests(f'retrieve {len(ids)} points of {self.collection}'):
      records = await self._client.retrieve(self.collection, list(ids), with_payload=True)
    by_id = {str(record.id): record.payload or {} for record in records}
    return [StoredPoint(id=point_id, payload=by_id[point_id]) for point_id in ids]

  async def point_ids(self) -> list[str]:
    """The ids of every point in the collection, sorted."""
    ids: list[str] = []
    offset: Any = None
    with _requests(f'list the points of {self.collection}'):
      while True:
        records, offset = await self._client.scroll(
          self.collection, limit=_SCROLL_PAGE, offset=offset, with_payload=False, with_vectors=False
        )
        ids.extend(str(record.id) for record in records)
        if offset is None:
          return sorted(ids)


@asynccontextmanager
async def open_vector_store(settings: QdrantSettings) -> AsyncIterator[VectorStore]:
  client = AsyncQdrantClient(url=settings.url, api_key=settings.api_key, timeout=_TIMEOUT_SECONDS)
  try:
    yield QdrantVectorStore(client)
  finally:
    await client.close()


@asynccontextmanager
async def open_search_index(settings: QdrantSettings, collection: str) -> AsyncIterator[QdrantSearchIndex]:
  client = AsyncQdrantClient(url=settings.url, api_key=settings.api_key, timeout=_TIMEOUT_SECONDS)
  try:
    yield await search_index(client, collection)
  finally:
    await client.close()


async def search_index(client: AsyncQdrantClient, collection: str) -> QdrantSearchIndex:
  """The search index of a loaded collection; one loaded without a hybrid search's vectors and metadata is refused."""
  with _requests(f'read collection {collection}'):
    if not await client.collection_exists(collection):
      raise ValueError(f'Qdrant has no collection {collection}; load it with `syllo-exp index load`')
    info = await client.get_collection(collection)
  config = info.config
  metadata = config.metadata or {}
  dimension = metadata.get('output_dimension')
  if (
    not isinstance(dimension, int)
    or not _has_search_vectors(config.params, dimension)
    or 'bm25_average_length' not in metadata
  ):
    raise ValueError(
      f'Collection {collection} was not loaded for hybrid search, with a dense and a BM25 vector; delete it and load it'
      ' again'
    )
  return QdrantSearchIndex(client, collection, metadata, str(info.status.value))


def _tie_runs_past(ranking: Sequence[models.ScoredPoint], fetched: int) -> bool:
  """Whether the score at the cut may continue past the fetched points: they fill the fetch and end on that score."""
  return len(ranking) == fetched and ranking[RANKING_LIMIT - 1].score == ranking[-1].score


def _has_search_vectors(params: models.CollectionParams, dimension: int) -> bool:
  vectors, sparse = params.vectors, params.sparse_vectors or {}
  dense = vectors.get(DENSE_VECTOR) if isinstance(vectors, dict) else None
  return (
    dense is not None
    and (dense.size, dense.distance) == (dimension, models.Distance.COSINE)
    and BM25_VECTOR in sparse
    and sparse[BM25_VECTOR].modifier == models.Modifier.IDF
  )


@contextmanager
def _requests(action: str) -> Iterator[None]:
  try:
    yield
  except (UnexpectedResponse, ResponseHandlingException, ResourceExhaustedResponse) as error:
    raise VectorStoreError(f'Qdrant could not {action}: {error}') from error
