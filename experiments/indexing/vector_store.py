"""The vector store that holds the search indexes: a Qdrant server, reached through its REST API."""

from collections.abc import AsyncIterator, Iterator, Mapping, Sequence
from contextlib import asynccontextmanager, contextmanager
from dataclasses import dataclass
from typing import Protocol
from uuid import UUID

from pydantic import JsonValue
from pydantic_settings import SettingsConfigDict
from qdrant_client import AsyncQdrantClient, models
from qdrant_client.common.client_exceptions import ResourceExhaustedResponse
from qdrant_client.http.exceptions import ResponseHandlingException, UnexpectedResponse

from syllo_eval.settings import EnvSettings

# Upserts wait until the points are stored, often beyond the client's 5-second default.
_TIMEOUT_SECONDS = 120


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


@dataclass(frozen=True, slots=True)
class IndexPoint:
  id: UUID
  vector: Sequence[float]
  payload: Mapping[str, JsonValue]


class VectorStore(Protocol):
  async def ensure_collection(self, spec: CollectionSpec) -> None:
    """Create the collection for cosine similarity, or check that an existing one has the same vector size."""

  async def upsert(self, collection: str, points: Sequence[IndexPoint]) -> None:
    """Store the points, replacing any stored under the same ids."""

  async def count(self, collection: str) -> int:
    """The exact number of points in the collection."""


class QdrantVectorStore:
  """Collections with one unnamed dense vector per point, compared by cosine and kept in memory."""

  def __init__(self, client: AsyncQdrantClient) -> None:
    self._client = client

  async def ensure_collection(self, spec: CollectionSpec) -> None:
    expected = models.VectorParams(size=spec.dimension, distance=models.Distance.COSINE)
    with _requests(f'prepare collection {spec.name}'):
      if not await self._client.collection_exists(spec.name):
        await self._client.create_collection(spec.name, vectors_config=expected)
        return
      vectors = (await self._client.get_collection(spec.name)).config.params.vectors
    if not isinstance(vectors, models.VectorParams) or (vectors.size, vectors.distance) != (
      expected.size,
      expected.distance,
    ):
      raise ValueError(
        f'Collection {spec.name} exists with vectors other than {spec.dimension} dimensions compared by cosine; '
        'delete it to load again'
      )

  async def upsert(self, collection: str, points: Sequence[IndexPoint]) -> None:
    structs = [
      models.PointStruct(id=str(point.id), vector=list(point.vector), payload=dict(point.payload)) for point in points
    ]
    with _requests(f'upsert {len(points)} points into {collection}'):
      await self._client.upsert(collection, points=structs, wait=True)

  async def count(self, collection: str) -> int:
    with _requests(f'count the points of {collection}'):
      return (await self._client.count(collection, exact=True)).count


@asynccontextmanager
async def open_vector_store(settings: QdrantSettings) -> AsyncIterator[VectorStore]:
  client = AsyncQdrantClient(url=settings.url, api_key=settings.api_key, timeout=_TIMEOUT_SECONDS)
  try:
    yield QdrantVectorStore(client)
  finally:
    await client.close()


@contextmanager
def _requests(action: str) -> Iterator[None]:
  try:
    yield
  except (UnexpectedResponse, ResponseHandlingException, ResourceExhaustedResponse) as error:
    raise VectorStoreError(f'Qdrant could not {action}: {error}') from error
