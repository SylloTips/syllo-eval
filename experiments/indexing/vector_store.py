"""The vector store that holds the search indexes. Its Qdrant implementation is not written yet."""

from collections.abc import AsyncIterator, Mapping, Sequence
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Protocol
from uuid import UUID

from pydantic import JsonValue


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


@asynccontextmanager
async def open_vector_store() -> AsyncIterator[VectorStore]:
  raise NotImplementedError('The Qdrant vector store is not implemented yet, so the indexes cannot be loaded')
  yield
