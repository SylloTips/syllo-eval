import unittest
from uuid import uuid4

import httpx
from qdrant_client import AsyncQdrantClient, models

from indexing.vector_store import (
  CollectionSpec,
  IndexPoint,
  QdrantSettings,
  QdrantVectorStore,
  VectorStoreError,
  open_vector_store,
)


class QdrantVectorStoreTest(unittest.IsolatedAsyncioTestCase):
  async def asyncSetUp(self) -> None:
    self.client = AsyncQdrantClient(location=':memory:')
    self.store = QdrantVectorStore(self.client)

  async def asyncTearDown(self) -> None:
    await self.client.close()

  async def test_a_new_collection_compares_vectors_of_the_given_size_by_cosine(self) -> None:
    await self.store.ensure_collection(CollectionSpec(name='kb', dimension=3))

    vectors = (await self.client.get_collection('kb')).config.params.vectors
    self.assertEqual(vectors, models.VectorParams(size=3, distance=models.Distance.COSINE))
    self.assertEqual(await self.store.count('kb'), 0)

  async def test_an_existing_collection_is_kept_only_if_its_vectors_match(self) -> None:
    await self.store.ensure_collection(CollectionSpec(name='kb', dimension=3))
    await self.store.upsert('kb', [IndexPoint(id=uuid4(), vector=[1.0, 0.0, 0.0], payload={})])

    await self.store.ensure_collection(CollectionSpec(name='kb', dimension=3))
    self.assertEqual(await self.store.count('kb'), 1)
    with self.assertRaisesRegex(ValueError, 'Collection kb exists with vectors other than 4 dimensions'):
      await self.store.ensure_collection(CollectionSpec(name='kb', dimension=4))

    await self.client.create_collection('dot', vectors_config=models.VectorParams(size=3, distance=models.Distance.DOT))
    with self.assertRaisesRegex(ValueError, 'compared by cosine'):
      await self.store.ensure_collection(CollectionSpec(name='dot', dimension=3))

  async def test_upserting_the_same_id_replaces_the_point_and_its_payload(self) -> None:
    point_id = uuid4()
    await self.store.ensure_collection(CollectionSpec(name='kb', dimension=2))
    await self.store.upsert('kb', [IndexPoint(id=point_id, vector=[1.0, 0.0], payload={'document_id': 'a', 'v': 1})])
    await self.store.upsert('kb', [IndexPoint(id=point_id, vector=[0.0, 1.0], payload={'document_id': 'a', 'v': 2})])

    self.assertEqual(await self.store.count('kb'), 1)
    [record] = await self.client.retrieve('kb', [str(point_id)], with_vectors=True)
    self.assertEqual((record.payload, record.vector), ({'document_id': 'a', 'v': 2}, [0.0, 1.0]))


class OpenVectorStoreTest(unittest.IsolatedAsyncioTestCase):
  async def test_throttling_raises_a_vector_store_error(self) -> None:
    transport = httpx.MockTransport(
      lambda request: httpx.Response(429, headers={'Retry-After': '1'}, json={'status': {'error': 'throttled'}})
    )
    client = AsyncQdrantClient(url='http://qdrant:6333', transport=transport, check_compatibility=False)
    try:
      with self.assertRaisesRegex(VectorStoreError, 'Qdrant could not count the points of kb: throttled'):
        await QdrantVectorStore(client).count('kb')
    finally:
      await client.close()

  async def test_an_unreachable_server_raises_a_vector_store_error(self) -> None:
    async with open_vector_store(QdrantSettings(url='http://127.0.0.1:9')) as store:
      with self.assertRaisesRegex(VectorStoreError, 'Qdrant could not count the points of kb'):
        await store.count('kb')


if __name__ == '__main__':
  unittest.main()
