import math
import unittest
from types import SimpleNamespace
from typing import Any, cast
from uuid import UUID, uuid4

import httpx
from qdrant_client import AsyncQdrantClient, models

from indexing.vector_store import (
  BM25_VECTOR,
  DENSE_VECTOR,
  RANKING_LIMIT,
  CollectionSpec,
  IndexPoint,
  QdrantSearchIndex,
  QdrantSettings,
  QdrantVectorStore,
  VectorStoreError,
  bm25_average_length,
  bm25_length,
  open_vector_store,
  reciprocal_rank_fusion,
  search_index,
)


def _spec(name: str = 'kb', dimension: int = 3, **metadata: Any) -> CollectionSpec:
  fields = {'model': 'embed-test', 'output_dimension': dimension, 'bm25_average_length': 2.0, **metadata}
  return CollectionSpec(name=name, dimension=dimension, bm25_average_length=2.0, metadata=fields)


class Bm25LengthTest(unittest.TestCase):
  def test_counts_the_words_that_are_not_stopwords(self) -> None:
    self.assertEqual(bm25_length('How do I connect a domain to my site? The DNS-settings page.'), 6)
    # Qdrant's tokenizer splits words at underscores too.
    self.assertEqual(bm25_length('support_engineer: there is a fix'), 3)
    self.assertEqual(bm25_average_length(['refund policy and terms', 'the refund', '']), 1.3)

  def test_the_average_length_is_at_least_one_as_qdrant_requires(self) -> None:
    self.assertEqual(bm25_average_length(['', 'the']), 1.0)
    self.assertEqual(bm25_average_length([]), 1.0)


class ReciprocalRankFusionTest(unittest.TestCase):
  def test_an_id_scores_the_sum_of_one_over_k_plus_its_ranks(self) -> None:
    # a: 1/(60+1) + 1/(60+3); b: 1/(60+2); c: 1/(60+3) + 1/(60+1).
    self.assertEqual(reciprocal_rank_fusion([['a', 'b', 'c'], ['c', 'd', 'a']]), ['a', 'c', 'b', 'd'])

  def test_ties_go_to_the_best_rank_then_to_the_earlier_ranking_then_to_the_smaller_id(self) -> None:
    # x is 2nd in the first ranking and y 2nd in the second: equal scores and best ranks, so the first ranking wins.
    self.assertEqual(reciprocal_rank_fusion([['a', 'x'], ['a', 'y']]), ['a', 'x', 'y'])
    self.assertEqual(reciprocal_rank_fusion([['n'], ['m']]), ['n', 'm'])
    self.assertEqual(reciprocal_rank_fusion([['n', 'm'], ['m', 'n']]), ['n', 'm'])


class QdrantVectorStoreTest(unittest.IsolatedAsyncioTestCase):
  """Collection setup, in qdrant-client's local mode; BM25 needs a server (QdrantServerTest)."""

  async def asyncSetUp(self) -> None:
    self.client = AsyncQdrantClient(location=':memory:')
    self.store = QdrantVectorStore(self.client)

  async def asyncTearDown(self) -> None:
    await self.client.close()

  async def test_a_new_collection_has_a_dense_and_a_bm25_vector_and_the_metadata(self) -> None:
    await self.store.ensure_collection(_spec())

    config = (await self.client.get_collection('kb')).config
    self.assertEqual(
      config.params.vectors, {DENSE_VECTOR: models.VectorParams(size=3, distance=models.Distance.COSINE)}
    )
    self.assertEqual(
      config.params.sparse_vectors, {BM25_VECTOR: models.SparseVectorParams(modifier=models.Modifier.IDF)}
    )
    self.assertEqual(config.metadata, dict(_spec().metadata))
    self.assertEqual(await self.store.count('kb'), 0)

  async def test_an_existing_collection_is_kept_only_if_its_vectors_and_metadata_match(self) -> None:
    await self.store.ensure_collection(_spec())
    await self.store.ensure_collection(_spec())

    for other in (_spec(dimension=4), _spec(model='embed-other'), _spec(bm25_average_length=3.0)):
      with self.subTest(other=other), self.assertRaisesRegex(ValueError, 'exists with other vectors or metadata'):
        await self.store.ensure_collection(other)
    await self.client.create_collection(
      'old', vectors_config=models.VectorParams(size=3, distance=models.Distance.COSINE)
    )
    with self.assertRaisesRegex(ValueError, 'Collection old exists with other vectors'):
      await self.store.ensure_collection(_spec(name='old'))

  async def test_an_existing_collection_with_the_metadata_but_other_vectors_is_refused(self) -> None:
    cosine = models.VectorParams(size=3, distance=models.Distance.COSINE)
    idf = models.SparseVectorParams(modifier=models.Modifier.IDF)
    others = {
      'dot': ({DENSE_VECTOR: models.VectorParams(size=3, distance=models.Distance.DOT)}, {BM25_VECTOR: idf}),
      'size': ({DENSE_VECTOR: models.VectorParams(size=4, distance=models.Distance.COSINE)}, {BM25_VECTOR: idf}),
      'no-idf': ({DENSE_VECTOR: cosine}, {BM25_VECTOR: models.SparseVectorParams()}),
    }
    for name, (dense, sparse) in others.items():
      await self.client.create_collection(
        name, vectors_config=dense, sparse_vectors_config=sparse, metadata=dict(_spec().metadata)
      )
      with self.subTest(name=name), self.assertRaisesRegex(ValueError, 'exists with other vectors or metadata'):
        await self.store.ensure_collection(_spec(name=name))
      with self.subTest(name=name), self.assertRaisesRegex(ValueError, 'not loaded for hybrid search'):
        await search_index(self.client, name)

  async def test_searching_needs_a_collection_loaded_with_bm25_vectors(self) -> None:
    await self.store.ensure_collection(_spec())
    self.assertEqual((await search_index(self.client, 'kb')).metadata['bm25_average_length'], 2.0)

    with self.assertRaisesRegex(ValueError, 'no collection missing; load it'):
      await search_index(self.client, 'missing')
    await self.client.create_collection(
      'old', vectors_config=models.VectorParams(size=3, distance=models.Distance.COSINE)
    )
    with self.assertRaisesRegex(ValueError, 'Collection old was not loaded for hybrid search'):
      await search_index(self.client, 'old')


def _point(n: int, score: float) -> models.ScoredPoint:
  return models.ScoredPoint(id=str(UUID(int=n)), version=0, score=score)


class FakeQdrant:
  """Answers each ranking with its points in a fixed order, cut at the request's limit, as Qdrant cuts rankings."""

  def __init__(self, rankings: list[list[models.ScoredPoint]]) -> None:
    self.rankings = rankings
    self.limits: list[int | None] = []

  async def query_batch_points(self, collection: str, requests: list[models.QueryRequest]) -> list[SimpleNamespace]:
    self.limits.append(requests[0].limit)
    return [SimpleNamespace(points=ranking[: request.limit]) for ranking, request in zip(self.rankings, requests)]

  async def retrieve(self, collection: str, ids: list[str], with_payload: bool) -> list[models.Record]:
    return [models.Record(id=point_id, payload={}) for point_id in ids]


class RankingCutTest(unittest.IsolatedAsyncioTestCase):
  async def test_documents_that_tie_at_the_cut_are_taken_by_id_however_qdrant_lists_them(self) -> None:
    # BM25: 49 documents score 2, then 80 tie at 1 across the cut at rank 50 and past the first 100 fetched. Qdrant
    # lists the tied ones from the largest id, 179, which its own cut would keep. The dense ranking holds 179 at
    # rank 50 and leaves out 100, the smallest tied id: 179 would reach the top 10 only if BM25 kept it too.
    keywords = [_point(n, 2.0) for n in range(49)] + [_point(n, 1.0) for n in range(179, 99, -1)]
    dense = [_point(n, 1 - (n - 130) / 1000) for n in range(130, 180)] + [_point(n, 0.5) for n in range(100, 130)]
    fake = FakeQdrant([dense, keywords])
    index = QdrantSearchIndex(cast(AsyncQdrantClient, fake), 'kb', {'bm25_average_length': 2.0})

    found = await index.search([1.0, 0.0, 0.0], 'zeta', limit=10)

    by_id = [str(UUID(int=n)) for n in range(49)] + [str(UUID(int=100))]
    expected = reciprocal_rank_fusion([[str(point.id) for point in dense[:RANKING_LIMIT]], by_id])[:10]
    self.assertEqual([point.id for point in found], expected)
    self.assertNotIn(str(UUID(int=179)), expected)
    self.assertEqual(fake.limits, [2 * RANKING_LIMIT, 4 * RANKING_LIMIT])


class QdrantServerTest(unittest.IsolatedAsyncioTestCase):
  """BM25 and hybrid search on the Qdrant server at QDRANT_URL, in a temporary collection; skipped without a server."""

  async def asyncSetUp(self) -> None:
    self.client = AsyncQdrantClient(url=QdrantSettings().url, api_key=QdrantSettings().api_key)
    self.addAsyncCleanup(self.client.close)
    try:
      await self.client.get_collections()
    except Exception as error:
      self.skipTest(f'No Qdrant server: {error}')
    self.spec = _spec(name=f'test-search-{uuid4().hex[:8]}')
    # Registered before the collection exists: Qdrant accepts deleting a missing collection.
    self.addAsyncCleanup(self.client.delete_collection, self.spec.name)
    self.store = QdrantVectorStore(self.client)
    await self.store.ensure_collection(self.spec)
    self.ids = {name: uuid4() for name in ('refund', 'domain', 'mobile')}
    await self.store.upsert(
      self.spec,
      [
        IndexPoint(id=self.ids['refund'], vector=[1.0, 0.0, 0.0], text='Refund policy for invoices', payload={'n': 1}),
        IndexPoint(id=self.ids['domain'], vector=[0.0, 1.0, 0.0], text='Connect a domain name', payload={'n': 2}),
        IndexPoint(id=self.ids['mobile'], vector=[0.0, 0.0, 1.0], text='Mobile app notifications', payload={'n': 3}),
      ],
    )

  async def test_hybrid_search_fuses_the_keyword_and_the_semantic_ranking(self) -> None:
    index = await search_index(self.client, self.spec.name)

    # The embedding is closest to the domain article; only the refund article has the keyword.
    points = await index.search([0.0, 1.0, 0.0], 'refunds', limit=3)

    self.assertEqual([point.id for point in points], [str(self.ids[name]) for name in ('refund', 'domain', 'mobile')])
    self.assertEqual(points[0].payload, {'n': 1})

  async def test_retrieves_points_and_lists_every_id(self) -> None:
    index = await search_index(self.client, self.spec.name)

    retrieved = await index.retrieve([str(self.ids['mobile']), str(self.ids['refund'])])

    self.assertEqual([point.payload['n'] for point in retrieved], [3, 1])
    self.assertEqual(await index.point_ids(), sorted(str(point_id) for point_id in self.ids.values()))

  async def test_upserting_the_same_id_replaces_the_point_and_its_payload(self) -> None:
    point = IndexPoint(id=self.ids['refund'], vector=[0.0, 0.0, 1.0], text='Shipping times', payload={'n': 4})
    await self.store.upsert(self.spec, [point])

    self.assertEqual(await self.store.count(self.spec.name), 3)
    [record] = await self.client.retrieve(self.spec.name, [str(point.id)], with_vectors=[DENSE_VECTOR])
    self.assertEqual((record.payload, record.vector), ({'n': 4}, {DENSE_VECTOR: [0.0, 0.0, 1.0]}))

  async def test_documents_that_tie_at_the_cut_of_a_ranking_give_the_same_results_whatever_the_load_order(self) -> None:
    # 49 documents match 'zeta' twice, then 80 identical ones match it once: BM25 ties them across its cut at rank 50,
    # and past the first 100 documents fetched. Every document is at a different angle from the query embedding.
    texts = ['zeta zeta'] * 49 + ['zeta omega'] * 80
    points = [
      IndexPoint(id=uuid4(), vector=[math.cos(n / 100), math.sin(n / 100), 0.0], text=text, payload={'n': n})
      for n, text in enumerate(texts)
    ]
    dense = [str(point.id) for point in points]
    keywords = sorted(dense[:49]) + sorted(dense[49:])
    expected = reciprocal_rank_fusion([dense[:RANKING_LIMIT], keywords[:RANKING_LIMIT]])[:10]

    for order, loaded in (('forward', points), ('reverse', points[::-1])):
      spec = _spec(name=f'{self.spec.name}-{order}', bm25_average_length=2.0)
      self.addAsyncCleanup(self.client.delete_collection, spec.name)
      await self.store.ensure_collection(spec)
      for start in range(0, len(loaded), 16):
        await self.store.upsert(spec, loaded[start : start + 16])
      found = await (await search_index(self.client, spec.name)).search([1.0, 0.0, 0.0], 'zeta', limit=10)

      with self.subTest(order=order):
        self.assertEqual([point.id for point in found], expected)


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
