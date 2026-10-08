import asyncio
import hashlib
import json
import tempfile
import unittest
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import httpx
import pyarrow.parquet as pq
from pydantic import JsonValue

from benchmarks.common import KnowledgeDocument
from config import BenchmarkSource, EmbeddingConfig
from indexing import pipeline
from indexing.embedding import (
  EMBED_PATH,
  AzureFoundrySettings,
  CohereEmbeddingClient,
  EmbeddingError,
  Embeddings,
  InputType,
  TokenRateLimiter,
  request_batches,
)
from indexing.vector_store import CollectionSpec, IndexPoint

DIMENSION = 4


def _articles(count: int) -> list[dict[str, str]]:
  return [
    {
      'id': f'k{index}',
      'url': f'https://support.example/{index}',
      'title': f'Article {index}',
      'contents': f'Article {index}\nBody {index}' + ('\x00' if index == 3 else ''),
      'html_content': '<div></div>',
      'article_type': 'feature_request' if index == 4 else 'article',
    }
    for index in range(count)
  ]


def _article_number(text: str) -> float:
  """The number of a synthetic article, from its embedded text 'Article <n>\\nBody <n>'."""
  return float(text.split()[1])


def _config(**overrides: Any) -> EmbeddingConfig:
  fields: dict[str, Any] = {
    'provider': 'azure_foundry',
    'model': 'embed-test',
    'label': 'Embed test',
    'output_dimension': DIMENSION,
    'timeout_seconds': 10,
    'max_retries': 3,
    'max_concurrent_requests': 2,
    'tokens_per_minute': 1_000_000,
    'max_request_tokens': 1_000,
  }
  return EmbeddingConfig.model_validate({**fields, **overrides})


def _document(title: str, text: str) -> KnowledgeDocument:
  return KnowledgeDocument(document_id='d', source_document_id='d', title=title, text=text)


class FakeEmbedder:
  """Embeds an article as [its number, the call, 1, 0]; ``fail_on_call`` makes that call fail."""

  def __init__(self, fail_on_call: int | None = None):
    self.calls: list[list[str]] = []
    self._fail_on_call = fail_on_call

  async def embed(self, texts: Sequence[str], input_type: InputType) -> Embeddings:
    assert input_type == 'search_document'
    self.calls.append(list(texts))
    call = len(self.calls)
    if call == self._fail_on_call:
      raise EmbeddingError('The embedding request was rejected: 400 bad request')
    return Embeddings(
      vectors=[[_article_number(text), float(call), 1.0, 0.0] for text in texts], input_tokens=10 * len(texts)
    )


class FakeVectorStore:
  def __init__(self) -> None:
    self.specs: dict[str, CollectionSpec] = {}
    self.points: dict[str, dict[Any, IndexPoint]] = {}
    self.upserts: list[int] = []

  async def ensure_collection(self, spec: CollectionSpec) -> None:
    assert self.specs.setdefault(spec.name, spec) == spec
    self.points.setdefault(spec.name, {})

  async def upsert(self, spec: CollectionSpec, points: Sequence[IndexPoint]) -> None:
    assert self.specs[spec.name] == spec
    self.upserts.append(len(points))
    self.points[spec.name].update({point.id: point for point in points})

  async def count(self, collection: str) -> int:
    return len(self.points[collection])


class EmbeddingTextTest(unittest.TestCase):
  def test_the_title_comes_first_unless_the_text_already_starts_with_it(self) -> None:
    self.assertEqual(pipeline.embedding_text(_document('Runbook', '## Purpose')), 'Runbook\n\n## Purpose')
    self.assertEqual(pipeline.embedding_text(_document('Domains', 'Domains\nBody')), 'Domains\nBody')
    self.assertEqual(pipeline.embedding_text(_document('general', '')), 'general')
    self.assertEqual(pipeline.embedding_text(_document('', 'Body')), 'Body')

  def test_the_title_counts_as_present_only_as_a_whole_word(self) -> None:
    # ERB Slack titles are channel names, and messages start with their speaker.
    self.assertEqual(pipeline.embedding_text(_document('devex', 'devex: quick ping')), 'devex: quick ping')
    self.assertEqual(pipeline.embedding_text(_document('Plan', 'Plan (draft) v2')), 'Plan (draft) v2')
    for speaker in ('support-alex', 'support_josh', 'supporter'):
      with self.subTest(speaker=speaker):
        text = f'{speaker}: customer reports 429s'
        self.assertEqual(pipeline.embedding_text(_document('support', text)), f'support\n\n{text}')


class PointTest(unittest.TestCase):
  def test_point_ids_are_stable_and_distinct_per_dataset_and_document(self) -> None:
    self.assertEqual(pipeline.point_id('erb-1', 'dsid_a'), pipeline.point_id('erb-1', 'dsid_a'))
    self.assertNotEqual(pipeline.point_id('erb-1', 'dsid_a'), pipeline.point_id('erb-1', 'dsid_a__2'))
    self.assertNotEqual(pipeline.point_id('erb-1', 'dsid_a'), pipeline.point_id('erb-2', 'dsid_a'))

  def test_the_payload_keeps_both_ids_the_dataset_the_text_and_the_benchmark_fields(self) -> None:
    document = KnowledgeDocument(
      document_id='dsid_x__2', source_document_id='dsid_x', title='T', text='B', metadata={'source_type': 'slack'}
    )

    self.assertEqual(
      pipeline.point_payload('erb-1', document),
      {
        'document_id': 'dsid_x__2',
        'source_document_id': 'dsid_x',
        'dataset': 'erb-1',
        'title': 'T',
        'text': 'B',
        'source_type': 'slack',
      },
    )


class RequestBatchesTest(unittest.TestCase):
  def test_a_request_carries_at_most_96_texts(self) -> None:
    self.assertEqual(
      request_batches(['short'] * 200, max_tokens=10_000), [range(0, 96), range(96, 192), range(192, 200)]
    )

  def test_a_request_stays_within_the_token_budget_and_a_long_text_goes_alone(self) -> None:
    # 30 characters are estimated at 11 tokens, 300 at 101.
    texts = ['a' * 30, 'b' * 30, 'c' * 30, 'd' * 300, 'e' * 30]

    self.assertEqual(request_batches(texts, max_tokens=25), [range(0, 2), range(2, 3), range(3, 4), range(4, 5)])


class FakeClock:
  def __init__(self) -> None:
    self.now = 0.0
    self.sleeps: list[float] = []

  async def sleep(self, seconds: float) -> None:
    self.sleeps.append(seconds)
    self.now += seconds


class TokenRateLimiterTest(unittest.IsolatedAsyncioTestCase):
  """A quota of 600 tokens a minute, 10 a second; requests may run 5 seconds, 50 tokens, ahead of that pace."""

  def setUp(self) -> None:
    self.clock = FakeClock()
    self.limiter = TokenRateLimiter(600, burst_seconds=5, clock=lambda: self.clock.now, sleep=self.clock.sleep)

  async def test_after_a_short_burst_requests_are_spaced_at_the_quota_pace(self) -> None:
    for _ in range(4):
      await self.limiter.reserve(30)

    # Two requests go at once; then each waits until the 30 tokens before it are paid for at 10 a second.
    self.assertEqual(self.clock.sleeps, [1.0, 3.0])

  async def test_over_time_the_tokens_flow_at_the_quota(self) -> None:
    for _ in range(100):
      await self.limiter.reserve(30)

    self.assertAlmostEqual(self.clock.now, (100 * 30 - 30 - 50) / 10)

  async def test_settling_at_the_billed_count_returns_the_overestimate(self) -> None:
    first = await self.limiter.reserve(30)
    await self.limiter.reserve(30)
    self.limiter.settle(first, 10)

    await self.limiter.reserve(30)

    self.assertEqual(self.clock.sleeps, [])

  async def test_a_waiting_request_checks_again_when_a_reservation_settles(self) -> None:
    async def never(seconds: float) -> None:
      await asyncio.Event().wait()

    limiter = TokenRateLimiter(600, burst_seconds=0, clock=lambda: 0.0, sleep=never)
    first = await limiter.reserve(60)
    waiting = asyncio.ensure_future(limiter.reserve(60))
    await asyncio.sleep(0)
    self.assertFalse(waiting.done())

    limiter.settle(first, 0)

    self.assertEqual((await asyncio.wait_for(waiting, 1)).tokens, 60)

  async def test_a_request_larger_than_the_burst_goes_alone_on_pace(self) -> None:
    await self.limiter.reserve(100)
    await self.limiter.reserve(1)

    self.assertEqual(self.clock.sleeps, [5.0])


def _connection_reset(request: httpx.Request) -> httpx.Response:
  raise httpx.ConnectError('connection reset', request=request)


def _embed_response(count: int, tokens: int = 7, dimension: int = DIMENSION) -> httpx.Response:
  return httpx.Response(
    200,
    json={
      'id': 'response-id',
      'embeddings': {'float': [[0.5] * dimension for _ in range(count)]},
      'texts': ['ignored'] * count,
      'meta': {'api_version': {'version': '2'}, 'billed_units': {'input_tokens': tokens}},
      'response_type': 'embeddings_by_type',
    },
  )


class CohereEmbeddingClientTest(unittest.IsolatedAsyncioTestCase):
  def setUp(self) -> None:
    self.requests: list[httpx.Request] = []
    self.responses: list[Callable[[httpx.Request], httpx.Response]] = []
    self.sleeps: list[float] = []

  async def asyncTearDown(self) -> None:
    if hasattr(self, 'client'):
      await self.client.aclose()

  def _client(self, **config: Any) -> CohereEmbeddingClient:
    def handle(request: httpx.Request) -> httpx.Response:
      self.requests.append(request)
      return self.responses.pop(0)(request)

    async def sleep(seconds: float) -> None:
      self.sleeps.append(seconds)

    settings = AzureFoundrySettings(base_url='https://foundry.example/', api_key='secret')
    self.client = CohereEmbeddingClient(_config(**config), settings, transport=httpx.MockTransport(handle), sleep=sleep)
    return self.client

  async def test_sends_a_cohere_embed_request_and_returns_the_vectors_and_billed_tokens(self) -> None:
    self.responses.append(lambda request: _embed_response(2, tokens=9))

    embeddings = await self._client().embed(['first', 'second'], 'search_document')

    self.assertEqual(embeddings, Embeddings(vectors=[[0.5] * DIMENSION] * 2, input_tokens=9))
    [request] = self.requests
    self.assertEqual(str(request.url), f'https://foundry.example{EMBED_PATH}')
    self.assertEqual(request.headers['api-key'], 'secret')
    self.assertEqual(
      json.loads(request.content),
      {
        'model': 'embed-test',
        'texts': ['first', 'second'],
        'input_type': 'search_document',
        'embedding_types': ['float'],
        'output_dimension': DIMENSION,
        'truncate': 'NONE',
      },
    )

  async def test_retries_throttling_and_server_errors_honouring_the_requested_wait(self) -> None:
    self.responses += [
      lambda request: httpx.Response(429, headers={'retry-after': '7'}, json={'error': {'message': 'Slow down'}}),
      lambda request: httpx.Response(503, text='unavailable'),
      _connection_reset,
      lambda request: _embed_response(1),
    ]

    with self.assertLogs('indexing.embedding', 'WARNING') as logs:
      embeddings = await self._client(max_retries=4).embed(['text'], 'search_query')

    self.assertEqual(len(embeddings.vectors), 1)
    self.assertEqual(len(self.requests), 4)
    self.assertEqual(self.sleeps[0], 7.0)
    self.assertEqual(len(self.sleeps), 3)
    self.assertIn('429 Slow down; retrying in 7.0 s', logs.output[0])
    self.assertIn('ConnectError: connection reset', logs.output[2])

  async def test_a_rejected_request_fails_at_once_with_the_api_message(self) -> None:
    self.responses.append(lambda request: httpx.Response(400, json={'id': 'x', 'message': 'invalid request: texts'}))

    with self.assertRaisesRegex(EmbeddingError, '400 invalid request: texts'):
      await self._client().embed(['text'], 'search_document')

    self.assertEqual((len(self.requests), self.sleeps), (1, []))

  async def test_gives_up_after_the_configured_attempts(self) -> None:
    self.responses += [lambda request: httpx.Response(500, text='boom')] * 3

    with self.assertLogs('indexing.embedding', 'WARNING'):
      with self.assertRaisesRegex(EmbeddingError, 'failed 3 times, last with 500 boom'):
        await self._client(max_retries=3).embed(['text'], 'search_document')

    self.assertEqual((len(self.requests), len(self.sleeps)), (3, 2))

  async def test_rejects_a_response_with_missing_or_misshapen_vectors(self) -> None:
    self.responses += [lambda request: _embed_response(1), lambda request: _embed_response(2, dimension=3)]
    client = self._client()

    for _ in range(2):
      with self.subTest(), self.assertRaisesRegex(EmbeddingError, 'Expected 2 embeddings of 4 values'):
        await client.embed(['a', 'b'], 'search_document')

  async def test_needs_the_foundry_url_and_key(self) -> None:
    with self.assertRaisesRegex(ValueError, 'AZURE_FOUNDRY_API_KEY'):
      CohereEmbeddingClient(_config(), AzureFoundrySettings(base_url='https://foundry.example', api_key=None))


def _pin(path: str, content: bytes) -> dict[str, Any]:
  return {'path': path, 'size': len(content), 'sha256': hashlib.sha256(content).hexdigest()}


class PipelineTest(unittest.IsolatedAsyncioTestCase):
  """Indexes a five-article WixQA knowledge base in shards of two documents."""

  def setUp(self) -> None:
    self._directory = tempfile.TemporaryDirectory()
    self.data_dir = Path(self._directory.name)
    (self.data_dir / 'wixqa' / 'raw').mkdir(parents=True)
    self._write_knowledge_base(_articles(5))
    self.index = self.data_dir / 'wixqa' / pipeline.INDEX_DIR

  def tearDown(self) -> None:
    self._directory.cleanup()

  def _write_knowledge_base(self, articles: list[dict[str, str]]) -> None:
    content = ''.join(json.dumps(article) + '\n' for article in articles).encode()
    (self.data_dir / 'wixqa' / 'raw' / 'kb.jsonl').write_bytes(content)
    # The questions file is pinned but never fetched: indexing reads only the knowledge base.
    self.source = BenchmarkSource.model_validate(
      {
        'dataset_name': 'wixqa-test',
        'expected_samples': 1,
        'base_url': f'https://example.org/wixqa/{"a" * 40}',
        'files': {'knowledge_base': _pin('kb.jsonl', content), 'expertwritten': _pin('q.jsonl', b'{}')},
      }
    )

  async def _embed(self, embedder: FakeEmbedder, config: EmbeddingConfig | None = None) -> pipeline.EmbedOutcome:
    return await pipeline.embed_knowledge_base(
      'wixqa', self.source, self.data_dir, embedder, config or _config(), shard_size=2
    )

  async def _load(self, store: FakeVectorStore) -> pipeline.LoadOutcome:
    return await pipeline.load_knowledge_base('wixqa', self.source, self.data_dir, store, _config(), batch_size=2)

  async def test_embeds_every_document_into_shards_and_reports_the_billed_tokens(self) -> None:
    progress: list[pipeline.ShardProgress] = []
    embedder = FakeEmbedder()

    outcome = await pipeline.embed_knowledge_base(
      'wixqa', self.source, self.data_dir, embedder, _config(), shard_size=2, progress=progress.append
    )

    self.assertEqual(outcome, pipeline.EmbedOutcome(5, 3, 3, 50, 50))
    self.assertEqual([(p.shard, p.documents, p.input_tokens) for p in progress], [(0, 2, 20), (1, 4, 40), (2, 5, 50)])
    shard = pq.read_table(pipeline.shard_path(self.index, 1))
    self.assertEqual(shard.column('document_id').to_pylist(), ['k2', 'k3'])
    self.assertEqual(shard.column('embedding').to_pylist(), [[2.0, 2.0, 1.0, 0.0], [3.0, 2.0, 1.0, 0.0]])
    # The NUL character is gone before embedding.
    self.assertEqual(embedder.calls[1], ['Article 2\nBody 2', 'Article 3\nBody 3'])
    report = json.loads((self.index / pipeline.REPORT_FILE).read_text(encoding='utf-8'))
    self.assertEqual((report['documents'], report['shards'], report['input_tokens']), (5, 3, 50))
    self.assertEqual(report['spec'], pipeline.index_spec('wixqa', self.source, _config()))

  async def test_an_interrupted_run_resumes_at_the_first_missing_shard(self) -> None:
    # Two documents fit one request, so the second call embeds the second shard.
    with self.assertRaises(EmbeddingError):
      await self._embed(FakeEmbedder(fail_on_call=2))
    self.assertTrue(pipeline.shard_path(self.index, 0).exists())
    self.assertFalse((self.index / pipeline.REPORT_FILE).exists())

    embedder = FakeEmbedder()
    outcome = await self._embed(embedder)

    self.assertEqual([call for texts in embedder.calls for call in texts][0], 'Article 2\nBody 2')
    self.assertEqual((outcome.embedded_shards, outcome.run_input_tokens, outcome.input_tokens), (2, 30, 50))
    # A complete index is only checked again.
    self.assertEqual((await self._embed(embedder)).embedded_shards, 0)

  async def test_refuses_to_resume_shards_embedded_with_another_spec(self) -> None:
    with self.assertRaises(EmbeddingError):
      await self._embed(FakeEmbedder(fail_on_call=2))

    with self.assertRaisesRegex(ValueError, 'another model, settings or source'):
      await self._embed(FakeEmbedder(), _config(output_dimension=8))

  async def test_renaming_the_dataset_or_repinning_other_files_keeps_the_index(self) -> None:
    await self._embed(FakeEmbedder())
    pins = self.source.model_dump(mode='json')
    self.source = BenchmarkSource.model_validate(
      {
        'dataset_name': 'wixqa-renamed',
        'expected_samples': 2,
        'base_url': f'https://example.org/wixqa/{"b" * 40}',
        'files': {**pins['files'], 'expertwritten': _pin('q.jsonl', b'[]')},
      }
    )

    outcome = await self._embed(FakeEmbedder())
    loaded = await self._load(FakeVectorStore())

    self.assertEqual(outcome.embedded_shards, 0)
    self.assertEqual(loaded, pipeline.LoadOutcome(collection='wixqa-renamed', points=5))

  async def test_the_spec_can_change_while_no_shard_is_on_disk(self) -> None:
    with self.assertRaises(EmbeddingError):
      await self._embed(FakeEmbedder(fail_on_call=1))

    outcome = await self._embed(FakeEmbedder(), _config(model='embed-other'))

    self.assertEqual(outcome.embedded_shards, 3)

  async def test_refuses_a_shard_that_holds_other_documents(self) -> None:
    await self._embed(FakeEmbedder())
    pipeline.shard_path(self.index, 0).replace(pipeline.shard_path(self.index, 1))

    with self.assertRaisesRegex(ValueError, 'holds other documents'):
      await self._embed(FakeEmbedder())

  async def test_refuses_to_run_while_another_run_embeds_the_same_index(self) -> None:
    with pipeline._exclusive(self.index), self.assertRaises(pipeline.IndexInProgressError):
      await self._embed(FakeEmbedder())

  async def test_a_failing_request_cancels_the_others(self) -> None:
    cancelled = asyncio.Event()

    class Embedder:
      async def embed(self, texts: Sequence[str], input_type: InputType) -> Embeddings:
        if texts[0].startswith('Article 0'):
          await asyncio.sleep(0)
          raise EmbeddingError('rejected')
        try:
          await asyncio.Event().wait()
        finally:
          cancelled.set()
        raise AssertionError('unreachable')

    # Each document exceeds a 10-token request budget, so every document is its own request.
    with self.assertRaisesRegex(EmbeddingError, 'rejected'):
      await pipeline.embed_knowledge_base(
        'wixqa', self.source, self.data_dir, Embedder(), _config(max_request_tokens=4), shard_size=2
      )

    self.assertTrue(cancelled.is_set())
    self.assertFalse(pipeline.shard_path(self.index, 0).exists())

  async def test_loads_every_document_with_its_vector_and_payload(self) -> None:
    await self._embed(FakeEmbedder())
    store = FakeVectorStore()

    outcome = await self._load(store)

    self.assertEqual(outcome, pipeline.LoadOutcome(collection='wixqa-test', points=5))
    # Every article has the four BM25 tokens 'article', its number, 'body' and its number.
    metadata: dict[str, JsonValue] = {
      'dataset': 'wixqa-test',
      'model': 'embed-test',
      'output_dimension': DIMENSION,
      'bm25_average_length': 4.0,
    }
    self.assertEqual(
      store.specs,
      {'wixqa-test': CollectionSpec('wixqa-test', DIMENSION, bm25_average_length=4.0, metadata=metadata)},
    )
    self.assertEqual(store.upserts, [2, 2, 1])
    point = store.points['wixqa-test'][pipeline.point_id('wixqa-test', 'k3')]
    self.assertEqual(list(point.vector), [3.0, 2.0, 1.0, 0.0])
    self.assertEqual(point.text, 'Article 3\nBody 3')
    self.assertEqual(
      point.payload,
      {
        'document_id': 'k3',
        'source_document_id': 'k3',
        'dataset': 'wixqa-test',
        'title': 'Article 3',
        'text': 'Article 3\nBody 3',
        'url': 'https://support.example/3',
        'article_type': 'article',
      },
    )

  async def test_vectors_stay_with_their_documents_across_requests_shards_and_upserts(self) -> None:
    self._write_knowledge_base(_articles(10))

    class Embedder:
      """Embeds an article as [its number, 0, 1, 0]; the requests sent last return first."""

      async def embed(self, texts: Sequence[str], input_type: InputType) -> Embeddings:
        [number] = [_article_number(text) for text in texts]
        for _ in range(10 - int(number)):
          await asyncio.sleep(0)
        return Embeddings(vectors=[[number, 0.0, 1.0, 0.0]], input_tokens=1)

    # Each text is estimated above 4 tokens, so every document is its own request.
    await pipeline.embed_knowledge_base(
      'wixqa', self.source, self.data_dir, Embedder(), _config(max_request_tokens=4), shard_size=4
    )
    store = FakeVectorStore()
    await pipeline.load_knowledge_base('wixqa', self.source, self.data_dir, store, _config(), batch_size=3)

    for shard in range(3):
      table = pq.read_table(pipeline.shard_path(self.index, shard))
      for document_id, vector in zip(table.column('document_id').to_pylist(), table.column('embedding').to_pylist()):
        self.assertEqual(vector[0], float(document_id[1:]), document_id)
    self.assertEqual(store.upserts, [3, 1, 3, 1, 2])
    points = list(store.points['wixqa-test'].values())
    self.assertEqual(len(points), 10)
    for point in points:
      self.assertEqual(point.vector[0], float(str(point.payload['document_id'])[1:]), point.payload['document_id'])

  async def test_loading_again_replaces_the_same_points(self) -> None:
    await self._embed(FakeEmbedder())
    store = FakeVectorStore()

    await self._load(store)
    outcome = await self._load(store)

    self.assertEqual((outcome.points, await store.count('wixqa-test')), (5, 5))

  async def test_load_refuses_an_incomplete_index(self) -> None:
    with self.assertRaises(EmbeddingError):
      await self._embed(FakeEmbedder(fail_on_call=2))

    with self.assertRaisesRegex(FileNotFoundError, 'no complete index'):
      await self._load(FakeVectorStore())

  async def test_load_refuses_an_index_embedded_with_another_spec(self) -> None:
    await self._embed(FakeEmbedder(), _config(model='embed-other'))

    with self.assertRaisesRegex(ValueError, 'another model, settings or source'):
      await self._load(FakeVectorStore())

  async def test_load_fails_when_the_collection_holds_other_points(self) -> None:
    await self._embed(FakeEmbedder())
    store = FakeVectorStore()
    await self._load(store)
    stale = IndexPoint(id=pipeline.point_id('wixqa-test', 'gone'), vector=[0.0] * DIMENSION, text='', payload={})
    await store.upsert(store.specs['wixqa-test'], [stale])

    with self.assertRaisesRegex(ValueError, 'holds 6 points, but the index has 5 documents'):
      await self._load(store)


if __name__ == '__main__':
  unittest.main()
