import json
import statistics
import tempfile
import unittest
from collections.abc import Sequence
from pathlib import Path

from fastmcp import Client

from indexing.embedding import EmbeddingError, Embeddings, InputType
from indexing.vector_store import StoredPoint
from search_tool.search import RESULTS, TRUNCATION_MARK, CallLog, KnowledgeBaseSearch, RandomSwap, SearchError
from search_tool.server import TOOL_NAME, build_server

POINTS = [
  StoredPoint(
    id=f'p{index:02d}',
    payload={'document_id': f'doc{index:02d}', 'title': f'Title {index}', 'text': f'Text of document {index}'},
  )
  for index in range(40)
]
POINT_IDS = [point.id for point in POINTS]


class FakeIndex:
  """Ranks the points in their order."""

  def __init__(self) -> None:
    self.searches: list[tuple[list[float], str, int]] = []
    self.retrievals: list[list[str]] = []

  async def search(self, vector: Sequence[float], query: str, limit: int) -> list[StoredPoint]:
    self.searches.append((list(vector), query, limit))
    return POINTS[:limit]

  async def retrieve(self, ids: Sequence[str]) -> list[StoredPoint]:
    self.retrievals.append(list(ids))
    by_id = {point.id: point for point in POINTS}
    return [by_id[point_id] for point_id in ids]


class FakeEmbedder:
  def __init__(self, error: Exception | None = None) -> None:
    self.calls: list[tuple[list[str], InputType]] = []
    self._error = error

  async def embed(self, texts: Sequence[str], input_type: InputType) -> Embeddings:
    self.calls.append((list(texts), input_type))
    if self._error is not None:
      raise self._error
    return Embeddings(vectors=[[0.25, 0.75]], input_tokens=3)


class RandomSwapTest(unittest.TestCase):
  def test_a_fraction_of_a_half_always_replaces_five_of_the_ten_results(self) -> None:
    swap = RandomSwap(0.5, POINT_IDS)

    self.assertEqual({len(swap.replacements(f'query {n}', POINT_IDS[:RESULTS])) for n in range(200)}, {5})

  def test_a_quarter_replaces_two_or_three_results_and_a_quarter_on_average(self) -> None:
    swap = RandomSwap(0.25, POINT_IDS)

    counts = [len(swap.replacements(f'query {n}', POINT_IDS[:RESULTS])) for n in range(4_000)]

    self.assertEqual(set(counts), {2, 3})
    self.assertAlmostEqual(statistics.mean(counts) / RESULTS, 0.25, delta=0.005)

  def test_random_documents_are_new_distinct_and_take_any_rank(self) -> None:
    swap = RandomSwap(0.5, POINT_IDS)
    positions: set[int] = set()

    for n in range(300):
      replacements = swap.replacements(f'query {n}', POINT_IDS[:RESULTS])
      positions.update(replacements)
      self.assertTrue(set(replacements.values()).isdisjoint(POINT_IDS[:RESULTS]))
      self.assertEqual(len(set(replacements.values())), len(replacements))

    self.assertEqual(positions, set(range(RESULTS)))

  def test_the_draws_depend_only_on_the_seed_and_the_query(self) -> None:
    queries = [f'query {n}' for n in range(20)]
    first = [RandomSwap(0.5, POINT_IDS, seed=1).replacements(query, POINT_IDS[:RESULTS]) for query in queries]
    again = [RandomSwap(0.5, POINT_IDS, seed=1).replacements(query, POINT_IDS[:RESULTS]) for query in queries]
    other = [RandomSwap(0.5, POINT_IDS, seed=2).replacements(query, POINT_IDS[:RESULTS]) for query in queries]

    self.assertEqual(first, again)
    self.assertNotEqual(first, other)

  def test_rejects_fractions_outside_zero_and_one_and_a_knowledge_base_too_small_to_draw_from(self) -> None:
    for fraction in (0.0, 1.0, -0.25):
      with self.subTest(fraction=fraction), self.assertRaisesRegex(ValueError, 'above 0 and below 1'):
        RandomSwap(fraction, POINT_IDS)
    with self.assertRaisesRegex(ValueError, 'at least 20 documents'):
      RandomSwap(0.5, POINT_IDS[:19])


class KnowledgeBaseSearchTest(unittest.IsolatedAsyncioTestCase):
  async def test_returns_the_ten_best_documents_for_the_embedded_query(self) -> None:
    index, embedder = FakeIndex(), FakeEmbedder()

    found = await KnowledgeBaseSearch(index, embedder).search('How do refunds work?')

    self.assertEqual(embedder.calls, [(['How do refunds work?'], 'search_query')])
    self.assertEqual(index.searches, [([0.25, 0.75], 'How do refunds work?', RESULTS)])
    self.assertEqual([result.rank for result in found.results], list(range(1, RESULTS + 1)))
    self.assertEqual(
      found.results[2].model_dump(), {'rank': 3, 'id': 'doc02', 'title': 'Title 2', 'text': 'Text of document 2'}
    )
    self.assertEqual(index.retrievals, [])

  async def test_each_search_gets_its_own_id_and_echoes_the_query(self) -> None:
    search = KnowledgeBaseSearch(FakeIndex(), FakeEmbedder())

    first, second = await search.search('refunds'), await search.search('refunds')

    self.assertEqual((first.query, len(first.call_id)), ('refunds', 32))
    self.assertNotEqual(first.call_id, second.call_id)
    self.assertEqual(first.results, second.results)

  async def test_the_call_log_records_what_each_search_returned_and_replaced(self) -> None:
    with tempfile.TemporaryDirectory() as directory:
      log = CallLog(Path(directory) / 'calls' / 'kb.jsonl', {'collection': 'kb', 'swap_fraction': 0.5})
      search = KnowledgeBaseSearch(FakeIndex(), FakeEmbedder(), swap=RandomSwap(0.5, POINT_IDS), call_log=log)

      found = await search.search('refunds')
      [line] = log.path.read_text(encoding='utf-8').splitlines()

    record = json.loads(line)
    self.assertEqual(
      {key: record[key] for key in ('collection', 'swap_fraction', 'call_id', 'query')},
      {'collection': 'kb', 'swap_fraction': 0.5, 'call_id': found.call_id, 'query': 'refunds'},
    )
    self.assertEqual(record['ids'], [result.id for result in found.results])
    self.assertEqual(len(record['random_ranks']), 5)
    self.assertEqual(record['replaced_ids'], [f'doc{rank - 1:02d}' for rank in record['random_ranks']])
    self.assertTrue(set(record['replaced_ids']).isdisjoint(record['ids']))

  async def test_a_failed_search_is_logged_with_its_error_and_raised_as_a_search_error(self) -> None:
    failing = FakeEmbedder(EmbeddingError('The embedding request was rejected: 401 denied'))
    with tempfile.TemporaryDirectory() as directory:
      log = CallLog(Path(directory) / 'kb.jsonl', {'collection': 'kb'})
      for embedder, query, error in (
        (failing, 'refunds', 'The embedding request was rejected: 401 denied'),
        (FakeEmbedder(), ' ', 'the query is empty'),
      ):
        with self.subTest(query=query), self.assertRaisesRegex(SearchError, error):
          await KnowledgeBaseSearch(FakeIndex(), embedder, call_log=log).search(query)
      records = [json.loads(line) for line in log.path.read_text(encoding='utf-8').splitlines()]

    self.assertEqual(
      [(record['collection'], record['query'], record['error']) for record in records],
      [('kb', 'refunds', 'The embedding request was rejected: 401 denied'), ('kb', ' ', 'the query is empty')],
    )
    self.assertTrue(all(len(record['call_id']) == 32 and 'ids' not in record for record in records))

  async def test_random_documents_take_the_ranks_the_swap_chose(self) -> None:
    index = FakeIndex()
    swap = RandomSwap(0.5, POINT_IDS, seed=7)
    replacements = RandomSwap(0.5, POINT_IDS, seed=7).replacements('refunds', POINT_IDS[:RESULTS])

    found = await KnowledgeBaseSearch(index, FakeEmbedder(), swap=swap).search('refunds')

    self.assertEqual(index.retrievals, [list(replacements.values())])
    expected = [replacements.get(position, POINT_IDS[position]) for position in range(RESULTS)]
    self.assertEqual([result.id for result in found.results], [f'doc{point_id[1:]}' for point_id in expected])
    self.assertEqual([result.rank for result in found.results], list(range(1, RESULTS + 1)))

  async def test_long_documents_are_cut_with_a_mark(self) -> None:
    found = await KnowledgeBaseSearch(FakeIndex(), FakeEmbedder(), max_document_chars=7).search('refunds')

    self.assertEqual(found.results[0].text, 'Text of' + TRUNCATION_MARK)


class ServerTest(unittest.IsolatedAsyncioTestCase):
  def _client(self, embedder: FakeEmbedder | None = None) -> Client:
    return Client(build_server(KnowledgeBaseSearch(FakeIndex(), embedder or FakeEmbedder())))

  async def test_offers_one_read_only_tool_that_takes_a_query(self) -> None:
    async with self._client() as client:
      [tool] = await client.list_tools()

    self.assertEqual(tool.name, TOOL_NAME)
    self.assertEqual(list(tool.inputSchema['properties']), ['query'])
    self.assertEqual(tool.inputSchema['required'], ['query'])
    assert tool.annotations is not None
    self.assertTrue(tool.annotations.readOnlyHint)

  async def test_returns_the_results_as_json_text_and_as_structured_content(self) -> None:
    async with self._client() as client:
      result = await client.call_tool(TOOL_NAME, {'query': 'refunds'})

    assert result.structured_content is not None
    self.assertEqual(list(result.structured_content), ['call_id', 'query', 'results'])
    self.assertEqual(result.structured_content['query'], 'refunds')
    results = result.structured_content['results']
    self.assertEqual([item['id'] for item in results], [f'doc{index:02d}' for index in range(RESULTS)])
    self.assertEqual(results[0], {'rank': 1, 'id': 'doc00', 'title': 'Title 0', 'text': 'Text of document 0'})
    [content] = result.content
    self.assertEqual(json.loads(getattr(content, 'text')), result.structured_content)

  async def test_an_empty_query_and_a_failed_search_are_tool_errors(self) -> None:
    failing = FakeEmbedder(EmbeddingError('The embedding request failed 8 times, last with 503 busy'))
    for embedder, query, message in (
      (None, '  ', 'The search failed: the query is empty'),
      (failing, 'refunds', 'The search failed: The embedding request failed 8 times, last with 503 busy'),
    ):
      async with self._client(embedder) as client:
        result = await client.call_tool(TOOL_NAME, {'query': query}, raise_on_error=False)

      with self.subTest(query=query):
        self.assertTrue(result.is_error)
        self.assertIn(message, getattr(result.content[0], 'text'))


if __name__ == '__main__':
  unittest.main()
