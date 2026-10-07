"""Search indexes of the ERB and WixQA knowledge bases: one point per document, never chunked.

The benchmarks label relevance per document, so each document is embedded whole and indexed under the id its gold
lists use. Indexing runs in two steps, and both are safe to run again:

- ``embed`` writes the embeddings to ``<data dir>/<benchmark>/index/``, in shards of consecutive documents. A shard is
  written once all its documents are embedded, so an interrupted run resumes at the first missing shard. The spec
  records what the shards are embedded with, and the report, written last, marks a complete index.
- ``load`` upserts every document with its embedding into the benchmark's collection. Point ids derive from the
  document ids, so loading again replaces the same points.
"""

import asyncio
import fcntl
import itertools
import json
import re
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from uuid import NAMESPACE_URL, UUID, uuid5

import pyarrow as pa
import pyarrow.parquet as pq
from pydantic import JsonValue

from benchmarks import erb, wixqa
from benchmarks.common import KnowledgeDocument
from benchmarks.download import verify_pinned_file
from config import BenchmarkSource, EmbeddingConfig
from indexing.embedding import DOCUMENT_INPUT_TYPE, Embedder, request_batches
from indexing.vector_store import CollectionSpec, IndexPoint, VectorStore


@dataclass(frozen=True, slots=True)
class KnowledgeBase:
  # The role of the pinned file that holds the documents, and the reader of that file.
  file: str
  read: Callable[[Mapping[str, Path]], Iterator[KnowledgeDocument]]


KNOWLEDGE_BASES = {
  'erb': KnowledgeBase(file='documents', read=erb.knowledge_base),
  'wixqa': KnowledgeBase(file='knowledge_base', read=wixqa.knowledge_base),
}
INDEX_DIR = 'index'
SPEC_FILE = 'spec.json'
REPORT_FILE = 'report.json'
SHARD_SIZE = 2_048
UPSERT_BATCH_SIZE = 256
# A point's id is the UUIDv5 of its dataset name and document id in this namespace.
POINT_NAMESPACE = uuid5(NAMESPACE_URL, 'https://github.com/SylloTips/syllo-eval/experiments/indexing')
# How ``embedding_text`` builds the embedded text; describe any change here, so that no index resumes across it.
EMBEDDED_TEXT = 'title, blank line, text; the text alone when it starts with the title as a whole word'


class IndexInProgressError(RuntimeError):
  """Raised when another run is already embedding the same knowledge base."""


@dataclass(frozen=True, slots=True)
class ShardProgress:
  shard: int
  # Documents of the shards done so far, and tokens billed by this run.
  documents: int
  input_tokens: int


@dataclass(frozen=True, slots=True)
class EmbedOutcome:
  documents: int
  shards: int
  # Shards this run embedded; the others were already on disk.
  embedded_shards: int
  # Tokens billed for the whole index, and by this run.
  input_tokens: int
  run_input_tokens: int


@dataclass(frozen=True, slots=True)
class LoadOutcome:
  collection: str
  points: int


def embedding_text(document: KnowledgeDocument) -> str:
  """The text the model embeds: the title, a blank line and the text, or the text alone if it starts with the title.

  The title must open the text as a whole word. Every WixQA article starts with its title, as do 709 ERB documents;
  an ERB Slack message from ``support-alex`` in channel ``support`` does not.
  """
  if not document.title or re.match(re.escape(document.title) + r'(?![\w-])', document.text):
    return document.text
  return f'{document.title}\n\n{document.text}' if document.text else document.title


def collection_name(source: BenchmarkSource) -> str:
  return source.dataset_name


def point_id(dataset_name: str, document_id: str) -> UUID:
  return uuid5(POINT_NAMESPACE, f'{dataset_name}/{document_id}')


def point_payload(dataset_name: str, document: KnowledgeDocument) -> dict[str, JsonValue]:
  """The fields stored with a document's vector: its ids, dataset, title and text, then its benchmark fields."""
  return {
    'document_id': document.document_id,
    'source_document_id': document.source_document_id,
    'dataset': dataset_name,
    'title': document.title,
    'text': document.text,
    **document.metadata,
  }


def index_spec(name: str, source: BenchmarkSource, config: EmbeddingConfig) -> dict[str, JsonValue]:
  """What the embeddings depend on: the knowledge-base file, the model, its settings and the embedded text.

  The dataset name and the other pinned files are left out: the shards hold only document ids and vectors.
  """
  return {
    'benchmark': name,
    'file': source.files[KNOWLEDGE_BASES[name].file].model_dump(mode='json'),
    'model': config.model,
    'output_dimension': config.output_dimension,
    'input_type': DOCUMENT_INPUT_TYPE,
    'embedded_text': EMBEDDED_TEXT,
  }


async def embed_knowledge_base(
  name: str,
  source: BenchmarkSource,
  data_dir: Path,
  embedder: Embedder,
  config: EmbeddingConfig,
  *,
  shard_size: int = SHARD_SIZE,
  progress: Callable[[ShardProgress], None] | None = None,
) -> EmbedOutcome:
  """Embed the documents of every shard not yet on disk, then write the report.

  Shards on disk are kept only if they were embedded with the same spec and hold the same documents.
  """
  directory = data_dir / name / INDEX_DIR
  with _exclusive(directory):
    knowledge_base = _read_knowledge_base(name, source, data_dir)
    spec = index_spec(name, source, config)
    _record_spec(directory, spec)
    (directory / REPORT_FILE).unlink(missing_ok=True)
    documents = shards = embedded_shards = input_tokens = run_input_tokens = 0
    for shard, batch in enumerate(itertools.batched(knowledge_base, shard_size)):
      path = shard_path(directory, shard)
      document_ids = [document.document_id for document in batch]
      shards += 1
      documents += len(batch)
      if path.exists():
        input_tokens += _check_shard(path, document_ids)
        continue
      vectors, tokens = await _embed_documents(embedder, batch, config.max_request_tokens)
      _write_shard(path, document_ids, vectors, tokens)
      embedded_shards += 1
      input_tokens += tokens
      run_input_tokens += tokens
      if progress is not None:
        progress(ShardProgress(shard=shard, documents=documents, input_tokens=run_input_tokens))
    report = {'spec': spec, 'documents': documents, 'shards': shards, 'input_tokens': input_tokens}
    (directory / REPORT_FILE).write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
  return EmbedOutcome(
    documents=documents,
    shards=shards,
    embedded_shards=embedded_shards,
    input_tokens=input_tokens,
    run_input_tokens=run_input_tokens,
  )


async def load_knowledge_base(
  name: str,
  source: BenchmarkSource,
  data_dir: Path,
  store: VectorStore,
  config: EmbeddingConfig,
  *,
  batch_size: int = UPSERT_BATCH_SIZE,
) -> LoadOutcome:
  """Upsert every embedded document into the benchmark's collection, which must then hold exactly these points."""
  directory = data_dir / name / INDEX_DIR
  if not (directory / REPORT_FILE).exists():
    raise FileNotFoundError(f'{name} has no complete index; run `syllo-exp index embed --only {name}`')
  report = json.loads((directory / REPORT_FILE).read_text(encoding='utf-8'))
  if report['spec'] != index_spec(name, source, config):
    raise ValueError(f'{name} was embedded with another model, settings or source; delete {directory} and embed again')
  documents = _read_knowledge_base(name, source, data_dir)
  collection = collection_name(source)
  await store.ensure_collection(CollectionSpec(name=collection, dimension=config.output_dimension))

  points = 0
  for shard in range(report['shards']):
    table = pq.read_table(shard_path(directory, shard))
    batch = list(itertools.islice(documents, table.num_rows))
    if [document.document_id for document in batch] != table.column('document_id').to_pylist():
      raise ValueError(f'{shard_path(directory, shard)} does not match the knowledge base; delete {directory}')
    vectors = table.column('embedding').to_pylist()
    for start in range(0, len(batch), batch_size):
      await store.upsert(
        collection,
        [
          IndexPoint(
            id=point_id(source.dataset_name, document.document_id),
            vector=vector,
            payload=point_payload(source.dataset_name, document),
          )
          for document, vector in zip(batch[start : start + batch_size], vectors[start : start + batch_size])
        ],
      )
    points += len(batch)
  if next(documents, None) is not None:
    raise ValueError(f'The knowledge base of {name} has more documents than its index; delete {directory}')

  stored = await store.count(collection)
  if stored != points:
    raise ValueError(f'Collection {collection} holds {stored} points, but the index has {points} documents')
  return LoadOutcome(collection=collection, points=points)


def shard_path(directory: Path, shard: int) -> Path:
  return directory / f'part-{shard:05d}.parquet'


def _read_knowledge_base(name: str, source: BenchmarkSource, data_dir: Path) -> Iterator[KnowledgeDocument]:
  """The documents of the knowledge-base file, once the file is checked against its pin."""
  knowledge_base = KNOWLEDGE_BASES[name]
  pinned = source.files[knowledge_base.file]
  path = data_dir / name / 'raw' / pinned.path
  verify_pinned_file(pinned, path)
  return knowledge_base.read({knowledge_base.file: path})


def _record_spec(directory: Path, spec: Mapping[str, Any]) -> None:
  """Write the spec, refusing to change it while shards embedded under the previous one are on disk."""
  path = directory / SPEC_FILE
  if path.exists() and json.loads(path.read_text(encoding='utf-8')) != spec and any(directory.glob('part-*.parquet')):
    raise ValueError(
      f'{directory} holds embeddings made with another model, settings or source; delete it to start over'
    )
  path.write_text(json.dumps(spec, ensure_ascii=False, indent=2), encoding='utf-8')


async def _embed_documents(
  embedder: Embedder, documents: Sequence[KnowledgeDocument], max_request_tokens: int
) -> tuple[list[list[float]], int]:
  """Embed the documents with concurrent requests; if one fails, the others are cancelled."""
  texts = [embedding_text(document) for document in documents]
  requests = [
    asyncio.ensure_future(embedder.embed(texts[batch.start : batch.stop], DOCUMENT_INPUT_TYPE))
    for batch in request_batches(texts, max_request_tokens)
  ]
  try:
    results = await asyncio.gather(*requests)
  except BaseException:
    for request in requests:
      request.cancel()
    await asyncio.gather(*requests, return_exceptions=True)
    raise
  return [vector for result in results for vector in result.vectors], sum(result.input_tokens for result in results)


def _write_shard(
  path: Path, document_ids: Sequence[str], vectors: Sequence[Sequence[float]], input_tokens: int
) -> None:
  """Write a shard with the tokens billed for it, through a temporary file so that no shard is ever partial."""
  values = pa.array([value for vector in vectors for value in vector], type=pa.float32())
  table = pa.table(
    {
      'document_id': pa.array(document_ids, type=pa.string()),
      'embedding': pa.FixedSizeListArray.from_arrays(values, len(vectors[0])),
    }
  ).replace_schema_metadata({'input_tokens': str(input_tokens)})
  partial = path.with_name(path.name + '.part')
  # Dictionary pages would add half the size of the vectors and save nothing on unique ids.
  pq.write_table(table, partial, use_dictionary=False)
  partial.replace(path)


def _check_shard(path: Path, document_ids: Sequence[str]) -> int:
  """The tokens billed for a shard on disk, after checking that it holds these documents."""
  if pq.read_table(path, columns=['document_id']).column('document_id').to_pylist() != list(document_ids):
    raise ValueError(f'{path} holds other documents than the knowledge base has there; delete it to embed them again')
  return int(pq.read_schema(path).metadata[b'input_tokens'])


@contextmanager
def _exclusive(directory: Path) -> Iterator[None]:
  """Hold an exclusive lock on the index, so two runs never embed the same shards."""
  directory.mkdir(parents=True, exist_ok=True)
  with (directory / '.lock').open('a') as lock:
    try:
      fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
      raise IndexInProgressError(f'Another run is already embedding {directory}') from None
    yield
