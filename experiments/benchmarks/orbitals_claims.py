import json
from collections.abc import Awaitable, Callable, Sequence
from pathlib import Path

import httpx
from pydantic import BaseModel, JsonValue, TypeAdapter

from syllo_eval.datasets.model import DatasetJsonPayload
from syllo_eval.evaluation.claim_extractor import ExtractedClaim
from syllo_eval.settings import OrbitalsSettings

from benchmarks.common import Claim
from config import ClaimExtractorConfig

CLAIMS_KEY = 'expected_claims_orbitals'
CLAIMS_FILE = 'orbitals_claims.json'

Conversation = list[dict[str, str]]


class _Extractions(BaseModel):
  claims: list[ExtractedClaim]


class Extraction(BaseModel):
  extractions: _Extractions
  model: str


_EXTRACTIONS = TypeAdapter(list[Extraction])


def batch_extractor(
  client: httpx.AsyncClient, settings: OrbitalsSettings, config: ClaimExtractorConfig, description: str
) -> Callable[[list[Conversation]], Awaitable[list[Extraction]]]:
  async def extract_batch(conversations: list[Conversation]) -> list[Extraction]:
    response = await client.post(
      f'{settings.base_url.rstrip("/")}/orbitals/claim-extractor/batch-extract',
      headers={'X-API-Key': settings.api_key or ''},
      json={'conversations': conversations, 'model': config.model, 'ai_service_description': description},
      timeout=config.timeout_seconds,
    )
    response.raise_for_status()
    return _EXTRACTIONS.validate_python(response.json())

  return extract_batch


async def extract(
  payload: DatasetJsonPayload,
  records: Sequence[dict[str, JsonValue]],
  path: Path,
  extract_batch: Callable[[list[Conversation]], Awaitable[list[Extraction]]],
  batch_size: int,
) -> dict[str, int]:
  extracted = json.loads(path.read_text(encoding='utf-8')) if path.exists() else {}
  pending = [(s, r) for s, r in zip(payload.samples, records, strict=True) if s.input_prompt not in extracted]
  for start in range(0, len(pending), batch_size):
    batch = pending[start : start + batch_size]
    results = await extract_batch(
      [
        [
          {'role': 'user', 'content': sample.input_prompt},
          {'role': 'assistant', 'content': sample.ground_truth_output or ''},
        ]
        for sample, _ in batch
      ]
    )
    if len(results) != len(batch):
      raise ValueError(f'Orbitals returned {len(results)} extractions for {len(batch)} conversations')
    for (sample, record), result in zip(batch, results):
      claims = result.extractions.claims
      kept = [claim for claim in claims if claim.subtype.lower() != 'unverifiable']
      extracted[sample.input_prompt] = {
        'model': result.model,
        'excluded_unverifiable': len(claims) - len(kept),
        'claims': [
          {'id': f'{record["sample_key"]}-c{index:02d}', 'text': claim.content, 'subtype': claim.subtype}
          for index, claim in enumerate(kept, 1)
        ],
      }
    partial = path.with_suffix('.partial')
    partial.write_text(json.dumps(extracted, ensure_ascii=False, indent=2), encoding='utf-8')
    partial.replace(path)
  return {
    'samples': len(extracted),
    'claims': sum(len(entry['claims']) for entry in extracted.values()),
    'excluded_unverifiable': sum(entry['excluded_unverifiable'] for entry in extracted.values()),
  }


def read(directory: Path) -> dict[str, dict[str, list[Claim]]]:
  path = directory / CLAIMS_FILE
  if not path.exists():
    return {}
  extracted = json.loads(path.read_text(encoding='utf-8'))
  claims = {
    prompt: [Claim(id=claim['id'], text=claim['text']) for claim in entry['claims']]
    for prompt, entry in extracted.items()
    if entry['claims']
  }
  return {CLAIMS_KEY: claims}
