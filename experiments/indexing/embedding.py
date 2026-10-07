"""Embeddings from Cohere's v2 embed API, served by an Azure AI Foundry deployment.

Foundry serves Cohere's embedding models only on the provider's own route: its model-inference and OpenAI-compatible
embedding routes answer ``api_not_supported``. A request carries at most 96 texts. Texts are never truncated: a text
longer than the model's context fails its request instead.
"""

import asyncio
import logging
import random
import time
from collections.abc import AsyncIterator, Awaitable, Callable, Sequence
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Literal, Protocol

import httpx
from pydantic import BaseModel, Field, ValidationError
from pydantic_settings import SettingsConfigDict

from syllo_eval.settings import EnvSettings

from config import EmbeddingConfig

EMBED_PATH = '/providers/cohere/v2/embed'
MAX_TEXTS_PER_REQUEST = 96
InputType = Literal['search_document', 'search_query']
DOCUMENT_INPUT_TYPE: InputType = 'search_document'
QUERY_INPUT_TYPE: InputType = 'search_query'

# Cohere's tokenizer averages 3.7 characters per token on ERB, so 3 rarely estimates below the billed count.
_CHARS_PER_TOKEN = 3
# How far requests may run ahead of the quota's even pace, in seconds of tokens.
_BURST_SECONDS = 2.0
_RETRYABLE_STATUSES = frozenset({408, 429, 500, 502, 503, 504})
_BACKOFF_SECONDS = 2.0
_MAX_BACKOFF_SECONDS = 60.0

Sleep = Callable[[float], Awaitable[None]]

logger = logging.getLogger(__name__)


class AzureFoundrySettings(EnvSettings):
  model_config = SettingsConfigDict(env_prefix='AZURE_FOUNDRY_')

  base_url: str | None = None
  api_key: str | None = None


class EmbeddingError(RuntimeError):
  """A request that failed for good: rejected by the API, or still failing after every retry."""


@dataclass(frozen=True, slots=True)
class Embeddings:
  vectors: list[list[float]]
  # Tokens the API billed for the request.
  input_tokens: int


class Embedder(Protocol):
  async def embed(self, texts: Sequence[str], input_type: InputType) -> Embeddings: ...


def estimate_tokens(text: str) -> int:
  return len(text) // _CHARS_PER_TOKEN + 1


def request_batches(texts: Sequence[str], max_tokens: int) -> list[range]:
  """Split ``texts`` into consecutive requests of at most 96 texts and ``max_tokens`` estimated tokens.

  A text estimated above ``max_tokens`` is sent alone.
  """
  batches: list[range] = []
  start, tokens = 0, 0
  for index, text in enumerate(texts):
    cost = estimate_tokens(text)
    if index > start and (index - start == MAX_TEXTS_PER_REQUEST or tokens + cost > max_tokens):
      batches.append(range(start, index))
      start, tokens = index, 0
    tokens += cost
  if start < len(texts):
    batches.append(range(start, len(texts)))
  return batches


@dataclass(slots=True)
class Reservation:
  # Estimated when the request is sent, and settled to the billed count once it returns.
  tokens: int


class TokenRateLimiter:
  """Spaces requests so that their tokens flow at the deployment's quota, ``tokens_per_minute / 60`` a second.

  Azure enforces the quota over windows of seconds, so a burst is throttled even when the minute's total is within the
  quota. A request goes once the requests before it are at most ``burst_seconds`` ahead of the even pace, so the
  tokens sent run ahead by at most that plus one request. Settling a request at its billed count pulls the pace back
  by the tokens it was overestimated by; a waiting request checks again then.
  """

  def __init__(
    self,
    tokens_per_minute: int,
    *,
    burst_seconds: float = _BURST_SECONDS,
    clock: Callable[[], float] = time.monotonic,
    sleep: Sleep = asyncio.sleep,
  ):
    self._rate = tokens_per_minute / 60
    self._burst = burst_seconds
    self._clock = clock
    self._sleep = sleep
    # When the tokens reserved so far are paid for at the even pace.
    self._paid_at = clock()
    self._lock = asyncio.Lock()
    self._settled = asyncio.Event()

  async def reserve(self, tokens: int) -> Reservation:
    """Wait until ``tokens`` keep the requests on pace, then count them."""
    async with self._lock:
      while True:
        now = self._clock()
        wait = self._paid_at - self._burst - now
        if wait <= 0:
          self._paid_at = max(self._paid_at, now) + tokens / self._rate
          return Reservation(tokens=tokens)
        self._settled.clear()
        pace = asyncio.ensure_future(self._sleep(wait))
        settled = asyncio.ensure_future(self._settled.wait())
        try:
          await asyncio.wait((pace, settled), return_when=asyncio.FIRST_COMPLETED)
        finally:
          pace.cancel()
          settled.cancel()

  def settle(self, reservation: Reservation, tokens: int) -> None:
    """Count a returned request at its billed tokens, which may let a waiting one go sooner."""
    self._paid_at -= (reservation.tokens - tokens) / self._rate
    reservation.tokens = tokens
    self._settled.set()


class _BilledUnits(BaseModel):
  input_tokens: int = Field(ge=0)


class _Meta(BaseModel):
  billed_units: _BilledUnits


class _Vectors(BaseModel):
  values: list[list[float]] = Field(alias='float')


class _EmbedResponse(BaseModel):
  embeddings: _Vectors
  meta: _Meta


class CohereEmbeddingClient:
  """Embeds texts within the deployment's quota, retrying rate limits and transient failures."""

  def __init__(
    self,
    config: EmbeddingConfig,
    settings: AzureFoundrySettings,
    *,
    transport: httpx.AsyncBaseTransport | None = None,
    sleep: Sleep = asyncio.sleep,
  ):
    if settings.base_url is None or settings.api_key is None:
      raise ValueError('The embedding model needs AZURE_FOUNDRY_BASE_URL and AZURE_FOUNDRY_API_KEY')
    self._config = config
    self._sleep = sleep
    self._limiter = TokenRateLimiter(config.tokens_per_minute, sleep=sleep)
    self._requests = asyncio.Semaphore(config.max_concurrent_requests)
    self._client = httpx.AsyncClient(
      base_url=settings.base_url.rstrip('/'),
      headers={'api-key': settings.api_key},
      timeout=config.timeout_seconds,
      transport=transport,
    )

  async def embed(self, texts: Sequence[str], input_type: InputType) -> Embeddings:
    if not 0 < len(texts) <= MAX_TEXTS_PER_REQUEST:
      raise ValueError(f'A request carries 1 to {MAX_TEXTS_PER_REQUEST} texts, got {len(texts)}')
    body = {
      'model': self._config.model,
      'texts': list(texts),
      'input_type': input_type,
      'embedding_types': ['float'],
      'output_dimension': self._config.output_dimension,
      'truncate': 'NONE',
    }
    estimate = sum(estimate_tokens(text) for text in texts)
    async with self._requests:
      attempt = 0
      while True:
        attempt += 1
        reservation = await self._limiter.reserve(estimate)
        delay: float | None = None
        try:
          response = await self._client.post(EMBED_PATH, json=body)
        except httpx.TransportError as error:
          failure = f'{type(error).__name__}: {error}'
        else:
          if response.is_success:
            embeddings = self._parse(response, len(texts))
            self._limiter.settle(reservation, embeddings.input_tokens)
            return embeddings
          failure = f'{response.status_code} {_error_message(response)}'
          if response.status_code not in _RETRYABLE_STATUSES:
            raise EmbeddingError(f'The embedding request to {response.request.url} was rejected: {failure}')
          delay = _retry_after(response)
        if attempt == self._config.max_retries:
          raise EmbeddingError(f'The embedding request failed {attempt} times, last with {failure}')
        wait = delay if delay is not None else _backoff(attempt)
        logger.warning('Embedding request failed with %s; retrying in %.1f s', failure, wait)
        await self._sleep(wait)

  def _parse(self, response: httpx.Response, count: int) -> Embeddings:
    try:
      parsed = _EmbedResponse.model_validate_json(response.content)
    except ValidationError as error:
      raise EmbeddingError(f'Unexpected embedding response: {error}') from error
    vectors = parsed.embeddings.values
    dimension = self._config.output_dimension
    if len(vectors) != count or any(len(vector) != dimension for vector in vectors):
      raise EmbeddingError(f'Expected {count} embeddings of {dimension} values, got {len(vectors)}')
    return Embeddings(vectors=vectors, input_tokens=parsed.meta.billed_units.input_tokens)

  async def aclose(self) -> None:
    await self._client.aclose()


@asynccontextmanager
async def open_embedding_client(
  config: EmbeddingConfig, settings: AzureFoundrySettings
) -> AsyncIterator[CohereEmbeddingClient]:
  client = CohereEmbeddingClient(config, settings)
  try:
    yield client
  finally:
    await client.aclose()


def _error_message(response: httpx.Response) -> str:
  """The message of an error response: Cohere's ``message``, or Azure's ``error.message``."""
  try:
    body = response.json()
  except ValueError:
    body = None
  if isinstance(body, dict):
    error = body.get('error')
    message = error.get('message') if isinstance(error, dict) else body.get('message')
    if message:
      return str(message)
  return response.text[:500]


def _retry_after(response: httpx.Response) -> float | None:
  """The wait a throttled response asks for, in seconds, when it names one."""
  for header, seconds in (('retry-after-ms', 0.001), ('retry-after', 1.0)):
    try:
      return float(response.headers[header]) * seconds
    except (KeyError, ValueError):
      continue
  return None


def _backoff(attempt: int) -> float:
  return min(_MAX_BACKOFF_SECONDS, _BACKOFF_SECONDS * 2 ** (attempt - 1)) * random.uniform(0.5, 1.0)
