from typing import Any

import httpx
from pydantic import BaseModel, ValidationError

from syllo_eval.evaluation.claim_extractor import ClaimExtractionResult, ClaimExtractorMessage, ExtractedClaim
from syllo_eval.infrastructure.exceptions import ConfigurationError, DataMappingError, ExternalServiceError
from syllo_eval.settings import OrbitalsSettings


class _OrbitalsExtractions(BaseModel):
  claims: list[ExtractedClaim]


class _OrbitalsResponse(BaseModel):
  extractions: _OrbitalsExtractions
  model: str
  usage: dict[str, int] | None = None
  time_taken: float | None = None


class OrbitalsClaimExtractorClient:
  """HTTP client for Orbitals ClaimExtractor."""

  _EXTRACT_PATH = '/orbitals/claim-extractor/extract'

  def __init__(self, config: OrbitalsSettings, client: httpx.AsyncClient | None = None):
    api_key = config.api_key
    if api_key is None:
      raise ConfigurationError('orbitals', 'api_key', 'must be configured')

    self._config = config
    self._api_key = api_key
    self._client = client or httpx.AsyncClient(timeout=config.timeout_seconds)
    self._owns_client = client is None

  async def extract(self, conversation: list[ClaimExtractorMessage]) -> ClaimExtractionResult:
    body: dict[str, Any] = {
      'conversation': [message.model_dump() for message in conversation],
      'model': self._config.claim_extractor_model,
    }
    if self._config.ai_service_description is not None:
      body['ai_service_description'] = self._config.ai_service_description

    try:
      response = await self._client.post(
        f'{self._config.base_url.rstrip("/")}{self._EXTRACT_PATH}',
        headers={'X-API-Key': self._api_key},
        json=body,
      )
      response.raise_for_status()
    except httpx.HTTPError as error:
      raise ExternalServiceError('orbitals', 'claim_extractor.extract', error) from error

    try:
      payload = _OrbitalsResponse.model_validate(response.json())
    except (ValidationError, ValueError) as error:
      raise DataMappingError('orbitals', 'invalid ClaimExtractor response', error) from error

    return ClaimExtractionResult(
      claims=payload.extractions.claims,
      model=payload.model,
      usage=payload.usage,
      time_taken=payload.time_taken,
    )

  async def aclose(self) -> None:
    if self._owns_client:
      await self._client.aclose()
