from typing import Literal, Protocol

from pydantic import BaseModel


class ClaimExtractorMessage(BaseModel):
  role: Literal['user', 'assistant']
  content: str


class ExtractedClaim(BaseModel):
  subtype: str
  content: str
  evidences: list[str]


class ClaimExtractionResult(BaseModel):
  claims: list[ExtractedClaim]
  model: str
  usage: dict[str, int] | None = None
  time_taken: float | None = None


class ClaimExtractorClient(Protocol):
  async def extract(self, conversation: list[ClaimExtractorMessage]) -> ClaimExtractionResult:
    """Extract claims from the final message in a conversation."""

  async def aclose(self) -> None:
    """Release client resources."""
