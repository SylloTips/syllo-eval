import json
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, JsonValue, field_validator

from syllo_eval.model import ExpectedClaim


class DatasetJsonPlanStep(BaseModel):
  """One expected step in a dataset sample plan."""

  model_config = ConfigDict(extra='forbid')

  operation: str = Field(min_length=1)
  instruction: str
  parameters: dict[str, JsonValue] | None = None


class DatasetJsonSample(BaseModel):
  """One sample from an imported dataset JSON file."""

  input_prompt: str
  ground_truth_output: str | None = None
  # None means unlabeled; an empty list means labeled with no relevant IDs.
  snippet_ids: list[str] | None = None
  document_ids: list[str] | None = None
  plan: list[DatasetJsonPlanStep] | None = None
  claims: list[ExpectedClaim] | None = None

  @field_validator('claims')
  @classmethod
  def claim_ids_are_unique(cls, claims: list[ExpectedClaim] | None) -> list[ExpectedClaim] | None:
    if claims is not None and len({claim.id for claim in claims}) != len(claims):
      raise ValueError('claim ids must be unique within a sample')
    return claims


class DatasetJsonPayload(BaseModel):
  """Top-level dataset JSON payload."""

  samples: list[DatasetJsonSample]


def load_dataset_json(dataset_path: Path) -> DatasetJsonPayload:
  """Load and validate a dataset JSON file from disk."""
  payload = json.loads(dataset_path.read_text(encoding='utf-8'))
  return DatasetJsonPayload.model_validate(payload)
