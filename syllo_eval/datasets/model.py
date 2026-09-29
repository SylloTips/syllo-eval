import json
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, JsonValue


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
  snippet_ids: list[str] = Field(default_factory=list)
  document_ids: list[str] = Field(default_factory=list)
  plan: list[DatasetJsonPlanStep] | None = None


class DatasetJsonPayload(BaseModel):
  """Top-level dataset JSON payload."""

  samples: list[DatasetJsonSample]


def load_dataset_json(dataset_path: Path) -> DatasetJsonPayload:
  """Load and validate a dataset JSON file from disk."""
  payload = json.loads(dataset_path.read_text(encoding='utf-8'))
  return DatasetJsonPayload.model_validate(payload)
