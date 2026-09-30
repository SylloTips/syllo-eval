from collections.abc import Sequence
from pathlib import Path
from typing import Annotated, Any
from uuid import UUID

from pydantic import BaseModel, Field, StringConstraints, ValidationError

from syllo_eval.evaluation.trace_adapter import TraceAdapter, TraceProcessingResult
from syllo_eval.evaluation.trace_processor import normalize_trace
from syllo_eval.model import Sample


class TraceImportError(ValueError):
  """Raised when imported traces cannot be normalized or bound to dataset samples."""


class ImportedTrace(BaseModel):
  """One trace as exported by its source: provider records are passed unchanged to the adapter."""

  trace_id: Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]
  spans: list[dict[str, Any]] = Field(min_length=1)


def load_trace_file(path: Path) -> ImportedTrace:
  try:
    return ImportedTrace.model_validate_json(path.read_bytes())
  except ValidationError as err:
    raise TraceImportError(f'Invalid trace file {path}: {err}') from err


def normalize_imported_trace(adapter: TraceAdapter, trace: ImportedTrace) -> TraceProcessingResult:
  try:
    return normalize_trace(adapter, trace.trace_id, trace.spans)
  except (KeyError, TypeError, ValueError) as err:
    raise TraceImportError(f'Trace {trace.trace_id} could not be normalized: {err}') from err


def match_traces_to_samples(results: Sequence[TraceProcessingResult], samples: Sequence[Sample]) -> dict[UUID, str]:
  """Bind each trace to the one sample whose prompt equals the trace request; every trace must bind."""
  samples_by_prompt: dict[str, list[Sample]] = {}
  for sample in samples:
    samples_by_prompt.setdefault(sample.input_prompt.strip(), []).append(sample)

  trace_ids_by_sample_id: dict[UUID, str] = {}
  for result in results:
    trace_id = result.trace.external_id
    candidates = samples_by_prompt.get(_request(result), [])
    if len(candidates) != 1:
      raise TraceImportError(f'Trace {trace_id} request matches {len(candidates)} dataset samples; expected one')
    sample_id = candidates[0].id
    if sample_id in trace_ids_by_sample_id:
      raise TraceImportError(f'Traces {trace_ids_by_sample_id[sample_id]} and {trace_id} match the same sample')
    trace_ids_by_sample_id[sample_id] = trace_id
  return trace_ids_by_sample_id


def _request(result: TraceProcessingResult) -> str:
  roots = [span for span in result.spans if span.span_type == 'agent_root']
  if len(roots) != 1:
    raise TraceImportError(f'Trace {result.trace.external_id} has {len(roots)} agent_root spans; expected one')
  request = roots[0].semantics.request
  if request is None and isinstance(roots[0].input_data, str):
    request = roots[0].input_data
  if request is None or not request.strip():
    raise TraceImportError(f'Trace {result.trace.external_id} root span has no request text')
  return request.strip()
