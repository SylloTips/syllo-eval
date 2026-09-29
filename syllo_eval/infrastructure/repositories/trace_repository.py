"""
Repository for Trace entity operations.
"""

from typing import Dict, Any

from syllo_eval.model import Trace
from syllo_eval.infrastructure.repositories.base import BaseRepository


class TraceRepository(BaseRepository[Trace]):
  """Repository for managing Trace entities."""

  @property
  def table_name(self) -> str:
    return 'trace'

  @property
  def model_class(self) -> type[Trace]:
    return Trace

  @property
  def id_column(self) -> str:
    return 'external_id'

  def _get_insert_fields(self, entity: Trace) -> Dict[str, Any]:
    return {
      'external_id': entity.external_id,
      'start_time': entity.start_time,
      'end_time': entity.end_time,
      'schema_version': entity.schema_version,
      'source': entity.source,
      'adapter': entity.adapter,
    }

  def _get_update_fields(self, entity: Trace) -> Dict[str, Any]:
    return {
      'start_time': entity.start_time,
      'end_time': entity.end_time,
      'schema_version': entity.schema_version,
      'source': entity.source,
      'adapter': entity.adapter,
    }
