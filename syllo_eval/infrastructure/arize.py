import asyncio
import json
import logging
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List

import httpx

from syllo_eval.infrastructure.exceptions import ConfigurationError, DataMappingError, ExternalServiceError
from syllo_eval.settings import PhoenixSettings

logger = logging.getLogger(__name__)


class PhoenixClient:
  """Client for fetching Phoenix traces and resolving request IDs."""

  _GRAPHQL_PATH = '/graphql'
  _SPANS_PAGE_SIZE = 50
  _REQUEST_ID_LOOKUP_PAGE_SIZE = 10

  _TRACE_SPANS_QUERY = (
    'query GetTraceSpans($traceId: String!, $first: Int!, $after: String) {\n'
    '  getTraceByOtelId(traceId: $traceId) {\n'
    '    spans(first: $first, after: $after) {\n'
    '      edges {\n'
    '        node {\n'
    '          name\n'
    '          spanKind\n'
    '          statusCode\n'
    '          statusMessage\n'
    '          startTime\n'
    '          endTime\n'
    '          parentId\n'
    '          context { spanId traceId }\n'
    '          attributes\n'
    '        }\n'
    '      }\n'
    '      pageInfo { hasNextPage endCursor }\n'
    '    }\n'
    '  }\n'
    '}'
  )

  _REQUEST_ID_LOOKUP_QUERY = (
    'query LookupTraceIdByRequestId('
    '$projectName: String!, $filterCondition: String!, $first: Int!, $start: DateTime!, $end: DateTime!'
    ') {\n'
    '  projects(filter: {col: name, value: $projectName}, first: 1) {\n'
    '    edges {\n'
    '      node {\n'
    '        spans('
    'first: $first, rootSpansOnly: true, '
    'filterCondition: $filterCondition, '
    'timeRange: {start: $start, end: $end}'
    ') {\n'
    '          edges { node { name context { traceId } } }\n'
    '        }\n'
    '      }\n'
    '    }\n'
    '  }\n'
    '}'
  )

  def __init__(self, config: PhoenixSettings):
    self.config = config

  async def get_trace_json(self, trace_id: str) -> List[Dict[str, Any]]:
    """Fetch all spans for a trace via Phoenix GraphQL, returning flattened records."""
    self._validate_trace_id(trace_id)
    nodes = await self._fetch_trace_spans_via_graphql(trace_id)
    return [self._normalize_span_node(node) for node in nodes]

  async def get_trace_id_by_request_id(self, request_id: str) -> str | None:
    """Resolve a Phoenix trace_id from an agent request_id using a recent root-span lookup."""
    self._validate_request_id(request_id)
    attempts = max(1, self.config.request_id_lookup_max_attempts)
    operation = f'get_trace_id_by_request_id(request_id={request_id})'

    for attempt in range(1, attempts + 1):
      try:
        trace_id = await self._lookup_trace_id(request_id)
      except httpx.HTTPError as err:
        if attempt == attempts:
          raise ExternalServiceError('phoenix', operation, err) from err
        backoff_seconds = self._request_id_lookup_backoff_seconds(attempt)
        logger.warning(
          'Phoenix request_id lookup attempt %s/%s failed request_id=%s backoff_seconds=%s: %s',
          attempt,
          attempts,
          request_id,
          backoff_seconds,
          err,
        )
        await asyncio.sleep(backoff_seconds)
        continue

      if trace_id is None:
        if attempt == attempts:
          logger.warning(
            'Phoenix request_id lookup exhausted attempts request_id=%s attempts=%s',
            request_id,
            attempts,
          )
          return None
        backoff_seconds = self._request_id_lookup_backoff_seconds(attempt)
        logger.info(
          'Phoenix request_id lookup returned no rows yet request_id=%s attempt=%s/%s backoff_seconds=%s',
          request_id,
          attempt,
          attempts,
          backoff_seconds,
        )
        await asyncio.sleep(backoff_seconds)
        continue

      return trace_id

    return None

  def _request_id_lookup_backoff_seconds(self, attempt: int) -> float:
    return min(
      self.config.request_id_lookup_initial_backoff_seconds * (2 ** (attempt - 1)),
      self.config.request_id_lookup_max_backoff_seconds,
    )

  def _trace_fetch_backoff_seconds(self, attempt: int) -> float:
    return min(
      self.config.trace_fetch_initial_backoff_seconds * (2 ** (attempt - 1)),
      self.config.trace_fetch_max_backoff_seconds,
    )

  @staticmethod
  def _validate_trace_id(trace_id: str) -> None:
    if not trace_id or not trace_id.strip():
      raise ConfigurationError('phoenix', 'trace_id', 'must be a non-empty string')

  @staticmethod
  def _validate_request_id(request_id: str) -> None:
    if not request_id or not request_id.strip():
      raise ConfigurationError('phoenix', 'request_id', 'must be a non-empty string')

  async def _post_graphql_with_retries(
    self,
    query: str,
    variables: Dict[str, Any],
    operation: str,
    log_label: str,
  ) -> Dict[str, Any]:
    last_error: httpx.HTTPError | None = None
    attempts = max(1, self.config.max_retries)
    for attempt in range(1, attempts + 1):
      try:
        return await self._post_graphql(query, variables)
      except httpx.HTTPError as err:
        last_error = err
        if attempt < attempts:
          backoff_seconds = self._trace_fetch_backoff_seconds(attempt)
          logger.warning(
            '%s attempt %s/%s failed backoff_seconds=%s: %s',
            log_label,
            attempt,
            attempts,
            backoff_seconds,
            err,
          )
          await asyncio.sleep(backoff_seconds)
        else:
          logger.warning('%s attempt %s/%s failed: %s', log_label, attempt, attempts, err)

    raise ExternalServiceError('phoenix', operation, last_error)

  async def _post_graphql(self, query: str, variables: Dict[str, Any]) -> Dict[str, Any]:
    headers: Dict[str, str] = {'content-type': 'application/json', 'accept-encoding': 'gzip'}
    if self.config.api_key:
      headers['authorization'] = f'Bearer {self.config.api_key}'
    url = f'{self.config.base_url.rstrip("/")}{self._GRAPHQL_PATH}'
    async with httpx.AsyncClient(timeout=self.config.timeout) as client:
      response = await client.post(url, json={'query': query, 'variables': variables}, headers=headers)
      response.raise_for_status()
      try:
        body = response.json()
      except ValueError as err:
        raise DataMappingError('phoenix', 'GraphQL response is not valid JSON', err) from err
      if not isinstance(body, dict):
        raise DataMappingError('phoenix', 'GraphQL response is not a JSON object')
      if body.get('errors'):
        raise DataMappingError('phoenix', f'GraphQL errors: {body["errors"]}')
      return body

  async def _fetch_trace_spans_via_graphql(self, trace_id: str) -> List[Dict[str, Any]]:
    nodes: List[Dict[str, Any]] = []
    cursor: str | None = None
    page = 0
    operation = f'get_trace_json(trace_id={trace_id})'
    while True:
      page += 1
      body = await self._post_graphql_with_retries(
        self._TRACE_SPANS_QUERY,
        {'traceId': trace_id, 'first': self._SPANS_PAGE_SIZE, 'after': cursor},
        operation=operation,
        log_label=f'Phoenix trace fetch trace_id={trace_id} page={page}',
      )
      trace = (body.get('data') or {}).get('getTraceByOtelId')
      if trace is None:
        return []
      spans_connection = trace.get('spans') or {}
      for edge in spans_connection.get('edges') or []:
        node = edge.get('node')
        if node is not None:
          nodes.append(node)
      page_info = spans_connection.get('pageInfo') or {}
      if not page_info.get('hasNextPage'):
        break
      next_cursor = page_info.get('endCursor')
      if next_cursor is None:
        raise DataMappingError('phoenix', f'missing next cursor while fetching trace_id={trace_id}')
      if next_cursor == cursor:
        raise DataMappingError('phoenix', f'repeated next cursor while fetching trace_id={trace_id}')
      cursor = next_cursor
    return nodes

  async def _lookup_trace_id(self, request_id: str) -> str | None:
    if not self.config.project_id:
      raise ConfigurationError('phoenix', 'project_id', 'must be set for request_id lookup')

    end_time = datetime.now(timezone.utc)
    start_time = end_time - timedelta(seconds=self.config.request_id_lookup_time_window_seconds)
    body = await self._post_graphql(
      self._REQUEST_ID_LOOKUP_QUERY,
      {
        'projectName': self.config.project_id,
        'filterCondition': f"attributes['request_id'] == {_filter_literal(request_id)}",
        'first': self._REQUEST_ID_LOOKUP_PAGE_SIZE,
        'start': _format_iso_datetime(start_time),
        'end': _format_iso_datetime(end_time),
      },
    )

    project_edges = ((body.get('data') or {}).get('projects') or {}).get('edges') or []
    if not project_edges:
      raise DataMappingError('phoenix', f'project not found: {self.config.project_id}')
    spans_connection = ((project_edges[0].get('node') or {}).get('spans')) or {}
    span_edges = spans_connection.get('edges') or []
    if not span_edges:
      return None

    trace_id = self._select_request_trace_id(span_edges)
    if trace_id is None or not str(trace_id).strip():
      return None
    return str(trace_id)

  def _select_request_trace_id(self, span_edges: list[dict[str, Any]]) -> str | None:
    for edge in span_edges:
      node = edge.get('node') or {}
      span_name = str(node.get('name') or '').strip().lower()
      if span_name in self.config.request_id_excluded_root_span_names:
        continue

      trace_id = (node.get('context') or {}).get('traceId')
      if trace_id is not None and str(trace_id).strip():
        return str(trace_id)

    return None

  @staticmethod
  def _normalize_span_node(node: Dict[str, Any]) -> Dict[str, Any]:
    record: Dict[str, Any] = {
      'name': node.get('name'),
      'span_kind': node.get('spanKind'),
      'status_code': node.get('statusCode'),
      'status_message': node.get('statusMessage'),
      'parent_id': node.get('parentId'),
      'start_time': _parse_iso_datetime(node.get('startTime')),
      'end_time': _parse_iso_datetime(node.get('endTime')),
    }
    context = node.get('context') or {}
    record['context.span_id'] = context.get('spanId')
    record['context.trace_id'] = context.get('traceId')

    raw_attributes = node.get('attributes')
    if raw_attributes:
      try:
        parsed_attributes = json.loads(raw_attributes) if isinstance(raw_attributes, str) else raw_attributes
      except (TypeError, ValueError) as err:
        raise DataMappingError('phoenix', 'failed to parse span attributes JSON', err) from err
      if isinstance(parsed_attributes, dict):
        for dotted_key, value in _flatten_dict(parsed_attributes, prefix='attributes').items():
          record[dotted_key] = value

    return record


def _parse_iso_datetime(value: Any) -> datetime | None:
  if value is None:
    return None
  if isinstance(value, datetime):
    return value
  if not isinstance(value, str):
    return None
  text = value.strip()
  if not text:
    return None
  if text.endswith('Z'):
    text = text[:-1] + '+00:00'
  try:
    return datetime.fromisoformat(text)
  except ValueError:
    return None


def _flatten_dict(value: Dict[str, Any], prefix: str) -> Dict[str, Any]:
  flat: Dict[str, Any] = {}
  for key, inner in value.items():
    composite_key = f'{prefix}.{key}' if prefix else str(key)
    if isinstance(inner, dict):
      flat.update(_flatten_dict(inner, composite_key))
    else:
      flat[composite_key] = inner
  return flat


def _filter_literal(value: str) -> str:
  return repr(value)


def _format_iso_datetime(value: datetime) -> str:
  return value.isoformat().replace('+00:00', 'Z')
