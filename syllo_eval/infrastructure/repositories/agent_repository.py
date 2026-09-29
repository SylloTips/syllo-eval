"""
Repository for Agent entity operations.
"""

import logging
from typing import Dict, Any, Optional, List
from uuid import uuid4

from syllo_eval.infrastructure.exceptions import DuplicateError
from syllo_eval.model import Agent
from syllo_eval.infrastructure.repositories.base import BaseRepository
from psycopg.rows import class_row

logger = logging.getLogger(__name__)


class AgentRepository(BaseRepository[Agent]):
  """Repository for managing Agent entities."""

  @property
  def table_name(self) -> str:
    return 'agent'

  @property
  def model_class(self) -> type[Agent]:
    return Agent

  def _get_insert_fields(self, entity: Agent) -> Dict[str, Any]:
    """Get fields for inserting a new agent."""
    fields: Dict[str, Any] = {
      'name': entity.name,
      'version_tag': entity.version_tag,
    }

    if entity.id is not None:
      fields['id'] = entity.id

    return fields

  def _get_update_fields(self, entity: Agent) -> Dict[str, Any]:
    """Get fields for updating an agent."""
    return {
      'name': entity.name,
      'version_tag': entity.version_tag,
    }

  async def get_by_name_and_version(self, name: str, version_tag: str) -> Optional[Agent]:
    """
    Get agent by name and version tag.

    Args:
        name: Agent name
        version_tag: Agent version tag

    Returns:
        Agent if found, None otherwise
    """
    try:
      async with self._get_connection() as conn:
        query = """
                    SELECT * FROM agent
                    WHERE name = %s AND version_tag = %s
                """

        async with conn.cursor(row_factory=class_row(Agent)) as cur:
          await cur.execute(query, (name, version_tag))
          return await cur.fetchone()

    except Exception as e:
      self._handle_db_error(e, f'get_by_name_and_version({name}, {version_tag})')

  async def get_or_create_by_name_and_version(self, name: str, version_tag: str) -> Agent:
    """Get an agent by name/version, creating it when missing."""
    agent = await self.get_by_name_and_version(name=name, version_tag=version_tag)
    if agent is not None:
      return agent

    try:
      agent = await self.create(
        Agent(
          id=uuid4(),
          name=name,
          version_tag=version_tag,
        )
      )
      logger.info('Created agent name="%s" version_tag="%s"', name, version_tag)
      return agent
    except DuplicateError:
      agent = await self.get_by_name_and_version(name=name, version_tag=version_tag)
      if agent is None:
        raise
      return agent

  async def list_by_name(self, name: str) -> List[Agent]:
    """
    List all versions of an agent by name.

    Args:
        name: Agent name

    Returns:
        List of agents with the given name
    """
    try:
      async with self._get_connection() as conn:
        query = """
                    SELECT * FROM agent
                    WHERE name = %s
                    ORDER BY created_at DESC
                """

        async with conn.cursor(row_factory=class_row(Agent)) as cur:
          await cur.execute(query, (name,))
          return await cur.fetchall()

    except Exception as e:
      self._handle_db_error(e, f'list_by_name({name})')

  async def list_distinct_names(self) -> List[str]:
    """
    List all distinct agent names.

    Returns:
        List of unique agent names
    """
    try:
      async with self._get_connection() as conn:
        query = """
                    SELECT DISTINCT name FROM agent
                    ORDER BY name
                """

        async with conn.cursor() as cur:
          await cur.execute(query)
          results = await cur.fetchall()
          return [row[0] for row in results]

    except Exception as e:
      self._handle_db_error(e, 'list_distinct_names')
