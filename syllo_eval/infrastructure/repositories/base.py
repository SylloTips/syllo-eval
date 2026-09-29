"""
Base repository providing generic CRUD operations.

This module defines the base repository class that all specific repositories
inherit from.
"""

import logging
from abc import ABC, abstractmethod
from contextlib import asynccontextmanager
from typing import Generic, TypeVar, Optional, List, Any, Dict, AsyncIterator, NoReturn
from uuid import UUID

import psycopg
from psycopg import AsyncConnection
from psycopg.rows import class_row

from syllo_eval.infrastructure.database import DatabaseManager
from syllo_eval.infrastructure.exceptions import (
  NotFoundError,
  DuplicateError,
  IntegrityError,
  QueryError,
)

logger = logging.getLogger(__name__)

T = TypeVar('T')


class BaseRepository(ABC, Generic[T]):
  """
  Base repository providing common CRUD operations.

  All concrete repositories should inherit from this class and implement
  the abstract methods to define their specific table and model mappings.
  """

  def __init__(self, db_manager: DatabaseManager, connection: Optional[AsyncConnection[Any]] = None):
    """
    Initialize repository.

    Args:
        db_manager: Database manager for connection handling
    """
    self.db_manager = db_manager
    self._connection = connection

  @asynccontextmanager
  async def _get_connection(self) -> AsyncIterator[AsyncConnection[Any]]:
    """
    Get a connection for repository operations.

    Uses a shared transaction connection when provided; otherwise pulls
    from the pool for each operation.
    """
    if self._connection is not None:
      yield self._connection
    else:
      async with self.db_manager.get_async_connection() as conn:
        yield conn

  @property
  @abstractmethod
  def table_name(self) -> str:
    """Return the name of the database table."""
    pass

  @property
  @abstractmethod
  def model_class(self) -> type[T]:
    """Return the model class for this repository."""
    pass

  @property
  def id_column(self) -> str:
    """Return the name of the ID column (default: 'id')."""
    return 'id'

  def _handle_db_error(self, error: Exception, operation: str) -> NoReturn:
    """
    Handle database errors and convert to appropriate exceptions.

    Args:
        error: Original database error
        operation: Operation that failed (for logging)

    Raises:
        Appropriate PersistenceError subclass
    """
    if isinstance(error, psycopg.errors.UniqueViolation):
      constraint = getattr(error.diag, 'constraint_name', 'unknown')
      raise DuplicateError(self.table_name, constraint, str(error))
    elif isinstance(error, psycopg.errors.ForeignKeyViolation):
      details = str(error)
      raise IntegrityError(self.table_name, 'foreign_key', 'referenced_entity', details)
    else:
      logger.error(f'{operation} failed: {error}')
      raise QueryError(f'{operation} on {self.table_name}', error)

  async def create(self, entity: T) -> T:
    """
    Create a new entity in the database.

    Args:
        entity: Entity to create

    Returns:
        Created entity with generated ID and timestamps

    Raises:
        DuplicateError: If entity violates unique constraints
        IntegrityError: If entity violates foreign key constraints
    """
    try:
      async with self._get_connection() as conn:
        fields = self._get_insert_fields(entity)
        columns = ', '.join(fields.keys())
        placeholders = ', '.join(['%s'] * len(fields))
        values = tuple(fields.values())

        query = f"""
                    INSERT INTO {self.table_name} ({columns})
                    VALUES ({placeholders})
                    RETURNING *
                """

        async with conn.cursor(row_factory=class_row(self.model_class)) as cur:
          await cur.execute(query, values)
          result = await cur.fetchone()

          if result is None:
            raise QueryError(f'Failed to create {self.table_name}', Exception('No result returned'))

          logger.debug(f'Created {self.table_name} with id {getattr(result, self.id_column, "N/A")}')
          return result

    except Exception as e:
      self._handle_db_error(e, 'create')

  async def get_by_id(self, entity_id: UUID | str) -> Optional[T]:
    """
    Retrieve an entity by its ID.

    Args:
        entity_id: ID of the entity to retrieve

    Returns:
        Entity if found, None otherwise
    """
    try:
      async with self._get_connection() as conn:
        query = f"""
                    SELECT * FROM {self.table_name}
                    WHERE {self.id_column} = %s
                """

        async with conn.cursor(row_factory=class_row(self.model_class)) as cur:
          await cur.execute(query, (entity_id,))
          result = await cur.fetchone()

          if result:
            logger.debug(f'Retrieved {self.table_name} with id {entity_id}')
          return result

    except Exception as e:
      self._handle_db_error(e, f'get_by_id({entity_id})')

  async def get_by_id_or_raise(self, entity_id: UUID | str) -> T:
    """
    Retrieve an entity by its ID or raise an error if not found.

    Args:
        entity_id: ID of the entity to retrieve

    Returns:
        Entity

    Raises:
        NotFoundError: If entity not found
    """
    result = await self.get_by_id(entity_id)
    if result is None:
      raise NotFoundError(self.table_name, entity_id)
    return result

  async def update(self, entity: T) -> T:
    """
    Update an existing entity.

    Args:
        entity: Entity with updated values

    Returns:
        Updated entity

    Raises:
        NotFoundError: If entity doesn't exist
    """
    try:
      async with self._get_connection() as conn:
        fields = self._get_update_fields(entity)
        entity_id = getattr(entity, self.id_column)

        if not fields:
          return await self.get_by_id_or_raise(entity_id)

        set_clause = ', '.join([f'{k} = %s' for k in fields.keys()])
        values = tuple(fields.values()) + (entity_id,)

        query = f"""
                    UPDATE {self.table_name}
                    SET {set_clause}
                    WHERE {self.id_column} = %s
                    RETURNING *
                """

        async with conn.cursor(row_factory=class_row(self.model_class)) as cur:
          await cur.execute(query, values)
          result = await cur.fetchone()

          if result is None:
            raise NotFoundError(self.table_name, entity_id)

          logger.debug(f'Updated {self.table_name} with id {entity_id}')
          return result

    except NotFoundError:
      raise
    except Exception as e:
      self._handle_db_error(e, 'update')

  async def delete(self, entity_id: UUID | str) -> bool:
    """
    Delete an entity by its ID.

    Args:
        entity_id: ID of the entity to delete

    Returns:
        True if entity was deleted, False if not found
    """
    try:
      async with self._get_connection() as conn:
        query = f"""
                    DELETE FROM {self.table_name}
                    WHERE {self.id_column} = %s
                """

        async with conn.cursor() as cur:
          await cur.execute(query, (entity_id,))
          deleted = cur.rowcount > 0

          if deleted:
            logger.debug(f'Deleted {self.table_name} with id {entity_id}')
          return deleted

    except Exception as e:
      self._handle_db_error(e, f'delete({entity_id})')

  async def list_all(self, limit: Optional[int] = None, offset: int = 0) -> List[T]:
    """
    List all entities with optional pagination.

    Args:
        limit: Maximum number of entities to return
        offset: Number of entities to skip

    Returns:
        List of entities
    """
    try:
      async with self._get_connection() as conn:
        query = f'SELECT * FROM {self.table_name} ORDER BY created_at DESC'

        if limit is not None:
          query += f' LIMIT {limit} OFFSET {offset}'

        async with conn.cursor(row_factory=class_row(self.model_class)) as cur:
          await cur.execute(query)
          results = await cur.fetchall()

          logger.debug(f'Retrieved {len(results)} {self.table_name} records')
          return results

    except Exception as e:
      self._handle_db_error(e, 'list_all')

  async def count(self) -> int:
    """
    Count total number of entities.

    Returns:
        Total count
    """
    try:
      async with self._get_connection() as conn:
        query = f'SELECT COUNT(*) FROM {self.table_name}'

        async with conn.cursor() as cur:
          await cur.execute(query)
          result = await cur.fetchone()
          return result[0] if result else 0

    except Exception as e:
      self._handle_db_error(e, 'count')

  async def exists(self, entity_id: UUID | str) -> bool:
    """
    Check if an entity exists by ID.

    Args:
        entity_id: ID to check

    Returns:
        True if exists, False otherwise
    """
    try:
      async with self._get_connection() as conn:
        query = f"""
                    SELECT EXISTS(
                        SELECT 1 FROM {self.table_name}
                        WHERE {self.id_column} = %s
                    )
                """

        async with conn.cursor() as cur:
          await cur.execute(query, (entity_id,))
          result = await cur.fetchone()
          return result[0] if result else False

    except Exception as e:
      self._handle_db_error(e, f'exists({entity_id})')

  @abstractmethod
  def _get_insert_fields(self, entity: T) -> Dict[str, Any]:
    """
    Get fields to insert for a new entity.

    Args:
        entity: Entity to insert

    Returns:
        Dictionary of column names to values
    """
    pass

  @abstractmethod
  def _get_update_fields(self, entity: T) -> Dict[str, Any]:
    """
    Get fields to update for an existing entity.

    Args:
        entity: Entity to update

    Returns:
        Dictionary of column names to values (excluding ID and timestamps)
    """
    pass
