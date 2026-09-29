"""
Database configuration and connection pool management using psycopg.
"""

import logging
from contextlib import asynccontextmanager, contextmanager
from typing import Optional, AsyncIterator, Iterator

from psycopg import AsyncConnection, Connection
from psycopg.rows import tuple_row
from psycopg_pool import AsyncConnectionPool, ConnectionPool

logger = logging.getLogger(__name__)


class DatabaseConfig:
  """Database configuration settings."""

  def __init__(
    self,
    host: str = 'localhost',
    port: int = 5432,
    database: str = 'pg-syllo-eval',
    user: str = 'postgres',
    password: str = '',
    min_size: int = 2,
    max_size: int = 10,
    timeout: float = 30.0,
  ):
    """
    Initialize database configuration.

    Args:
        host: Database host
        port: Database port
        database: Database name
        user: Database user
        password: Database password
        min_size: Minimum number of connections in pool
        max_size: Maximum number of connections in pool
        timeout: Connection timeout in seconds
    """
    self.host = host
    self.port = port
    self.database = database
    self.user = user
    self.password = password
    self.min_size = min_size
    self.max_size = max_size
    self.timeout = timeout

  @property
  def connection_string(self) -> str:
    """Generate psycopg connection string."""
    return (
      f'host={self.host} '
      f'port={self.port} '
      f'dbname={self.database} '
      f'user={self.user} '
      f'password={self.password} '
      f'connect_timeout={int(self.timeout)}'
    )


class DatabaseManager:
  """
  Manages database connection pools for both sync and async operations.
  """

  def __init__(self, config: DatabaseConfig):
    """
    Initialize database manager.

    Args:
        config: Database configuration
    """
    self.config = config
    self._async_pool: Optional[AsyncConnectionPool] = None
    self._sync_pool: Optional[ConnectionPool] = None
    self._is_initialized = False

  async def initialize_async(self) -> None:
    """Initialize async connection pool."""
    if self._async_pool is not None:
      logger.warning('Async pool already initialized')
      return

    logger.info(f'Initializing async connection pool: min_size={self.config.min_size}, max_size={self.config.max_size}')

    self._async_pool = AsyncConnectionPool(
      conninfo=self.config.connection_string,
      min_size=self.config.min_size,
      max_size=self.config.max_size,
      timeout=self.config.timeout,
      open=False,
    )
    await self._async_pool.open()
    self._is_initialized = True
    logger.info('Async connection pool initialized successfully')

  def initialize_sync(self) -> None:
    """Initialize sync connection pool."""
    if self._sync_pool is not None:
      logger.warning('Sync pool already initialized')
      return

    logger.info(f'Initializing sync connection pool: min_size={self.config.min_size}, max_size={self.config.max_size}')

    self._sync_pool = ConnectionPool(
      conninfo=self.config.connection_string,
      min_size=self.config.min_size,
      max_size=self.config.max_size,
      timeout=self.config.timeout,
    )
    self._sync_pool.open()
    self._is_initialized = True
    logger.info('Sync connection pool initialized successfully')

  async def close_async(self) -> None:
    """Close async connection pool."""
    if self._async_pool is not None:
      logger.info('Closing async connection pool')
      await self._async_pool.close()
      self._async_pool = None

  def close_sync(self) -> None:
    """Close sync connection pool."""
    if self._sync_pool is not None:
      logger.info('Closing sync connection pool')
      self._sync_pool.close()
      self._sync_pool = None

  @asynccontextmanager
  async def get_async_connection(self, row_factory=tuple_row) -> AsyncIterator[AsyncConnection]:
    """
    Get an async database connection from the pool.

    Args:
        row_factory: Row factory for query results (default: tuple_row)

    Yields:
        AsyncConnection: Database connection

    Raises:
        RuntimeError: If pool not initialized
    """
    if self._async_pool is None:
      raise RuntimeError('Async pool not initialized. Call initialize_async() first.')

    async with self._async_pool.connection() as conn:
      conn.row_factory = row_factory
      yield conn

  @contextmanager
  def get_sync_connection(self, row_factory=tuple_row) -> Iterator[Connection]:
    """
    Get a sync database connection from the pool.

    Args:
        row_factory: Row factory for query results (default: tuple_row)

    Yields:
        Connection: Database connection

    Raises:
        RuntimeError: If pool not initialized
    """
    if self._sync_pool is None:
      raise RuntimeError('Sync pool not initialized. Call initialize_sync() first.')

    with self._sync_pool.connection() as conn:
      conn.row_factory = row_factory
      yield conn

  @asynccontextmanager
  async def transaction(self) -> AsyncIterator[AsyncConnection]:
    """
    Context manager for async transactions.

    Automatically commits on success, rolls back on exception.

    Yields:
        AsyncConnection: Database connection in transaction

    Example:
        async with db_manager.transaction() as conn:
            await conn.execute("INSERT INTO table VALUES (%s)", (value,))
            # Automatically committed on success
    """
    async with self.get_async_connection() as conn:
      async with conn.transaction():
        yield conn

  @contextmanager
  def sync_transaction(self) -> Iterator[Connection]:
    """
    Context manager for sync transactions.

    Automatically commits on success, rolls back on exception.

    Yields:
        Connection: Database connection in transaction

    Example:
        with db_manager.sync_transaction() as conn:
            conn.execute("INSERT INTO table VALUES (%s)", (value,))
            # Automatically committed on success
    """
    with self.get_sync_connection() as conn:
      with conn.transaction():
        yield conn

  async def health_check(self) -> bool:
    """
    Perform async health check on database connection.

    Returns:
        bool: True if database is accessible, False otherwise
    """
    try:
      async with self.get_async_connection(row_factory=tuple_row) as conn:
        async with conn.cursor() as cur:
          await cur.execute('SELECT 1')
          result = await cur.fetchone()
          return result is not None and result[0] == 1
    except Exception as e:
      logger.error(f'Database health check failed: {e}')
      return False

  def sync_health_check(self) -> bool:
    """
    Perform sync health check on database connection.

    Returns:
        bool: True if database is accessible, False otherwise
    """
    try:
      with self.get_sync_connection(row_factory=tuple_row) as conn:
        with conn.cursor() as cur:
          cur.execute('SELECT 1')
          result = cur.fetchone()
          return result is not None and result[0] == 1
    except Exception as e:
      logger.error(f'Database health check failed: {e}')
      return False
