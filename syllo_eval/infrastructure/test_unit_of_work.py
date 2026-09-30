import asyncio
import unittest
from unittest.mock import ANY, MagicMock, Mock, patch

from psycopg import AsyncConnection

from syllo_eval.infrastructure import DatabaseConfig, DatabaseManager, TransactionalUnitOfWork, UnitOfWork


class TestUnitOfWork(unittest.IsolatedAsyncioTestCase):
  async def test_plain_unit_of_work_caches_repositories_without_opening_a_connection(self) -> None:
    db_manager = Mock(spec=DatabaseManager)
    uow = UnitOfWork(db_manager)

    async with uow as active:
      self.assertIs(active, uow)
      self.assertIs(uow.agents, uow.agents)
      self.assertIs(uow.span_metric_computations, uow.metric_computations)
      self.assertIsNone(uow.agents._connection)

    db_manager.get_async_connection.assert_not_called()
    db_manager.transaction.assert_not_called()


class TestTransactionalUnitOfWorkLifecycle(unittest.IsolatedAsyncioTestCase):
  def setUp(self) -> None:
    self.db_manager = DatabaseManager(DatabaseConfig())
    self.connection = MagicMock(spec=AsyncConnection)
    self.connection_context = MagicMock()
    self.connection_context.__aenter__.return_value = self.connection
    self.transaction = self.connection.transaction.return_value
    patcher = patch.object(self.db_manager, 'get_async_connection', return_value=self.connection_context)
    self.addCleanup(patcher.stop)
    patcher.start()
    self.uow = TransactionalUnitOfWork(self.db_manager)

  async def test_repositories_share_the_transaction_connection(self) -> None:
    async with self.uow as uow:
      self.assertIs(uow, self.uow)
      self.assertIs(uow.agents, uow.agents)
      self.assertIs(uow.span_metric_computations, uow.metric_computations)
      self.assertIs(uow.agents._connection, self.connection)
      self.assertIs(uow.datasets._connection, self.connection)

    self.transaction.__aexit__.assert_awaited_once_with(None, None, None)
    self.connection_context.__aexit__.assert_awaited_once_with(None, None, None)

  async def test_repository_access_before_entry_raises(self) -> None:
    with self.assertRaisesRegex(RuntimeError, 'inside its async with block'):
      self.uow.agents

    self.connection_context.__aenter__.assert_not_awaited()

  async def test_cached_and_uncached_repository_access_after_exit_raises(self) -> None:
    async with self.uow as uow:
      uow.agents

    for name in ('agents', 'datasets', 'span_metric_computations'):
      with self.subTest(repository=name):
        with self.assertRaisesRegex(RuntimeError, 'inside its async with block'):
          getattr(self.uow, name)

  async def test_reentry_raises(self) -> None:
    async with self.uow:
      with self.assertRaisesRegex(RuntimeError, 'single-use'):
        async with self.uow:
          self.fail('Nested entry must not succeed')

    with self.assertRaisesRegex(RuntimeError, 'single-use'):
      async with self.uow:
        self.fail('Reentry must not succeed')

    self.connection_context.__aenter__.assert_awaited_once()

  async def test_exceptions_and_cancellation_reach_the_transaction(self) -> None:
    for error in (ValueError('failed'), asyncio.CancelledError()):
      with self.subTest(error_type=type(error).__name__):
        self.transaction.reset_mock()
        self.connection_context.reset_mock()
        with self.assertRaises(type(error)):
          async with TransactionalUnitOfWork(self.db_manager):
            raise error

        self.transaction.__aexit__.assert_awaited_once_with(type(error), error, ANY)
        self.connection_context.__aexit__.assert_awaited_once_with(type(error), error, ANY)

  async def test_transaction_entry_failure_releases_the_connection(self) -> None:
    error = RuntimeError('transaction entry failed')
    self.transaction.__aenter__.side_effect = error

    with self.assertRaisesRegex(RuntimeError, 'transaction entry failed'):
      async with self.uow:
        self.fail('Body must not run after entry fails')

    self.transaction.__aexit__.assert_not_awaited()
    self.connection_context.__aexit__.assert_awaited_once_with(RuntimeError, error, ANY)
    with self.assertRaisesRegex(RuntimeError, 'inside its async with block'):
      self.uow.agents

  async def test_transaction_exit_failure_releases_the_connection(self) -> None:
    error = RuntimeError('transaction exit failed')
    self.transaction.__aexit__.side_effect = error

    with self.assertRaisesRegex(RuntimeError, 'transaction exit failed'):
      async with self.uow as uow:
        uow.agents

    self.connection_context.__aexit__.assert_awaited_once_with(RuntimeError, error, ANY)
    with self.assertRaisesRegex(RuntimeError, 'inside its async with block'):
      self.uow.agents


if __name__ == '__main__':
  unittest.main()
