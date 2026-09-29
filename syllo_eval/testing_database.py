from syllo_eval.infrastructure import DatabaseConfig
from syllo_eval.infrastructure.database import DatabaseManager
from syllo_eval.settings import DatabaseSettings, load_settings_env


async def setup_test_database() -> DatabaseManager:
  load_settings_env()
  settings = DatabaseSettings()
  if settings.password is None:
    raise RuntimeError('DB_PASSWORD not set in environment or .env file')

  db_manager = DatabaseManager(
    DatabaseConfig(
      host=settings.host,
      port=settings.port,
      database=settings.database,
      user=settings.user,
      password=settings.password,
      min_size=settings.min_size,
      max_size=settings.max_size,
      timeout=settings.timeout,
    )
  )
  await db_manager.initialize_async()

  if not await db_manager.health_check():
    await db_manager.close_async()
    raise RuntimeError('Database health check failed')

  return db_manager
