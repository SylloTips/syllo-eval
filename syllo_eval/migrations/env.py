from alembic import context
from sqlalchemy import URL, create_engine, pool

from syllo_eval.settings import DatabaseSettings, load_settings_env

target_metadata = None


def _database_url() -> URL:
  load_settings_env(override=False)
  database_settings = DatabaseSettings()
  if not database_settings.password:
    raise RuntimeError('DB_PASSWORD not set. Configure it in environment or .env.')

  return URL.create(
    'postgresql+psycopg',
    username=database_settings.user,
    password=database_settings.password,
    host=database_settings.host,
    port=database_settings.port,
    database=database_settings.database,
  )


def run_migrations_offline() -> None:
  context.configure(
    url=_database_url().render_as_string(hide_password=False),
    target_metadata=target_metadata,
    literal_binds=True,
    dialect_opts={'paramstyle': 'named'},
  )

  with context.begin_transaction():
    context.run_migrations()


def run_migrations_online() -> None:
  engine = create_engine(_database_url(), poolclass=pool.NullPool)

  try:
    with engine.connect() as connection:
      context.configure(connection=connection, target_metadata=target_metadata)

      with context.begin_transaction():
        context.run_migrations()
  finally:
    engine.dispose()


if context.is_offline_mode():
  run_migrations_offline()
else:
  run_migrations_online()
