FROM python:3.12-slim

# libpq for psycopg
RUN apt-get update && apt-get install -y --no-install-recommends libpq5 && rm -rf /var/lib/apt/lists/*
RUN pip install --no-cache-dir poetry==1.8.3
ENV POETRY_NO_INTERACTION=1 POETRY_VIRTUALENVS_CREATE=false PYTHONUNBUFFERED=1 ALEMBIC_CONFIG=/app/alembic.ini

WORKDIR /app
# The dependency layer is rebuilt only when syllo-eval or the lock file changes.
COPY pyproject.toml README.md alembic.ini ./
COPY syllo_eval ./syllo_eval
COPY experiments/pyproject.toml experiments/poetry.lock experiments/README.md ./experiments/
WORKDIR /app/experiments
RUN poetry install --only main,migrations --no-root
COPY experiments ./
RUN poetry install --only-root
