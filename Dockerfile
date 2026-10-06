FROM python:3.12-slim AS builder

RUN pip install --no-cache-dir poetry==1.8.3
ENV POETRY_NO_INTERACTION=1 POETRY_VIRTUALENVS_IN_PROJECT=1
WORKDIR /app

COPY pyproject.toml poetry.lock ./
RUN poetry install --only main,migrations --no-root
COPY README.md ./
COPY syllo_eval ./syllo_eval
RUN poetry build --format wheel && .venv/bin/pip install --no-deps dist/*.whl

FROM python:3.12-slim

# libpq for psycopg
RUN apt-get update && apt-get install -y --no-install-recommends libpq5 && rm -rf /var/lib/apt/lists/*
RUN useradd --create-home syllo
WORKDIR /app
COPY --from=builder /app/.venv /app/.venv
COPY alembic.ini ./
ENV PATH="/app/.venv/bin:$PATH" PYTHONUNBUFFERED=1 ALEMBIC_CONFIG=/app/alembic.ini
USER syllo

EXPOSE 8005
CMD ["uvicorn", "syllo_eval.API.app:app", "--host", "0.0.0.0", "--port", "8005"]
