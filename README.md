# syllo-eval

Syllotips evaluation framework for enterprise agents.

## How it works

A fresh evaluation runs, for each dataset sample:

```
persist run plan → call agent → fetch, normalize, and persist trace → plan metrics → compute and persist results
```

- **Agent callers** execute your agent for a sample and return a request ID.
- **Trace source clients** resolve that request ID to a trace and fetch its records.
  [Arize Phoenix](https://github.com/Arize-ai/phoenix) is supported out of the box.
- **Trace adapters** turn raw records into canonical spans with typed semantics: request, answer, retrieval result sets,
  plan snapshots, executed steps, and LLM usage.
- **Metrics** read only those canonical observations, so they work with any agent whose adapter emits them.

A **repeat** evaluation reuses the traces stored by an earlier run and only recomputes metrics. It never calls the
agent and needs no agent credentials.

## Requirements

- Python 3.12 or 3.13
- [Poetry](https://python-poetry.org/)
- PostgreSQL (a `docker-compose.yml` is included) and the `libpq` client library, for example
  `brew install libpq` or `apt install libpq5`
- A Phoenix instance your agent reports traces to, or your own trace source client
- Optional: an OpenAI-compatible or Gemini API key for the LLM-judge metrics, and an
  [Orbitals](https://orbitals.principled.app) API key for the claim-extractor metrics

## Quick start

```bash
git clone https://github.com/SylloTips/syllo-eval.git
cd syllo-eval
poetry install

cp .env.example .env    # set DB_PASSWORD at least; see Configuration
docker compose up -d db
poetry run alembic upgrade head
```

Migrations are applied manually and never on startup. Check which database your `.env` points to before running them.

Import a dataset and list what is registered:

```bash
poetry run syllo-eval dataset import path/to/my-dataset.json --name my-dataset
poetry run syllo-eval dataset list
```

The engine ships no agent integration, so a fresh evaluation needs a caller for your agent first. See
[Connecting an agent](#connecting-an-agent).

## Datasets

A dataset is a JSON file with a list of samples. Only `input_prompt` is required; each other field is the ground truth
for a family of metrics.

```json
{
  "samples": [
    {
      "input_prompt": "What is the warranty period for the X200 router?",
      "ground_truth_output": "The X200 has a 24-month warranty.",
      "document_ids": ["doc-warranty-policy"],
      "snippet_ids": ["doc-warranty-policy#3"],
      "plan": [
        {"operation": "retrieve", "instruction": "Find the X200 warranty terms", "parameters": {"top_k": 5}},
        {"operation": "answer", "instruction": "Answer with the warranty period"}
      ]
    }
  ]
}
```

| Field | Used by |
|---|---|
| `ground_truth_output` | answer correctness, contextual precision and recall |
| `document_ids`, `snippet_ids` | set precision and recall, nDCG@10 |
| `plan` | plan efficiency and plan correctness |

Plan steps accept exactly `operation`, `instruction`, and optional JSON `parameters`; any other field is rejected.
Dataset names are unique, so importing a name that already exists fails.

## Connecting an agent

A caller implements one method: run the agent for a sample and return the request ID that identifies its trace.

```python
import httpx

from syllo_eval.model import Sample


class MyAgentCaller:
  async def call(self, sample: Sample) -> str:
    async with httpx.AsyncClient(base_url='http://localhost:8080') as client:
      response = await client.post('/chat', json={'message': sample.input_prompt})
    response.raise_for_status()
    return response.json()['request_id']
```

With the default Phoenix client, the request ID must be recorded as a `request_id` attribute on the trace's root span.

An adapter turns the fetched records into canonical observations for metrics to read. This one extends the built-in
Phoenix adapter with the question, the answer, and the selected snippets, assuming the agent's root span records the
input `{"message": ...}` and the output `{"answer": ..., "snippet_ids": [...]}`, with snippets in no relevance order:

```python
from typing import Any

from syllo_eval.evaluation.trace_adapter import PhoenixTraceAdapter, TraceProcessingResult
from syllo_eval.trace_semantics import Answer, RetrievalItem, RetrievalResult


class MyTraceAdapter(PhoenixTraceAdapter):
  def normalize(self, trace_id: str, records: list[dict[str, Any]]) -> TraceProcessingResult:
    result = super().normalize(trace_id, records)
    for span in result.spans:
      if span.span_type == 'agent_root' and isinstance(span.input_data, dict) and isinstance(span.output_data, dict):
        request: dict[str, Any] = span.input_data
        output: dict[str, Any] = span.output_data
        span.semantics.request = request['message']
        span.semantics.answer = Answer(text=output['answer'])
        items = [RetrievalItem(id=snippet_id) for snippet_id in output['snippet_ids']]
        span.semantics.retrieval = [RetrievalResult(kind='snippet', stage='selected', items=items, ranked=False)]
    return result
```

Register callers and adapters by agent name in a service factory:

```python
from syllo_eval.service import EvaluationService
from syllo_eval.settings import Settings


def build_service(settings: Settings) -> EvaluationService:
  return EvaluationService(
    settings,
    callers_by_agent_name={'my-agent': MyAgentCaller()},
    trace_adapters_by_agent_name={'my-agent': MyTraceAdapter()},
  )
```

Agents without a registered adapter get no answer, retrieval, or plan, and metrics missing their input are recorded as
skipped. See [trace normalization](docs/trace-normalization.md) for the full adapter contract.

To read traces from a backend other than Phoenix, pass a `trace_client` implementing `TraceSourceClient`
(`get_trace_id_by_request_id` and `get_trace_json`), plus an adapter for its records unless they have Phoenix's shape.
A `trace_integration` replaces fetching, normalization, and persistence altogether: it must return a validated
`TraceProcessingResult` that is already stored in the evaluation database.

Callers and adapters you inject stay owned by you, and must support concurrent calls.

### Your own CLI and HTTP API

Both entry points accept a service factory, so a small package of your own gets the full CLI and API with your agents
registered:

```python
# my_eval/cli.py
from functools import partial

from syllo_eval.API import cli
from my_eval.service import build_service

main = partial(cli.main, service_factory=build_service, prog='my-eval')
```

```toml
# pyproject.toml
[tool.poetry.scripts]
my-eval = "my_eval.cli:main"
```

```python
# my_eval/app.py
from syllo_eval.API.app import create_app
from my_eval.service import build_service

app = create_app(service_factory=build_service, title='my-eval')
```

The engine's migrations ship inside the `syllo_eval` package. To run them from your package, add `alembic` and
`sqlalchemy` to your dependencies and create an `alembic.ini` with:

```ini
[alembic]
script_location = syllo_eval:migrations
```

## Running evaluations

### CLI

```bash
# Fresh run: call the agent for every sample, then score
poetry run my-eval --agent-name my-agent --agent-version-tag v1 --dataset-id <dataset-uuid> \
  --metrics answer_correctness_judge,set_recall_snippet \
  --report-path reports/run.json

# Recompute metrics over the traces of an earlier run, without calling the agent
poetry run my-eval repeat <evaluation-run-uuid> --metrics set_recall_snippet
```

Without `--metrics`, a fresh run uses every metric available with your configuration, and a repeat uses the metrics of
the source run. Each command prints a JSON summary with the run ID and status. The exit code is 0 for `COMPLETED` and
`PARTIALLY_COMPLETED` runs, and non-zero when the run fails. Run `--help` on any command for all options.

### HTTP API

```bash
poetry run uvicorn my_eval.app:app --port 8005
```

| Method | Path | Description |
|---|---|---|
| `POST` | `/evaluations` | Start a run in the background (`agent_name`, `agent_version_tag`, `dataset_id`, optional `metrics`) |
| `GET` | `/evaluations` | List runs, newest first |
| `GET` | `/evaluations/{id}` | Run status with per-sample counts |
| `GET` | `/evaluations/{id}/report` | Aggregated report of a finished run |
| `POST` | `/evaluations/{id}/repeat` | Recompute metrics over the stored traces of a finished run |
| `POST` | `/evaluations/{id}/cancel` | Best-effort cancellation; the run ends as `FAILED` |
| `POST` | `/datasets` | Import a dataset payload |
| `GET` | `/datasets` | List datasets |

Interactive documentation is served at `/docs`.

### Python

```python
import asyncio
from uuid import UUID

from syllo_eval.settings import Settings, load_settings_env


async def main() -> None:
  load_settings_env()
  async with build_service(Settings()) as service:
    run = await service.run_evaluation(
      agent_name='my-agent',
      agent_version_tag='v1',
      dataset_id=UUID('<dataset-uuid>'),
    )
    report = await service.get_evaluation_report(run.id)
    print(report.model_dump_json(indent=2))


asyncio.run(main())
```

## Metrics

| Metric | Measures | Needs |
|---|---|---|
| `llm_calls` | Number of LLM spans in the trace | — |
| `plan_efficiency` | Expected plan steps divided by executed steps | `plan` |
| `set_precision_document`, `set_precision_snippet` | Precision of the selected IDs | `document_ids` / `snippet_ids` |
| `set_recall_document`, `set_recall_snippet` | Recall of the selected IDs | `document_ids` / `snippet_ids` |
| `ndcg_at_10_document`, `ndcg_at_10_snippet` | Binary-relevance nDCG@10 of the ranked selection | `document_ids` / `snippet_ids` |
| `answer_correctness_judge` | Agreement of the answer with the expected output | LLM judge, `ground_truth_output` |
| `plan_correctness_judge` | Agreement of the executed plan with the expected plan | LLM judge, `plan` |
| `contextual_precision_document_judge`, `contextual_precision_snippet_judge` | Whether relevant context ranks first | LLM judge, `ground_truth_output` |
| `contextual_recall_document_judge`, `contextual_recall_snippet_judge` | How much of the expected answer the context supports | LLM judge, `ground_truth_output` |
| `contextual_recall_document_claim_extractor`, `contextual_recall_snippet_claim_extractor` | Contextual recall over extracted claims | LLM judge, Orbitals, `ground_truth_output` |

Retrieval metrics score the agent's final `selected` context, separately for documents and snippets. Rank-based
metrics skip result sets the adapter didn't mark as ranked. Judge metrics are available only when `LLM_JUDGE_PROVIDER`
is set, and claim-extractor metrics only when `ORBITALS_API_KEY` is also set.

To add your own, subclass `SpanEvaluationMetric` or `SpanGroupEvaluationMetric` from
`syllo_eval.evaluation.metrics.contracts` and pass instances as `custom_metrics=[...]` to `EvaluationService`.
`syllo_eval.evaluation.metrics.implementations.demo.response_length` is a minimal example.

## Configuration

Settings are read from environment variables, and entry points also load a `.env` file from the working directory.
In Python, you can build `Settings` directly instead.

| Variable | Default | Purpose |
|---|---|---|
| `DB_HOST`, `DB_PORT`, `DB_NAME`, `DB_USER` | `localhost`, `5432`, `pg-syllo-eval`, `postgres` | PostgreSQL connection |
| `DB_PASSWORD` | — | Required |
| `PHOENIX_BASE_URL` | `http://localhost:6006` | Phoenix server |
| `PHOENIX_API_KEY`, `PHOENIX_PROJECT_ID` | — | Phoenix authentication and project |
| `PHOENIX_REQUEST_ID_EXCLUDED_ROOT_SPAN_NAMES` | `[]` | JSON list of root span names to ignore when resolving request IDs |
| `LLM_JUDGE_PROVIDER` | — | `openai` or `gemini`; enables the judge metrics |
| `OPENAI_API_KEY`, `OPENAI_BASE_URL`, `OPENAI_MODEL` | —, `https://api.openai.com/v1`, `gpt-5-mini` | OpenAI judge |
| `OPENAI_USE_RESPONSES_API` | `true` | Set to `false` for OpenAI-compatible endpoints without the Responses API |
| `GOOGLE_API_KEY`, `GEMINI_MODEL` | —, `gemini-3.1-flash-lite` | Gemini judge |
| `LLM_JUDGE_MAX_CONCURRENT_REQUESTS` | `5` | Concurrent judge requests |
| `ORBITALS_API_KEY` | — | Enables the claim-extractor metrics |
| `EVALUATION_MAX_CONCURRENT_SAMPLES` | `1` | Samples evaluated in parallel |
| `EVALUATION_MAX_CONCURRENT_TASKS` | `10` | Metric computations in parallel per sample |
| `EVALUATION_SAMPLE_TRACE_TIMEOUT_SECONDS` | — | Deadline for the agent call and trace ingestion per sample |
| `EVALUATION_SAMPLE_COMPUTE_TIMEOUT_SECONDS` | — | Deadline for metric computation per sample |
| `LOG_LEVEL` | `INFO` | Logging level |

`syllo_eval/settings.py` lists every option, including timeouts and retry backoffs.

## Development

```bash
poetry install
poetry run ruff check .
poetry run ruff format .
poetry run mypy syllo_eval --check-untyped-defs --explicit-package-bases
poetry run python -m unittest discover -s . -p 'test_*.py'
```

The full suite includes database integration tests, which use the database configured in `.env`.

Further reading:

- [Trace normalization](docs/trace-normalization.md): the adapter contract and retrieval semantics
- [Database diagram](docs/ER_diagram.mmd)
- [Adding a metric](.agents/skills/add-metric/SKILL.md)
- [AGENTS.md](AGENTS.md): architecture boundaries and contributor rules
