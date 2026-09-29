# AGENTS.md

Guidance for working on Syllo-eval, an open-source agent evaluation library shared by Python, CLI, and HTTP consumers.
Keep this file focused on working rules, architectural boundaries, and where to look.

## Working rules

- Make the smallest correct change. Reuse existing helpers; avoid speculative abstractions and redundant defensive checks.
- Preserve validation at external boundaries, meaningful error handling, and resource cleanup.
- Use 2-space indentation, single quotes, and a 120-character line limit; Ruff enforces formatting.
- Update affected architecture documentation in the same change. Update this file when a boundary, working rule,
  or navigation entry changes; put implementation details in the owning documentation or code.
- This repository is public. Never add code, names, fixtures, or configuration specific to a particular agent or
  deployment; downstream packages supply those through the extension points. Tests and examples use synthetic data.

## Commands and verification

```bash
poetry install
poetry run ruff check .
poetry run ruff format .
poetry run mypy syllo_eval --check-untyped-defs --explicit-package-bases

# Full suite; includes database integration tests
poetry run python -m unittest discover -s . -p 'test_*.py'

poetry build
```

After each file edit, run the formatter and mypy commands above. Run focused tests appropriate to the change;
do not repeat passing tests without new changes or evidence. Check the configured database before integration tests
or migrations: tests may load `.env`.

## Architecture and extension boundaries

Fresh evaluation: **persist run plan → call agent → fetch/normalize/persist trace → plan metrics → compute/persist results**.
Repeat evaluation reuses stored traces and runs only metric planning and computation; it must not require agent credentials
or construct callers/trace integrations.

- **Public boundary:** `EvaluationService` serves library, CLI, and HTTP consumers. The package is `syllo_eval`,
  distributed as `syllo-eval`. Downstream packages extend it by injecting callers, trace clients, adapters, and metrics.
- **Agent independence:** callers execute agents, source clients fetch records, adapters produce canonical traces,
  and metrics consume canonical observations. Agent-specific interpretation belongs in downstream adapters or custom
  metrics. The engine registers no caller by default.
- **Metrics:** built-ins and custom instances can be mixed. Keep built-in inventory in `evaluation/metrics/available.py`;
  the in-memory registry owns implementations, the planner owns span selection/grouping, and the executor owns
  computation status and persistence. Preserve custom span types, selection hooks, and group targets.
- **Traces:** validate identities, parent graphs, timestamps, and semantic references before transactional persistence.
  Namespace source IDs when necessary. Identical ingestion is idempotent; conflicting snapshots must not overwrite repeat inputs.
  Missing observations differ from observed empty values; never invent retrieval membership, ranks, step links, or timing.
- **Benchmark semantics:** built-in RAG metrics assess the `selected` retrieval stage, separately by result kind;
  rank metrics skip result sets not marked `ranked`.
  Custom metrics may target other stages/spans. Do not silently concatenate independent rankings.
  Dataset plans use generic `operation`, `instruction`, and optional JSON parameters.

## Configuration and resource ownership

- Declare configuration in `EnvSettings` blocks with an `env_prefix`; use `validation_alias` for fixed deployment names.
  Engine settings live in `settings.py`; integration/process settings live with their owner. Never read the environment outside
  settings blocks, except the documented presence checks in logging.
- Explicit settings override environment values. Blank values are unset; malformed nonblank values must fail validation.
  Entry points load `.env`; package imports must not. Keep `from_env()` only for process-level behavior that cannot be declarative.
- Code defaults are deployment defaults; update `test_settings.py` when changing a default. CLI defaults must not override
  configured values accidentally.
- Resources created by the service/prepared execution are owned and closed by it, including on preparation failure or
  cancellation. Injected pools, callers, clients, and custom metric resources remain caller-owned; injected pools must already
  be initialized. Do not introduce a process-global database pool. Custom metric instances must support concurrent calls.
- Allocate judge/claim clients only for selected metrics. Cancel and await sibling judge calls on failure/cancellation;
  preserve usage from completed responses. Judge clients validate structured output before returning it.

## Persistence and execution contracts

- Use repository operations through `UnitOfWork`; use `TransactionalUnitOfWork` for atomic multi-operation changes.
  Plain `UnitOfWork` does not create a transaction.
- Agent/trace failures fail the sample. Individual metric failures become failed computations; orchestration or persistence
  failures fail the compute phase. Persist missing-target skips so reporting retains metric coverage.
- Trace and compute have separate optional deadlines. The trace deadline includes the agent call; lifecycle persistence stays
  outside deadlines. On cancellation, persist sample failure with shielding, then propagate cancellation.
- Reports read persisted data. Use saved sample/metric plans for totals and coverage, including missing samples and metrics
  with zero computations. Keep agent usage separate from judge usage and deduplicate by trace/call identity.
- Alembic migrations are manually applied raw SQL in `migrations/`, without ORM autogeneration. Never run migrations automatically
  from application startup or container commands. Confirm the database target before `poetry run alembic upgrade head`.

## Where to look

Paths below are relative to `syllo_eval/`, unless linked otherwise. Read only the areas relevant to the task.

| Task | Start here |
|---|---|
| Service/API/CLI assembly | `service.py`, `API/config.py`, `API/app.py`, `API/cli.py` |
| Execution, cancellation, repeats | `orchestration/evaluation_orchestrator.py`, `execution/sample_executor.py`, `execution/agent_caller/` |
| Canonical models, adapters, ingestion | `model.py`, `trace_semantics.py`, `evaluation/trace_adapter.py`, `evaluation/trace_processor.py` |
| Metrics, judges, reporting | `evaluation/metrics/`, `evaluation/metric_planner.py`, `evaluation/plan_executor.py`, `evaluation/judge/`, `evaluation/evaluation_report.py` |
| Datasets, settings, persistence | `datasets/`, `settings.py`, `infrastructure/`, root `migrations/` |

Do not append endpoint catalogs, field inventories, provider workarounds, release history, or agent-specific walkthroughs here.
Keep those with their implementation or focused documentation, and link to them when useful.
