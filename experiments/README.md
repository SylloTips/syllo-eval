# Paper experiments

This folder reproduces Section 5 of *Evaluating Enterprise Agents From the Traces They Leave*. It is a separate Poetry
project and is not part of the published package. It plugs into `syllo-eval` through its public extension points:
callers, trace clients, adapters and custom metrics.

It also uses the library's persistence layer, the `UnitOfWork` repositories in `syllo_eval.infrastructure`, for
ground truths under its own keys, which the dataset format cannot carry (`benchmarks/common.py`). It relies on
`datasets.get_by_name`, `samples.list_by_dataset`, `ground_truths.list_by_key` and `ground_truths.bulk_create`. The
library's test suite does not run this folder's tests, so changes to these methods must keep them passing.

## What the experiments measure

Syllo-eval scores an agent from the trace it leaves in an observability platform, and each metric reads only the part
of the trace it assesses. The paper asks three questions:

| | Question | Evidence | Paper |
|---|---|---|---|
| RQ1 | Do the judges agree with human and benchmark references? | judge decisions vs. gold documents and facts; answer grades vs. two annotators; plan grades vs. task rewards | Table 2 |
| RQ2 | Do the metrics rank agents of different quality correctly? | Kendall τ against reference rankings; significant pairs; configurations degraded on purpose | Table 3 |
| RQ3 | Do the judges stay accurate, and affordable, as inputs grow? | retrieved lists of 5–80 documents; trace-length terciles and padding up to 128k tokens; judge cost | Table 4 |

- **Benchmarks:** EnterpriseRAG-Bench (ERB), WixQA and τ²-bench retail.
- **Agents:** a Dify ReAct agent, a smolagents CodeAgent and Open Deep Research, plus τ²-bench's own tool-calling agent.
- **Models and configurations:** two agent models; [`configs/configurations.yaml`](configs/configurations.yaml) lists
  all 22 configurations.
- **Comparisons:** Syllo-eval runs against DeepEval and two ablations, all with the same judge model:
  - Syllo-eval-SC judges all items or claims in a single call;
  - Syllo-eval-WT reads the whole trace instead of the target spans.

[`METHODOLOGY.md`](METHODOLOGY.md) fixes the protocol: evaluation units, aggregation and statistics.
[`REPRODUCIBILITY.md`](REPRODUCIBILITY.md) lists the commands a reviewer runs to reproduce the results, with the
expected output of each step.

## Layout

| Path | Contents |
|---|---|
| `configs/` | every parameter of the study; values still red in the paper draft are marked "paper placeholder" |
| `cli.py`, `config.py`, `manifest.py`, `judge.py`, `service.py` | the CLI, configuration, run manifest, shared judge and evaluation services; this folder is their import root |
| `benchmarks/` | pinned downloads, converters and import of the three benchmarks |
| `tests/` | tests on synthetic data; `ImportBenchmarkTest` also writes to the configured database (see Verification) |
| `outputs/` | gitignored: manifest, raw trace exports, computations, annotation packets |
| `data/` | gitignored: benchmark downloads and search indexes |

## Setup

Run all commands from this folder. The parent project's database and migrations are shared: start Postgres with the
root `docker-compose.yml` and apply migrations from the root folder.

```bash
poetry install
poetry run syllo-exp configurations   # validate the configs and list the agent configurations
poetry run syllo-exp steps            # state of every recorded experiment step
```

Settings such as database credentials and the judge API key are read by `syllo-eval`'s settings blocks. Commands that
need them load `experiments/.env` first and then the repository's `.env`, so a value in the first wins. Process
environment variables override both.

Use a dedicated database for the campaign, for example `DB_NAME=syllo-eval-paper` in `experiments/.env`, with
`DB_TIMEOUT=30` and `DB_MAX_SIZE=40`. Alembic reads only the `.env` of the folder it runs from, so create and migrate
that database from the repository root, naming it in the environment:

```bash
docker compose exec db createdb -U postgres syllo-eval-paper
DB_NAME=syllo-eval-paper poetry run alembic upgrade head
```

## Benchmarks

```bash
poetry run syllo-exp benchmarks fetch     # pinned files into data/<benchmark>/raw, size and SHA-256 checked (1.5 GB)
poetry run syllo-exp benchmarks convert   # dataset.json, samples.jsonl, claims.json and report.json per benchmark
poetry run syllo-exp benchmarks import    # the datasets and their claim ground truths, into the configured database
```

- **Pins:** [`configs/benchmarks.yaml`](configs/benchmarks.yaml) fixes every file by commit, size and SHA-256.
- **Scope:** each step takes `--only erb wixqa tau2`, records its outcome in the manifest, and is safe to re-run.
  Fetching resumes interrupted downloads.
- **Checks:** a conversion fails on a wrong sample count, a duplicate or unstripped prompt, a gold document missing
  from the knowledge base, or a malformed claim list. A failed conversion leaves no report, and the import refuses a
  conversion that failed or was made from pins other than the current ones.
- **Re-runs:** importing again changes nothing in an existing dataset. It first checks that the dataset holds the same
  prompts, answers, built-in ground truths and claims, and refuses otherwise. A fetch refuses to run while another
  fetch of the same file is in progress.

| Dataset | Samples | Gold documents | Claims (`expected_claims_gold`) |
|---|---|---|---|
| `erb-69916e3` | 480 | 470 samples, 723 distinct of 511,962 | 300 samples, 1,013 claims |
| `wixqa-d662dc4` | 400 | 400 samples, 340 distinct of 6,221 | none |
| `tau2-retail-v1.0.1` | 114 | none | none (112 expected plans, 550 steps) |

`samples.jsonl` keeps each sample's benchmark fields (question id and type, config and row, tau2 split and grading
fields), keyed by `sample_key` and joined to the database through the input prompt. `report.json` adds the length
statistics that the search tool's per-document cap will be chosen from. [`METHODOLOGY.md`](METHODOLOGY.md) lists the
conversion rules.

The benchmarks are MIT-licensed. A release of derived data must keep their notices:
- ERB: Copyright (c) 2026 DanswerAI, Inc.
- WixQA: cite "Wix.com AI Research".
- τ²-bench: Copyright (c) 2025 Sierra Research.

## Verification

```bash
poetry run python -m unittest discover -s tests
poetry run mypy . --check-untyped-defs --explicit-package-bases
```

`ImportBenchmarkTest` (in `tests/test_benchmarks_common.py`) runs against the database that the `.env` files select:
- it creates and deletes `test-benchmark-*` datasets;
- it is skipped when that database is unreachable, and fails when the database is not migrated.

Check which database is configured before running the suite.

The parent project's `poetry run ruff check .` and `poetry run ruff format .` also cover this folder.

## How the campaign runs

1. **Collect.** Fresh runs call each agent and archive its raw traces. They compute only deterministic metrics.
2. **Judge.** Every judge evaluation repeats a collection run on its stored traces:
   - the main pass and two retests;
   - the gold-claim pass;
   - the SC and WT ablations.

   DeepEval scores the same units offline.
3. **Construct.** RQ3 lists and padded traces are imported as new traces.
4. **Annotate.** Humans grade 180 answers and verify ERB facts.
5. **Analyze.** Tables 2–4 and every red placeholder are regenerated from the exports.

A pilot on a small slice gates the full campaign. It fixes the judge's temperature and thinking level and checks cost.
