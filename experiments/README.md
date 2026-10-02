# Paper experiments

This folder reproduces Section 5 of *Evaluating Enterprise Agents From the Traces They Leave*. It is a separate Poetry
project that uses `syllo-eval` only through its public extension points: callers, trace clients, adapters and custom
metrics. It is not part of the published package.

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

## Layout

| Path | Contents |
|---|---|
| `configs/` | every parameter of the study; values still red in the paper draft are marked "paper placeholder" |
| `cli.py`, `config.py`, `manifest.py`, `judge.py`, `service.py` | the CLI, configuration, run manifest, shared judge and evaluation services; this folder is their import root |
| `tests/` | unit tests on synthetic data |
| `outputs/` | gitignored: manifest, raw trace exports, computations, annotation packets |
| `data/` | gitignored: benchmark downloads and search indexes |

## Setup

Run all commands from this folder. The parent project's database and migrations are shared: start Postgres with the
root `docker-compose.yml`, and apply migrations from the root folder after confirming which database `.env` points to.

```bash
poetry install
poetry run syllo-exp configurations   # validate the configs and list the agent configurations
poetry run syllo-exp steps            # state of every recorded experiment step
```

Settings such as database credentials and the judge API key are read by `syllo-eval`'s settings blocks. Use a dedicated
database for the campaign (for example `DB_NAME=syllo-eval-paper`) with `DB_TIMEOUT=30` and `DB_MAX_SIZE=40`.

## Verification

```bash
poetry run python -m unittest discover -s tests
poetry run mypy . --check-untyped-defs --explicit-package-bases
```

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
