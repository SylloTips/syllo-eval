# Reproducing the paper's results

This guide lists the commands a reviewer runs to reproduce the results of *Evaluating Enterprise Agents From the
Traces They Leave*, and the output each command should produce. It grows with the reproduction package. For now it
covers the first stage: obtaining the three benchmarks and loading them into the evaluation database.

The protocol behind each step is in [`METHODOLOGY.md`](METHODOLOGY.md).

## Requirements

- macOS or Linux. The downloader uses POSIX file locks.
- Python 3.12 or 3.13, and [Poetry](https://python-poetry.org/) 1.8 or later.
- Docker with Docker Compose, to run PostgreSQL. Any PostgreSQL server reachable through the `DB_*` settings works too.
- The `libpq` client library, for example `brew install libpq` or `apt install libpq5`.
- About 2 GB of free disk space and 4 GB of free memory: converting the largest benchmark loads a 2.5 GB text column.
- Network access to `huggingface.co` and `raw.githubusercontent.com`. This stage needs no accounts or API keys.

## Setup

Clone the repository and check out the commit or tag cited in the paper. Then run these commands from the repository
root:

```bash
poetry install                        # the syllo-eval library and its migration tool
cp .env.example .env                  # then set DB_PASSWORD; the other keys are not needed yet
docker compose up -d db               # PostgreSQL, with a new database named pg-syllo-eval
poetry run alembic upgrade head       # create the evaluation tables
```

The migrations apply to the database that `.env` points to. By default, that is the new database inside the Compose
container.

The experiments are a separate Poetry project. Install it, then run every later command from `experiments/`:

```bash
cd experiments
poetry install
```

## Stage 1: the benchmarks

The paper uses three public benchmarks:

| Benchmark | Source, pinned to a commit | What the experiments use |
|---|---|---|
| EnterpriseRAG-Bench (ERB) | Hugging Face `onyx-dot-app/EnterpriseRAG-Bench` @ `69916e3` | 480 answerable questions over 511,962 company documents |
| WixQA | Hugging Face `Wix/WixQA` @ `d662dc4` | 400 customer-support queries over 6,221 help-center articles |
| τ²-bench retail | GitHub `sierra-research/tau2-bench` @ `fc0055d` (v1.0.1) | 114 customer-service tasks with reference tool calls, and the retail environment its agent runs in |

[`configs/benchmarks.yaml`](configs/benchmarks.yaml) pins every file by full commit, size and SHA-256. Each step
records its outcome in `outputs/manifest.jsonl` and is safe to run again.

### 1. Download

```bash
poetry run syllo-exp benchmarks fetch
```

- **What it does:** downloads ten files, about 1.5 GB, into `data/<benchmark>/raw/`. A file is kept only once its
  size and SHA-256 match its pin.
- **Interruptions:** an interrupted download resumes where it stopped when you run the command again.
- **Files already present** are verified again rather than downloaded.

Expected output (paths shortened):

```
erb: 2 files verified under .../experiments/data/erb/raw
wixqa: 3 files verified under .../experiments/data/wixqa/raw
tau2: 5 files verified under .../experiments/data/tau2/raw
```

To check the bytes independently of our code, hash the files and compare each hash with the `sha256` values in
`configs/benchmarks.yaml`:

```bash
find data -path '*/raw/*' -type f ! -name '*.lock' -exec shasum -a 256 {} +    # Linux: sha256sum
```

### 2. Convert and check

```bash
poetry run syllo-exp benchmarks convert
```

This turns each benchmark into a syllo-eval dataset and checks it. For each benchmark it writes four files to
`data/<benchmark>/`:

| File | Contents |
|---|---|
| `dataset.json` | the samples to import |
| `samples.jsonl` | each sample's benchmark fields, such as the question id and category |
| `claims.json` | the gold claims |
| `report.json` | the checks and the statistics to compare with the table below |

The conversion rules are listed under Datasets in [`METHODOLOGY.md`](METHODOLOGY.md).

Expected output: three `completed` lines and one warning. The warning is expected: ERB's ten `high_level` questions
have no gold documents by design.

```
erb: completed, written to .../experiments/data/erb
  warning: 10 samples have no gold documents; set recall scores 1.0 on empty labels, so label-based analyses must exclude them: 'qst_0471', 'qst_0472', 'qst_0473', 'qst_0474', 'qst_0475' and 5 more
wixqa: completed, written to .../experiments/data/wixqa
tau2: completed, written to .../experiments/data/tau2
```

Each `report.json` should show no errors and these values:

| Benchmark | Values in `report.json` |
|---|---|
| ERB | `samples` 480 (`basic` 175, `semantic` 125, others 180); `samples_with_gold_documents` 470; `distinct_gold_documents` 723; `claims` 300 samples and 1,013 claims; `knowledge_base.documents` 511,962, with 4 ids renamed |
| WixQA | `samples` 400 (ExpertWritten 200, Simulated 200); `gold_references` 488; `distinct_gold_documents` 340; `knowledge_base.articles` 6,221 |
| τ²-bench | `samples` 114 (train 74, test 40); `plan_steps` 550; `samples_without_plan` tasks 24 and 57; `samples_with_nl_assertions` 40 tasks |

For example, `python3 -m json.tool data/erb/report.json` prints the full ERB report.

### 3. Import

```bash
poetry run syllo-exp benchmarks import
```

This stores the three datasets and ERB's gold claims in the database. The first run prints the following; the dataset
ids differ on every machine:

```
erb: dataset erb-69916e3 imported (<id>); claims added: {'expected_claims_gold': 300}
wixqa: dataset wixqa-d662dc4 imported (<id>); claims added: {}
tau2: dataset tau2-retail-v1.0.1 imported (<id>); claims added: {}
```

Running it again prints `already present` for each dataset and adds nothing. It first checks that the stored datasets
are identical to the conversion.

To check the database directly, run this from the repository root:

```bash
docker compose exec -T db psql -U postgres -d pg-syllo-eval \
  -c "select d.name, count(*) as samples from dataset d join sample s on s.dataset_id = d.id group by d.name order by d.name"
```

```
        name        | samples
--------------------+---------
 erb-69916e3        |     480
 tau2-retail-v1.0.1 |     114
 wixqa-d662dc4      |     400
```

### Checking the stage

`poetry run syllo-exp steps` lists every recorded step. After this stage it shows nine `completed` steps: `fetch`,
`convert` and `import` for each benchmark.

## Troubleshooting

| Symptom | What to do |
|---|---|
| The download stopped or failed | Run `benchmarks fetch` again. It resumes from the partial `.part` file and retries transient network errors. |
| `Another fetch is already downloading` | Another `fetch` is still running. Wait for it to finish. The `.lock` file it leaves is harmless. |
| `... has SHA-256 ..., expected ...` for a file in `raw/` | The local copy is damaged. Delete that file and run `benchmarks fetch` again. |
| `... was converted from other pins` at import | The configuration changed after the conversion. Run `benchmarks convert` again. |
| Database connection errors | Check that `docker compose ps` shows the `db` service as healthy, and that `.env` sets `DB_PASSWORD`. |

## Data licenses

The three benchmarks are released under the MIT license. Their notices apply to any data derived from them:
- ERB: Copyright (c) 2026 DanswerAI, Inc.
- WixQA: cite "Wix.com AI Research".
- τ²-bench: Copyright (c) 2025 Sierra Research.
