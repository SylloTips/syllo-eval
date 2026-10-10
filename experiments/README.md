# Paper experiments

This folder reproduces Section 5 of *Evaluating Enterprise Agents From the Traces They Leave*. It is a separate Poetry
project and is not part of the published package. It plugs into `syllo-eval` through its public extension points:
callers, trace clients, adapters and custom metrics.

It also uses the library's persistence layer, the `UnitOfWork` repositories in `syllo_eval.infrastructure`, for
ground truths under its own keys, which the dataset format cannot carry (`benchmarks/common.py`), and for the stored
traces that the whole-trace ablation reads (`ablation_metrics.py`) and the failures a judge run recorded (`cli.py`).
It relies on `datasets.get_by_name`, `samples.list_by_dataset`, `ground_truths.list_by_key`, `ground_truths.bulk_create`,
`spans.list_by_trace` and `metric_computations.list_by_evaluation_run`.

The ablations and the DeepEval baseline also rely on a few library internals:
- `_judge_client` and `_rubric_addition`, which the built-in judge metrics set from their constructor arguments, and
  `_extract_expected_answer` of Answer Correctness;
- the `syllo_eval.evaluation.metric_support` helpers that `ablation_metrics.py` and `deepeval_baseline.py` import to
  render prompts, traces and test cases;
- the built-in v1 prompt templates that `prompts/` edits: a new built-in prompt version needs new ablation prompts;
- `LangChainLlmJudgeClient._usage` and `GeminiLlmJudgeClient._retry_delay_seconds`, with which the DeepEval judge reads
  usage and retries rate limits as the library's judge client does.

Collection wraps the library's `PhoenixClient` to archive the traces it fetches (`collect.py`). The τ²-bench,
smolagents and Open Deep Research adapters read the flattened records that client returns, `attributes.*` keys
included, after `PhoenixTraceAdapter` has decoded them (`agents/tau2.py`, `agents/code_agent.py`, `agents/odr.py`).

The library's test suite does not run this folder's tests, so changes to any of these must keep them passing.

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
| `collect.py`, `run_outputs.py`, `agents/` | collection runs, the files they write, and each agent stack's caller and trace adapter |
| `tau2_runtime/` | τ²-bench's run of one task with traces, run in its own environment (see τ²-bench agent) |
| `ablation_metrics.py`, `prompts/` | the SC and WT ablations, recall over the gold claims, and the ablation prompts |
| `deepeval_baseline.py`, `deepeval_env.py` | the DeepEval baseline, and the switches DeepEval reads when it is imported |
| `dify/` | the Dify agent's DSL, configuration guide, and the Compose override that runs Dify beside Phoenix |
| `indexing/` | the search indexes of the ERB and WixQA knowledge bases: embedding model, Qdrant vector store, pipeline |
| `search_tool/` | the agents' search tool: an MCP server over the search index of one knowledge base |
| `scripts/` | `index-benchmark.sh`, which launches the indexing of one knowledge base; `setup-tau2.sh`, which creates the τ²-bench environment; `dify.sh`, which runs Dify |
| `docker-compose.yml`, `Dockerfile` | the campaign environment (see Setup) |
| `tests/` | tests on synthetic data; `ImportBenchmarkTest` and `QdrantServerTest` also write to the configured database and Qdrant (see Verification) |
| `outputs/` | gitignored: manifest, raw trace exports, τ²-bench simulations and rewards, computations, annotation packets |
| `data/` | gitignored: benchmark downloads and search indexes |

## Setup

Run all commands from this folder.

### Campaign environment

`docker-compose.yml` here runs Postgres (the campaign and Phoenix databases), Qdrant, Phoenix, and an `experiments`
image that runs `syllo-exp` and the migrations. Settings come only from `experiments/.env`, which must set
`DB_PASSWORD`; `DB_NAME` defaults to `syllo-eval-paper`. Ports bind to localhost, and Postgres uses host port 5433.

```bash
docker compose up -d
docker compose run --rm experiments alembic upgrade head   # after checking DB_NAME
docker compose run --rm experiments syllo-exp configurations
```

Run `docker compose build experiments` after changing the code. `data/` and `outputs/` are mounted from this folder.

### Without Docker

The parent project's database and migrations are shared: start Postgres with the root `docker-compose.yml` and apply
migrations from the root folder.

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

- **Pins:** [`configs/benchmarks.yaml`](configs/benchmarks.yaml) fixes every file by commit, size and SHA-256. For
  τ²-bench, it also fixes the files that its agent runs on: the retail database, the agent's policy and the customer
  simulator's guidelines.
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

## Search indexes

The agents search the ERB and WixQA knowledge bases through the search tool, which reads one vector-store collection
per knowledge base. A collection holds one point per document, with its embedding and its BM25 term weights: the
benchmarks label relevance per document, so documents are never chunked.

```bash
scripts/index-benchmark.sh wixqa   # about 3M tokens, a few minutes: a quick check of the setup
scripts/index-benchmark.sh erb     # about 670M tokens, about 12 hours at the deployment's 1M tokens a minute
```

- **Launch:** the script runs `syllo-exp index embed --only <benchmark>`, then `syllo-exp index load --only
  <benchmark>`, which can also run alone. It writes a copy of its output to `outputs/logs/` and keeps a Mac awake while
  it runs. Both steps resume, so after an interruption run it again.
- **Model:** `embedding` in [`configs/models.yaml`](configs/models.yaml), Cohere Embed 5 Fast at 2,048 dimensions, its
  full size. Documents are embedded as `search_document`; the search tool must embed queries with the same model and
  dimension, as `search_query`.
  - The model is a deployment of an Azure AI Foundry resource, called through Cohere's v2 embed API at
    `<AZURE_FOUNDRY_BASE_URL>/providers/cohere/v2/embed` with the key `AZURE_FOUNDRY_API_KEY`.
  - `AZURE_FOUNDRY_BASE_URL` is the resource endpoint, `https://<resource>.services.ai.azure.com`, without `/models`
    or `/openai/v1`.
- **Embed:** writes the vectors to `data/<benchmark>/index/` in Parquet shards of 2,048 documents, each with the tokens
  billed for it; ERB's take about 4.2 GB.
  - **Resume:** an interrupted run starts again at the first missing shard. Two runs cannot embed the same index at
    once.
  - **Spec:** `spec.json` records the pin of the knowledge-base file, the model, the dimension and the embedded text.
    Shards made with another spec are refused: delete the folder to embed again. Renaming the dataset or re-pinning
    the other files of the benchmark keeps them.
  - **Report:** `report.json`, written last, marks a complete index, with the documents, shards and billed tokens.
  - **Pacing:** requests are spaced evenly at the deployment's `tokens_per_minute`: a request goes once the requests
    before it are at most 2 seconds ahead of that pace, because Azure throttles bursts over windows of seconds.
    Throttled requests, server errors and network failures are retried, each retry printing a warning.
- **Load:** upserts every document into the collection named after its dataset, `erb-69916e3` or `wixqa-d662dc4`,
  then checks that the collection holds exactly the index's documents.
  - The vector store is Qdrant (`indexing/vector_store.py`), at `QDRANT_URL` (default `http://localhost:6333`, the
    Compose service) with the optional key `QDRANT_API_KEY`. A failing request fails only that knowledge base.
  - Each point has two vectors. `dense` is the embedding, compared by cosine: the vectors of the longest documents
    are not unit length. `bm25` holds the BM25 weights that Qdrant computes from the embedded text; Qdrant weighs
    them by inverse document frequency when it searches.
  - The collection's metadata records the dataset, the model, the dimension and the documents' mean BM25 length. An
    existing collection with other vectors or metadata is refused: delete it to load again.
  - The collection keeps its vectors in memory: for ERB, about 4.2 GB of dense vectors and 1 to 2 GB for the BM25
    index, more while loading.
- **Points:** a point's id is the UUIDv5 of the dataset name and document id, so loading again replaces the same
  points. A point's payload:

| Field | Content |
|---|---|
| `document_id` | the id that `dataset.json`'s gold lists use, so a retrieved document is compared with the labels as is |
| `source_document_id` | the id in the benchmark file; ERB's four renamed duplicates have `document_id` `<doc_id>__2` |
| `dataset` | the dataset name, which carries the pinned commit |
| `title`, `text` | the document's title and body, without NUL characters |
| `source_type` | ERB: where the document comes from, such as `slack`, `gmail` or `confluence` |
| `url`, `article_type` | WixQA: the Help Center URL, and `article`, `feature_request` or `known_issue` |

[`METHODOLOGY.md`](METHODOLOGY.md) lists the indexing rules.

## Search tool

The agents search through an MCP server, which offers a single tool, `search_knowledge_base(query)`. It returns the
10 best documents of a hybrid search: the documents whose embeddings are nearest to the query's, and the best BM25
matches, fused by reciprocal rank.

```bash
poetry run syllo-exp search-server --collection erb-69916e3 --port 8101                        # erb, f = 0
poetry run syllo-exp search-server --collection erb-69916e3 --port 8101 --swap-fraction 0.25  # erb/react/sonnet/f0.25
poetry run syllo-exp search-server --collection wixqa-d662dc4 --port 8101                      # wixqa
```

- **Switching configurations:** the agents keep one URL. To switch them to another configuration, stop the server and
  relaunch it on the same port with that configuration's collection and swap fraction, as above. Every call-log record
  carries the server's settings and its time, so a search can be traced to the configuration that ran then.
- **Settings:** the server also publishes its settings (collection, swap fraction, seed and document cap) as the MCP
  resource `search://settings`. The agents list only tools, so they never see it; `collect` reads it to refuse a
  server left on another configuration (see Collection).
- **Endpoint:** streamable HTTP at `http://<host>:<port>/mcp`, stateless, so concurrent agents share one server. It
  listens on 127.0.0.1 unless `--host` says otherwise. The tool's name, description and schema are the same on every
  server.
- **Startup:** a server refuses a collection loaded without its dense and BM25 vectors, or embedded with another model
  or dimension than [`configs/models.yaml`](configs/models.yaml)'s, and warns while Qdrant is still optimizing the
  collection. It needs the Azure Foundry key, to embed queries. Run the servers on the host with `poetry run`: the
  Compose `experiments` container publishes no ports.
- **Output:** one JSON object, both as text and as structured content. `id` is the `document_id` of the gold lists,
  and `call_id` identifies the search in the call log:

  ```json
  {"call_id": "87e80b95…", "query": "…", "results": [{"rank": 1, "id": "…", "title": "…", "text": "…"}, …]}
  ```

  A search unit is a tool output that parses as such an object; anything else is a failed call, since the agents'
  clients do not all keep MCP's error flag. The server's own failures read `The search failed: <reason>`.
- **Determinism:** the same query always returns the same documents. Qdrant ranks the documents both ways, and the
  server cuts and fuses the rankings itself, ordering equal scores by id: Qdrant's own fusion orders ties at random,
  and its cut picks among tied documents by how it stores them.
- **Degraded configurations:** `--swap-fraction f` replaces a fraction f of the results with documents drawn at random
  from the same collection, for RQ2's known order. The draws depend only on `--seed` and the query.
- **Long documents:** `--max-document-chars N` cuts each returned text after N characters and appends ` [truncated]`.
  It is off by default: the cap is still to be chosen from the gold length statistics of the conversion,
  `gold_document_chars` and `longest_gold_document_chars_per_sample` in `data/<benchmark>/report.json`.
- **Call log:** every search that reaches the tool is appended to `--call-log`, by default
  `outputs/search_calls/<collection>-<port>.jsonl`. A record holds the server's settings, the call id, time and query,
  and either the returned ids, the ranks of the random documents and the ids they replaced, or the error. Traces can be
  checked against it: Dify, for one, can leave a call out of its traces. A call that MCP rejects, without a string
  `query`, never reaches the tool.

### Connecting the agents

- **Dify** (1.17.1): add the server under Integrations > Tools > MCP > Add MCP Server (HTTP), at
  `http://host.docker.internal:<port>/mcp`.
  - Dify calls external servers through its `ssrf_proxy`, which refuses private addresses. Set
    `SSRF_PROXY_ALLOW_PRIVATE_DOMAINS=host.docker.internal` in Dify's `docker/.env` and recreate `ssrf_proxy`. On
    Linux, also give `ssrf_proxy` `extra_hosts: ['host.docker.internal:host-gateway']`, and serve on `--host 0.0.0.0`.
  - Dify's classic Agent app switches to function calling whenever the model supports it. A ReAct agent needs a
    Chatflow or Workflow Agent node with the ReAct strategy.
- **smolagents:** the caller connects at `SMOLAGENTS_SEARCH_URL` (see smolagents CodeAgent).
- **Open Deep Research:** the caller connects at `ODR_SEARCH_URL` (see Open Deep Research).

## Agents

### Dify

The low-code agent is built in Dify's UI, as a low-code user would, and called through Dify's service API. Dify traces
it to the campaign's Phoenix with its built-in Arize Phoenix integration; the agent is not instrumented.

```bash
docker compose up -d        # first: Dify's api and worker join this environment's network to reach Phoenix
scripts/dify.sh up -d       # Dify 1.17.1; any docker compose arguments, e.g. scripts/dify.sh down
```

- **Stack:** the first run downloads Dify's `docker/` folder at the pinned release into `data/dify/`, checking its
  SHA-256, and creates Dify's own settings, `data/dify/.env`, from its example. The UI is at `http://localhost`, on
  the port `EXPOSE_NGINX_PORT` sets there. [`dify/compose.override.yaml`](dify/compose.override.yaml) binds Dify's
  ports to localhost and connects its api and worker to Phoenix.
- **App:** [`dify/react-deepseek.yml`](dify/react-deepseek.yml) is the DSL export of the agent, a Chatflow whose
  Agent node uses the ReAct strategy: a classic Agent app switches to function calling for tool-capable models. The
  model is `DeepSeek-V4.1-Flash` on Azure AI Foundry, through Dify's Azure AI Studio provider. The export holds no
  credentials or workspace settings: [`dify/README.md`](dify/README.md) is the configuration guide for the plugins,
  model provider, MCP server, tracing and API key that each Dify instance needs from the UI.
- **One app per configuration:** a configuration's model and benchmark are fixed by the Dify app it calls, not by
  `collect`. Each `react` configuration therefore runs on an app that uses its model and traces to its benchmark's
  Phoenix project (`erb-69916e3` or `wixqa-d662dc4`), with the search server on that benchmark's collection and the
  configuration's swap fraction. `DIFY_API_KEY` selects the app; set it to that app's key before each collection.
  `DIFY_SEARCH_URL` is the search server that the app calls, as this host reaches it (`http://127.0.0.1:8101/mcp` by
  default), which `collect` checks.
- **Request ID:** the caller sends its own request ID as Dify's `trace_id`. Dify writes it as `dify_trace_id` on the
  root of the workflow trace, which holds the agent and its searches; `collect` looks Dify's traces up by that
  attribute. The message ID would find Dify's separate message trace, which holds only the question and the answer.
- **Caller:** `agents/dify.py` streams the answer, because Dify's agent apps do not answer in blocking mode, through
  Dify's nginx at `DIFY_BASE_URL` (`http://localhost/v1` by default).
- **Check:** a pilot (see Collection) runs the app on a few questions and archives their traces. Dify's searches are
  not retrieval spans until Dify has its own trace adapter, so the search metrics skip its runs until then.

### smolagents CodeAgent

The pro-code agent is smolagents' `CodeAgent` (`agents/code_agent.py`), which writes Python that calls the search tool
and revises its plan every 3 steps. It runs in `syllo-exp`'s own process, one new agent per sample, and is traced to
Phoenix with OpenInference's smolagents instrumentation.

- **Model:** the configuration's model through litellm, with the provider's defaults, as in the Dify agent.
  `ANTHROPIC_API_KEY` for Sonnet; DeepSeek on the Foundry resource of `AZURE_FOUNDRY_BASE_URL` and
  `AZURE_FOUNDRY_API_KEY`.
- **Search:** `SMOLAGENTS_SEARCH_URL`, by default `http://127.0.0.1:8101/mcp`, with the server on the configuration's
  collection. MCP results come as text, so the agent reads the same JSON as the other agents, and `json` is among its
  authorized imports to parse them.
- **Trace:** the caller's root span `smolagents.request` holds the `request_id`, the question and the answer. Every
  search whose output is the search tool's JSON is a retrieval span; a failed search stays a tool span. Planning calls
  are LLM spans directly under `CodeAgent.run`.
- **What the agent reads:** the retrieval spans hold all ten documents, but the agent reads only what its code prints
  (cut by smolagents at 50,000 characters per step) and the value of its last line (at 20,000).

### Open Deep Research

Open Deep Research (ODR) is LangChain's research agent: a supervisor splits the question into topics, researchers
search each topic in parallel and compress their findings, and a last call writes the report, which is the answer. Its
LangGraph graph runs unmodified in `syllo-exp`'s own process (`agents/odr.py`), at the commit `pyproject.toml` pins,
and OpenInference's LangChain instrumentation traces it to Phoenix.

- **Settings:** [`configs/odr.yaml`](configs/odr.yaml), with ODR's own names and defaults, except that web search is
  off (`search_api: none`), so the search tool is its only source, and that it never asks a clarifying question.
  `mcp_prompt` is the only prompt that tells the researchers about the tool: ODR's own prompts name its web search.
- **Model:** the configuration's model in every phase: the research brief, the supervisor, the researchers, the
  compression of their findings and the report. ODR sets only their output limits; every other parameter is the
  provider's default, as in the Dify agent.
  - ODR builds each phase's model from one module-level model, configured per phase. The caller replaces it with one
    that defaults to the configuration's model and carries its connection, which ODR's settings cannot name.
  - Sonnet runs on Anthropic's API with `ANTHROPIC_API_KEY`.
  - DeepSeek runs on the Foundry deployment at `AZURE_FOUNDRY_BASE_URL`, through its Azure OpenAI API (version
    `ODR_AZURE_API_VERSION`), with `AZURE_FOUNDRY_API_KEY`. ODR passes no key for it, which the other routes to the
    deployment would fill with `OPENAI_API_KEY` or `DEEPSEEK_API_KEY`.
  - The adapter refuses a trace with an LLM call to any other model.
- **Search:** `ODR_SEARCH_URL`, by default `http://127.0.0.1:8101`, with the server on the configuration's collection.
  It is the server's base URL: ODR appends `/mcp` itself.
- **Trace:** the caller's root span `odr.request` holds the `request_id`, the question and the report as text, without
  the thinking that Sonnet's reply also holds. Every search whose output is the search tool's JSON, which LangChain's
  MCP client returns as text blocks, is a retrieval span with its ten documents; a failed search stays a tool span.
- **Runs again:** ODR turns many failures into text or silence, so a failed run can look complete. The caller watches
  the run through LangChain's callbacks, and a question runs again from the start, with a new request id, up to
  `ODR_MAX_ATTEMPTS` attempts (3), `ODR_RETRY_DELAY_SECONDS` (60) apart, when:
  - the search tool was unreachable: before the run, for a researcher's model call or tool step, or for a search that
    got no answer from the server. ODR drops a server it cannot reach without an error, and its researchers go on
    without searching;
  - a provider's rate limit, outage or connection error outlasted the retries of ODR and of the client, in a researcher
    too: ODR then ends its research and writes the report from what the earlier rounds found;
  - ODR wrote an error in place of its report;
  - spans of the run did not reach Phoenix;
  - the run took longer than `ODR_TIMEOUT_SECONDS` (3600): Anthropic's client sets no request timeout.

  Any other failure fails the sample at once. The agent's own failures count as they happen:
  - a search that the server answered with an error, such as an empty query, is a failed call;
  - a researcher's other failures, such as a tool the model made up, end ODR's research as they would anyway;
  - a researcher whose findings ODR could not compress is only logged, and ODR reports what the others found.
- **Leftover researchers:** ODR does not stop the other researchers when one fails, so the caller cancels them before
  the call returns: the trace then holds only what the run used.
- **Environment:** ODR reads environment variables named after its settings (`SEARCH_API`, `RESEARCH_MODEL`, and so
  on) before its run config, so the caller refuses to start while any of them is set.
- **One collection per process:** the replaced model and the instrumentation are process-wide.

## Metrics

The main pass runs the built-in metrics. `ablation_metrics.py` adds recall over the ERB gold claims and the two
ablations. Each ablation subclasses the metric it is compared with and overrides one library hook, so both arms score
the same units with the same rubric, scoring, failure handling and result metadata; an ablation only adds metadata
fields. `deepeval_baseline.py` adds the DeepEval baseline the same way: each DeepEval metric replaces how the compared
metric judges, and keeps its units, skip rules and result metadata. The retrieval metrics also keep its scoring; the
answer metric's score is G-Eval's. [`METHODOLOGY.md`](METHODOLOGY.md) describes what each ablation and the baseline
change.

| Metric | Unit | Judge calls per unit | Compared with |
|---|---|---|---|
| `contextual_precision_document_judge` (built-in) | one search call | one per document | |
| `contextual_precision_document_judge_sc` | one search call | one | `contextual_precision_document_judge` |
| `contextual_recall_gold_claims` | one search call | one per gold claim | |
| `contextual_recall_gold_claims_sc` | one search call | one | `contextual_recall_gold_claims` |
| `answer_correctness_judge` (built-in) | one question | one | |
| `answer_correctness_judge_wt` | one question | one | `answer_correctness_judge` |
| `plan_correctness_judge` (built-in) | one τ²-bench trajectory | one | |
| `plan_correctness_judge_wt` | one τ²-bench trajectory | one | `plan_correctness_judge` |
| `deepeval_contextual_precision` | one search call | one | `contextual_precision_document_judge` |
| `deepeval_contextual_recall_gold_claims` | one search call | one | `contextual_recall_gold_claims` |
| `deepeval_answer_correctness` | one question | one | `answer_correctness_judge` |

- **Search units:** the retrieval metrics are built with `target_span_types=(SEARCH_SPAN_TYPE,)`, so each search call
  is one unit. Each agent's adapter must emit every search call as a `retrieval` span, with the user question as its
  request and the returned documents as one ranked `selected` document result.
- **Claims:** `contextual_recall_gold_claims` is the built-in stored-claims recall reading `expected_claims_gold`: it
  never decomposes the expected answer, so a unit makes one judge call per claim. Recall over `decomposed_claims` (RQ2)
  and `expected_claims_verified` (RQ3) is the same subclass with another key, once those ground truths are imported.
  DeepEval's recall over `expected_claims_verified` is likewise its gold-claims metric with that key.
- **Construction:** single-call metrics take the judge's `output_token_limit` from `configs/models.yaml`. Whole-trace
  metrics take a span loader; a run passes `stored_span_loader(db_manager)` with the step's pool.
- **Failures:** a failed unit records its cause in `metadata['failure']` (the library's classes, plus `trace_load` for
  a whole-trace unit whose trace could not be loaded) and the usage of every judge call.

## DeepEval baseline

```bash
poetry run syllo-exp deepeval --source-run <run id>                       # precision, recall and answer
poetry run syllo-exp deepeval --source-run <run id> --metrics precision   # WixQA: Table 2 compares relevance only
```

- **What it does:** it repeats a collection run on its stored traces with the selected DeepEval metrics. The new run's
  report counts DeepEval's judge calls and tokens as it counts Syllo-eval's.
- **Manifest:** the step is `deepeval:<source run>:<metrics>`, with the new run's id, the metric names, the DeepEval
  version and the failed units per metric and failure class.
- **Completion:** the step completes when the run does and every failed unit failed with an outcome of the judge,
  which counts as wrong ([`METHODOLOGY.md`](METHODOLOGY.md), Failures). A unit that failed in any other way, such as
  a provider error, is unscored, so the step is recorded as failed and the command exits with 1.
- **Requirements:** the database and `GOOGLE_API_KEY` for the shared judge (`configs/models.yaml`). No agent is called.
- **Cost:** Table 4 measures one sample at a time. That is the default unless `EVALUATION_MAX_CONCURRENT_SAMPLES` is
  set, and `--max-concurrent-samples 1` makes it explicit.
- **Offline:** `deepeval_env.py` turns off DeepEval's telemetry and its loading of `.env` files, and keeps its working
  files in `outputs/deepeval/`. DeepEval reads these switches when it is imported, so code that uses DeepEval imports
  `deepeval_baseline` first, which imports the switches before DeepEval. `pyproject.toml` disables DeepEval's pytest
  plugin, which pytest would import first.

## Collection

```bash
poetry run syllo-exp collect --configuration tau2/llm-agent/sonnet/trial-1 --max-concurrent-samples 4
```

- **What it does:** a fresh syllo-eval run of one configuration on every sample of its benchmark. The run calls the
  agent on each sample and stores its trace. It computes only the metrics that need no judge: Plan Efficiency on
  τ²-bench, and set precision, set recall and NDCG@10 of each search on ERB and WixQA. Judge metrics repeat the run
  later, on its stored traces.
- **Agents:** every stack has a caller: τ²-bench's agent, the Dify agent, the smolagents CodeAgent and Open Deep
  Research. A stack without one fails type checking in `collect.build_integration`.
- **Search server:** before any sample, the collection of an agent that searches reads the settings of the server that
  the agent calls, at `DIFY_SEARCH_URL`, `SMOLAGENTS_SEARCH_URL` or `ODR_SEARCH_URL`. Unless the server serves the
  configuration's collection and swap fraction, the step fails with the `search-server` command that (re)starts it.
- **Phoenix:** each benchmark has one project, named after its dataset: `erb-69916e3`, `wixqa-d662dc4` and
  `tau2-retail-v1.0.1`.
  - Every agent of a benchmark sends its spans to that project.
  - A sample's trace is the one whose root span carries the call's request ID, in the attribute `request_id`, or
    `dify_trace_id` for Dify. It is looked up among the root spans of the last 24 hours, because one agent call can
    outlast the library's 10-minute default. The attribute is set per stack, so `PHOENIX_REQUEST_ID_ATTRIBUTE` does
    not apply to collection.
- **Raw traces:** every trace the run fetches is also written to `outputs/traces/<run id>/<trace id>.json`. The format
  is the one syllo-eval imports: the `trace_id`, and Phoenix's records as `spans`.
- **Manifest:** the step is `collect:<configuration id>`, with the run id, the Phoenix project and the sample counts.
  It completes only when every sample completed; a run with failed samples is recorded as failed, and those samples
  must run again.
- **Requirements:** the configured database with the benchmark imported, Phoenix at `PHOENIX_BASE_URL` (by default the
  Compose service on `localhost:6006`), and the agent's own requirements.
- **Pilot:** a pilot runs a configuration on a few samples of its benchmark first:

  ```bash
  poetry run syllo-exp benchmarks pilot --benchmark tau2 --samples 0 24 40
  poetry run syllo-exp collect --configuration tau2/llm-agent/sonnet/trial-1 --pilot
  ```

  - `benchmarks pilot` takes the given samples of the checked conversion, with their ground truths and claims. It
    imports them as the dataset `<dataset>-pilot`, and writes them to `data/<benchmark>/pilot/`, gitignored like
    the rest of `data/`. The keys are `samples.jsonl`'s `sample_key`: τ²-bench task ids, for example.
  - `collect --pilot` runs on that dataset, as the step `collect:<configuration id>:pilot`, so a pilot never
    completes the real step. Its spans go to the benchmark's Phoenix project, like the full run's.
  - A pilot dataset holds one selection: importing other samples under its name is refused.
  - On τ²-bench, tasks 0, 24 and 40 cover a five-call plan rewarded on the database alone, a task without an expected
    plan, and a task whose assertions the grader checks.
- **Where:** run it on the host with `poetry run`, like the search servers, where the agents can reach the services.
  `--max-concurrent-samples` sets how many samples run at once; the default is `EVALUATION_MAX_CONCURRENT_SAMPLES`,
  or 1.

## τ²-bench agent

τ²-bench runs its own tool-calling agent against a simulated customer in its retail environment, and rewards each
conversation by checking the final database and the task's assertions. One sample is one simulation of its task.

```bash
scripts/setup-tau2.sh                                   # once: the τ²-bench environment, in .venv-tau2
poetry run syllo-exp benchmarks fetch --only tau2       # the tasks, and the environment's database, policy and guidelines
poetry run syllo-exp collect --configuration tau2/llm-agent/sonnet/trial-1 --max-concurrent-samples 4
```

- **Environment:** τ²-bench's dependencies conflict with syllo-eval's, so it runs in an environment of its own,
  `.venv-tau2`. `scripts/setup-tau2.sh` creates it from `tau2_runtime/requirements.txt`, which fixes every package:
  - τ²-bench at the commit the benchmark is pinned to;
  - litellm, which τ²-bench calls the models with;
  - OpenTelemetry and OpenInference's litellm instrumentation.

  `TAU2_PYTHON` points the caller at another interpreter.
- **One process per simulation:** the caller (`agents/tau2.py`) maps the sample's prompt to its task. It then runs
  `tau2_runtime/run_simulation.py` with that environment's interpreter.
  - The process runs τ²-bench's own orchestrator, agent, customer simulator and evaluator, unmodified but for how
    the grader's replies are read.
  - It reads the pinned files under `data/tau2/raw/data`, the caller's `TAU2_DATA_DIR`.
- **Models and credentials** ([`configs/tau2.yaml`](configs/tau2.yaml)):
  - The agent is the configuration's model, with the arguments of `agent_llm_args`.
  - The customer is `customer_simulator` in [`configs/models.yaml`](configs/models.yaml).
  - τ²-bench's assertion grader is the agent model.
  - litellm reads `ANTHROPIC_API_KEY` for Sonnet. The `azure/` and `azure_ai/` models, GPT 6 Luna and
    DeepSeek-V4.1-Flash, are deployments of the Foundry resource at `AZURE_FOUNDRY_BASE_URL`, called with
    `AZURE_FOUNDRY_API_KEY`. The `azure/` ones also use the API version `TAU2_AZURE_API_VERSION`.
- **Grader replies:** τ²-bench fails a simulation whose grader reply is not JSON, such as JSON in a Markdown fence.
  The process reads such a reply as the JSON inside it, with τ²-bench's own helper; a JSON reply is read unchanged
  ([`METHODOLOGY.md`](METHODOLOGY.md), τ²-bench runs).
- **Cost map:** litellm's model cost map tells it which parameters each model accepts, and it drops the others without
  an error. The process loads the map at the commit `cost_map_url` pins, and refuses to run on litellm's bundled copy,
  which predates the paper's models.
- **Traces:** each simulation is one trace in Phoenix:

  ```
  tau2.simulation          root: request_id, the user scenario in, the reward out
  ├─ tau2-llm-agent        the agent: its turns, each with its LLM call and the tool calls it requested
  ├─ user_turn             the customer simulator's turns, with their LLM calls
  └─ evaluate_simulation   τ²-bench's reward, with the grader's LLM calls
  ```

  Every LLM call is traced by OpenInference's instrumentation, which the process installs before it imports τ²-bench.
  Each tool call is a span with its arguments and result, marked as an error when the environment rejected it.
- **Adapter:** the agent's span is the agent root. Its request is the task's user scenario, which is the sample's
  prompt, and its executed steps are its tool calls, failed ones included, in order.
  - Each step's instruction renders the call's arguments as the dataset renders the expected plan.
  - The customer's and the grader's LLM spans become `customer_llm` and `grader_llm`. They, and litellm's nested
    retries (`llm_internal`), carry no usage, so a run's report counts only the agent's own calls.
  - Agent token usage and cost come from `PhoenixTraceAdapter`. A zero cost from litellm means the model could not
    be priced, so the adapter leaves cost and currency unset while retaining token counts.
- **Rewards:** the reward is the success label of RQ1. Each simulation adds one line to
  `outputs/tau2/<run id>/rewards.jsonl`, with the sample, request id, task, trial, seed, termination reason, reward
  and τ²-bench's reward details.
  - Next to it, `simulations/<request id>.json` is τ²-bench's own record of the simulation: messages, reward and
    costs.
  - `.spec.json` and `.log` hold what the process was asked to run and what it printed.
- **Failures:** some failures are transient:
  - a provider's rate limit, outage or connection error;
  - an unreachable cost map;
  - spans that Phoenix did not receive, which would leave an incomplete trace.

  These start a new simulation, with a new request id, up to `TAU2_MAX_ATTEMPTS` attempts (3), `TAU2_RETRY_DELAY_SECONDS`
  apart (60). Any other failure fails the sample at once. A cancelled call stops its process, which still exports the
  spans it has.

## Verification

```bash
poetry run python -m unittest discover -s tests
poetry run mypy . --check-untyped-defs --explicit-package-bases
```

`ImportBenchmarkTest` (in `tests/test_benchmarks_common.py`) runs against the database that the `.env` files select:
- it creates and deletes `test-benchmark-*` datasets;
- it is skipped when that database is unreachable, and fails when the database is not migrated.

`QdrantServerTest` (in `tests/test_vector_store.py`) runs against the Qdrant at `QDRANT_URL`, by default the Compose
service: it creates and deletes `test-search-*` collections, and it is skipped when that server is unreachable.

Check which database and Qdrant are configured before running the suite.

`OdrGraphTest` (in `tests/test_odr.py`) runs ODR's graph offline, with a scripted model, against the search server on
a free local port. It checks that every search reaches the trace with its ten documents, whole and in rank order, and
that the researchers a failed one leaves running stop with the call.

The τ²-bench runtime has its own tests, which run in its environment on the fetched files. They check the spans of a
simulation offline, with litellm's mock responses:

```bash
.venv-tau2/bin/python -m unittest discover -s tau2_runtime
```

The parent project's `poetry run ruff check .` and `poetry run ruff format .` also cover this folder.

## How the campaign runs

1. **Collect.** Fresh runs call each agent and archive its raw traces (`syllo-exp collect`). They compute only
   deterministic metrics.
2. **Judge.** Every judge evaluation repeats a collection run on its stored traces:
   - the main pass and two retests;
   - the gold-claim pass;
   - the SC and WT ablations;
   - the DeepEval baseline (`syllo-exp deepeval`).
3. **Construct.** RQ3 lists and padded traces are imported as new traces.
4. **Annotate.** Humans grade 180 answers and verify ERB facts.
5. **Analyze.** Tables 2–4 and every red placeholder are regenerated from the exports.

A pilot on a small slice gates the full campaign (`benchmarks pilot`, then `collect --pilot`). It fixes the judge's
temperature and thinking level and checks cost.
