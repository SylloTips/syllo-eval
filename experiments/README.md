# Paper experiments

This folder reproduces Section 5 of *Evaluating Enterprise Agents From the Traces They Leave*. It is a separate Poetry
project and is not part of the published package. It plugs into `syllo-eval` through its public extension points:
callers, trace clients, adapters and custom metrics.

It also uses the library's persistence layer, the `UnitOfWork` repositories in `syllo_eval.infrastructure`, for
ground truths under its own keys, which the dataset format cannot carry (`benchmarks/common.py`), and for the stored
traces that the whole-trace ablation reads (`ablation_metrics.py`), the failures a judge run recorded (`cli.py`) and
the judge runs the analysis reads (`analysis/`). It relies on `datasets.get_by_name`, `samples.list_by_dataset`,
`ground_truths.list_by_key`, `ground_truths.bulk_create`, `spans.list_by_trace`, `evaluation_runs.list_runs`,
`evaluation_run_samples.list_by_evaluation_run` and `metric_computations.list_by_evaluation_run`. The analysis also
reads the start time of search spans with SQL, which no repository returns.

The ablations and the DeepEval, AgentEvals and OpenEvals baselines also rely on a few library internals:
- `_judge_client` and `_rubric_addition`, which the built-in judge metrics set from their constructor arguments, and
  `_extract_expected_answer` of Answer Correctness;
- the `syllo_eval.evaluation.metric_support` helpers that `ablation_metrics.py` and the baselines' modules import to
  render prompts, traces, test cases and trajectories, and `normalize_relevant_ids`, with which the analysis reads
  gold documents;
- the built-in v1 prompt templates that `prompts/` edits (v2 for Plan and Answer Correctness): a new built-in
  prompt version needs new ablation prompts;
- `LangChainLlmJudgeClient._usage` and `GeminiLlmJudgeClient._retry_delay_seconds`, with which the baselines' judge
  (`baseline_judge.py`) reads usage and retries rate limits as the library's judge client does.

Collection wraps the library's `PhoenixClient` to archive the traces it fetches (`collect.py`). The τ²-bench,
smolagents, Open Deep Research, deep agent and Dify adapters read the flattened records that client returns,
`attributes.*` keys included, after `PhoenixTraceAdapter` has decoded them (`agents/tau2.py`, `agents/code_agent.py`,
`agents/langchain_runs.py`, `agents/dify.py`).

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
- **Agents:** a Dify ReAct agent, a smolagents CodeAgent, Open Deep Research and LangChain's deep agent, plus
  τ²-bench's own tool-calling agent.
- **Models and configurations:** two agent models; [`configs/configurations.yaml`](configs/configurations.yaml) lists
  all 26 configurations.
- **Comparisons:** Syllo-eval runs against three frameworks and two ablations, all with the same judge model:
  - DeepEval scores relevance, fact support and answers;
  - LangChain's AgentEvals scores τ²-bench plans, the only kind of unit it evaluates;
  - LangChain's OpenEvals scores relevance, fact support, answers and plans;
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
| `collect.py`, `run_outputs.py`, `agents/` | collection runs, the files they write, and each agent stack's caller and trace adapter; `agents/deep_research/` vendors the deep agent's example |
| `tau2_runtime/` | τ²-bench's run of one task with traces, run in its own environment (see τ²-bench agent) |
| `ablation_metrics.py`, `prompts/` | the SC and WT ablations, recall over the gold claims, and the ablation prompts |
| `deepeval_baseline.py`, `deepeval_env.py` | the DeepEval baseline, and the switches DeepEval reads when it is imported |
| `agentevals_baseline.py`, `openevals_baseline.py` | the AgentEvals and OpenEvals baselines, and the shared judge as OpenEvals calls it |
| `baseline_judge.py` | the shared judge model as the baselines call it, the calls each unit made, and the document text they judge |
| `dify/` | the Dify agent's DSL, configuration guide, and the Compose override that runs Dify beside Phoenix |
| `indexing/` | the search indexes of the ERB and WixQA knowledge bases: embedding model, Qdrant vector store, pipeline |
| `search_tool/` | the agents' search tool: an MCP server over the search index of one knowledge base |
| `scripts/` | `index-benchmark.sh`, which launches the indexing of one knowledge base; `setup-tau2.sh`, which creates the τ²-bench environment; `dify.sh`, which runs Dify; `dify-patch-plugins.sh`, which adapts Dify's plugins to GPT models and token counting |
| `docker-compose.yml`, `Dockerfile` | the campaign environment (see Setup) |
| `analysis/` | the results tables, computed from the stored judge runs (see Analysis) |
| `tests/` | tests on synthetic data; `ImportBenchmarkTest` and `QdrantServerTest` also write to the configured database and Qdrant (see Verification) |
| `outputs/` | gitignored: manifest, raw trace exports, τ²-bench simulations and rewards, computations, annotation packets, stored CP verdicts |
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
poetry run syllo-exp benchmarks claims    # WixQA: the expected answers' claims, extracted once with Orbitals
poetry run syllo-exp benchmarks import    # the datasets and their claim ground truths, into the configured database
```

- **Pins:** [`configs/benchmarks.yaml`](configs/benchmarks.yaml) fixes every file by commit, size and SHA-256. For
  τ²-bench, it also fixes the files that its agent runs on: the retail database, the agent's policy and the customer
  simulator's guidelines.
- **Scope:** each step takes `--only erb wixqa tau2`, records its outcome in the manifest, and is safe to re-run.
  Fetching resumes interrupted downloads.
- **Orbitals claims:** `claims` sends each WixQA question and expected answer to Orbitals' claim extractor
  (`ORBITALS_API_KEY`), as the library's claim-extractor recall does, and drops the claims it types as unverifiable.
  It calls the batch API, `batch_size` answers at a time, with the model and the description of the WixQA assistant
  that `claim_extractor` in [`configs/models.yaml`](configs/models.yaml) fixes; a benchmark without a description
  there has no Orbitals claims. It writes `data/wixqa/orbitals_claims.json` after every batch, with the model and each
  claim's type, so it resumes where it stopped and `convert` never overwrites it. `import` then stores the claims as
  `expected_claims_orbitals`; run it after `claims`, since an import refuses to change claims it already stored.
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
poetry run syllo-exp search-server --collection erb-69916e3 --port 8101 --swap-fraction 0.25  # erb/react/gpt/f0.25
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

  A successful search is a tool output that parses as such an object, and a failed one reads
  `The search failed: <reason>`: the adapters read the text, since the agents' clients do not all keep MCP's error
  flag. Any other output is unreadable, and makes the question's retrieved evidence unavailable. ODR's and the deep
  agent's adapters also read the error that LangChain recorded (`search_span_retrieval`): a search that got any error
  from the server, such as arguments the tool does not take, or that was cancelled under way, is a failed call.
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
    `SSRF_PROXY_ALLOW_PRIVATE_DOMAINS=host.docker.internal` in Dify's `docker/.env` and recreate `ssrf_proxy`.
    [`dify/compose.override.yaml`](dify/compose.override.yaml) maps `host.docker.internal` to the host for
    `ssrf_proxy`. On Linux, the host is the Docker bridge, `172.17.0.1` by default: serve there with `--host`, and
    point every agent's search URL at it.
  - Dify's classic Agent app switches to function calling whenever the model supports it. A ReAct agent needs a
    Chatflow or Workflow Agent node with the ReAct strategy.
- **smolagents:** the caller connects at `SMOLAGENTS_SEARCH_URL` (see smolagents CodeAgent).
- **Open Deep Research:** the caller connects at `ODR_SEARCH_URL` (see Open Deep Research).
- **Deep agent:** the caller connects at `DEEPAGENTS_SEARCH_URL` (see Deep agent).

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
- **Check:** a pilot (see Collection) runs the app on a few questions and archives their traces.
- **Trace:** Dify exports no spans below its nodes, so the adapter turns the agent node's ReAct log into spans: rounds,
  model calls and tool calls. A search whose output is the search tool's JSON is a retrieval span with its ten
  documents; a failed search stays a tool span ([`dify/README.md`](dify/README.md#what-the-trace-holds)).

### smolagents CodeAgent

The pro-code agent is smolagents' `CodeAgent` (`agents/code_agent.py`), which writes Python that calls the search tool
and revises its plan every 3 steps. It runs in `syllo-exp`'s own process, one new agent per sample, and is traced to
Phoenix with OpenInference's smolagents instrumentation.

- **Model:** the configuration's model through litellm, with the provider's defaults, as in the Dify agent. Both
  models are deployments of the Foundry resource at `AZURE_FOUNDRY_BASE_URL`, called with `AZURE_FOUNDRY_API_KEY`:
  GPT 6.1 Sol through its Azure OpenAI API (version `SMOLAGENTS_AZURE_API_VERSION`), DeepSeek through its model
  inference API.
  - GPT reasoning models refuse the `stop` parameter. smolagents then cuts each reply at its stop sequences itself,
    but recognizes such models by name, up to the GPT-5 family; the caller adds GPT 6 and later.
  - GPT 6.1 Sol requests time out after `SMOLAGENTS_AZURE_TIMEOUT_SECONDS` (120) and are retried up to
    `SMOLAGENTS_AZURE_MAX_RETRIES` (5) times. About 1 call in 10 reasons for tens of thousands of hidden tokens or
    never returns, while 99.6% of its WixQA calls took under 120 s; litellm's default 600 s timeout with 2 retries
    made such a step cost up to 30 minutes and then fail the sample. The WixQA GPT runs predate this limit.
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
  - GPT 6.1 Sol and DeepSeek run on their Foundry deployments at `AZURE_FOUNDRY_BASE_URL`, through its Azure OpenAI
    API (version `ODR_AZURE_API_VERSION`), with `AZURE_FOUNDRY_API_KEY`. ODR passes no key for them, which the other
    routes to the deployments would fill with `OPENAI_API_KEY` or `DEEPSEEK_API_KEY`.
  - The adapter refuses a trace with an LLM call to any other model. A dated version of the model, as Azure reports
    GPT models (`gpt-6.1-sol-2026-09-29`), is the model.
- **Search:** `ODR_SEARCH_URL`, by default `http://127.0.0.1:8101`, with the server on the configuration's collection.
  It is the server's base URL: ODR appends `/mcp` itself.
  - ODR looks the tool up before every researcher step and goes on without it when the lookup fails. The caller
    replaces that lookup with one that runs once per server and process, and fails instead of returning no tools.
  - Each search still opens its own connection, with a 2-second connect timeout and two retries. A connection that
    fails never reaches the server, so a retry repeats no search or embedding; the read timeout is unchanged.
- **Trace:** the caller's root span `odr.request` holds the `request_id`, the question and the report as text. Every
  search whose output is the search tool's JSON, which LangChain's MCP client returns as text blocks, is a retrieval
  span with its ten documents. A search that returned no documents stays a tool span: the server rejected it, or it
  was cancelled under way.
- **Runs again:** ODR turns many failures into text or silence, so a failed run can look complete. The caller watches
  the run through LangChain's callbacks, and a question runs again from the start, with a new request id, up to
  `ODR_MAX_ATTEMPTS` attempts (3), `ODR_RETRY_DELAY_SECONDS` (60) apart, when:
  - the search tool was unreachable: before the run, for a researcher's model call or tool step, or for a search that
    got no answer from the server after the connection retries;
  - a provider's rate limit, outage or connection error outlasted the retries of ODR and of the client, in a researcher
    too: ODR then ends its research and writes the report from what the earlier rounds found;
  - ODR wrote an error in place of its report;
  - spans of the run did not reach Phoenix;
  - the run took longer than `ODR_TIMEOUT_SECONDS` (1200): a provider can hold a request open for good.

  Any other failure fails the sample at once. The agent's own failures count as they happen:
  - a search that the server answered with an error, such as an empty query, is a failed call;
  - a researcher's other failures, such as a tool the model made up, end ODR's research as they would anyway;
  - a researcher whose findings ODR could not compress is only logged, and ODR reports what the others found.
- **Leftover researchers:** ODR does not stop the other researchers when one fails, so the caller cancels them before
  the call returns. It lets the callbacks that end their runs finish, so every span of the trace keeps its parent. A
  search they had under way stays in the trace as a failed call; a model call under way leaves no span.
- **Environment:** ODR reads environment variables named after its settings (`SEARCH_API`, `RESEARCH_MODEL`, and so
  on) before its run config, so the caller refuses to start while any of them is set.
- **One collection per process:** the replaced model and the instrumentation are process-wide.

### Deep agent

LangChain's deep agent is the `deep_research` example of deepagents: an orchestrator plans with `write_todos`,
delegates the research to sub-agents through deepagents' `task` tool, and writes the report, which is the answer. It is
built with `create_deep_agent` in `syllo-exp`'s own process (`agents/deep_agent.py`), with deepagents 0.7.34, and
OpenInference's LangChain instrumentation traces it to Phoenix.

- **Example:** its prompts and `think_tool`, vendored at the commit of the 0.7.34 release under the example's MIT
  license (`agents/deep_research/`), and wired as its `agent.py` does: the example's research sub-agent and limits (3
  sub-agents at once, 3 delegation rounds), and deepagents' defaults otherwise, with its general-purpose sub-agent,
  except the thresholds of the settings below.
  - `prompts.py` is byte for byte the example's, which a test checks. The caller edits the prompts with exact
    replacements, which fail if they no longer apply: the search tool for the web search, document titles and ids for
    URLs, the report as the last message instead of `/final_report.md`, and no clarifying questions, as ODR asks none.
  - The example's own lock pins deepagents 0.6.12. deepagents 0.7 dropped its base prompt, its tools' usage guidance
    and `write_todos` from the defaults; the orchestrator gets `write_todos` back, as the example's workflow starts
    with it.
- **Settings:** [`configs/deepagents.yaml`](configs/deepagents.yaml):
  - the 272,000-token window of both models: deepagents summarizes the older history once a call's input reaches 85%
    of it (231,200 tokens);
  - the 100,000 tokens above which a tool result moves to the agent's files.
- **Model:** the configuration's model in every call, the sub-agents' and the summaries' included, on its Foundry
  deployment at `AZURE_FOUNDRY_BASE_URL`, through its Azure OpenAI API (version `DEEPAGENTS_AZURE_API_VERSION`), with
  `AZURE_FOUNDRY_API_KEY`. Every model parameter is the provider's default, as in the Dify agent: the agents differ
  only in their framework. LangChain sends GPT 6.1 Sol's calls with tools to the Responses API, as for ODR. Each call
  waits up to `DEEPAGENTS_REQUEST_TIMEOUT_SECONDS` (600), and the client retries a rate limit, an outage or a
  connection error `DEEPAGENTS_MAX_RETRIES` times (5). The adapter refuses a trace with an LLM call to any other
  model.
- **Search:** `DEEPAGENTS_SEARCH_URL`, by default `http://127.0.0.1:8101/mcp`, with the server on the configuration's
  collection. The caller lists the server's tools when it builds the agent, at its first call; each search opens its
  own connection, as ODR's do. The tool hands the server's errors, such as an empty query, back to the agent, which
  would otherwise end the run.
- **Trace:** the caller's root span `deepagents.request` holds the `request_id`, the question and the answer as text.
  The sub-agents' spans are under the `task` tool's. Every search, the orchestrator's or a sub-agent's, is a retrieval
  span with its ten documents; a search that the server rejected stays a tool span.
  - A tool result moved to the agent's files is moved after the search returns, so its span holds the ten documents.
    The model gets a preview of the result and its path; reading it there with the file tools leaves tool spans,
    which are no new searches. The caller logs every result moved, which no search of ERB or WixQA is: their ten
    longest documents would make about 372,000 and 281,000 characters of JSON.
- **Runs again:** a question runs again from the start, with a new request id, up to `DEEPAGENTS_MAX_ATTEMPTS`
  attempts (3), `DEEPAGENTS_RETRY_DELAY_SECONDS` (60) apart, when:
  - the search tool was unreachable when the agent was built, or a search got no answer from the server;
  - a provider's rate limit, outage or connection error outlasted the client's retries;
  - a model call was cut off, at its output limit or by a content filter;
  - the run ended without an answer;
  - spans of the run did not reach Phoenix;
  - the run took longer than `DEEPAGENTS_TIMEOUT_SECONDS` (3600).

  Any other failure fails the sample at once, a refusal of the orchestrator's model included; a sub-agent's refusal
  reaches the orchestrator as a notice, as deepagents handles it. If the agent writes `/final_report.md` anyway, the
  caller logs it: the last message is still the answer.
- **One collection per process:** the instrumentation is process-wide.

## Metrics

The main pass runs the built-in metrics. `ablation_metrics.py` adds recall over the ERB gold claims and the two
ablations. Each ablation subclasses the metric it is compared with and overrides one library hook, so both arms score
the same units with the same rubric, scoring, failure handling and result metadata; an ablation only adds metadata
fields. `deepeval_baseline.py` adds the DeepEval baseline the same way: each DeepEval metric replaces how the compared
metric judges, and keeps its units, skip rules and result metadata. The retrieval metrics also keep its scoring; the
answer metric's score is G-Eval's. `agentevals_baseline.py` adds the AgentEvals baseline for plans the same way: its
metric replaces how Plan Correctness judges, and its score is AgentEvals' verdict, 1.0 or 0.0. `openevals_baseline.py`
adds the OpenEvals baseline, one metric for each judge metric of the main arm. Its retrieval metrics judge each document
or claim in its own call and keep the compared metric's scoring; its answer and plan metrics score OpenEvals' verdict.
[`METHODOLOGY.md`](METHODOLOGY.md) describes what each ablation and baseline changes.

| Metric | Unit | Judge calls per unit | Compared with |
|---|---|---|---|
| `contextual_precision_document_judge` (built-in) | one search call | one per document | |
| `contextual_precision_document_judge_sc` | one search call | one | `contextual_precision_document_judge` |
| `contextual_recall_gold_claims_siblings_question` | one question | one per gold claim | |
| `contextual_recall_gold_claims_sc_question` | one question | one | `contextual_recall_gold_claims_siblings_question` |
| `answer_correctness_judge` (built-in) | one question | one | |
| `answer_correctness_judge_wt` | one question | one | `answer_correctness_judge` |
| `plan_correctness_judge` (built-in) | one τ²-bench trajectory | one | |
| `plan_correctness_judge_wt` | one τ²-bench trajectory | one | `plan_correctness_judge` |
| `deepeval_contextual_precision` | one search call | one | `contextual_precision_document_judge` |
| `deepeval_contextual_recall_gold_claims` | one question | one | `contextual_recall_gold_claims_siblings_question` |
| `deepeval_answer_correctness` | one question | one | `answer_correctness_judge` |
| `agentevals_trajectory_accuracy` | one τ²-bench trajectory | one | `plan_correctness_judge` |
| `openevals_retrieval_relevance` | one search call | one per document | `contextual_precision_document_judge` |
| `openevals_groundedness_gold_claims` | one question | one per gold claim | `contextual_recall_gold_claims` |
| `openevals_groundedness_orbitals_claims` | one question | one per claim | `contextual_recall_orbitals_claims` |
| `openevals_correctness` | one question | one | `answer_correctness_judge` |
| `openevals_plan_adherence` | one τ²-bench trajectory | one | `plan_correctness_judge` |

- **Retrieval units:** each agent's adapter reads every search output with `search_retrieval`
  (`search_tool/search.py`):
  - the tool's results become a `retrieval` span, with the user question as its request and the documents as one
    ranked `selected` document result;
  - the tool's error message (`The search failed: …`) leaves a tool span, as, for ODR and the deep agent, any error
    of the server and a search cancelled under way do;
  - any other output becomes a `retrieval` span whose result is `unavailable`.

  `with_retrieved_evidence` then gives the agent root the retrieved evidence: one unranked document result at the
  `retrieved` stage, holding each document of the question's searches once, in order of first appearance. It is
  empty when no search succeeded, and `unavailable` when a search output could not be read. Contextual precision and
  NDCG@10 are built with `target_span_types=(SEARCH_SPAN_TYPE,)`, so each search call is one unit.
  `contextual_recall_gold_claims_siblings_question`, `set_precision_retrieved_document` and
  `set_recall_retrieved_document` read the root's `retrieved` stage, so each question is one unit; the built-in
  metrics read `selected` and skip the root.
- **Claims:** `contextual_recall_gold_claims_siblings_question` is the built-in stored-claims recall reading
  `expected_claims_gold`: it never decomposes the expected answer, so a unit makes one judge call per claim; it serves
  RQ1 attribution and RQ2. On WixQA, `contextual_recall_orbitals_claims_siblings_question` is that subclass reading
  `expected_claims_orbitals`. Both use the library's recall prompt v2, which gives each call the question and every
  claim of the unit; their names say so, because the main runs made before v2 scored `contextual_recall_gold_claims`
  and `contextual_recall_orbitals_claims` with v1, which the analysis no longer reads. The first v2 pass ran as the
  arm `crgs`, before v2 was the library's default; the analysis reads those runs as the `main` arm.
- **Construction:** single-call metrics take the judge's `output_token_limit` from `configs/models.yaml`. Whole-trace
  metrics take a span loader; a run passes `stored_span_loader(db_manager)` with the step's pool.
- **Failures:** a failed unit records its cause in `metadata['failure']` (the library's classes, plus `trace_load` for
  a whole-trace unit whose trace could not be loaded) and the usage of every judge call.

## Judge

```bash
poetry run syllo-exp judge --source-run <collection run id> --arm main --repetition 1   # 2 and 3 are the retests
poetry run syllo-exp judge --source-run <collection run id> --arm main --repetition 2 --recall-only
poetry run syllo-exp judge --source-run <collection run id> --arm sc --pilot
poetry run syllo-exp judge --source-run <collection run id> --arm wt --retry <judge run id>
```

- **What it does:** it repeats a finished collection run (`COMPLETED` or `PARTIALLY_COMPLETED`, of a benchmark or of
  its pilot dataset) on its stored traces with the judge metrics of one arm. The source run's benchmark selects them:

  | Arm | ERB | WixQA | τ²-bench |
  |---|---|---|---|
  | `main` | AC, CP, CR over `expected_claims_gold` | AC, CP, CR over `expected_claims_orbitals` | PC |
  | `sc` | CP and CR SC | CP SC | refused |
  | `wt` | AC WT, on the annotated questions | refused | PC WT |

- **Samples:** every traced sample, except that ERB `wt` scores the `annotated_samples` of `configs/judge.yaml` and
  `--pilot` scores its `pilot_samples`, both by sample key within the source run's dataset. A sample without a trace
  is a collection failure: it is counted in the manifest, never judged, and left out of those selections.
- **Run metadata:** each judge run carries the step, configuration, arm, repetition and judge settings as its
  `metadata`, so the database alone tells the runs of a source run apart.
- **Recall only:** `--recall-only` scores only the arm's contextual recall, as its own step with its own retries. The
  analysis joins it to the arm's other runs of the same repetition, so a retest of recall reuses the stored scores of
  the other metrics.
- **Manifest:** the step is `judge:<source run>:<arm>:<repetition>`, with `:recall` appended for `--recall-only` and
  `:pilot` or `:retry-<n>` for those runs, and records the run, the configuration, the metrics, the judge settings,
  the collection failures, the failed units per metric and class, and the unresolved samples.
- **Completion:** a sample is unresolved until it completes and each of its metrics has an outcome: an interrupted
  sample can hold only part of its computations, and a computation failed without an outcome of the judge
  ([`METHODOLOGY.md`](METHODOLOGY.md), Failures) is none. `--retry <judge run>` takes the latest attempt of the same step,
  never a pilot, and scores its unresolved samples again in a new run, which replaces their whole earlier attempt.
  Every other sample's latest attempt is then resolved, so the step completes when a retry leaves nothing unresolved,
  listing every attempt in order; for each sample, the latest run that scored it holds its results. Every attempt's cost stays recorded.
- **Rejected outputs:** the step logs to stderr with the run and sample of each line, so every rejected judge output
  (see the root README) can be traced back to its sample, for example with
  `docker logs <container> 2>&1 | grep 'Judge output rejected'`.
- **Requirements:** the database, the converted benchmarks for `--pilot` and ERB `wt`, and `GOOGLE_API_KEY` for the
  shared judge (`configs/models.yaml`). No agent is called.

## Baselines at a glance

Each framework's metrics score the units of the Syllo-eval metric they are compared with, with the same judge model
(`configs/models.yaml`), and record every judge call. ✅ runs with `syllo-exp <framework>`; "not run" means a
counterpart exists but no comparison of the paper uses it.

| Syllo-eval metric | What it measures | DeepEval | AgentEvals | OpenEvals |
|---|---|---|---|---|
| NDCG@10 (`ndcg_at_10_document`, `_snippet`) | rank quality of each search against the gold documents (deterministic) | none | none | none |
| Set precision (`set_precision_document`, `_snippet`, `_retrieved_document`) | share of retrieved documents that are gold (deterministic) | none | none | none |
| Set recall (`set_recall_document`, `_snippet`, `_retrieved_document`) | share of gold documents retrieved (deterministic) | none | none | none |
| Contextual precision (`contextual_precision_document_judge`, `_snippet`) | judge: are the relevant documents ranked first? | ✅ `ContextualPrecisionMetric`, a ranking in one call | none | ✅ retrieval relevance, one call per document |
| Contextual recall over stored claims (`…_stored_claims`, ERB gold claims, WixQA Orbitals claims) | judge: is each claim supported by the retrieved documents? | ✅ `ContextualRecallMetric`, all claims in one call, ERB gold claims only | none | ✅ groundedness, one call per claim, ERB and WixQA |
| Contextual recall over claims the judge splits off (`contextual_recall_document_judge`, `_snippet`) | the same, after the judge splits the expected answer into claims | not run: `ContextualRecallMetric` splits the expected answer into sentences itself | none | none: OpenEvals cannot split an answer into claims |
| Contextual recall over extracted claims (`…_claim_extractor`) | the same, with Orbitals extracting the claims | not run: as for stored claims | none | not run: as for stored claims |
| Answer correctness (`answer_correctness_judge`) | judge: does the answer match the expected one? | ✅ G-Eval Correctness, 0–10 divided by 10 | none | ✅ correctness, 1.0 or 0.0 |
| Plan correctness (`plan_correctness_judge`) | judge: does the plan reach the reference's outcome? | none: its plan metrics grade no reference plan | ✅ trajectory accuracy, 1.0 or 0.0 | ✅ plan adherence, 1.0 or 0.0 |
| Plan efficiency (`plan_efficiency`) | expected steps divided by executed steps (deterministic) | none: `StepEfficiencyMetric` judges DeepEval's own traces, without a reference | none: its judge folds efficiency into one verdict, and its matches check which calls were made, not how many | none |
| LLM calls (`llm_calls`) | number of LLM spans in the trace | none | none | none |

The deterministic metrics score typed trace observations against gold labels or the trace's structure: gold document
ids, ranks, step counts and LLM spans. None of the frameworks reads those from a test case: DeepEval's trace metrics
need DeepEval's own traces, AgentEvals reads messages or LangGraph thread histories, and OpenEvals reads text. In the
paper they are the references the judges are ranked against (RQ2). DeepEval's `ToolCorrectnessMetric` and AgentEvals'
trajectory match, which OpenEvals copies, compare executed tool calls with expected ones; no Syllo-eval metric does,
and the plan analysis takes that check from τ²-bench itself.

## DeepEval baseline

```bash
poetry run syllo-exp deepeval --source-run <run id>                       # precision, recall and answer
poetry run syllo-exp deepeval --source-run <run id> --metrics precision   # WixQA: Table 2 compares relevance only
```

- **What it does:** it repeats a collection run on its stored traces with the selected DeepEval metrics. The new run's
  report counts DeepEval's judge calls and tokens as it counts Syllo-eval's.
- **Run metadata:** the step, configuration, arm `deepeval` and repetition 1, as a judge run carries them.
- **Manifest:** the step is `deepeval:<source run>:<metrics>`, with the new run's id, the metric names, the DeepEval
  version and the failed units per metric and failure class.
- **Completion:** the step completes when the run does and every failed unit failed with an outcome of the judge,
  which is counted ([`METHODOLOGY.md`](METHODOLOGY.md), Failures). A unit that failed in any other way, such as
  a provider error, is unscored, so the step is recorded as failed and the command exits with 1.
- **Requirements:** the database and `GOOGLE_API_KEY` for the shared judge (`configs/models.yaml`). No agent is called.
- **Cost:** Table 4 measures one sample at a time. That is the default unless `EVALUATION_MAX_CONCURRENT_SAMPLES` is
  set, and `--max-concurrent-samples 1` makes it explicit.
- **Offline:** `deepeval_env.py` turns off DeepEval's telemetry and its loading of `.env` files, and keeps its working
  files in `outputs/deepeval/`. DeepEval reads these switches when it is imported, so code that uses DeepEval imports
  `deepeval_baseline` first, which imports the switches before DeepEval. `pyproject.toml` disables DeepEval's pytest
  plugin, which pytest would import first.

## AgentEvals baseline

```bash
poetry run syllo-exp agentevals --source-run <τ²-bench collection run id>
poetry run syllo-exp agentevals --source-run <τ²-bench collection run id> --retry <run id>   # unresolved samples
```

- **What it does:** it repeats a finished τ²-bench collection run, as `judge` takes it, on its stored traces with
  AgentEvals' trajectory accuracy, which is compared with Plan Correctness. AgentEvals has no retrieval or answer
  metric, so the command refuses a run of another benchmark. The new run's report counts AgentEvals' judge calls and
  tokens as it counts Syllo-eval's.
- **Run metadata:** the step, configuration, arm `agentevals` and repetition 1, as a judge run carries them.
- **Manifest:** the step is `agentevals:<source run>`, with the new run's id, the configuration, the metric name, the
  AgentEvals and OpenEvals versions and the failed units per failure class.
- **Completion:** as for a judge step: the step completes once no sample is unresolved, a unit failed with an outcome of
  the judge counting as resolved, and the manifest records the unresolved samples. `--retry <run>` takes the latest
  run of the same step and scores its unresolved samples again, in a new run recorded as `<step>:retry-<n>`; a retry
  that leaves none completes the step, listing every attempt in order.
- **Requirements and cost:** as for the DeepEval baseline.
- **Pins:** `pyproject.toml` pins AgentEvals 0.0.9, its latest release, and OpenEvals 0.2.2, whose judge AgentEvals
  calls. Later OpenEvals releases need langchain-core 1.8, above the version the agent stacks are locked to; on the
  judge's path they differ from 0.2.2 only in the description of the boolean score.
- **Offline:** OpenEvals traces its evaluators to LangSmith only when LangSmith tracing is turned on in the environment,
  which the experiments never do. AgentEvals sends no telemetry.

## OpenEvals baseline

```bash
poetry run syllo-exp openevals --source-run <collection run id>                       # every metric of its benchmark
poetry run syllo-exp openevals --source-run <collection run id> --metrics relevance   # some of them
poetry run syllo-exp openevals --source-run <collection run id> --retry <run id>      # unresolved samples
```

- **What it does:** it repeats a finished collection run, as `judge` takes it, on its stored traces with the OpenEvals
  metrics of its benchmark, one for each judge metric of the main arm:

  | Benchmark | `relevance` | `recall` | `answer` | `plan` |
  |---|---|---|---|---|
  | ERB | retrieval relevance (vs CP) | groundedness of each gold claim (vs CR over `expected_claims_gold`) | correctness (vs AC) | |
  | WixQA | retrieval relevance (vs CP) | groundedness of each Orbitals claim (vs CR over `expected_claims_orbitals`) | correctness (vs AC) | |
  | τ²-bench | | | | plan adherence (vs PC) |

  `--metrics` selects some of them; a metric the run's benchmark lacks is refused.
- **Run metadata:** the step, configuration, arm `openevals` and repetition 1, as a judge run carries them.
- **Manifest:** the step is `openevals:<source run>`, followed by `:<metrics>` when `--metrics` selects them, with the
  new run's id, the configuration, the metric names, the OpenEvals version and the failed units per failure class.
- **Completion, retries, requirements, cost, pins and offline:** as for the AgentEvals baseline. A run of an ERB or
  WixQA configuration makes one judge call per document of every search and per claim, so `--retry` re-scores only
  the samples a provider error left unresolved.

## Collection

```bash
poetry run syllo-exp collect --configuration tau2/llm-agent/gpt/trial-1 --max-concurrent-samples 4
```

- **What it does:** a fresh syllo-eval run of one configuration on every sample of its benchmark. The run calls the
  agent on each sample and stores its trace. It computes only the metrics that need no judge: Plan Efficiency on
  τ²-bench, and on ERB and WixQA set precision and set recall of each question's retrieved evidence and NDCG@10 of
  each search. Judge metrics repeat the run later, on its stored traces.
- **Agents:** every stack has a caller: τ²-bench's agent, the Dify agent, the smolagents CodeAgent, Open Deep Research
  and the deep agent. A stack without one fails type checking in `collect.build_integration`.
- **Search server:** before any sample, the collection of an agent that searches reads the settings of the server that
  the agent calls, at `DIFY_SEARCH_URL`, `SMOLAGENTS_SEARCH_URL`, `ODR_SEARCH_URL` or `DEEPAGENTS_SEARCH_URL`. Unless
  the server serves the configuration's collection and swap fraction, the step fails with the `search-server` command
  that (re)starts it.
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
  poetry run syllo-exp collect --configuration tau2/llm-agent/gpt/trial-1 --pilot
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
- **Hangs:** `kill -USR1 <pid>` makes a running `syllo-exp` print the stack of every thread to its stderr.

## τ²-bench agent

τ²-bench runs its own tool-calling agent against a simulated customer in its retail environment, and rewards each
conversation by checking the final database and the task's assertions. One sample is one simulation of its task.

```bash
scripts/setup-tau2.sh                                   # once: the τ²-bench environment, in .venv-tau2
poetry run syllo-exp benchmarks fetch --only tau2       # the tasks, and the environment's database, policy and guidelines
poetry run syllo-exp collect --configuration tau2/llm-agent/gpt/trial-1 --max-concurrent-samples 4
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
  - The models of `responses_api_models`, GPT 6.1 Sol, are called through Azure's Responses API, on litellm's
    `azure/responses/` route: on Chat Completions, GPT 6.1 Sol takes function tools only with reasoning turned off.
  - The `azure/` and `azure_ai/` models, GPT 6.1 Sol, GPT 6 Luna and DeepSeek-V4.1-Flash, are deployments of the
    Foundry resource at `AZURE_FOUNDRY_BASE_URL`, called with `AZURE_FOUNDRY_API_KEY`. The `azure/` ones also use the
    API version `TAU2_AZURE_API_VERSION`.
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
  - Each step's input is the call's arguments, and its output is the call's result.
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

## Analysis

`syllo-exp analyze <table>` computes a results table from the completed judge runs, prints it and writes its rows to
`outputs/analysis/<table>.json`. It reads the finished runs by the `configuration`, `arm` and `repetition` that the
`judge` and `deepeval` steps record in each run's metadata, leaving out pilots and the Open Deep Research runs, which
the paper does not report. The runs of one configuration, arm and
repetition (a first attempt, its retries, or several DeepEval runs) form a chain: each sample counts in the latest run
that completed it with the analyzed metric, as for a judge step's retries. A chain over two source runs stops the
analysis. It only reads: run it with `PGOPTIONS='-c default_transaction_read_only=on'`, so the database refuses any
write.

- `relevance` (RQ1 Relevance) compares the contextual precision decisions of the selected arms (`--arms`: `main`,
  `sc`, `deepeval`; default `main sc`) with the gold documents, as METHODOLOGY's *RQ1 references* define them:
  - one decision per configuration, question and document, from the document's first search in time order;
  - a unit failed with an outcome of the judge counts each document of its ranking, read from the NDCG@10 of the
    source run, as a wrong decision; any other failure stops the analysis;
  - questions without gold documents, τ²-bench and degraded runs are left out.

  It reports, per run, the questions, the gold and other documents with the share of each that the judge accepts, and
  Cohen's κ. A `<benchmark>/reported` row per arm and repetition pools the six configurations.
- `attribution` (RQ1 Fact Support) compares the contextual recall decisions on ERB's gold facts (Basic and Semantic
  questions; `--arms`: `main`, `sc`, `deepeval`) with whether each fact's gold document is in the trace's retrieved
  evidence, read from the source run's set recall: one decision per configuration, question and fact. A fact of a
  trace that retrieved nothing is unsupported and rejected without a judge call, as contextual recall scores it 0; the
  `unjudged` count reports those decisions. A unit failed with an outcome of the judge counts each fact of its ground
  truth as a wrong decision. Its rows have the shape of `relevance`'s, with supported and unsupported facts in place of
  gold and other documents.
- `plans` (RQ1 Plans; `--arms`: `main`, `wt`) compares Plan Correctness with τ²-bench's reward, read with its action
  checks from `outputs/tau2/<collection run>/rewards.jsonl` (each sample's latest attempt). Per run, per agent model
  and pooled, it gives the AUROC of Plan Correctness and of a deterministic baseline, 1 − the normalized edit distance
  between the executed tool calls (name and arguments) and the expected plan, on the trajectories Plan Correctness
  scored. It also counts the rewarded trajectories that fail τ²-bench's action check, which deviate from the reference,
  with both metrics' mean on them.
- `answers` (RQ1 Answers; `--arms`: `main`, `wt`, `deepeval`) compares Answer Correctness with the human grades of
  the 180 annotated ERB answers, read from `outputs/annotations/erb-task1/`: the four annotators' exports
  (`erb-task1-<A-D>-*.json`), the tie-break file (`erb-task1-ties-*.json`), the packet they graded
  (`task1-items.json`, whose SHA-256 each of them names), and the coordinator's key (`coordinator-key.csv`) and
  answers (`answers.jsonl`). The key must agree with `answers.jsonl` on each item's configuration, collection run,
  question and trace, and that trace's question and answer must be the ones the packet shows for the item, so a key
  of another packet or with swapped items or questions stops the analysis. An answer's final grade is the majority;
  2–2 ties and flagged answers take the tie-break grade. Each answer is matched to its run sample by the key's trace,
  and only the annotated answers' AC units are read, so a failure elsewhere in a run does not stop the analysis.
  Per arm and repetition, it gives Spearman's ρ of AC against the final grades and against each annotator, with their
  mean. For the annotators, it gives ρ for each pair and their mean, Fleiss' κ on the answers all four graded, the
  4–0, 3–1 and 2–2 splits, the flagged and tie-break counts, and each annotator's share of answers graded correct.
- `retest` (RQ1 Retest; `--arms`: `main`) compares the three passes of the main arm, pair by pair (1-2, 1-3, 2-3),
  over the items both passes have: Cohen's κ on the `relevance` and `attribution` decisions, Spearman's ρ on the
  per-question CP, CR, AC and PC scores of `scores`, pooled per benchmark over the six configurations (and τ²-bench's
  for PC). A `mean` row per benchmark and measure averages the three pairs once all three are in.
- `scores` (Table 3) gives each run's configuration scores (`--arms`: `main`, `sc`, `wt`, `deepeval`), the mean of its
  per-question scores, as METHODOLOGY's *Aggregation* defines them, with the number of questions behind each; R and
  NDCG@10 come from the source run. `R on CR` is R over the questions CR scores, such as ERB's Basic and Semantic
  questions; its count matches CR's when every such question has an R. The run report's score statistics pool every
  computation instead, so they weigh a question by its number of searches and are not these scores. Printed values
  round half up, as the paper does.
- `rankings` (RQ2; `--arms`: `main`, `sc`, `deepeval`) ranks the six reported configurations of each benchmark,
  arm and repetition by their `scores` (Table 3) for AC, CP, CR, R, `R on CR` and NDCG@10, each with its number of
  questions. Kendall τ-b compares each ranking with its reference from *RQ2 references*: CP's with NDCG@10's, CR's with
  R's and with `R on CR`'s. Tied scores share a rank. A configuration without a run, or without scores for the metric,
  shows –, and leaves the metric and its reference without τ.
- `power` (RQ2 Discriminative power; `--arms`: `main`, `sc`, `deepeval`) gives, for each benchmark, arm,
  repetition and metric of `rankings`, the share of the 15 pairs of the six configurations whose scores differ
  significantly, as METHODOLOGY's *RQ2 references* define it: a paired bootstrap of each pair's per-question
  differences, on the questions both configurations scored, with a fixed seed, then Holm's correction over the 15
  p-values. The JSON rows hold each pair's question count, mean difference, p and corrected p. A pair with fewer
  than two shared questions has no p, and leaves the metric's power empty.
- `answer-rankings` (RQ2 Answers; `--arms`: `main`, `wt`, `deepeval`) reads the annotations as `answers` does and
  keeps AC to the 30 annotated ERB questions. For the final grades and for each arm and repetition, it gives each
  configuration's mean over its annotated answers, with their number and its rank, and the discriminative power as
  `power` computes it; for AC, Kendall τ-b compares its ranking with the final grades'. A configuration with fewer
  scored answers, such as WT's failed ones, is still compared with the humans' mean over all 30.
- `known-order` (RQ2 Known order; `--arms`: `main`, `sc`, `deepeval`) compares each degraded configuration's base
  (ReAct/GPT on ERB) with its degraded runs, for each arm and repetition that has them: the `scores` of each metric
  at f = 0, 0.25 and 0.5, Kendall τ-b against the known order f = 0 ≻ 0.25 ≻ 0.5, and the relative drop from f = 0 to
  f = 0.5, which compares how much the retrieval metrics and AC fall. R and NDCG@10 check that the swap took effect:
  below τ-b = 1, the degraded runs must be collected again. They are read on every question of the collection run,
  not only on those the judge run completed, so the check does not depend on the judge's coverage.
- `trace-length` (RQ3 Trace length; `--arms`: `main`, `wt`) splits the 180 annotated ERB answers and the τ²-bench
  trajectories into three parts of equal size by the length of their rendered whole trace, read from the WT units
  (`trace_render.chars`) whatever `--arms` selects. In each part and for each arm and repetition, it gives `answers`'
  Spearman ρ of AC against the final grades and `plans`' AUROC of PC and of matching against the reward, with the
  scored items and the positives (answers graded correct, rewarded trajectories) among them. WT's failed units have no
  score, so its long part can hold fewer scored answers than main's.

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
that the researchers a failed one leaves running stop with the call. `DeepAgentGraphTest` (in
`tests/test_deep_agent.py`) runs the deep agent's graph the same way: the searches of the orchestrator and of a
sub-agent, a rejected search, and a search result moved to the agent's files, whose span still holds the ten documents.

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
   - the main pass and two retests, and the SC and WT ablations (`syllo-exp judge`);
   - the DeepEval baseline (`syllo-exp deepeval`);
   - the AgentEvals baseline, on the τ²-bench runs (`syllo-exp agentevals`);
   - the OpenEvals baseline (`syllo-exp openevals`).
3. **Construct.** RQ3 lists and padded traces are imported as new traces.
4. **Annotate.** Humans grade 240 answers and verify ERB facts.
5. **Analyze.** Tables 2–4 and every red placeholder are regenerated from the exports.

A pilot on a small slice gates the full campaign (`benchmarks pilot`, then `collect --pilot`), and `judge --pilot`
checks each arm and its cost range before the judge passes. The judge runs at the model's default temperature and
thinking level.
