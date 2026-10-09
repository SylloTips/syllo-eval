# Dify agent: configuration

[`react-deepseek.yml`](react-deepseek.yml) is the DSL export of the paper's low-code agent, a Chatflow built in Dify's
UI: Start, an Agent node with the ReAct strategy, and Answer. It defines the instruction, the model, the tool, at most
100 iterations, and no memory, so every question is answered on its own.

The export holds no credentials and no workspace settings. Each Dify instance needs them set in the UI, in this order,
because the import resolves the model and the tool by name:

1. [Stack](#1-stack)
2. [Plugins](#2-plugins)
3. [Model provider](#3-model-provider)
4. [MCP server](#4-mcp-server)
5. [Import and publish](#5-import-and-publish)
6. [Tracing](#6-tracing)
7. [API key](#7-api-key)
8. [Check](#8-check)

## 1. Stack

From `experiments/`, start the campaign environment first, then Dify, as described in the
[README](../README.md#dify):

```bash
docker compose up -d
scripts/dify.sh up -d
```

Dify calls the search tool through its `ssrf_proxy`, which refuses private addresses. Add
`SSRF_PROXY_ALLOW_PRIVATE_DOMAINS=host.docker.internal` to `data/dify/.env`, then run
`scripts/dify.sh up -d --force-recreate ssrf_proxy`. On Linux, see also
[Connecting the agents](../README.md#connecting-the-agents).

Create the admin account at `http://localhost/install`.

## 2. Plugins

Install from Plugins > Marketplace:

| Plugin | Version | Provides |
|---|---|---|
| `langgenius/agent` | 0.0.50 | the ReAct strategy of the Agent node |
| `langgenius/deepseek` | 0.0.24 | the model provider |

The export pins the agent plugin, but its `dependencies` list is empty, so the import does not offer to install either
plugin.

## 3. Model provider

Under Settings > Model Provider > DeepSeek, enter the API key and leave the base URL empty
(`https://api.deepseek.com`).

The Agent node selects `deepseek-flash`, which plugin 0.0.24 labels DeepSeek V4.1 Flash and sends to DeepSeek under
that name. DeepSeek can move this alias to a later Flash release, so check the label in the model picker before each
campaign. The export sets no model parameters, so the plugin's defaults apply: thinking mode is on, which drops
`temperature`.

**Changing provider** (DeepSeek is to move to Azure): the provider and the model are the `model` block of the Agent
node (`provider: langgenius/deepseek/deepseek`, `model: deepseek-flash`). Install the new provider's plugin, select the
same model in the Agent node, check that the instruction, tool and iterations are unchanged, then export again and
replace this file and the plugin table above. Update the agent's model in
[`configs/models.yaml`](../configs/models.yaml) if its name changes.

## 4. MCP server

Start the search server on the host, on port 8101, as in [Search tool](../README.md#search-tool). Under Tools > MCP >
Add MCP Server (HTTP), set:

| Field | Value |
|---|---|
| Server URL | `http://host.docker.internal:8101/mcp` |
| Name | `Knowledge base` |
| Server identifier | `knowledge-base` |

Leave authentication and headers empty. Once saved, the server lists `search_knowledge_base`.

The Agent node refers to the tool by the server identifier (`provider_name: knowledge-base`). With any other
identifier, the import leaves the node without its tool. The URL stays the same for every configuration: switch
configurations by relaunching the server on the same port.

## 5. Import and publish

In Studio, choose Import DSL file and select `react-deepseek.yml`. In the Agent node, check that the strategy is ReAct,
the model is the one above, `search_knowledge_base` is enabled and Maximum Iterations is 100.

Then select Publish. The service API runs the published version; the studio's Preview runs the draft, so a draft-only
app answers in Preview but not through the caller.

## 6. Tracing

Tracing is configured per app and is not exported, so an imported app starts with tracing off. Under the app's
Monitoring > Tracing, configure Arize Phoenix:

| Field | Value |
|---|---|
| Endpoint | `http://phoenix:6006` |
| Project | the project of the app's benchmark, `erb-69916e3` or `wixqa-d662dc4`: `collect` looks there |
| API key | empty: the campaign's Phoenix has no authentication |

Then turn tracing on. Dify's api and worker reach `phoenix` because
[`compose.override.yaml`](compose.override.yaml) connects them to the campaign network.

## 7. API key

Under the app's API Access, create an API key. Each `react` configuration runs on its own app, with its model and its
benchmark's project, so set `DIFY_API_KEY` to the key of the configuration's app when collecting it:

```bash
DIFY_API_KEY=app-... poetry run syllo-exp collect --configuration wixqa/react/deepseek
```

## 8. Check

Run a pilot of the configuration on a few questions (see [Collection](../README.md#collection)):

```bash
poetry run syllo-exp benchmarks pilot --benchmark wixqa --samples <sample keys>
DIFY_API_KEY=app-... poetry run syllo-exp collect --configuration wixqa/react/deepseek --pilot
```

Each question's trace is saved to `outputs/traces/<run id>/<trace id>.json`. Its spans read:

```text
<workflow run id> [chain]
  workflow_<workflow run id> [chain]
    start_Start [chain]
    agent_Agent [agent]
    answer_Answer [chain]
```

Check that:

- the workflow span's `metadata.triggered_from` is `app-run`, so the published version ran;
- every search is in the agent span, with all its documents in rank order, ids and full texts, as the search server's
  call log (`outputs/search_calls/`) records them.

## What the trace holds

The trace adapter reads Dify's traces as follows:

- **Two traces per message.** Dify exports a workflow trace and a separate message trace. The workflow trace holds
  the nodes. Its root carries `dify_trace_id`, set to the `trace_id` the caller sends, which is the request ID. The
  message trace has a root named `Dify` with the message ID, and only a `message` span and an `llm` span, whose input
  is empty and whose token counts are zero. A lookup by message ID would find this trace, so the caller sends its own
  ID instead.
- **No tool or LLM spans.** The ReAct rounds are in the `agent_Agent` span, as `metadata.agent_log.<n>.*` attributes
  and in the `json` list of its `output.value`. Each round has three entries:
  - `ROUND <k>`, with the thought, `action_name`, `action_input` and `observation`;
  - `<model> Thought`, the model call, with its total tokens;
  - `CALL search_knowledge_base`, with `tool_call_args` and the output `tool response: <search JSON>`.

  Token usage is only a total per call: prompt and completion tokens read zero.
- **Large outputs.** Dify stores node outputs above 100,000 characters per string, or 1000 KiB in all, in its file
  storage, keeps a truncated copy in its database, and reloads the full output when it exports the trace.
