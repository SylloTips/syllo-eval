# Trace normalization

Spans carry raw JSON input/output plus typed `SpanSemantics`: request, answer, ordered retrieval result sets,
plan snapshots, executed steps, and own-call LLM usage. Public metrics read only the typed observations.

## Adapters

A `TraceAdapter` implements `normalize(trace_id, records) -> TraceProcessingResult`. Register one per agent:

```python
EvaluationService(
  settings=settings,
  callers_by_agent_name={'my-agent': caller},
  trace_adapters_by_agent_name={'my-agent': MyTraceAdapter()},
)
```

- Imports select an adapter by name from `trace_adapters_by_name`, plus the built-in `phoenix`; the file's `spans`
  reach `normalize` unchanged.
- Unregistered agents use `PhoenixTraceAdapter`, which decodes telemetry and status but infers no agent-specific semantics.
- The processor validates the whole bundle and persists it transactionally. Identical re-ingestion is idempotent;
  a changed interpretation needs a new trace identity. Namespace source IDs another provider could reuse.
- Adapter-emitted custom span types are upserted, so custom metrics can target them.
- Adapter instances are caller-owned and must support concurrent normalization.

## Observations

- Missing is not empty: `available` with no items is an observed empty result; `unavailable`, `not_attempted`, or an
  absent result set mean the observation is not established. Never reconstruct it from reprs, summaries, or guessed timing.
- Retrieval sets declare `kind` and `stage` (`retrieved`, `reranked`, `selected`, `generation_context`). List order is rank
  only when `ranked` (the default); adapters set `ranked=False` for filtered or accumulated lists, and nDCG and contextual
  precision skip those. Scores are optional and never compared across searches. A malformed item makes only its set `unavailable`, with item
  indices in `reason`, so metrics skip rather than score a shortened list.
- Built-in RAG metrics score `agent_root` at stage `selected`, separately for documents and snippets; their
  `target_span_types` argument, or `EVALUATION_RETRIEVAL_SPAN_TYPES`, makes them score other spans, such as one span per
  search call. Several ranked result sets on one span are separate rankings: rank metrics score each one and average
  them, and never concatenate them. Custom subclasses can override `retrieval_stage`; GROUP metrics choose how to
  aggregate several searches.
- `PlanningData.plans` holds versioned plan snapshots; `executed_steps` holds actual attempts, including errors. Do not infer
  dependencies or plan membership from equal instruction text. Dataset expected plans use `operation`, `instruction`,
  and optional JSON `parameters`; unknown fields are rejected. The plan judge sees each executed step's `status`,
  `instruction`, `input` and `output`, and each expected step's `instruction` and `parameters`; it omits unobserved
  fields, including an `unknown` status, so put tool arguments in `input` rather than in the instruction text.
- LLM usage is optional (missing is not zero) and is counted once per trace/call identity, separately from judge usage.

## Request-ID lookup

Callers return a request ID; the Phoenix client resolves it to the trace whose root span carries a matching `request_id`
attribute. Root spans named in `PhoenixSettings.request_id_excluded_root_span_names` (case-insensitive) are skipped.
