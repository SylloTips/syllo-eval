---
name: add-metric
description: Use when adding a new evaluation metric to this codebase — covers picking a base class in contracts.py, implementing the contract, placing the implementation, and registering it in available.py so sync_with_persistence() upserts it.
---

# Adding a New Metric

1. Pick a base class in `syllo_eval/evaluation/metrics/contracts.py`:
   - `SpanEvaluationMetric` for per-span metrics (`targeting_mode = SINGLE`)
   - `SpanGroupEvaluationMetric` for ordered-group metrics (`targeting_mode = GROUP`)
   - For LLM-as-a-judge metrics, extend `BaseLlmJudgeMetric` from `syllo_eval/evaluation/judge/metric_base.py`.
2. Set the class attributes `metric_name` and `metric_description` — do **not** override `name`/`description` as properties; `EvaluationMetric` derives them. The name must be readable before an instance exists, because the inventory is inspected to decide which provider clients a run allocates.
3. Implement `target_span_types`, `ground_truth_keys` (the `GroundTruthKey` values the metric reads; every declared key is required unless `ground_truth_skip_reason` says otherwise, e.g. `return None` for optional keys), and `compute(target, ground_truths)` (where `target` is a `Span` or a `Sequence[Span]` depending on targeting mode, and `ground_truths` maps each declared key the sample has to its `GroundTruth`).
4. If the metric needs a provider client, declare it on the class: `requires_judge_client = True` and/or `requires_claim_extractor_client = True`. Extending `BaseLlmJudgeMetric` already declares the judge. Accept the dependency as a **keyword-only** argument of the same name (`def __init__(self, *, judge_client: LlmJudgeClient, rubric_addition: str | None = None)`) — that is how the runtime injects it. Every built-in judge metric also accepts `rubric_addition`, the run's extra requirements for that metric.
5. Make judge calls through `judge_batch` (`syllo_eval/evaluation/judge/batch.py`), so a failure becomes a FAILED result with a `metadata.failure` class and the usage of every call; `BaseLlmJudgeMetric` already does, and its async `judge_input` hook builds a prompt that needs awaited data.
6. For judge metrics, put the prompt wording in versioned templates under `syllo_eval/evaluation/metrics/prompts/<prompt>/v1/` (`system.md`, `user.md`), set `prompt_version = 'v1'`, and build prompts with `render_prompt(f'<prompt>/{self.prompt_version}/system.md', ...)`. `$name` placeholders take the values; a paragraph whose value is `None` is omitted. Keep prompts cache-friendly: the system prompt must not vary by sample, sample values go at the end of the user prompt, and the system template ends with the rubric-addition paragraph (`$rubric_addition`) used by the existing templates. To change a released prompt, add `v2` and bump `prompt_version`.
7. Place the implementation under `syllo_eval/evaluation/metrics/implementations/<domain>/` and export it from the relevant `__init__.py`.
8. Add the class to the single `BUILTIN_METRICS` tuple in `syllo_eval/evaluation/metrics/available.py` so that `sync_with_persistence()` upserts it. There is no separate table per dependency kind — availability, client requirements and construction all derive from the class attributes.
