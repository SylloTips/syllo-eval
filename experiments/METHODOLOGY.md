# Methodology

These rules are pre-registered. They are frozen before collection starts, and changing one afterwards requires a new
pilot and a note in the manifest. Section numbers refer to the paper draft.

## Datasets

The three benchmarks are pinned by commit and converted as follows (`benchmarks/`).

- **EnterpriseRAG-Bench, 480 samples:**
  - Every question except the 20 `info_not_found` ones is kept.
  - The prompt is the question, verbatim. The expected answer is the gold answer.
  - The gold documents are `expected_doc_ids`.
  - **Repeated ids:** four knowledge-base ids each label two documents (an original and its near-duplicate rewrite).
    The later row gets a `__2` suffix, so every id is unique. A gold id that names two documents makes both gold; this
    affects only `qst_0413`.
  - The 10 `high_level` questions have no gold documents, by design, and are excluded from label-based analyses.
  - **Gold claims** are the answer facts of the 300 Basic and Semantic questions: 1,013 claims, each with a single
    gold document. Many facts of constrained and conflicting questions are grading instructions, so they are not used.
- **WixQA, 400 samples:**
  - ExpertWritten (200) comes first, then Simulated (200), in file order.
  - Prompts and answers are stripped at the ends only; two prompts change.
  - A sample is keyed by config and 0-based row.
- **τ²-bench retail, 114 samples:**
  - **Prompt:** the user scenario, rendered exactly as tau2 renders it for its user simulator. This was verified
    byte-identical with tau2 on all 114 tasks.
  - **Expected plan:** the reference tool calls in order. Each step's parameters are the call's arguments, and its
    instruction is empty.
  - Tasks 24 and 57 have no reference calls, so Plan Correctness skips them. For 45 tasks (36 and 70 to 113), the reference lists only the write calls.
  - **Reward at the pin:** 112 tasks multiply the database check by the natural-language assertions, and tasks 33
    and 34 use the database check alone.
  - Only 40 of those 112 tasks have assertions (61 in total), graded by tau2's LLM judge. tau2 scores a task without
    assertions as met, so the other 74 tasks are rewarded on the database check alone.
  - Action matching never enters the reward.

## Evaluation units

- **Retrieval (CP, CR, R, NDCG@10):**
  - One search call is one unit: the ten documents it returned, in rank order. DeepEval and the ablations receive the
    same units with the same document text.
  - The search span carries the user question, so the judge sees the same request as in a single-search agent.
- **Answers (AC):** one unit per question.
- **Plans (PC):** one unit per τ²-bench trajectory. The executed steps are its tool calls, with each call's arguments
  as the step input and its result as the step output. The plan judge also sees each step's status.

## Claims

Claims are stored as ground truth, never re-derived during a run:

| Key | Content | Used by |
|---|---|---|
| `expected_claims_gold` | ERB answer facts, for Basic and Semantic questions (single gold document) | RQ1 attribution |
| `decomposed_claims` | each expected answer decomposed once with the built-in decomposition prompt, after the pilot | RQ2 CR |
| `expected_claims_verified` | ERB facts an annotator found in the gold document, as truncated by the search tool | RQ3 CR |

With stored claims, CR makes one judge call per claim, not the |C| + 1 calls of Table 1.

## Ablations

Each ablation changes one design choice of the metric it is compared with (`ablation_metrics.py`). The judge, units,
skip rules, rubric text, scoring and result metadata stay the same; an ablation only adds metadata fields (WT: the
rendered trace's size; a failed SC unit: its output budget and, for CP, its rank alignment).

- **Syllo-eval-SC** judges all the documents (CP) or all the claims (CR) of a search unit in one call:
  - The prompts are the built-in v1 prompts turned to the plural: each document or statement gets one judgment, in
    rank or given order. The claims are numbered.
  - The output budget is the per-item call's 2,000 tokens per item, capped at the judge's output limit of 65,536.
  - CP judgments are matched to documents by rank, and CR judgments to claims by position. As in the per-item pass,
    the ids and statements the judge echoes are not checked; mismatches are only counted.
  - A unit fails as misaligned unless its judgments match it one to one: in CP each of the unit's ranks exactly once,
    in CR exactly one judgment per claim. It fails as truncated instead when the judgment count is wrong and the
    output used its whole budget.
- **Syllo-eval-WT** reads the agent's whole canonical trace instead of the observations of the target span:
  - The trace is the agent root and its descendants, depth-first, with ordinal span ids and no timestamps. Children
    are ordered by start time, then end time, type, name and finally source span id. Spans outside the agent root,
    such as a reward grader's, are not part of it.
  - Each span shows its typed observations, rendered as the target metrics render them: its request before its
    children, then its retrieval results, plans, executed steps and answer after them. A span without typed
    observations shows its raw input and output instead.
  - The system prompt is the built-in one. The user prompt puts the trace in place of the answer or plan paragraph,
    and its first paragraph ends with one added sentence saying that the answer or plan is the agent root's.
  - Every unit that loads its trace records the rendered trace's span count and length in characters, judge failures
    included. A unit skipped by the built-in rule, or failing to load its trace, records none.
  - It is compared with the built-in Answer and Plan Correctness, which record judge failures the same way.

## Aggregation

- **Question scores:**
  - A question's retrieval score is the mean over its search units.
  - A question with no search scores 0 on the retrieval metrics.
  - An agent failure that persists after retries scores 0 on AC.
- **Configuration scores:** a configuration's score is the mean over questions.
- **Pairing:** rankings and bootstraps use the questions scored for every configuration, and their count is reported.
- **Sensitivity analyses**, reported next to the primary results:
  - R over the union of a question's retrieved documents;
  - CR as "claim supported in any search";
  - CP and NDCG@10 from the first search only.

## RQ1 references

- **Decisions:**
  - One decision per (configuration, question, document), taking the first occurrence.
  - Report the share of gold items accepted and the share of other items accepted. The second share is an upper bound
    on false positives, because non-gold items can be relevant. Cohen's κ is secondary.
  - Degraded runs are excluded.
- **Answers:**
  - Two annotators grade on the AC anchors (1.0, 0.7, 0.4, 0.0), blind to configuration and judge output, and an
    adjudicator resolves every disagreement.
  - Report Spearman ρ of the judge vs. the adjudicated grades.
  - Report judge vs. each annotator next to annotator vs. annotator.
- **Plans:**
  - Plan Correctness "accepts" a trajectory at a score of at least 0.7.
  - The deterministic baseline is τ²-bench's action check: every gold action is present with matching arguments, in any
    order. A strict sequence match is reported as secondary.
  - A deviating success has reward 1 and fails the action check. Acceptance on deviating failures is a control.
  - AUROC is pooled over trajectories, with a task-level cluster bootstrap, plus within-task AUROC on tasks with mixed
    outcomes.
- **Retest:** three Syllo-eval passes on the same stored traces; mean pairwise κ for decisions and ρ for scores.

## RQ2 references

- AC is ranked against human grades on the annotated questions, CP against NDCG@10, and CR against set recall.
- **Kendall τ-b** compares the rankings of the six configurations.
- **Discriminative power:**
  - It is the share of the 15 configuration pairs that differ significantly.
  - Significance uses a paired bootstrap over questions (B = 10,000) with Holm correction per metric at p < 0.05.
  - It is also computed for the human grades, R and NDCG@10, and for AC on the annotated questions.
- **Known order:** ReAct/Sonnet on ERB with f = 0 ≻ 0.25 ≻ 0.5, checked with R and NDCG@10.

## RQ3 constructions

- **Lists:**
  - Each query appears, for every n ∈ {5, 10, 20, 40, 80}, with the relevant item first, middle, last, and absent.
  - Negatives exclude near-duplicates of the gold item, and for CR also documents containing a fact.
  - Lists are nested across n and seeded.
  - Balanced accuracy is computed over item (CP) or claim (CR) decisions, per n and per position.
- **Traces:**
  - Terciles use the length of the rendered whole trace in characters, which every judged WT unit records. Input
    tokens reported by the judge would leave out the units that failed, which are the longest.
  - Padding adds non-gold search spans as children of the agent root, so that they are part of the trace WT reads:
    +32k, +64k and +128k tokens. Their start times decide where they render among the agent's own spans, so the
    construction fixes them before the pilot.
  - Syllo-eval's answer prompt must be byte-identical under padding, so it is not re-run.
- **Cost:** judge calls, input tokens, cached input tokens and latency per sample. They are measured one sample at a
  time, with the same judge concurrency for every framework.

## Failures

- Whole-trace context overflows and timeouts count as wrong, as do misaligned or truncated single-call and DeepEval
  outputs.
- Excluding instead every unit that failed with an outcome of the judge (listed below), in every arm, is reported only
  as a sensitivity analysis.
- The experiment metrics record why a unit failed in `metadata['failure']`, the same way in every arm (main pass,
  retests and ablations):
  - `misaligned`, `truncated`, `invalid_output`, `context_overflow` and `timeout` are outcomes of the judge, and count
    as wrong in every arm;
  - `invalid_output` is an output that does not parse or validate, or a refusal; an output that does not parse because
    it reached its token limit is `truncated`;
  - `provider` and `trace_load` are infrastructure failures: the unit is re-run in every arm, not counted.
- Each document or claim of a failed CP or CR unit counts as a wrong decision; the computation's span and ground truth
  identify them. A failed AC or PC unit has no decisions, only a missing score.
- A failed unit counts every judge call it made, with the usage the provider reported, failed calls included.
