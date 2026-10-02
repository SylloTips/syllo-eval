# Methodology

These rules are pre-registered. They are frozen before collection starts, and changing one afterwards requires a new
pilot and a note in the manifest. Section numbers refer to the paper draft.

## Evaluation units

- **Retrieval (CP, CR, R, NDCG@10):**
  - One search call is one unit: the ten documents it returned, in rank order. DeepEval and the ablations receive the
    same units with the same document text.
  - The search span carries the user question, so the judge sees the same request as in a single-search agent.
- **Answers (AC):** one unit per question.
- **Plans (PC):** one unit per τ²-bench trajectory. The executed steps are its tool calls, and each call's arguments
  appear in its instruction text, because the plan judge renders operation, instruction and output only.

## Claims

Claims are stored as ground truth, never re-derived during a run:

| Key | Content | Used by |
|---|---|---|
| `expected_claims_gold` | ERB answer facts, for Basic and Semantic questions (single gold document) | RQ1 attribution |
| `decomposed_claims` | each expected answer decomposed once with the built-in decomposition prompt, after the pilot | RQ2 CR |
| `expected_claims_verified` | ERB facts an annotator found in the gold document, as truncated by the search tool | RQ3 CR |

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
  - Terciles use the input tokens the whole-trace judge reported.
  - Padding adds non-gold search spans: +32k, +64k and +128k tokens.
  - Syllo-eval's answer prompt must be byte-identical under padding, so it is not re-run.
- **Cost:** judge calls, input tokens, cached input tokens and latency per sample. They are measured one sample at a
  time, with the same judge concurrency for every framework.

## Failures

- Whole-trace context overflows and timeouts count as wrong, as do misaligned or truncated single-call and DeepEval
  outputs.
- Excluding them instead is reported only as a sensitivity analysis.
