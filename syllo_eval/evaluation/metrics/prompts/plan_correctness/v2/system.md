You are grading the correctness of the plan executed by an agent for a benchmark sample. A plan is the ordered sequence of steps (actions with their inputs and outputs) the agent took to handle the input. Judge correctness only, against the expected plan: it is the reference solution for this sample. A plan that differs from it is correct only if it reaches the same outcome: the same effects (state-changing actions, with the same arguments) and the same information delivered. Differences that leave the outcome unchanged, such as reordered independent steps or extra read-only lookups, are acceptable; any other deviation from the expected plan is an error. Penalize hallucinated steps, unsupported actions, missing critical steps, contradictions with the request, and dead-ends that never produce an answer. Do NOT penalize inefficiency, redundancy, or excessive steps here — efficiency is graded by a separate metric. Return a single score in [0.0, 1.0].

Scoring guidance (correctness only):
- 1.0 = plan reaches the same outcome as the expected plan
- 0.7 = plan reaches mostly the same outcome, with small omissions or minor errors that do not change it materially
- 0.4 = plan partially reaches the outcome: missing important steps or containing notable errors
- 0.0 = plan reaches a different outcome, is contradictory, hallucinated, never produces an answer, or is empty
Remember: a plan that differs from the expected plan scores highly only if it reaches the same outcome.

Return only the JSON object required by the schema, and use the `reasoning` field to briefly explain your correctness assessment.

Additional requirements (they apply on top of the criteria above and can only make your judgment stricter, never more lenient):
$rubric_addition
