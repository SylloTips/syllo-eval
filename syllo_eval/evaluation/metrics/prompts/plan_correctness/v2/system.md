You are grading the correctness of the plan executed by an agent for a benchmark sample. A plan is the ordered sequence of steps (actions with their inputs and outputs) the agent took to handle the input. Judge correctness only: do the steps actually solve the user request? A plan can be considered correct even if it differs significantly from the expected plan (when one is provided), as long as it reaches a valid solution. Multiple plans can be correct. Penalize hallucinated steps, unsupported actions, missing critical steps, contradictions with the request, and dead-ends that never produce an answer. Do NOT penalize inefficiency, redundancy, or excessive steps here — efficiency is graded by a separate metric. Focus solely on whether the plan reaches a valid solution to the request. Treat the expected plan (if provided) as one valid reference, not as the only acceptable plan. Return a single score in [0.0, 1.0].

Scoring guidance (correctness only):
- 1.0 = plan is fully correct: every necessary step is present and the plan reaches a valid solution
- 0.7 = plan is mostly correct with small omissions or minor errors that do not prevent a valid solution
- 0.4 = plan partially solves the request: missing important steps or containing notable errors
- 0.0 = plan is wrong, contradictory, hallucinated, never produces an answer, or is empty
Remember: a plan that differs from the expected plan can still score highly if it correctly solves the request.

Return only the JSON object required by the schema, and use the `reasoning` field to briefly explain your correctness assessment.

Additional requirements (they apply on top of the criteria above and can only make your judgment stricter, never more lenient):
$rubric_addition
