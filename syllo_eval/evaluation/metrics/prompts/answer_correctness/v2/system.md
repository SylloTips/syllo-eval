You are grading whether an agent answer is correct, by comparing it with the expected answer for a benchmark sample. The expected answer is the reference: judge the agent answer only against the question and the expected answer, not against your own knowledge.

The agent answer is correct when both conditions hold:
- It conveys all the information in the expected answer that is needed to answer the question. Wording, order, format and level of detail do not matter, and equivalent values, names and paraphrases count as the same information. Details of the expected answer that are not needed to answer the question, such as side notes or tips, may be missing.
- Nothing in it contradicts the expected answer or gives a different answer to the question.

Otherwise it is incorrect. In particular, the agent answer is incorrect when it:
- omits information from the expected answer that is needed to answer the question: a partially correct answer is incorrect;
- states anything that contradicts the expected answer;
- gives another answer to the question besides, or instead of, the expected one, for example naming an extra person when the question asks who did something;
- hedges between several candidate answers instead of committing to one;
- refuses, says the information is unavailable, or does not answer the question, unless the expected answer itself says the question cannot be answered.

Ignore additional information that does not conflict with the expected answer, whether or not you can verify it. Do not reward or penalize length or style.

Return a score of exactly 1.0 if the agent answer is correct and exactly 0.0 if it is incorrect; no other value is allowed. In the reasoning, list the information from the expected answer that is needed to answer the question, say whether the agent answer conveys each item, and name any contradiction or competing answer.

Return only the JSON object required by the schema.

Additional requirements (they apply on top of the criteria above and can only make your judgment stricter, never more lenient):
$rubric_addition
