You judge whether each factual statement from the expected answer is attributable to retrieved ${variant}s. A statement is attributable when the retrieved ${variant}s contain enough information to directly support it. Return exactly one binary attribution judgment for each statement, in the order given. Return only the JSON object required by the schema.

Additional requirements (they apply on top of the criteria above and can only make your judgment stricter, never more lenient):
$rubric_addition

The statements are part of the expected answer to the question below. Use the question only to understand what the statements refer to: it is not evidence, so a statement is attributable only when the retrieved ${variant}s support it.
Question:
$question
