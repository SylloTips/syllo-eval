You judge whether one factual statement from the expected answer is attributable to retrieved ${variant}s. A statement is attributable when the retrieved ${variant}s contain enough information to directly support it. Return exactly one binary attribution judgment for the statement. Return only the JSON object required by the schema.

Additional requirements (they apply on top of the criteria above and can only make your judgment stricter, never more lenient):
$rubric_addition

The statement is part of the expected answer to the question below. Use the question only to understand what the statement refers to: it is not evidence, so the statement is attributable only when the retrieved ${variant}s support it.
Question:
$question
