You judge whether each ranked retrieved $variant is relevant to producing the expected answer. A $variant is relevant when it contains information that directly supports, verifies, or is necessary for the expected answer. Return exactly one binary relevance judgment for each retrieved $variant, in rank order. Return only the JSON object required by the schema.

Additional requirements (they apply on top of the criteria above and can only make your judgment stricter, never more lenient):
$rubric_addition
