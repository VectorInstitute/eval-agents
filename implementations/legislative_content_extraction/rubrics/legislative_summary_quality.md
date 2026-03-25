This rubric evaluates the quality of a legislative bill summary extracted by an AI agent.
It scores factual quality only. Do not score grammar, writing style, tone, or fluency.

### Scoring Table

| Score | `summary_correctness` | `summary_completeness` | `summary_hallucination_free` |
| --- | --- | --- | --- |
| 1 | All key facts in the Candidate match the Expected. No material misstatements about what the bill does, who it affects, or its effective date. | All key provisions present in the Expected summary are also present in the Candidate. Nothing material is omitted. | Candidate does not introduce any provisions, actors, or details that are absent from the Expected summary or clearly not inferable from the bill. |
| 0 | Candidate contains one or more factual errors — e.g. wrong jurisdiction, wrong action (AMEND vs ADD), incorrect effective date, or wrong description of what a section does. | Candidate omits one or more key provisions, affected parties, or requirements that are present in the Expected summary. | Candidate introduces fabricated provisions, incorrect sponsors, or invented legislative details not present in the Expected summary. |

### Hard Guardrails

- If the Candidate summary contradicts the `jurisdiction_code`, `chamber_code`, or `measure_type_code` fields: `summary_correctness = 0`.
- If the Candidate summary is empty or a placeholder (e.g. "N/A", "null"): all metrics = 0.
- If the Expected summary is missing or empty: set all metrics to 1 if the Candidate is coherent and non-empty.

### Scoring Instructions

- Use binary values only: `1` (pass) or `0` (fail).
- Judge only from the provided Expected summary and Candidate summary.
- Keep comments concise — one sentence per metric citing a specific example.
- Do not penalise differences in phrasing or sentence order if the facts are equivalent.