This rubric scores investigation reasoning quality only. It does not score grammar, writing style, tone, or fluency.

### Scoring Table

| Score | `summary_narrative_quality` | `pattern_description_quality` |
| --- | --- | --- |
| 5 | Evidence-grounded and coherent investigation logic. Explicitly considers and rules out plausible benign explanations. Conclusion is fully consistent with cited evidence and final decision fields. | Clear mechanism-level typology description: flow shape, actor roles, fund movement pattern, and temporal logic are explicit and consistent with the conclusion. |
| 4 | Strong evidence linkage and mostly coherent logic, with only minor omissions or weak spots. | Mechanism is mostly correct and specific, with minor incompleteness. |
| 3 | Mixed quality. Some grounding and reasoning are present, but analysis is partially generic, incomplete, or weakly connected to evidence. | Partially correct mechanism but vague, generic, or only partially connected to evidence. |
| 2 | Weak evidence grounding with major reasoning gaps, leaps, or unsupported inferences. | Mostly vague description and/or materially incomplete with partial inaccuracies. |
| 1 | Reasoning is unsupported, contradictory, or materially inconsistent with available evidence. | Incorrect, contradictory, or effectively empty mechanism description. |

### Hard Guardrails (Reasoning Quality With Hard Floors)

- If the narrative contains material unsupported claims: `summary_narrative_quality <= 2`.
- If the narrative contradicts final decision fields (`is_laundering`, `pattern_type`): `summary_narrative_quality = 1`.
- If the pattern description contradicts decision fields or typology semantics: `pattern_description_quality = 1`.
- If the pattern description is effectively placeholder or non-informative: `pattern_description_quality <= 2`.

### Scoring Instructions

- Use integers only: `1`, `2`, `3`, `4`, `5`.
- Judge only from the provided input, expected output, and candidate output.
- Keep comments concise and evidence-focused.

### Special Cases

- If ground-truth `pattern_description` is missing, `N/A`, or equivalent placeholder text, treat any coherent candidate pattern description as valid when it is consistent with other fields (especially `is_laundering` and `pattern_type`).
- Ground-truth `pattern_description` may be terse typology shorthand (for example, `Max 1-degree Fan-In, Max 10-degree Fan-Out, Max 7 hops`). In these cases, evaluate semantic consistency with typology mechanics rather than exact phrasing.
