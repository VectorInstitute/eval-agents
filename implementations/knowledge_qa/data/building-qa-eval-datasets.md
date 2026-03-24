# Building Ground-Truth QA Datasets for Agent Evaluation

This guide distills the DeepSearchQA methodology into a reusable pattern for building evaluation datasets for your own agent use case.

---

## Core Idea

An evaluation dataset is a list of `(question, ground_truth)` pairs where ground truth is a **minimal, verifiable set of facts**. An LLM judge checks whether the agent's response contains those facts, and you measure precision, recall, and F1.

---

## Dataset Structure

Each example needs four fields:

| Field | Type | Description |
|---|---|---|
| `id` | int | Unique identifier |
| `question` | str | The query given to the agent |
| `answer` | str | The ground truth (see below) |
| `answer_type` | str | `"Single Answer"` or `"Set Answer"` |

**`Single Answer`** — one correct fact. Wording flexibility is fine; the judge checks semantic equivalence.
```
question: "Who wrote Pride and Prejudice?"
answer:   "Jane Austen"
```

**`Set Answer`** — multiple independent facts that must all be present.
```
question: "Which countries border Switzerland?"
answer:   "France, Germany, Austria, Italy, Liechtenstein"
```

---

## How the Judge Works

The judge (an LLM) receives `(question, agent_response, ground_truth, answer_type)` and returns:

- **Correctness Details**: `{"France": true, "Germany": true, "Austria": false, ...}`
- **Excessive Answers**: items in the response not in the ground truth

From that, standard IR metrics are computed:

```
Precision = matched / (matched + extraneous)  # Is the agent's answer focused?
Recall    = matched / total_ground_truth       # Did the agent find everything?
F1        = 2·P·R / (P+R)                     # Overall quality
```

**Four outcomes:** `fully_correct`, `correct_with_extraneous`, `partially_correct`, `fully_incorrect`.

---

## Tips for Writing Good Ground Truth

### 1. Use atomic facts, not prose

Bad:
```
answer: "The Treaty of Westphalia was signed in 1648 and ended the Thirty Years' War in Europe."
```
Good (Set Answer):
```
answer: "1648, Thirty Years' War"
```
Atomic items are unambiguous for the judge to verify independently.

### 2. Match specificity to what you're testing

If your agent should return a year, put a year — not a full date unless the full date matters. Ground truth that's overly specific fails agents that are technically correct.

### 3. Be careful with Set Answer ordering

The judge ignores order by default. Only specify order constraints explicitly in the question if order matters: `"List the top 3 countries by GDP in order"`.

### 4. Decide what counts as extraneous

Extraneous items lower precision. If your use case tolerates verbose but accurate responses, weight recall higher. If you need tight, factual answers (e.g., structured data extraction), penalize precision misses.

### 5. Avoid unanswerable or ambiguous questions

If a question has multiple valid answers depending on interpretation, the judge will inconsistently flag extraneous items. Either narrow the question or add constraints (`"as of 2024"`, `"according to source X"`).

---

## How to Construct Your Dataset

### Option A: Manual curation (highest quality)
1. Write questions that reflect real user queries in your domain.
2. Look up the answer from a trusted source.
3. Decompose the answer into atomic facts.
4. Assign `answer_type` based on whether it's one fact or many.

### Option B: LLM-assisted generation + human review
Use an LLM to generate `(question, answer)` pairs from a reference document, then have a human verify each answer against the source. Prompt pattern:

```
Given this document: <doc>
Generate 5 factual questions and their concise ground truth answers.
For each answer, decompose it into atomic facts if there are multiple.
Format: {"question": "...", "answer": "...", "answer_type": "Single Answer" | "Set Answer"}
```

### Option C: Mine from existing logs (fastest)
Take real queries from your system logs. For each, manually write the correct answer. This gives you realistic distribution coverage quickly.

---

## Example: Customer Support Agent

```python
examples = [
    {
        "id": 1,
        "question": "What is the return window for electronics?",
        "answer": "30 days",
        "answer_type": "Single Answer",
    },
    {
        "id": 2,
        "question": "What payment methods do you accept?",
        "answer": "Visa, Mastercard, PayPal, Apple Pay",
        "answer_type": "Set Answer",
    },
    {
        "id": 3,
        "question": "Who should I contact for a damaged item?",
        "answer": "support@example.com",
        "answer_type": "Single Answer",
    },
]
```

---

## Common Mistakes

| Mistake | Problem | Fix |
|---|---|---|
| Ground truth is a full sentence | Judge has to parse semantics; more error-prone | Use the minimal fact only |
| Set Answer with 10+ items | Noisy; hard to verify | Split into sub-questions or prune to what matters |
| Questions with "best" or "most" | Subjective; no stable ground truth | Add a reference: "according to X" |
| Reusing training-distribution questions | Tests memorization, not reasoning | Use held-out or novel queries |

---

## Minimum Viable Dataset Size

| Use Case | Recommended Size |
|---|---|
| Quick sanity check | 20–50 examples |
| Dev evaluation loop | 100–200 examples |
| Benchmark / publication | 500+ examples, balanced by category |

Stratify by question type or domain category to ensure you're not accidentally testing only easy cases.
