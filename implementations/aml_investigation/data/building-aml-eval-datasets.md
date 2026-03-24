# Extending AML Agent Evaluation and Datasets

This guide explains how the AML investigation workflow is wired today and how to extend it in a way that stays aligned with the code in:

- [`aieng-eval-agents/aieng/agent_evals/aml_investigation`](../../../aieng-eval-agents/aieng/agent_evals/aml_investigation)
- [`implementations/aml_investigation`](../)

It is meant to help you do two related things:

1. build or extend AML evaluation datasets;
2. add new evaluation dimensions and improve the AML agent itself.

---

## Core Idea

The AML agent is not a plain QA agent. Each evaluation item is a structured investigation case:

- the agent receives a `CaseFile`;
- the evaluator holds a `GroundTruth`;
- the agent returns an `AnalystOutput`.

That means evaluation is also structured. The current system checks:

- whether the agent reached the right laundering verdict;
- whether it picked the right typology;
- whether it flagged the right transaction IDs;
- whether its narrative is evidence-grounded;
- whether its SQL trace stayed read-only, respected the case window, and avoided redundant querying.

The key design principle is: **treat AML evaluation as a combination of outcome quality, reasoning quality, and investigation-process quality**.

---

## Where the Extension Points Live

| Area | File(s) | What to change |
| --- | --- | --- |
| Case schema | [`cases.py`](../../../aieng-eval-agents/aieng/agent_evals/aml_investigation/data/cases.py) | Add new input, ground-truth, or output fields |
| Case generation | [`cases.py`](../../../aieng-eval-agents/aieng/agent_evals/aml_investigation/data/cases.py), [`cli.py`](cli.py) | Build new datasets or new case categories |
| Agent task wrapper | [`task.py`](../../../aieng-eval-agents/aieng/agent_evals/aml_investigation/task.py) | Parse/validate new output schema |
| Agent behavior | [`agent.py`](../../../aieng-eval-agents/aieng/agent_evals/aml_investigation/agent.py) | Prompt, tools, output schema |
| Item-level deterministic metrics | [`item.py`](../../../aieng-eval-agents/aieng/agent_evals/aml_investigation/graders/item.py) | Add field-by-field checks |
| Trace-level metrics | [`trace.py`](../../../aieng-eval-agents/aieng/agent_evals/aml_investigation/graders/trace.py) | Add tool-use or SQL-process checks |
| Run-level metrics | [`run.py`](../../../aieng-eval-agents/aieng/agent_evals/aml_investigation/graders/run.py) | Add aggregate metrics and slices |
| LLM-judge rubric | [`narrative_pattern_quality.md`](../rubrics/narrative_pattern_quality.md) | Change reasoning-quality criteria |
| Evaluation wiring | [`evaluate.py`](../evaluate.py) | Register new evaluators in the experiment |

---

## Current Dataset Structure

The canonical schema lives in [`cases.py`](../../../aieng-eval-agents/aieng/agent_evals/aml_investigation/data/cases.py).

### Agent input: `CaseFile`

| Field | Description |
| --- | --- |
| `case_id` | Unique identifier for the case |
| `seed_transaction_id` | The transaction that triggered the investigation |
| `seed_timestamp` | End of the case window |
| `window_start` | Start of the case window |
| `trigger_label` | Upstream alert label or heuristic hint |

### Ground truth: `GroundTruth`

| Field | Description |
| --- | --- |
| `is_laundering` | Whether the case is truly laundering |
| `pattern_type` | One of the supported typologies plus `NONE` |
| `pattern_description` | Human description of the true pattern |
| `attempt_transaction_ids` | Comma-separated causal-chain transaction IDs |

### Agent output: `AnalystOutput`

| Field | Description |
| --- | --- |
| `summary_narrative` | Evidence summary and reasoning |
| `is_laundering` | Agent verdict |
| `pattern_type` | Agent typology classification |
| `pattern_description` | Agent explanation of the typology |
| `flagged_transaction_ids` | Transactions the agent believes form the laundering pattern |

The dataset file is JSONL, one `CaseRecord` per line:

```json
{
  "input": {
    "case_id": "case-001",
    "seed_transaction_id": "txn-123",
    "seed_timestamp": "2022-09-12T09:05:00",
    "window_start": "2022-09-01T00:00:00",
    "trigger_label": "RANDOM_REVIEW"
  },
  "expected_output": {
    "is_laundering": true,
    "pattern_type": "STACK",
    "pattern_description": "Funds move through sequential intermediary accounts before the seed transaction.",
    "attempt_transaction_ids": "txn-100,txn-101,txn-123"
  },
  "output": null
}
```

---

## What the AML Evaluation Already Measures

Today’s evaluation in [`evaluate.py`](../evaluate.py) combines five distinct perspectives:

| Level | Evaluator | What it measures |
| --- | --- | --- |
| Item | deterministic grader | Verdict correctness, typology correctness, consistency for benign cases, flagged-ID precision/coverage |
| Item | LLM judge | Narrative quality and pattern-description quality via rubric |
| Trace | deterministic grader | Presence of SQL, read-only compliance, time-window discipline, redundant queries |
| Trace | groundedness judge | Whether the final narrative is supported by tool outputs |
| Run | aggregate grader | Precision/recall/F1 for `is_laundering`, macro F1 and confusion matrix for `pattern_type` |

This baseline separates:

- **what answer the agent gave**;
- **why it said that**;
- **how it investigated**.

---

## How to Build New Datasets

### Option A: Use the built-in case generator

The fastest path is the data CLI in [`cli.py`](cli.py):

```bash
uv run --env-file .env python -m implementations.aml_investigation.data.cli create-db \
  --illicit-ratio HI \
  --transactions-size Small

uv run --env-file .env python -m implementations.aml_investigation.data.cli create-cases \
  --illicit-ratio HI \
  --transactions-size Small \
  --num-laundering-cases 20 \
  --num-false-positive-cases 10 \
  --num-false-negative-cases 10 \
  --num-normal-cases 10 \
  --lookback-days 10 \
  --output-dir implementations/aml_investigation/data
```

This route uses:

- [`normalize_transactions_data()`](../../../aieng-eval-agents/aieng/agent_evals/aml_investigation/data/utils.py)
- [`parse_patterns_file()`](../../../aieng-eval-agents/aieng/agent_evals/aml_investigation/data/cases.py)
- [`build_cases()`](../../../aieng-eval-agents/aieng/agent_evals/aml_investigation/data/cases.py)

### Option B: Bring your own transaction dataset

If you already have transaction data:

1. normalize it to the schema expected by [`normalize_transactions_data()`](../../../aieng-eval-agents/aieng/agent_evals/aml_investigation/data/utils.py);
2. load it into a database compatible with [`schema.ddl`](schema.ddl);
3. generate `CaseRecord` rows directly, or create a compatible patterns file and call `build_cases()`.

This is the right path when your real-world typologies, alert labels, or entity behavior differ from the Kaggle/IBM sample dataset.

### Option C: Use an LLM with few-shot ground-truth patterns

Another practical path is to use an LLM to draft new cases from a small set of validated laundering patterns.

This works especially well when you already have:

- a handful of trusted `CaseRecord` examples;
- known typologies with correct `attempt_transaction_ids`;
- seed examples of good false positives and false negatives.

The few-shot examples teach the model:

- how `CaseFile`, `GroundTruth`, and `CaseRecord` are structured;
- how to keep `attempt_transaction_ids` limited to the causal chain;
- how to vary `trigger_label` to create easy, weak-signal, or misleading cases;
- how to write short but concrete `pattern_description` values.

Good uses of LLM assistance:

- drafting additional case files from known laundering chains;
- generating hard-negative variants by weakening or misleading the `trigger_label`;

Prompt pattern:

```text
You are helping build an AML evaluation dataset.

Here are 3 validated examples of AML CaseRecord objects:
<few-shot examples>

Now generate 10 new CaseRecord objects that follow the same schema.
Requirements:
- Keep input and expected_output consistent.
- Only include causal-chain transactions in attempt_transaction_ids.
- Include a mix of laundering, false positive, false negative, and normal cases.
- Vary trigger_label difficulty.
- Return JSONL-ready objects only.
```

Important: treat the LLM as a drafting assistant, not the source of truth. Every generated case should still be checked against the underlying transactions or patterns file before it becomes part of the benchmark.

### Option D: Curate cases by hand

Manual curation is best for high-signal benchmark sets.

Good candidates for hand-built AML cases:

- typologies your current agent often confuses;
- benign cases that look suspicious at first glance;
- cases with incomplete but realistic evidence;
- cases where `trigger_label` is misleading or noisy;
- cases with long chains where over-flagging is common.

### Optional: add metadata for slice analysis

The Langfuse uploader in [`langfuse.py`](../../../aieng-eval-agents/aieng/agent_evals/langfuse.py) accepts optional per-record `metadata`.

That is useful for labeling slices such as:

- `difficulty: easy|medium|hard`
- `signal_regime: strong_trigger|weak_trigger|misleading_trigger`
- `window_regime: narrow|standard|noisy`
- `source: synthetic|manual|production_like`

Example:

```json
{
  "input": { "...": "..." },
  "expected_output": { "...": "..." },
  "metadata": {
    "difficulty": "hard",
    "signal_regime": "misleading_trigger",
    "window_regime": "noisy"
  }
}
```

Once those tags exist, it becomes much easier to add run-level reporting by slice instead of relying only on one global F1 score.

---

## Tips for Writing Better AML Cases

### 1. Treat `trigger_label` as a difficulty control

The generator already uses this idea:

- laundering cases often get strong labels;
- false negatives get low-signal review labels;
- false positives get alarming labels for benign activity.

If you want to test robustness, add:

- more low-signal labels;
- labels from your real alert taxonomy instead of typology names.

### 2. Keep `attempt_transaction_ids` minimal

The current item grader compares set overlap between:

- `expected_output.attempt_transaction_ids`
- `output.flagged_transaction_ids`

Only include the true causal chain. If you dump every ambient transaction in the window into ground truth, both `id_precision_like` and `id_coverage` become noisy.

### 3. Vary the window width intentionally

`window_start` is part of the task, not just metadata. Small changes in window width can make the same case:

- easy, by exposing the full chain;
- hard, by burying the chain in benign traffic;
- unfair, by hiding crucial earlier hops.

Create separate slices for narrow, medium, and noisy windows.

### 4. Cover every typology you care about

Run-level pattern scoring is macro F1, so missing typologies matter. If you add a new pattern label, you should also update:

- [`LaunderingPattern`](../../../aieng-eval-agents/aieng/agent_evals/aml_investigation/data/cases.py);
- the agent instructions in [`agent.py`](../../../aieng-eval-agents/aieng/agent_evals/aml_investigation/agent.py);
- any deterministic graders that assume the existing label set;
- your dataset distribution.

---

## How to Add New Evaluation Dimensions

There are three main ways to extend evaluation.

### 1. Add a new deterministic item-level grader

Use this when you can compute the score directly from `input`, `output`, and `expected_output`.

Good AML-specific examples:

- whether the seed transaction is always included when laundering is predicted;
- whether the agent defaults to `NONE` when evidence is weak;
- whether the predicted chain preserves temporal order;
- whether the agent over-relies on `trigger_label`;
- whether the narrative mentions the same typology as the structured field.

Minimal pattern:

```python
from aieng.agent_evals.evaluation import Evaluation


def seed_transaction_flagged_grader(*, input, output, expected_output, metadata=None, **kwargs):
    del expected_output, metadata, kwargs

    predicted_is_laundering = output.get("is_laundering")
    predicted_ids = {
        token.strip()
        for token in str(output.get("flagged_transaction_ids", "")).split(",")
        if token.strip()
    }
    seed_id = input.get("seed_transaction_id")

    applicable = predicted_is_laundering is True
    passed = (seed_id in predicted_ids) if applicable else True

    return [
        Evaluation(
            name="seed_transaction_flagged",
            value=1.0 if passed else 0.0,
            metadata={"applicable": applicable, "seed_transaction_id": seed_id},
        )
    ]
```

Then register it in [`evaluate.py`](../evaluate.py) by adding it to `evaluators=[...]`.

### 2. Add a new LLM-judge rubric

Use this when the metric depends on judgment rather than exact matching.

Good candidates:

- quality of benign-explanation analysis;
- clarity of escalation rationale;
- whether the narrative distinguishes evidence from speculation;
- whether the agent explains why alternative typologies were rejected.

Create a new rubric markdown file under [`implementations/aml_investigation/rubrics`](../rubrics), then wire it in with:

```python
custom_evaluator = create_llm_as_judge_evaluator(
    name="benign_hypothesis_quality",
    rubric_markdown="implementations/aml_investigation/rubrics/benign_hypothesis_quality.md",
)
```

### 3. Add a new trace-level evaluator

Use trace evaluators when you care about investigation process, not just the final answer.

Good candidates:

- number of unique accounts investigated before concluding;
- whether the agent started with schema/aggregate queries before raw detail pulls;
- whether it queried both sides of the seed transaction;
- whether it repeatedly pulled large raw result sets instead of narrowing with aggregates.

You can either:

- extend [`trace.py`](../../../aieng-eval-agents/aieng/agent_evals/aml_investigation/graders/trace.py), or
- create a separate trace grader file and add it to `trace_evaluators=[...]` in [`evaluate.py`](../evaluate.py).

---

## Recommended New Metrics for This Agent

If you want the next most valuable additions, these are the best candidates:

| Metric | Why it is useful | Best level |
| --- | --- | --- |
| `seed_transaction_flagged` | Catches cases where the returned chain does not anchor to the case trigger | Item |
| `typology_narrative_consistency` | Detects disagreement between structured fields and prose | Item |
| `benign_hypothesis_quality` | Forces the agent to justify why activity is benign or suspicious | LLM judge |
| `account_coverage` | Measures whether the agent inspected both inbound and outbound sides of critical accounts | Trace |
| `aggregate_first_discipline` | Rewards starting with summaries instead of dumping raw transactions | Trace |
| `trigger_label_dependence` | Measures whether strong labels are inflating performance | Run / slice |
| `performance_by_pattern_type` | Makes blind spots obvious without reading the full confusion matrix | Run |
| `performance_by_window_width` | Shows when the agent breaks under noise or limited context | Run |

The highest-leverage improvement is usually **slice-based reporting**. Even if overall F1 looks good, the agent may still fail badly on:

- low-signal cases;
- benign-but-anomalous cases;
- long-hop stack/layering cases;
- rare typologies.

---

## How to Add a New Output Field

If you want to evaluate something the current output schema does not expose, add it explicitly.

Examples:

- `confidence`
- `benign_explanation`
- `missing_information`
- `investigated_accounts`
- `evidence_transaction_ids`

When you add a new output field, update these pieces together:

1. [`AnalystOutput`](../../../aieng-eval-agents/aieng/agent_evals/aml_investigation/data/cases.py)
2. the output instructions in [`agent.py`](../../../aieng-eval-agents/aieng/agent_evals/aml_investigation/agent.py)
3. any dataset rows or ground-truth fields needed to score it
4. [`task.py`](../../../aieng-eval-agents/aieng/agent_evals/aml_investigation/task.py), if parsing assumptions change
5. your grader(s) in [`item.py`](../../../aieng-eval-agents/aieng/agent_evals/aml_investigation/graders/item.py) or new evaluator files

As a rule: **do not try to infer important evaluation fields back out of free text if you can ask the agent to emit them structurally instead**.

---

## How to Extend the AML Agent Itself

The main agent factory is [`create_aml_investigation_agent()`](../../../aieng-eval-agents/aieng/agent_evals/aml_investigation/agent.py). Right now it uses:

- a read-only schema tool;
- a read-only SQL execution tool;
- a strict `AnalystOutput` response schema.

### 1. Add higher-level domain tools

The current agent can do everything with SQL, but not always efficiently. Good additions would be:

- `get_seed_transaction_context(seed_transaction_id, window_start, seed_timestamp)`
- `summarize_account_activity(account_id, window_start, seed_timestamp)`
- `find_counterparty_clusters(account_id, window_start, seed_timestamp)`
- `trace_fund_path(seed_transaction_id, max_hops)`
- `compute_account_features(account_id, window_start, seed_timestamp)`

These tools reduce prompt load, standardize analysis, and make trace evaluation easier because the intended workflow becomes more explicit.

### 2. Add graph-oriented tooling

AML patterns are graph problems. A dedicated tool that returns:

- inbound degree;
- outbound degree;
- repeating counterparties;
- connected components;
- short path summaries;
- cycle candidates;

would likely improve typology classification more than prompt tuning alone.

### 3. Add a code interpreter

This is optional, but it can be useful for:

- multi-step feature calculations;
- clustering or graph analysis over query results;
- validating candidate chains before writing the final structured output.

If you add it, pair it with trace evaluation so you can measure whether it actually improves difficult cases instead of just increasing tool churn.

A relatively easy way to add a code interpreter is to use the `aieng-agents` package:

```bash
pip install aieng-agents
```

Then create a new agent factory that extends the existing one with code-interpreter tools:

```python
from aieng.agents.tools import CodeInterpreter

interpreter = CodeInterpreter()
# add interpreter to the agent's toolset
```

### 4. Add account/entity enrichment

The current database has `accounts` and an `account_transactions` view in [`schema.ddl`](schema.ddl), but the agent could benefit from richer context such as:

- jurisdiction or bank risk tiers;
- business/entity categories;
- known customer profile expectations;
- payment-channel risk metadata.

If you add enrichment, also add evaluation slices to test whether the agent uses it correctly instead of just becoming more verbose.

### 5. Tighten the prompt around investigation discipline

The existing prompt already says “start with aggregates” and “default benign unless evidence contradicts this.” You can strengthen it further by requiring:

- an explicit benign hypothesis section;
- a hard query budget;
- a requirement to explain why the seed transaction belongs in the final chain;
- a requirement to cite the specific observations behind the verdict.

### 6. Make the output more audit-friendly

Consider extending the output schema with:

- `decision_rationale`
- `key_evidence`
- `alternative_explanations_considered`
- `recommended_follow_up`
- `confidence`

Those fields make both human review and automated scoring easier.

---

## A Good Pattern for Co-Evolving Agent and Evaluation

The good workflow is:

1. add a new structured output field or tool;
2. add a deterministic or rubric-based metric that specifically checks whether the new capability is used well;
3. add dataset slices where that capability matters;
4. compare overall metrics and slice metrics before keeping the change.

Examples:

- If you add a graph tool, add trace metrics for whether it was used and whether pattern accuracy improves on multi-hop cases.
- If you add a `confidence` field, add calibration slices that compare confidence to correctness.
- If you add a benign-explanation field, add a rubric that rewards concrete benign reasoning and penalizes unsupported suspicion.

---

## Suggested First Extensions

If you want a practical roadmap, start here:

1. Add a `seed_transaction_flagged` deterministic metric.
2. Add a second rubric focused on benign-hypothesis quality.
3. Report metrics by `trigger_label` regime and `pattern_type`.
4. Add one higher-level account-summary tool so the agent relies less on repeated raw SQL.
5. Extend the dataset with more benign-but-suspicious cases to pressure test false-positive behavior.

That combination usually gives the best signal quickly: it improves both agent behavior and the quality of the benchmark you use to judge it.

---

## Recommended Validation Loop

After changing the dataset, agent, or graders, run the full loop:

```bash
uv run --env-file .env implementations/aml_investigation/cli.py \
  --input-path implementations/aml_investigation/data/aml_cases.jsonl \
  --output-path implementations/aml_investigation/data/aml_cases_with_output.jsonl

uv run --env-file .env implementations/aml_investigation/evaluate.py \
  --dataset-path implementations/aml_investigation/data/aml_cases.jsonl \
  --dataset-name AML-investigation
```

Use the first command when you want to inspect raw outputs case by case. Use the second when you want the full Langfuse-backed evaluation stack, including trace evaluators.
