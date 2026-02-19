# Anti-Money Laundering Investigation Agent

This folder contains a teaching-focused implementation of an AML (Anti‑Money Laundering) investigation workflow:

- a small SQLite database built from a transactions dataset,
- a case generator that produces investigation “case files” (JSONL),
- an ADK agent that uses a **read-only SQL tool** to investigate each case and write an analysis.

The goal is to mirror a real analyst workflow: start from an alert/seed, pull relevant activity in a time window, and decide whether the evidence supports laundering or a benign explanation.

## Setup

1. If not already present, create a `.env` file (copy from `.env.example`):

```bash
cp .env.example .env
```

1. Set the required API key (Gemini via OpenAI-compatible endpoint is used in this repo):

```bash
GOOGLE_API_KEY="your-api-key"
```

1. Configure the AML database connection (SQLite by default):

```bash
AML_DB__DRIVER="sqlite"
AML_DB__DATABASE="implementations/aml_investigation/data/aml_transactions.db"
AML_DB__QUERY__MODE="ro"
```

1. Install dependencies:

```bash
uv sync
```

## Create the Database

The CLI downloads a dataset and builds a local SQLite database using the schema in `implementations/aml_investigation/data/schema.ddl`.

```bash
uv run --env-file .env python -m implementations.aml_investigation.data.cli create-db \
  --illicit-ratio HI \
  --transactions-size Small
```

This writes the SQLite DB at `implementations/aml_investigation/data/aml_transactions.db` by default.

## Generate Case Files (JSONL)

Cases are generated as `CaseRecord` objects (case metadata + ground truth, and later optional model analysis).

```bash
uv run --env-file .env python -m implementations.aml_investigation.data.cli create-cases \
  --illicit-ratio HI \
  --transactions-size Small \
  --output-dir implementations/aml_investigation/data
```

Output:

- `implementations/aml_investigation/data/aml_cases.jsonl`

Each case contains:

- `seed_transaction_id`: where the investigation starts,
- `window_start` and `seed_timestamp`: the time window the analyst must stay within,
- `trigger_label`: a free-form label describing why the case exists (rule alert, review sample, retrospective review, etc.).

## Run the Agent (Batch)

This reads `aml_cases.jsonl`, runs the agent over any cases missing `output`, and writes:

- `implementations/aml_investigation/data/aml_cases_with_output.jsonl`

```bash
uv run --env-file .env implementations/aml_investigation/cli.py
```

The script prints a simple confusion matrix for `is_laundering` based on the cases that have `output`.

## Evaluate the Agent

The evaluation script uploads the AML investigation dataset to Langfuse, runs a comprehensive evaluation experiment with multiple types of evaluators, and displays results in the console.

```bash
uv run --env-file .env implementations/aml_investigation/evaluate.py \
  --dataset-path implementations/aml_investigation/data/aml_cases.jsonl \
  --dataset-name AML-investigation
```

### Evaluation Levels

The evaluation framework assesses agent performance at three levels:

**Item-Level Evaluators** — Score each individual case prediction:

- **Deterministic grader**: Checks correctness of `is_laundering`, `pattern_type`, and flagged transaction IDs against ground truth
- **Narrative quality evaluator**: LLM-as-judge that scores the investigation reasoning and pattern description quality using the rubric in `rubrics/narrative_pattern_quality.md`

**Trace-Level Evaluators** — Analyze tool-use trajectories for each agent run:

- **Trace deterministic grader**: Validates SQL safety (read-only compliance), time window adherence, and query redundancy metrics
- **Trace groundedness evaluator**: LLM-based assessment of whether the agent's narrative is grounded in the actual tool outputs

**Run-Level Grader** — Aggregates results across all cases:

- Computes precision, recall, and F1-score for `is_laundering` detection
- Generates macro F1-score and confusion matrix for `pattern_type` classification

### CLI Options

Key options you may want to adjust:

- `--agent-timeout`: Timeout in seconds for each agent run (default: 300)
- `--llm-judge-timeout`: Timeout for LLM judge evaluations (default: 120)
- `--llm-judge-retries`: Retry attempts for LLM judge failures (default: 3)
- `--max-concurrent-cases`: Maximum concurrent cases to process (default: 5)
- `--max-concurrent-traces`: Maximum concurrent trace evaluations (default: 10)
- `--max-trace-wait-time`: Maximum seconds to wait for trace data (default: 300)

### Output

The evaluation displays:

1. **Per-item metrics tables**: Shows deterministic and narrative quality scores for each case
2. **Run-level aggregate metrics**: Overall precision, recall, F1-score, and confusion matrix
3. **Trace evaluation summary**: Count of successful, skipped, and failed trace evaluations

All results are uploaded to Langfuse for further analysis and visualization.

## Run with ADK Web UI

If you want to inspect the agent interactively, the module exposes a top-level `root_agent` for ADK discovery.

Run:

```bash
uv run adk web --port 8000 --reload --reload_agents implementations/
```

## Safety Notes (Why Read‑Only SQL?)

Agents' access to operational databases should be limited to prevent accidental or malicious data modification.
This repo’s SQL tool is designed to be read-only and defensive:

- it allows only a small set of statement roots (e.g., `SELECT`),
- it blocks write/DDL nodes anywhere in the parsed SQL AST,
- it limits rows returned and applies a timeout,
- it formats results as a small markdown table for LLM consumption.

For an extra layer of security, access control can be enforced at the database user/role level.

See `aieng-eval-agents/aieng/agent_evals/tools/sql_database.py`.
