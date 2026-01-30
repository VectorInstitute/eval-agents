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

This reads `aml_cases.jsonl`, runs the agent over any cases missing `analysis`, and writes:

- `implementations/aml_investigation/data/aml_cases_with_analysis.jsonl`

```bash
uv run --env-file .env implementations/aml_investigation/agent.py
```

The script prints a simple confusion matrix for `is_laundering` based on the cases that have `analysis`.

## Run with ADK Web UI

If you want to inspect the agent interactively, the module exposes a top-level `root_agent` for ADK discovery.

From `implementations/aml_investigation/`:

```bash
uv run adk web --port 8000 --reload --reload_agents implementations/
```

The DB tool is initialized lazily when a tool call happens (so importing the module doesn’t keep a DB connection open).

## Safety Notes (Why Read‑Only SQL?)

Agents should not be allowed to write to operational databases. This repo’s SQL tool is designed to be read-only and defensive:

- it allows only a small set of statement roots (e.g., `SELECT`),
- it blocks write/DDL nodes anywhere in the parsed SQL AST,
- it limits rows returned and applies a timeout,
- it formats results as a small markdown table for LLM consumption.

See `aieng-eval-agents/aieng/agent_evals/tools/sql_database.py`.
