# aieng-eval-agents

[![PyPI](https://img.shields.io/pypi/v/aieng-eval-agents)](https://pypi.org/project/aieng-eval-agents/)
[![Python](https://img.shields.io/badge/python-3.12%2B-blue)](https://pypi.org/project/aieng-eval-agents/)
[![License](https://img.shields.io/github/license/VectorInstitute/eval-agents)](https://github.com/VectorInstitute/eval-agents/blob/main/LICENSE)

Shared library for Vector Institute's Agentic AI Evaluation Bootcamp. Provides reusable components for building, running, and evaluating LLM agents with [Google ADK](https://google.github.io/adk-docs/) and [Langfuse](https://langfuse.com/).

## What's included

### Agent implementations

| Module | Description |
|---|---|
| `aieng.agent_evals.knowledge_qa` | ReAct agent that answers questions using live web search. Includes evaluation against the DeepSearchQA benchmark with LLM-as-a-judge metrics (precision/recall/F1). |
| `aieng.agent_evals.aml_investigation` | Agent that investigates Anti-Money Laundering cases by querying a SQLite database of financial transactions via a read-only SQL tool. |
| `aieng.agent_evals.report_generation` | Agent that generates structured Excel reports from a relational database based on natural language queries. |

### Reusable tools (`aieng.agent_evals.tools`)

- `search` — Google Search with response grounding and citations
- `web` — HTML and PDF content fetching
- `file` — Download and search data files (CSV, XLSX, JSON)
- `sql_database` — Read-only SQL database access via `ReadOnlySqlDatabase`

### Evaluation harness (`aieng.agent_evals.evaluation`)

Wrappers around Langfuse for running agent experiments:

- `run_experiment` — Run a dataset through an agent and score outputs
- `run_experiment_with_trace_evals` — Run experiments with trace-level evaluation
- `run_trace_evaluations` — Score existing Langfuse traces with LLM-based or heuristic graders

### Utilities

- `display` — Rich-based terminal and Jupyter display helpers for metrics and agent responses
- `progress` — Progress tracking for batch evaluation runs
- `configs` — Pydantic-based configuration loading from `.env`
- `langfuse` — Langfuse client and trace utilities
- `db_manager` — Database connection management

## Installation

```bash
pip install aieng-eval-agents
```

Requires Python 3.12+.

## Source

Full reference implementations and documentation are in the [eval-agents](https://github.com/VectorInstitute/eval-agents) repository.
