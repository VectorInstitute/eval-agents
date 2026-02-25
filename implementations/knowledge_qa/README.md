# Knowledge-Grounded QA Agent

This implementation demonstrates a knowledge-grounded question answering agent using **Google ADK** with a **PlanReAct** architecture, evaluated on the **DeepSearchQA** benchmark.

## Overview

The agent combines two patterns: **PlanReAct** (creates an explicit numbered research plan before executing) and a **ReAct loop** within each step (Thought → Tool Call → Observation). It searches the live web to find and verify facts rather than relying on training data.

## Features

- **PlanReAct Architecture**: Explicit research plan with step statuses, revised mid-run if needed
- **Five Tools**: `google_search`, `web_fetch`, `fetch_file`, `grep_file`, `read_file`
- **Source Citation**: Extracts and cites source URLs from search results
- **DeepSearchQA Evaluation**: LLM-as-judge evaluation on the DeepSearchQA benchmark (896 questions)
- **Multi-turn Conversations**: Session management via ADK's `InMemorySessionService`

## Setup

1. **Configure environment variables** in `.env`:

```bash
# Required: Google API key (get from https://aistudio.google.com/apikey)
GOOGLE_API_KEY="your-api-key"

# Optional: Langfuse for tracing
LANGFUSE_PUBLIC_KEY="pk-lf-..."
LANGFUSE_SECRET_KEY="sk-lf-..."
```

2. **Install dependencies**:

```bash
uv sync
```

## Usage

### Programmatic Usage

```python
from aieng.agent_evals.knowledge_qa import KnowledgeGroundedAgent

agent = KnowledgeGroundedAgent()

# In async context (Jupyter notebooks, async functions)
response = await agent.answer_async("What is the current population of Tokyo?")

# In sync context (scripts)
response = agent.answer("What is the current population of Tokyo?")

print(response.text)
print(f"Sources: {[s.uri for s in response.sources]}")
print(f"Tool calls: {response.tool_calls}")
```

### Evaluation on DeepSearchQA

Use the main evaluation script to run comprehensive evaluations:

```bash
# Run evaluation on 3 samples
python implementations/knowledge_qa/evaluate.py --samples 3

# Run with specific example IDs
python implementations/knowledge_qa/evaluate.py --ids 123 456 789

# Enable trace groundedness evaluation
ENABLE_TRACE_GROUNDEDNESS=true python implementations/knowledge_qa/evaluate.py
```

Or use the CLI:

```bash
# Run evaluation via CLI
uv run --env-file .env knowledge-qa eval --samples 3
uv run --env-file .env knowledge-qa eval --ids 123 456 --show-plan
```

## Run with ADK Web UI

To inspect the agent interactively, the module exposes a top-level `root_agent` for ADK discovery.

```bash
uv run adk web --port 8000 --reload --reload_agents implementations/
```

## Notebooks

1. **01_dataset_and_tools.ipynb**: The DeepSearchQA dataset and the agent's five tools
2. **02_running_the_agent.ipynb**: PlanReAct architecture, live progress display, multi-turn conversations, and Langfuse tracing
3. **03_evaluation.ipynb**: Systematic evaluation with `run_experiment`, LLM-as-judge grading, and result inspection

## Architecture

```
aieng.agent_evals.knowledge_qa/
├── agent.py                # KnowledgeGroundedAgent (ADK Agent + Runner)
├── data/                   # DeepSearchQA dataset loader
├── deepsearchqa_grader.py  # LLM-as-judge evaluation
├── planner.py              # Research planning
├── token_tracker.py        # Token usage tracking
└── cli.py                  # Rich CLI interface

aieng.agent_evals/
├── configs.py              # Configuration (Pydantic settings)
├── evaluation/             # Evaluation harness
│   ├── experiment.py       # Langfuse experiment runner
│   └── graders/            # Evaluators (trace groundedness, etc.)
└── tools/                  # Shared tools
    ├── search.py           # GoogleSearchTool wrapper
    ├── web.py              # web_fetch for HTML/PDF
    └── file.py             # fetch_file, grep_file, read_file
```

## DeepSearchQA Dataset

The [DeepSearchQA](https://www.kaggle.com/datasets/deepmind/deepsearchqa) benchmark consists of 896 "causal chain" research tasks across 17 categories. These questions require:

- Multi-source lookups
- Statistical comparisons
- Real-time web search

Example question:
> "Consider the OECD countries whose total population was composed of at least 20% of foreign-born populations as of 2023. Amongst them, which country saw their overall criminality score increase by at least +0.2 point between 2021 and 2023?"

## Models

The agent supports Gemini models via Google ADK:

| Model | Best For |
|-------|----------|
| `gemini-2.5-flash` (default) | Fast, cost-effective |
| `gemini-2.5-pro` | Complex reasoning |

See [Gemini models documentation](https://ai.google.dev/gemini-api/docs/models) for the full list.

## References

- [Google ADK (Agent Development Kit)](https://google.github.io/adk-docs/)
- [DeepSearchQA Dataset - Kaggle](https://www.kaggle.com/datasets/deepmind/deepsearchqa)
- [Google GenAI Python SDK](https://github.com/googleapis/python-genai)
