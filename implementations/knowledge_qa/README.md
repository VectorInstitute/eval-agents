# Knowledge-Grounded QA Agent

This implementation demonstrates a knowledge-grounded question answering agent using **Google ADK** with explicit **Google Search tool calls**, evaluated on the **DeepSearchQA** benchmark.

## Overview

The knowledge agent uses a ReAct (Reasoning + Acting) architecture powered by Google ADK. It explicitly calls Google Search as a tool, making the reasoning process transparent through observable Thought → Action → Observation cycles. This approach searches the live web to find relevant information for questions requiring real-time data.

## Features

- **ReAct Architecture**: Explicit tool calls with traceable reasoning (Thought → Action → Observation)
- **Google Search Tool**: Uses ADK's `GoogleSearchTool` for real-time web search
- **Source Citation**: Automatically extracts and includes source URLs from search results
- **DeepSearchQA Evaluation**: Built-in evaluation on the DeepSearchQA benchmark (900 research tasks)
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

## Notebooks

1. **01_agent_basics.ipynb**: Agent architecture, tools, running queries, and single-sample evaluation with LLM-as-judge
2. **02_langfuse_tracing.ipynb**: Observability with Langfuse - enabling tracing, viewing traces, and loading traces via API

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

The [DeepSearchQA](https://www.kaggle.com/datasets/deepmind/deepsearchqa) benchmark consists of 900 "causal chain" research tasks across 17 categories. These questions require:

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
