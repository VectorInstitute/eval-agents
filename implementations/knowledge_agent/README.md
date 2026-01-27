# Knowledge-Grounded QA Agent

This implementation demonstrates a knowledge-grounded question answering agent using **Gemini's Google Search grounding** capability, evaluated on the **DeepSearchQA** benchmark.

## Overview

The knowledge agent uses Gemini's built-in Google Search tool to answer questions that require real-time information. Unlike traditional RAG systems that rely on pre-indexed documents, this approach searches the live web to find relevant information.

## Features

- **Google Search Grounding**: Uses Gemini's native search capability for real-time information retrieval
- **Source Citation**: Automatically includes source URLs in responses
- **DeepSearchQA Evaluation**: Built-in evaluation on the DeepSearchQA benchmark (900 research tasks)
- **Multi-turn Conversations**: Session management for follow-up questions
- **Gradio Interface**: Interactive chat UI for testing

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

### Interactive Chat

Run the Gradio app:

```bash
uv run --env-file .env gradio implementations/knowledge_agent/gradio_app.py
```

### Programmatic Usage

```python
from aieng.agent_evals.knowledge_agent import KnowledgeGroundedAgent

agent = KnowledgeGroundedAgent()
response = agent.answer("What is the current population of Tokyo?")

print(response.text)
print(f"Sources: {[s.uri for s in response.sources]}")
```

### Evaluation on DeepSearchQA

```python
from aieng.agent_evals.knowledge_agent import (
    KnowledgeGroundedAgent,
    DeepSearchQADataset,
    DeepSearchQAEvaluator,
)

agent = KnowledgeGroundedAgent()
evaluator = DeepSearchQAEvaluator(agent)

# Evaluate a sample
results = evaluator.evaluate_sample(n=10, random_state=42)

# Convert to DataFrame for analysis
df = evaluator.results_to_dataframe(results)
print(df[["example_id", "ground_truth", "prediction", "sources_used"]])
```

## Notebooks

1. **01_grounding_basics.ipynb**: Introduction to Gemini's Google Search grounding
2. **02_agent_basics.ipynb**: Creating and using the knowledge agent
3. **03_multi_turn.ipynb**: Multi-turn conversations and DeepSearchQA evaluation

## Architecture

```
aieng.agent_evals.knowledge_agent/
├── config.py          # Configuration (Pydantic settings)
├── grounding_tool.py  # Gemini Google Search grounding tool
├── agent.py           # KnowledgeGroundedAgent class
├── session.py         # Multi-turn session management
├── evaluation.py      # DeepSearchQA dataset and evaluator
└── wikipedia_tool.py  # Alternative Wikipedia search tool
```

## DeepSearchQA Dataset

The [DeepSearchQA](https://www.kaggle.com/datasets/deepmind/deepsearchqa) benchmark consists of 900 "causal chain" research tasks across 17 categories. These questions require:

- Multi-source lookups
- Statistical comparisons
- Real-time web search

Example question:
> "Consider the OECD countries whose total population was composed of at least 20% of foreign-born populations as of 2023. Amongst them, which country saw their overall criminality score increase by at least +0.2 point between 2021 and 2023?"

## Models

The agent supports Gemini models with Google Search grounding:

| Model | Best For |
|-------|----------|
| `gemini-2.5-flash` (default) | Fast, cost-effective |
| `gemini-2.5-pro` | Complex reasoning |
| `gemini-3-flash-preview` | Latest capabilities (preview) |

See [Gemini models documentation](https://ai.google.dev/gemini-api/docs/models) for the full list.

## References

- [Grounding with Google Search - Gemini API](https://ai.google.dev/gemini-api/docs/google-search)
- [DeepSearchQA Dataset - Kaggle](https://www.kaggle.com/datasets/deepmind/deepsearchqa)
- [Google GenAI Python SDK](https://github.com/googleapis/python-genai)
