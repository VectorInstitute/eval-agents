# Agentic AI Evaluation Bootcamp

----------------------------------------------------------------------------------------

[![code checks](https://github.com/VectorInstitute/eval-agents/actions/workflows/code_checks.yml/badge.svg)](https://github.com/VectorInstitute/eval-agents/actions/workflows/code_checks.yml)
[![unit tests](https://github.com/VectorInstitute/eval-agents/actions/workflows/unit_tests.yml/badge.svg)](https://github.com/VectorInstitute/eval-agents/actions/workflows/unit_tests.yml)
[![codecov](https://codecov.io/github/VectorInstitute/eval-agents/graph/badge.svg?token=83MYFZ3UPA)](https://codecov.io/github/VectorInstitute/eval-agents)
[![GitHub License](https://img.shields.io/github/license/VectorInstitute/eval-agents)](https://img.shields.io/github/license/VectorInstitute/eval-agents)

This is a collection of reference implementations for Vector Institute's **Agentic AI Evaluation Bootcamp**.

## Reference Implementations

This repository includes three modules, each demonstrating a different aspect of building and evaluating agent-based systems:

- **[Knowledge-Grounded QA Agent](implementations/knowledge_qa/README.md)**
  A ReAct agent using Google ADK and Google Search to answer questions grounded in live web content. Evaluated on the DeepSearchQA benchmark using LLM-as-a-judge metrics.

- **[AML Investigation Agent](implementations/aml_investigation/README.md)**
  An agent that investigates Anti-Money Laundering cases by querying a SQLite database of financial transactions via a read-only SQL tool. Produces structured analysis and supports batch evaluation.

- **[Report Generation Agent](implementations/report_generation/README.md)**
  An agent that accepts natural language queries and generates downloadable Excel reports from a relational database. Includes a Gradio demo UI and Langfuse-integrated evaluations.

## Getting Started

Set your API keys in `.env`. Use `.env.example` as a template.

```bash
cp -v .env.example .env
```

Run integration tests to validate that your API keys are set up correctly.

```bash
uv run --env-file .env pytest -sv tests/tool_tests/test_integration.py
```

## Running the Implementations

For "Gradio App" reference implementations, running the script would print out a "public URL" ending in `gradio.live` (might take a few seconds to appear.) To access the gradio app with the full streaming capabilities, copy and paste this `gradio.live` URL into a new browser tab.

For all reference implementations, to exit, press "Ctrl/Control-C" and wait up to ten seconds. If you are a Mac user, you should use "Control-C" and not "Command-C". Please note that by default, the gradio web app reloads automatically as you edit the Python script. There is no need to manually stop and restart the program each time you make some code changes.

You might see warning messages like the following:

```json
ERROR:openai.agents:[non-fatal] Tracing client error 401: {
  "error": {
    "message": "Incorrect API key provided. You can find your API key at https://platform.openai.com/account/api-keys.",
    "type": "invalid_request_error",
    "param": null,
    "code": "invalid_api_key"
  }
}
```

These warnings can be safely ignored, as they are the result of a bug in the upstream libraries. Your agent traces will be uploaded to LangFuse as configured.

## Requirements

- Python 3.12+
- API keys as configured in `.env`.

### Tidbit

If you're curious about what "uv" stands for, it appears to have been more or
less chosen [randomly](https://github.com/astral-sh/uv/issues/1349#issuecomment-1986451785).
