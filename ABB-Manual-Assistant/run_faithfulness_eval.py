"""
Run the ABB Manual Assistant against a Langfuse dataset and evaluate its
responses with the RAGAS Faithfulness metric.

Faithfulness (``faithfulness``): how well the agent's answer is grounded in the
context retrieved from the ABB knowledge base. Uses an LLM.

The retrieval used for the Faithfulness context reproduces the same Vertex AI
Search call the Search Agent performs internally
(``SearchAgent`` -> ``VertexSearchTool`` -> ``vertex_search``), so faithfulness
is measured against the project's real knowledge base.

A score is uploaded only when:
1. The ABB agent returns a non-empty answer without an error.
2. Retrieved context is available and the metric produced a valid numeric value.
3. A Langfuse trace ID is available.

Usage:
    python run_faithfulness_eval.py \
        --langfuse_dataset_name "SME Test Set - Formatted" \
        --run_name "Faithfulness_Eval_Run_01" \
        --limit 10
"""

import argparse
import asyncio
import csv
import os
import re
import traceback
from typing import Any

from dotenv import load_dotenv
from langfuse import get_client
from langfuse._client.datasets import DatasetItemClient
from openai import AsyncOpenAI
from rich.progress import track

from orchestrator_agent import Orchestrator
from ragas.llms import llm_factory
from ragas.metrics.collections import Faithfulness

# Reuse the agent-running helpers already validated in run_eval.py. Importing
# this module also installs the OpenAI Agents instrumentation exactly once, so
# agent runs continue to be traced in Langfuse.
from run_eval import (
    LangFuseTracedResponse,
    extract_dataset_text,
    run_agent_with_trace,
)
from utils.langfuse.shared_client import flush_langfuse, langfuse_client
from utils.tools.vertex_search import vertex_search


load_dotenv()

langfuse = get_client()

openai_api_key = os.getenv("OPENAI_API_KEY")
openai_base_url = os.getenv("OPENAI_BASE_URL")

if not openai_api_key:
    raise RuntimeError(
        "OPENAI_API_KEY is not set. Check your .env file or shell environment."
    )

ragas_client_kwargs: dict[str, Any] = {
    "api_key": openai_api_key,
}

if openai_base_url:
    ragas_client_kwargs["base_url"] = openai_base_url

# RAGAS talks to the same OpenAI-compatible endpoint as the rest of the project
# (the Gemini endpoint by default). Model names must be valid for that endpoint,
# so the default below is a Gemini model rather than an OpenAI one.
ragas_openai_client = AsyncOpenAI(**ragas_client_kwargs)

RAGAS_LLM_MODEL = os.getenv("RAGAS_LLM_MODEL", "gemini-2.5-flash")

# The RAGAS LLM default output cap is only 1024 tokens, which truncates the
# Faithfulness statement-decomposition step for long answers (raising
# instructor's IncompleteOutputException). gemini-2.5-flash is also a reasoning
# model, so hidden reasoning tokens share this budget -- give it plenty of room.
RAGAS_MAX_TOKENS = int(os.getenv("RAGAS_MAX_TOKENS", "16384"))

ragas_llm = llm_factory(
    RAGAS_LLM_MODEL,
    client=ragas_openai_client,
    max_tokens=RAGAS_MAX_TOKENS,
)

faithfulness_metric = Faithfulness(llm=ragas_llm)


async def retrieve_contexts(query: str) -> list[str]:
    """
    Retrieve grounded context from the ABB knowledge base for a query.

    The Faithfulness metric checks whether the agent's answer is grounded in
    retrieved context. We reproduce the same Vertex AI Search retrieval that the
    Search Agent uses internally so faithfulness is measured against the real
    knowledge base rather than an unrelated source.
    """
    try:
        result = await vertex_search(query)

    except Exception as exc:
        print(
            f"[RAGAS] Context retrieval failed: {type(exc).__name__}: {exc}",
            flush=True,
        )
        return []

    if result.get("status") != "success":
        print(
            f"[RAGAS] Context retrieval error: {result.get('error', 'unknown')}",
            flush=True,
        )
        return []

    summary = (result.get("summary") or "").strip()

    if not summary:
        print("[RAGAS] Retrieval returned no summary text.", flush=True)
        return []

    return [summary]


async def score_faithfulness(
    query: str,
    answer: str,
    contexts: list[str],
) -> tuple[float | None, str | None]:
    """Compute the RAGAS Faithfulness score for one answer against its context."""
    if not contexts:
        print(
            "[RAGAS] No retrieved contexts available; skipping faithfulness.",
            flush=True,
        )
        return None, None

    try:
        result = await faithfulness_metric.ascore(
            user_input=query,
            response=answer,
            retrieved_contexts=contexts,
        )

    except Exception as exc:
        print("\n[RAGAS] FAITHFULNESS FAILED", flush=True)
        print(f"Query: {query}", flush=True)
        print(f"Error: {type(exc).__name__}: {exc}", flush=True)
        print(traceback.format_exc(), flush=True)
        return None, None

    return getattr(result, "value", None), getattr(result, "reason", None)


def extract_query_or_none(item: DatasetItemClient) -> str | None:
    """
    Return the item's input text, or None if it is blank or missing.

    Blank inputs are skipped entirely so the agent is never run on an empty
    prompt and no empty input is recorded in Langfuse.
    """
    try:
        return extract_dataset_text(item.input, "input")
    except ValueError:
        return None


# The orchestrator appends the raw retrieved context inside a collapsible
# "<details>...</details>" block at the bottom of every answer. That block is not
# part of the actual answer and can be very large (it repeats full manual
# excerpts), which skews the metric and overflows the RAGAS LLM output budget
# (IncompleteOutputException). Strip it before scoring.
_DETAILS_BLOCK_RE = re.compile(r"<details\b.*?</details>", re.IGNORECASE | re.DOTALL)


def strip_search_details(answer: str) -> str:
    """Remove the appended collapsible search-results block from an answer."""
    cleaned = _DETAILS_BLOCK_RE.sub("", answer)
    # Drop any horizontal-rule separators left dangling once the block is gone.
    cleaned = re.sub(r"\n[*-]{3,}\s*$", "", cleaned.strip()).strip()
    return cleaned or answer.strip()


async def run_and_evaluate(
    run_name: str,
    orchestrator: Orchestrator,
    item: DatasetItemClient,
    query: str,
) -> tuple[LangFuseTracedResponse, float | None, str | None]:
    print(f"\nRunning query: {query}", flush=True)

    with item.run(run_name=run_name) as span:
        span.update(input=query)

        traced_response = await run_agent_with_trace(
            orchestrator=orchestrator,
            query=query,
        )

        span.update(
            output={
                "answer": traced_response.answer,
                "error": traced_response.error,
            }
        )

    # Failed agent call: no metric and no score.
    if traced_response.error is not None:
        print(f"Agent failed: {traced_response.error}", flush=True)
        return traced_response, None, None

    # Defensive check: blank answer means no metric and no score.
    if traced_response.answer is None:
        print("Agent returned no answer; skipping metric.", flush=True)
        return traced_response, None, None

    print(f"Agent response: {traced_response.answer}", flush=True)

    answer_for_scoring = strip_search_details(traced_response.answer)
    contexts = await retrieve_contexts(query)

    value, reason = await score_faithfulness(query, answer_for_scoring, contexts)

    print(f"Faithfulness: {value}", flush=True)

    return traced_response, value, reason


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run ABB Manual Assistant RAGAS faithfulness evaluations from a "
            "Langfuse dataset."
        )
    )

    parser.add_argument(
        "--langfuse_dataset_name",
        required=True,
        help="Name of the Langfuse dataset to evaluate.",
    )

    parser.add_argument(
        "--run_name",
        required=True,
        help="Label for this evaluation run.",
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional maximum number of dataset items to evaluate.",
    )

    parser.add_argument(
        "--output_file",
        default=None,
        help=(
            "Optional path to a local CSV file listing just the per-test scores. "
            "Defaults to 'faithfulness_<run_name>.csv'."
        ),
    )

    return parser.parse_args()


def _build_comment(model: str, reason: str | None) -> str:
    """Build a human-readable comment for an uploaded score."""
    base = f"RAGAS ({model})"

    if reason:
        return f"{base}: {reason}"

    return base


def _default_output_file(metric_name: str, run_name: str) -> str:
    """Build a filesystem-safe default CSV path from the run name."""
    safe_run_name = "".join(
        char if char.isalnum() or char in ("-", "_") else "_" for char in run_name
    )

    return f"{metric_name}_{safe_run_name}.csv"


def write_scores_file(
    output_file: str,
    metric_name: str,
    results: list[tuple[LangFuseTracedResponse, float | None, str | None]],
) -> None:
    """
    Write only the per-test metric numbers to a local CSV file.

    The file intentionally contains just a test number and the metric value for
    each evaluated item; prompts, answers, and other context are not written.
    """
    with open(output_file, "w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["test_number", metric_name])

        for test_number, (_traced_response, value, _reason) in enumerate(
            results,
            start=1,
        ):
            writer.writerow([test_number, "" if value is None else value])

    print(
        f"Wrote {len(results)} {metric_name} score(s) to {output_file}",
        flush=True,
    )


async def main() -> None:
    args = parse_args()

    dataset = langfuse.get_dataset(args.langfuse_dataset_name)
    items = dataset.items

    if args.limit is not None:
        items = items[: args.limit]

    if not items:
        raise RuntimeError(
            f"No items found in dataset: {args.langfuse_dataset_name!r}"
        )

    # Sequential execution keeps the shared Orchestrator free of concurrency and
    # shared-state issues, matching run_eval.py.
    orchestrator = Orchestrator()

    results: list[tuple[LangFuseTracedResponse, float | None, str | None]] = []
    skipped = 0

    for item in items:
        query = extract_query_or_none(item)

        # Never send a blank or missing input to the agent or to Langfuse.
        if query is None:
            print(
                f"Skipping dataset item {getattr(item, 'id', '<unknown>')!r}: "
                "blank or missing input.",
                flush=True,
            )
            skipped += 1
            continue

        result = await run_and_evaluate(
            run_name=args.run_name,
            orchestrator=orchestrator,
            item=item,
            query=query,
        )
        results.append(result)

    if skipped:
        print(
            f"\nSkipped {skipped} dataset item(s) with blank or missing input.",
            flush=True,
        )

    uploads = 0

    for traced_response, value, reason in track(
        results,
        total=len(results),
        description="Uploading scores",
    ):
        # A score requires a valid metric value and a Langfuse trace to attach to.
        if traced_response.trace_id is None or value is None:
            continue

        langfuse_client.create_score(
            name="faithfulness",
            value=float(value),
            comment=_build_comment(RAGAS_LLM_MODEL, reason),
            trace_id=traced_response.trace_id,
        )
        uploads += 1

    print(f"\nUploaded {uploads} faithfulness score(s).", flush=True)

    if uploads == 0:
        print(
            "No successful evaluations to score. No Langfuse scores were "
            "uploaded.",
            flush=True,
        )

    output_file = args.output_file or _default_output_file(
        "faithfulness",
        args.run_name,
    )
    write_scores_file(output_file, "faithfulness", results)

    flush_langfuse()


if __name__ == "__main__":
    asyncio.run(main())