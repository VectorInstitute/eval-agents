"""
Run the standalone Search Agent against a Langfuse dataset and evaluate whether
its output follows the required JSON format.

This benchmark is intentionally scoped to the Search Agent only. It does not
invoke the Orchestrator, Work Order Agent, or LLM judge.

Expected Search Agent output format:

[
  {
    "source": "...",
    "url": "...",
    "excerpt": "...",
    "confidence": 1.0
  }
]

A score is uploaded only when:
- The Search Agent returns a non-empty answer without an error.
- The JSON format evaluator returns a valid result.
- A Langfuse trace ID is available.

Scores uploaded:
- search_json_format
- search_json_format_passed
- search_json_format_latency_seconds

Usage:
python run_search_format_eval.py \
  --langfuse_dataset_name "SME Test Set - Formatted" \
  --run_name "json_format_test1" \
  --limit 1
"""

import argparse
import asyncio
import json
import os
import statistics
import time
import traceback
from dataclasses import dataclass
from typing import Any

from agents import Runner
from dotenv import load_dotenv
from langfuse import get_client
from langfuse._client.datasets import DatasetItemClient
from openinference.instrumentation.openai_agents import OpenAIAgentsInstrumentor
from pydantic import BaseModel
from rich.progress import track

from search_agent import SearchAgent
from utils.langfuse.shared_client import flush_langfuse, langfuse_client


load_dotenv()
OpenAIAgentsInstrumentor().instrument()

langfuse = get_client()

openai_api_key = os.getenv("OPENAI_API_KEY")

if not openai_api_key:
    raise RuntimeError(
        "OPENAI_API_KEY is not set. Check your .env file or shell environment."
    )


DEFAULT_FORMAT_TARGET = 0.90
DEFAULT_LATENCY_TARGET_SECONDS = 10.0

REQUIRED_KEYS = {
    "source",
    "url",
    "excerpt",
    "confidence",
}


class LangFuseTracedResponse(BaseModel):
    answer: str | None = None
    trace_id: str | None = None
    error: str | None = None
    latency_seconds: float | None = None


class SearchJsonFormatEvaluation(BaseModel):
    score: float
    passed: bool
    explanation: str
    parsed_count: int = 0
    valid_count: int = 0


@dataclass
class SearchFormatEvaluationOutcome:
    traced_response: LangFuseTracedResponse
    evaluation: SearchJsonFormatEvaluation | None
    format_target: float


def format_score(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{value:.3f}"


def percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0

    ordered = sorted(values)

    if len(ordered) == 1:
        return ordered[0]

    rank = (len(ordered) - 1) * pct
    lower_index = int(rank)
    upper_index = min(lower_index + 1, len(ordered) - 1)
    weight = rank - lower_index

    return ordered[lower_index] * (1 - weight) + ordered[upper_index] * weight


def get_current_trace_id() -> str | None:
    if langfuse_client is None:
        return None
    return langfuse_client.get_current_trace_id()


def parse_item_input(item: DatasetItemClient) -> str:
    """
    Extract the search query from a Langfuse dataset item.

    Supported input shapes:
    - {"text": "..."}
    - {"query": "..."}
    - {"question": "..."}
    - plain string
    """
    if isinstance(item.input, str):
        query = item.input.strip()
        if query:
            return query

    if isinstance(item.input, dict):
        for key in ("text", "query", "question", "input", "content"):
            value = item.input.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()

    raise ValueError(
        "Dataset item input must contain a non-empty text/query/question field. "
        f"Got: {item.input!r}"
    )


def clean_agent_output(value: Any) -> str:
    """
    Extract text from common OpenAI Agents Runner result shapes.
    """
    if value is None:
        return ""

    if isinstance(value, str):
        return value.strip()

    for attr in ("final_output", "output", "answer", "content", "text", "response"):
        if hasattr(value, attr):
            candidate = getattr(value, attr)
            if isinstance(candidate, str):
                return candidate.strip()

    return str(value).strip()


def get_search_agent_object(search_agent: SearchAgent) -> Any:
    """
    Find the actual OpenAI Agents SDK Agent object inside SearchAgent.

    Different wrappers may expose the underlying agent using different names.
    """
    candidate_attrs = (
        "agent",
        "search_agent",
        "assistant",
        "runner_agent",
        "_agent",
    )

    for attr in candidate_attrs:
        if hasattr(search_agent, attr):
            candidate = getattr(search_agent, attr)
            if candidate is not None:
                return candidate

    raise AttributeError(
        "Could not find the underlying Agents SDK Agent object on SearchAgent. "
        "Expected one of these attributes: "
        f"{', '.join(candidate_attrs)}. "
        "Open search_agent.py and check what attribute Agent(...) is assigned to."
    )


def strip_markdown_code_fence(text: str) -> str:
    """
    Remove ```json ... ``` or ``` ... ``` wrappers if the model adds them.
    """
    stripped = text.strip()

    if stripped.startswith("```"):
        lines = stripped.splitlines()

        if lines and lines[0].strip().startswith("```"):
            lines = lines[1:]

        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]

        return "\n".join(lines).strip()

    return stripped


def extract_json_candidate(text: str) -> str:
    """
    Return the JSON-looking portion of output.

    Preferred:
    - whole response is JSON

    Fallback:
    - first '[' to last ']'
    """
    cleaned = strip_markdown_code_fence(text)

    if cleaned.startswith("[") and cleaned.endswith("]"):
        return cleaned

    start = cleaned.find("[")
    end = cleaned.rfind("]")

    if start != -1 and end != -1 and end > start:
        return cleaned[start : end + 1]

    return cleaned


def evaluate_search_json_format(
    raw_output: str,
    format_target: float,
) -> SearchJsonFormatEvaluation:
    """
    Heuristic JSON schema evaluation.

    Required format:
    [
      {
        "source": string,
        "url": string,
        "excerpt": string,
        "confidence": number
      }
    ]

    Scoring:
    - invalid JSON: 0.0
    - valid JSON but root is not list: 0.0
    - []: 1.0, because SearchAgent allows empty list when no relevant info found
    - non-empty list: valid_items / total_items
    """
    if not raw_output or not raw_output.strip():
        return SearchJsonFormatEvaluation(
            score=0.0,
            passed=False,
            explanation="Search Agent returned an empty output.",
        )

    candidate = extract_json_candidate(raw_output)

    try:
        parsed = json.loads(candidate)
    except Exception as exc:
        return SearchJsonFormatEvaluation(
            score=0.0,
            passed=False,
            explanation=f"Output is not valid JSON. Parse error: {exc}",
        )

    if not isinstance(parsed, list):
        return SearchJsonFormatEvaluation(
            score=0.0,
            passed=False,
            explanation="Output is valid JSON, but the root value is not a list.",
        )

    if len(parsed) == 0:
        return SearchJsonFormatEvaluation(
            score=1.0,
            passed=True,
            explanation=(
                "Output is a valid empty JSON list []. This is acceptable when "
                "no relevant search results are found."
            ),
            parsed_count=0,
            valid_count=0,
        )

    valid_count = 0
    invalid_reasons: list[str] = []

    for index, item in enumerate(parsed):
        if not isinstance(item, dict):
            invalid_reasons.append(f"Item {index} is not an object.")
            continue

        missing_keys = REQUIRED_KEYS - set(item.keys())
        if missing_keys:
            invalid_reasons.append(
                f"Item {index} is missing keys: {sorted(missing_keys)}."
            )
            continue

        if not isinstance(item.get("source"), str):
            invalid_reasons.append(f"Item {index} field 'source' is not a string.")
            continue

        if not isinstance(item.get("url"), str):
            invalid_reasons.append(f"Item {index} field 'url' is not a string.")
            continue

        if not isinstance(item.get("excerpt"), str):
            invalid_reasons.append(f"Item {index} field 'excerpt' is not a string.")
            continue

        if not isinstance(item.get("confidence"), (int, float)):
            invalid_reasons.append(
                f"Item {index} field 'confidence' is not numeric."
            )
            continue

        valid_count += 1

    score = round(valid_count / len(parsed), 4)
    passed = score >= format_target

    if passed:
        explanation = (
            f"{valid_count}/{len(parsed)} search result objects matched the required "
            f"JSON schema. Score {score:.2f} meets target {format_target:.2f}."
        )
    else:
        issues = " ".join(invalid_reasons[:5])
        explanation = (
            f"{valid_count}/{len(parsed)} search result objects matched the required "
            f"JSON schema. Score {score:.2f} is below target {format_target:.2f}. "
            f"Issues: {issues}"
        )

    return SearchJsonFormatEvaluation(
        score=score,
        passed=passed,
        explanation=explanation,
        parsed_count=len(parsed),
        valid_count=valid_count,
    )


async def run_search_with_trace(
    search_agent: SearchAgent,
    query: str,
) -> LangFuseTracedResponse:
    """
    Run the Search Agent.

    This function assumes it is called inside item.run(...), so
    get_current_trace_id() should return the dataset-run trace ID.
    """
    start_time = time.perf_counter()

    try:
        if hasattr(search_agent, "run") and callable(getattr(search_agent, "run")):
            raw_result = await search_agent.run(query)
            answer = clean_agent_output(raw_result)
        else:
            agent_obj = get_search_agent_object(search_agent)
            raw_result = await Runner.run(
                agent_obj,
                input=query,
            )
            answer = clean_agent_output(raw_result)

        return LangFuseTracedResponse(
            answer=answer,
            trace_id=get_current_trace_id(),
            latency_seconds=time.perf_counter() - start_time,
        )

    except asyncio.CancelledError:
        raise

    except Exception as exc:
        error_message = f"{type(exc).__name__}: {exc}"

        print("\nSEARCH AGENT EXECUTION FAILED", flush=True)
        print(f"Query: {query}", flush=True)
        print(f"Error: {error_message}", flush=True)
        print(traceback.format_exc(), flush=True)

        return LangFuseTracedResponse(
            answer=None,
            trace_id=get_current_trace_id(),
            error=error_message,
            latency_seconds=time.perf_counter() - start_time,
        )


async def run_and_evaluate(
    run_name: str,
    search_agent: SearchAgent,
    item: DatasetItemClient,
    format_target: float,
) -> SearchFormatEvaluationOutcome:
    item_id = getattr(item, "id", "<unknown>")
    query = parse_item_input(item)

    print(
        f"\nRunning Search Agent JSON format evaluation for item {item_id}: "
        f"{query[:160]}",
        flush=True,
    )

    with item.run(run_name=run_name) as span:
        span.update(
            input={
                "agent_type": "search",
                "eval_type": "search_json_format",
                "query": query,
                "format_target": format_target,
            }
        )

        traced_response = await run_search_with_trace(
            search_agent=search_agent,
            query=query,
        )

        if traced_response.answer is not None:
            evaluation = evaluate_search_json_format(
                raw_output=traced_response.answer,
                format_target=format_target,
            )
        else:
            evaluation = None

        span.update(
            output={
                "answer": traced_response.answer,
                "error": traced_response.error,
                "latency_seconds": traced_response.latency_seconds,
                "json_format_score": evaluation.score if evaluation else None,
                "json_format_passed": evaluation.passed if evaluation else None,
            }
        )

    if traced_response.error is not None:
        print(f"Search Agent failed: {traced_response.error}", flush=True)
        return SearchFormatEvaluationOutcome(
            traced_response=traced_response,
            evaluation=None,
            format_target=format_target,
        )

    if traced_response.answer is None:
        print("Search Agent returned no answer; skipping format evaluation.", flush=True)
        return SearchFormatEvaluationOutcome(
            traced_response=traced_response,
            evaluation=None,
            format_target=format_target,
        )

    print("Search Agent raw output:", flush=True)
    print(traced_response.answer, flush=True)

    if evaluation is not None:
        print("JSON format evaluation summary:", flush=True)
        print(f"- item_id: {item_id}", flush=True)
        print(f"- trace_id: {traced_response.trace_id or 'n/a'}", flush=True)
        print(
            f"- latency_seconds: {format_score(traced_response.latency_seconds)}",
            flush=True,
        )
        print(f"- format_target: {format_target:.2f}", flush=True)
        print(f"- search_json_format: {format_score(evaluation.score)}", flush=True)
        print(f"- passed: {evaluation.passed}", flush=True)
        print(f"- explanation: {evaluation.explanation}", flush=True)

    return SearchFormatEvaluationOutcome(
        traced_response=traced_response,
        evaluation=evaluation,
        format_target=format_target,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run Search Agent JSON format evaluations from a Langfuse dataset."
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
        "--format_target",
        type=float,
        default=DEFAULT_FORMAT_TARGET,
        help="Pass threshold for JSON format score. Default: 0.90.",
    )

    return parser.parse_args()


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

    search_agent = SearchAgent()
    outcomes: list[SearchFormatEvaluationOutcome] = []

    for item in items:
        outcome = await run_and_evaluate(
            run_name=args.run_name,
            search_agent=search_agent,
            item=item,
            format_target=args.format_target,
        )
        outcomes.append(outcome)

    scorable_outcomes = [
        outcome
        for outcome in outcomes
        if (
            outcome.traced_response.error is None
            and outcome.traced_response.answer is not None
            and outcome.traced_response.trace_id is not None
            and outcome.evaluation is not None
        )
    ]

    skipped_count = len(outcomes) - len(scorable_outcomes)

    if skipped_count:
        print(
            f"\nSkipping score upload for {skipped_count} failed or incomplete evaluation(s).",
            flush=True,
        )

    if langfuse_client is None:
        print("Langfuse client is unavailable; skipping score upload.", flush=True)
    else:
        for outcome in track(
            scorable_outcomes,
            total=len(scorable_outcomes),
            description="Uploading scores",
        ):
            traced_response = outcome.traced_response
            evaluation = outcome.evaluation

            assert evaluation is not None
            assert traced_response.trace_id is not None

            langfuse_client.create_score(
                name="search_json_format",
                value=evaluation.score,
                comment=evaluation.explanation,
                trace_id=traced_response.trace_id,
            )

            langfuse_client.create_score(
                name="search_json_format_passed",
                value=evaluation.passed,
                comment=f"Format target: {outcome.format_target:.2f}",
                trace_id=traced_response.trace_id,
            )

            if traced_response.latency_seconds is not None:
                langfuse_client.create_score(
                    name="search_json_format_latency_seconds",
                    value=round(traced_response.latency_seconds, 4),
                    trace_id=traced_response.trace_id,
                )

                langfuse_client.create_score(
                    name="search_json_format_latency_passed",
                    value=(
                        traced_response.latency_seconds
                        <= DEFAULT_LATENCY_TARGET_SECONDS
                    ),
                    comment=(
                        f"Latency target: "
                        f"{DEFAULT_LATENCY_TARGET_SECONDS:.2f} seconds"
                    ),
                    trace_id=traced_response.trace_id,
                )

    latency_values = [
        outcome.traced_response.latency_seconds
        for outcome in outcomes
        if outcome.traced_response.latency_seconds is not None
    ]

    format_scores = [
        outcome.evaluation.score
        for outcome in outcomes
        if outcome.evaluation is not None
    ]

    passed_values = [
        outcome.evaluation.passed
        for outcome in outcomes
        if outcome.evaluation is not None
    ]

    if latency_values:
        print(
            "\nLatency summary:"
            f" avg={statistics.fmean(latency_values):.2f}s"
            f" p95={percentile(latency_values, 0.95):.2f}s",
            flush=True,
        )

    if format_scores:
        pass_rate = sum(1 for passed in passed_values if passed) / len(passed_values)

        print(
            "JSON format summary:"
            f" avg={statistics.fmean(format_scores):.3f}"
            f" pass_rate={pass_rate:.1%}",
            flush=True,
        )

    flush_langfuse()


if __name__ == "__main__":
    asyncio.run(main())