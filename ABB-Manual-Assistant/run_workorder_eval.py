"""
Run the standalone Work Order Agent against a Langfuse dataset and score its
summaries with an LLM judge.

This benchmark is intentionally scoped to the Work Order Agent only. It does
not invoke the Orchestrator or Search Agent.
"""

import argparse
import asyncio
import os
import statistics
import time
import traceback
from typing import Any

from agents import Agent, OpenAIChatCompletionsModel, Runner
from dotenv import load_dotenv
from langfuse import get_client
from langfuse._client.datasets import DatasetItemClient
from openai import AsyncOpenAI
from openinference.instrumentation.openai_agents import OpenAIAgentsInstrumentor
from pydantic import BaseModel, Field
from rich.progress import track

from utils.langfuse.shared_client import flush_langfuse, langfuse_client
from workorder_agent import WorkorderAgent


load_dotenv()
OpenAIAgentsInstrumentor().instrument()

langfuse = get_client()

openai_api_key = os.getenv("OPENAI_API_KEY")
openai_base_url = os.getenv("OPENAI_BASE_URL")

if not openai_api_key:
    raise RuntimeError(
        "OPENAI_API_KEY is not set. Check your .env file or shell environment."
    )

openai_client_kwargs: dict[str, Any] = {
    "api_key": openai_api_key,
}

if openai_base_url:
    openai_client_kwargs["base_url"] = openai_base_url

async_openai_client = AsyncOpenAI(**openai_client_kwargs)


DEFAULT_LATENCY_TARGET_SECONDS = 10.0
DEFAULT_QUALITY_TARGET = 0.90
SCORE_WEIGHTS = {
    "issue_coverage": 0.35,
    "factual_accuracy": 0.20,
    "actionability": 0.15,
    "missing_information_handling": 0.15,
    "conciseness": 0.15,
}

EVALUATOR_INSTRUCTIONS = """
Evaluate how well the proposed work order summarizes the source conversation.

Score each category from 0.00 to 1.00.

Scoring guidance:
- issue_coverage: captures the core issue and important symptoms.
- factual_accuracy: preserves the technical details present in the conversation
  and does not invent unsupported facts.
- actionability: gives a technician useful next steps or clearly states what is
  already completed or planned.
- missing_information_handling: correctly identifies unknowns instead of
  pretending the conversation included them.
- conciseness: concise, readable, and formatted as an operational work order.

Return scores only from the evidence in the conversation and reference answer.
Do not reward hallucinated details.
Provide a short explanation that mentions the biggest strength and biggest gap.
""".strip()

EVALUATOR_TEMPLATE = """\
# Conversation
{conversation}

# Reference Answer
{reference_answer}

# Source Question
{source_question}

# Proposed Work Order
{proposed_workorder}
"""


class LangFuseTracedResponse(BaseModel):
    answer: str | None = None
    trace_id: str | None = None
    error: str | None = None
    latency_seconds: float | None = None


class WorkOrderEvaluatorQuery(BaseModel):
    conversation: str
    reference_answer: str
    source_question: str
    proposed_workorder: str

    def get_query(self) -> str:
        return EVALUATOR_TEMPLATE.format(**self.model_dump())


class WorkOrderEvaluatorResponse(BaseModel):
    issue_coverage: float = Field(ge=0, le=1)
    factual_accuracy: float = Field(ge=0, le=1)
    actionability: float = Field(ge=0, le=1)
    missing_information_handling: float = Field(ge=0, le=1)
    conciseness: float = Field(ge=0, le=1)
    explanation: str


class ScoredWorkOrderEvaluation(BaseModel):
    issue_coverage: float
    factual_accuracy: float
    actionability: float
    missing_information_handling: float
    conciseness: float
    overall_score: float
    passed: bool
    explanation: str


def format_score(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{value:.3f}"


def clamp_score(value: float) -> float:
    return max(0.0, min(1.0, value))


def compute_overall_score(
    evaluation: WorkOrderEvaluatorResponse,
    quality_target: float,
) -> ScoredWorkOrderEvaluation:
    weighted_score = (
        clamp_score(evaluation.issue_coverage) * SCORE_WEIGHTS["issue_coverage"]
        + clamp_score(evaluation.factual_accuracy) * SCORE_WEIGHTS["factual_accuracy"]
        + clamp_score(evaluation.actionability) * SCORE_WEIGHTS["actionability"]
        + clamp_score(evaluation.missing_information_handling)
        * SCORE_WEIGHTS["missing_information_handling"]
        + clamp_score(evaluation.conciseness) * SCORE_WEIGHTS["conciseness"]
    )
    overall_score = round(weighted_score, 4)

    return ScoredWorkOrderEvaluation(
        issue_coverage=clamp_score(evaluation.issue_coverage),
        factual_accuracy=clamp_score(evaluation.factual_accuracy),
        actionability=clamp_score(evaluation.actionability),
        missing_information_handling=clamp_score(
            evaluation.missing_information_handling
        ),
        conciseness=clamp_score(evaluation.conciseness),
        overall_score=overall_score,
        passed=overall_score >= quality_target,
        explanation=evaluation.explanation,
    )


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


def parse_item_input(item: DatasetItemClient) -> tuple[str, str, str, float]:
    if not isinstance(item.input, dict):
        raise ValueError(
            "Dataset item input must be an object. "
            f"Got: {item.input!r}"
        )

    conversation = item.input.get("conversation") or item.input.get("text")

    if not isinstance(conversation, str) or not conversation.strip():
        raise ValueError(
            "Dataset item input must contain a non-empty 'conversation' or 'text' field. "
            f"Got: {item.input!r}"
        )

    expected_output = item.expected_output

    if isinstance(expected_output, dict):
        reference_answer = str(expected_output.get("reference_answer") or expected_output.get("text") or "").strip()
        source_question = str(expected_output.get("source_question") or "").strip()
        quality_target = float(
            expected_output.get("quality_target", DEFAULT_QUALITY_TARGET)
        )
    else:
        reference_answer = str(expected_output or "").strip()
        source_question = ""
        quality_target = DEFAULT_QUALITY_TARGET

    return conversation.strip(), reference_answer, source_question, quality_target


async def run_workorder_with_trace(
    workorder_agent: WorkorderAgent,
    conversation: str,
) -> LangFuseTracedResponse:
    start_time = time.perf_counter()

    try:
        answer = await workorder_agent.run(conversation)

        return LangFuseTracedResponse(
            answer=answer,
            trace_id=get_current_trace_id(),
            latency_seconds=time.perf_counter() - start_time,
        )

    except asyncio.CancelledError:
        raise

    except Exception as exc:
        error_message = f"{type(exc).__name__}: {exc}"

        print("\nWORK ORDER AGENT EXECUTION FAILED", flush=True)
        print(f"Conversation: {conversation}", flush=True)
        print(f"Error: {error_message}", flush=True)
        print(traceback.format_exc(), flush=True)

        return LangFuseTracedResponse(
            answer=None,
            trace_id=get_current_trace_id(),
            error=error_message,
            latency_seconds=time.perf_counter() - start_time,
        )


async def run_evaluator_agent(
    evaluator_query: WorkOrderEvaluatorQuery,
) -> WorkOrderEvaluatorResponse:
    evaluator_agent = Agent(
        name="Work Order Evaluator",
        instructions=EVALUATOR_INSTRUCTIONS,
        output_type=WorkOrderEvaluatorResponse,
        model=OpenAIChatCompletionsModel(
            model="gemini-2.5-flash",
            openai_client=async_openai_client,
        ),
    )

    result = await Runner.run(
        evaluator_agent,
        input=evaluator_query.get_query(),
    )

    return result.final_output_as(WorkOrderEvaluatorResponse)


async def run_and_evaluate(
    run_name: str,
    workorder_agent: WorkorderAgent,
    item: DatasetItemClient,
) -> tuple[LangFuseTracedResponse, ScoredWorkOrderEvaluation | None, float]:
    item_id = getattr(item, "id", "<unknown>")
    conversation, reference_answer, source_question, quality_target = parse_item_input(
        item
    )

    print(
        f"\nRunning work order evaluation for item {item_id}: "
        f"{source_question or conversation[:120]}",
        flush=True,
    )

    with item.run(run_name=run_name) as span:
        span.update(
            input={
                "agent_type": "workorder",
                "conversation": conversation,
            }
        )

        traced_response = await run_workorder_with_trace(
            workorder_agent=workorder_agent,
            conversation=conversation,
        )

        span.update(
            output={
                "answer": traced_response.answer,
                "error": traced_response.error,
                "latency_seconds": traced_response.latency_seconds,
            }
        )

    if traced_response.error is not None:
        print(f"Work Order Agent failed: {traced_response.error}", flush=True)
        return traced_response, None, quality_target

    if traced_response.answer is None:
        print("Work Order Agent returned no answer; skipping evaluator.", flush=True)
        return traced_response, None, quality_target

    print(f"Reference answer:\n{reference_answer}", flush=True)
    print(f"Work Order Agent response:\n{traced_response.answer}", flush=True)

    try:
        evaluator_response = await run_evaluator_agent(
            WorkOrderEvaluatorQuery(
                conversation=conversation,
                reference_answer=reference_answer,
                source_question=source_question,
                proposed_workorder=traced_response.answer,
            )
        )
        scored_response = compute_overall_score(
            evaluation=evaluator_response,
            quality_target=quality_target,
        )

        print("Evaluation summary:", flush=True)
        print(f"- item_id: {item_id}", flush=True)
        print(f"- trace_id: {traced_response.trace_id or 'n/a'}", flush=True)
        print(
            f"- latency_seconds: {format_score(traced_response.latency_seconds)}",
            flush=True,
        )
        print(f"- quality_target: {quality_target:.2f}", flush=True)
        print(
            f"- overall_score: {format_score(scored_response.overall_score)}",
            flush=True,
        )
        print(f"- passed: {scored_response.passed}", flush=True)
        print(
            f"- issue_coverage: {format_score(scored_response.issue_coverage)}",
            flush=True,
        )
        print(
            f"- factual_accuracy: {format_score(scored_response.factual_accuracy)}",
            flush=True,
        )
        print(
            f"- actionability: {format_score(scored_response.actionability)}",
            flush=True,
        )
        print(
            "- missing_information_handling: "
            f"{format_score(scored_response.missing_information_handling)}",
            flush=True,
        )
        print(
            f"- conciseness: {format_score(scored_response.conciseness)}",
            flush=True,
        )
        print(f"- judge_explanation: {scored_response.explanation}", flush=True)
        return traced_response, scored_response, quality_target

    except Exception as exc:
        print("\nEVALUATOR EXECUTION FAILED", flush=True)
        print(f"Error: {type(exc).__name__}: {exc}", flush=True)
        print(traceback.format_exc(), flush=True)
        return traced_response, None, quality_target


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run Work Order Agent evaluations from a Langfuse dataset."
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

    workorder_agent = WorkorderAgent()
    results: list[tuple[LangFuseTracedResponse, ScoredWorkOrderEvaluation | None, float]] = []

    for item in items:
        result = await run_and_evaluate(
            run_name=args.run_name,
            workorder_agent=workorder_agent,
            item=item,
        )
        results.append(result)

    scorable_results = [
        (traced_response, evaluator_response, quality_target)
        for traced_response, evaluator_response, quality_target in results
        if (
            traced_response.error is None
            and traced_response.answer is not None
            and traced_response.trace_id is not None
            and evaluator_response is not None
        )
    ]

    skipped_count = len(results) - len(scorable_results)

    if skipped_count:
        print(
            f"\nSkipping score upload for {skipped_count} failed or incomplete evaluation(s).",
            flush=True,
        )

    if langfuse_client is None:
        print("Langfuse client is unavailable; skipping score upload.", flush=True)
    else:
        for traced_response, evaluator_response, quality_target in track(
            scorable_results,
            total=len(scorable_results),
            description="Uploading scores",
        ):
            langfuse_client.create_score(
                name="workorder_overall_score",
                value=evaluator_response.overall_score,
                comment=evaluator_response.explanation,
                trace_id=traced_response.trace_id,
            )
            langfuse_client.create_score(
                name="workorder_passed",
                value=evaluator_response.passed,
                comment=f"Quality target: {quality_target:.2f}",
                trace_id=traced_response.trace_id,
            )
            langfuse_client.create_score(
                name="workorder_issue_coverage",
                value=evaluator_response.issue_coverage,
                trace_id=traced_response.trace_id,
            )
            langfuse_client.create_score(
                name="workorder_factual_accuracy",
                value=evaluator_response.factual_accuracy,
                trace_id=traced_response.trace_id,
            )
            langfuse_client.create_score(
                name="workorder_actionability",
                value=evaluator_response.actionability,
                trace_id=traced_response.trace_id,
            )
            langfuse_client.create_score(
                name="workorder_missing_information_handling",
                value=evaluator_response.missing_information_handling,
                trace_id=traced_response.trace_id,
            )
            langfuse_client.create_score(
                name="workorder_conciseness",
                value=evaluator_response.conciseness,
                trace_id=traced_response.trace_id,
            )
            if traced_response.latency_seconds is not None:
                langfuse_client.create_score(
                    name="workorder_latency_seconds",
                    value=round(traced_response.latency_seconds, 4),
                    trace_id=traced_response.trace_id,
                )
                langfuse_client.create_score(
                    name="workorder_latency_passed",
                    value=traced_response.latency_seconds <= DEFAULT_LATENCY_TARGET_SECONDS,
                    comment=f"Latency target: {DEFAULT_LATENCY_TARGET_SECONDS:.2f} seconds",
                    trace_id=traced_response.trace_id,
                )

    latency_values = [
        traced_response.latency_seconds
        for traced_response, _, _ in results
        if traced_response.latency_seconds is not None
    ]
    quality_values = [
        evaluator_response.overall_score
        for _, evaluator_response, _ in results
        if evaluator_response is not None
    ]
    passed_values = [
        evaluator_response.passed
        for _, evaluator_response, _ in results
        if evaluator_response is not None
    ]

    if latency_values:
        print(
            "\nLatency summary:"
            f" avg={statistics.fmean(latency_values):.2f}s"
            f" p95={percentile(latency_values, 0.95):.2f}s",
            flush=True,
        )

    if quality_values:
        pass_rate = sum(1 for passed in passed_values if passed) / len(passed_values)
        print(
            "Quality summary:"
            f" avg={statistics.fmean(quality_values):.3f}"
            f" pass_rate={pass_rate:.1%}",
            flush=True,
        )
        print(
            "Use Langfuse trace totals to inspect token usage and production cost for each Work Order run.",
            flush=True,
        )

    flush_langfuse()


if __name__ == "__main__":
    asyncio.run(main())
