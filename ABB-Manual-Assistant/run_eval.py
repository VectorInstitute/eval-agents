"""
Run the ABB Manual Assistant against a Langfuse dataset and evaluate its
responses using an LLM judge.

A score is uploaded only when:
1. The ABB agent returns a non-empty answer without an error.
2. The evaluator returns a valid evaluation result.
3. A Langfuse trace ID is available.

Usage:
    python run_eval.py \
        --langfuse_dataset_name "LLM_Judge_Errors" \
        --run_name "ABB_Eval_Run_01" \
        --limit 10
"""

import argparse
import asyncio
import inspect
import os
import traceback
from typing import Any

from agents import Agent, OpenAIChatCompletionsModel, Runner
from dotenv import load_dotenv
from langfuse import get_client
from langfuse._client.datasets import DatasetItemClient
from openai import AsyncOpenAI
from orchestrator_agent import Orchestrator
from pydantic import BaseModel
from rich.progress import track
from openinference.instrumentation.openai_agents import OpenAIAgentsInstrumentor
from utils.langfuse.shared_client import flush_langfuse, langfuse_client



# Replaced the old langfuse setup
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



EVALUATOR_INSTRUCTIONS = """
Evaluate whether the Proposed Answer matches the Ground Truth for the Question.

Judge factual correctness, relevance, completeness, and safety where applicable.
Return a concise explanation and a boolean correctness result.
""".strip()

EVALUATOR_TEMPLATE = """\
# Question
{question}

# Ground Truth
{ground_truth}

# Proposed Answer
{proposed_response}
"""


class LangFuseTracedResponse(BaseModel):
    answer: str | None = None
    trace_id: str | None = None
    error: str | None = None


class EvaluatorQuery(BaseModel):
    question: str
    ground_truth: str
    proposed_response: str

    def get_query(self) -> str:
        return EVALUATOR_TEMPLATE.format(**self.model_dump())


class EvaluatorResponse(BaseModel):
    explanation: str
    is_answer_correct: bool


def extract_dataset_text(value: Any, field_label: str) -> str:
    if isinstance(value, str):
        if value.strip():
            return value

        raise ValueError(f"Dataset item {field_label} must not be blank.")

    if isinstance(value, dict) and "text" in value:
        text = value["text"]

        if isinstance(text, str) and text.strip():
            return text

    raise ValueError(
        f"Dataset item {field_label} must be a non-empty string or an object "
        f"containing a non-empty 'text' field. Got: {value!r}"
    )



ANSWER_FIELD_NAMES = (
    "final_output",
    "output",
    "answer",
    "content",
    "text",
    "response",
)


def compact_repr(value: Any, max_length: int = 1200) -> str:
    """Return a bounded repr for readable debug logs."""
    value_repr = repr(value)

    if len(value_repr) <= max_length:
        return value_repr

    return f"{value_repr[:max_length]}... <truncated>"


def clean_text(value: Any) -> str | None:
    """
    Return text exactly as emitted by a stream chunk.

    Do not strip individual chunks: leading spaces and newlines are meaningful
    in token streaming and must survive until the final assembled answer.
    """
    if not isinstance(value, str):
        return None

    return value if value.strip() else None


def extract_agent_answer(value: Any, depth: int = 0) -> str | None:
    """
    Attempt to extract actual response text from common agent result shapes.

    This supports:
    - Plain strings
    - Dicts with output / answer / content fields
    - Objects with final_output / output / answer attributes
    - OpenAI-style chunk objects with choices[0].delta.content
    - Tuple yields such as (answer_chunk, history)
    """
    if value is None or depth > 5:
        return None

    direct_text = clean_text(value)

    if direct_text is not None:
        return direct_text

    # Some streaming generators yield (answer_or_chunk, history).
    # Only inspect the first tuple element. Do not inspect arbitrary history.
    if isinstance(value, tuple):
        if not value:
            return None

        return extract_agent_answer(value[0], depth + 1)

    if isinstance(value, dict):
        for field_name in ANSWER_FIELD_NAMES:
            if field_name not in value:
                continue

            extracted = extract_agent_answer(value[field_name], depth + 1)

            if extracted is not None:
                return extracted

        choices = value.get("choices")

        if isinstance(choices, (list, tuple)) and choices:
            return extract_agent_answer(choices[0], depth + 1)

        delta = value.get("delta")

        if delta is not None:
            return extract_agent_answer(delta, depth + 1)

        message = value.get("message")

        if message is not None:
            return extract_agent_answer(message, depth + 1)

        return None

    for field_name in ANSWER_FIELD_NAMES:
        attribute_value = getattr(value, field_name, None)

        if attribute_value is None:
            continue

        extracted = extract_agent_answer(attribute_value, depth + 1)

        if extracted is not None:
            return extracted

    choices = getattr(value, "choices", None)

    if isinstance(choices, (list, tuple)) and choices:
        extracted = extract_agent_answer(choices[0], depth + 1)

        if extracted is not None:
            return extracted

    delta = getattr(value, "delta", None)

    if delta is not None:
        extracted = extract_agent_answer(delta, depth + 1)

        if extracted is not None:
            return extracted

    message = getattr(value, "message", None)

    if message is not None:
        extracted = extract_agent_answer(message, depth + 1)

        if extracted is not None:
            return extracted

    return None


def merge_stream_text(current_text: str, incoming_text: str) -> str:
    """
    Combine stream chunks without removing meaningful whitespace.

    Supports normal delta chunks and protects against a duplicate/final
    cumulative snapshot being yielded by some streaming implementations.
    """
    if not current_text:
        return incoming_text

    if incoming_text == current_text:
        return current_text

    if incoming_text.startswith(current_text):
        return incoming_text

    if current_text.startswith(incoming_text):
        return current_text

    return current_text + incoming_text

async def consume_orchestrator_stream(stream: Any) -> str:
    """
    Consume an async generator returned by Orchestrator.run() and build its
    final response text from yielded chunks or final response objects.
    """
    accumulated_answer = ""
    event_count = 0
    first_event_repr: str | None = None
    last_event_repr: str | None = None

    async for event in stream:
        event_count += 1

        event_repr = compact_repr(event)

        if first_event_repr is None:
            first_event_repr = event_repr

        last_event_repr = event_repr


        event_text = extract_agent_answer(event)

        if event_text is not None:
            accumulated_answer = merge_stream_text(
                accumulated_answer,
                event_text,
            )

    if event_count == 0:
        raise RuntimeError(
            "Orchestrator.run() completed without yielding any stream events."
        )

    accumulated_answer = accumulated_answer.strip()

    if not accumulated_answer:
        raise RuntimeError(
            "Orchestrator.run() yielded stream events but no answer text could "
            "be extracted. "
            f"First event: {first_event_repr}. "
            f"Last event: {last_event_repr}."
        )

    print(
        f"Consumed {event_count} orchestrator stream event(s).",
        flush=True,
    )

    return accumulated_answer



async def run_agent_with_trace(
    orchestrator: Orchestrator,
    query: str,
) -> LangFuseTracedResponse:
    """
    Run one independent dataset question through the ABB orchestrator.

    history=[] is intentional: each Langfuse dataset item is treated as an
    independent evaluation rather than sharing a conversation across rows.
    """
    try:
        run_result = orchestrator.run(query, history=[])

        if hasattr(run_result, "__aiter__"):
            answer = await consume_orchestrator_stream(run_result)

        elif inspect.isawaitable(run_result):
            resolved_result = await run_result

            print(
                f"Raw orchestrator result type: "
                f"{type(resolved_result).__name__}",
                flush=True,
            )
            print(
                f"Raw orchestrator result repr: "
                f"{compact_repr(resolved_result)}",
                flush=True,
            )

            answer = extract_agent_answer(resolved_result)

            if answer is None:
                raise RuntimeError(
                    "Could not extract answer text from the non-streaming "
                    "Orchestrator.run() result. "
                    f"Result: {compact_repr(resolved_result)}"
                )

        else:
            print(
                f"Raw synchronous orchestrator result type: "
                f"{type(run_result).__name__}",
                flush=True,
            )
            print(
                f"Raw synchronous orchestrator result repr: "
                f"{compact_repr(run_result)}",
                flush=True,
            )

            answer = extract_agent_answer(run_result)

            if answer is None:
                raise RuntimeError(
                    "Could not extract answer text from the synchronous "
                    "Orchestrator.run() result. "
                    f"Result: {compact_repr(run_result)}"
                )

        return LangFuseTracedResponse(
            answer=answer,
            trace_id=langfuse_client.get_current_trace_id(),
        )

    except asyncio.CancelledError:
        raise

    except Exception as exc:
        error_message = f"{type(exc).__name__}: {exc}"

        print("\nAGENT EXECUTION FAILED", flush=True)
        print(f"Query: {query}", flush=True)
        print(f"Error: {error_message}", flush=True)
        print(traceback.format_exc(), flush=True)

        return LangFuseTracedResponse(
            answer=None,
            trace_id=langfuse_client.get_current_trace_id(),
            error=error_message,
        )



async def run_evaluator_agent(
    evaluator_query: EvaluatorQuery,
) -> EvaluatorResponse:
    evaluator_agent = Agent(
        name="ABB Evaluator",
        instructions=EVALUATOR_INSTRUCTIONS,
        output_type=EvaluatorResponse,
        model=OpenAIChatCompletionsModel(
            model="gemini-2.5-flash",
            openai_client=async_openai_client,
        ),
    )

    result = await Runner.run(
        evaluator_agent,
        input=evaluator_query.get_query(),
    )

    return result.final_output_as(EvaluatorResponse)



async def run_and_evaluate(
    run_name: str,
    orchestrator: Orchestrator,
    item: DatasetItemClient,
) -> tuple[LangFuseTracedResponse, EvaluatorResponse | None]:
    query = extract_dataset_text(item.input, "input")
    ground_truth = extract_dataset_text(item.expected_output, "expected_output")

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

    # Failed agent call: do not ask the evaluator and do not upload a score.
    if traced_response.error is not None:
        print(f"Agent failed: {traced_response.error}", flush=True)
        return traced_response, None

    # Defensive check: blank answer means no evaluator and no score.
    if traced_response.answer is None:
        print("Agent returned no answer; skipping evaluator.", flush=True)
        return traced_response, None

    print(f"Agent response: {traced_response.answer}", flush=True)

    try:
        evaluator_response = await run_evaluator_agent(
            EvaluatorQuery(
                question=query,
                ground_truth=ground_truth,
                proposed_response=traced_response.answer,
            )
        )

        return traced_response, evaluator_response

    except Exception as exc:
        print("\nEVALUATOR EXECUTION FAILED", flush=True)
        print(f"Query: {query}", flush=True)
        print(f"Error: {type(exc).__name__}: {exc}", flush=True)
        print(traceback.format_exc(), flush=True)

        # Evaluator failure: do not upload a score.
        return traced_response, None



def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run ABB Manual Assistant evaluations from a Langfuse dataset."
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
        items = items[:args.limit]

    if not items:
        raise RuntimeError(
            f"No items found in dataset: {args.langfuse_dataset_name!r}"
        )

    # Sequential execution is deliberate while validating stream behavior and
    # avoids shared-Orchestrator concurrency/state issues.
    orchestrator = Orchestrator()

    results: list[tuple[LangFuseTracedResponse, EvaluatorResponse | None]] = []

    for item in items:
        result = await run_and_evaluate(
            run_name=args.run_name,
            orchestrator=orchestrator,
            item=item,
        )
        results.append(result)

    # A score is uploaded only for a fully successful evaluation.
    scorable_results = [
        (traced_response, evaluator_response)
        for traced_response, evaluator_response in results
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
            f"\nSkipping score upload for {skipped_count} failed or incomplete "
            "evaluation(s).",
            flush=True,
        )

    if not scorable_results:
        print(
            "No successful evaluations to score. No Langfuse scores will be "
            "uploaded.",
            flush=True,
        )
    else:
        for traced_response, evaluator_response in track(
            scorable_results,
            total=len(scorable_results),
            description="Uploading scores",
        ):
            langfuse_client.create_score(
                name="is_answer_correct",
                value=evaluator_response.is_answer_correct,
                comment=evaluator_response.explanation,
                trace_id=traced_response.trace_id,
            )

    flush_langfuse()


if __name__ == "__main__":
    asyncio.run(main())