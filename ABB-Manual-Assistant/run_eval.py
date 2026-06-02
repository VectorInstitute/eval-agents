"""
This script runs the ABB Manual assistent agent on a Langfuse dataset and evaluates it;s responses using an LLM as a judge. Results are uploaded to langfuse for traceability
Include the following when you run this script:
--langfuse_dataset_name: Name of the dataset in Langfuse
--run_name: Label for this evaluation run
--limit: (Optional) Number of items to evaluate
Example
python run_eval.py --langfuse_dataset_name LLM_Judge_Errors --run_name ABB_Eval_Run_01 --limit 10
"""
import argparse
import asyncio
import os
from openai import AsyncOpenAI
from dotenv import load_dotenv
from langfuse._client.datasets import DatasetItemClient
from langfuse import get_client
from rich.progress import track

from orchestrator_agent import Orchestrator
from utils.langfuse.shared_client import langfuse_client, flush_langfuse
from utils import setup_langfuse_tracer
from agents import Agent, Runner, OpenAIChatCompletionsModel, function_tool
from pydantic import BaseModel

# --- Load environment and Langfuse ---
load_dotenv()
langfuse = get_client()
setup_langfuse_tracer()

# Load your OpenAI API key from .env or environment
openai_api_key = os.getenv("OPENAI_API_KEY")

# Create the async client
async_openai_client = AsyncOpenAI(api_key=openai_api_key)


# --- Evaluation Prompt Templates ---
EVALUATOR_INSTRUCTIONS = "Evaluate whether the 'Proposed Answer' to the given 'Question' matches the 'Ground Truth'."
EVALUATOR_TEMPLATE = """\
# Question
{question}

# Ground Truth
{ground_truth}

# Proposed Answer
{proposed_response}
"""

# --- Data Models ---
class LangFuseTracedResponse(BaseModel):
    answer: str | None
    trace_id: str | None

class EvaluatorQuery(BaseModel):
    question: str
    ground_truth: str
    proposed_response: str

    def get_query(self) -> str:
        return EVALUATOR_TEMPLATE.format(**self.model_dump())

class EvaluatorResponse(BaseModel):
    explanation: str
    is_answer_correct: bool

# --- Agent Execution ---
async def run_agent_with_trace(orchestrator: Orchestrator, query: str) -> LangFuseTracedResponse:
    try:
        result = await orchestrator.run(query)
        answer = getattr(result, "final_output", str(result))
    except Exception:
        answer = None

    return LangFuseTracedResponse(
        answer=answer,
        trace_id=langfuse_client.get_current_trace_id()
    )

# --- Evaluation Agent ---
async def run_evaluator_agent(evaluator_query: EvaluatorQuery) -> EvaluatorResponse:
    evaluator_agent = Agent(
        name="ABB Evaluator",
        instructions=EVALUATOR_INSTRUCTIONS,
        output_type=EvaluatorResponse,
        model=OpenAIChatCompletionsModel(model="gemini-2.5-flash", openai_client=async_openai_client)
    )
    result = await Runner.run(evaluator_agent, input=evaluator_query.get_query())
    return result.final_output_as(EvaluatorResponse)

# --- Main Evaluation Loop ---
async def run_and_evaluate(run_name: str, orchestrator: Orchestrator, item: DatasetItemClient):
    expected_output = item.expected_output
    assert expected_output is not None

    with item.run(run_name=run_name) as span:
        span.update(input=item.input["text"])
        traced_response = await run_agent_with_trace(orchestrator, item.input["text"])
        span.update(output=traced_response.answer)
    
    print(f"Running query: {item.input['text']}")
    print(f"Agent response: {traced_response.answer}")


    if traced_response.answer is None:
        return traced_response, None

    evaluator_response = await run_evaluator_agent(EvaluatorQuery(
        question=item.input["text"],
        ground_truth=expected_output["text"],
        proposed_response=traced_response.answer
    ))

    return traced_response, evaluator_response

# --- CLI Entrypoint ---
parser = argparse.ArgumentParser()
parser.add_argument("--langfuse_dataset_name", required=True)
parser.add_argument("--run_name", required=True)
parser.add_argument("--limit", type=int)

if __name__ == "__main__":
    args = parser.parse_args()

    items = langfuse.get_dataset(args.langfuse_dataset_name).items
    if args.limit:
        items = items[:args.limit]

    orchestrator = Orchestrator()
    coros = [run_and_evaluate(args.run_name, orchestrator, item) for item in items]
    
    async def main():
        return await asyncio.gather(*coros)

    results = asyncio.run(main())


    for traced_response, eval_output in track(results, total=len(results), description="Uploading scores"):
        if eval_output:
            langfuse_client.create_score(
                name="is_answer_correct",
                value=eval_output.is_answer_correct,
                comment=eval_output.explanation,
                trace_id=traced_response.trace_id
            )

    flush_langfuse()



    
