"""Gradio app for the Knowledge-Grounded QA Agent.

This app provides an interactive chat interface for testing the
knowledge-grounded QA agent with Google ADK and explicit Google Search tool calls.

Run with:
    uv run --env-file .env gradio implementations/knowledge_agent/gradio_app.py
"""

import asyncio
import logging
import uuid
from typing import Any, Generator

import gradio as gr
from aieng.agent_evals.knowledge_agent import (
    DeepSearchQADataset,
    KnowledgeAgentManager,
)
from dotenv import load_dotenv
from gradio.components.chatbot import ChatMessage


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def format_sources(sources: list) -> str:
    """Format sources for display."""
    if not sources:
        return ""

    lines = ["\n\n---\n**Sources:**"]
    for i, source in enumerate(sources, 1):
        if source.uri:
            lines.append(f"[{i}] [{source.title or 'Source'}]({source.uri})")
    return "\n".join(lines)


def format_tool_calls(tool_calls: list) -> str:
    """Format tool calls for display."""
    if not tool_calls:
        return ""

    lines = []
    for tc in tool_calls:
        name = tc.get("name", "unknown")
        args = tc.get("args", {})
        if "search" in name.lower():
            query = args.get("query", str(args))
            lines.append(f'Searched: "{query}"')
        else:
            lines.append(f"{name}({args})")
    return ", ".join(lines)


def chat(
    query: str,
    history: list[ChatMessage],
    session_state: dict[str, Any],
) -> Generator[list[ChatMessage], None, None]:
    """Handle chat interactions with the agent.

    Parameters
    ----------
    query : str
        The user's question.
    history : list[ChatMessage]
        The conversation history.
    session_state : dict[str, Any]
        Gradio session state for maintaining conversation context.

    Yields
    ------
    list[ChatMessage]
        Updated conversation history with agent response.
    """
    # Get or create session_id for multi-turn context (ADK handles the actual session)
    if "session_id" not in session_state:
        session_state["session_id"] = str(uuid.uuid4())
    session_id = session_state["session_id"]

    # Show thinking indicator
    thinking_message = ChatMessage(
        role="assistant",
        content="Searching the web...",
        metadata={"title": "ReAct Agent Processing"},
    )
    yield [thinking_message]

    try:
        # Get response from agent with session_id for multi-turn context
        response = asyncio.run(client_manager.agent.answer_async(query, session_id=session_id))

        # Format response with sources
        formatted_response = response.text
        if response.sources:
            formatted_response += format_sources(response.sources)

        # Create metadata showing tool calls (ReAct trace)
        metadata: dict[str, str] | None = None
        tool_info = format_tool_calls(response.tool_calls)
        if tool_info:
            metadata = {"title": f"Tools: {tool_info}"}
        elif response.search_queries:
            title = f"Searched: {', '.join(response.search_queries[:2])}"
            if len(response.search_queries) > 2:
                title += f" (+{len(response.search_queries) - 2} more)"
            metadata = {"title": title}

        # Create final message
        final_message = ChatMessage(
            role="assistant",
            content=formatted_response,
            metadata=metadata,  # type: ignore[arg-type]
        )

        yield [final_message]

    except Exception as e:
        logger.error(f"Error processing query: {e}")
        error_message = ChatMessage(
            role="assistant",
            content=f"Sorry, I encountered an error: {str(e)}",
            metadata={"title": "Error"},
        )
        yield [error_message]


def load_example_questions() -> list[list[str]]:
    """Load example questions from DeepSearchQA dataset."""
    try:
        dataset = DeepSearchQADataset()
        # Get a diverse sample of questions
        sample = dataset.sample(n=5, random_state=42)
        return [[ex.problem] for ex in sample]
    except Exception as e:
        logger.warning(f"Could not load DeepSearchQA examples: {e}")
        return [
            ["What is the current population of Tokyo?"],
            ["Who won the Nobel Prize in Physics in 2024?"],
            ["What are the latest developments in fusion energy?"],
        ]


if __name__ == "__main__":
    load_dotenv(verbose=True)

    # Initialize agent manager
    client_manager = KnowledgeAgentManager()
    logger.info(f"Using model: {client_manager.config.default_worker_model}")

    # Load example questions
    examples = load_example_questions()

    # Create Gradio interface
    demo = gr.ChatInterface(
        chat,
        chatbot=gr.Chatbot(
            height=600,
            render_markdown=True,
        ),
        textbox=gr.Textbox(
            lines=2,
            placeholder="Ask a question that requires current information...",
        ),
        examples=examples,
        additional_inputs=gr.State(value={}, render=False),
        title="Knowledge-Grounded QA Agent (ADK + Google Search)",
        description=(
            "Ask questions that require up-to-date information. "
            "This agent uses Google ADK with explicit Google Search tool calls, "
            "making the reasoning process observable. Tool calls are shown in the message header."
        ),
    )

    try:
        demo.launch(share=True)
    finally:
        client_manager.close()
