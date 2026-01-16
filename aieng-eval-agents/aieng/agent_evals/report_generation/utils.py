"""Utility functions for the report generation agent."""

import uuid
from typing import Any

from agents import SQLiteSession, StreamEvent, stream_events
from agents.items import ToolCallOutputItem
from gradio.components.chatbot import ChatMessage, MetadataDict
from openai.types.responses import ResponseFunctionToolCall, ResponseOutputText
from openai.types.responses.response_completed_event import ResponseCompletedEvent
from openai.types.responses.response_output_message import ResponseOutputMessage
from pydantic import AliasChoices, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


def oai_agent_stream_to_gradio_messages(stream_event: StreamEvent) -> list[ChatMessage]:
    """Parse agent sdk "stream event" into a list of gr messages.

    Adds extra data for tool use to make the gradio display informative.
    """
    output: list[ChatMessage] = []

    if isinstance(stream_event, stream_events.RawResponsesStreamEvent):
        data = stream_event.data
        if isinstance(data, ResponseCompletedEvent):
            # The completed event may contain multiple output messages,
            # including tool calls and final outputs.
            # If there is at least one tool call, we mark the response as a thought.
            is_thought = len(data.response.output) > 1 and any(
                isinstance(message, ResponseFunctionToolCall)
                for message in data.response.output
            )

            for message in data.response.output:
                if isinstance(message, ResponseOutputMessage):
                    for _item in message.content:
                        if isinstance(_item, ResponseOutputText):
                            output.append(
                                ChatMessage(
                                    role="assistant",
                                    content=_item.text,
                                    metadata={
                                        "title": "🧠 Thought",
                                        "id": data.sequence_number,
                                    }
                                    if is_thought
                                    else MetadataDict(),
                                )
                            )
                elif isinstance(message, ResponseFunctionToolCall):
                    output.append(
                        ChatMessage(
                            role="assistant",
                            content=f"```\n{message.arguments}\n```",
                            metadata={
                                "title": f"🛠️ Used tool `{message.name}`",
                            },
                        )
                    )

    elif isinstance(stream_event, stream_events.RunItemStreamEvent):
        name = stream_event.name
        item = stream_event.item

        if name == "tool_output" and isinstance(item, ToolCallOutputItem):
            output.append(
                ChatMessage(
                    role="assistant",
                    content=f"```\n{item.output}\n```",
                    metadata={
                        "title": "*Tool call output*",
                        "status": "done",  # This makes it collapsed by default
                    },
                )
            )

    return output


def get_or_create_session(
    history: list[ChatMessage],
    session_state: dict[str, Any],
) -> SQLiteSession:
    """Get existing session or create a new one for conversation persistence.

    Args:
        history: The history of the conversation.
        session_state: The state of the session.

    Returns
    -------
        The session.
    """
    if len(history) == 0:
        session = SQLiteSession(session_id=str(uuid.uuid4()))
        session_state["session"] = session
    else:
        session = session_state["session"]
    return session


class Configs(BaseSettings):
    """Configuration settings loaded from environment variables.

    This class automatically loads configuration values from environment variables
    and a .env file, and provides type-safe access to all settings. It validates
    environment variables on instantiation.

    Attributes
    ----------
    openai_base_url : str
        Base URL for OpenAI-compatible API (defaults to Gemini endpoint).
    openai_api_key : str
        API key for OpenAI-compatible API (accepts OPENAI_API_KEY, GEMINI_API_KEY,
        or GOOGLE_API_KEY).
    default_planner_model : str, default='gemini-2.5-pro'
        Model name for planning tasks. This is typically a more capable and expensive
        model.
    default_worker_model : str, default='gemini-2.5-flash'
        Model name for worker tasks. This is typically a less expensive model.
    embedding_base_url : str
        Base URL for embedding API service.
    embedding_api_key : str
        API key for embedding service.
    embedding_model_name : str, default='@cf/baai/bge-m3'
        Name of the embedding model.
    weaviate_collection_name : str, default='enwiki_20250520'
        Name of the Weaviate collection to use.
    weaviate_api_key : str
        API key for Weaviate cloud instance.
    weaviate_http_host : str
        Weaviate HTTP host (must end with .weaviate.cloud).
    weaviate_grpc_host : str
        Weaviate gRPC host (must start with grpc- and end with .weaviate.cloud).
    weaviate_http_port : int, default=443
        Port for Weaviate HTTP connections.
    weaviate_grpc_port : int, default=443
        Port for Weaviate gRPC connections.
    weaviate_http_secure : bool, default=True
        Use secure HTTP connection.
    weaviate_grpc_secure : bool, default=True
        Use secure gRPC connection.
    langfuse_public_key : str
        Langfuse public key (must start with pk-lf-).
    langfuse_secret_key : str
        Langfuse secret key (must start with sk-lf-).
    langfuse_host : str, default='https://us.cloud.langfuse.com'
        Langfuse host URL.
    e2b_api_key : str or None
        Optional E2B.dev API key for code interpreter (must start with e2b_).
    default_code_interpreter_template : str or None
        Optional default template name or ID for E2B.dev code interpreter.
    web_search_base_url : str or None
        Optional base URL for web search service.
    web_search_api_key : str or None
        Optional API key for web search service.

    Examples
    --------
    >>> from src.utils.env_vars import Configs
    >>> config = Configs()
    >>> print(config.default_planner_model)
    'gemini-2.5-pro'

    Notes
    -----
    Create a .env file in your project root with the required environment
    variables. The class will automatically load and validate them.
    """

    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", env_ignore_empty=True
    )

    openai_base_url: str = "https://generativelanguage.googleapis.com/v1beta/openai/"
    openai_api_key: str = Field(
        validation_alias=AliasChoices(
            "OPENAI_API_KEY", "GEMINI_API_KEY", "GOOGLE_API_KEY"
        )
    )

    default_planner_model: str = "gemini-2.5-pro"
    default_worker_model: str = "gemini-2.5-flash"

    embedding_base_url: str
    embedding_api_key: str
    embedding_model_name: str = "@cf/baai/bge-m3"

    weaviate_collection_name: str = "enwiki_20250520"
    weaviate_api_key: str
    # ends with .weaviate.cloud, or it's "localhost"
    weaviate_http_host: str = Field(pattern=r"^.*\.weaviate\.cloud$|localhost")
    # starts with grpc- ends with .weaviate.cloud, or it's "localhost"
    weaviate_grpc_host: str = Field(pattern=r"^grpc-.*\.weaviate\.cloud$|localhost")
    weaviate_http_port: int = 443
    weaviate_grpc_port: int = 443
    weaviate_http_secure: bool = True
    weaviate_grpc_secure: bool = True

    langfuse_public_key: str = Field(pattern=r"^pk-lf-.*$")
    langfuse_secret_key: str = Field(pattern=r"^sk-lf-.*$")
    langfuse_host: str = "https://us.cloud.langfuse.com"

    # Optional E2B.dev API key for Python Code Interpreter tool
    e2b_api_key: str | None = Field(default=None, pattern=r"^e2b_.*$")
    default_code_interpreter_template: str | None = "9p6favrrqijhasgkq1tv"

    # Optional configs for web search tool
    web_search_base_url: str | None = None
    web_search_api_key: str | None = None
