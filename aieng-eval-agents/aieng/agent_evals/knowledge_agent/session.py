"""Session management for multi-turn conversations.

This module provides utilities for managing conversation sessions,
maintaining history across multiple turns of a conversation.
"""

import uuid
from typing import Any

from pydantic import BaseModel


class Message(BaseModel):
    """A single message in a conversation.

    Attributes
    ----------
    role : str
        The role of the message sender ("user" or "assistant").
    content : str
        The content of the message.
    """

    role: str
    content: str


class ConversationSession:
    """Manages multi-turn conversation state.

    This class provides a simple interface for managing conversation
    sessions, including history tracking and context management.

    Attributes
    ----------
    session_id : str
        Unique identifier for this session.
    history : list[Message]
        List of conversation messages.

    Examples
    --------
    >>> session = ConversationSession()
    >>> session.add_message("user", "What is the capital of France?")
    >>> session.add_message("assistant", "The capital of France is Paris.")
    >>> print(len(session))
    2
    """

    def __init__(self, session_id: str | None = None) -> None:
        """Initialize a conversation session.

        Parameters
        ----------
        session_id : str, optional
            Unique identifier for the session. If not provided, generates a UUID.
        """
        self.session_id = session_id or str(uuid.uuid4())
        self.history: list[Message] = []
        self.metadata: dict[str, Any] = {}

    def add_message(self, role: str, content: str) -> None:
        """Add a message to the conversation history.

        Parameters
        ----------
        role : str
            The role of the message sender ("user" or "assistant").
        content : str
            The content of the message.
        """
        self.history.append(Message(role=role, content=content))

    def add_user_message(self, content: str) -> None:
        """Add a user message to the conversation.

        Parameters
        ----------
        content : str
            The user's message content.
        """
        self.add_message("user", content)

    def add_assistant_message(self, content: str) -> None:
        """Add an assistant message to the conversation.

        Parameters
        ----------
        content : str
            The assistant's message content.
        """
        self.add_message("assistant", content)

    def get_history(self) -> list[dict[str, str]]:
        """Get the full conversation history as dictionaries.

        Returns
        -------
        list[dict[str, str]]
            List of messages with role and content keys.
        """
        return [{"role": m.role, "content": m.content} for m in self.history]

    def get_history_as_text(self) -> str:
        """Get the conversation history formatted as text.

        Returns
        -------
        str
            Formatted conversation history.
        """
        lines = []
        for msg in self.history:
            prefix = "User" if msg.role == "user" else "Assistant"
            lines.append(f"{prefix}: {msg.content}")
        return "\n\n".join(lines)

    def get_last_n_messages(self, n: int) -> list[Message]:
        """Get the last n messages from the conversation.

        Parameters
        ----------
        n : int
            Number of recent messages to retrieve.

        Returns
        -------
        list[Message]
            The last n messages from the conversation.
        """
        return self.history[-n:] if len(self.history) >= n else self.history.copy()

    def clear(self) -> None:
        """Clear the conversation history and reset the session."""
        self.history = []
        self.metadata = {}
        self.session_id = str(uuid.uuid4())

    def __len__(self) -> int:
        """Return the number of messages in the conversation."""
        return len(self.history)

    def __repr__(self) -> str:
        """Return a string representation of the session."""
        return f"ConversationSession(id={self.session_id}, messages={len(self)})"


def get_or_create_session(
    session_state: dict[str, Any],
    session_key: str = "session",
) -> ConversationSession:
    """Get existing session or create a new one from Gradio state.

    This function manages conversation sessions for multi-turn interactions
    in Gradio applications.

    Parameters
    ----------
    session_state : dict[str, Any]
        The Gradio session state dictionary.
    session_key : str, optional
        The key to use for storing the session, by default "session".

    Returns
    -------
    ConversationSession
        The conversation session.

    Examples
    --------
    >>> # In a Gradio app
    >>> def chat_handler(query, history, session_state):
    ...     session = get_or_create_session(session_state)
    ...     session.add_user_message(query)
    ...     # Process and get response...
    ...     session.add_assistant_message(response)
    ...     return response
    """
    if session_key not in session_state:
        session_state[session_key] = ConversationSession()
    return session_state[session_key]
