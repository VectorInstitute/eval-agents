"""Tests for session management utilities."""

from aieng.agent_evals.knowledge_agent.session import (
    ConversationSession,
    Message,
    get_or_create_session,
)


class TestMessage:
    """Tests for the Message class."""

    def test_message_creation(self):
        """Test creating a message."""
        msg = Message(role="user", content="Hello")
        assert msg.role == "user"
        assert msg.content == "Hello"

    def test_message_roles(self):
        """Test different message roles."""
        user_msg = Message(role="user", content="Hi")
        assistant_msg = Message(role="assistant", content="Hello!")

        assert user_msg.role == "user"
        assert assistant_msg.role == "assistant"


class TestConversationSession:
    """Tests for the ConversationSession class."""

    def test_session_creation(self):
        """Test creating a new session."""
        session = ConversationSession()
        assert session.session_id is not None
        assert len(session) == 0
        assert session.history == []

    def test_session_with_custom_id(self):
        """Test creating a session with a custom ID."""
        session = ConversationSession(session_id="test-123")
        assert session.session_id == "test-123"

    def test_add_message(self):
        """Test adding messages to a session."""
        session = ConversationSession()
        session.add_message("user", "Hello")
        session.add_message("assistant", "Hi there!")

        assert len(session) == 2
        assert session.history[0].role == "user"
        assert session.history[0].content == "Hello"
        assert session.history[1].role == "assistant"
        assert session.history[1].content == "Hi there!"

    def test_add_user_message(self):
        """Test adding a user message."""
        session = ConversationSession()
        session.add_user_message("What is AI?")

        assert len(session) == 1
        assert session.history[0].role == "user"
        assert session.history[0].content == "What is AI?"

    def test_add_assistant_message(self):
        """Test adding an assistant message."""
        session = ConversationSession()
        session.add_assistant_message("AI stands for Artificial Intelligence.")

        assert len(session) == 1
        assert session.history[0].role == "assistant"

    def test_get_history(self):
        """Test getting history as dictionaries."""
        session = ConversationSession()
        session.add_user_message("Question?")
        session.add_assistant_message("Answer!")

        history = session.get_history()
        assert len(history) == 2
        assert history[0] == {"role": "user", "content": "Question?"}
        assert history[1] == {"role": "assistant", "content": "Answer!"}

    def test_get_history_as_text(self):
        """Test getting history as formatted text."""
        session = ConversationSession()
        session.add_user_message("Hello")
        session.add_assistant_message("Hi!")

        text = session.get_history_as_text()
        assert "User: Hello" in text
        assert "Assistant: Hi!" in text

    def test_get_last_n_messages(self):
        """Test getting last n messages."""
        session = ConversationSession()
        for i in range(5):
            session.add_user_message(f"Message {i}")

        last_2 = session.get_last_n_messages(2)
        assert len(last_2) == 2
        assert last_2[0].content == "Message 3"
        assert last_2[1].content == "Message 4"

    def test_get_last_n_messages_when_fewer_messages(self):
        """Test getting more messages than available."""
        session = ConversationSession()
        session.add_user_message("Only message")

        last_5 = session.get_last_n_messages(5)
        assert len(last_5) == 1

    def test_clear(self):
        """Test clearing a session."""
        session = ConversationSession(session_id="original-id")
        session.add_user_message("Test")
        session.metadata["key"] = "value"

        old_id = session.session_id
        session.clear()

        assert len(session) == 0
        assert session.metadata == {}
        assert session.session_id != old_id

    def test_repr(self):
        """Test string representation."""
        session = ConversationSession(session_id="test-id")
        session.add_user_message("Hello")

        repr_str = repr(session)
        assert "test-id" in repr_str
        assert "1" in repr_str  # 1 message


class TestGetOrCreateSession:
    """Tests for the get_or_create_session function."""

    def test_creates_new_session(self):
        """Test creating a new session when none exists."""
        state: dict = {}
        session = get_or_create_session(state)

        assert isinstance(session, ConversationSession)
        assert "session" in state
        assert state["session"] is session

    def test_returns_existing_session(self):
        """Test returning existing session."""
        state: dict = {}
        session1 = get_or_create_session(state)
        session1.add_user_message("Hello")

        session2 = get_or_create_session(state)

        assert session1 is session2
        assert len(session2) == 1

    def test_custom_session_key(self):
        """Test using a custom session key."""
        state: dict = {}
        session = get_or_create_session(state, session_key="my_session")

        assert "my_session" in state
        assert state["my_session"] is session
