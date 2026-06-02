from __future__ import annotations

# Gradio powers the UI
import gradio as gr

# Your local persistence layer (SQLite wrapper you already have)
from agent_utils.memory_store import MemoryStore

# Our tiny “agent” that owns conversation state and history (lives in project root)
from conversation_manager import ConversationManagerAgent

# Orchestrator remains the single “brain” that decides how to answer and streams tokens/chunks
from orchestrator_agent import Orchestrator

# Your tracing setup (left intact so nothing breaks)
from utils import setup_langfuse_tracer
from utils.langfuse.shared_client import langfuse_client  # noqa: F401  (imported for tracer wiring)

# -----------------------------------------------------------------------------
# Storage + manager agent
# -----------------------------------------------------------------------------
store = MemoryStore()
conv_agent = ConversationManagerAgent(store)   # The app talks to the conversation manager, not raw DB


# -----------------------------------------------------------------------------
# Small helpers for the dropdown and history mapping
# -----------------------------------------------------------------------------
def _label_for(id_: str, title: str) -> str:
    """Pretty labels for the chat selector dropdown: 'Title · abc123'."""
    return f"{title} · {id_[:6]}"

def _choices_and_maps():
    """
    Build dropdown choices and mapping between labels and conversation IDs.
    Ensures at least one conversation exists.
    """
    rows = store.list_conversations()
    if not rows:
        store.create_conversation()
        rows = store.list_conversations()
    choices = [_label_for(r[0], r[1]) for r in rows]
    id_by_label = { _label_for(r[0], r[1]): r[0] for r in rows }
    label_by_id = { r[0]: _label_for(r[0], r[1]) for r in rows }
    return choices, id_by_label, label_by_id, choices[0]  # default = first

def _history_messages(cid: str):
    """
    Read messages from the ConversationManagerAgent and convert to
    Chatbot(type='messages') format: [{role, content}, ...]
    """
    rows = conv_agent.get_history_messages(cid, limit=1000)
    return [{"role": r["role"], "content": r["content"]} for r in rows]


# -----------------------------------------------------------------------------
# The Gradio App
# -----------------------------------------------------------------------------
class GradioApp:
    def __init__(self):
        # One Orchestrator instance for the whole app
        self.orchestrator = Orchestrator()

    async def run_search_stream(self, message: str, chat_history):
        """
        Delegate to Orchestrator and stream chunks back.
        This is intentionally simple: the Orchestrator decides how to answer.
        """
        async for chunk in self.orchestrator.run(message, chat_history):
            # Each 'chunk' is a piece of the assistant's text (token/phrase/etc)
            yield chunk

    def launch(self):
        """
        Build and launch the Gradio UI.
        """
        with gr.Blocks(title="ABB Knowledgebase Search") as demo:
            # Top title
            gr.Markdown("### ABB Knowledgebase Search — Multi-Chat (per-chat history)")

            # Global UI state: we store the currently selected conversation_id here
            state = gr.State({})

            with gr.Row():
                # ----------------------- Sidebar: Conversation management -----------------------
                with gr.Column(scale=1, min_width=320):
                    gr.Markdown("**Chats**")

                    # Dropdown for selecting which conversation is active
                    chat_dropdown = gr.Dropdown(
                        choices=[],
                        value=None,
                        label="Select chat",
                        interactive=True
                    )

                    # Buttons/inputs for CRUD on conversations
                    new_title = gr.Textbox(label="New chat title", placeholder="Optional (auto if blank)")
                    btn_new   = gr.Button("New chat")

                    rename_to = gr.Textbox(label="Rename to")
                    btn_rename = gr.Button("Rename")

                    btn_delete = gr.Button("Delete chat")
                    btn_refresh = gr.Button("Refresh list")

                # ----------------------- Main chat panel -----------------------
                with gr.Column(scale=3):
                    # Chatbot uses OpenAI-style message dicts when type='messages'
                    chatbot = gr.Chatbot(label="Chat", height=520, type="messages")

                    # Input row: one-line textbox + Send button
                    with gr.Row():
                        msg = gr.Textbox(
                            placeholder="Ask me something about ABB errors...",
                            scale=9,
                            label="Your Question",
                            lines=1
                        )
                        send = gr.Button("Send", variant="primary", scale=1)

            # Some convenience example prompts
            examples = [
                "What is Error code 10039 and possible solution?",
                "What is a reference error?",
                "We are getting a Motor phase short circuit. Where should we look?",
                "How to set up an IRC5, and what is it?",
            ]
            gr.Examples(examples=examples, inputs=msg, label="Try one of these example questions:")

            # ----------------------- Callbacks -----------------------

            # Initialize the UI on page load:
            # - Populate dropdown
            # - Set default conversation_id in state
            # - Load that conversation's message history
            def _init():
                choices, id_by_label, label_by_id, default_label = _choices_and_maps()
                cid = id_by_label[default_label]
                return (
                    gr.update(choices=choices, value=default_label),  # dropdown options + selection
                    {"conversation_id": cid},                         # state
                    _history_messages(cid),                           # initial chat messages
                )

            demo.load(_init, inputs=None, outputs=[chat_dropdown, state, chatbot])

            # When the user switches the dropdown, update the active conversation and show its history
            def _select_chat(label, current_state):
                choices, id_by_label, label_by_id, default_label = _choices_and_maps()
                if not label or label not in id_by_label:
                    label = default_label
                cid = id_by_label[label]
                current_state = (current_state or {})
                current_state["conversation_id"] = cid
                return current_state, _history_messages(cid)

            chat_dropdown.change(_select_chat, inputs=[chat_dropdown, state], outputs=[state, chatbot])

            # The main send handler — this is an async generator that yields intermediate UI updates,
            # so you see the assistant's reply grow inside the *same* chat bubble (streaming).
            async def _send(user_text, current_state, current_msgs):
                # Guard: ignore empty messages but keep UI outputs consistent
                if not user_text or not user_text.strip():
                    yield current_msgs, "", (current_state or {})
                    return

                # Resolve the active conversation ID (default to first if none)
                choices, id_by_label, label_by_id, default_label = _choices_and_maps()
                cid = (current_state or {}).get("conversation_id") or id_by_label[default_label]
                current_state = {"conversation_id": cid}

                # Persist the user's message immediately
                conv_agent.save_user(cid, user_text)

                # Start from canonical history for this conversation
                messages = _history_messages(cid)

                # Append user's new message
                messages.append({"role": "user", "content": user_text})

                chat_history = messages

                # Append an *empty* assistant bubble that we'll fill as chunks arrive
                # Seed it with a quick typing indicator so the user sees a response is coming
                messages.append({"role": "assistant", "content": "…"})
                # Push user + typing indicator to UI immediately
                yield messages, "", current_state

                # Now stream the assistant reply into that last message in-place
                partial = ""
                try:
                    async for chunk in self.run_search_stream(user_text, chat_history):
                        partial += chunk          # grow the assistant's partial reply
                        messages[-1]["content"] = partial  # update the *same* assistant bubble
                        conv_agent.set_assistant_partial(cid, partial)  # keep partial in RAM (not DB)
                        yield messages, "", current_state  # push incremental update to the UI
                finally:
                    # When stream ends (or errors), persist the final assistant message once
                    conv_agent.finalize_assistant(cid)

                    # Reload canonical history from storage (ensures what you see is exactly what we saved)
                    messages = _history_messages(cid)
                    yield messages, "", current_state

            # Wire the Send button and Enter key to the same streaming handler
            send.click(_send, inputs=[msg, state, chatbot], outputs=[chatbot, msg, state])
            msg.submit(_send, inputs=[msg, state, chatbot], outputs=[chatbot, msg, state])

            # Create a new conversation; select it and show an empty history
            def _new_chat(title):
                res = conv_agent.create(title or None)
                cid = res["conversation_id"]
                choices, id_by_label, label_by_id, _ = _choices_and_maps()
                label = label_by_id[cid]
                return (
                    gr.update(choices=choices, value=label),  # select new chat in dropdown
                    {"conversation_id": cid},                 # set state
                    _history_messages(cid),                   # show empty (or fresh) history
                )

            btn_new.click(_new_chat, inputs=new_title, outputs=[chat_dropdown, state, chatbot])

            # Rename the current conversation; update the dropdown label
            def _rename_chat(new_title, current_state):
                choices, id_by_label, label_by_id, default_label = _choices_and_maps()
                cid = (current_state or {}).get("conversation_id") or id_by_label[default_label]
                if new_title and new_title.strip():
                    conv_agent.rename(cid, new_title.strip())
                choices, id_by_label, label_by_id, _ = _choices_and_maps()
                label = label_by_id[cid]
                return gr.update(choices=choices, value=label)

            btn_rename.click(_rename_chat, inputs=rename_to, outputs=chat_dropdown)

            # Delete the current conversation; ensure one remains and switch to it
            def _delete_chat(current_state):
                choices, id_by_label, label_by_id, default_label = _choices_and_maps()
                cid = (current_state or {}).get("conversation_id") or id_by_label[default_label]
                conv_agent.delete(cid)

                # Recompute choices; pick default; load its history
                choices, id_by_label, label_by_id, default_label = _choices_and_maps()
                new_cid = id_by_label[default_label]
                return (
                    gr.update(choices=choices, value=default_label),
                    {"conversation_id": new_cid},
                    _history_messages(new_cid),
                )

            btn_delete.click(_delete_chat, inputs=state, outputs=[chat_dropdown, state, chatbot])

            # Just refresh the dropdown list, keeping the same selection if still valid
            def _refresh_list(current_state):
                choices, id_by_label, label_by_id, default_label = _choices_and_maps()
                cid = (current_state or {}).get("conversation_id")
                if cid and cid in label_by_id:
                    label = label_by_id[cid]
                else:
                    label = default_label
                return gr.update(choices=choices, value=label)

            btn_refresh.click(_refresh_list, inputs=state, outputs=chat_dropdown)

        # Start the Gradio server
        demo.launch(server_name="0.0.0.0")


# -----------------------------------------------------------------------------
# Entry point (kept as-is for your tracing + app start)
# -----------------------------------------------------------------------------
def main():
    setup_langfuse_tracer()
    app = GradioApp()
    app.launch()

if __name__ == "__main__":
    main()
