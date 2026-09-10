# agents/conversation_manager.py
from __future__ import annotations

from typing import Any, Dict, List, Optional

from agent_utils.memory_store import MemoryStore


class ConversationManagerAgent:
    """
    Minimal 'agent' wrapper around MemoryStore so other agents/tools
    can call a stable interface. Keeps things simple:
    - make sure a conversation exists
    - append user/assistant turns
    - read history
    - rename/delete/list conversations
    - optional: keep assistant partials in RAM; persist only on finalize
    """

    def __init__(self, store: Optional[MemoryStore] = None):
        self.store = store or MemoryStore()
        # simple in-RAM buffer for current assistant partials per conversation
        self._partials: Dict[str, str] = {}

    # ---- conversation mgmt ----
    def ensure_conversation(self, conversation_id: Optional[str]) -> str:
        rows = self.store.list_conversations()
        if not rows:
            return self.store.create_conversation()
        if conversation_id:
            return conversation_id
        # default to newest (list_conversations should return DESC by created_at)
        return rows[0][0]

    def list_conversations(self) -> List[Dict[str, Any]]:
        rows = self.store.list_conversations()
        return [{"id": r[0], "title": r[1], "created_at": r[2]} for r in rows]

    def rename(self, conversation_id: str, title: str) -> Dict[str, Any]:
        self.store.rename_conversation(conversation_id, title)
        return {"ok": True, "conversation_id": conversation_id, "title": title}

    def delete(self, conversation_id: str) -> Dict[str, Any]:
        self.store.delete_conversation(conversation_id)
        self._partials.pop(conversation_id, None)
        return {"ok": True}

    def create(self, title: Optional[str] = None) -> Dict[str, Any]:
        cid = self.store.create_conversation(title)
        return {"ok": True, "conversation_id": cid}

    # ---- messages ----
    def save_user(self, conversation_id: str, content: str) -> Dict[str, Any]:
        cid = self.ensure_conversation(conversation_id)
        self.store.log_message(cid, "user", content)
        return {"ok": True, "conversation_id": cid}

    def set_assistant_partial(self, conversation_id: str, partial_text: str) -> Dict[str, Any]:
        # Keep partials in RAM for simplicity/quickness
        cid = self.ensure_conversation(conversation_id)
        self._partials[cid] = partial_text
        return {"ok": True, "conversation_id": cid}

    def finalize_assistant(self, conversation_id: str) -> Dict[str, Any]:
        cid = self.ensure_conversation(conversation_id)
        final_text = self._partials.pop(cid, "")
        # Only persist once at the end of the stream
        self.store.log_message(cid, "assistant", final_text)
        return {"ok": True, "conversation_id": cid, "content_len": len(final_text)}

    def get_history_messages(self, conversation_id: str, limit: int = 1000) -> List[Dict[str, Any]]:
        cid = self.ensure_conversation(conversation_id)
        return self.store.get_history(cid, limit=limit)
