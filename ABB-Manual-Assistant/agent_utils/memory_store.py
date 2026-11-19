import datetime
import os
import sqlite3
import uuid


class MemoryStore:
    def __init__(self, db_path="data/memory.db"):
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        self._init_db()

    def _init_db(self):
        cur = self.conn.cursor()
        cur.execute("""
        CREATE TABLE IF NOT EXISTS conversations (
            id TEXT PRIMARY KEY,
            title TEXT,
            created_at TEXT
        )
        """)
        cur.execute("""
        CREATE TABLE IF NOT EXISTS messages (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            conversation_id TEXT,
            role TEXT,
            content TEXT,
            timestamp TEXT
        )
        """)
        self.conn.commit()

    def create_conversation(self, title=None):
        cid = str(uuid.uuid4())
        if not title:
            title = f"Chat {datetime.datetime.now().strftime('%Y-%m-%d %H:%M')}"
        self.conn.execute(
            "INSERT INTO conversations VALUES (?, ?, ?)", (cid, title, datetime.datetime.now().isoformat())
        )
        self.conn.commit()
        return cid

    def list_conversations(self):
        return self.conn.execute("SELECT id, title, created_at FROM conversations ORDER BY created_at DESC").fetchall()

    def get_history(self, conversation_id, limit=100):
        rows = self.conn.execute(
            "SELECT role, content FROM messages WHERE conversation_id=? ORDER BY id ASC LIMIT ?",
            (conversation_id, limit),
        ).fetchall()
        return [{"role": r[0], "content": r[1]} for r in rows]

    def log_message(self, conversation_id, role, content):
        self.conn.execute(
            "INSERT INTO messages (conversation_id, role, content, timestamp) VALUES (?, ?, ?, ?)",
            (conversation_id, role, content, datetime.datetime.now().isoformat()),
        )
        self.conn.commit()

    def rename_conversation(self, conversation_id, new_title):
        self.conn.execute("UPDATE conversations SET title=? WHERE id=?", (new_title, conversation_id))
        self.conn.commit()

    def delete_conversation(self, conversation_id):
        self.conn.execute("DELETE FROM messages WHERE conversation_id=?", (conversation_id,))
        self.conn.execute("DELETE FROM conversations WHERE id=?", (conversation_id,))
        self.conn.commit()
