"""Conversation memory: saved chats, recent history, and follow-up question rewriting.

Chats are kept in a small SQLite file (data/chats.sqlite3), so they survive restarts.

Why rewriting? The search engine sees one question at a time. "What about part-time
staff?" means nothing on its own, so before searching, a small LLM call turns it into
"How many days of annual leave do part-time employees get?" using the chat history.
The answer model also receives the recent history, so its reply continues the
conversation naturally."""

import json
import sqlite3
import time
import uuid
from contextlib import closing, contextmanager
from pathlib import Path
from typing import Protocol

from rag.models import Turn
from rag.openrouter import call_with_retry

DEFAULT_TITLE = "New chat"
MAX_TITLE = 60


class ChatStore:
    """Saved conversations. A connection is opened per call and closed straight away,
    so the database file is never left locked (Windows cannot delete a locked file)."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._db() as con:
            con.executescript("""
                CREATE TABLE IF NOT EXISTS chats (
                    id TEXT PRIMARY KEY,
                    title TEXT NOT NULL,
                    created_at REAL NOT NULL,
                    updated_at REAL NOT NULL
                );
                CREATE TABLE IF NOT EXISTS messages (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    chat_id TEXT NOT NULL REFERENCES chats(id) ON DELETE CASCADE,
                    role TEXT NOT NULL,
                    content TEXT NOT NULL,
                    meta_json TEXT NOT NULL DEFAULT '{}',
                    created_at REAL NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_messages_chat ON messages(chat_id, id);
            """)

    @contextmanager
    def _db(self):
        """A connection that commits on success, rolls back on error, and always closes."""
        with closing(sqlite3.connect(self.path)) as con:
            con.row_factory = sqlite3.Row
            con.execute("PRAGMA foreign_keys = ON")
            with con:            # transaction: commit or roll back
                yield con

    def create_chat(self, title: str = DEFAULT_TITLE) -> str:
        chat_id = uuid.uuid4().hex[:12]
        now = time.time()
        with self._db() as con:
            con.execute("INSERT INTO chats VALUES (?, ?, ?, ?)", (chat_id, title, now, now))
        return chat_id

    def list_chats(self) -> list[dict]:
        """Most recently used first."""
        with self._db() as con:
            rows = con.execute(
                "SELECT id, title, updated_at FROM chats ORDER BY updated_at DESC, rowid DESC"
            ).fetchall()
        return [dict(r) for r in rows]

    def exists(self, chat_id: str) -> bool:
        with self._db() as con:
            return con.execute("SELECT 1 FROM chats WHERE id = ?", (chat_id,)).fetchone() is not None

    def add_message(self, chat_id: str, role: str, content: str, **meta) -> None:
        """Extra fields (sources, cited, images, search_query) are stored as JSON."""
        now = time.time()
        with self._db() as con:
            con.execute(
                "INSERT INTO messages (chat_id, role, content, meta_json, created_at) "
                "VALUES (?, ?, ?, ?, ?)",
                (chat_id, role, content, json.dumps(meta), now),
            )
            con.execute("UPDATE chats SET updated_at = ? WHERE id = ?", (now, chat_id))
            if role == "user":   # the first question names the chat
                title = " ".join(content.split())
                title = title if len(title) <= MAX_TITLE else title[:MAX_TITLE - 3] + "..."
                con.execute("UPDATE chats SET title = ? WHERE id = ? AND title = ?",
                            (title, chat_id, DEFAULT_TITLE))

    def messages(self, chat_id: str) -> list[dict]:
        """Oldest first: {"role", "content", **meta}."""
        with self._db() as con:
            rows = con.execute(
                "SELECT role, content, meta_json FROM messages WHERE chat_id = ? ORDER BY id",
                (chat_id,),
            ).fetchall()
        return [{"role": r["role"], "content": r["content"], **json.loads(r["meta_json"])}
                for r in rows]

    def rename_chat(self, chat_id: str, title: str) -> None:
        with self._db() as con:
            con.execute("UPDATE chats SET title = ? WHERE id = ?", (title, chat_id))

    def delete_chat(self, chat_id: str) -> None:
        with self._db() as con:
            con.execute("DELETE FROM chats WHERE id = ?", (chat_id,))


def recent_history(messages: list[dict] | list[Turn], max_messages: int = 6,
                   max_chars: int = 4000, max_message_chars: int = 1200) -> list[Turn]:
    """The last few messages, as Turns, kept within a character budget.

    Older messages are dropped first. Failed answers (marked error=True) are skipped,
    so an error message is never treated as something the assistant said."""
    turns = []
    for m in messages:
        if isinstance(m, Turn):
            turns.append(m)
        elif not m.get("error") and m.get("role") in ("user", "assistant"):
            turns.append(Turn(m["role"], m["content"]))
    turns = [Turn(t.role, t.content[:max_message_chars]) for t in turns[-max_messages:]]
    while turns and sum(len(t.content) for t in turns) > max_chars:
        turns.pop(0)
    while turns and turns[0].role != "user":     # start the history with a question
        turns.pop(0)
    return turns


class QuestionRewriter(Protocol):
    def rewrite(self, history: list[Turn], question: str) -> str: ...


REWRITE_PROMPT = """
You rewrite a user's latest message into one standalone question for a document search engine.

Use the conversation to replace words like "it", "that", "they", "the chart" or
"what about ..." with what they refer to, so the question makes sense on its own.
Keep the user's meaning and language. Do not answer the question.
If the latest message is already standalone, return it unchanged.
Reply with the question only.
"""


class OpenRouterRewriter:
    def __init__(self, client, model: str = "openai/gpt-4o-mini",
                 fallback_model: str | None = None, max_tokens: int = 256):
        self.client = client
        self.model = model
        self.fallback_model = fallback_model
        self.max_tokens = max_tokens

    def rewrite(self, history: list[Turn], question: str) -> str:
        if not history:
            return question                       # nothing to resolve: no API call
        transcript = "\n".join(f"{t.role.capitalize()}: {t.content}" for t in history)
        response = call_with_retry(
            lambda model: self.client.chat.completions.create(
                model=model,
                max_tokens=self.max_tokens,
                messages=[
                    {"role": "system", "content": REWRITE_PROMPT},
                    {"role": "user", "content": f"Conversation:\n{transcript}\n\n"
                                                f"Latest message: {question}"},
                ],
            ),
            [m for m in (self.model, self.fallback_model) if m],
        )
        text = (response.choices[0].message.content or "").strip()
        text = text.splitlines()[0].strip().strip('"').strip() if text else ""
        return text if 0 < len(text) <= 500 else question
