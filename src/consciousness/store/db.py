"""SQLite persistence for conversations, messages, and projects."""

import json
import sqlite3
from datetime import datetime
from pathlib import Path

from consciousness.models import Conversation, Message, Project, Role

SCHEMA = """
PRAGMA journal_mode=WAL;
PRAGMA foreign_keys=ON;

CREATE TABLE IF NOT EXISTS projects (
    id          TEXT PRIMARY KEY,
    name        TEXT NOT NULL,
    created_at  TEXT
);

CREATE TABLE IF NOT EXISTS conversations (
    id          TEXT PRIMARY KEY,
    title       TEXT NOT NULL,
    project_id  TEXT REFERENCES projects(id),
    created_at  TEXT NOT NULL,
    updated_at  TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS messages (
    id              TEXT PRIMARY KEY,
    conversation_id TEXT NOT NULL REFERENCES conversations(id) ON DELETE CASCADE,
    role            TEXT NOT NULL CHECK(role IN ('human', 'assistant')),
    content         TEXT NOT NULL,
    timestamp       TEXT,
    position        INTEGER NOT NULL,
    attachments     TEXT DEFAULT '[]'
);

CREATE INDEX IF NOT EXISTS idx_messages_conv ON messages(conversation_id, position);
CREATE INDEX IF NOT EXISTS idx_conversations_project ON conversations(project_id);
CREATE INDEX IF NOT EXISTS idx_conversations_updated ON conversations(updated_at DESC);
"""


def _ts(dt: datetime | None) -> str | None:
    return dt.isoformat() if dt else None


def _from_ts(s: str | None) -> datetime | None:
    return datetime.fromisoformat(s) if s else None


class Database:
    def __init__(self, path: Path):
        self.path = path
        self._conn: sqlite3.Connection | None = None

    def connect(self) -> "Database":
        self._conn = sqlite3.connect(self.path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.executescript(SCHEMA)
        self._conn.commit()
        return self

    def close(self):
        if self._conn:
            self._conn.close()
            self._conn = None

    def __enter__(self):
        return self.connect()

    def __exit__(self, *_):
        self.close()

    @property
    def conn(self) -> sqlite3.Connection:
        if not self._conn:
            raise RuntimeError("Database not connected — call connect() first")
        return self._conn

    # ── write ──────────────────────────────────────────────────────────────

    def upsert_project(self, project: Project):
        self.conn.execute(
            "INSERT OR REPLACE INTO projects(id, name, created_at) VALUES (?,?,?)",
            (project.id, project.name, _ts(project.created_at)),
        )

    def upsert_conversation(self, conv: Conversation):
        self.conn.execute(
            """INSERT OR REPLACE INTO conversations(id, title, project_id, created_at, updated_at)
               VALUES (?,?,?,?,?)""",
            (conv.id, conv.title, conv.project_id, _ts(conv.created_at), _ts(conv.updated_at)),
        )
        for msg in conv.messages:
            self.upsert_message(msg)

    def upsert_message(self, msg: Message):
        self.conn.execute(
            """INSERT OR REPLACE INTO messages(id, conversation_id, role, content, timestamp, position, attachments)
               VALUES (?,?,?,?,?,?,?)""",
            (
                msg.id,
                msg.conversation_id,
                msg.role.value,
                msg.content,
                _ts(msg.timestamp),
                msg.position,
                json.dumps([a.model_dump() for a in msg.attachments]),
            ),
        )

    def commit(self):
        self.conn.commit()

    # ── read ───────────────────────────────────────────────────────────────

    def list_projects(self) -> list[Project]:
        rows = self.conn.execute(
            """SELECT p.*, COUNT(c.id) as conversation_count
               FROM projects p
               LEFT JOIN conversations c ON c.project_id = p.id
               GROUP BY p.id ORDER BY p.name"""
        ).fetchall()
        return [
            Project(
                id=r["id"],
                name=r["name"],
                created_at=_from_ts(r["created_at"]),
                conversation_count=r["conversation_count"],
            )
            for r in rows
        ]

    def list_conversations(
        self,
        project_id: str | None = None,
        limit: int = 50,
        offset: int = 0,
    ) -> list[Conversation]:
        q = "SELECT * FROM conversations"
        params: list = []
        if project_id:
            q += " WHERE project_id = ?"
            params.append(project_id)
        q += " ORDER BY updated_at DESC LIMIT ? OFFSET ?"
        params += [limit, offset]

        rows = self.conn.execute(q, params).fetchall()
        return [self._conv_from_row(r, include_messages=False) for r in rows]

    def get_conversation(self, conversation_id: str) -> Conversation | None:
        row = self.conn.execute(
            "SELECT * FROM conversations WHERE id = ?", (conversation_id,)
        ).fetchone()
        if not row:
            return None
        return self._conv_from_row(row, include_messages=True)

    def get_messages(self, conversation_id: str) -> list[Message]:
        rows = self.conn.execute(
            "SELECT * FROM messages WHERE conversation_id = ? ORDER BY position",
            (conversation_id,),
        ).fetchall()
        return [self._msg_from_row(r) for r in rows]

    def stats(self) -> dict:
        row = self.conn.execute(
            """SELECT
                (SELECT COUNT(*) FROM conversations) as conversations,
                (SELECT COUNT(*) FROM messages) as messages,
                (SELECT COUNT(*) FROM projects) as projects
            """
        ).fetchone()
        return dict(row)

    # ── helpers ────────────────────────────────────────────────────────────

    def _conv_from_row(self, r: sqlite3.Row, include_messages: bool) -> Conversation:
        messages = self.get_messages(r["id"]) if include_messages else []
        return Conversation(
            id=r["id"],
            title=r["title"],
            project_id=r["project_id"],
            created_at=_from_ts(r["created_at"]),
            updated_at=_from_ts(r["updated_at"]),
            messages=messages,
        )

    def _msg_from_row(self, r: sqlite3.Row) -> Message:
        return Message(
            id=r["id"],
            conversation_id=r["conversation_id"],
            role=Role(r["role"]),
            content=r["content"],
            timestamp=_from_ts(r["timestamp"]),
            position=r["position"],
        )
