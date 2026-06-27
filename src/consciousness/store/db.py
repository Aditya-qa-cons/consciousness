"""SQLite persistence for conversations, messages, projects, and extracted knowledge."""

import json
import sqlite3
import uuid
from datetime import datetime
from pathlib import Path

from consciousness.models import Conversation, Decision, ExcludeRule, Message, Preference, Project, Role, TechChoice

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

CREATE TABLE IF NOT EXISTS decisions (
    id              TEXT PRIMARY KEY,
    topic           TEXT NOT NULL,
    conclusion      TEXT NOT NULL,
    confidence      REAL NOT NULL DEFAULT 0.75,
    conversation_id TEXT REFERENCES conversations(id) ON DELETE CASCADE,
    extracted_at    TEXT NOT NULL,
    superseded_by   TEXT REFERENCES decisions(id)
);

CREATE TABLE IF NOT EXISTS preferences (
    id              TEXT PRIMARY KEY,
    area            TEXT NOT NULL,
    preference      TEXT NOT NULL,
    conversation_id TEXT REFERENCES conversations(id) ON DELETE CASCADE,
    extracted_at    TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS tech_choices (
    id              TEXT PRIMARY KEY,
    technology      TEXT NOT NULL,
    verdict         TEXT NOT NULL,
    rationale       TEXT,
    conversation_id TEXT REFERENCES conversations(id) ON DELETE CASCADE,
    extracted_at    TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS exclude_rules (
    pattern     TEXT PRIMARY KEY,
    rule_type   TEXT NOT NULL,
    created_at  TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_messages_conv       ON messages(conversation_id, position);
CREATE INDEX IF NOT EXISTS idx_conversations_proj  ON conversations(project_id);
CREATE INDEX IF NOT EXISTS idx_conversations_upd   ON conversations(updated_at DESC);
CREATE INDEX IF NOT EXISTS idx_decisions_topic     ON decisions(topic);
CREATE INDEX IF NOT EXISTS idx_decisions_conv      ON decisions(conversation_id);
CREATE INDEX IF NOT EXISTS idx_preferences_area    ON preferences(area);
CREATE INDEX IF NOT EXISTS idx_tech_choices_tech   ON tech_choices(technology);
"""


def _ts(dt: datetime | None) -> str | None:
    return dt.isoformat() if dt else None


def _from_ts(s: str | None) -> datetime | None:
    return datetime.fromisoformat(s) if s else None


def _new_id() -> str:
    return str(uuid.uuid4())


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

    # ── conversations + messages ───────────────────────────────────────────────

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

    def list_projects(self) -> list[Project]:
        rows = self.conn.execute(
            """SELECT p.*, COUNT(c.id) as conversation_count
               FROM projects p
               LEFT JOIN conversations c ON c.project_id = p.id
               GROUP BY p.id ORDER BY p.name"""
        ).fetchall()
        return [
            Project(
                id=r["id"], name=r["name"],
                created_at=_from_ts(r["created_at"]),
                conversation_count=r["conversation_count"],
            )
            for r in rows
        ]

    def list_conversations(self, project_id: str | None = None, limit: int = 50, offset: int = 0) -> list[Conversation]:
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
        row = self.conn.execute("SELECT * FROM conversations WHERE id = ?", (conversation_id,)).fetchone()
        if not row:
            return None
        return self._conv_from_row(row, include_messages=True)

    def get_conversation_updated_at(self, conversation_id: str) -> "datetime | None":
        row = self.conn.execute(
            "SELECT updated_at FROM conversations WHERE id = ?", (conversation_id,)
        ).fetchone()
        return _from_ts(row["updated_at"]) if row else None

    def delete_knowledge_for_conversation(self, conversation_id: str):
        """Remove all extracted knowledge rows for a conversation before re-extraction."""
        self.conn.execute("DELETE FROM decisions WHERE conversation_id = ?", (conversation_id,))
        self.conn.execute("DELETE FROM preferences WHERE conversation_id = ?", (conversation_id,))
        self.conn.execute("DELETE FROM tech_choices WHERE conversation_id = ?", (conversation_id,))

    def get_messages(self, conversation_id: str) -> list[Message]:
        rows = self.conn.execute(
            "SELECT * FROM messages WHERE conversation_id = ? ORDER BY position", (conversation_id,)
        ).fetchall()
        return [self._msg_from_row(r) for r in rows]

    def stats(self) -> dict:
        row = self.conn.execute(
            """SELECT
                (SELECT COUNT(*) FROM conversations) as conversations,
                (SELECT COUNT(*) FROM messages)      as messages,
                (SELECT COUNT(*) FROM projects)      as projects,
                (SELECT COUNT(*) FROM decisions)     as decisions,
                (SELECT COUNT(*) FROM preferences)   as preferences,
                (SELECT COUNT(*) FROM tech_choices)  as tech_choices
            """
        ).fetchone()
        return dict(row)

    # ── decisions ─────────────────────────────────────────────────────────────

    def upsert_decision(self, decision: Decision):
        self.conn.execute(
            "INSERT OR REPLACE INTO decisions"
            "(id, topic, conclusion, confidence, conversation_id, extracted_at, superseded_by)"
            " VALUES (?,?,?,?,?,?,?)",
            (decision.id, decision.topic, decision.conclusion, decision.confidence,
             decision.conversation_id, _ts(decision.extracted_at), decision.superseded_by),
        )

    def find_active_decisions(self, topic: str) -> list[Decision]:
        """Return non-superseded decisions whose topic contains the search term."""
        rows = self.conn.execute(
            "SELECT * FROM decisions WHERE lower(topic) LIKE ? AND superseded_by IS NULL ORDER BY extracted_at DESC",
            (f"%{topic.lower()}%",),
        ).fetchall()
        return [self._decision_from_row(r) for r in rows]

    def list_decisions(self, limit: int = 50) -> list[Decision]:
        rows = self.conn.execute(
            "SELECT * FROM decisions WHERE superseded_by IS NULL ORDER BY extracted_at DESC LIMIT ?", (limit,)
        ).fetchall()
        return [self._decision_from_row(r) for r in rows]

    def supersede_decision(self, old_id: str, new_id: str):
        self.conn.execute("UPDATE decisions SET superseded_by = ? WHERE id = ?", (new_id, old_id))

    # ── preferences ───────────────────────────────────────────────────────────

    def upsert_preference(self, pref: Preference):
        self.conn.execute(
            "INSERT OR REPLACE INTO preferences"
            "(id, area, preference, conversation_id, extracted_at) VALUES (?,?,?,?,?)",
            (pref.id, pref.area, pref.preference, pref.conversation_id, _ts(pref.extracted_at)),
        )

    def list_preferences(self) -> list[Preference]:
        rows = self.conn.execute("SELECT * FROM preferences ORDER BY extracted_at DESC").fetchall()
        return [self._pref_from_row(r) for r in rows]

    # ── tech choices ───────────────────────────────────────────────────────────

    def upsert_tech_choice(self, tc: TechChoice):
        self.conn.execute(
            """INSERT OR REPLACE INTO tech_choices(id, technology, verdict, rationale, conversation_id, extracted_at)
               VALUES (?,?,?,?,?,?)""",
            (tc.id, tc.technology, tc.verdict, tc.rationale, tc.conversation_id, _ts(tc.extracted_at)),
        )

    def list_tech_choices(self) -> list[TechChoice]:
        rows = self.conn.execute("SELECT * FROM tech_choices ORDER BY extracted_at DESC").fetchall()
        return [self._tech_from_row(r) for r in rows]

    # ── exclude rules ─────────────────────────────────────────────────────────

    def add_exclude_rule(self, rule: ExcludeRule):
        self.conn.execute(
            "INSERT OR REPLACE INTO exclude_rules(pattern, rule_type, created_at) VALUES (?,?,?)",
            (rule.pattern, rule.rule_type, _ts(rule.created_at)),
        )

    def remove_exclude_rule(self, pattern: str):
        self.conn.execute("DELETE FROM exclude_rules WHERE pattern = ?", (pattern,))

    def list_exclude_rules(self) -> list[ExcludeRule]:
        rows = self.conn.execute("SELECT * FROM exclude_rules ORDER BY created_at").fetchall()
        return [
            ExcludeRule(pattern=r["pattern"], rule_type=r["rule_type"], created_at=_from_ts(r["created_at"]))
            for r in rows
        ]

    def is_excluded(self, conv: Conversation) -> bool:
        """Return True if conv matches any active exclude rule."""
        import fnmatch

        rules = self.list_exclude_rules()
        for rule in rules:
            if rule.rule_type == "conversation_id" and rule.pattern == conv.id:
                return True
            if rule.rule_type == "project_id" and rule.pattern == conv.project_id:
                return True
            if rule.rule_type == "title_glob" and fnmatch.fnmatch(conv.title.lower(), rule.pattern.lower()):
                return True
        return False

    # ── helpers ────────────────────────────────────────────────────────────────

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

    def _decision_from_row(self, r: sqlite3.Row) -> Decision:
        return Decision(
            id=r["id"],
            topic=r["topic"],
            conclusion=r["conclusion"],
            confidence=r["confidence"],
            conversation_id=r["conversation_id"],
            extracted_at=_from_ts(r["extracted_at"]),
            superseded_by=r["superseded_by"],
        )

    def _pref_from_row(self, r: sqlite3.Row) -> Preference:
        return Preference(
            id=r["id"],
            area=r["area"],
            preference=r["preference"],
            conversation_id=r["conversation_id"],
            extracted_at=_from_ts(r["extracted_at"]),
        )

    def _tech_from_row(self, r: sqlite3.Row) -> TechChoice:
        return TechChoice(
            id=r["id"],
            technology=r["technology"],
            verdict=r["verdict"],
            rationale=r["rationale"],
            conversation_id=r["conversation_id"],
            extracted_at=_from_ts(r["extracted_at"]),
        )
