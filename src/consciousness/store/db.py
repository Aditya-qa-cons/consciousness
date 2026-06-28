"""SQLite persistence for conversations, messages, projects, and extracted knowledge."""

import json
import sqlite3
import uuid
from datetime import datetime
from pathlib import Path

from consciousness.models import (
    Conversation,
    ConversationSummary,
    Decision,
    ExcludeRule,
    KGEdge,
    KGNode,
    Message,
    Preference,
    Project,
    Role,
    TechChoice,
)

SCHEMA = """
PRAGMA journal_mode=WAL;
PRAGMA foreign_keys=ON;

CREATE TABLE IF NOT EXISTS projects (
    id          TEXT PRIMARY KEY,
    name        TEXT NOT NULL,
    created_at  TEXT,
    account_id  TEXT
);

CREATE TABLE IF NOT EXISTS conversations (
    id           TEXT PRIMARY KEY,
    title        TEXT NOT NULL,
    project_id   TEXT REFERENCES projects(id),
    created_at   TEXT NOT NULL,
    updated_at   TEXT NOT NULL,
    account_id   TEXT,
    content_hash TEXT
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
CREATE INDEX IF NOT EXISTS idx_conversations_hash  ON conversations(content_hash);
CREATE INDEX IF NOT EXISTS idx_conversations_upd   ON conversations(updated_at DESC);
CREATE INDEX IF NOT EXISTS idx_decisions_topic     ON decisions(topic);
CREATE INDEX IF NOT EXISTS idx_decisions_conv      ON decisions(conversation_id);
CREATE INDEX IF NOT EXISTS idx_preferences_area    ON preferences(area);
CREATE INDEX IF NOT EXISTS idx_tech_choices_tech   ON tech_choices(technology);

CREATE TABLE IF NOT EXISTS summaries (
    conversation_id TEXT PRIMARY KEY REFERENCES conversations(id) ON DELETE CASCADE,
    summary         TEXT NOT NULL,
    generated_at    TEXT NOT NULL,
    model           TEXT
);

CREATE VIRTUAL TABLE IF NOT EXISTS messages_fts USING fts5(
    message_id   UNINDEXED,
    conversation_id UNINDEXED,
    role         UNINDEXED,
    content,
    tokenize     = 'unicode61 remove_diacritics 1'
);

CREATE TABLE IF NOT EXISTS kg_nodes (
    id    TEXT PRIMARY KEY,
    type  TEXT NOT NULL,
    label TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS kg_edges (
    src_id   TEXT NOT NULL REFERENCES kg_nodes(id) ON DELETE CASCADE,
    dst_id   TEXT NOT NULL REFERENCES kg_nodes(id) ON DELETE CASCADE,
    relation TEXT NOT NULL,
    weight   REAL NOT NULL DEFAULT 1.0,
    PRIMARY KEY (src_id, dst_id, relation)
);

CREATE INDEX IF NOT EXISTS idx_kg_edges_src ON kg_edges(src_id);
CREATE INDEX IF NOT EXISTS idx_kg_edges_dst ON kg_edges(dst_id);
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
        self._migrate()
        self._conn.commit()
        return self

    def _migrate(self):
        """Forward-only column additions for existing databases."""
        for stmt in [
            "ALTER TABLE conversations ADD COLUMN account_id TEXT",
            "ALTER TABLE conversations ADD COLUMN content_hash TEXT",
            "ALTER TABLE projects ADD COLUMN account_id TEXT",
        ]:
            try:
                self._conn.execute(stmt)
            except Exception:
                pass  # column already exists

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
            "INSERT OR REPLACE INTO projects(id, name, created_at, account_id) VALUES (?,?,?,?)",
            (project.id, project.name, _ts(project.created_at), project.account_id),
        )

    def upsert_conversation(self, conv: Conversation):
        cols = "id, title, project_id, created_at, updated_at, account_id, content_hash"
        self.conn.execute(
            f"INSERT OR REPLACE INTO conversations({cols}) VALUES (?,?,?,?,?,?,?)",
            (conv.id, conv.title, conv.project_id, _ts(conv.created_at), _ts(conv.updated_at),
             conv.account_id, conv.content_hash),
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
        # Keep FTS index in sync — FTS5 has no INSERT OR REPLACE, so delete then insert.
        self.conn.execute("DELETE FROM messages_fts WHERE message_id = ?", (msg.id,))
        self.conn.execute(
            "INSERT INTO messages_fts(message_id, conversation_id, role, content) VALUES (?,?,?,?)",
            (msg.id, msg.conversation_id, msg.role.value, msg.content),
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
                account_id=r["account_id"],
            )
            for r in rows
        ]

    def find_by_content_hash(self, content_hash: str) -> str | None:
        """Return the conversation_id if a conversation with this hash already exists."""
        row = self.conn.execute(
            "SELECT id FROM conversations WHERE content_hash = ?", (content_hash,)
        ).fetchone()
        return row["id"] if row else None

    def list_accounts(self) -> list[str]:
        """Return distinct non-null account IDs across all conversations."""
        rows = self.conn.execute(
            "SELECT DISTINCT account_id FROM conversations WHERE account_id IS NOT NULL ORDER BY account_id"
        ).fetchall()
        return [r["account_id"] for r in rows]

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

    def fulltext_search(
        self,
        query: str,
        limit: int = 20,
        conversation_ids: list[str] | None = None,
        role: str | None = None,
    ) -> list[dict]:
        """BM25 full-text search over message content. Returns dicts with message_id,
        conversation_id, role, snippet, and rank (lower rank = better match)."""
        where_parts = ["messages_fts MATCH ?"]
        params: list = [query]
        if conversation_ids:
            placeholders = ",".join("?" * len(conversation_ids))
            where_parts.append(f"conversation_id IN ({placeholders})")
            params.extend(conversation_ids)
        if role:
            where_parts.append("role = ?")
            params.append(role)
        params.append(limit)

        rows = self.conn.execute(
            f"""SELECT message_id, conversation_id, role,
                       snippet(messages_fts, 3, '**', '**', '…', 24) AS snippet,
                       rank
                FROM messages_fts
                WHERE {' AND '.join(where_parts)}
                ORDER BY rank
                LIMIT ?""",
            params,
        ).fetchall()
        return [
            {
                "message_id": r["message_id"],
                "conversation_id": r["conversation_id"],
                "role": r["role"],
                "snippet": r["snippet"],
                "rank": r["rank"],
            }
            for r in rows
        ]

    def rebuild_fts(self):
        """Repopulate the FTS index from the messages table. Used by rebuild-index."""
        self.conn.execute("DELETE FROM messages_fts")
        rows = self.conn.execute("SELECT id, conversation_id, role, content FROM messages").fetchall()
        self.conn.executemany(
            "INSERT INTO messages_fts(message_id, conversation_id, role, content) VALUES (?,?,?,?)",
            [(r["id"], r["conversation_id"], r["role"], r["content"]) for r in rows],
        )

    def stats(self) -> dict:
        row = self.conn.execute(
            """SELECT
                (SELECT COUNT(*) FROM conversations) as conversations,
                (SELECT COUNT(*) FROM messages)      as messages,
                (SELECT COUNT(*) FROM projects)      as projects,
                (SELECT COUNT(*) FROM decisions)     as decisions,
                (SELECT COUNT(*) FROM preferences)   as preferences,
                (SELECT COUNT(*) FROM tech_choices)  as tech_choices,
                (SELECT COUNT(*) FROM summaries)     as summaries,
                (SELECT COUNT(*) FROM kg_nodes)      as kg_nodes,
                (SELECT COUNT(*) FROM kg_edges)      as kg_edges,
                (SELECT COUNT(DISTINCT account_id) FROM conversations WHERE account_id IS NOT NULL) as accounts
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

    # ── summaries ─────────────────────────────────────────────────────────────

    def upsert_summary(self, s: ConversationSummary):
        self.conn.execute(
            """INSERT OR REPLACE INTO summaries(conversation_id, summary, generated_at, model)
               VALUES (?,?,?,?)""",
            (s.conversation_id, s.summary, _ts(s.generated_at), s.model),
        )

    def get_summary(self, conversation_id: str) -> ConversationSummary | None:
        row = self.conn.execute(
            "SELECT * FROM summaries WHERE conversation_id = ?", (conversation_id,)
        ).fetchone()
        return self._summary_from_row(row) if row else None

    def get_summaries(self, conversation_ids: list[str]) -> dict[str, ConversationSummary]:
        if not conversation_ids:
            return {}
        placeholders = ",".join("?" * len(conversation_ids))
        rows = self.conn.execute(
            f"SELECT * FROM summaries WHERE conversation_id IN ({placeholders})",
            conversation_ids,
        ).fetchall()
        return {r["conversation_id"]: self._summary_from_row(r) for r in rows}

    # ── knowledge graph ────────────────────────────────────────────────────────

    def clear_kg(self):
        self.conn.execute("DELETE FROM kg_edges")
        self.conn.execute("DELETE FROM kg_nodes")

    def upsert_kg_node(self, node: KGNode):
        self.conn.execute(
            "INSERT OR REPLACE INTO kg_nodes(id, type, label) VALUES (?,?,?)",
            (node.id, node.type, node.label),
        )

    def upsert_kg_edge(self, edge: KGEdge):
        self.conn.execute(
            """INSERT INTO kg_edges(src_id, dst_id, relation, weight) VALUES (?,?,?,?)
               ON CONFLICT(src_id, dst_id, relation) DO UPDATE SET weight = excluded.weight""",
            (edge.src_id, edge.dst_id, edge.relation, edge.weight),
        )

    def get_kg_node(self, node_id: str) -> KGNode | None:
        row = self.conn.execute("SELECT * FROM kg_nodes WHERE id = ?", (node_id,)).fetchone()
        return self._kg_node_from_row(row) if row else None

    def get_kg_neighbors(self, node_id: str, relation: str | None = None) -> list[tuple[KGEdge, KGNode]]:
        """Return (edge, neighbor_node) pairs for all edges touching node_id (both directions)."""
        rel_clause = " AND e.relation = ?" if relation else ""
        params = [node_id] + ([relation] if relation else [])

        out_rows = self.conn.execute(
            f"SELECT e.src_id, e.dst_id, e.relation, e.weight, n.id AS nid, n.type AS ntype, n.label"
            f" FROM kg_edges e JOIN kg_nodes n ON n.id = e.dst_id"
            f" WHERE e.src_id = ?{rel_clause}",
            params,
        ).fetchall()
        in_rows = self.conn.execute(
            f"SELECT e.src_id, e.dst_id, e.relation, e.weight, n.id AS nid, n.type AS ntype, n.label"
            f" FROM kg_edges e JOIN kg_nodes n ON n.id = e.src_id"
            f" WHERE e.dst_id = ?{rel_clause}",
            params,
        ).fetchall()

        result = []
        for r in out_rows + in_rows:
            edge = KGEdge(src_id=r["src_id"], dst_id=r["dst_id"], relation=r["relation"], weight=r["weight"])
            node = KGNode(id=r["nid"], type=r["ntype"], label=r["label"])
            result.append((edge, node))
        return result

    def co_occurring_technologies(self, min_weight: float = 1.0, limit: int = 20) -> list[tuple[str, str, float]]:
        """Return (tech1_label, tech2_label, weight) pairs sorted by weight desc."""
        rows = self.conn.execute(
            """SELECT n1.label, n2.label, e.weight
               FROM kg_edges e
               JOIN kg_nodes n1 ON n1.id = e.src_id
               JOIN kg_nodes n2 ON n2.id = e.dst_id
               WHERE e.relation = 'co_occurs_with' AND e.weight >= ?
               ORDER BY e.weight DESC LIMIT ?""",
            (min_weight, limit),
        ).fetchall()
        return [(r[0], r[1], r[2]) for r in rows]

    def revisited_topics(self, limit: int = 20) -> list[tuple[str, int]]:
        """Decision topics that appear in more than one decision row."""
        rows = self.conn.execute(
            """SELECT topic, COUNT(*) AS cnt
               FROM decisions
               GROUP BY lower(topic)
               HAVING cnt > 1
               ORDER BY cnt DESC LIMIT ?""",
            (limit,),
        ).fetchall()
        return [(r["topic"], r["cnt"]) for r in rows]

    def list_all_decisions(self) -> list[Decision]:
        """All decisions including superseded ones."""
        rows = self.conn.execute("SELECT * FROM decisions ORDER BY extracted_at DESC").fetchall()
        return [self._decision_from_row(r) for r in rows]

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
            account_id=r["account_id"],
            content_hash=r["content_hash"],
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

    def _summary_from_row(self, r: sqlite3.Row) -> ConversationSummary:
        return ConversationSummary(
            conversation_id=r["conversation_id"],
            summary=r["summary"],
            generated_at=_from_ts(r["generated_at"]) or datetime.now(),
            model=r["model"],
        )

    def _kg_node_from_row(self, r: sqlite3.Row) -> KGNode:
        return KGNode(id=r["id"], type=r["type"], label=r["label"])

    def _tech_from_row(self, r: sqlite3.Row) -> TechChoice:
        return TechChoice(
            id=r["id"],
            technology=r["technology"],
            verdict=r["verdict"],
            rationale=r["rationale"],
            conversation_id=r["conversation_id"],
            extracted_at=_from_ts(r["extracted_at"]),
        )
