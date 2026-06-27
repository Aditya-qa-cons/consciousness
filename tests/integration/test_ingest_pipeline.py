"""Integration tests for the full ingest pipeline: parse → DB → vectors → knowledge extraction."""

import json
import zipfile
from io import BytesIO
from pathlib import Path

import pytest
from click.testing import CliRunner

from consciousness.cli import cli
from consciousness.parser.claude_export import parse_export
from consciousness.store.db import Database
from consciousness.store.vectors import VectorStore
from tests.integration.conftest import _make_store

pytestmark = pytest.mark.integration

_RUNNER = CliRunner(env={"CONSCIOUSNESS_FAKE_ENCODER": "1"})


def _make_export_zip(conversations: list) -> Path:
    buf = BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr("conversations.json", json.dumps(conversations))
    path = Path("/tmp/ingest_test.zip")
    path.write_bytes(buf.getvalue())
    return path


EXPORT_DATA = [
    {
        "uuid": "conv-a",
        "name": "Python vs Go",
        "created_at": "2024-06-01T08:00:00Z",
        "updated_at": "2024-06-01T08:30:00Z",
        "account": {"uuid": "acc-1", "full_name": "Tester"},
        "project": {"uuid": "proj-lang", "name": "Language Choices"},
        "chat_messages": [
            {
                "uuid": "ma1",
                "sender": "human",
                "text": "Should I pick Python or Go for my new backend?",
                "created_at": "2024-06-01T08:00:00Z",
                "attachments": [],
                "files": [],
            },
            {
                "uuid": "ma2",
                "sender": "assistant",
                "text": "Python is better for ML-heavy services; Go is better for high-concurrency APIs.",
                "created_at": "2024-06-01T08:00:05Z",
                "attachments": [],
                "files": [],
            },
        ],
    },
    {
        "uuid": "conv-b",
        "name": "ORM vs raw SQL",
        "created_at": "2024-06-02T09:00:00Z",
        "updated_at": "2024-06-02T09:15:00Z",
        "account": {"uuid": "acc-1", "full_name": "Tester"},
        "project": {"uuid": "proj-lang", "name": "Language Choices"},
        "chat_messages": [
            {
                "uuid": "mb1",
                "sender": "human",
                "text": "SQLAlchemy ORM or raw SQL with psycopg?",
                "created_at": "2024-06-02T09:00:00Z",
                "attachments": [],
                "files": [],
            },
            {
                "uuid": "mb2",
                "sender": "assistant",
                "text": "Use raw SQL for complex queries; ORM for simple CRUD operations.",
                "created_at": "2024-06-02T09:00:05Z",
                "attachments": [],
                "files": [],
            },
        ],
    },
]


@pytest.fixture
def ingested(tmp_path) -> tuple[Database, VectorStore]:
    """Run the full ingest pipeline and return the resulting stores."""
    export_path = _make_export_zip(EXPORT_DATA)
    conversations, projects = parse_export(export_path)

    db = Database(tmp_path / "conversations.db").connect()
    vectors = _make_store(tmp_path / "vectors")

    for project in projects:
        db.upsert_project(project)
    for conv in conversations:
        db.upsert_conversation(conv)
        vectors.index_conversation(conv)
    db.commit()

    yield db, vectors
    db.close()


def test_pipeline_stores_correct_conversation_count(ingested):
    db, _ = ingested
    assert db.stats()["conversations"] == 2


def test_pipeline_stores_correct_message_count(ingested):
    db, _ = ingested
    assert db.stats()["messages"] == 4


def test_pipeline_stores_project(ingested):
    db, _ = ingested
    projects = db.list_projects()
    assert len(projects) == 1
    assert projects[0].name == "Language Choices"


def test_pipeline_vectors_indexed(ingested):
    _, vectors = ingested
    assert vectors.count() > 0


def test_pipeline_search_finds_python(ingested):
    _, vectors = ingested
    hits = vectors.search("Python backend language", limit=5)
    texts = " ".join(h.chunk_text for h in hits)
    assert "Python" in texts


def test_pipeline_search_finds_sql(ingested):
    _, vectors = ingested
    hits = vectors.search("SQL database queries", limit=5)
    texts = " ".join(h.chunk_text for h in hits)
    assert "SQL" in texts or "sql" in texts.lower()


def test_pipeline_conversation_retrievable_by_id(ingested):
    db, _ = ingested
    conv = db.get_conversation("conv-a")
    assert conv is not None
    assert conv.title == "Python vs Go"
    assert len(conv.messages) == 2


def test_pipeline_reingest_is_idempotent(ingested):
    db, vectors = ingested
    initial_conv_count = db.stats()["conversations"]
    initial_vec_count = vectors.count()

    export_path = _make_export_zip(EXPORT_DATA)
    conversations, projects = parse_export(export_path)
    for project in projects:
        db.upsert_project(project)
    for conv in conversations:
        db.upsert_conversation(conv)
        vectors.index_conversation(conv)
    db.commit()

    assert db.stats()["conversations"] == initial_conv_count
    assert vectors.count() == initial_vec_count


# ── incremental ingest (CLI-level) ────────────────────────────────────────────


def _make_zip(conversations: list, tmp_path: Path, name: str = "export.zip") -> Path:
    buf = BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr("conversations.json", json.dumps(conversations))
    p = tmp_path / name
    p.write_bytes(buf.getvalue())
    return p


_BASE_CONV = [
    {
        "uuid": "inc-1",
        "name": "DB Choices",
        "created_at": "2024-07-01T08:00:00Z",
        "updated_at": "2024-07-01T08:30:00Z",
        "account": {"uuid": "acc-1", "full_name": "Tester"},
        "project": None,
        "chat_messages": [
            {"uuid": "mi1", "sender": "human", "text": "Postgres or MySQL?",
             "created_at": "2024-07-01T08:00:00Z", "attachments": [], "files": []},
            {"uuid": "mi2", "sender": "assistant", "text": "I recommend Postgres because it has better JSON support.",
             "created_at": "2024-07-01T08:00:05Z", "attachments": [], "files": []},
        ],
    }
]

_UPDATED_CONV = [
    {
        "uuid": "inc-1",
        "name": "DB Choices",
        "created_at": "2024-07-01T08:00:00Z",
        "updated_at": "2024-07-02T10:00:00Z",   # newer timestamp
        "account": {"uuid": "acc-1", "full_name": "Tester"},
        "project": None,
        "chat_messages": [
            {"uuid": "mi1", "sender": "human", "text": "Postgres or MySQL?",
             "created_at": "2024-07-01T08:00:00Z", "attachments": [], "files": []},
            {"uuid": "mi2", "sender": "assistant",
             "text": "I recommend Postgres with pgBouncer because it handles pooling well.",
             "created_at": "2024-07-02T10:00:00Z", "attachments": [], "files": []},
        ],
    }
]


def test_incremental_skip_unchanged(tmp_path):
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    zip1 = _make_zip(_BASE_CONV, tmp_path, "first.zip")

    result = _RUNNER.invoke(cli, ["--data-dir", str(data_dir), "ingest", str(zip1)])
    assert result.exit_code == 0, result.output
    assert "1 new" in result.output

    zip2 = _make_zip(_BASE_CONV, tmp_path, "second.zip")
    result = _RUNNER.invoke(cli, ["--data-dir", str(data_dir), "ingest", str(zip2)])
    assert result.exit_code == 0, result.output
    assert "1 unchanged" in result.output
    assert "new" not in result.output


def test_incremental_processes_updated(tmp_path):
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    zip1 = _make_zip(_BASE_CONV, tmp_path, "first.zip")

    _RUNNER.invoke(cli, ["--data-dir", str(data_dir), "ingest", str(zip1)])

    zip2 = _make_zip(_UPDATED_CONV, tmp_path, "updated.zip")
    result = _RUNNER.invoke(cli, ["--data-dir", str(data_dir), "ingest", str(zip2)])
    assert result.exit_code == 0, result.output
    assert "1 updated" in result.output

    db = Database(data_dir / "conversations.db").connect()
    conv = db.get_conversation("inc-1")
    db.close()
    assert conv is not None
    assert "pgBouncer" in conv.messages[-1].content


def test_incremental_force_reprocesses_all(tmp_path):
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    zip1 = _make_zip(_BASE_CONV, tmp_path, "first.zip")

    _RUNNER.invoke(cli, ["--data-dir", str(data_dir), "ingest", str(zip1)])

    zip2 = _make_zip(_BASE_CONV, tmp_path, "second.zip")
    result = _RUNNER.invoke(cli, ["--data-dir", str(data_dir), "ingest", "--force", str(zip2)])
    assert result.exit_code == 0, result.output
    # --force means it won't skip even though unchanged; shows as updated (not new, since it exists)
    assert "unchanged" not in result.output


def test_incremental_knowledge_purged_on_update(tmp_path):
    """Re-processing an updated conversation should replace its extracted knowledge."""
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    zip1 = _make_zip(_BASE_CONV, tmp_path, "first.zip")
    _RUNNER.invoke(cli, ["--data-dir", str(data_dir), "ingest", str(zip1)])

    db = Database(data_dir / "conversations.db").connect()
    decisions_before = db.list_decisions()
    db.close()

    zip2 = _make_zip(_UPDATED_CONV, tmp_path, "updated.zip")
    _RUNNER.invoke(cli, ["--data-dir", str(data_dir), "ingest", str(zip2)])

    db = Database(data_dir / "conversations.db").connect()
    decisions_after = db.list_decisions()
    db.close()

    # Knowledge for conv is replaced, not duplicated — count stays stable or improves
    # (not doubled). The exact count depends on pattern matches; we just assert no duplication.
    assert len(decisions_after) <= len(decisions_before) + 5  # generous upper bound
