"""Unit tests for the local web UI (FastAPI routes)."""

import pytest

fastapi = pytest.importorskip("fastapi")
import uuid  # noqa: E402

from starlette.testclient import TestClient  # noqa: E402 — after importorskip

from consciousness.models import Decision, Preference, Role, TechChoice  # noqa: E402
from consciousness.store.db import Database  # noqa: E402
from consciousness.web.app import create_app  # noqa: E402

# Re-use conftest helpers
from tests.conftest import make_conversation, make_message, make_project, utc  # noqa: E402


@pytest.fixture
def client(tmp_path):
    """TestClient backed by a seeded in-memory-ish SQLite database."""
    db = Database(tmp_path / "conversations.db").connect()

    proj = make_project(id="proj-1", name="Backend")
    db.upsert_project(proj)

    conv1 = make_conversation(
        id="conv-1",
        title="Database choice",
        project_id="proj-1",
        messages=[
            make_message("m1", "conv-1", Role.human, "Should I use Postgres or MySQL?", 0),
            make_message("m2", "conv-1", Role.assistant, "Use Postgres for production.", 1),
        ],
        updated_at=utc(2024, 6, 1),
    )
    conv2 = make_conversation(
        id="conv-2",
        title="Auth strategy",
        project_id="proj-1",
        messages=[
            make_message("m3", "conv-2", Role.human, "JWT or sessions?", 0),
            make_message("m4", "conv-2", Role.assistant, "Sessions are simpler.", 1),
        ],
        updated_at=utc(2024, 6, 2),
    )
    db.upsert_conversation(conv1)
    db.upsert_conversation(conv2)

    decision = Decision(
        id=str(uuid.uuid4()),
        topic="Database selection",
        conclusion="Use Postgres.",
        confidence=0.9,
        conversation_id="conv-1",
    )
    pref = Preference(
        id=str(uuid.uuid4()),
        area="Language",
        preference="Prefers Python.",
        conversation_id="conv-1",
    )
    tc = TechChoice(
        id=str(uuid.uuid4()),
        technology="Postgres",
        verdict="Recommended.",
        rationale="Best JSON support.",
        conversation_id="conv-1",
    )
    db.upsert_decision(decision)
    db.upsert_preference(pref)
    db.upsert_tech_choice(tc)
    db.commit()
    db.close()

    app = create_app(tmp_path)
    with TestClient(app) as c:
        yield c


# ── home / projects ───────────────────────────────────────────────────────────

def test_home_returns_200(client):
    r = client.get("/")
    assert r.status_code == 200


def test_home_shows_project(client):
    r = client.get("/")
    assert "Backend" in r.text


def test_home_shows_stats(client):
    r = client.get("/")
    assert "Conversations" in r.text or "conversations" in r.text


# ── conversation list ─────────────────────────────────────────────────────────

def test_conversations_list_returns_200(client):
    r = client.get("/conversations")
    assert r.status_code == 200


def test_conversations_list_shows_titles(client):
    r = client.get("/conversations")
    assert "Database choice" in r.text
    assert "Auth strategy" in r.text


def test_conversations_filter_by_project(client):
    r = client.get("/conversations?project_id=proj-1")
    assert r.status_code == 200
    assert "Database choice" in r.text


def test_conversations_filter_unknown_project_empty(client):
    r = client.get("/conversations?project_id=nonexistent")
    assert r.status_code == 200
    assert "No conversations found" in r.text


# ── conversation detail ───────────────────────────────────────────────────────

def test_conversation_detail_returns_200(client):
    r = client.get("/conversations/conv-1")
    assert r.status_code == 200


def test_conversation_detail_shows_title(client):
    r = client.get("/conversations/conv-1")
    assert "Database choice" in r.text


def test_conversation_detail_shows_messages(client):
    r = client.get("/conversations/conv-1")
    assert "Postgres" in r.text
    assert "Human" in r.text
    assert "Assistant" in r.text


def test_conversation_detail_404(client):
    r = client.get("/conversations/does-not-exist")
    assert r.status_code == 404


# ── search ────────────────────────────────────────────────────────────────────

def test_search_empty_query_returns_200(client):
    r = client.get("/search")
    assert r.status_code == 200


def test_search_with_query_returns_200(client):
    r = client.get("/search?q=Postgres")
    assert r.status_code == 200


def test_search_shows_matching_conversation(client):
    r = client.get("/search?q=Postgres")
    assert "Database choice" in r.text


def test_search_no_results_shows_empty_message(client):
    r = client.get("/search?q=xyzzynonexistent123")
    assert r.status_code == 200
    assert "No results" in r.text


# ── decisions dashboard ───────────────────────────────────────────────────────

def test_decisions_returns_200(client):
    r = client.get("/decisions")
    assert r.status_code == 200


def test_decisions_shows_decision(client):
    r = client.get("/decisions")
    assert "Database selection" in r.text


def test_decisions_shows_preference(client):
    r = client.get("/decisions")
    assert "Language" in r.text


def test_decisions_shows_tech_choice(client):
    r = client.get("/decisions")
    assert "Postgres" in r.text
