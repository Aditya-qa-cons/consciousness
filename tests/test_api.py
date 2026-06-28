"""Tests for the REST API server (cross-assistant portability)."""

import os

import pytest
from starlette.testclient import TestClient

from consciousness.api.app import _OPENAI_TOOLS
from consciousness.models import Decision, Role, TechChoice
from consciousness.store.db import Database
from consciousness.store.vectors import VectorStore
from tests.conftest import make_conversation, make_message, make_project, utc

# ── fixtures ───────────────────────────────────────────────────────────────────


@pytest.fixture
def api_client(tmp_path):
    """TestClient backed by a seeded in-memory DB and FakeEncoder vector store."""
    os.environ["CONSCIOUSNESS_FAKE_ENCODER"] = "1"

    db = Database(tmp_path / "test.db").connect()
    vectors = VectorStore(tmp_path / "vectors").connect()

    # Seed a project and two conversations
    proj = make_project(id="proj-1", name="Backend")
    db.upsert_project(proj)

    conv1 = make_conversation(
        id="conv-1", title="Database choice", project_id="proj-1",
        messages=[
            make_message("m1", "conv-1", Role.human, "Should I use Postgres?", 0),
            make_message("m2", "conv-1", Role.assistant, "Use Postgres for production.", 1),
        ],
        updated_at=utc(2024, 6, 1, 10),
    )
    conv2 = make_conversation(
        id="conv-2", title="Auth strategy", project_id="proj-1",
        messages=[
            make_message("m3", "conv-2", Role.human, "JWT or sessions?", 0),
            make_message("m4", "conv-2", Role.assistant, "Sessions are simpler.", 1),
        ],
        updated_at=utc(2024, 6, 2, 9),
    )
    db.upsert_conversation(conv1)
    db.upsert_conversation(conv2)

    decision = Decision(
        id="dec-1", topic="database", conclusion="Use Postgres", confidence=0.9,
        conversation_id="conv-1",
    )
    db.upsert_decision(decision)

    tc = TechChoice(
        id="tc-1", technology="Postgres", verdict="preferred",
        rationale="Battle-tested", conversation_id="conv-1",
    )
    db.upsert_tech_choice(tc)

    db.commit()
    vectors.index_conversation(conv1)
    vectors.index_conversation(conv2)

    # Build app manually without calling lifespan (TestClient handles it)
    from datetime import datetime, timedelta, timezone

    from fastapi import FastAPI, HTTPException
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.responses import PlainTextResponse

    from consciousness.api.app import (
        _OPENAI_TOOLS,
        OpenAIToolCallRequest,
        _handle_explore_kg,
        _handle_get_conversation,
        _handle_get_project_context,
        _handle_get_recent_context,
        _handle_list_projects,
        _handle_recall_decision,
        _handle_search,
        _handle_synthesize_memory,
    )

    fast_app = FastAPI(title="Test API")
    fast_app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["GET", "POST"])

    @fast_app.get("/api/v1/health")
    async def health():
        s = db.stats()
        return {"status": "ok", "conversations": s["conversations"]}

    @fast_app.get("/api/v1/stats")
    async def stats():
        return db.stats()

    @fast_app.get("/api/v1/projects")
    async def list_projects():
        return {"projects": [p.model_dump() for p in db.list_projects()]}

    @fast_app.get("/api/v1/conversations")
    async def list_conversations(project_id: str = "", page: int = 1, limit: int = 30):
        offset = (page - 1) * limit
        convs = db.list_conversations(project_id=project_id or None, limit=limit + 1, offset=offset)
        has_next = len(convs) > limit
        page_convs = convs[:limit]
        summaries = db.get_summaries([c.id for c in page_convs])
        return {
            "page": page, "limit": limit, "has_next": has_next,
            "conversations": [
                {**c.model_dump(exclude={"messages"}), "message_count": c.message_count,
                 "summary": summaries[c.id].summary if c.id in summaries else None}
                for c in page_convs
            ],
        }

    @fast_app.get("/api/v1/conversations/{conv_id}")
    async def get_conversation(conv_id: str):
        conv = db.get_conversation(conv_id)
        if not conv:
            raise HTTPException(status_code=404, detail="Conversation not found")
        return conv.model_dump()

    @fast_app.get("/api/v1/search")
    async def search(q: str = "", limit: int = 8, project: str | None = None, role: str | None = None):
        if not q.strip():
            return {"query": q, "results": []}
        text = await _handle_search(db, vectors, {"query": q, "limit": limit, "project": project, "role": role})
        return {"query": q, "text": text}

    @fast_app.get("/api/v1/decisions")
    async def decisions(topic: str = "", limit: int = 10):
        if topic.strip():
            ds = db.find_active_decisions(topic)[:limit]
        else:
            ds = db.list_decisions(limit=limit)
        return {"decisions": [d.model_dump() for d in ds]}

    @fast_app.get("/api/v1/recent")
    async def recent(days: int = 7, project: str | None = None):
        cutoff = datetime.now(timezone.utc) - timedelta(days=days)
        convs = db.list_conversations(limit=500)
        recent_convs = [c for c in convs if c.updated_at and c.updated_at >= cutoff]
        summaries = db.get_summaries([c.id for c in recent_convs])
        return {
            "days": days,
            "conversations": [
                {**c.model_dump(exclude={"messages"}), "message_count": c.message_count,
                 "summary": summaries[c.id].summary if c.id in summaries else None}
                for c in recent_convs
            ],
        }

    @fast_app.get("/api/v1/knowledge-graph")
    async def knowledge_graph(
        query: str = "co_occurring_technologies", technology: str | None = None, limit: int = 10,
    ):
        if query == "co_occurring_technologies":
            pairs = db.co_occurring_technologies(limit=limit)
            return {
                "query": query,
                "pairs": [{"tech1": t1, "tech2": t2, "conversations": int(w)} for t1, t2, w in pairs],
            }
        if query == "revisited_topics":
            topics = db.revisited_topics(limit=limit)
            return {"query": query, "topics": [{"topic": t, "decision_count": c} for t, c in topics]}
        raise HTTPException(status_code=422, detail="unknown query")

    @fast_app.get("/api/v1/context.md", response_class=PlainTextResponse)
    async def context_md():
        decisions_list = db.list_decisions(limit=20)
        lines = ["# Your Context\n"]
        if decisions_list:
            lines.append("## Recent Decisions\n")
            for d in decisions_list[:10]:
                lines.append(f"- **{d.topic}**: {d.conclusion[:200]}")
        return PlainTextResponse("\n".join(lines), media_type="text/markdown")

    @fast_app.get("/api/v1/openai/tools")
    async def openai_tools():
        return _OPENAI_TOOLS

    @fast_app.post("/api/v1/openai/tool-call")
    async def openai_tool_call(body: OpenAIToolCallRequest):
        name, args = body.name, body.arguments
        if name == "search_history":
            result = await _handle_search(db, vectors, args)
        elif name == "list_projects":
            result = await _handle_list_projects(db)
        elif name == "get_project_context":
            result = await _handle_get_project_context(db, args)
        elif name == "get_conversation":
            result = await _handle_get_conversation(db, args)
        elif name == "recall_decision":
            result = await _handle_recall_decision(db, vectors, args)
        elif name == "get_recent_context":
            result = await _handle_get_recent_context(db, args)
        elif name == "synthesize_memory":
            result = await _handle_synthesize_memory(db, args)
        elif name == "explore_knowledge_graph":
            result = await _handle_explore_kg(db, args)
        else:
            raise HTTPException(status_code=404, detail=f"Unknown tool: {name}")
        return {"role": "tool", "name": name, "content": result}

    with TestClient(fast_app) as client:
        yield client

    db.close()
    os.environ.pop("CONSCIOUSNESS_FAKE_ENCODER", None)


# ── health / stats ─────────────────────────────────────────────────────────────


def test_health_returns_ok(api_client):
    r = api_client.get("/api/v1/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"
    assert r.json()["conversations"] == 2


def test_stats_returns_counts(api_client):
    r = api_client.get("/api/v1/stats")
    assert r.status_code == 200
    data = r.json()
    assert data["conversations"] == 2
    assert data["messages"] == 4


# ── projects ───────────────────────────────────────────────────────────────────


def test_list_projects_returns_projects(api_client):
    r = api_client.get("/api/v1/projects")
    assert r.status_code == 200
    projects = r.json()["projects"]
    assert len(projects) == 1
    assert projects[0]["name"] == "Backend"


# ── conversations ──────────────────────────────────────────────────────────────


def test_list_conversations_returns_all(api_client):
    r = api_client.get("/api/v1/conversations")
    assert r.status_code == 200
    data = r.json()
    assert len(data["conversations"]) == 2
    assert data["page"] == 1
    assert not data["has_next"]


def test_list_conversations_pagination(api_client):
    r = api_client.get("/api/v1/conversations?limit=1")
    assert r.status_code == 200
    data = r.json()
    assert len(data["conversations"]) == 1
    assert data["has_next"] is True


def test_list_conversations_has_message_count_field(api_client):
    r = api_client.get("/api/v1/conversations")
    convs = r.json()["conversations"]
    assert all("message_count" in c for c in convs)
    assert all(isinstance(c["message_count"], int) for c in convs)


def test_get_conversation_returns_messages(api_client):
    r = api_client.get("/api/v1/conversations/conv-1")
    assert r.status_code == 200
    data = r.json()
    assert data["id"] == "conv-1"
    assert data["title"] == "Database choice"
    assert len(data["messages"]) == 2


def test_get_conversation_404(api_client):
    r = api_client.get("/api/v1/conversations/nonexistent")
    assert r.status_code == 404


# ── decisions ──────────────────────────────────────────────────────────────────


def test_decisions_list_all(api_client):
    r = api_client.get("/api/v1/decisions")
    assert r.status_code == 200
    assert len(r.json()["decisions"]) == 1


def test_decisions_filter_by_topic(api_client):
    r = api_client.get("/api/v1/decisions?topic=database")
    assert r.status_code == 200
    decisions = r.json()["decisions"]
    assert len(decisions) == 1
    assert decisions[0]["topic"] == "database"


def test_decisions_topic_no_match_returns_empty(api_client):
    r = api_client.get("/api/v1/decisions?topic=nonexistent")
    assert r.status_code == 200
    assert r.json()["decisions"] == []


# ── context.md ─────────────────────────────────────────────────────────────────


def test_context_md_returns_markdown(api_client):
    r = api_client.get("/api/v1/context.md")
    assert r.status_code == 200
    assert "text/markdown" in r.headers["content-type"]
    assert "# Your Context" in r.text


def test_context_md_includes_decisions(api_client):
    r = api_client.get("/api/v1/context.md")
    assert "database" in r.text.lower()


# ── openai tools ───────────────────────────────────────────────────────────────


def test_openai_tools_returns_list(api_client):
    r = api_client.get("/api/v1/openai/tools")
    assert r.status_code == 200
    tools = r.json()
    assert isinstance(tools, list)
    assert len(tools) >= 7


def test_openai_tools_have_correct_shape(api_client):
    tools = api_client.get("/api/v1/openai/tools").json()
    for tool in tools:
        assert tool["type"] == "function"
        assert "name" in tool["function"]
        assert "description" in tool["function"]
        assert "parameters" in tool["function"]


def test_openai_tools_includes_search_history(api_client):
    tools = api_client.get("/api/v1/openai/tools").json()
    names = [t["function"]["name"] for t in tools]
    assert "search_history" in names
    assert "list_projects" in names
    assert "recall_decision" in names


# ── openai tool-call ──────────────────────────────────────────────────────────


def test_openai_tool_call_list_projects(api_client):
    r = api_client.post("/api/v1/openai/tool-call", json={"name": "list_projects", "arguments": {}})
    assert r.status_code == 200
    data = r.json()
    assert data["role"] == "tool"
    assert data["name"] == "list_projects"
    assert "Backend" in data["content"]


def test_openai_tool_call_get_conversation(api_client):
    r = api_client.post(
        "/api/v1/openai/tool-call",
        json={"name": "get_conversation", "arguments": {"conversation_id": "conv-1"}},
    )
    assert r.status_code == 200
    assert "Database choice" in r.json()["content"]


def test_openai_tool_call_get_conversation_missing(api_client):
    r = api_client.post(
        "/api/v1/openai/tool-call",
        json={"name": "get_conversation", "arguments": {"conversation_id": "nope"}},
    )
    assert r.status_code == 200
    assert "not found" in r.json()["content"].lower()


def test_openai_tool_call_recall_decision(api_client):
    r = api_client.post(
        "/api/v1/openai/tool-call",
        json={"name": "recall_decision", "arguments": {"topic": "database"}},
    )
    assert r.status_code == 200
    assert "Postgres" in r.json()["content"]


def test_openai_tool_call_unknown_tool_returns_404(api_client):
    r = api_client.post("/api/v1/openai/tool-call", json={"name": "does_not_exist", "arguments": {}})
    assert r.status_code == 404


def test_openai_tool_call_get_project_context(api_client):
    r = api_client.post(
        "/api/v1/openai/tool-call",
        json={"name": "get_project_context", "arguments": {"project_name": "Backend"}},
    )
    assert r.status_code == 200
    assert "Backend" in r.json()["content"]


def test_openai_tool_call_get_recent_context(api_client):
    r = api_client.post(
        "/api/v1/openai/tool-call",
        json={"name": "get_recent_context", "arguments": {"days": 3650}},
    )
    assert r.status_code == 200
    data = r.json()
    assert data["role"] == "tool"


# ── knowledge graph endpoint ──────────────────────────────────────────────────


def test_knowledge_graph_co_occurring_empty(api_client):
    r = api_client.get("/api/v1/knowledge-graph?query=co_occurring_technologies")
    assert r.status_code == 200
    assert r.json()["query"] == "co_occurring_technologies"
    assert "pairs" in r.json()


def test_knowledge_graph_revisited_topics_empty(api_client):
    r = api_client.get("/api/v1/knowledge-graph?query=revisited_topics")
    assert r.status_code == 200
    assert "topics" in r.json()


# ── CORS headers ──────────────────────────────────────────────────────────────


def test_cors_header_present(api_client):
    r = api_client.get("/api/v1/health", headers={"Origin": "https://example.com"})
    assert r.status_code == 200
    assert r.headers.get("access-control-allow-origin") == "*"


# ── openai tools list is complete ────────────────────────────────────────────


def test_openai_tools_module_level_constant():
    names = {t["function"]["name"] for t in _OPENAI_TOOLS}
    assert names == {
        "search_history", "list_projects", "get_project_context",
        "get_conversation", "recall_decision", "get_recent_context",
        "synthesize_memory", "explore_knowledge_graph",
    }
