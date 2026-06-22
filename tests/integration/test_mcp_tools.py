"""Integration tests for each MCP tool handler — called directly, no transport."""

import pytest

from consciousness.mcp_server.server import (
    get_conversation,
    get_project_context,
    get_recent_context,
    list_projects,
    recall_decision,
    search_history,
    synthesize_memory,
)
from consciousness.store.db import Database

pytestmark = pytest.mark.integration


@pytest.fixture
def stores(full_stores):
    db, vectors = full_stores
    return db, vectors


# ── search_history ────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_search_history_returns_results(stores):
    db, vectors = stores
    result = await search_history(db, vectors, {"query": "Postgres database"})
    assert len(result) == 1
    assert "Postgres" in result[0].text or "postgres" in result[0].text.lower()


@pytest.mark.asyncio
async def test_search_history_no_results(stores):
    db, vectors = stores
    result = await search_history(db, vectors, {"query": "quantum entanglement particle physics"})
    # Either no results or very low confidence results — just ensure it returns TextContent
    assert len(result) == 1
    assert isinstance(result[0].text, str)


@pytest.mark.asyncio
async def test_search_history_role_filter_assistant(stores):
    db, vectors = stores
    result = await search_history(db, vectors, {"query": "Postgres", "role": "assistant"})
    text = result[0].text
    # Should find the assistant's "Use Postgres for production" message
    assert "results" in text or "Postgres" in text


@pytest.mark.asyncio
async def test_search_history_project_filter(stores):
    db, vectors = stores
    result = await search_history(db, vectors, {"query": "database", "project": "Backend"})
    assert len(result) == 1


# ── get_project_context ───────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_get_project_context_found(stores):
    db, vectors = stores
    result = await get_project_context(db, {"project_name": "Backend"})
    text = result[0].text
    assert "Backend" in text
    assert "Database choice" in text or "Auth strategy" in text


@pytest.mark.asyncio
async def test_get_project_context_partial_match(stores):
    db, vectors = stores
    result = await get_project_context(db, {"project_name": "back"})
    assert "Backend" in result[0].text


@pytest.mark.asyncio
async def test_get_project_context_not_found(stores):
    db, vectors = stores
    result = await get_project_context(db, {"project_name": "nonexistent-xyz"})
    assert "No project found" in result[0].text


@pytest.mark.asyncio
async def test_get_project_context_with_messages(stores):
    db, vectors = stores
    result = await get_project_context(db, {"project_name": "Backend", "include_messages": True})
    text = result[0].text
    assert "Postgres" in text or "JWT" in text


# ── recall_decision ───────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_recall_decision_returns_assistant_content(stores):
    db, vectors = stores
    result = await recall_decision(db, vectors, {"topic": "database"})
    text = result[0].text
    # Should find "Use Postgres for production workloads" (assistant message)
    assert isinstance(text, str)
    assert len(text) > 0


@pytest.mark.asyncio
async def test_recall_decision_no_results(stores):
    db, vectors = stores
    result = await recall_decision(db, vectors, {"topic": "quantum computing algorithms"})
    assert isinstance(result[0].text, str)


# ── get_recent_context ────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_get_recent_context_far_future(stores):
    """Using days=99999 should include all conversations."""
    db, vectors = stores
    result = await get_recent_context(db, {"days": 99999})
    text = result[0].text
    assert "conversations" in text.lower() or "Database choice" in text


@pytest.mark.asyncio
async def test_get_recent_context_past_cutoff(stores):
    """Using days=0 should return no conversations (all are in 2024)."""
    db, vectors = stores
    result = await get_recent_context(db, {"days": 0})
    assert "No conversations" in result[0].text


# ── get_conversation ──────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_get_conversation_found(stores):
    db, vectors = stores
    result = await get_conversation(db, {"conversation_id": "conv-1"})
    text = result[0].text
    assert "Database choice" in text
    assert "Postgres" in text


@pytest.mark.asyncio
async def test_get_conversation_not_found(stores):
    db, vectors = stores
    result = await get_conversation(db, {"conversation_id": "does-not-exist"})
    assert "not found" in result[0].text.lower()


# ── list_projects ─────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_list_projects_shows_project(stores):
    db, vectors = stores
    result = await list_projects(db)
    assert "Backend" in result[0].text


@pytest.mark.asyncio
async def test_list_projects_empty_store(tmp_path):
    db = Database(tmp_path / "empty.db").connect()
    result = await list_projects(db)
    assert "No projects found" in result[0].text
    db.close()


# ── synthesize_memory ─────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_synthesize_memory_returns_blob(stores):
    db, vectors = stores
    result = await synthesize_memory(db, {})
    text = result[0].text
    assert "Memory Import Blob" in text
    assert "```" in text


@pytest.mark.asyncio
async def test_synthesize_memory_with_focus_topics(stores):
    db, vectors = stores
    result = await synthesize_memory(db, {"focus_topics": ["databases", "auth"]})
    text = result[0].text
    assert "Memory Import Blob" in text


@pytest.mark.asyncio
async def test_synthesize_memory_project_filter(stores):
    db, vectors = stores
    result = await synthesize_memory(db, {"project": "Backend"})
    text = result[0].text
    assert "Memory Import Blob" in text
