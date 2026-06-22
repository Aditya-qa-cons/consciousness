"""Integration tests for the vector store — real embeddings + ChromaDB."""

import pytest

from consciousness.models import Role
from tests.conftest import make_message
from tests.integration.conftest import _make_store

pytestmark = pytest.mark.integration


def test_index_and_count(seeded_vector_store):
    assert seeded_vector_store.count() > 0


def test_search_returns_results(seeded_vector_store):
    hits = seeded_vector_store.search("database", limit=5)
    assert len(hits) > 0


def test_search_postgres_query_finds_postgres_content(seeded_vector_store):
    hits = seeded_vector_store.search("Postgres database", limit=3)
    texts = " ".join(h.chunk_text for h in hits).lower()
    assert "postgres" in texts


def test_search_jwt_query_finds_jwt_content(seeded_vector_store):
    hits = seeded_vector_store.search("JWT authentication tokens", limit=3)
    texts = " ".join(h.chunk_text for h in hits).lower()
    assert "jwt" in texts or "token" in texts or "pyjwt" in texts.lower()


def test_role_filter_human_only(seeded_vector_store):
    hits = seeded_vector_store.search("database", limit=10, role_filter=Role.human)
    assert all(h.message_id in {"m1", "m3"} for h in hits)


def test_role_filter_assistant_only(seeded_vector_store):
    hits = seeded_vector_store.search("production", limit=10, role_filter=Role.assistant)
    assert all(h.message_id in {"m2", "m4"} for h in hits)


def test_conversation_id_filter(seeded_vector_store):
    hits = seeded_vector_store.search("database", limit=10, conversation_ids=["conv-1"])
    assert all(h.conversation_id == "conv-1" for h in hits)


def test_empty_store_returns_no_results(tmp_path):
    store = _make_store(tmp_path / "empty")
    hits = store.search("anything", limit=5)
    assert hits == []


def test_long_message_is_chunked(tmp_path):
    store = _make_store(tmp_path / "chunked")
    long_text = "The quick brown fox jumps. " * 100  # ~2600 chars, > 512 chunk size
    msg = make_message("long-msg", "conv-x", Role.human, long_text, 0)
    store.index_message(msg)
    # Should have multiple chunks indexed
    assert store.count() > 1


def test_empty_message_not_indexed(tmp_path):
    store = _make_store(tmp_path / "empty-msg")
    msg = make_message("blank", "conv-x", Role.human, "   ", 0)
    store.index_message(msg)
    assert store.count() == 0


def test_upsert_idempotent(tmp_path):
    store = _make_store(tmp_path / "idem")
    msg = make_message("m1", "conv-1", Role.human, "Hello world", 0)
    store.index_message(msg)
    count_after_first = store.count()
    store.index_message(msg)  # re-index same message
    assert store.count() == count_after_first  # should not duplicate
