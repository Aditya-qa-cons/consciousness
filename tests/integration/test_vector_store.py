"""Integration tests for the vector store — exchange-level chunking + ChromaDB."""

import pytest

from consciousness.models import Role
from tests.conftest import make_conversation, make_message
from tests.integration.conftest import _make_store

pytestmark = pytest.mark.integration


def test_index_and_count(seeded_vector_store):
    assert seeded_vector_store.count() > 0


def test_search_returns_results(seeded_vector_store):
    hits = seeded_vector_store.search("Postgres", limit=5)
    assert len(hits) > 0


def test_search_postgres_query_finds_postgres_content(seeded_vector_store):
    hits = seeded_vector_store.search("Postgres database", limit=5)
    texts = " ".join(h.chunk_text for h in hits).lower()
    assert "postgres" in texts


def test_search_jwt_query_finds_jwt_content(seeded_vector_store):
    hits = seeded_vector_store.search("JWT authentication tokens", limit=5)
    texts = " ".join(h.chunk_text for h in hits).lower()
    assert "jwt" in texts or "token" in texts or "pyjwt" in texts


def test_role_filter_human_only(seeded_vector_store):
    hits = seeded_vector_store.search("database", limit=10, role_filter=Role.human)
    assert all(h.human_message_id in {"m1", "m3"} for h in hits)


def test_role_filter_assistant_only(seeded_vector_store):
    hits = seeded_vector_store.search("production workloads", limit=10, role_filter=Role.assistant)
    # Assistant chunks carry human_message_id of their paired human message
    assert all(h.human_message_id in {"m1", "m2", "m3", "m4"} for h in hits)


def test_conversation_id_filter(seeded_vector_store):
    hits = seeded_vector_store.search("database", limit=10, conversation_ids=["conv-1"])
    assert all(h.conversation_id == "conv-1" for h in hits)


def test_hit_has_relevance_label(seeded_vector_store):
    hits = seeded_vector_store.search("Postgres", limit=3)
    assert len(hits) > 0
    assert hits[0].relevance_label in {"high", "medium", "low"}


def test_exchange_deduplication(seeded_vector_store):
    """Each exchange should appear at most once in results even with multiple chunks."""
    hits = seeded_vector_store.search("Postgres database production", limit=10)
    exchange_ids = [h.exchange_id for h in hits]
    assert len(exchange_ids) == len(set(exchange_ids))


def test_empty_store_returns_no_results(tmp_path):
    store = _make_store(tmp_path / "empty")
    hits = store.search("anything", limit=5)
    assert hits == []


def test_long_message_is_chunked(tmp_path):
    store = _make_store(tmp_path / "chunked")
    long_text = "The quick brown fox jumps over the lazy dog. " * 60  # ~2700 chars
    conv = make_conversation(
        id="conv-x",
        messages=[
            make_message("long-q", "conv-x", Role.human, "Tell me about something.", 0),
            make_message("long-a", "conv-x", Role.assistant, long_text, 1),
        ],
    )
    store.index_conversation(conv)
    assert store.count() > 2  # human chunk + multiple assistant chunks


def test_empty_message_not_indexed(tmp_path):
    store = _make_store(tmp_path / "empty-msg")
    conv = make_conversation(
        id="conv-x",
        messages=[make_message("blank", "conv-x", Role.human, "   ", 0)],
    )
    store.index_conversation(conv)
    assert store.count() == 0


def test_upsert_idempotent(tmp_path):
    store = _make_store(tmp_path / "idem")
    conv = make_conversation(
        id="conv-1",
        messages=[
            make_message("m1", "conv-1", Role.human, "Hello world", 0),
            make_message("m2", "conv-1", Role.assistant, "Hi there!", 1),
        ],
    )
    store.index_conversation(conv)
    count_after_first = store.count()
    store.index_conversation(conv)
    assert store.count() == count_after_first


def test_score_threshold_filters_poor_matches(tmp_path):
    store = _make_store(tmp_path / "thresh")
    conv = make_conversation(
        id="conv-1",
        messages=[
            make_message("m1", "conv-1", Role.human, "Python vs Go for backend services?", 0),
            make_message("m2", "conv-1", Role.assistant, "Go for high concurrency, Python for ML.", 1),
        ],
    )
    store.index_conversation(conv)
    # A wildly off-topic query should return no results at default threshold
    hits = store.search("recipes for chocolate cake dessert baking", limit=5, score_threshold=0.3)
    assert hits == []
