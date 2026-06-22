"""Fixtures for integration tests — real sentence-transformers + ChromaDB."""

import pytest

from consciousness.store.db import Database
from consciousness.store.vectors import VectorStore
from tests.conftest import Role, make_conversation, make_message, make_project, utc


@pytest.fixture(scope="session")
def vector_store(tmp_path_factory):
    """Single VectorStore instance shared across the session to avoid re-loading the model."""
    path = tmp_path_factory.mktemp("vectors")
    store = VectorStore(path).connect()
    yield store


@pytest.fixture
def seeded_vector_store(vector_store, tmp_path):
    """VectorStore with a few indexed messages. Uses a fresh ChromaDB collection per test."""
    # Each test gets its own store to avoid cross-contamination
    store = VectorStore(tmp_path / "vectors").connect()

    conv1 = make_conversation(
        id="conv-1",
        messages=[
            make_message("m1", "conv-1", Role.human, "Should I use Postgres or SQLite?", 0),
            make_message("m2", "conv-1", Role.assistant, "Use Postgres for production workloads.", 1),
        ],
    )
    conv2 = make_conversation(
        id="conv-2",
        messages=[
            make_message("m3", "conv-2", Role.human, "How do I set up JWT authentication?", 0),
            make_message(
                "m4",
                "conv-2",
                Role.assistant,
                "Install PyJWT, create a secret key, sign tokens on login.",
                1,
            ),
        ],
    )
    for conv in [conv1, conv2]:
        for msg in conv.messages:
            store.index_message(msg)

    yield store


@pytest.fixture
def full_stores(tmp_path):
    """Both DB and VectorStore seeded with the same two conversations."""
    db = Database(tmp_path / "test.db").connect()
    vectors = VectorStore(tmp_path / "vectors").connect()

    p = make_project()
    db.upsert_project(p)

    conv1 = make_conversation(
        id="conv-1",
        title="Database choice",
        messages=[
            make_message("m1", "conv-1", Role.human, "Should I use Postgres or SQLite?", 0),
            make_message("m2", "conv-1", Role.assistant, "Use Postgres for production workloads.", 1),
        ],
        updated_at=utc(2024, 6, 1, 10),
    )
    conv2 = make_conversation(
        id="conv-2",
        title="Auth strategy",
        messages=[
            make_message("m3", "conv-2", Role.human, "JWT or sessions?", 0),
            make_message(
                "m4",
                "conv-2",
                Role.assistant,
                "Sessions are simpler; use JWT for stateless APIs.",
                1,
            ),
        ],
        updated_at=utc(2024, 6, 2, 9),
    )
    for conv in [conv1, conv2]:
        db.upsert_conversation(conv)
        for msg in conv.messages:
            vectors.index_message(msg)
    db.commit()

    yield db, vectors

    db.close()
