"""Fixtures for integration tests — ChromaDB + FakeEncoder (no network required).

FakeEncoder produces bag-of-words vectors: each dimension maps to a word hash.
Documents sharing words get higher cosine similarity, so keyword-based
search assertions still hold without downloading any model from HuggingFace.
"""

import hashlib

import numpy as np
import pytest

from consciousness.store.db import Database
from consciousness.store.vectors import VectorStore
from tests.conftest import Role, make_conversation, make_message, make_project, utc

_DIMS = 384


class FakeEncoder:
    """Deterministic bag-of-words encoder — no network, no model download."""

    def encode(self, texts: list[str], show_progress_bar: bool = False) -> np.ndarray:
        vecs = []
        for text in texts:
            vec = np.zeros(_DIMS, dtype=np.float32)
            for word in text.lower().split():
                idx = int(hashlib.md5(word.encode()).hexdigest(), 16) % _DIMS
                vec[idx] += 1.0
            norm = np.linalg.norm(vec)
            vecs.append(vec / norm if norm > 0 else vec)
        return np.array(vecs)


def _make_store(path) -> VectorStore:
    return VectorStore(path, encoder=FakeEncoder()).connect()


# ── fixtures ──────────────────────────────────────────────────────────────────


@pytest.fixture
def seeded_vector_store(tmp_path) -> VectorStore:
    store = _make_store(tmp_path / "vectors")

    conv1 = make_conversation(
        id="conv-1",
        messages=[
            make_message("m1", "conv-1", Role.human, "Should I use Postgres or SQLite?", 0),
            make_message("m2", "conv-1", Role.assistant, "Use Postgres for production workloads.", 1),
        ],
    )
    conv2 = make_conversation(
        id="conv-2",
        title="Auth strategy",
        messages=[
            make_message("m3", "conv-2", Role.human, "How do I set up JWT authentication?", 0),
            make_message("m4", "conv-2", Role.assistant,
                         "Install PyJWT, create a secret key, sign tokens on login.", 1),
        ],
    )
    store.index_conversation(conv1)
    store.index_conversation(conv2)

    yield store


@pytest.fixture
def full_stores(tmp_path) -> tuple[Database, VectorStore]:
    """Both DB and VectorStore seeded with the same two conversations."""
    db = Database(tmp_path / "test.db").connect()
    vectors = _make_store(tmp_path / "vectors")

    db.upsert_project(make_project())

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
            make_message("m4", "conv-2", Role.assistant, "Sessions are simpler; use JWT for stateless APIs.", 1),
        ],
        updated_at=utc(2024, 6, 2, 9),
    )
    for conv in [conv1, conv2]:
        db.upsert_conversation(conv)
        vectors.index_conversation(conv)
    db.commit()

    yield db, vectors
    db.close()
