"""Shared pytest fixtures available to all tests."""

from datetime import datetime, timezone

import pytest

from consciousness.models import Conversation, Message, Project, Role
from consciousness.store.db import Database


def utc(year, month, day, hour=0, minute=0) -> datetime:
    return datetime(year, month, day, hour, minute, tzinfo=timezone.utc)


# ── domain-model factories ────────────────────────────────────────────────────


def make_project(id="proj-1", name="Backend") -> Project:
    return Project(id=id, name=name, created_at=utc(2024, 1, 1))


def make_message(
    id="msg-1",
    conversation_id="conv-1",
    role=Role.human,
    content="Hello",
    position=0,
    timestamp=None,
) -> Message:
    return Message(
        id=id,
        conversation_id=conversation_id,
        role=role,
        content=content,
        timestamp=timestamp or utc(2024, 6, 1, 10, position),
        position=position,
    )


def make_conversation(
    id="conv-1",
    title="Test Conversation",
    project_id="proj-1",
    messages=None,
    updated_at=None,
) -> Conversation:
    if messages is None:
        messages = [
            make_message("msg-1", id, Role.human, "Should I use Postgres?", 0),
            make_message("msg-2", id, Role.assistant, "Use Postgres for production.", 1),
        ]
    return Conversation(
        id=id,
        title=title,
        project_id=project_id,
        created_at=utc(2024, 6, 1),
        updated_at=updated_at or utc(2024, 6, 1, 10, 1),
        messages=messages,
    )


# ── database fixture ──────────────────────────────────────────────────────────


@pytest.fixture
def db(tmp_path) -> Database:
    d = Database(tmp_path / "test.db").connect()
    yield d
    d.close()


@pytest.fixture
def seeded_db(tmp_path) -> Database:
    """Database pre-loaded with one project and two conversations."""
    d = Database(tmp_path / "seeded.db").connect()

    p = make_project()
    d.upsert_project(p)

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
            make_message("m3", "conv-2", Role.human, "JWT or sessions for auth?", 0),
            make_message("m4", "conv-2", Role.assistant, "Sessions are simpler; use JWT for stateless APIs.", 1),
        ],
        updated_at=utc(2024, 6, 2, 9),
    )
    d.upsert_conversation(conv1)
    d.upsert_conversation(conv2)
    d.commit()

    yield d
    d.close()
