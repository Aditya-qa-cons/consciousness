"""Tests for SQLite database layer."""

from datetime import datetime
from pathlib import Path

import pytest

from consciousness.store.db import Database
from consciousness.models import Conversation, Message, Project, Role


@pytest.fixture
def db(tmp_path):
    d = Database(tmp_path / "test.db").connect()
    yield d
    d.close()


def make_project() -> Project:
    return Project(id="proj-1", name="Test Project", created_at=datetime(2024, 1, 1))


def make_conversation(include_messages=True) -> Conversation:
    messages = []
    if include_messages:
        messages = [
            Message(id="msg-1", conversation_id="conv-1", role=Role.human, content="Hello", timestamp=datetime(2024, 6, 1, 10), position=0),
            Message(id="msg-2", conversation_id="conv-1", role=Role.assistant, content="Hi there!", timestamp=datetime(2024, 6, 1, 10, 1), position=1),
        ]
    return Conversation(
        id="conv-1",
        title="Test Conversation",
        project_id="proj-1",
        created_at=datetime(2024, 6, 1),
        updated_at=datetime(2024, 6, 1, 10, 1),
        messages=messages,
    )


def test_upsert_and_retrieve_project(db):
    p = make_project()
    db.upsert_project(p)
    db.commit()
    projects = db.list_projects()
    assert len(projects) == 1
    assert projects[0].name == "Test Project"


def test_upsert_conversation_with_messages(db):
    db.upsert_project(make_project())
    conv = make_conversation()
    db.upsert_conversation(conv)
    db.commit()

    retrieved = db.get_conversation("conv-1")
    assert retrieved is not None
    assert retrieved.title == "Test Conversation"
    assert len(retrieved.messages) == 2
    assert retrieved.messages[0].role == Role.human
    assert retrieved.messages[1].content == "Hi there!"


def test_list_conversations_by_project(db):
    db.upsert_project(make_project())
    db.upsert_conversation(make_conversation())
    db.commit()

    convs = db.list_conversations(project_id="proj-1")
    assert len(convs) == 1
    assert convs[0].id == "conv-1"

    convs_other = db.list_conversations(project_id="other-proj")
    assert len(convs_other) == 0


def test_stats(db):
    db.upsert_project(make_project())
    db.upsert_conversation(make_conversation())
    db.commit()

    s = db.stats()
    assert s["conversations"] == 1
    assert s["messages"] == 2
    assert s["projects"] == 1


def test_get_nonexistent_conversation(db):
    assert db.get_conversation("nonexistent") is None
