"""Tests for the export parser — validate against synthetic Claude.ai export shape."""

import json
import zipfile
from io import BytesIO
from pathlib import Path

import pytest

from consciousness.parser.claude_export import parse_export
from consciousness.models import Role


def make_export_zip(conversations: list[dict]) -> Path:
    buf = BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr("conversations.json", json.dumps(conversations))
    tmp = Path("/tmp/test_export.zip")
    tmp.write_bytes(buf.getvalue())
    return tmp


SAMPLE_EXPORT = [
    {
        "uuid": "conv-1",
        "name": "Database design discussion",
        "created_at": "2024-06-01T10:00:00.000Z",
        "updated_at": "2024-06-01T10:30:00.000Z",
        "account": {"uuid": "acc-1", "full_name": "Test User"},
        "project": {"uuid": "proj-1", "name": "Backend"},
        "chat_messages": [
            {
                "uuid": "msg-1",
                "sender": "human",
                "text": "Should I use Postgres or SQLite?",
                "created_at": "2024-06-01T10:00:00.000Z",
                "attachments": [],
                "files": [],
            },
            {
                "uuid": "msg-2",
                "sender": "assistant",
                "text": "For a production app, use Postgres. SQLite is great for dev/testing.",
                "created_at": "2024-06-01T10:00:05.000Z",
                "attachments": [],
                "files": [],
            },
        ],
    },
    {
        "uuid": "conv-2",
        "name": "Auth strategy",
        "created_at": "2024-06-02T09:00:00.000Z",
        "updated_at": "2024-06-02T09:15:00.000Z",
        "account": {"uuid": "acc-1", "full_name": "Test User"},
        "project": None,
        "chat_messages": [
            {
                "uuid": "msg-3",
                "sender": "human",
                "text": "JWT or sessions?",
                "created_at": "2024-06-02T09:00:00.000Z",
                "attachments": [],
                "files": [],
            },
        ],
    },
]


def test_parse_zip():
    path = make_export_zip(SAMPLE_EXPORT)
    conversations, projects = parse_export(path)

    assert len(conversations) == 2
    assert len(projects) == 1

    conv1 = next(c for c in conversations if c.id == "conv-1")
    assert conv1.title == "Database design discussion"
    assert conv1.project_name == "Backend"
    assert len(conv1.messages) == 2
    assert conv1.messages[0].role == Role.human
    assert conv1.messages[1].role == Role.assistant


def test_conversation_without_project():
    path = make_export_zip(SAMPLE_EXPORT)
    conversations, _ = parse_export(path)
    conv2 = next(c for c in conversations if c.id == "conv-2")
    assert conv2.project_id is None


def test_message_content_preserved():
    path = make_export_zip(SAMPLE_EXPORT)
    conversations, _ = parse_export(path)
    conv1 = next(c for c in conversations if c.id == "conv-1")
    assert "Postgres" in conv1.messages[1].content


def test_project_conversation_count():
    path = make_export_zip(SAMPLE_EXPORT)
    _, projects = parse_export(path)
    backend = next(p for p in projects if p.name == "Backend")
    assert backend.conversation_count == 1
