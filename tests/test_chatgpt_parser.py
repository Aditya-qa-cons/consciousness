"""Unit tests for the ChatGPT export adapter."""

import json
import zipfile
from io import BytesIO
from pathlib import Path

import pytest

from consciousness.models import Role
from consciousness.parser.chatgpt_export import ChatGPTExportAdapter
from consciousness.parser.claude_export import ClaudeExportAdapter

# ── shared test data ──────────────────────────────────────────────────────────

_CHATGPT_CONV = {
    "id": "chatgpt-conv-1",
    "title": "Database choices",
    "create_time": 1717228800.0,   # 2024-06-01T08:00:00Z
    "update_time": 1717232400.0,   # 2024-06-01T09:00:00Z
    "mapping": {
        "root": {
            "id": "root",
            "message": None,
            "parent": None,
            "children": ["sys-1"],
        },
        "sys-1": {
            "id": "sys-1",
            "message": {
                "id": "sys-1",
                "author": {"role": "system"},
                "create_time": None,
                "content": {"content_type": "text", "parts": [""]},
            },
            "parent": "root",
            "children": ["user-1"],
        },
        "user-1": {
            "id": "user-1",
            "message": {
                "id": "user-1",
                "author": {"role": "user"},
                "create_time": 1717228800.0,
                "content": {"content_type": "text", "parts": ["Should I use Postgres or MySQL?"]},
            },
            "parent": "sys-1",
            "children": ["asst-1"],
        },
        "asst-1": {
            "id": "asst-1",
            "message": {
                "id": "asst-1",
                "author": {"role": "assistant"},
                "create_time": 1717228860.0,
                "content": {"content_type": "text", "parts": ["I recommend Postgres for most use cases."]},
            },
            "parent": "user-1",
            "children": ["user-2"],
        },
        "user-2": {
            "id": "user-2",
            "message": {
                "id": "user-2",
                "author": {"role": "user"},
                "create_time": 1717229000.0,
                "content": {"content_type": "text", "parts": ["Why not MySQL?"]},
            },
            "parent": "asst-1",
            "children": ["asst-2"],
        },
        "asst-2": {
            "id": "asst-2",
            "message": {
                "id": "asst-2",
                "author": {"role": "assistant"},
                "create_time": 1717229060.0,
                "content": {"content_type": "text", "parts": ["MySQL has weaker JSON support."]},
            },
            "parent": "user-2",
            "children": [],
        },
    },
}

_CLAUDE_CONV = {
    "uuid": "claude-conv-1",
    "name": "Test Claude convo",
    "created_at": "2024-06-01T08:00:00Z",
    "updated_at": "2024-06-01T09:00:00Z",
    "account": {"uuid": "acc-1", "full_name": "Tester"},
    "project": None,
    "chat_messages": [
        {"uuid": "m1", "sender": "human", "text": "Hello", "created_at": "2024-06-01T08:00:00Z",
         "attachments": [], "files": []},
        {"uuid": "m2", "sender": "assistant", "text": "Hi!", "created_at": "2024-06-01T08:00:01Z",
         "attachments": [], "files": []},
    ],
}


def _make_zip(conversations: list, tmp_path: Path, name: str = "export.zip") -> Path:
    buf = BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr("conversations.json", json.dumps(conversations))
    p = tmp_path / name
    p.write_bytes(buf.getvalue())
    return p


# ── can_handle ────────────────────────────────────────────────────────────────


def test_chatgpt_adapter_handles_chatgpt_zip(tmp_path):
    z = _make_zip([_CHATGPT_CONV], tmp_path)
    assert ChatGPTExportAdapter().can_handle(z) is True


def test_chatgpt_adapter_rejects_claude_zip(tmp_path):
    z = _make_zip([_CLAUDE_CONV], tmp_path)
    assert ChatGPTExportAdapter().can_handle(z) is False


def test_chatgpt_adapter_rejects_json_file(tmp_path):
    p = tmp_path / "convs.json"
    p.write_text(json.dumps([_CHATGPT_CONV]))
    assert ChatGPTExportAdapter().can_handle(p) is False


def test_claude_adapter_rejects_chatgpt_zip(tmp_path):
    z = _make_zip([_CHATGPT_CONV], tmp_path)
    assert ClaudeExportAdapter().can_handle(z) is False


def test_claude_adapter_handles_claude_zip(tmp_path):
    z = _make_zip([_CLAUDE_CONV], tmp_path)
    assert ClaudeExportAdapter().can_handle(z) is True


# ── conversation parsing ──────────────────────────────────────────────────────


@pytest.fixture
def parsed(tmp_path):
    z = _make_zip([_CHATGPT_CONV], tmp_path)
    conversations, projects = ChatGPTExportAdapter().parse(z)
    return conversations, projects


def test_parse_returns_one_conversation(parsed):
    conversations, _ = parsed
    assert len(conversations) == 1


def test_parse_conversation_title(parsed):
    conversations, _ = parsed
    assert conversations[0].title == "Database choices"


def test_parse_conversation_id(parsed):
    conversations, _ = parsed
    assert conversations[0].id == "chatgpt-conv-1"


def test_parse_timestamps(parsed):
    conversations, _ = parsed
    conv = conversations[0]
    assert conv.created_at is not None
    assert conv.updated_at is not None
    assert conv.updated_at > conv.created_at


def test_parse_skips_system_messages(parsed):
    conversations, _ = parsed
    roles = [m.role for m in conversations[0].messages]
    assert Role.human in roles
    assert Role.assistant in roles
    assert all(r in (Role.human, Role.assistant) for r in roles)


def test_parse_message_count(parsed):
    conversations, _ = parsed
    # 2 user + 2 assistant = 4; system skipped
    assert len(conversations[0].messages) == 4


def test_parse_message_roles_alternate(parsed):
    conversations, _ = parsed
    roles = [m.role for m in conversations[0].messages]
    assert roles == [Role.human, Role.assistant, Role.human, Role.assistant]


def test_parse_message_content(parsed):
    conversations, _ = parsed
    contents = [m.content for m in conversations[0].messages]
    assert "Postgres" in contents[1]
    assert "MySQL" in contents[3]


def test_parse_message_positions_sequential(parsed):
    conversations, _ = parsed
    positions = [m.position for m in conversations[0].messages]
    assert positions == list(range(len(positions)))


def test_parse_produces_default_project(parsed):
    _, projects = parsed
    assert len(projects) == 1
    assert projects[0].name == "ChatGPT"


def test_parse_conversation_assigned_to_project(parsed):
    conversations, projects = parsed
    assert conversations[0].project_id == projects[0].id


# ── auto-detection via parse_export ──────────────────────────────────────────


def test_auto_detect_chatgpt(tmp_path):
    from consciousness.parser import parse_export
    z = _make_zip([_CHATGPT_CONV], tmp_path)
    conversations, _ = parse_export(z)
    assert len(conversations) == 1
    assert "Postgres" in conversations[0].messages[1].content


def test_auto_detect_claude(tmp_path):
    from consciousness.parser import parse_export
    z = _make_zip([_CLAUDE_CONV], tmp_path)
    conversations, _ = parse_export(z)
    assert len(conversations) == 1
    assert conversations[0].title == "Test Claude convo"


# ── edge cases ────────────────────────────────────────────────────────────────


def test_empty_message_content_skipped(tmp_path):
    conv = {
        **_CHATGPT_CONV,
        "id": "edge-1",
        "mapping": {
            "root": {"id": "root", "message": None, "parent": None, "children": ["user-1"]},
            "user-1": {
                "id": "user-1",
                "message": {
                    "id": "user-1",
                    "author": {"role": "user"},
                    "create_time": 1717228800.0,
                    "content": {"content_type": "text", "parts": [""]},  # empty
                },
                "parent": "root", "children": ["asst-1"],
            },
            "asst-1": {
                "id": "asst-1",
                "message": {
                    "id": "asst-1",
                    "author": {"role": "assistant"},
                    "create_time": 1717228860.0,
                    "content": {"content_type": "text", "parts": ["Use Postgres."]},
                },
                "parent": "user-1", "children": [],
            },
        },
    }
    z = _make_zip([conv], tmp_path, "edge.zip")
    conversations, _ = ChatGPTExportAdapter().parse(z)
    # Only the non-empty assistant message survives
    assert len(conversations[0].messages) == 1
    assert conversations[0].messages[0].role == Role.assistant


def test_branching_follows_last_child(tmp_path):
    """When a user regenerates a response, we take the last (most recent) branch."""
    conv = {
        "id": "branch-1",
        "title": "Branched conv",
        "create_time": 1717228800.0,
        "update_time": 1717229000.0,
        "mapping": {
            "root": {"id": "root", "message": None, "parent": None, "children": ["user-1"]},
            "user-1": {
                "id": "user-1",
                "message": {
                    "id": "user-1", "author": {"role": "user"},
                    "create_time": 1717228800.0,
                    "content": {"content_type": "text", "parts": ["Which database?"]},
                },
                "parent": "root",
                # Two children = user regenerated the response; last one is the kept version
                "children": ["asst-old", "asst-new"],
            },
            "asst-old": {
                "id": "asst-old",
                "message": {
                    "id": "asst-old", "author": {"role": "assistant"},
                    "create_time": 1717228820.0,
                    "content": {"content_type": "text", "parts": ["Use MySQL."]},
                },
                "parent": "user-1", "children": [],
            },
            "asst-new": {
                "id": "asst-new",
                "message": {
                    "id": "asst-new", "author": {"role": "assistant"},
                    "create_time": 1717228860.0,
                    "content": {"content_type": "text", "parts": ["Use Postgres."]},
                },
                "parent": "user-1", "children": [],
            },
        },
    }
    z = _make_zip([conv], tmp_path, "branch.zip")
    conversations, _ = ChatGPTExportAdapter().parse(z)
    assert len(conversations[0].messages) == 2
    assert "Postgres" in conversations[0].messages[1].content
