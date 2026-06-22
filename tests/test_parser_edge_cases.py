"""Edge-case tests for the export parser."""

import json
import zipfile
from io import BytesIO
from pathlib import Path

import pytest

from consciousness.models import Role
from consciousness.parser.claude_export import ExportParseError, parse_export


def _make_zip(conversations: list, projects: list | None = None) -> Path:
    buf = BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr("conversations.json", json.dumps(conversations))
        if projects is not None:
            zf.writestr("projects.json", json.dumps(projects))
    tmp = Path("/tmp/test_edge.zip")
    tmp.write_bytes(buf.getvalue())
    return tmp


def _make_json(data) -> Path:
    tmp = Path("/tmp/test_edge.json")
    tmp.write_text(json.dumps(data))
    return tmp


BASE_MSG = {
    "uuid": "msg-1",
    "sender": "human",
    "text": "Hello",
    "created_at": "2024-06-01T10:00:00.000Z",
    "attachments": [],
    "files": [],
}

BASE_CONV = {
    "uuid": "conv-1",
    "name": "Test",
    "created_at": "2024-06-01T10:00:00.000Z",
    "updated_at": "2024-06-01T10:00:00.000Z",
    "account": {"uuid": "acc-1", "full_name": "Test"},
    "project": None,
    "chat_messages": [BASE_MSG],
}


# ── parsing formats ───────────────────────────────────────────────────────────


def test_parse_bare_json_array(tmp_path):
    path = _make_json([BASE_CONV])
    convs, projects = parse_export(path)
    assert len(convs) == 1


def test_parse_json_envelope(tmp_path):
    path = _make_json({"conversations": [BASE_CONV], "projects": []})
    convs, _ = parse_export(path)
    assert len(convs) == 1


def test_unsupported_extension_raises():
    path = Path("/tmp/export.txt")
    path.write_text("nope")
    with pytest.raises(ExportParseError, match="Unsupported"):
        parse_export(path)


def test_zip_missing_conversations_json(tmp_path):
    buf = BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr("other.json", "[]")
    path = tmp_path / "bad.zip"
    path.write_bytes(buf.getvalue())
    with pytest.raises(ExportParseError, match="conversations.json not found"):
        parse_export(path)


# ── conversation edge cases ───────────────────────────────────────────────────


def test_conversation_with_no_messages():
    conv = dict(BASE_CONV, chat_messages=[])
    path = _make_zip([conv])
    convs, _ = parse_export(path)
    assert convs[0].message_count == 0


def test_conversation_with_null_name():
    conv = dict(BASE_CONV, name=None)
    path = _make_zip([conv])
    convs, _ = parse_export(path)
    assert convs[0].title == "Untitled"


def test_conversation_with_missing_project():
    conv = dict(BASE_CONV, project=None)
    path = _make_zip([conv])
    convs, _ = parse_export(path)
    assert convs[0].project_id is None
    assert convs[0].project_name is None


def test_multiple_projects_inferred_from_conversations():
    conv1 = dict(BASE_CONV, uuid="c1", project={"uuid": "p1", "name": "Backend"})
    conv2 = dict(BASE_CONV, uuid="c2", project={"uuid": "p2", "name": "Frontend"})
    path = _make_zip([conv1, conv2])
    _, projects = parse_export(path)
    names = {p.name for p in projects}
    assert names == {"Backend", "Frontend"}


def test_explicit_projects_json_takes_precedence():
    conv = dict(BASE_CONV, project={"uuid": "p1", "name": "Old Name"})
    explicit_project = {"uuid": "p1", "name": "Authoritative Name", "created_at": "2024-01-01T00:00:00Z"}
    path = _make_zip([conv], projects=[explicit_project])
    _, projects = parse_export(path)
    assert projects[0].name == "Authoritative Name"


def test_project_conversation_count_accurate():
    conv1 = dict(BASE_CONV, uuid="c1", project={"uuid": "p1", "name": "Backend"})
    conv2 = dict(BASE_CONV, uuid="c2", project={"uuid": "p1", "name": "Backend"})
    conv3 = dict(BASE_CONV, uuid="c3", project=None)
    path = _make_zip([conv1, conv2, conv3])
    _, projects = parse_export(path)
    backend = next(p for p in projects if p.name == "Backend")
    assert backend.conversation_count == 2


# ── message edge cases ────────────────────────────────────────────────────────


def test_message_with_empty_text():
    msg = dict(BASE_MSG, text="")
    conv = dict(BASE_CONV, chat_messages=[msg])
    path = _make_zip([conv])
    convs, _ = parse_export(path)
    assert convs[0].messages[0].content == ""


def test_message_positions_are_sequential():
    msgs = [dict(BASE_MSG, uuid=f"m{i}", sender="human" if i % 2 == 0 else "assistant") for i in range(4)]
    conv = dict(BASE_CONV, chat_messages=msgs)
    path = _make_zip([conv])
    convs, _ = parse_export(path)
    positions = [m.position for m in convs[0].messages]
    assert positions == list(range(4))


def test_assistant_message_role():
    msg = dict(BASE_MSG, uuid="m2", sender="assistant", text="I am Claude.")
    conv = dict(BASE_CONV, chat_messages=[msg])
    path = _make_zip([conv])
    convs, _ = parse_export(path)
    assert convs[0].messages[0].role == Role.assistant


def test_timestamp_parsed_correctly():
    path = _make_zip([BASE_CONV])
    convs, _ = parse_export(path)
    ts = convs[0].created_at
    assert ts is not None
    assert ts.year == 2024
    assert ts.month == 6
    assert ts.day == 1
