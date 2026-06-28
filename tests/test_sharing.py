"""Tests for share-export, share-import, and shared exclude rules."""

import io
import json
import zipfile
from datetime import datetime, timezone
from pathlib import Path

import pytest
from click.testing import CliRunner

from consciousness.cli import cli
from consciousness.models import Conversation, Decision, Message, Project, Role, TechChoice
from consciousness.store.db import Database


def _utc(*args) -> datetime:
    return datetime(*args, tzinfo=timezone.utc)


@pytest.fixture
def runner():
    return CliRunner(env={"CONSCIOUSNESS_FAKE_ENCODER": "1"})


@pytest.fixture
def data_dir(tmp_path) -> Path:
    """Data dir pre-seeded with two projects and three conversations."""
    d = tmp_path / "data"
    d.mkdir()

    db = Database(d / "conversations.db").connect()

    db.upsert_project(Project(id="proj-a", name="Alpha", created_at=_utc(2024, 1, 1)))
    db.upsert_project(Project(id="proj-b", name="Beta", created_at=_utc(2024, 1, 2)))

    def _msg(mid, cid, role, text, pos):
        return Message(id=mid, conversation_id=cid, role=Role(role),
                       content=text, timestamp=_utc(2024, 6, 1), position=pos)

    conv1 = Conversation(
        id="conv-a1", title="Alpha One", project_id="proj-a",
        created_at=_utc(2024, 6, 1), updated_at=_utc(2024, 6, 1),
        messages=[_msg("m1", "conv-a1", "human", "Hello", 0),
                  _msg("m2", "conv-a1", "assistant", "Hi there", 1)],
    )
    conv2 = Conversation(
        id="conv-a2", title="Alpha Two", project_id="proj-a",
        created_at=_utc(2024, 6, 2), updated_at=_utc(2024, 6, 2),
        messages=[_msg("m3", "conv-a2", "human", "Question", 0)],
    )
    conv3 = Conversation(
        id="conv-b1", title="Beta One", project_id="proj-b",
        created_at=_utc(2024, 6, 3), updated_at=_utc(2024, 6, 3),
        messages=[_msg("m4", "conv-b1", "human", "Beta msg", 0)],
    )

    for conv in (conv1, conv2, conv3):
        db.upsert_conversation(conv)

    db.upsert_decision(Decision(
        id="d1", topic="database", conclusion="Use Postgres",
        confidence=0.9, conversation_id="conv-a1", extracted_at=_utc(2024, 6, 1),
    ))
    db.upsert_tech_choice(TechChoice(
        id="tc1", technology="Redis", verdict="cache layer",
        conversation_id="conv-a1", extracted_at=_utc(2024, 6, 1),
    ))

    db.commit()
    db.close()
    return d


# ── share-export ──────────────────────────────────────────────────────────────


def _read_share_json(bundle_path: Path) -> dict:
    with zipfile.ZipFile(bundle_path) as zf:
        return json.loads(zf.read("share.json"))


def test_share_export_project_creates_valid_zip(runner, data_dir, tmp_path):
    out = tmp_path / "out.consciousness"
    result = runner.invoke(cli, ["--data-dir", str(data_dir), "share-export",
                                 str(out), "--project", "proj-a", "--namespace", "alice"])
    assert result.exit_code == 0, result.output
    assert out.exists()
    payload = _read_share_json(out)
    assert payload["format"] == "consciousness-share"
    assert payload["namespace"] == "alice"


def test_share_export_project_filters_conversations(runner, data_dir, tmp_path):
    out = tmp_path / "out.consciousness"
    runner.invoke(cli, ["--data-dir", str(data_dir), "share-export",
                        str(out), "--project", "proj-a"])
    payload = _read_share_json(out)
    exported_ids = {c["id"] for c in payload["conversations"]}
    assert "conv-a1" in exported_ids
    assert "conv-a2" in exported_ids
    assert "conv-b1" not in exported_ids


def test_share_export_specific_conversation(runner, data_dir, tmp_path):
    out = tmp_path / "out.consciousness"
    runner.invoke(cli, ["--data-dir", str(data_dir), "share-export",
                        str(out), "--conversation", "conv-b1"])
    payload = _read_share_json(out)
    assert len(payload["conversations"]) == 1
    assert payload["conversations"][0]["id"] == "conv-b1"


def test_share_export_includes_messages(runner, data_dir, tmp_path):
    out = tmp_path / "out.consciousness"
    runner.invoke(cli, ["--data-dir", str(data_dir), "share-export",
                        str(out), "--conversation", "conv-a1"])
    payload = _read_share_json(out)
    msgs = payload["conversations"][0]["messages"]
    assert len(msgs) == 2
    assert any(m["content"] == "Hello" for m in msgs)


def test_share_export_includes_decisions(runner, data_dir, tmp_path):
    out = tmp_path / "out.consciousness"
    runner.invoke(cli, ["--data-dir", str(data_dir), "share-export",
                        str(out), "--conversation", "conv-a1"])
    payload = _read_share_json(out)
    decisions = payload["conversations"][0]["decisions"]
    assert any(d["topic"] == "database" for d in decisions)


def test_share_export_no_project_no_conv_exits_nonzero(runner, data_dir, tmp_path):
    out = tmp_path / "out.consciousness"
    result = runner.invoke(cli, ["--data-dir", str(data_dir), "share-export", str(out)])
    assert result.exit_code != 0
    assert "Specify" in result.output


def test_share_export_no_data_exits_nonzero(runner, tmp_path):
    empty = tmp_path / "empty"
    empty.mkdir()
    out = tmp_path / "out.consciousness"
    result = runner.invoke(cli, ["--data-dir", str(empty), "share-export",
                                 str(out), "--project", "proj-a"])
    assert result.exit_code != 0
    assert "No data found" in result.output


def test_share_export_respects_exclude_rules(runner, data_dir, tmp_path):
    # Exclude conv-a2 by title glob
    runner.invoke(cli, ["--data-dir", str(data_dir), "exclude", "add", "--title", "*alpha two*"])
    out = tmp_path / "out.consciousness"
    runner.invoke(cli, ["--data-dir", str(data_dir), "share-export",
                        str(out), "--project", "proj-a"])
    payload = _read_share_json(out)
    exported_ids = {c["id"] for c in payload["conversations"]}
    assert "conv-a2" not in exported_ids
    assert "conv-a1" in exported_ids


def test_share_export_bundles_shared_exclude_rules(runner, data_dir, tmp_path):
    runner.invoke(cli, ["--data-dir", str(data_dir), "exclude", "add", "--title", "*secret*", "--shared"])
    out = tmp_path / "out.consciousness"
    runner.invoke(cli, ["--data-dir", str(data_dir), "share-export",
                        str(out), "--project", "proj-a"])
    payload = _read_share_json(out)
    assert len(payload["shared_exclude_rules"]) == 1
    assert payload["shared_exclude_rules"][0]["pattern"] == "*secret*"


def test_share_export_does_not_bundle_non_shared_rules(runner, data_dir, tmp_path):
    runner.invoke(cli, ["--data-dir", str(data_dir), "exclude", "add", "--title", "*private*"])
    out = tmp_path / "out.consciousness"
    runner.invoke(cli, ["--data-dir", str(data_dir), "share-export",
                        str(out), "--project", "proj-a"])
    payload = _read_share_json(out)
    assert payload["shared_exclude_rules"] == []


def test_share_export_adds_consciousness_extension(runner, data_dir, tmp_path):
    out = tmp_path / "noext"
    runner.invoke(cli, ["--data-dir", str(data_dir), "share-export",
                        str(out), "--project", "proj-a"])
    assert (tmp_path / "noext.consciousness").exists()


# ── share-import ──────────────────────────────────────────────────────────────


def _make_share_bundle(namespace: str, conversations: list, projects: list,
                       shared_rules: list | None = None) -> bytes:
    payload = {
        "version": 1,
        "format": "consciousness-share",
        "namespace": namespace,
        "exported_at": "2024-06-01T00:00:00+00:00",
        "shared_exclude_rules": shared_rules or [],
        "projects": projects,
        "conversations": conversations,
    }
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr("share.json", json.dumps(payload))
    return buf.getvalue()


def _make_bundle_file(tmp_path: Path, namespace: str = "bob") -> Path:
    bundle = _make_share_bundle(
        namespace=namespace,
        projects=[{"id": "proj-x", "name": "Extern", "created_at": "2024-01-01T00:00:00+00:00"}],
        conversations=[{
            "id": "conv-x1", "title": "External Chat", "project_id": "proj-x",
            "created_at": "2024-06-01T00:00:00+00:00",
            "updated_at": "2024-06-01T00:00:00+00:00",
            "messages": [
                {"id": "mx1", "role": "human", "content": "Hey Bob",
                 "timestamp": "2024-06-01T00:00:00+00:00", "position": 0},
            ],
            "decisions": [
                {"id": "dx1", "topic": "auth", "conclusion": "Use JWT",
                 "confidence": 0.8, "extracted_at": "2024-06-01T00:00:00+00:00"},
            ],
            "preferences": [],
            "tech_choices": [],
        }],
    )
    p = tmp_path / "bob.consciousness"
    p.write_bytes(bundle)
    return p


def test_share_import_loads_conversations(runner, tmp_path):
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    bundle = _make_bundle_file(tmp_path)

    result = runner.invoke(cli, ["--data-dir", str(data_dir), "share-import",
                                 str(bundle), "--namespace", "bob", "--no-rebuild"])
    assert result.exit_code == 0, result.output
    assert "1 conversation" in result.output

    db = Database(data_dir / "conversations.db").connect()
    conv = db.get_conversation("bob/conv-x1")
    assert conv is not None
    assert conv.title == "External Chat"
    assert conv.account_id == "bob"
    db.close()


def test_share_import_prefixes_ids_with_namespace(runner, tmp_path):
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    bundle = _make_bundle_file(tmp_path)

    runner.invoke(cli, ["--data-dir", str(data_dir), "share-import",
                        str(bundle), "--namespace", "bob", "--no-rebuild"])

    db = Database(data_dir / "conversations.db").connect()
    # project id is also prefixed
    projects = db.list_projects()
    assert any(p.id == "bob/proj-x" for p in projects)
    db.close()


def test_share_import_imports_decisions(runner, tmp_path):
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    bundle = _make_bundle_file(tmp_path)

    runner.invoke(cli, ["--data-dir", str(data_dir), "share-import",
                        str(bundle), "--namespace", "bob", "--no-rebuild"])

    db = Database(data_dir / "conversations.db").connect()
    decisions = db.find_active_decisions("auth")
    assert any(d.id == "bob/dx1" for d in decisions)
    db.close()


def test_share_import_is_idempotent(runner, tmp_path):
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    bundle = _make_bundle_file(tmp_path)

    for _ in range(2):
        runner.invoke(cli, ["--data-dir", str(data_dir), "share-import",
                            str(bundle), "--namespace", "bob", "--no-rebuild"])

    db = Database(data_dir / "conversations.db").connect()
    convs = db.list_conversations()
    ids = [c.id for c in convs]
    assert ids.count("bob/conv-x1") == 1
    db.close()


def test_share_import_import_excludes(runner, tmp_path):
    data_dir = tmp_path / "data"
    data_dir.mkdir()

    bundle_bytes = _make_share_bundle(
        namespace="alice",
        projects=[],
        conversations=[{
            "id": "cx", "title": "X", "project_id": None,
            "created_at": "2024-01-01T00:00:00+00:00",
            "updated_at": "2024-01-01T00:00:00+00:00",
            "messages": [
                {"id": "mx", "role": "human", "content": "hi", "timestamp": "2024-01-01T00:00:00+00:00", "position": 0},
            ],
            "decisions": [], "preferences": [], "tech_choices": [],
        }],
        shared_rules=[{"pattern": "*salary*", "rule_type": "title_glob", "created_at": "2024-01-01T00:00:00+00:00"}],
    )
    bundle_path = tmp_path / "alice.consciousness"
    bundle_path.write_bytes(bundle_bytes)

    runner.invoke(cli, ["--data-dir", str(data_dir), "share-import",
                        str(bundle_path), "--namespace", "alice", "--no-rebuild", "--import-excludes"])

    db = Database(data_dir / "conversations.db").connect()
    rules = db.list_exclude_rules()
    assert any(r.pattern == "*salary*" and r.shared for r in rules)
    db.close()


def test_share_import_invalid_bundle_exits_nonzero(runner, tmp_path):
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    bad = tmp_path / "bad.consciousness"
    # A zip with conversations.db (normal bundle format, not share format)
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr("conversations.db", b"sqlite3 data")
    bad.write_bytes(buf.getvalue())

    result = runner.invoke(cli, ["--data-dir", str(data_dir), "share-import",
                                 str(bad), "--namespace", "x", "--no-rebuild"])
    assert result.exit_code != 0
    assert "share.json" in result.output or "Invalid" in result.output


def test_share_import_namespace_isolation(runner, tmp_path):
    """Conversations from different namespaces don't collide even with same original IDs."""
    data_dir = tmp_path / "data"
    data_dir.mkdir()

    for name in ("alice", "bob"):
        b = _make_share_bundle(
            namespace=name,
            projects=[{"id": "proj-x", "name": name.capitalize(), "created_at": "2024-01-01T00:00:00+00:00"}],
            conversations=[{
                "id": "conv-x1", "title": f"{name.capitalize()} Conv", "project_id": "proj-x",
                "created_at": "2024-01-01T00:00:00+00:00", "updated_at": "2024-01-01T00:00:00+00:00",
                "messages": [
                    {"id": "m1", "role": "human", "content": name,
                     "timestamp": "2024-01-01T00:00:00+00:00", "position": 0},
                ],
                "decisions": [], "preferences": [], "tech_choices": [],
            }],
        )
        bp = tmp_path / f"{name}.consciousness"
        bp.write_bytes(b)
        runner.invoke(cli, ["--data-dir", str(data_dir), "share-import",
                            str(bp), "--namespace", name, "--no-rebuild"])

    db = Database(data_dir / "conversations.db").connect()
    alice_conv = db.get_conversation("alice/conv-x1")
    bob_conv = db.get_conversation("bob/conv-x1")
    assert alice_conv is not None
    assert bob_conv is not None
    assert alice_conv.account_id == "alice"
    assert bob_conv.account_id == "bob"
    db.close()


# ── shared exclude rules ──────────────────────────────────────────────────────


def test_exclude_add_shared_marks_rule(runner, data_dir):
    runner.invoke(cli, ["--data-dir", str(data_dir), "exclude", "add", "--title", "*secret*", "--shared"])
    db = Database(data_dir / "conversations.db").connect()
    rules = db.list_shared_exclude_rules()
    assert any(r.pattern == "*secret*" for r in rules)
    db.close()


def test_exclude_add_non_shared_not_in_shared_list(runner, data_dir):
    runner.invoke(cli, ["--data-dir", str(data_dir), "exclude", "add", "--title", "*private*"])
    db = Database(data_dir / "conversations.db").connect()
    shared = db.list_shared_exclude_rules()
    assert not any(r.pattern == "*private*" for r in shared)
    db.close()


def test_exclude_list_shows_shared_column(runner, data_dir):
    runner.invoke(cli, ["--data-dir", str(data_dir), "exclude", "add", "--title", "*team-secret*", "--shared"])
    result = runner.invoke(cli, ["--data-dir", str(data_dir), "exclude", "list"])
    assert "yes" in result.output
    assert "*team-secret*" in result.output
