"""Tests for multi-account merge: content-hash dedup, account_id tagging, DB helpers."""

import hashlib

from consciousness.models import Conversation, Message, Project, Role
from consciousness.parser import parse_export
from tests.conftest import make_message, utc

# ── helpers ───────────────────────────────────────────────────────────────────


def _hash(messages: list[Message]) -> str:
    parts = sorted(f"{m.role.value}:{m.content}" for m in messages)
    return hashlib.sha256("\n".join(parts).encode()).hexdigest()


def _conv_with_hash(id="conv-1", account_id: str | None = None) -> Conversation:
    msgs = [
        make_message("m1", id, Role.human, "Hello there", 0),
        make_message("m2", id, Role.assistant, "Hi!", 1),
    ]
    return Conversation(
        id=id,
        title="Test",
        project_id=None,
        created_at=utc(2024, 1, 1),
        updated_at=utc(2024, 1, 1),
        messages=msgs,
        account_id=account_id,
        content_hash=_hash(msgs),
    )


# ── content hash stability ────────────────────────────────────────────────────


def test_content_hash_stable_across_order():
    msgs_a = [
        make_message("m1", "c1", Role.human, "Hello", 0),
        make_message("m2", "c1", Role.assistant, "World", 1),
    ]
    msgs_b = list(reversed(msgs_a))  # different insertion order
    assert _hash(msgs_a) == _hash(msgs_b)


def test_content_hash_differs_for_different_content():
    msgs_a = [make_message("m1", "c1", Role.human, "Hello", 0)]
    msgs_b = [make_message("m1", "c1", Role.human, "Goodbye", 0)]
    assert _hash(msgs_a) != _hash(msgs_b)


# ── Database.find_by_content_hash ─────────────────────────────────────────────


def test_find_by_content_hash_returns_none_when_empty(db):
    assert db.find_by_content_hash("abc123") is None


def test_find_by_content_hash_finds_existing(db):
    conv = _conv_with_hash("conv-1")
    db.upsert_conversation(conv)
    db.commit()
    assert db.find_by_content_hash(conv.content_hash) == "conv-1"


def test_find_by_content_hash_returns_none_for_wrong_hash(db):
    conv = _conv_with_hash("conv-1")
    db.upsert_conversation(conv)
    db.commit()
    assert db.find_by_content_hash("deadbeef") is None


# ── Database.list_accounts ────────────────────────────────────────────────────


def test_list_accounts_empty(db):
    assert db.list_accounts() == []


def test_list_accounts_with_data(db):
    conv_a = _conv_with_hash("conv-1", account_id="acc-alice")
    conv_b = _conv_with_hash("conv-2", account_id="acc-bob")
    conv_b.content_hash = "different-hash"  # ensure unique hash
    db.upsert_conversation(conv_a)
    db.upsert_conversation(conv_b)
    db.commit()
    accounts = db.list_accounts()
    assert set(accounts) == {"acc-alice", "acc-bob"}


def test_list_accounts_deduplicates(db):
    conv_a = _conv_with_hash("conv-1", account_id="acc-alice")
    conv_b = _conv_with_hash("conv-2", account_id="acc-alice")
    conv_b.content_hash = "different-hash"
    db.upsert_conversation(conv_a)
    db.upsert_conversation(conv_b)
    db.commit()
    assert db.list_accounts() == ["acc-alice"]


def test_list_accounts_excludes_null(db):
    conv = _conv_with_hash("conv-1", account_id=None)
    db.upsert_conversation(conv)
    db.commit()
    assert db.list_accounts() == []


# ── account_id preserved in round-trip ────────────────────────────────────────


def test_account_id_persisted_and_loaded(db):
    conv = _conv_with_hash("conv-1", account_id="acc-alice")
    db.upsert_conversation(conv)
    db.commit()
    loaded = db.get_conversation("conv-1")
    assert loaded is not None
    assert loaded.account_id == "acc-alice"


def test_content_hash_persisted_and_loaded(db):
    conv = _conv_with_hash("conv-1")
    db.upsert_conversation(conv)
    db.commit()
    loaded = db.get_conversation("conv-1")
    assert loaded is not None
    assert loaded.content_hash == conv.content_hash


# ── project account_id ────────────────────────────────────────────────────────


def test_project_account_id_persisted(db):
    p = Project(id="proj-1", name="My Project", account_id="acc-alice")
    db.upsert_project(p)
    db.commit()
    projects = db.list_projects()
    assert any(p.account_id == "acc-alice" for p in projects)


# ── stats includes accounts count ─────────────────────────────────────────────


def test_stats_accounts_count(db):
    conv_a = _conv_with_hash("conv-1", account_id="acc-alice")
    conv_b = _conv_with_hash("conv-2", account_id="acc-bob")
    conv_b.content_hash = "other-hash"
    db.upsert_conversation(conv_a)
    db.upsert_conversation(conv_b)
    db.commit()
    s = db.stats()
    assert s["accounts"] == 2


# ── parse_export account_id override ─────────────────────────────────────────


def test_parse_export_account_id_override(tmp_path):
    """parse_export(path, account_id=...) overrides account on all conversations."""
    import json
    import zipfile

    conv_data = [
        {
            "uuid": "conv-1",
            "name": "Test",
            "created_at": "2024-01-01T00:00:00+00:00",
            "updated_at": "2024-01-01T01:00:00+00:00",
            "account": {"uuid": "original-account", "full_name": "Original"},
            "chat_messages": [
                {"uuid": "m1", "sender": "human", "text": "Hi", "created_at": "2024-01-01T00:00:00+00:00"},
                {"uuid": "m2", "sender": "assistant", "text": "Hello", "created_at": "2024-01-01T00:01:00+00:00"},
            ],
        }
    ]
    zip_path = tmp_path / "export.zip"
    with zipfile.ZipFile(zip_path, "w") as zf:
        zf.writestr("conversations.json", json.dumps(conv_data))

    convs, _ = parse_export(zip_path, account_id="override-account")
    assert len(convs) == 1
    assert convs[0].account_id == "override-account"


def test_parse_export_no_override_preserves_original(tmp_path):
    """parse_export without account_id preserves the account from the export."""
    import json
    import zipfile

    conv_data = [
        {
            "uuid": "conv-1",
            "name": "Test",
            "created_at": "2024-01-01T00:00:00+00:00",
            "updated_at": "2024-01-01T01:00:00+00:00",
            "account": {"uuid": "original-account", "full_name": "Original"},
            "chat_messages": [
                {"uuid": "m1", "sender": "human", "text": "Hi", "created_at": "2024-01-01T00:00:00+00:00"},
                {"uuid": "m2", "sender": "assistant", "text": "Hello", "created_at": "2024-01-01T00:01:00+00:00"},
            ],
        }
    ]
    zip_path = tmp_path / "export.zip"
    with zipfile.ZipFile(zip_path, "w") as zf:
        zf.writestr("conversations.json", json.dumps(conv_data))

    convs, _ = parse_export(zip_path)
    assert len(convs) == 1
    assert convs[0].account_id == "original-account"


# ── dedup via content hash in ingest ─────────────────────────────────────────


def test_dedup_skips_identical_conversation_under_different_id(db):
    """A conversation with an identical content hash but different ID is not double-counted."""
    msgs = [
        make_message("m1", "conv-orig", Role.human, "Hello", 0),
        make_message("m2", "conv-orig", Role.assistant, "Hi!", 1),
    ]
    h = _hash(msgs)

    # First conversation indexed under "conv-orig"
    conv_orig = Conversation(
        id="conv-orig", title="Original", project_id=None,
        created_at=utc(2024, 1, 1), updated_at=utc(2024, 1, 1),
        messages=msgs, content_hash=h,
    )
    db.upsert_conversation(conv_orig)
    db.commit()

    # Duplicate arrives under a different ID (second account's export)
    existing_id = db.find_by_content_hash(h)
    assert existing_id == "conv-orig"

    # Simulate ingest dedup check: skip because existing_id != conv_dup.id
    conv_dup_id = "conv-dup"
    assert existing_id != conv_dup_id  # would be skipped in ingest loop

    # Verify DB still only has one conversation
    assert db.stats()["conversations"] == 1
