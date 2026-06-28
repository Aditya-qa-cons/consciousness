"""Tests for SQLite database layer — conversations, knowledge tables, exclude rules."""

from datetime import datetime, timezone

import pytest

from consciousness.models import Conversation, Decision, ExcludeRule, Message, Preference, Project, Role, TechChoice
from consciousness.store.db import Database


@pytest.fixture
def db(tmp_path):
    d = Database(tmp_path / "test.db").connect()
    yield d
    d.close()


def _utc(*args):
    return datetime(*args, tzinfo=timezone.utc)


def make_project() -> Project:
    return Project(id="proj-1", name="Test Project", created_at=_utc(2024, 1, 1))


def make_conversation(include_messages=True) -> Conversation:
    messages = []
    if include_messages:
        messages = [
            Message(
                id="msg-1", conversation_id="conv-1", role=Role.human,
                content="Hello", timestamp=_utc(2024, 6, 1, 10), position=0,
            ),
            Message(
                id="msg-2", conversation_id="conv-1", role=Role.assistant,
                content="Hi there!", timestamp=_utc(2024, 6, 1, 10, 1), position=1,
            ),
        ]
    return Conversation(
        id="conv-1",
        title="Test Conversation",
        project_id="proj-1",
        created_at=_utc(2024, 6, 1),
        updated_at=_utc(2024, 6, 1, 10, 1),
        messages=messages,
    )


# ── conversations ─────────────────────────────────────────────────────────────


def test_upsert_and_retrieve_project(db):
    db.upsert_project(make_project())
    db.commit()
    projects = db.list_projects()
    assert len(projects) == 1
    assert projects[0].name == "Test Project"


def test_upsert_conversation_with_messages(db):
    db.upsert_project(make_project())
    db.upsert_conversation(make_conversation())
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
    assert len(db.list_conversations(project_id="proj-1")) == 1
    assert len(db.list_conversations(project_id="other-proj")) == 0


def test_stats_includes_knowledge_counts(db):
    db.upsert_project(make_project())
    db.upsert_conversation(make_conversation())
    db.commit()
    s = db.stats()
    assert s["conversations"] == 1
    assert s["messages"] == 2
    assert s["projects"] == 1
    assert s["decisions"] == 0
    assert s["tech_choices"] == 0


def test_get_nonexistent_conversation(db):
    assert db.get_conversation("nonexistent") is None


# ── decisions ─────────────────────────────────────────────────────────────────


def test_upsert_and_find_decision(db):
    db.upsert_project(make_project())
    db.upsert_conversation(make_conversation())
    d = Decision(
        id="d1", topic="Postgres", conclusion="Use Postgres for production.",
        confidence=0.9, conversation_id="conv-1",
    )
    db.upsert_decision(d)
    db.commit()

    found = db.find_active_decisions("postgres")
    assert len(found) == 1
    assert found[0].topic == "Postgres"


def test_find_decisions_partial_match(db):
    db.upsert_project(make_project())
    db.upsert_conversation(make_conversation())
    db.upsert_decision(Decision(
        id="d1", topic="database choice", conclusion="Use Postgres.",
        confidence=0.8, conversation_id="conv-1",
    ))
    db.commit()

    assert len(db.find_active_decisions("database")) == 1
    assert len(db.find_active_decisions("auth")) == 0


def test_supersede_decision(db):
    db.upsert_project(make_project())
    db.upsert_conversation(make_conversation())
    db.upsert_decision(Decision(
        id="old", topic="Postgres", conclusion="Use Postgres.",
        confidence=0.8, conversation_id="conv-1",
    ))
    db.upsert_decision(Decision(
        id="new", topic="Postgres", conclusion="Use Postgres with pgBouncer.",
        confidence=0.9, conversation_id="conv-1",
    ))
    db.supersede_decision("old", "new")
    db.commit()

    active = db.find_active_decisions("postgres")
    ids = [d.id for d in active]
    assert "new" in ids
    assert "old" not in ids


def test_list_decisions_excludes_superseded(db):
    db.upsert_project(make_project())
    db.upsert_conversation(make_conversation())
    db.upsert_decision(Decision(
        id="old", topic="Auth", conclusion="Use sessions.",
        confidence=0.7, conversation_id="conv-1",
    ))
    db.upsert_decision(Decision(
        id="new", topic="Auth", conclusion="Use JWT.",
        confidence=0.9, conversation_id="conv-1",
    ))
    db.supersede_decision("old", "new")
    db.commit()

    all_active = db.list_decisions()
    assert not any(d.id == "old" for d in all_active)


# ── preferences ───────────────────────────────────────────────────────────────


def test_upsert_and_list_preference(db):
    db.upsert_project(make_project())
    db.upsert_conversation(make_conversation())
    pref = Preference(
        id="p1", area="TypeScript",
        preference="I prefer TypeScript over JavaScript.", conversation_id="conv-1",
    )
    db.upsert_preference(pref)
    db.commit()

    prefs = db.list_preferences()
    assert len(prefs) == 1
    assert prefs[0].area == "TypeScript"


# ── tech choices ──────────────────────────────────────────────────────────────


def test_upsert_and_list_tech_choice(db):
    db.upsert_project(make_project())
    db.upsert_conversation(make_conversation())
    tc = TechChoice(id="tc1", technology="Redis", verdict="Use Redis for caching.", conversation_id="conv-1")
    db.upsert_tech_choice(tc)
    db.commit()

    choices = db.list_tech_choices()
    assert len(choices) == 1
    assert choices[0].technology == "Redis"


# ── exclude rules ─────────────────────────────────────────────────────────────


def test_add_and_list_exclude_rule(db):
    rule = ExcludeRule(pattern="conv-secret", rule_type="conversation_id")
    db.add_exclude_rule(rule)
    db.commit()

    rules = db.list_exclude_rules()
    assert len(rules) == 1
    assert rules[0].pattern == "conv-secret"


def test_remove_exclude_rule(db):
    db.add_exclude_rule(ExcludeRule(pattern="*private*", rule_type="title_glob"))
    db.commit()
    db.remove_exclude_rule("*private*")
    db.commit()
    assert len(db.list_exclude_rules()) == 0


def test_is_excluded_by_conversation_id(db):
    db.upsert_project(make_project())
    conv = make_conversation()
    db.upsert_conversation(conv)
    db.add_exclude_rule(ExcludeRule(pattern="conv-1", rule_type="conversation_id"))
    db.commit()
    assert db.is_excluded(conv) is True


def test_is_excluded_by_title_glob(db):
    db.upsert_project(make_project())
    conv = make_conversation()  # title = "Test Conversation"
    db.upsert_conversation(conv)
    db.add_exclude_rule(ExcludeRule(pattern="*test*", rule_type="title_glob"))
    db.commit()
    assert db.is_excluded(conv) is True


def test_not_excluded_when_no_rules_match(db):
    db.upsert_project(make_project())
    conv = make_conversation()
    db.upsert_conversation(conv)
    db.add_exclude_rule(ExcludeRule(pattern="other-conv", rule_type="conversation_id"))
    db.commit()
    assert db.is_excluded(conv) is False


# ── incremental ingest helpers ────────────────────────────────────────────────


def test_get_conversation_updated_at_returns_none_for_missing(db):
    assert db.get_conversation_updated_at("nonexistent") is None


def test_get_conversation_updated_at_returns_timestamp(db):
    db.upsert_project(make_project())
    db.upsert_conversation(make_conversation())
    db.commit()
    ts = db.get_conversation_updated_at("conv-1")
    assert ts == _utc(2024, 6, 1, 10, 1)


def test_delete_knowledge_for_conversation(db):
    db.upsert_project(make_project())
    db.upsert_conversation(make_conversation())
    db.upsert_decision(Decision(
        id="d1", topic="Auth", conclusion="Use JWT.", confidence=0.9, conversation_id="conv-1",
    ))
    db.upsert_preference(Preference(
        id="p1", area="Python", preference="I prefer Python.", conversation_id="conv-1",
    ))
    db.upsert_tech_choice(TechChoice(
        id="tc1", technology="Redis", verdict="Use Redis.", conversation_id="conv-1",
    ))
    db.commit()

    assert db.stats()["decisions"] == 1
    assert len(db.list_preferences()) == 1
    assert len(db.list_tech_choices()) == 1

    db.delete_knowledge_for_conversation("conv-1")
    db.commit()

    assert db.stats()["decisions"] == 0
    assert len(db.list_preferences()) == 0
    assert len(db.list_tech_choices()) == 0


def test_delete_knowledge_leaves_other_conversations_untouched(db):
    db.upsert_project(make_project())
    db.upsert_conversation(make_conversation())
    db.upsert_project(Project(id="proj-2", name="Other Project", created_at=_utc(2024, 1, 1)))
    conv2 = Conversation(
        id="conv-2", title="Other", project_id="proj-2",
        created_at=_utc(2024, 6, 2), updated_at=_utc(2024, 6, 2),
    )
    db.upsert_conversation(conv2)
    db.upsert_decision(Decision(
        id="d1", topic="Auth", conclusion="Use JWT.", confidence=0.9, conversation_id="conv-1",
    ))
    db.upsert_decision(Decision(
        id="d2", topic="DB", conclusion="Use Postgres.", confidence=0.9, conversation_id="conv-2",
    ))
    db.commit()

    db.delete_knowledge_for_conversation("conv-1")
    db.commit()

    remaining = db.list_decisions()
    assert len(remaining) == 1
    assert remaining[0].id == "d2"


# ── full-text search ──────────────────────────────────────────────────────────


def test_fulltext_search_finds_exact_word(db):
    db.upsert_project(make_project())
    db.upsert_conversation(make_conversation())
    db.commit()
    results = db.fulltext_search("Hello")
    assert len(results) > 0
    assert any(r["conversation_id"] == "conv-1" for r in results)


def test_fulltext_search_returns_snippet(db):
    db.upsert_project(make_project())
    db.upsert_conversation(make_conversation())
    db.commit()
    results = db.fulltext_search("Hello")
    assert results[0]["snippet"] != ""


def test_fulltext_search_no_match_returns_empty(db):
    db.upsert_project(make_project())
    db.upsert_conversation(make_conversation())
    db.commit()
    results = db.fulltext_search("xyzzy_nonexistent_word")
    assert results == []


def test_fulltext_search_role_filter(db):
    db.upsert_project(make_project())
    db.upsert_conversation(make_conversation())
    db.commit()
    human_results = db.fulltext_search("Hello", role="human")
    assistant_results = db.fulltext_search("Hello", role="assistant")
    assert all(r["role"] == "human" for r in human_results)
    assert all(r["role"] == "assistant" for r in assistant_results)


def test_fulltext_search_conversation_id_filter(db):
    db.upsert_project(make_project())
    db.upsert_conversation(make_conversation())
    db.commit()
    results = db.fulltext_search("Hello", conversation_ids=["conv-1"])
    assert all(r["conversation_id"] == "conv-1" for r in results)
    results_other = db.fulltext_search("Hello", conversation_ids=["other-conv"])
    assert results_other == []


def test_rebuild_fts_repopulates_from_messages(db):
    db.upsert_project(make_project())
    db.upsert_conversation(make_conversation())
    db.commit()
    # Manually wipe FTS to simulate a pre-FTS install
    db.conn.execute("DELETE FROM messages_fts")
    db.commit()
    assert db.fulltext_search("Hello") == []

    db.rebuild_fts()
    db.commit()
    assert len(db.fulltext_search("Hello")) > 0


# ── config ─────────────────────────────────────────────────────────────────────


def test_get_config_returns_none_for_missing_key(db):
    assert db.get_config("nonexistent") is None


def test_set_and_get_config_roundtrip(db):
    db.set_config("last_ingested_at", "2024-06-01T10:00:00+00:00")
    db.commit()
    assert db.get_config("last_ingested_at") == "2024-06-01T10:00:00+00:00"


def test_set_config_overwrites_existing_value(db):
    db.set_config("key", "old")
    db.set_config("key", "new")
    db.commit()
    assert db.get_config("key") == "new"


def test_set_config_multiple_keys_are_independent(db):
    db.set_config("a", "alpha")
    db.set_config("b", "beta")
    db.commit()
    assert db.get_config("a") == "alpha"
    assert db.get_config("b") == "beta"
