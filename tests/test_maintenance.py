"""Tests for automatic memory maintenance — stale decisions, conflicts, digest."""

import json
from datetime import datetime, timedelta, timezone

import pytest
from click.testing import CliRunner

from consciousness.cli import cli
from consciousness.models import Decision, TechChoice
from consciousness.store.db import Database


def _utc(*args) -> datetime:
    return datetime(*args, tzinfo=timezone.utc)


def _ago(days: int) -> datetime:
    return datetime.now(timezone.utc) - timedelta(days=days)


@pytest.fixture
def db(tmp_path) -> Database:
    """Empty database with a single seeded conversation so FK constraints pass."""

    d = Database(tmp_path / "test.db").connect()
    d.conn.execute(
        "INSERT OR IGNORE INTO projects(id, name) VALUES ('proj-test', 'Test')"
    )
    d.conn.execute(
        "INSERT OR IGNORE INTO conversations(id, title, project_id, created_at, updated_at)"
        " VALUES ('conv-1', 'Test', 'proj-test', '2024-01-01T00:00:00+00:00', '2024-01-01T00:00:00+00:00')"
    )
    d.conn.commit()
    yield d
    d.close()


@pytest.fixture
def runner():
    return CliRunner(env={"CONSCIOUSNESS_FAKE_ENCODER": "1"})


def _make_decision(
    id: str,
    topic: str,
    conclusion: str,
    extracted_at: datetime,
    superseded_by: str | None = None,
    conversation_id: str = "conv-1",
) -> Decision:
    return Decision(
        id=id, topic=topic, conclusion=conclusion, confidence=0.8,
        conversation_id=conversation_id, extracted_at=extracted_at,
        superseded_by=superseded_by,
    )


def _make_tc(id: str, technology: str, verdict: str, extracted_at: datetime) -> TechChoice:
    return TechChoice(
        id=id, technology=technology, verdict=verdict,
        rationale="test", conversation_id="conv-1", extracted_at=extracted_at,
    )


# ── list_stale_decisions ──────────────────────────────────────────────────────


def test_stale_decisions_returns_old_active_decisions(db):
    old = _make_decision("d1", "database", "Use Postgres", _ago(120))
    db.upsert_decision(old)
    db.commit()

    stale = db.list_stale_decisions(older_than_days=90)
    assert len(stale) == 1
    assert stale[0].id == "d1"


def test_stale_decisions_excludes_recent_decisions(db):
    recent = _make_decision("d1", "database", "Use Postgres", _ago(10))
    db.upsert_decision(recent)
    db.commit()

    assert db.list_stale_decisions(older_than_days=90) == []


def test_stale_decisions_excludes_superseded(db):
    old = _make_decision("d1", "database", "Use SQLite", _ago(120))
    newer = _make_decision("d2", "database", "Use Postgres", _ago(10))
    db.upsert_decision(old)
    db.upsert_decision(newer)
    db.supersede_decision("d1", "d2")
    db.commit()

    # Only the active (non-superseded) decision should appear, and d2 is recent
    assert db.list_stale_decisions(older_than_days=90) == []


def test_stale_decisions_ordered_oldest_first(db):
    db.upsert_decision(_make_decision("d1", "auth", "Use sessions", _ago(200)))
    db.upsert_decision(_make_decision("d2", "cache", "Use Redis", _ago(150)))
    db.commit()

    stale = db.list_stale_decisions(older_than_days=90)
    assert stale[0].id == "d1"
    assert stale[1].id == "d2"


def test_stale_decisions_custom_threshold(db):
    db.upsert_decision(_make_decision("d1", "db", "Postgres", _ago(30)))
    db.commit()

    # 90-day threshold → not stale
    assert db.list_stale_decisions(older_than_days=90) == []
    # 20-day threshold → stale
    assert len(db.list_stale_decisions(older_than_days=20)) == 1


# ── find_conflicting_decisions ────────────────────────────────────────────────


def test_no_conflicts_when_topics_differ(db):
    db.upsert_decision(_make_decision("d1", "database", "Use Postgres", _ago(10)))
    db.upsert_decision(_make_decision("d2", "auth", "Use sessions", _ago(10)))
    db.commit()

    assert db.find_conflicting_decisions() == []


def test_conflict_detected_for_same_topic(db):
    db.upsert_decision(_make_decision("d1", "database", "Use Postgres", _ago(20)))
    db.upsert_decision(_make_decision("d2", "database", "Use SQLite instead", _ago(5)))
    db.commit()

    pairs = db.find_conflicting_decisions()
    assert len(pairs) == 1
    ids = {pairs[0][0].id, pairs[0][1].id}
    assert ids == {"d1", "d2"}


def test_conflict_detection_is_case_insensitive(db):
    db.upsert_decision(_make_decision("d1", "Database", "Use Postgres", _ago(20)))
    db.upsert_decision(_make_decision("d2", "database", "Use MySQL", _ago(5)))
    db.commit()

    assert len(db.find_conflicting_decisions()) == 1


def test_no_conflict_if_one_is_superseded(db):
    db.upsert_decision(_make_decision("d1", "database", "Use SQLite", _ago(20)))
    db.upsert_decision(_make_decision("d2", "database", "Use Postgres", _ago(5)))
    db.supersede_decision("d1", "d2")
    db.commit()

    # d1 is superseded so not active — no conflict
    assert db.find_conflicting_decisions() == []


def test_three_decisions_same_topic_returns_all_pairs(db):
    db.upsert_decision(_make_decision("d1", "cache", "Use Redis", _ago(30)))
    db.upsert_decision(_make_decision("d2", "cache", "Use Memcached", _ago(20)))
    db.upsert_decision(_make_decision("d3", "cache", "No cache needed", _ago(10)))
    db.commit()

    pairs = db.find_conflicting_decisions()
    assert len(pairs) == 3  # (d1,d2), (d1,d3), (d2,d3)


# ── recent_changes ────────────────────────────────────────────────────────────


def test_recent_changes_empty_when_no_activity(db):
    result = db.recent_changes(days=7)
    assert result["conversations"] == []
    assert result["decisions"] == []
    assert result["tech_choices"] == []


def test_recent_changes_includes_recent_decisions(db):
    db.upsert_decision(_make_decision("d1", "database", "Use Postgres", _ago(2)))
    db.upsert_decision(_make_decision("d2", "auth", "Use sessions", _ago(30)))
    db.commit()

    result = db.recent_changes(days=7)
    assert len(result["decisions"]) == 1
    assert result["decisions"][0].id == "d1"


def test_recent_changes_includes_recent_tech_choices(db):
    db.upsert_tech_choice(_make_tc("tc1", "Postgres", "preferred", _ago(3)))
    db.upsert_tech_choice(_make_tc("tc2", "Oracle", "avoid", _ago(100)))
    db.commit()

    result = db.recent_changes(days=7)
    assert len(result["tech_choices"]) == 1
    assert result["tech_choices"][0].technology == "Postgres"


def test_recent_changes_custom_window(db):
    db.upsert_decision(_make_decision("d1", "db", "Postgres", _ago(10)))
    db.commit()

    assert len(db.recent_changes(days=7)["decisions"]) == 0
    assert len(db.recent_changes(days=14)["decisions"]) == 1


# ── maintenance CLI command ───────────────────────────────────────────────────


@pytest.fixture
def seeded_data_dir(tmp_path):
    """Data dir with a mix of stale, recent, and conflicting decisions."""
    data_dir = tmp_path / "data"
    data_dir.mkdir()

    db = Database(data_dir / "conversations.db").connect()
    # Seed minimal conversation so FK constraints pass
    db.conn.execute(
        "INSERT OR IGNORE INTO projects(id, name) VALUES ('proj-test', 'Test')"
    )
    db.conn.execute(
        "INSERT OR IGNORE INTO conversations(id, title, project_id, created_at, updated_at)"
        " VALUES ('conv-1', 'Test', 'proj-test', '2024-01-01T00:00:00+00:00', '2024-01-01T00:00:00+00:00')"
    )
    db.conn.commit()
    db.upsert_decision(_make_decision("d-stale", "deployment", "Deploy to Heroku", _ago(180)))
    db.upsert_decision(_make_decision("d-recent", "testing", "Use pytest", _ago(3)))
    db.upsert_decision(_make_decision("d-conf-a", "database", "Use Postgres", _ago(50)))
    db.upsert_decision(_make_decision("d-conf-b", "database", "Use SQLite", _ago(40)))
    db.upsert_tech_choice(_make_tc("tc1", "pytest", "preferred", _ago(3)))
    db.commit()
    db.close()

    return data_dir


def test_maintenance_command_exits_zero(runner, seeded_data_dir):
    result = runner.invoke(cli, ["--data-dir", str(seeded_data_dir), "maintenance"])
    assert result.exit_code == 0, result.output


def test_maintenance_reports_stale_section(runner, seeded_data_dir):
    result = runner.invoke(cli, ["--data-dir", str(seeded_data_dir), "maintenance", "--stale-days", "90"])
    assert "Stale Decisions" in result.output
    assert "deployment" in result.output.lower() or "Heroku" in result.output


def test_maintenance_reports_conflict_section(runner, seeded_data_dir):
    result = runner.invoke(cli, ["--data-dir", str(seeded_data_dir), "maintenance"])
    assert "Potential Memory Conflicts" in result.output
    assert "database" in result.output.lower()


def test_maintenance_reports_digest_section(runner, seeded_data_dir):
    result = runner.invoke(cli, ["--data-dir", str(seeded_data_dir), "maintenance", "--digest-days", "7"])
    assert "Recent Activity" in result.output
    assert "testing" in result.output.lower() or "pytest" in result.output.lower()


def test_maintenance_json_output_is_valid_json(runner, seeded_data_dir):
    result = runner.invoke(cli, ["--data-dir", str(seeded_data_dir), "maintenance", "--json"])
    assert result.exit_code == 0, result.output
    data = json.loads(result.output)
    assert "stale_decisions" in data
    assert "conflicting_pairs" in data
    assert "recent_changes" in data


def test_maintenance_json_stale_decisions_have_expected_keys(runner, seeded_data_dir):
    result = runner.invoke(cli, ["--data-dir", str(seeded_data_dir), "maintenance", "--json", "--stale-days", "90"])
    data = json.loads(result.output)
    for d in data["stale_decisions"]:
        assert "id" in d
        assert "topic" in d
        assert "conclusion" in d
        assert "extracted_at" in d


def test_maintenance_json_conflicts_have_expected_keys(runner, seeded_data_dir):
    result = runner.invoke(cli, ["--data-dir", str(seeded_data_dir), "maintenance", "--json"])
    data = json.loads(result.output)
    for pair in data["conflicting_pairs"]:
        assert "decision_a" in pair
        assert "decision_b" in pair
        assert pair["decision_a"]["topic"].lower() == pair["decision_b"]["topic"].lower()


def test_maintenance_no_data_exits_nonzero(runner, tmp_path):
    empty_dir = tmp_path / "empty"
    empty_dir.mkdir()
    result = runner.invoke(cli, ["--data-dir", str(empty_dir), "maintenance"])
    assert result.exit_code == 1
    assert "No data found" in result.output


def test_maintenance_all_green_when_no_stale_or_conflicts(runner, tmp_path):
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    db = Database(data_dir / "conversations.db").connect()
    db.conn.execute("INSERT OR IGNORE INTO projects(id, name) VALUES ('p', 'P')")
    db.conn.execute(
        "INSERT OR IGNORE INTO conversations(id, title, project_id, created_at, updated_at)"
        " VALUES ('conv-1', 'T', 'p', '2024-01-01T00:00:00+00:00', '2024-01-01T00:00:00+00:00')"
    )
    db.conn.commit()
    db.upsert_decision(_make_decision("d1", "testing", "Use pytest", _ago(5)))
    db.commit()
    db.close()

    result = runner.invoke(cli, ["--data-dir", str(data_dir), "maintenance", "--stale-days", "90"])
    assert result.exit_code == 0
    assert "None" in result.output  # both stale and conflicts show "None"
