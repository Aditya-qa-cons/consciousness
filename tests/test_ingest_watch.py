"""Tests for incremental / watch-mode ingest and directory scanning."""

import json
import os
import time
import zipfile
from datetime import datetime, timedelta, timezone
from io import BytesIO
from pathlib import Path
from unittest.mock import patch

import pytest
from click.testing import CliRunner

from consciousness.cli import cli
from consciousness.store.db import Database

# ── helpers ────────────────────────────────────────────────────────────────────

_SAMPLE_CONV = [
    {
        "uuid": "conv-watch-1",
        "name": "Watch test conversation",
        "created_at": "2024-06-01T10:00:00.000Z",
        "updated_at": "2024-06-01T10:30:00.000Z",
        "account": {"uuid": "acc-1", "full_name": "Test User"},
        "project": None,
        "chat_messages": [
            {
                "uuid": "msg-w1",
                "sender": "human",
                "text": "Hello from watch test",
                "created_at": "2024-06-01T10:00:00.000Z",
                "attachments": [],
                "files": [],
            },
            {
                "uuid": "msg-w2",
                "sender": "assistant",
                "text": "Watch mode is working.",
                "created_at": "2024-06-01T10:00:05.000Z",
                "attachments": [],
                "files": [],
            },
        ],
    }
]

_SAMPLE_CONV_2 = [
    {
        "uuid": "conv-watch-2",
        "name": "Second watch conversation",
        "created_at": "2024-06-03T08:00:00.000Z",
        "updated_at": "2024-06-03T08:15:00.000Z",
        "account": {"uuid": "acc-1", "full_name": "Test User"},
        "project": None,
        "chat_messages": [
            {
                "uuid": "msg-w3",
                "sender": "human",
                "text": "Second file ingested",
                "created_at": "2024-06-03T08:00:00.000Z",
                "attachments": [],
                "files": [],
            },
        ],
    }
]


def _make_zip(conversations: list[dict]) -> bytes:
    buf = BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr("conversations.json", json.dumps(conversations))
    return buf.getvalue()


@pytest.fixture
def runner():
    return CliRunner(env={"CONSCIOUSNESS_FAKE_ENCODER": "1"})


@pytest.fixture
def data_dir(tmp_path) -> Path:
    d = tmp_path / "data"
    d.mkdir()
    return d


@pytest.fixture
def export_zip(tmp_path) -> Path:
    p = tmp_path / "export.zip"
    p.write_bytes(_make_zip(_SAMPLE_CONV))
    return p


# ── single-file ingest ────────────────────────────────────────────────────────


def test_ingest_sets_last_ingested_at(runner, data_dir, export_zip):
    result = runner.invoke(cli, ["--data-dir", str(data_dir), "ingest", str(export_zip)])
    assert result.exit_code == 0, result.output

    db = Database(data_dir / "conversations.db").connect()
    val = db.get_config("last_ingested_at")
    db.close()

    assert val is not None
    ts = datetime.fromisoformat(val)
    assert ts.tzinfo is not None
    # timestamp should be within the last minute
    assert datetime.now(timezone.utc) - ts < timedelta(minutes=1)


def test_ingest_last_ingested_at_updates_on_second_run(runner, data_dir, export_zip):
    runner.invoke(cli, ["--data-dir", str(data_dir), "ingest", str(export_zip)])
    db = Database(data_dir / "conversations.db").connect()
    first_ts = datetime.fromisoformat(db.get_config("last_ingested_at"))
    db.close()

    # Small sleep so second timestamp is strictly later
    time.sleep(0.05)
    runner.invoke(cli, ["--data-dir", str(data_dir), "ingest", str(export_zip)])
    db = Database(data_dir / "conversations.db").connect()
    second_ts = datetime.fromisoformat(db.get_config("last_ingested_at"))
    db.close()

    assert second_ts > first_ts


# ── directory mode ────────────────────────────────────────────────────────────


def test_ingest_directory_ingests_all_zips(runner, data_dir, tmp_path):
    export_dir = tmp_path / "exports"
    export_dir.mkdir()
    (export_dir / "a.zip").write_bytes(_make_zip(_SAMPLE_CONV))
    (export_dir / "b.zip").write_bytes(_make_zip(_SAMPLE_CONV_2))

    result = runner.invoke(cli, ["--data-dir", str(data_dir), "ingest", str(export_dir)])
    assert result.exit_code == 0, result.output

    db = Database(data_dir / "conversations.db").connect()
    assert db.stats()["conversations"] == 2
    db.close()


def test_ingest_directory_empty_exits_nonzero(runner, data_dir, tmp_path):
    export_dir = tmp_path / "empty"
    export_dir.mkdir()

    result = runner.invoke(cli, ["--data-dir", str(data_dir), "ingest", str(export_dir)])
    assert result.exit_code == 1
    assert "No ZIP files found" in result.output


def test_ingest_directory_skips_zips_older_than_last_ingested_at(runner, data_dir, tmp_path):
    export_dir = tmp_path / "exports"
    export_dir.mkdir()
    zip_path = export_dir / "old.zip"
    zip_path.write_bytes(_make_zip(_SAMPLE_CONV))

    # Wind the ZIP's mtime back 2 hours so it appears old
    old_mtime = time.time() - 7200
    os.utime(zip_path, (old_mtime, old_mtime))

    # Store last_ingested_at as 1 hour ago — newer than the ZIP's mtime
    db = Database(data_dir / "conversations.db").connect()
    cutoff = (datetime.now(timezone.utc) - timedelta(hours=1)).isoformat()
    db.set_config("last_ingested_at", cutoff)
    db.commit()
    db.close()

    result = runner.invoke(cli, ["--data-dir", str(data_dir), "ingest", str(export_dir)])
    assert result.exit_code == 1
    assert "No ZIP files" in result.output


def test_ingest_directory_processes_new_zip_after_cutoff(runner, data_dir, tmp_path):
    export_dir = tmp_path / "exports"
    export_dir.mkdir()

    # Old ZIP — will be skipped
    old_zip = export_dir / "old.zip"
    old_zip.write_bytes(_make_zip(_SAMPLE_CONV))
    old_mtime = time.time() - 7200
    os.utime(old_zip, (old_mtime, old_mtime))

    # New ZIP — will be processed (mtime = now, newer than the cutoff)
    new_zip = export_dir / "new.zip"
    new_zip.write_bytes(_make_zip(_SAMPLE_CONV_2))

    # Set cutoff to 1 hour ago — old ZIP is excluded, new ZIP is not
    db = Database(data_dir / "conversations.db").connect()
    db.set_config("last_ingested_at", (datetime.now(timezone.utc) - timedelta(hours=1)).isoformat())
    db.commit()
    db.close()

    result = runner.invoke(cli, ["--data-dir", str(data_dir), "ingest", str(export_dir)])
    assert result.exit_code == 0, result.output

    db = Database(data_dir / "conversations.db").connect()
    # Only conv-watch-2 (from new.zip) should be indexed
    assert db.stats()["conversations"] == 1
    assert db.get_conversation("conv-watch-2") is not None
    db.close()


# ── watch mode ────────────────────────────────────────────────────────────────


def test_watch_mode_loops_and_stops_on_ctrl_c(runner, data_dir, export_zip):
    """Watch mode runs a second iteration and exits cleanly on KeyboardInterrupt."""
    call_count = 0

    def fake_sleep(n):
        nonlocal call_count
        call_count += 1
        if call_count >= 1:
            raise KeyboardInterrupt

    with patch("consciousness.cli.time.sleep", side_effect=fake_sleep):
        result = runner.invoke(
            cli,
            ["--data-dir", str(data_dir), "ingest", str(export_zip), "--watch", "--interval", "1"],
        )

    assert "Watch stopped" in result.output
    assert call_count == 1


def test_watch_mode_directory_no_new_zips_sleeps_then_stops(runner, data_dir, tmp_path):
    """When no new ZIPs exist, watch mode sleeps and then stops on KeyboardInterrupt."""
    export_dir = tmp_path / "exports"
    export_dir.mkdir()
    zip_path = export_dir / "old.zip"
    zip_path.write_bytes(_make_zip(_SAMPLE_CONV))

    # Mark everything as already ingested (1 hour ago, ZIP is 2 hours old)
    old_mtime = time.time() - 7200
    os.utime(zip_path, (old_mtime, old_mtime))
    db = Database(data_dir / "conversations.db").connect()
    db.set_config("last_ingested_at", (datetime.now(timezone.utc) - timedelta(hours=1)).isoformat())
    db.commit()
    db.close()

    def fake_sleep(_):
        raise KeyboardInterrupt

    with patch("consciousness.cli.time.sleep", side_effect=fake_sleep):
        result = runner.invoke(
            cli,
            ["--data-dir", str(data_dir), "ingest", str(export_dir), "--watch", "--interval", "60"],
        )

    assert "Watch stopped" in result.output
