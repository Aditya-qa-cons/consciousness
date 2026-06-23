"""Integration tests for the export/import/rebuild-index pipeline."""

import pytest
from click.testing import CliRunner

from consciousness.cli import cli
from consciousness.store.db import Database

pytestmark = pytest.mark.integration

_RUNNER = CliRunner(env={"CONSCIOUSNESS_FAKE_ENCODER": "1"})


@pytest.fixture
def seeded_data_dir(tmp_path):
    """A data directory pre-seeded with two conversations via CLI ingest."""
    import json
    import zipfile
    from io import BytesIO

    conversations = [
        {
            "uuid": "c1",
            "name": "Export test convo",
            "created_at": "2024-06-01T08:00:00Z",
            "updated_at": "2024-06-01T08:30:00Z",
            "account": {"uuid": "acc-1", "full_name": "Tester"},
            "project": None,
            "chat_messages": [
                {"uuid": "m1", "sender": "human", "text": "Hello world",
                 "created_at": "2024-06-01T08:00:00Z", "attachments": [], "files": []},
                {"uuid": "m2", "sender": "assistant", "text": "Hi there!",
                 "created_at": "2024-06-01T08:00:01Z", "attachments": [], "files": []},
            ],
        }
    ]
    buf = BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr("conversations.json", json.dumps(conversations))
    zip_path = tmp_path / "export.zip"
    zip_path.write_bytes(buf.getvalue())

    data_dir = tmp_path / "data"
    data_dir.mkdir()

    result = _RUNNER.invoke(cli, ["--data-dir", str(data_dir), "ingest", str(zip_path)])
    assert result.exit_code == 0, result.output

    return data_dir


def test_export_creates_file(seeded_data_dir, tmp_path):
    out_path = tmp_path / "backup.consciousness"
    result = _RUNNER.invoke(cli, ["--data-dir", str(seeded_data_dir), "export", str(out_path)])
    assert result.exit_code == 0, result.output
    assert out_path.exists()
    assert out_path.stat().st_size > 0


def test_export_is_valid_zip(seeded_data_dir, tmp_path):
    import zipfile

    out_path = tmp_path / "backup.consciousness"
    _RUNNER.invoke(cli, ["--data-dir", str(seeded_data_dir), "export", str(out_path)])

    with zipfile.ZipFile(out_path) as zf:
        assert "conversations.db" in zf.namelist()
        assert "metadata.json" in zf.namelist()


def test_import_restores_database(seeded_data_dir, tmp_path):
    bundle_path = tmp_path / "backup.consciousness"
    _RUNNER.invoke(cli, ["--data-dir", str(seeded_data_dir), "export", str(bundle_path)])

    restore_dir = tmp_path / "restored"
    restore_dir.mkdir()
    result = _RUNNER.invoke(cli, ["--data-dir", str(restore_dir), "import-bundle", str(bundle_path)])
    assert result.exit_code == 0, result.output

    db = Database(restore_dir / "conversations.db").connect()
    assert db.stats()["conversations"] == 1
    db.close()


def test_rebuild_index_repopulates_vectors(seeded_data_dir, tmp_path):
    bundle_path = tmp_path / "backup.consciousness"
    _RUNNER.invoke(cli, ["--data-dir", str(seeded_data_dir), "export", str(bundle_path)])

    restore_dir = tmp_path / "restored2"
    restore_dir.mkdir()
    result = _RUNNER.invoke(
        cli, ["--data-dir", str(restore_dir), "import-bundle", "--no-rebuild", str(bundle_path)]
    )
    assert result.exit_code == 0, result.output

    result = _RUNNER.invoke(cli, ["--data-dir", str(restore_dir), "rebuild-index"])
    assert result.exit_code == 0, result.output
    assert "chunks indexed" in result.output


def test_exclude_add_and_list(seeded_data_dir):
    result = _RUNNER.invoke(cli, ["--data-dir", str(seeded_data_dir), "exclude", "add", "--title", "*private*"])
    assert result.exit_code == 0, result.output

    result = _RUNNER.invoke(cli, ["--data-dir", str(seeded_data_dir), "exclude", "list"])
    assert result.exit_code == 0, result.output
    assert "private" in result.output


def test_exclude_remove(seeded_data_dir):
    _RUNNER.invoke(cli, ["--data-dir", str(seeded_data_dir), "exclude", "add", "--title", "*temp*"])
    result = _RUNNER.invoke(cli, ["--data-dir", str(seeded_data_dir), "exclude", "remove", "*temp*"])
    assert result.exit_code == 0, result.output

    result = _RUNNER.invoke(cli, ["--data-dir", str(seeded_data_dir), "exclude", "list"])
    assert "*temp*" not in result.output
