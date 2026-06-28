"""Tests for the extractor plugin system — Protocol, registry, built-in plugin, CLI."""

import warnings
from datetime import datetime, timezone

import pytest
from click.testing import CliRunner

from consciousness.cli import cli
from consciousness.extractors.base import ExtractorPlugin, ExtractorResult
from consciousness.extractors.knowledge import RegexExtractor
from consciousness.extractors.registry import ENTRY_POINT_GROUP, load_plugins, run_plugins
from consciousness.models import Conversation, Decision, Message, Preference, Role, TechChoice


def _utc(*args) -> datetime:
    return datetime(*args, tzinfo=timezone.utc)


def _conv(assistant_text="Use Postgres for production.", human_text="Which database?") -> Conversation:
    return Conversation(
        id="conv-1", title="DB choice", project_id="proj-1",
        created_at=_utc(2024, 6, 1), updated_at=_utc(2024, 6, 1),
        messages=[
            Message(id="m1", conversation_id="conv-1", role=Role.human,
                    content=human_text, timestamp=_utc(2024, 6, 1), position=0),
            Message(id="m2", conversation_id="conv-1", role=Role.assistant,
                    content=assistant_text, timestamp=_utc(2024, 6, 1, 0, 1), position=1),
        ],
    )


@pytest.fixture
def runner():
    return CliRunner(env={"CONSCIOUSNESS_FAKE_ENCODER": "1"})


# ── ExtractorResult ───────────────────────────────────────────────────────────


def test_extractor_result_defaults_to_empty_lists():
    r = ExtractorResult()
    assert r.decisions == []
    assert r.preferences == []
    assert r.tech_choices == []


def test_extractor_result_accepts_initial_values():
    d = Decision(id="d1", topic="db", conclusion="Postgres",
                 confidence=0.9, conversation_id="conv-1")
    r = ExtractorResult(decisions=[d])
    assert len(r.decisions) == 1


# ── ExtractorPlugin Protocol ──────────────────────────────────────────────────


def test_extractor_plugin_protocol_is_runtime_checkable():
    class Good:
        name = "good"
        def extract(self, conv): return ExtractorResult()

    assert isinstance(Good(), ExtractorPlugin)


def test_class_missing_extract_is_not_a_plugin():
    class Bad:
        name = "bad"

    assert not isinstance(Bad(), ExtractorPlugin)


def test_class_missing_name_is_not_a_plugin():
    class Bad:
        def extract(self, conv): return ExtractorResult()

    assert not isinstance(Bad(), ExtractorPlugin)


# ── RegexExtractor (built-in) ─────────────────────────────────────────────────


def test_regex_extractor_name():
    assert RegexExtractor.name == "regex"


def test_regex_extractor_implements_protocol():
    assert isinstance(RegexExtractor(), ExtractorPlugin)


def test_regex_extractor_returns_extractor_result():
    result = RegexExtractor().extract(_conv())
    assert isinstance(result, ExtractorResult)


def test_regex_extractor_finds_decisions():
    conv = _conv(assistant_text="Use Postgres for production.")
    result = RegexExtractor().extract(conv)
    assert len(result.decisions) >= 1
    assert any("Postgres" in d.conclusion for d in result.decisions)


def test_regex_extractor_finds_preferences():
    conv = _conv(human_text="I prefer TypeScript over JavaScript.")
    result = RegexExtractor().extract(conv)
    assert len(result.preferences) >= 1


# ── load_plugins ──────────────────────────────────────────────────────────────


def test_load_plugins_returns_list():
    plugins = load_plugins()
    assert isinstance(plugins, list)


def test_load_plugins_includes_built_in_regex():
    plugins = load_plugins()
    names = [p.name for p in plugins]
    assert "regex" in names, f"Expected 'regex' in {names}"


def test_entry_point_group_constant():
    assert ENTRY_POINT_GROUP == "consciousness.extractors"


# ── run_plugins ───────────────────────────────────────────────────────────────


def test_run_plugins_merges_decisions_from_multiple_plugins():
    d1 = Decision(id="d1", topic="db", conclusion="Use Postgres",
                  confidence=0.9, conversation_id="conv-1")
    d2 = Decision(id="d2", topic="cache", conclusion="Use Redis",
                  confidence=0.8, conversation_id="conv-1")

    class PluginA:
        name = "a"
        def extract(self, conv): return ExtractorResult(decisions=[d1])

    class PluginB:
        name = "b"
        def extract(self, conv): return ExtractorResult(decisions=[d2])

    result = run_plugins([PluginA(), PluginB()], _conv())
    assert len(result.decisions) == 2


def test_run_plugins_merges_preferences():
    p = Preference(id="p1", area="TypeScript", preference="I prefer TypeScript.",
                   conversation_id="conv-1")

    class Plugin:
        name = "prefs"
        def extract(self, conv): return ExtractorResult(preferences=[p])

    result = run_plugins([Plugin(), Plugin()], _conv())
    assert len(result.preferences) == 2


def test_run_plugins_merges_tech_choices():
    tc = TechChoice(id="tc1", technology="Redis", verdict="preferred", conversation_id="conv-1")

    class Plugin:
        name = "tc"
        def extract(self, conv): return ExtractorResult(tech_choices=[tc])

    result = run_plugins([Plugin()], _conv())
    assert len(result.tech_choices) == 1


def test_run_plugins_skips_broken_plugin_with_warning():
    class Broken:
        name = "broken"
        def extract(self, conv): raise RuntimeError("oops")

    class Good:
        name = "good"
        def extract(self, conv):
            return ExtractorResult(decisions=[
                Decision(id="d1", topic="db", conclusion="Postgres",
                         confidence=0.9, conversation_id="conv-1")
            ])

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        result = run_plugins([Broken(), Good()], _conv())

    assert len(result.decisions) == 1
    assert any("broken" in str(warning.message).lower() for warning in w)


def test_run_plugins_empty_list_returns_empty_result():
    result = run_plugins([], _conv())
    assert result.decisions == []
    assert result.preferences == []
    assert result.tech_choices == []


def test_run_plugins_with_real_built_in():
    plugins = load_plugins()
    result = run_plugins(plugins, _conv("Use Postgres for production."))
    assert isinstance(result, ExtractorResult)
    assert len(result.decisions) >= 1


# ── plugins CLI command ───────────────────────────────────────────────────────


def test_plugins_command_exits_zero(runner):
    result = runner.invoke(cli, ["plugins"])
    assert result.exit_code == 0, result.output


def test_plugins_command_shows_regex(runner):
    result = runner.invoke(cli, ["plugins"])
    assert "regex" in result.output


def test_plugins_command_shows_entry_point_group(runner):
    result = runner.invoke(cli, ["plugins"])
    assert ENTRY_POINT_GROUP in result.output
