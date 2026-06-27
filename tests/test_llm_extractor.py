"""Unit tests for the LLM-assisted knowledge extractor."""

import json
from unittest.mock import MagicMock, patch

from consciousness.extractors.llm import LLMExtractor, _build_transcript, _parse_response
from consciousness.models import Role

# Re-use conftest helpers
from tests.conftest import make_conversation, make_message

# ── is_available ──────────────────────────────────────────────────────────────


def test_is_available_with_api_key():
    with patch("anthropic.Anthropic"):
        extractor = LLMExtractor(api_key="sk-ant-test")
    assert extractor.is_available() is True


def test_is_available_without_api_key(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    extractor = LLMExtractor(api_key=None)
    assert extractor.is_available() is False


def test_is_available_from_env(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-env")
    with patch("anthropic.Anthropic"):
        extractor = LLMExtractor()
    assert extractor.is_available() is True


# ── _build_transcript ─────────────────────────────────────────────────────────


def test_build_transcript_alternates_roles():
    conv = make_conversation(
        messages=[
            make_message("m1", "c1", Role.human, "Hello", 0),
            make_message("m2", "c1", Role.assistant, "Hi there!", 1),
        ]
    )
    transcript = _build_transcript(conv)
    assert "Human: Hello" in transcript
    assert "Assistant: Hi there!" in transcript


def test_build_transcript_respects_max_chars():
    long_text = "x" * 5000
    conv = make_conversation(
        messages=[
            make_message("m1", "c1", Role.human, long_text, 0),
            make_message("m2", "c1", Role.assistant, long_text, 1),
            make_message("m3", "c1", Role.human, long_text, 2),
            make_message("m4", "c1", Role.assistant, long_text, 3),
        ]
    )
    transcript = _build_transcript(conv, max_chars=8_000)
    assert len(transcript) <= 8_000 + 200  # small overhead for labels


# ── _parse_response ───────────────────────────────────────────────────────────


_VALID_JSON = {
    "decisions": [
        {"topic": "Database choice", "conclusion": "Use Postgres for production.", "confidence": 0.9},
    ],
    "preferences": [
        {"area": "Language", "preference": "Prefers Python over Go."},
    ],
    "tech_choices": [
        {"technology": "Postgres", "verdict": "Recommended for production.", "rationale": "Better JSON support."},
    ],
}


def test_parse_response_valid_json():
    decisions, prefs, tcs = _parse_response(json.dumps(_VALID_JSON), "conv-1")
    assert len(decisions) == 1
    assert decisions[0].topic == "Database choice"
    assert decisions[0].conclusion == "Use Postgres for production."
    assert decisions[0].confidence == 0.9
    assert decisions[0].conversation_id == "conv-1"


def test_parse_response_preferences():
    _, prefs, _ = _parse_response(json.dumps(_VALID_JSON), "conv-1")
    assert len(prefs) == 1
    assert prefs[0].area == "Language"
    assert prefs[0].preference == "Prefers Python over Go."


def test_parse_response_tech_choices():
    _, _, tcs = _parse_response(json.dumps(_VALID_JSON), "conv-1")
    assert len(tcs) == 1
    assert tcs[0].technology == "Postgres"
    assert tcs[0].verdict == "Recommended for production."
    assert tcs[0].rationale == "Better JSON support."


def test_parse_response_invalid_json():
    decisions, prefs, tcs = _parse_response("not valid json{", "conv-1")
    assert decisions == []
    assert prefs == []
    assert tcs == []


def test_parse_response_empty_arrays():
    raw = json.dumps({"decisions": [], "preferences": [], "tech_choices": []})
    decisions, prefs, tcs = _parse_response(raw, "conv-1")
    assert decisions == []
    assert prefs == []
    assert tcs == []


def test_parse_response_skips_missing_fields():
    raw = json.dumps({
        "decisions": [{"topic": "", "conclusion": "has no topic"}],
        "preferences": [{"area": "ok", "preference": ""}],
        "tech_choices": [{"technology": "", "verdict": "no tech name"}],
    })
    decisions, prefs, tcs = _parse_response(raw, "conv-1")
    assert decisions == []
    assert prefs == []
    assert tcs == []


def test_parse_response_clamps_confidence():
    raw = json.dumps({
        "decisions": [{"topic": "Test", "conclusion": "Use X.", "confidence": 5.0}],
        "preferences": [],
        "tech_choices": [],
    })
    decisions, _, _ = _parse_response(raw, "conv-1")
    assert decisions[0].confidence == 1.0


def test_parse_response_tech_choice_null_rationale():
    raw = json.dumps({
        "decisions": [],
        "preferences": [],
        "tech_choices": [{"technology": "Redis", "verdict": "Fast cache.", "rationale": None}],
    })
    _, _, tcs = _parse_response(raw, "conv-1")
    assert tcs[0].rationale is None


# ── LLMExtractor.extract ──────────────────────────────────────────────────────


def _make_mock_client(response_json: dict) -> MagicMock:
    mock_client = MagicMock()
    mock_msg = MagicMock()
    mock_msg.content = [MagicMock(text=json.dumps(response_json))]
    mock_client.messages.create.return_value = mock_msg
    return mock_client


def test_extract_returns_parsed_results():
    conv = make_conversation()
    extractor = LLMExtractor.__new__(LLMExtractor)
    extractor._model = "claude-haiku-4-5-20251001"
    extractor._client = _make_mock_client(_VALID_JSON)

    decisions, prefs, tcs = extractor.extract(conv)
    assert len(decisions) == 1
    assert len(prefs) == 1
    assert len(tcs) == 1


def test_extract_strips_markdown_fences():
    conv = make_conversation()
    extractor = LLMExtractor.__new__(LLMExtractor)
    extractor._model = "claude-haiku-4-5-20251001"

    fenced = f"```json\n{json.dumps(_VALID_JSON)}\n```"
    mock_client = MagicMock()
    mock_msg = MagicMock()
    mock_msg.content = [MagicMock(text=fenced)]
    mock_client.messages.create.return_value = mock_msg
    extractor._client = mock_client

    decisions, _, _ = extractor.extract(conv)
    assert len(decisions) == 1


def test_extract_returns_empty_on_api_error():
    conv = make_conversation()
    extractor = LLMExtractor.__new__(LLMExtractor)
    extractor._model = "claude-haiku-4-5-20251001"
    mock_client = MagicMock()
    mock_client.messages.create.side_effect = Exception("API error")
    extractor._client = mock_client

    decisions, prefs, tcs = extractor.extract(conv)
    assert decisions == []
    assert prefs == []
    assert tcs == []


def test_extract_returns_empty_when_no_client():
    conv = make_conversation()
    extractor = LLMExtractor.__new__(LLMExtractor)
    extractor._client = None

    decisions, prefs, tcs = extractor.extract(conv)
    assert decisions == []
    assert prefs == []
    assert tcs == []


def test_extract_skips_empty_conversation():

    conv = make_conversation(messages=[])
    extractor = LLMExtractor.__new__(LLMExtractor)
    extractor._model = "claude-haiku-4-5-20251001"
    extractor._client = _make_mock_client(_VALID_JSON)

    decisions, prefs, tcs = extractor.extract(conv)
    assert decisions == []
    assert prefs == []
    assert tcs == []
    # Client should NOT have been called for an empty conversation
    extractor._client.messages.create.assert_not_called()
