"""Unit tests for the per-conversation summarizer."""

from unittest.mock import MagicMock, patch

from consciousness.memory.summarizer import ConversationSummarizer, _fallback_summary
from consciousness.models import Role
from tests.conftest import make_conversation, make_message

# ── is_available ──────────────────────────────────────────────────────────────

def test_is_available_with_api_key():
    with patch("anthropic.Anthropic"):
        s = ConversationSummarizer(api_key="sk-ant-test")
    assert s.is_available() is True


def test_is_available_without_api_key(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    s = ConversationSummarizer(api_key=None)
    assert s.is_available() is False


def test_model_used_with_client():
    with patch("anthropic.Anthropic"):
        s = ConversationSummarizer(api_key="sk-ant-test")
    assert s.model_used() is not None


def test_model_used_without_client(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    s = ConversationSummarizer(api_key=None)
    assert s.model_used() is None


# ── fallback_summary ──────────────────────────────────────────────────────────

def test_fallback_uses_first_assistant_message():
    conv = make_conversation(
        messages=[
            make_message("m1", "c1", Role.human, "What database?", 0),
            make_message("m2", "c1", Role.assistant, "Use Postgres for production.", 1),
        ]
    )
    result = _fallback_summary(conv)
    assert "Postgres" in result


def test_fallback_truncates_long_message():
    long_text = "x" * 500
    conv = make_conversation(
        messages=[make_message("m1", "c1", Role.assistant, long_text, 0)]
    )
    result = _fallback_summary(conv)
    assert len(result) <= 283  # 280 + "…"
    assert result.endswith("…")


def test_fallback_returns_title_when_no_assistant_messages():
    conv = make_conversation(
        messages=[make_message("m1", "c1", Role.human, "Hello?", 0)],
        title="My conversation",
    )
    assert _fallback_summary(conv) == "My conversation"


def test_fallback_skips_empty_assistant_content():
    conv = make_conversation(
        messages=[
            make_message("m1", "c1", Role.assistant, "   ", 0),
            make_message("m2", "c1", Role.assistant, "Real answer.", 1),
        ]
    )
    assert _fallback_summary(conv) == "Real answer."


# ── summarize with mocked client ──────────────────────────────────────────────

def _make_summarizer_with_mock(response_text: str) -> ConversationSummarizer:
    s = ConversationSummarizer.__new__(ConversationSummarizer)
    s._model = "claude-haiku-4-5-20251001"
    mock_client = MagicMock()
    mock_msg = MagicMock()
    mock_msg.content = [MagicMock(text=response_text)]
    mock_client.messages.create.return_value = mock_msg
    s._client = mock_client
    return s


def test_summarize_returns_llm_text():
    conv = make_conversation()
    s = _make_summarizer_with_mock("The user asked about databases. Postgres was recommended.")
    result = s.summarize(conv)
    assert result == "The user asked about databases. Postgres was recommended."


def test_summarize_falls_back_on_api_error():
    conv = make_conversation(
        messages=[
            make_message("m1", "c1", Role.human, "Question?", 0),
            make_message("m2", "c1", Role.assistant, "Fallback answer.", 1),
        ]
    )
    s = ConversationSummarizer.__new__(ConversationSummarizer)
    s._model = "claude-haiku-4-5-20251001"
    mock_client = MagicMock()
    mock_client.messages.create.side_effect = Exception("API down")
    s._client = mock_client

    result = s.summarize(conv)
    assert "Fallback answer" in result


def test_summarize_uses_fallback_when_no_client(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    conv = make_conversation(
        messages=[
            make_message("m1", "c1", Role.human, "Q?", 0),
            make_message("m2", "c1", Role.assistant, "Fallback from no client.", 1),
        ]
    )
    s = ConversationSummarizer(api_key=None)
    result = s.summarize(conv)
    assert "Fallback from no client" in result


def test_summarize_falls_back_on_empty_llm_response():
    conv = make_conversation(
        messages=[make_message("m1", "c1", Role.assistant, "Real answer.", 0)]
    )
    s = _make_summarizer_with_mock("")  # empty response
    result = s.summarize(conv)
    assert result == "Real answer."


# ── DB round-trip ─────────────────────────────────────────────────────────────

def test_db_upsert_and_get_summary(db):
    from consciousness.models import ConversationSummary
    from tests.conftest import make_conversation, make_project

    proj = make_project()
    db.upsert_project(proj)
    conv = make_conversation()
    db.upsert_conversation(conv)

    summary = ConversationSummary(
        conversation_id="conv-1",
        summary="This is a test summary.",
        model="claude-haiku-4-5-20251001",
    )
    db.upsert_summary(summary)
    db.commit()

    retrieved = db.get_summary("conv-1")
    assert retrieved is not None
    assert retrieved.summary == "This is a test summary."
    assert retrieved.model == "claude-haiku-4-5-20251001"


def test_db_get_summary_returns_none_for_missing(db):
    assert db.get_summary("nonexistent-conv") is None


def test_db_get_summaries_bulk(db):
    from consciousness.models import ConversationSummary
    from tests.conftest import make_project

    db.upsert_project(make_project())
    conv1 = make_conversation(id="conv-1")
    conv2 = make_conversation(id="conv-2", title="Second conv")
    db.upsert_conversation(conv1)
    db.upsert_conversation(conv2)

    for cid, text in [("conv-1", "Summary one."), ("conv-2", "Summary two.")]:
        db.upsert_summary(ConversationSummary(conversation_id=cid, summary=text))
    db.commit()

    result = db.get_summaries(["conv-1", "conv-2", "unknown"])
    assert len(result) == 2
    assert result["conv-1"].summary == "Summary one."
    assert result["conv-2"].summary == "Summary two."
    assert "unknown" not in result


def test_db_stats_includes_summaries(db):
    from consciousness.models import ConversationSummary
    from tests.conftest import make_project

    db.upsert_project(make_project())
    conv = make_conversation()
    db.upsert_conversation(conv)
    db.upsert_summary(ConversationSummary(conversation_id="conv-1", summary="A summary."))
    db.commit()

    stats = db.stats()
    assert stats["summaries"] == 1
