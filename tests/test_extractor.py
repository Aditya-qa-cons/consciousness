"""Unit tests for pattern-based knowledge extraction."""


from consciousness.extractors.knowledge import (
    apply_temporal_tracking,
    extract_decisions,
    extract_preferences,
    extract_tech_choices,
)
from consciousness.extractors.sensitive import has_sensitive_content, redact
from tests.conftest import Role, make_conversation, make_message


def _conv_with_messages(human_text: str, assistant_text: str):
    return make_conversation(
        messages=[
            make_message("h1", "conv-1", Role.human, human_text, 0),
            make_message("a1", "conv-1", Role.assistant, assistant_text, 1),
        ]
    )


# ── decision extraction ───────────────────────────────────────────────────────


def test_extract_decision_recommend_pattern():
    conv = _conv_with_messages(
        "Which database should I use?",
        "I recommend Postgres for production workloads because it handles concurrent writes well.",
    )
    decisions = extract_decisions(conv)
    assert len(decisions) > 0
    topics = [d.topic.lower() for d in decisions]
    assert any("postgres" in t for t in topics)


def test_extract_decision_use_pattern():
    conv = _conv_with_messages(
        "JWT or sessions?",
        "Use JWT for stateless APIs because it scales horizontally without shared session storage.",
    )
    decisions = extract_decisions(conv)
    assert len(decisions) > 0


def test_extract_decision_avoid_pattern():
    conv = _conv_with_messages(
        "Should I use MongoDB?",
        "Avoid MongoDB because schema constraints are painful to enforce.",
    )
    decisions = extract_decisions(conv)
    assert len(decisions) > 0


def test_extract_no_decisions_from_questions():
    conv = _conv_with_messages(
        "What are the options for auth?",
        "There are many ways to do auth. Let me walk you through the options.",
    )
    decisions = extract_decisions(conv)
    assert len(decisions) == 0


def test_decision_has_required_fields():
    conv = _conv_with_messages(
        "Which ORM?",
        "I recommend SQLAlchemy for Python projects because it has mature async support.",
    )
    decisions = extract_decisions(conv)
    assert len(decisions) > 0
    d = decisions[0]
    assert d.id
    assert d.topic
    assert d.conclusion
    assert 0 < d.confidence <= 1.0
    assert d.conversation_id == "conv-1"


# ── preference extraction ─────────────────────────────────────────────────────


def test_extract_preference_i_prefer():
    conv = _conv_with_messages(
        "I prefer TypeScript over JavaScript because of the type safety.",
        "That makes sense for larger projects.",
    )
    prefs = extract_preferences(conv)
    assert len(prefs) > 0
    assert any("typescript" in p.area.lower() for p in prefs)


def test_extract_preference_go_to():
    conv = _conv_with_messages(
        "FastAPI is my go-to for Python APIs.",
        "Good choice.",
    )
    prefs = extract_preferences(conv)
    assert len(prefs) > 0


def test_no_preferences_from_assistant():
    conv = _conv_with_messages(
        "What should I use?",
        "I prefer React for frontend projects with complex state.",
    )
    # Preferences should only be extracted from human messages
    prefs = extract_preferences(conv)
    assert all(p.conversation_id == "conv-1" for p in prefs)
    # The assistant saying "I prefer" should NOT be treated as user preference
    # (preferences are scanned from human_turns only)
    assert len(prefs) == 0


# ── tech choice extraction ────────────────────────────────────────────────────


def test_extract_tech_choice_known_tech():
    conv = _conv_with_messages(
        "Should I use Redis?",
        "I recommend Redis for caching session data because it's blazing fast.",
    )
    choices = extract_tech_choices(conv)
    assert any(tc.technology == "Redis" for tc in choices)


def test_extract_tech_choice_multiple_techs():
    conv = _conv_with_messages(
        "Compare Postgres and MySQL.",
        "I recommend Postgres over MySQL for most use cases.",
    )
    choices = extract_tech_choices(conv)
    tech_names = {tc.technology for tc in choices}
    assert "Postgres" in tech_names or "PostgreSQL" in tech_names or "MySQL" in tech_names


# ── temporal tracking ─────────────────────────────────────────────────────────


def test_temporal_tracking_finds_supersession():

    from consciousness.models import Decision

    old = Decision(id="old-1", topic="Postgres", conclusion="Use Postgres", confidence=0.8, conversation_id="c1")
    new = Decision(
        id="new-1", topic="Postgres for production",
        conclusion="Use Postgres with connection pooling", confidence=0.9, conversation_id="c2",
    )
    supersessions = apply_temporal_tracking([new], [old])
    assert len(supersessions) == 1
    assert supersessions[0] == ("old-1", "new-1")


def test_temporal_tracking_no_overlap():
    from consciousness.models import Decision

    old = Decision(id="old-1", topic="database", conclusion="Use Postgres", confidence=0.8, conversation_id="c1")
    new = Decision(id="new-1", topic="authentication", conclusion="Use JWT", confidence=0.8, conversation_id="c2")
    supersessions = apply_temporal_tracking([new], [old])
    assert len(supersessions) == 0


# ── sensitive content ─────────────────────────────────────────────────────────


def test_detects_openai_key():
    text = "My key is sk-abcdefghijklmnopqrstuvwxyzABCDEF12"
    assert has_sensitive_content(text)


def test_detects_aws_access_key():
    assert has_sensitive_content("AKIAIOSFODNN7EXAMPLE")


def test_detects_password_assignment():
    assert has_sensitive_content("password=SuperSecret123!")


def test_redacts_key():
    text = "Use sk-abcdefghijklmnopqrstuvwxyzABCDEF12 for auth."
    clean, findings = redact(text)
    assert "sk-" not in clean
    assert "[REDACTED]" in clean
    assert len(findings) > 0


def test_clean_text_unchanged():
    text = "Use Postgres for production workloads."
    clean, findings = redact(text)
    assert clean == text
    assert findings == []
