"""Unit tests for domain models."""

from datetime import datetime

from consciousness.models import MemoryBlob, Role
from tests.conftest import make_conversation, make_message


def test_conversation_message_count():
    conv = make_conversation()
    assert conv.message_count == 2


def test_conversation_role_filtering():
    conv = make_conversation()
    assert len(conv.human_turns) == 1
    assert len(conv.assistant_turns) == 1
    assert conv.human_turns[0].role == Role.human
    assert conv.assistant_turns[0].role == Role.assistant


def test_conversation_as_text_format():
    conv = make_conversation(
        messages=[
            make_message("m1", "conv-1", Role.human, "Hello?", 0),
            make_message("m2", "conv-1", Role.assistant, "Hi there.", 1),
        ]
    )
    text = conv.as_text()
    assert "Test Conversation" in text
    assert "Human" in text
    assert "Hello?" in text
    assert "Assistant" in text
    assert "Hi there." in text


def test_conversation_as_text_ordering():
    conv = make_conversation(
        messages=[
            make_message("m1", "conv-1", Role.human, "First", 0),
            make_message("m2", "conv-1", Role.assistant, "Second", 1),
        ]
    )
    text = conv.as_text()
    assert text.index("First") < text.index("Second")


def test_memory_blob_render_includes_all_sections():
    blob = MemoryBlob(
        source_conversation_count=5,
        focus_topics=["databases"],
        sections={"About Me": "I am a developer.", "Tech Stack": "Python, Postgres"},
    )
    rendered = blob.render()
    assert "## About Me" in rendered
    assert "I am a developer." in rendered
    assert "## Tech Stack" in rendered
    assert "Python, Postgres" in rendered


def test_memory_blob_render_includes_metadata():
    blob = MemoryBlob(
        generated_at=datetime(2024, 6, 1),
        source_conversation_count=42,
        focus_topics=[],
        sections={"Note": "test"},
    )
    rendered = blob.render()
    assert "2024-06-01" in rendered
    assert "42" in rendered


def test_memory_blob_render_empty_sections():
    blob = MemoryBlob(source_conversation_count=0, focus_topics=[], sections={})
    rendered = blob.render()
    assert isinstance(rendered, str)
