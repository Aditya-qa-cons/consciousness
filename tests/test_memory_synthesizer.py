"""Unit tests for the memory synthesizer (no API key — fallback path only)."""

import pytest

from consciousness.memory.synthesizer import MemorySynthesizer
from consciousness.models import MemoryBlob
from tests.conftest import make_conversation


@pytest.fixture
def synthesizer():
    return MemorySynthesizer(api_key=None)


def test_fallback_returns_memory_blob(synthesizer):
    convs = [make_conversation()]
    blob = synthesizer.synthesize(convs)
    assert isinstance(blob, MemoryBlob)


def test_fallback_counts_conversations(synthesizer):
    convs = [make_conversation(id=f"conv-{i}") for i in range(5)]
    blob = synthesizer.synthesize(convs)
    assert blob.source_conversation_count == 5


def test_fallback_records_focus_topics(synthesizer):
    convs = [make_conversation()]
    blob = synthesizer.synthesize(convs, focus_topics=["databases", "auth"])
    assert "databases" in blob.focus_topics
    assert "auth" in blob.focus_topics


def test_fallback_sections_contain_api_key_hint(synthesizer):
    convs = [make_conversation()]
    blob = synthesizer.synthesize(convs)
    rendered = blob.render()
    assert "ANTHROPIC_API_KEY" in rendered


def test_empty_conversation_list(synthesizer):
    blob = synthesizer.synthesize([])
    assert blob.source_conversation_count == 0
    assert isinstance(blob.render(), str)


def test_corpus_respects_char_budget(synthesizer):
    # Build enough conversations to exceed 60_000 chars
    long_content = "x" * 3000
    from tests.conftest import Role, make_message

    convs = []
    for i in range(30):
        msg = make_message(f"m{i}", f"c{i}", Role.human, long_content, 0)
        conv = make_conversation(id=f"c{i}", messages=[msg])
        convs.append(conv)

    # Should not raise; corpus is capped internally
    corpus = synthesizer._build_corpus(convs, max_chars=10_000)
    assert len(corpus) <= 10_000 + 200  # small slack for snippet headers
