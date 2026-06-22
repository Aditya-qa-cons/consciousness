"""Synthesize a Claude memory-import blob from conversation history.

Uses Claude itself (via the Anthropic SDK) to generate structured summaries
across five memory sections. Falls back to a template-based approach if no
API key is configured.
"""

import os
from datetime import datetime

import anthropic

from consciousness.models import Conversation, MemoryBlob

_SYSTEM = """You are synthesizing a user's Claude conversation history into a
structured memory document. Be specific, concrete, and use first person ("I").
Capture durable facts, preferences, and decisions — not transient chit-chat.
Omit anything that seems sensitive (credentials, personal health details)."""

_SECTION_PROMPTS = {
    "About Me": (
        "Summarize durable facts about this person: role, domain expertise, location, background, "
        "and any personal context they've shared consistently."
    ),
    "Technical Stack & Preferences": (
        "List the languages, frameworks, tools, and architectural patterns this person uses or prefers. "
        "Include version preferences and strong opinions."
    ),
    "Working Style": (
        "Describe how this person likes to work with Claude: level of detail they prefer, "
        "whether they want code or explanations first, preferred response length, etc."
    ),
    "Key Projects & Context": (
        "Summarize the main projects, codebases, or domains this person has worked on. "
        "Include project names, goals, and key decisions reached."
    ),
    "Recurring Decisions & Opinions": (
        "List opinions, preferences, and decisions that appear multiple times — "
        "technology choices, design philosophies, approaches they've settled on."
    ),
}

_FALLBACK_TEMPLATE = """[Auto-generated from {count} conversations on {date}]

Review and edit before importing — this is a starting point, not a final draft.
"""


class MemorySynthesizer:
    def __init__(self, api_key: str | None = None):
        self._api_key = api_key or os.getenv("ANTHROPIC_API_KEY")
        self._client: anthropic.Anthropic | None = None
        if self._api_key:
            self._client = anthropic.Anthropic(api_key=self._api_key)

    def synthesize(
        self,
        conversations: list[Conversation],
        focus_topics: list[str] | None = None,
    ) -> MemoryBlob:
        if not self._client:
            return self._fallback(conversations, focus_topics or [])

        # Build a condensed transcript corpus — most recent conversations first
        corpus = self._build_corpus(conversations, max_chars=60_000)

        sections: dict[str, str] = {}
        for section_name, section_prompt in _SECTION_PROMPTS.items():
            sections[section_name] = self._call_claude(corpus, section_prompt, focus_topics)

        return MemoryBlob(
            source_conversation_count=len(conversations),
            focus_topics=focus_topics or [],
            sections=sections,
        )

    def _call_claude(
        self,
        corpus: str,
        section_prompt: str,
        focus_topics: list[str] | None,
    ) -> str:
        topic_clause = ""
        if focus_topics:
            topic_clause = f" Pay special attention to these topics: {', '.join(focus_topics)}."

        message = self._client.messages.create(
            model="claude-opus-4-8",
            max_tokens=800,
            system=_SYSTEM,
            messages=[
                {
                    "role": "user",
                    "content": (
                        f"Here are excerpts from my Claude conversation history:\n\n{corpus}"
                        f"\n\n---\n\n{section_prompt}{topic_clause}"
                        "\n\nBe concise (under 200 words). Use bullet points."
                    ),
                }
            ],
        )
        return message.content[0].text

    def _build_corpus(self, conversations: list[Conversation], max_chars: int) -> str:
        """Sample conversations into a text corpus, newest first, within char budget."""
        sorted_convs = sorted(conversations, key=lambda c: c.updated_at, reverse=True)
        parts = []
        total = 0
        for conv in sorted_convs:
            snippet = f"### {conv.title}\n{conv.as_text()[:2000]}"
            if total + len(snippet) > max_chars:
                break
            parts.append(snippet)
            total += len(snippet)
        return "\n\n---\n\n".join(parts)

    def _fallback(
        self, conversations: list[Conversation], focus_topics: list[str]
    ) -> MemoryBlob:
        return MemoryBlob(
            source_conversation_count=len(conversations),
            focus_topics=focus_topics,
            sections={
                "Note": _FALLBACK_TEMPLATE.format(
                    count=len(conversations),
                    date=datetime.utcnow().strftime("%Y-%m-%d"),
                )
                + "\n\nSet ANTHROPIC_API_KEY to enable AI-assisted synthesis."
            },
        )
