"""Per-conversation summarisation using Claude Haiku.

Produces a 2-3 sentence summary capturing the main question and key outcome.
Falls back to extracting the first assistant message when no API key is set or
on any API error — so a summary is always produced.
"""

import logging
import os

import anthropic

from consciousness.models import Conversation, Role

_MODEL = "claude-haiku-4-5-20251001"
_MAX_CONV_CHARS = 6_000

_SYSTEM = """\
Summarize this AI conversation in 2-3 sentences.
Capture: (1) the main question or task, (2) the key conclusion or recommendation.
Be specific and concrete. No filler phrases. Plain sentences only.
Respond with the summary text and nothing else."""

_logger = logging.getLogger(__name__)


class ConversationSummarizer:
    """Summarize individual conversations to 2-3 sentences."""

    def __init__(self, api_key: str | None = None, model: str = _MODEL):
        self._api_key = api_key or os.getenv("ANTHROPIC_API_KEY")
        self._model = model
        self._client: anthropic.Anthropic | None = None
        if self._api_key:
            self._client = anthropic.Anthropic(api_key=self._api_key)

    def is_available(self) -> bool:
        return self._client is not None

    def model_used(self) -> str | None:
        return self._model if self._client else None

    def summarize(self, conv: Conversation) -> str:
        """Return a 2-3 sentence summary. Falls back to text extraction on any error."""
        if not self._client:
            return _fallback_summary(conv)

        transcript = _build_transcript(conv)
        if not transcript.strip():
            return _fallback_summary(conv)

        try:
            message = self._client.messages.create(
                model=self._model,
                max_tokens=200,
                system=_SYSTEM,
                messages=[{"role": "user", "content": transcript}],
            )
            text = message.content[0].text.strip()
            return text if text else _fallback_summary(conv)
        except Exception:
            _logger.debug("LLM summarization failed for conv %s", conv.id, exc_info=True)
            return _fallback_summary(conv)


def _build_transcript(conv: Conversation, max_chars: int = _MAX_CONV_CHARS) -> str:
    parts: list[str] = []
    total = 0
    for msg in conv.messages:
        prefix = "Human" if msg.role == Role.human else "Assistant"
        line = f"{prefix}: {msg.content[:1500]}"
        if total + len(line) > max_chars:
            break
        parts.append(line)
        total += len(line)
    return "\n\n".join(parts)


def _fallback_summary(conv: Conversation) -> str:
    """Best-effort summary without LLM: first non-empty assistant message, truncated."""
    for msg in conv.messages:
        if msg.role == Role.assistant and msg.content.strip():
            text = msg.content.strip()
            if len(text) > 280:
                text = text[:277] + "…"
            return text
    return conv.title
