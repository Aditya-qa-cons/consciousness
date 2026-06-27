"""LLM-assisted knowledge extraction using Claude Haiku.

Produces higher recall than the regex approach by asking Claude to extract
decisions, preferences, and tech choices as structured JSON.  One API call
per conversation — gated on ANTHROPIC_API_KEY and the --llm-extract flag.

Falls back to empty results on any error so the caller can decide whether
to use the regex extractors as a fallback.
"""

import json
import logging
import os
import uuid

import anthropic

from consciousness.models import Conversation, Decision, Preference, TechChoice

_MODEL = "claude-haiku-4-5-20251001"
_MAX_CONV_CHARS = 8_000

_SYSTEM = """\
You are an expert at extracting structured knowledge from AI conversations.
Given a conversation transcript, identify only concrete, durable facts:

1. Decisions — settled conclusions or recommendations made by the assistant
2. Preferences — explicit preferences expressed by the human
3. Technology choices — verdicts about specific named technologies

Respond ONLY with valid JSON in this exact schema, no markdown fences:
{
  "decisions": [{"topic": "...", "conclusion": "...", "confidence": 0.85}],
  "preferences": [{"area": "...", "preference": "..."}],
  "tech_choices": [{"technology": "...", "verdict": "...", "rationale": "..."}]
}

Rules:
- topic: 3–60 chars, describes what the decision is about
- conclusion: the actual recommendation (one sentence)
- confidence: 0.9 for clear direct recommendations, 0.7 for softer suggestions
- area: the technology/tool/approach the human prefers
- technology: exact technology name (e.g. "Postgres", "React", "FastAPI")
- verdict: one sentence summary of the technology verdict
- rationale: optional explanation (null if not stated)
- Return empty arrays if nothing relevant is found
- Exclude transient questions, clarifications, and non-technical discussion
"""

_logger = logging.getLogger(__name__)


class LLMExtractor:
    """Extract knowledge from conversations using Claude Haiku."""

    def __init__(self, api_key: str | None = None, model: str = _MODEL):
        self._api_key = api_key or os.getenv("ANTHROPIC_API_KEY")
        self._model = model
        self._client: anthropic.Anthropic | None = None
        if self._api_key:
            self._client = anthropic.Anthropic(api_key=self._api_key)

    def is_available(self) -> bool:
        return self._client is not None

    def extract(self, conv: Conversation) -> tuple[list[Decision], list[Preference], list[TechChoice]]:
        """Return extracted knowledge or empty lists on any error."""
        if not self._client:
            return [], [], []

        transcript = _build_transcript(conv)
        if not transcript.strip():
            return [], [], []

        try:
            message = self._client.messages.create(
                model=self._model,
                max_tokens=1024,
                system=_SYSTEM,
                messages=[{
                    "role": "user",
                    "content": f"Extract knowledge from this conversation:\n\n{transcript}",
                }],
            )
            raw = message.content[0].text.strip()
            # Strip markdown code fences if the model wraps output anyway
            if raw.startswith("```"):
                parts = raw.split("```")
                raw = parts[1].lstrip("json").strip() if len(parts) > 1 else raw
            return _parse_response(raw, conv.id)
        except Exception:
            _logger.debug("LLM extraction failed for conv %s", conv.id, exc_info=True)
            return [], [], []


def _build_transcript(conv: Conversation, max_chars: int = _MAX_CONV_CHARS) -> str:
    parts: list[str] = []
    total = 0
    for msg in conv.messages:
        prefix = "Human" if msg.role.value == "human" else "Assistant"
        line = f"{prefix}: {msg.content[:1500]}"
        if total + len(line) > max_chars:
            break
        parts.append(line)
        total += len(line)
    return "\n\n".join(parts)


def _parse_response(
    raw: str,
    conversation_id: str,
) -> tuple[list[Decision], list[Preference], list[TechChoice]]:
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        _logger.debug("LLM returned non-JSON: %r", raw[:200])
        return [], [], []

    decisions: list[Decision] = []
    for item in data.get("decisions") or []:
        topic = str(item.get("topic", "")).strip()[:60]
        conclusion = str(item.get("conclusion", "")).strip()[:500]
        if not topic or not conclusion:
            continue
        try:
            confidence = float(item.get("confidence", 0.8))
        except (TypeError, ValueError):
            confidence = 0.8
        decisions.append(Decision(
            id=str(uuid.uuid4()),
            topic=topic,
            conclusion=conclusion,
            confidence=max(0.1, min(1.0, confidence)),
            conversation_id=conversation_id,
        ))

    preferences: list[Preference] = []
    for item in data.get("preferences") or []:
        area = str(item.get("area", "")).strip()[:60]
        preference = str(item.get("preference", "")).strip()[:500]
        if not area or not preference:
            continue
        preferences.append(Preference(
            id=str(uuid.uuid4()),
            area=area,
            preference=preference,
            conversation_id=conversation_id,
        ))

    tech_choices: list[TechChoice] = []
    for item in data.get("tech_choices") or []:
        technology = str(item.get("technology", "")).strip()[:100]
        verdict = str(item.get("verdict", "")).strip()[:300]
        raw_rationale = item.get("rationale")
        rationale = str(raw_rationale).strip()[:300] if raw_rationale else None
        if not technology or not verdict:
            continue
        tech_choices.append(TechChoice(
            id=str(uuid.uuid4()),
            technology=technology,
            verdict=verdict,
            rationale=rationale,
            conversation_id=conversation_id,
        ))

    return decisions, preferences, tech_choices
