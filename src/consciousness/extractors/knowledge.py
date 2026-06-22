"""Pattern-based knowledge extraction — no API key required.

Scans assistant messages for decisions, user messages for preferences,
and all messages for technology verdicts. Results are stored in the DB
alongside raw conversations so recall_decision can query structured facts
before falling back to vector search.

Temporal tracking: when a new decision on the same topic is extracted
from a later conversation, the earlier one is marked superseded_by.
"""

import re
import uuid

from consciousness.models import Conversation, Decision, Preference, TechChoice

# ── decision patterns (match assistant messages) ───────────────────────────────

_DECISION_PATTERNS: list[tuple[re.Pattern, float]] = [
    (re.compile(  # noqa: E501
        r"(?i)(?:i (?:would )?recommend|you should use|use|go with|go for|choose|stick with)"
        r"\s+([\w][\w\s\-\.]{1,30}?)(?:\s+for\b|\s+because\b|\s+over\b|\s+when\b|\.|,)"
    ), 0.70),
    (re.compile(  # noqa: E501
        r"(?i)(?:the )?(?:best|right|correct|recommended|ideal)"
        r"\s+(?:choice|option|approach|tool|solution|way|fit)"
        r"\s+(?:here\s+)?is\s+([\w][\w\s\-\.]{1,30}?)(?:\s+because\b|\.|,)"
    ), 0.80),
    (re.compile(
        r"(?i)([\w][\w\s\-\.]{1,25}?)\s+is\s+(?:the\s+)?"
        r"(?:better|best|preferred|recommended|right)\s+(?:choice|option|approach|fit|pick)"
    ), 0.75),
    (re.compile(
        r"(?i)(?:i'?d? suggest|i recommend|my recommendation is|i'd go with)"
        r"\s+([\w][\w\s\-\.]{1,30}?)(?:\s+for\b|\s+because\b|\.|,)"
    ), 0.80),
    (re.compile(
        r"(?i)(?:avoid|don'?t use|stay away from|skip)"
        r"\s+([\w][\w\s\-\.]{1,25}?)(?:\s+because\b|\s+—|\s+–|\.|,)"
    ), 0.70),
]

# ── preference patterns (match human messages) ─────────────────────────────────

_PREFERENCE_PATTERNS: list[tuple[re.Pattern, str]] = [
    (re.compile(
        r"(?i)i (?:prefer|like|love|always use|tend to use|usually use)"
        r"\s+([\w][\w\s\-\.]{1,30}?)(?:\s+over\b|\s+because\b|\.|,|$)"
    ), "general"),
    (re.compile(r"(?i)([\w][\w\s\-\.]{1,25}?)\s+is (?:my )?(?:preferred|favourite|go-to|default)"), "general"),
    (re.compile(
        r"(?i)i (?:don'?t like|dislike|hate|avoid)"
        r"\s+([\w][\w\s\-\.]{1,25}?)(?:\s+because\b|\.|,|$)"
    ), "dislike"),
]

# ── known technology terms ─────────────────────────────────────────────────────

_TECH_TERMS: set[str] = {
    # Languages
    "Python", "Go", "Golang", "Rust", "TypeScript", "JavaScript", "Java", "Kotlin",
    "Swift", "Ruby", "PHP", "C++", "C#", "Scala", "Elixir", "Haskell",
    # Databases
    "Postgres", "PostgreSQL", "MySQL", "SQLite", "MongoDB", "DynamoDB", "Redis",
    "Elasticsearch", "Cassandra", "CockroachDB", "Supabase", "PlanetScale",
    # Frontend
    "React", "Vue", "Angular", "Svelte", "Next.js", "Nuxt", "Remix", "Astro", "SvelteKit",
    # Backend frameworks
    "FastAPI", "Django", "Flask", "Express", "Fastify", "NestJS", "Rails", "Laravel",
    "Spring", "Gin", "Fiber", "Axum", "Phoenix",
    # Infra / DevOps
    "Docker", "Kubernetes", "K8s", "Terraform", "Ansible", "Helm", "ArgoCD",
    "AWS", "GCP", "Azure", "Cloudflare", "Vercel", "Railway", "Fly.io",
    # Auth
    "JWT", "OAuth", "OAuth2", "SAML", "Passkeys", "WebAuthn", "Clerk", "Auth0",
    # ORMs / data access
    "SQLAlchemy", "Prisma", "TypeORM", "Drizzle", "Hibernate", "ActiveRecord", "GORM",
    "PyJWT", "Alembic",
    # Testing
    "pytest", "Jest", "Vitest", "Playwright", "Cypress", "Testing Library",
    # Message queues / streaming
    "Kafka", "RabbitMQ", "SQS", "Celery", "Bull",
}

_VERDICT_WORDS = re.compile(
    r"(?i)\b(recommend|use|choose|prefer|better|best|avoid|don'?t use|go with|stick with|switch to)\b"
)


def _extract_sentence(text: str, match_start: int) -> str:
    """Return the full sentence containing the match position."""
    # Find sentence start (after . ! ? or start of text)
    start = max(text.rfind(".", 0, match_start), text.rfind("!", 0, match_start), text.rfind("?", 0, match_start))
    start = start + 1 if start >= 0 else 0
    # Find sentence end
    end_match = re.search(r"[.!?]", text[match_start:])
    end = match_start + end_match.start() + 1 if end_match else len(text)
    return text[start:end].strip()


def _short_topic(raw: str) -> str:
    """Clean up a raw regex-captured topic string."""
    return raw.strip().rstrip(".,;:").strip()


def extract_decisions(conv: Conversation) -> list[Decision]:
    results: list[Decision] = []
    for msg in conv.assistant_turns:
        for pattern, confidence in _DECISION_PATTERNS:
            for match in pattern.finditer(msg.content):
                topic = _short_topic(match.group(1))
                if len(topic) < 2 or len(topic) > 60:
                    continue
                conclusion = _extract_sentence(msg.content, match.start())
                if not conclusion:
                    conclusion = match.group(0).strip()
                results.append(Decision(
                    id=str(uuid.uuid4()),
                    topic=topic,
                    conclusion=conclusion[:500],
                    confidence=confidence,
                    conversation_id=conv.id,
                ))
    return results


def extract_preferences(conv: Conversation) -> list[Preference]:
    results: list[Preference] = []
    for msg in conv.human_turns:
        for pattern, area in _PREFERENCE_PATTERNS:
            for match in pattern.finditer(msg.content):
                subject = _short_topic(match.group(1))
                if len(subject) < 2 or len(subject) > 60:
                    continue
                sentence = _extract_sentence(msg.content, match.start())
                results.append(Preference(
                    id=str(uuid.uuid4()),
                    area=subject,
                    preference=sentence[:500] or match.group(0).strip(),
                    conversation_id=conv.id,
                ))
    return results


def extract_tech_choices(conv: Conversation) -> list[TechChoice]:
    results: list[TechChoice] = []
    seen: set[str] = set()

    all_text = " ".join(m.content for m in conv.messages)
    for tech in _TECH_TERMS:
        if tech.lower() not in all_text.lower():
            continue
        pattern = re.compile(rf"\b{re.escape(tech)}\b", re.IGNORECASE)
        for match in pattern.finditer(all_text):
            window_start = max(0, match.start() - 150)
            window_end = min(len(all_text), match.end() + 150)
            window = all_text[window_start:window_end]
            if not _VERDICT_WORDS.search(window):
                continue
            key = (tech.lower(), conv.id)
            if key in seen:
                continue
            seen.add(key)
            sentence = _extract_sentence(all_text, match.start())
            results.append(TechChoice(
                id=str(uuid.uuid4()),
                technology=tech,
                verdict=sentence[:300] or match.group(0),
                rationale=None,
                conversation_id=conv.id,
            ))
    return results


def apply_temporal_tracking(new_decisions: list[Decision], existing_decisions: list[Decision]) -> list[tuple[str, str]]:
    """Return (old_id, new_id) pairs where the new decision supersedes an older one.

    Matching is done on normalized topic overlap — conservative to avoid false positives.
    """
    supersessions: list[tuple[str, str]] = []
    for new in new_decisions:
        new_topic = new.topic.lower()
        for old in existing_decisions:
            if old.superseded_by:
                continue
            old_topic = old.topic.lower()
            # Topics overlap if one contains the other (e.g. "Postgres" ↔ "Postgres for databases")
            if new_topic in old_topic or old_topic in new_topic:
                supersessions.append((old.id, new.id))
    return supersessions
