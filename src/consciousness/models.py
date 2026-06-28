"""Core domain models — account-independent representations of Claude history."""

from datetime import datetime, timezone
from enum import Enum

from pydantic import BaseModel, Field


class Role(str, Enum):
    human = "human"
    assistant = "assistant"


class Attachment(BaseModel):
    file_name: str
    file_type: str
    file_size: int | None = None
    extracted_content: str | None = None


class Message(BaseModel):
    id: str
    conversation_id: str
    role: Role
    content: str
    timestamp: datetime
    position: int
    attachments: list[Attachment] = Field(default_factory=list)


class Conversation(BaseModel):
    id: str
    title: str
    project_id: str | None = None
    project_name: str | None = None
    created_at: datetime
    updated_at: datetime
    messages: list[Message] = Field(default_factory=list)
    account_id: str | None = None
    content_hash: str | None = None

    @property
    def message_count(self) -> int:
        return len(self.messages)

    @property
    def human_turns(self) -> list[Message]:
        return [m for m in self.messages if m.role == Role.human]

    @property
    def assistant_turns(self) -> list[Message]:
        return [m for m in self.messages if m.role == Role.assistant]

    def as_text(self) -> str:
        parts = [f"# {self.title}"]
        for msg in self.messages:
            speaker = "Human" if msg.role == Role.human else "Assistant"
            parts.append(f"\n**{speaker}:** {msg.content}")
        return "\n".join(parts)


class Project(BaseModel):
    id: str
    name: str
    created_at: datetime | None = None
    conversation_count: int = 0
    account_id: str | None = None


# ── extracted knowledge ────────────────────────────────────────────────────────


class Decision(BaseModel):
    """A settled conclusion extracted from an assistant message."""

    id: str
    topic: str
    conclusion: str
    confidence: float = 0.75
    conversation_id: str
    extracted_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    superseded_by: str | None = None  # ID of a later decision on the same topic


class Preference(BaseModel):
    """A recurring preference expressed by the user."""

    id: str
    area: str
    preference: str
    conversation_id: str
    extracted_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class TechChoice(BaseModel):
    """A technology verdict reached in a conversation."""

    id: str
    technology: str
    verdict: str
    rationale: str | None = None
    conversation_id: str
    extracted_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class ConversationSummary(BaseModel):
    """2-3 sentence summary of a single conversation."""

    conversation_id: str
    summary: str
    generated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    model: str | None = None  # None = text-extraction fallback


class ExcludeRule(BaseModel):
    """A rule that prevents a conversation or project from being indexed."""

    pattern: str
    rule_type: str  # 'conversation_id' | 'project_id' | 'title_glob'
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    shared: bool = False  # shared rules are bundled with share-export for team-wide policies


# ── knowledge graph ────────────────────────────────────────────────────────────


class KGNode(BaseModel):
    """A node in the knowledge graph — a technology name or decision topic."""

    id: str    # 'tech:postgres' or 'topic:database choice'
    type: str  # 'technology' | 'topic'
    label: str


class KGEdge(BaseModel):
    """A directed edge in the knowledge graph."""

    src_id: str
    dst_id: str
    relation: str  # 'co_occurs_with' | 'superseded_by' | 'relates_to'
    weight: float = 1.0


# ── search + memory ────────────────────────────────────────────────────────────


class SearchResult(BaseModel):
    conversation_id: str
    conversation_title: str
    project_name: str | None
    message_id: str
    role: Role
    snippet: str
    score: float
    timestamp: datetime


class MemoryBlob(BaseModel):
    """Structured payload ready for Claude's memory import box."""

    generated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    source_conversation_count: int
    focus_topics: list[str]
    sections: dict[str, str]

    def render(self) -> str:
        lines = [
            f"<!-- Generated {self.generated_at.strftime('%Y-%m-%d')} "
            f"from {self.source_conversation_count} conversations -->",
        ]
        for section, content in self.sections.items():
            lines.append(f"\n## {section}\n{content}")
        return "\n".join(lines)
