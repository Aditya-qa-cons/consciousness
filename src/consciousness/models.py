"""Core domain models — account-independent representations of Claude history."""

from datetime import datetime
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

    generated_at: datetime = Field(default_factory=datetime.utcnow)
    source_conversation_count: int
    focus_topics: list[str]
    sections: dict[str, str]

    def render(self) -> str:
        """Render as markdown for pasting into Claude memory import."""
        lines = [
            f"<!-- Generated {self.generated_at.strftime('%Y-%m-%d')} "
            f"from {self.source_conversation_count} conversations -->",
        ]
        for section, content in self.sections.items():
            lines.append(f"\n## {section}\n{content}")
        return "\n".join(lines)
