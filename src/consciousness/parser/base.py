"""SourceAdapter protocol — plug in any conversation export format."""

from pathlib import Path
from typing import Protocol

from consciousness.models import Conversation, Project


class SourceAdapter(Protocol):
    """Implement this to add a new conversation export source."""

    source_name: str

    def can_handle(self, path: Path) -> bool:
        """Return True if this adapter understands the file at path."""
        ...

    def parse(self, path: Path) -> tuple[list[Conversation], list[Project]]:
        """Parse the file and return (conversations, projects)."""
        ...
