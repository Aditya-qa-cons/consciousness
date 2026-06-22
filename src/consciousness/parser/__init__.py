"""Parser package — auto-detect the right adapter for any export format."""

from pathlib import Path

from consciousness.models import Conversation, Project
from consciousness.parser.base import SourceAdapter
from consciousness.parser.claude_export import ClaudeExportAdapter, ExportParseError

# Registry — add new adapters here
_ADAPTERS: list[SourceAdapter] = [
    ClaudeExportAdapter(),
]


def parse_export(path: Path) -> tuple[list[Conversation], list[Project]]:
    """Auto-detect the right adapter and parse the export file."""
    for adapter in _ADAPTERS:
        if adapter.can_handle(path):
            return adapter.parse(path)
    raise ExportParseError(f"No adapter found for: {path} — supported formats: ZIP (Claude.ai), JSON")


__all__ = ["parse_export", "SourceAdapter", "ClaudeExportAdapter", "ExportParseError"]
