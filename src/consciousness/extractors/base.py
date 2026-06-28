"""Base types for the extractor plugin system."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    from consciousness.models import Conversation, Decision, Preference, TechChoice


@dataclass
class ExtractorResult:
    """Combined output from a single extractor plugin run."""

    decisions: list[Decision] = field(default_factory=list)
    preferences: list[Preference] = field(default_factory=list)
    tech_choices: list[TechChoice] = field(default_factory=list)


@runtime_checkable
class ExtractorPlugin(Protocol):
    """Protocol that extractor plugins must implement.

    Register your plugin via Python entry points so consciousness discovers it:

        [project.entry-points."consciousness.extractors"]
        my_extractor = "my_package.extractors:MyExtractorClass"

    The class is instantiated once at plugin-load time (no arguments).
    ``extract`` is called for every conversation during ingest.
    """

    name: str

    def extract(self, conv: Conversation) -> ExtractorResult:
        ...
