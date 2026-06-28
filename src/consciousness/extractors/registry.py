"""Extractor plugin registry — discovers and runs plugins via Python entry points."""

from __future__ import annotations

import importlib.metadata
import warnings

from consciousness.models import Conversation

from .base import ExtractorPlugin, ExtractorResult

ENTRY_POINT_GROUP = "consciousness.extractors"


def load_plugins() -> list[ExtractorPlugin]:
    """Discover all extractor plugins registered under 'consciousness.extractors'.

    Returns an instance of each registered class, in the order they appear in the
    entry point metadata. A failed load emits a warning and skips that plugin.
    """
    plugins: list[ExtractorPlugin] = []
    for ep in importlib.metadata.entry_points(group=ENTRY_POINT_GROUP):
        try:
            cls = ep.load()
            plugins.append(cls())
        except Exception as exc:
            warnings.warn(
                f"Failed to load extractor plugin {ep.name!r} ({ep.value}): {exc}",
                stacklevel=2,
            )
    return plugins


def run_plugins(plugins: list[ExtractorPlugin], conv: Conversation) -> ExtractorResult:
    """Run all plugins against a conversation and merge their results.

    A plugin that raises is skipped with a warning — one broken plugin never
    prevents the rest from running.
    """
    merged = ExtractorResult()
    for plugin in plugins:
        try:
            result = plugin.extract(conv)
            merged.decisions.extend(result.decisions)
            merged.preferences.extend(result.preferences)
            merged.tech_choices.extend(result.tech_choices)
        except Exception as exc:
            warnings.warn(
                f"Extractor plugin {plugin.name!r} raised during extraction: {exc}",
                stacklevel=2,
            )
    return merged
