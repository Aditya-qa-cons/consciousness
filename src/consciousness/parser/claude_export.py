"""Parse Claude.ai data export into domain models.

Claude.ai Settings → Export produces a ZIP containing:
  - conversations.json  — array of all conversations
  - (optionally) projects.json

Conversation JSON shape (as observed from exports):
{
  "uuid": "...",
  "name": "Conversation title",
  "created_at": "2024-01-15T10:30:00.000000+00:00",
  "updated_at": "2024-01-15T11:00:00.000000+00:00",
  "account": {"uuid": "...", "full_name": "..."},
  "project": {"uuid": "...", "name": "..."} | null,
  "chat_messages": [
    {
      "uuid": "...",
      "sender": "human" | "assistant",
      "text": "...",
      "created_at": "...",
      "attachments": [...],
      "files": [...]
    }
  ]
}
"""

import json
import zipfile
from datetime import datetime
from pathlib import Path

from dateutil import parser as dateparser

from consciousness.models import Attachment, Conversation, Message, Project, Role


class ExportParseError(Exception):
    pass


def _parse_timestamp(raw: str | None) -> datetime | None:
    if raw is None:
        return None
    return dateparser.parse(raw)


def _parse_attachment(raw: dict) -> Attachment:
    return Attachment(
        file_name=raw.get("file_name", raw.get("name", "unknown")),
        file_type=raw.get("file_type", raw.get("media_type", "")),
        file_size=raw.get("file_size"),
        extracted_content=raw.get("extracted_content"),
    )


def _parse_message(raw: dict, conversation_id: str, position: int) -> Message:
    content = raw.get("text") or ""
    # Some exports nest content differently
    if not content and "content" in raw:
        blocks = raw["content"]
        if isinstance(blocks, list):
            content = "\n".join(
                b.get("text", "") for b in blocks if isinstance(b, dict)
            )
        elif isinstance(blocks, str):
            content = blocks

    attachments = [_parse_attachment(a) for a in raw.get("attachments", [])]
    attachments += [_parse_attachment(f) for f in raw.get("files", [])]

    return Message(
        id=raw["uuid"],
        conversation_id=conversation_id,
        role=Role(raw["sender"]),
        content=content,
        timestamp=_parse_timestamp(raw.get("created_at")),
        position=position,
        attachments=attachments,
    )


def _parse_conversation(raw: dict) -> Conversation:
    conv_id = raw["uuid"]
    project_raw = raw.get("project")

    messages = [
        _parse_message(m, conv_id, i)
        for i, m in enumerate(raw.get("chat_messages", []))
    ]

    return Conversation(
        id=conv_id,
        title=raw.get("name") or "Untitled",
        project_id=project_raw["uuid"] if project_raw else None,
        project_name=project_raw["name"] if project_raw else None,
        created_at=_parse_timestamp(raw.get("created_at")),
        updated_at=_parse_timestamp(raw.get("updated_at")),
        messages=messages,
    )


def parse_export(path: Path) -> tuple[list[Conversation], list[Project]]:
    """Parse a Claude.ai export ZIP or raw conversations.json.

    Returns (conversations, projects). Projects are derived from conversation
    metadata if no separate projects.json exists.
    """
    if path.suffix == ".zip":
        return _parse_zip(path)
    if path.suffix == ".json":
        return _parse_json_file(path)
    raise ExportParseError(f"Unsupported export format: {path.suffix}")


def _parse_zip(path: Path) -> tuple[list[Conversation], list[Project]]:
    with zipfile.ZipFile(path) as zf:
        names = zf.namelist()

        if "conversations.json" not in names:
            raise ExportParseError("conversations.json not found in export ZIP")

        with zf.open("conversations.json") as f:
            raw_conversations = json.load(f)

        raw_projects = []
        if "projects.json" in names:
            with zf.open("projects.json") as f:
                raw_projects = json.load(f)

    return _build_models(raw_conversations, raw_projects)


def _parse_json_file(path: Path) -> tuple[list[Conversation], list[Project]]:
    with open(path) as f:
        raw = json.load(f)
    # Support both bare array and {conversations: [...]} envelope
    if isinstance(raw, list):
        return _build_models(raw, [])
    if isinstance(raw, dict) and "conversations" in raw:
        return _build_models(raw["conversations"], raw.get("projects", []))
    raise ExportParseError("Unrecognized JSON structure")


def _build_models(
    raw_conversations: list[dict], raw_projects: list[dict]
) -> tuple[list[Conversation], list[Project]]:
    conversations = [_parse_conversation(c) for c in raw_conversations]

    # Build project index from explicit projects.json if available
    project_map: dict[str, Project] = {}
    for rp in raw_projects:
        p = Project(
            id=rp["uuid"],
            name=rp.get("name", "Unnamed Project"),
            created_at=_parse_timestamp(rp.get("created_at")),
        )
        project_map[p.id] = p

    # Supplement with projects inferred from conversation metadata
    for conv in conversations:
        if conv.project_id and conv.project_id not in project_map:
            project_map[conv.project_id] = Project(
                id=conv.project_id,
                name=conv.project_name or "Unknown Project",
            )

    # Tally conversation counts
    counts: dict[str, int] = {}
    for conv in conversations:
        if conv.project_id:
            counts[conv.project_id] = counts.get(conv.project_id, 0) + 1
    for pid, count in counts.items():
        if pid in project_map:
            project_map[pid].conversation_count = count

    return conversations, list(project_map.values())
