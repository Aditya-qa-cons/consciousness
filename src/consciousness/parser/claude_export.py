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

import hashlib
import json
import zipfile
from datetime import datetime
from pathlib import Path

from dateutil import parser as dateparser

from consciousness.models import Attachment, Conversation, Message, Project, Role


def _compute_content_hash(messages: list[Message]) -> str:
    parts = sorted(f"{m.role.value}:{m.content}" for m in messages)
    return hashlib.sha256("\n".join(parts).encode()).hexdigest()


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
    if not content and "content" in raw:
        blocks = raw["content"]
        if isinstance(blocks, list):
            content = "\n".join(b.get("text", "") for b in blocks if isinstance(b, dict))
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
    messages = [_parse_message(m, conv_id, i) for i, m in enumerate(raw.get("chat_messages", []))]
    account_raw = raw.get("account")
    account_id = account_raw.get("uuid") if account_raw else None
    return Conversation(
        id=conv_id,
        title=raw.get("name") or "Untitled",
        project_id=project_raw["uuid"] if project_raw else None,
        project_name=project_raw["name"] if project_raw else None,
        created_at=_parse_timestamp(raw.get("created_at")),
        updated_at=_parse_timestamp(raw.get("updated_at")),
        messages=messages,
        account_id=account_id,
        content_hash=_compute_content_hash(messages),
    )


def _build_models(raw_conversations: list[dict], raw_projects: list[dict]) -> tuple[list[Conversation], list[Project]]:
    conversations = [_parse_conversation(c) for c in raw_conversations]

    project_map: dict[str, Project] = {}
    for rp in raw_projects:
        p = Project(
            id=rp["uuid"], name=rp.get("name", "Unnamed Project"),
            created_at=_parse_timestamp(rp.get("created_at")),
        )
        project_map[p.id] = p

    for conv in conversations:
        if conv.project_id and conv.project_id not in project_map:
            project_map[conv.project_id] = Project(id=conv.project_id, name=conv.project_name or "Unknown Project")

    counts: dict[str, int] = {}
    for conv in conversations:
        if conv.project_id:
            counts[conv.project_id] = counts.get(conv.project_id, 0) + 1
    for pid, count in counts.items():
        if pid in project_map:
            project_map[pid].conversation_count = count

    return conversations, list(project_map.values())


class ClaudeExportAdapter:
    """Parses Claude.ai ZIP exports and bare conversations.json files."""

    source_name = "claude_export"

    def can_handle(self, path: Path) -> bool:
        if path.suffix == ".zip":
            try:
                with zipfile.ZipFile(path) as zf:
                    if "conversations.json" not in zf.namelist():
                        return False
                    with zf.open("conversations.json") as f:
                        data = json.load(f)
                # Claude format has 'chat_messages'; ChatGPT has 'mapping'
                if isinstance(data, list) and data:
                    return "chat_messages" in data[0]
                return True  # empty list — default to Claude
            except Exception:
                return False
        if path.suffix == ".json":
            return True
        return False

    def parse(self, path: Path) -> tuple[list[Conversation], list[Project]]:
        if path.suffix == ".zip":
            return self._parse_zip(path)
        if path.suffix == ".json":
            return self._parse_json_file(path)
        raise ExportParseError(f"Unsupported export format: {path.suffix}")

    def _parse_zip(self, path: Path) -> tuple[list[Conversation], list[Project]]:
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

    def _parse_json_file(self, path: Path) -> tuple[list[Conversation], list[Project]]:
        with open(path) as f:
            raw = json.load(f)
        if isinstance(raw, list):
            return _build_models(raw, [])
        if isinstance(raw, dict) and "conversations" in raw:
            return _build_models(raw["conversations"], raw.get("projects", []))
        raise ExportParseError("Unrecognized JSON structure")


# Module-level convenience — keeps existing call sites working
def parse_export(path: Path) -> tuple[list[Conversation], list[Project]]:
    adapter = ClaudeExportAdapter()
    return adapter.parse(path)
