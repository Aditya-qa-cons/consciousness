"""Parse ChatGPT data exports into domain models.

ChatGPT Settings → Data Controls → Export Data produces a ZIP containing:
  - conversations.json  — array of all conversations

ChatGPT conversation JSON shape:
{
  "id": "...",
  "title": "...",
  "create_time": 1706000000.0,   # Unix timestamp (float)
  "update_time": 1706001000.0,
  "mapping": {
    "<node-id>": {
      "id": "<node-id>",
      "message": {
        "id": "...",
        "author": {"role": "user" | "assistant" | "system" | "tool"},
        "create_time": 1706000000.0 | null,
        "content": {
          "content_type": "text",
          "parts": ["message text here"]
        }
      } | null,
      "parent": "<parent-node-id>" | null,
      "children": ["<child-node-id>", ...]
    },
    ...
  }
}

Messages are stored as a tree (to support edit/regenerate branching).
We linearize by following the last child at each node, which gives the
final version of each turn.
"""

import hashlib
import json
import zipfile
from datetime import datetime, timezone
from pathlib import Path

from consciousness.models import Conversation, Message, Project, Role
from consciousness.parser.base import SourceAdapter


def _compute_content_hash(messages: list[Message]) -> str:
    parts = sorted(f"{m.role.value}:{m.content}" for m in messages)
    return hashlib.sha256("\n".join(parts).encode()).hexdigest()

_CHATGPT_SOURCE_PROJECT_ID = "chatgpt-default"
_CHATGPT_SOURCE_PROJECT_NAME = "ChatGPT"


class ChatGPTExportAdapter(SourceAdapter):
    """Parses ChatGPT ZIP exports."""

    source_name = "chatgpt_export"

    def can_handle(self, path: Path) -> bool:
        if path.suffix != ".zip":
            return False
        try:
            with zipfile.ZipFile(path) as zf:
                if "conversations.json" not in zf.namelist():
                    return False
                with zf.open("conversations.json") as f:
                    data = json.load(f)
                # ChatGPT format uses a 'mapping' dict; Claude uses 'chat_messages'
                return isinstance(data, list) and bool(data) and "mapping" in data[0]
        except Exception:
            return False

    def parse(self, path: Path) -> tuple[list[Conversation], list[Project]]:
        with zipfile.ZipFile(path) as zf:
            with zf.open("conversations.json") as f:
                raw_conversations = json.load(f)

        conversations = [c for raw in raw_conversations if (c := _parse_conversation(raw)) is not None]
        project = Project(id=_CHATGPT_SOURCE_PROJECT_ID, name=_CHATGPT_SOURCE_PROJECT_NAME)
        project.conversation_count = len(conversations)

        return conversations, [project] if conversations else []


def _from_unix(ts: float | None) -> datetime | None:
    if ts is None:
        return None
    return datetime.fromtimestamp(ts, tz=timezone.utc)


def _extract_content(message: dict) -> str:
    content = message.get("content") or {}
    if isinstance(content, str):
        return content
    parts = content.get("parts") or []
    return "\n".join(str(p) for p in parts if isinstance(p, str) and p.strip())


def _linearize_mapping(mapping: dict) -> list[dict]:
    """Walk the message tree following the last child at each step.

    ChatGPT stores conversations as a tree to handle regeneration and edits.
    Taking the last child at every fork gives the final accepted version.
    """
    # Find root: the node whose parent is absent from the mapping
    root_id = None
    node_ids = set(mapping)
    for node_id, node in mapping.items():
        parent = node.get("parent")
        if parent is None or parent not in node_ids:
            root_id = node_id
            break

    if root_id is None:
        return []

    messages: list[dict] = []
    current_id: str | None = root_id
    while current_id:
        node = mapping.get(current_id)
        if not node:
            break
        msg = node.get("message")
        if msg:
            role = (msg.get("author") or {}).get("role")
            if role in ("user", "assistant"):
                messages.append(msg)
        children = node.get("children") or []
        current_id = children[-1] if children else None

    return messages


def _parse_conversation(raw: dict) -> Conversation | None:
    mapping = raw.get("mapping")
    if not mapping:
        return None

    linear_msgs = _linearize_mapping(mapping)
    if not linear_msgs:
        return None

    conv_id = raw.get("id") or raw.get("conversation_id") or raw.get("title", "unknown")
    created_at = _from_unix(raw.get("create_time"))
    updated_at = _from_unix(raw.get("update_time")) or created_at
    title = raw.get("title") or "Untitled"

    messages: list[Message] = []
    for position, msg in enumerate(linear_msgs):
        role_str = (msg.get("author") or {}).get("role", "")
        try:
            role = Role("human" if role_str == "user" else role_str)
        except ValueError:
            continue  # skip unknown roles

        content = _extract_content(msg)
        if not content.strip():
            continue

        messages.append(Message(
            id=msg.get("id") or f"{conv_id}-{position}",
            conversation_id=conv_id,
            role=role,
            content=content,
            timestamp=_from_unix(msg.get("create_time")) or created_at,
            position=position,
        ))

    if not messages:
        return None

    # Re-number positions after skipping empty/system messages
    for i, msg in enumerate(messages):
        msg.position = i

    return Conversation(
        id=conv_id,
        title=title,
        project_id=_CHATGPT_SOURCE_PROJECT_ID,
        project_name=_CHATGPT_SOURCE_PROJECT_NAME,
        created_at=created_at or datetime.now(timezone.utc),
        updated_at=updated_at or datetime.now(timezone.utc),
        messages=messages,
        content_hash=_compute_content_hash(messages),
    )
