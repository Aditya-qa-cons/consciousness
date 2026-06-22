"""MCP server — exposes your Claude history as account-independent tools.

Tool surface:
  search_history          semantic search over all conversations
  get_project_context     all conversations in a named project
  recall_decision         surface decisions/conclusions on a topic
  get_recent_context      summaries of recent N days of conversations
  get_conversation        retrieve a full conversation by ID
  list_projects           all projects with conversation counts
  synthesize_memory       generate a Claude memory-import blob
"""

from datetime import datetime, timedelta
from pathlib import Path

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import TextContent, Tool

from consciousness.memory.synthesizer import MemorySynthesizer
from consciousness.models import Role
from consciousness.store.db import Database
from consciousness.store.vectors import VectorStore

app = Server("consciousness")


def _get_stores(ctx) -> tuple[Database, VectorStore]:
    """Retrieve DB and vector store from server lifespan context."""
    return ctx.request_context.lifespan_context["db"], ctx.request_context.lifespan_context["vectors"]


# ── tool definitions ────────────────────────────────────────────────────────


@app.list_tools()
async def list_tools() -> list[Tool]:
    return [
        Tool(
            name="search_history",
            description=(
                "Semantic search over all your Claude conversations. "
                "Returns the most relevant message snippets with conversation context."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "What to search for"},
                    "limit": {"type": "integer", "default": 8, "description": "Max results to return"},
                    "project": {
                        "type": "string",
                        "description": "Optional: restrict search to a specific project name",
                    },
                    "role": {
                        "type": "string",
                        "enum": ["human", "assistant"],
                        "description": "Optional: only search messages from one role",
                    },
                },
                "required": ["query"],
            },
        ),
        Tool(
            name="get_project_context",
            description="Retrieve all conversations in a named project with their summaries.",
            inputSchema={
                "type": "object",
                "properties": {
                    "project_name": {
                        "type": "string",
                        "description": "Name of the project (partial match supported)",
                    },
                    "include_messages": {
                        "type": "boolean",
                        "default": False,
                        "description": "Include full message text",
                    },
                },
                "required": ["project_name"],
            },
        ),
        Tool(
            name="recall_decision",
            description=(
                "Find decisions, conclusions, and settled choices on a topic across your history. "
                "Searches assistant responses for actionable conclusions."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "topic": {
                        "type": "string",
                        "description": "Decision topic to look up (e.g. 'database choice', 'auth strategy')",
                    },
                    "limit": {"type": "integer", "default": 5},
                },
                "required": ["topic"],
            },
        ),
        Tool(
            name="get_recent_context",
            description="Summaries of conversations from the last N days, newest first.",
            inputSchema={
                "type": "object",
                "properties": {
                    "days": {"type": "integer", "default": 7, "description": "How many days back to look"},
                    "project": {"type": "string", "description": "Optional project filter"},
                },
            },
        ),
        Tool(
            name="get_conversation",
            description="Retrieve the full text of a specific conversation by ID.",
            inputSchema={
                "type": "object",
                "properties": {
                    "conversation_id": {"type": "string"},
                },
                "required": ["conversation_id"],
            },
        ),
        Tool(
            name="list_projects",
            description="List all projects with conversation counts and dates.",
            inputSchema={"type": "object", "properties": {}},
        ),
        Tool(
            name="synthesize_memory",
            description=(
                "Generate a structured memory-import blob from your history. "
                "Paste the output into Claude's memory import box to seed any account."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "focus_topics": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Optional topics to emphasize in the synthesis",
                    },
                    "project": {"type": "string", "description": "Optional: limit synthesis to one project"},
                },
            },
        ),
    ]


# ── tool handlers ────────────────────────────────────────────────────────────


@app.call_tool()
async def call_tool(name: str, arguments: dict, ctx=None) -> list[TextContent]:
    db, vectors = _get_stores(ctx)

    match name:
        case "search_history":
            return await _search_history(db, vectors, arguments)
        case "get_project_context":
            return await _get_project_context(db, arguments)
        case "recall_decision":
            return await _recall_decision(db, vectors, arguments)
        case "get_recent_context":
            return await _get_recent_context(db, arguments)
        case "get_conversation":
            return await _get_conversation(db, arguments)
        case "list_projects":
            return await _list_projects(db)
        case "synthesize_memory":
            return await _synthesize_memory(db, arguments)
        case _:
            return [TextContent(type="text", text=f"Unknown tool: {name}")]


async def _search_history(db: Database, vectors: VectorStore, args: dict) -> list[TextContent]:
    query = args["query"]
    limit = args.get("limit", 8)
    project_filter = args.get("project")
    role_filter = Role(args["role"]) if args.get("role") else None

    conv_ids = None
    if project_filter:
        projects = db.list_projects()
        matched = [p for p in projects if project_filter.lower() in p.name.lower()]
        if matched:
            convs = db.list_conversations(project_id=matched[0].id, limit=1000)
            conv_ids = [c.id for c in convs]

    hits = vectors.search(query, limit=limit, role_filter=role_filter, conversation_ids=conv_ids)

    if not hits:
        return [TextContent(type="text", text="No results found.")]

    lines = [f"Found {len(hits)} results for: **{query}**\n"]
    seen_convs: dict[str, str] = {}

    for hit in hits:
        if hit.conversation_id not in seen_convs:
            conv = db.get_conversation(hit.conversation_id)
            seen_convs[hit.conversation_id] = conv.title if conv else hit.conversation_id

        title = seen_convs[hit.conversation_id]
        lines.append(f"### [{title}] (id: {hit.conversation_id})")
        lines.append(f"> {hit.chunk_text[:400]}")
        lines.append("")

    return [TextContent(type="text", text="\n".join(lines))]


async def _get_project_context(db: Database, args: dict) -> list[TextContent]:
    name = args["project_name"]
    include_messages = args.get("include_messages", False)

    projects = db.list_projects()
    matched = [p for p in projects if name.lower() in p.name.lower()]

    if not matched:
        return [TextContent(type="text", text=f"No project found matching '{name}'")]

    project = matched[0]
    conversations = db.list_conversations(project_id=project.id, limit=200)

    lines = [f"## Project: {project.name}", f"{len(conversations)} conversations\n"]

    for conv in conversations:
        lines.append(f"### {conv.title}")
        lines.append(f"Updated: {conv.updated_at.strftime('%Y-%m-%d') if conv.updated_at else 'unknown'}")
        if include_messages:
            full = db.get_conversation(conv.id)
            if full:
                lines.append(full.as_text()[:3000])
        lines.append("")

    return [TextContent(type="text", text="\n".join(lines))]


async def _recall_decision(db: Database, vectors: VectorStore, args: dict) -> list[TextContent]:
    topic = args["topic"]
    limit = args.get("limit", 5)

    # Search only assistant messages — decisions live in responses
    hits = vectors.search(
        f"decision conclusion recommendation {topic}",
        limit=limit * 2,
        role_filter=Role.assistant,
    )
    hits = hits[:limit]

    if not hits:
        return [TextContent(type="text", text=f"No decisions found about '{topic}'.")]

    lines = [f"## Decisions & conclusions about: {topic}\n"]
    for hit in hits:
        conv = db.get_conversation(hit.conversation_id)
        title = conv.title if conv else hit.conversation_id
        lines.append(f"### {title}")
        lines.append(hit.chunk_text[:600])
        lines.append("")

    return [TextContent(type="text", text="\n".join(lines))]


async def _get_recent_context(db: Database, args: dict) -> list[TextContent]:
    days = args.get("days", 7)
    project_filter = args.get("project")
    cutoff = datetime.utcnow() - timedelta(days=days)

    project_id = None
    if project_filter:
        projects = db.list_projects()
        matched = [p for p in projects if project_filter.lower() in p.name.lower()]
        if matched:
            project_id = matched[0].id

    conversations = db.list_conversations(project_id=project_id, limit=500)
    recent = [c for c in conversations if c.updated_at and c.updated_at >= cutoff]

    if not recent:
        return [TextContent(type="text", text=f"No conversations in the last {days} days.")]

    lines = [f"## Recent context — last {days} days ({len(recent)} conversations)\n"]
    for conv in recent:
        lines.append(f"### {conv.title}")
        lines.append(f"Updated: {conv.updated_at.strftime('%Y-%m-%d %H:%M')}")
        full = db.get_conversation(conv.id)
        if full and full.messages:
            last_human = next((m for m in reversed(full.messages) if m.role == Role.human), None)
            if last_human:
                lines.append(f"Last question: {last_human.content[:200]}")
        lines.append("")

    return [TextContent(type="text", text="\n".join(lines))]


async def _get_conversation(db: Database, args: dict) -> list[TextContent]:
    conv = db.get_conversation(args["conversation_id"])
    if not conv:
        return [TextContent(type="text", text="Conversation not found.")]
    return [TextContent(type="text", text=conv.as_text())]


async def _list_projects(db: Database) -> list[TextContent]:
    projects = db.list_projects()
    if not projects:
        return [TextContent(type="text", text="No projects found. Run `consciousness ingest` first.")]

    lines = ["## Your Projects\n"]
    for p in projects:
        date = p.created_at.strftime("%Y-%m-%d") if p.created_at else "unknown"
        lines.append(f"- **{p.name}** — {p.conversation_count} conversations (created {date})")

    return [TextContent(type="text", text="\n".join(lines))]


async def _synthesize_memory(db: Database, args: dict) -> list[TextContent]:
    focus_topics = args.get("focus_topics", [])
    project_filter = args.get("project")

    project_id = None
    if project_filter:
        projects = db.list_projects()
        matched = [p for p in projects if project_filter.lower() in p.name.lower()]
        if matched:
            project_id = matched[0].id

    conversations = db.list_conversations(project_id=project_id, limit=500)
    full_convs = [db.get_conversation(c.id) for c in conversations[:100]]
    full_convs = [c for c in full_convs if c]

    synthesizer = MemorySynthesizer()
    blob = synthesizer.synthesize(full_convs, focus_topics=focus_topics or None)

    output = [
        "## Memory Import Blob",
        f"_Generated from {blob.source_conversation_count} conversations. "
        "Paste into Claude → Settings → Memory → Import._\n",
        "```",
        blob.render(),
        "```",
    ]
    return [TextContent(type="text", text="\n".join(output))]


# ── server entrypoint ────────────────────────────────────────────────────────


def create_server(data_dir: Path) -> tuple[Server, Database, VectorStore]:
    db = Database(data_dir / "conversations.db").connect()
    vectors = VectorStore(data_dir / "vectors").connect()
    return app, db, vectors


async def run(data_dir: Path):
    db = Database(data_dir / "conversations.db").connect()
    vectors = VectorStore(data_dir / "vectors").connect()

    async with stdio_server() as (read_stream, write_stream):
        await app.run(
            read_stream,
            write_stream,
            app.create_initialization_options(),
            # Pass stores into request context
            lifespan_context={"db": db, "vectors": vectors},
        )
