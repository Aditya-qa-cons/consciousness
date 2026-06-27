"""MCP server — exposes your Claude history as account-independent tools and resources.

Tool surface:
  search_history          semantic search over all conversations
  get_project_context     all conversations in a named project
  recall_decision         structured DB lookup first, vector search fallback
  get_recent_context      summaries of recent N days of conversations
  get_conversation        retrieve a full conversation by ID
  list_projects           all projects with conversation counts
  synthesize_memory       generate a Claude memory-import blob

Resource surface:
  memory://context        auto-injected condensed view of decisions + recent activity

Business-logic handlers are module-level async functions so tests can call
them directly without going through the MCP transport layer.
"""

from contextlib import asynccontextmanager
from datetime import datetime, timedelta, timezone
from pathlib import Path

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import Resource, TextContent, TextResourceContents, Tool
from pydantic import AnyUrl

from consciousness.memory.synthesizer import MemorySynthesizer
from consciousness.models import Role
from consciousness.store.db import Database
from consciousness.store.vectors import VectorStore

_data_dir: Path | None = None


@asynccontextmanager
async def _lifespan(server: Server):
    assert _data_dir is not None, "Call run(data_dir) to set _data_dir before serving"
    db = Database(_data_dir / "conversations.db").connect()
    vectors = VectorStore(_data_dir / "vectors").connect()
    try:
        yield {"db": db, "vectors": vectors}
    finally:
        db.close()


app = Server("consciousness", lifespan=_lifespan)


def _stores() -> tuple[Database, VectorStore]:
    ctx = app.request_context
    return ctx.lifespan_context["db"], ctx.lifespan_context["vectors"]


# ── resources ────────────────────────────────────────────────────────────────


@app.list_resources()
async def handle_list_resources() -> list[Resource]:
    return [
        Resource(
            uri=AnyUrl("memory://context"),
            name="Memory Context",
            description=(
                "Auto-injected at session start: your recent decisions, technology choices, "
                "preferences, and active conversations — no tool call needed."
            ),
            mimeType="text/markdown",
        )
    ]


@app.read_resource()
async def handle_read_resource(uri: AnyUrl) -> list[TextResourceContents]:
    db, _ = _stores()
    content = await _build_memory_resource(db)
    return [TextResourceContents(uri=uri, mimeType="text/markdown", text=content)]


async def _build_memory_resource(db: Database) -> str:
    """Condensed markdown document for automatic context injection."""
    lines = ["# Your Context\n"]

    # Recent decisions (non-superseded, last 20)
    decisions = db.list_decisions(limit=20)
    if decisions:
        lines.append("## Recent Decisions\n")
        for d in decisions[:10]:
            lines.append(f"- **{d.topic}**: {d.conclusion[:200]}")
        lines.append("")

    # Tech choices
    tech = db.list_tech_choices()
    if tech:
        lines.append("## Technology Choices\n")
        seen_tech: set[str] = set()
        for tc in tech:
            if tc.technology not in seen_tech:
                lines.append(f"- **{tc.technology}**: {tc.verdict[:150]}")
                seen_tech.add(tc.technology)
        lines.append("")

    # Active conversations — last 7 days
    cutoff = datetime.now(timezone.utc) - timedelta(days=7)
    recent_convs = db.list_conversations(limit=100)
    recent = [c for c in recent_convs if c.updated_at and c.updated_at >= cutoff]
    if recent:
        lines.append(f"## Active This Week ({len(recent)} conversations)\n")
        for conv in recent[:5]:
            lines.append(f"- {conv.title} ({conv.updated_at.strftime('%Y-%m-%d') if conv.updated_at else '?'})")
        lines.append("")

    return "\n".join(lines)


# ── tool definitions ────────────────────────────────────────────────────────


@app.list_tools()
async def list_tools() -> list[Tool]:
    return [
        Tool(
            name="search_history",
            description=(
                "Semantic search over all your Claude conversations. "
                "Returns the most relevant Q+A exchanges with relevance scores."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "What to search for"},
                    "limit": {"type": "integer", "default": 8, "description": "Max results to return"},
                    "project": {"type": "string", "description": "Optional: restrict to a specific project name"},
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
                    "project_name": {"type": "string", "description": "Name of the project (partial match supported)"},
                    "include_messages": {
                        "type": "boolean", "default": False,
                        "description": "Include full message text",
                    },
                },
                "required": ["project_name"],
            },
        ),
        Tool(
            name="recall_decision",
            description=(
                "Recall decisions and conclusions on a topic. "
                "Checks structured decision history first (fast, exact), then falls back to semantic search."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "topic": {"type": "string", "description": "Decision topic (e.g. 'database', 'auth strategy')"},
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
                "properties": {"conversation_id": {"type": "string"}},
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
                        "description": "Optional topics to emphasize",
                    },
                    "project": {"type": "string", "description": "Optional: limit to one project"},
                },
            },
        ),
    ]


# ── tool dispatcher ──────────────────────────────────────────────────────────


@app.call_tool()
async def call_tool(name: str, arguments: dict) -> list[TextContent]:
    db, vectors = _stores()
    match name:
        case "search_history":
            return await search_history(db, vectors, arguments)
        case "get_project_context":
            return await get_project_context(db, arguments)
        case "recall_decision":
            return await recall_decision(db, vectors, arguments)
        case "get_recent_context":
            return await get_recent_context(db, arguments)
        case "get_conversation":
            return await get_conversation(db, arguments)
        case "list_projects":
            return await list_projects(db)
        case "synthesize_memory":
            return await synthesize_memory(db, arguments)
        case _:
            return [TextContent(type="text", text=f"Unknown tool: {name}")]


# ── business-logic handlers ──────────────────────────────────────────────────


async def search_history(db: Database, vectors: VectorStore, args: dict) -> list[TextContent]:
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

    # Run vector search and FTS in parallel logical steps; merge with RRF.
    vector_hits = vectors.search(query, limit=limit * 2, role_filter=role_filter, conversation_ids=conv_ids)
    fts_role = role_filter.value if role_filter else None
    fts_hits = db.fulltext_search(query, limit=limit * 2, conversation_ids=conv_ids, role=fts_role)

    # RRF: score = Σ 1/(k + rank), k=60, rank is 1-based position in each list.
    _K = 60
    rrf: dict[str, float] = {}
    best_snippet: dict[str, str] = {}   # conversation_id → best text excerpt

    for rank, hit in enumerate(vector_hits, start=1):
        cid = hit.conversation_id
        rrf[cid] = rrf.get(cid, 0.0) + 1.0 / (_K + rank)
        if cid not in best_snippet:
            best_snippet[cid] = hit.chunk_text[:400]

    for rank, hit in enumerate(fts_hits, start=1):
        cid = hit["conversation_id"]
        rrf[cid] = rrf.get(cid, 0.0) + 1.0 / (_K + rank)
        if cid not in best_snippet:
            best_snippet[cid] = hit["snippet"]

    if not rrf:
        return [TextContent(type="text", text="No results found.")]

    ranked = sorted(rrf.items(), key=lambda x: x[1], reverse=True)[:limit]

    lines = [f"Found {len(ranked)} results for: **{query}**\n"]
    for conv_id, score in ranked:
        conv = db.get_conversation(conv_id)
        title = conv.title if conv else conv_id
        lines.append(f"### [{title}] (id: {conv_id})")
        lines.append(f"> {best_snippet[conv_id]}")
        lines.append("")

    return [TextContent(type="text", text="\n".join(lines))]


async def get_project_context(db: Database, args: dict) -> list[TextContent]:
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


async def recall_decision(db: Database, vectors: VectorStore, args: dict) -> list[TextContent]:
    topic = args["topic"]
    limit = args.get("limit", 5)

    # Structured DB lookup first — fast, exact, uses extracted facts
    structured = db.find_active_decisions(topic)
    lines = [f"## Decisions & conclusions about: {topic}\n"]

    if structured:
        lines.append("### From decision history\n")
        for d in structured[:limit]:
            conv = db.get_conversation(d.conversation_id)
            conv_title = conv.title if conv else d.conversation_id
            lines.append(f"**{d.topic}** _(confidence: {d.confidence:.0%}, from: {conv_title})_")
            lines.append(d.conclusion)
            lines.append("")

    # Supplement with vector search for full-sentence context
    hits = vectors.search(
        f"decision conclusion recommendation {topic}",
        limit=limit,
        role_filter=Role.assistant,
    )
    if hits:
        lines.append("### From conversation search\n")
        for hit in hits:
            conv = db.get_conversation(hit.conversation_id)
            title = conv.title if conv else hit.conversation_id
            lines.append(f"**{title}** _{hit.relevance_label} relevance_")
            lines.append(hit.chunk_text[:600])
            lines.append("")

    if len(lines) == 2:  # only the header
        return [TextContent(type="text", text=f"No decisions found about '{topic}'.")]

    return [TextContent(type="text", text="\n".join(lines))]


async def get_recent_context(db: Database, args: dict) -> list[TextContent]:
    days = args.get("days", 7)
    project_filter = args.get("project")
    cutoff = datetime.now(timezone.utc) - timedelta(days=days)

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


async def get_conversation(db: Database, args: dict) -> list[TextContent]:
    conv = db.get_conversation(args["conversation_id"])
    if not conv:
        return [TextContent(type="text", text="Conversation not found.")]
    return [TextContent(type="text", text=conv.as_text())]


async def list_projects(db: Database) -> list[TextContent]:
    projects = db.list_projects()
    if not projects:
        return [TextContent(type="text", text="No projects found. Run `consciousness ingest` first.")]

    lines = ["## Your Projects\n"]
    for p in projects:
        date = p.created_at.strftime("%Y-%m-%d") if p.created_at else "unknown"
        lines.append(f"- **{p.name}** — {p.conversation_count} conversations (created {date})")

    return [TextContent(type="text", text="\n".join(lines))]


async def synthesize_memory(db: Database, args: dict) -> list[TextContent]:
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


async def run(data_dir: Path):
    global _data_dir
    _data_dir = data_dir

    async with stdio_server() as (read_stream, write_stream):
        await app.run(read_stream, write_stream, app.create_initialization_options())
