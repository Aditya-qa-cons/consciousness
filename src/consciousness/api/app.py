"""Consciousness REST API — exposes all MCP tools over HTTP JSON.

Endpoints (prefix /api/v1):
  GET  /health                     liveness check
  GET  /stats                      store statistics
  GET  /projects                   list all projects
  GET  /conversations              paginated conversation list
  GET  /conversations/{id}         full conversation with messages
  GET  /search?q=&limit=&project=&role=  hybrid FTS + vector search
  GET  /decisions?topic=&limit=    structured decision recall
  GET  /recent?days=&project=      recent conversations
  POST /synthesize                 generate memory-import blob
  GET  /knowledge-graph?query=&technology=&limit=  KG exploration
  GET  /context.md                 memory://context resource as markdown

Cross-assistant portability:
  GET  /openai/tools               tool definitions in OpenAI function-calling format
  POST /openai/tool-call           invoke any tool in OpenAI tool-call format
"""

from contextlib import asynccontextmanager
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import PlainTextResponse
from pydantic import BaseModel

from consciousness.memory.synthesizer import MemorySynthesizer
from consciousness.models import Role
from consciousness.store.db import Database
from consciousness.store.vectors import VectorStore

# ── request / response models ─────────────────────────────────────────────────


class SynthesizeRequest(BaseModel):
    focus_topics: list[str] = []
    project: str | None = None


class OpenAIToolCallRequest(BaseModel):
    name: str
    arguments: dict[str, Any] = {}


# ── OpenAI tool definitions ────────────────────────────────────────────────────

_OPENAI_TOOLS: list[dict] = [
    {
        "type": "function",
        "function": {
            "name": "search_history",
            "description": (
                "Semantic + keyword search over conversation history. "
                "Returns the most relevant exchanges with snippets."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "What to search for"},
                    "limit": {"type": "integer", "default": 8, "description": "Max results"},
                    "project": {"type": "string", "description": "Restrict to a project name"},
                    "role": {
                        "type": "string",
                        "enum": ["human", "assistant"],
                        "description": "Only search messages from one role",
                    },
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_projects",
            "description": "List all projects with conversation counts.",
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_project_context",
            "description": "All conversations in a named project with summaries.",
            "parameters": {
                "type": "object",
                "properties": {
                    "project_name": {"type": "string"},
                    "include_messages": {"type": "boolean", "default": False},
                },
                "required": ["project_name"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_conversation",
            "description": "Retrieve the full text of a specific conversation by ID.",
            "parameters": {
                "type": "object",
                "properties": {"conversation_id": {"type": "string"}},
                "required": ["conversation_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "recall_decision",
            "description": (
                "Recall decisions on a topic. Checks structured history first, "
                "then falls back to semantic search."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "topic": {"type": "string"},
                    "limit": {"type": "integer", "default": 5},
                },
                "required": ["topic"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_recent_context",
            "description": "Summaries of conversations from the last N days.",
            "parameters": {
                "type": "object",
                "properties": {
                    "days": {"type": "integer", "default": 7},
                    "project": {"type": "string"},
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "synthesize_memory",
            "description": "Generate a structured memory-import blob from history.",
            "parameters": {
                "type": "object",
                "properties": {
                    "focus_topics": {"type": "array", "items": {"type": "string"}},
                    "project": {"type": "string"},
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "explore_knowledge_graph",
            "description": (
                "Explore the knowledge graph. "
                "query: co_occurring_technologies | revisited_topics | technology_context"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "enum": ["co_occurring_technologies", "revisited_topics", "technology_context"],
                    },
                    "technology": {"type": "string"},
                    "limit": {"type": "integer", "default": 10},
                },
                "required": ["query"],
            },
        },
    },
]


# ── app factory ────────────────────────────────────────────────────────────────


def create_api_app(data_dir: Path) -> FastAPI:
    """Return a configured FastAPI application exposing the REST API."""

    db = Database(data_dir / "conversations.db").connect()
    vectors = VectorStore(data_dir / "vectors").connect()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        yield
        db.close()

    app = FastAPI(
        title="Consciousness API",
        description=(
            "HTTP interface to your conversation history store. "
            "Import the OpenAI tool definitions from GET /api/v1/openai/tools "
            "to use this store as a tool source in any OpenAI-compatible assistant."
        ),
        version="1",
        lifespan=lifespan,
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["GET", "POST"],
        allow_headers=["*"],
    )

    # ── health / stats ─────────────────────────────────────────────────────────

    @app.get("/api/v1/health")
    async def health():
        s = db.stats()
        return {"status": "ok", "conversations": s["conversations"]}

    @app.get("/api/v1/stats")
    async def stats():
        return db.stats()

    # ── projects ───────────────────────────────────────────────────────────────

    @app.get("/api/v1/projects")
    async def list_projects_endpoint():
        projects = db.list_projects()
        return {"projects": [p.model_dump() for p in projects]}

    # ── conversations ──────────────────────────────────────────────────────────

    @app.get("/api/v1/conversations")
    async def list_conversations_endpoint(
        project_id: str = "", page: int = 1, limit: int = 30,
    ):
        offset = (page - 1) * limit
        convs = db.list_conversations(
            project_id=project_id or None,
            limit=limit + 1,
            offset=offset,
        )
        has_next = len(convs) > limit
        page_convs = convs[:limit]
        summaries = db.get_summaries([c.id for c in page_convs])
        return {
            "page": page,
            "limit": limit,
            "has_next": has_next,
            "conversations": [
                {
                    **c.model_dump(exclude={"messages"}),
                    "message_count": c.message_count,
                    "summary": summaries[c.id].summary if c.id in summaries else None,
                }
                for c in page_convs
            ],
        }

    @app.get("/api/v1/conversations/{conv_id}")
    async def get_conversation_endpoint(conv_id: str):
        conv = db.get_conversation(conv_id)
        if not conv:
            raise HTTPException(status_code=404, detail="Conversation not found")
        return conv.model_dump()

    # ── search ─────────────────────────────────────────────────────────────────

    @app.get("/api/v1/search")
    async def search_endpoint(
        q: str = "", limit: int = 8,
        project: str | None = None, role: str | None = None,
    ):
        if not q.strip():
            return {"query": q, "results": []}

        role_filter = Role(role) if role else None
        conv_ids = None
        if project:
            projects = db.list_projects()
            matched = [p for p in projects if project.lower() in p.name.lower()]
            if matched:
                convs = db.list_conversations(project_id=matched[0].id, limit=1000)
                conv_ids = [c.id for c in convs]

        vector_hits = vectors.search(q, limit=limit * 2, role_filter=role_filter, conversation_ids=conv_ids)
        fts_role = role_filter.value if role_filter else None
        fts_hits = db.fulltext_search(q, limit=limit * 2, conversation_ids=conv_ids, role=fts_role)

        _K = 60
        rrf: dict[str, float] = {}
        best_snippet: dict[str, str] = {}

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

        ranked = sorted(rrf.items(), key=lambda x: x[1], reverse=True)[:limit]
        results = []
        for conv_id, score in ranked:
            conv = db.get_conversation(conv_id)
            results.append({
                "conversation_id": conv_id,
                "title": conv.title if conv else conv_id,
                "project_name": conv.project_name if conv else None,
                "score": round(score, 6),
                "snippet": best_snippet.get(conv_id, ""),
            })

        return {"query": q, "results": results}

    # ── decisions ──────────────────────────────────────────────────────────────

    @app.get("/api/v1/decisions")
    async def decisions_endpoint(topic: str = "", limit: int = 10):
        if topic.strip():
            decisions = db.find_active_decisions(topic)[:limit]
        else:
            decisions = db.list_decisions(limit=limit)
        return {"decisions": [d.model_dump() for d in decisions]}

    # ── recent ─────────────────────────────────────────────────────────────────

    @app.get("/api/v1/recent")
    async def recent_endpoint(days: int = 7, project: str | None = None):
        cutoff = datetime.now(timezone.utc) - timedelta(days=days)
        project_id = None
        if project:
            projects = db.list_projects()
            matched = [p for p in projects if project.lower() in p.name.lower()]
            if matched:
                project_id = matched[0].id

        convs = db.list_conversations(project_id=project_id, limit=500)
        recent = [c for c in convs if c.updated_at and c.updated_at >= cutoff]
        summaries = db.get_summaries([c.id for c in recent])

        return {
            "days": days,
            "conversations": [
                {
                    **c.model_dump(exclude={"messages"}),
                    "message_count": c.message_count,
                    "summary": summaries[c.id].summary if c.id in summaries else None,
                }
                for c in recent
            ],
        }

    # ── synthesize ─────────────────────────────────────────────────────────────

    @app.post("/api/v1/synthesize")
    async def synthesize_endpoint(body: SynthesizeRequest):
        project_id = None
        if body.project:
            projects = db.list_projects()
            matched = [p for p in projects if body.project.lower() in p.name.lower()]
            if matched:
                project_id = matched[0].id

        convs = db.list_conversations(project_id=project_id, limit=500)
        full_convs = [db.get_conversation(c.id) for c in convs[:100]]
        full_convs = [c for c in full_convs if c]

        blob = MemorySynthesizer().synthesize(full_convs, focus_topics=body.focus_topics or None)
        return {
            "generated_at": blob.generated_at.isoformat(),
            "source_conversation_count": blob.source_conversation_count,
            "focus_topics": blob.focus_topics,
            "sections": blob.sections,
            "rendered": blob.render(),
        }

    # ── knowledge graph ─────────────────────────────────────────────────────────

    @app.get("/api/v1/knowledge-graph")
    async def knowledge_graph_endpoint(
        query: str = "co_occurring_technologies",
        technology: str | None = None,
        limit: int = 10,
    ):
        if query == "co_occurring_technologies":
            pairs = db.co_occurring_technologies(limit=limit)
            return {
                "query": query,
                "pairs": [{"tech1": t1, "tech2": t2, "conversations": int(w)} for t1, t2, w in pairs],
            }

        if query == "revisited_topics":
            topics = db.revisited_topics(limit=limit)
            return {
                "query": query,
                "topics": [{"topic": t, "decision_count": c} for t, c in topics],
            }

        if query == "technology_context":
            if not technology:
                raise HTTPException(status_code=422, detail="technology parameter required for technology_context")
            node_id = f"tech:{technology.lower()}"
            node = db.get_kg_node(node_id)
            if not node:
                raise HTTPException(status_code=404, detail=f"No graph node found for '{technology}'")
            neighbors = db.get_kg_neighbors(node_id)
            return {
                "query": query,
                "node": node.model_dump(),
                "neighbors": [
                    {"edge": e.model_dump(), "node": n.model_dump()} for e, n in neighbors
                ][:limit],
                "verdicts": [
                    tc.model_dump()
                    for tc in db.list_tech_choices()
                    if tc.technology.lower() == technology.lower()
                ][:5],
            }

        raise HTTPException(
            status_code=422,
            detail="query must be co_occurring_technologies, revisited_topics, or technology_context",
        )

    # ── context.md resource ────────────────────────────────────────────────────

    @app.get("/api/v1/context.md", response_class=PlainTextResponse)
    async def context_md_endpoint():
        lines = ["# Your Context\n"]
        decisions = db.list_decisions(limit=20)
        if decisions:
            lines.append("## Recent Decisions\n")
            for d in decisions[:10]:
                lines.append(f"- **{d.topic}**: {d.conclusion[:200]}")
            lines.append("")

        tech = db.list_tech_choices()
        if tech:
            lines.append("## Technology Choices\n")
            seen: set[str] = set()
            for tc in tech:
                if tc.technology not in seen:
                    lines.append(f"- **{tc.technology}**: {tc.verdict[:150]}")
                    seen.add(tc.technology)
            lines.append("")

        cutoff = datetime.now(timezone.utc) - timedelta(days=7)
        recent_convs = db.list_conversations(limit=100)
        recent = [c for c in recent_convs if c.updated_at and c.updated_at >= cutoff]
        if recent:
            lines.append(f"## Active This Week ({len(recent)} conversations)\n")
            for conv in recent[:5]:
                date = conv.updated_at.strftime("%Y-%m-%d") if conv.updated_at else "?"
                lines.append(f"- {conv.title} ({date})")
            lines.append("")

        return PlainTextResponse("\n".join(lines), media_type="text/markdown")

    # ── cross-assistant: OpenAI function definitions ───────────────────────────

    @app.get("/api/v1/openai/tools")
    async def openai_tools_endpoint():
        """Return all tools in OpenAI function-calling format.

        Use this URL as the tool source in any OpenAI-compatible assistant:
          import openai, httpx
          tools = httpx.get("http://localhost:8765/api/v1/openai/tools").json()
          response = openai.chat.completions.create(model="gpt-4o", tools=tools, ...)
        """
        return _OPENAI_TOOLS

    @app.post("/api/v1/openai/tool-call")
    async def openai_tool_call_endpoint(body: OpenAIToolCallRequest):
        """Invoke a tool by name with arguments dict, returning its text output.

        Compatible with OpenAI tool-call response format:
          tool_result = httpx.post(url, json={"name": tc.function.name,
                                               "arguments": json.loads(tc.function.arguments)})
        """
        name = body.name
        args = body.arguments

        if name == "search_history":
            result = await _handle_search(db, vectors, args)
        elif name == "list_projects":
            result = await _handle_list_projects(db)
        elif name == "get_project_context":
            result = await _handle_get_project_context(db, args)
        elif name == "get_conversation":
            result = await _handle_get_conversation(db, args)
        elif name == "recall_decision":
            result = await _handle_recall_decision(db, vectors, args)
        elif name == "get_recent_context":
            result = await _handle_get_recent_context(db, args)
        elif name == "synthesize_memory":
            result = await _handle_synthesize_memory(db, args)
        elif name == "explore_knowledge_graph":
            result = await _handle_explore_kg(db, args)
        else:
            raise HTTPException(status_code=404, detail=f"Unknown tool: {name}")

        return {"role": "tool", "name": name, "content": result}

    return app


# ── tool-call handler implementations ─────────────────────────────────────────
# These mirror the MCP server handlers but return plain strings instead of
# list[TextContent], keeping them reusable from both the REST API and tests.


async def _handle_search(db: Database, vectors: VectorStore, args: dict) -> str:
    query = args.get("query", "")
    limit = int(args.get("limit", 8))
    project_filter = args.get("project")
    role_filter = Role(args["role"]) if args.get("role") else None

    conv_ids = None
    if project_filter:
        projects = db.list_projects()
        matched = [p for p in projects if project_filter.lower() in p.name.lower()]
        if matched:
            convs = db.list_conversations(project_id=matched[0].id, limit=1000)
            conv_ids = [c.id for c in convs]

    vector_hits = vectors.search(query, limit=limit * 2, role_filter=role_filter, conversation_ids=conv_ids)
    fts_role = role_filter.value if role_filter else None
    fts_hits = db.fulltext_search(query, limit=limit * 2, conversation_ids=conv_ids, role=fts_role)

    _K = 60
    rrf: dict[str, float] = {}
    best_snippet: dict[str, str] = {}
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
        return "No results found."

    ranked = sorted(rrf.items(), key=lambda x: x[1], reverse=True)[:limit]
    lines = [f"Found {len(ranked)} results for: **{query}**\n"]
    for conv_id, _ in ranked:
        conv = db.get_conversation(conv_id)
        title = conv.title if conv else conv_id
        lines.append(f"### [{title}] (id: {conv_id})")
        lines.append(f"> {best_snippet[conv_id]}")
        lines.append("")
    return "\n".join(lines)


async def _handle_list_projects(db: Database) -> str:
    projects = db.list_projects()
    if not projects:
        return "No projects found."
    lines = ["## Your Projects\n"]
    for p in projects:
        date = p.created_at.strftime("%Y-%m-%d") if p.created_at else "unknown"
        acct = f" · account: {p.account_id}" if p.account_id else ""
        lines.append(f"- **{p.name}** — {p.conversation_count} conversations (created {date}){acct}")
    return "\n".join(lines)


async def _handle_get_project_context(db: Database, args: dict) -> str:
    name = args.get("project_name", "")
    include_messages = args.get("include_messages", False)
    projects = db.list_projects()
    matched = [p for p in projects if name.lower() in p.name.lower()]
    if not matched:
        return f"No project found matching '{name}'"
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
    return "\n".join(lines)


async def _handle_get_conversation(db: Database, args: dict) -> str:
    conv = db.get_conversation(args.get("conversation_id", ""))
    if not conv:
        return "Conversation not found."
    return conv.as_text()


async def _handle_recall_decision(db: Database, vectors: VectorStore, args: dict) -> str:
    topic = args.get("topic", "")
    limit = int(args.get("limit", 5))
    structured = db.find_active_decisions(topic)
    lines = [f"## Decisions about: {topic}\n"]
    if structured:
        lines.append("### From decision history\n")
        for d in structured[:limit]:
            conv = db.get_conversation(d.conversation_id)
            conv_title = conv.title if conv else d.conversation_id
            lines.append(f"**{d.topic}** _(confidence: {d.confidence:.0%}, from: {conv_title})_")
            lines.append(d.conclusion)
            lines.append("")
    hits = vectors.search(f"decision conclusion recommendation {topic}", limit=limit, role_filter=Role.assistant)
    if hits:
        lines.append("### From conversation search\n")
        for hit in hits:
            conv = db.get_conversation(hit.conversation_id)
            title = conv.title if conv else hit.conversation_id
            lines.append(f"**{title}** _{hit.relevance_label} relevance_")
            lines.append(hit.chunk_text[:600])
            lines.append("")
    if len(lines) == 2:
        return f"No decisions found about '{topic}'."
    return "\n".join(lines)


async def _handle_get_recent_context(db: Database, args: dict) -> str:
    days = int(args.get("days", 7))
    project_filter = args.get("project")
    cutoff = datetime.now(timezone.utc) - timedelta(days=days)
    project_id = None
    if project_filter:
        projects = db.list_projects()
        matched = [p for p in projects if project_filter.lower() in p.name.lower()]
        if matched:
            project_id = matched[0].id
    convs = db.list_conversations(project_id=project_id, limit=500)
    recent = [c for c in convs if c.updated_at and c.updated_at >= cutoff]
    if not recent:
        return f"No conversations in the last {days} days."
    summaries = db.get_summaries([c.id for c in recent])
    lines = [f"## Recent context — last {days} days ({len(recent)} conversations)\n"]
    for conv in recent:
        lines.append(f"### {conv.title}")
        lines.append(f"Updated: {conv.updated_at.strftime('%Y-%m-%d %H:%M')}")
        s = summaries.get(conv.id)
        if s:
            lines.append(s.summary)
        lines.append("")
    return "\n".join(lines)


async def _handle_synthesize_memory(db: Database, args: dict) -> str:
    focus_topics = args.get("focus_topics", [])
    project_filter = args.get("project")
    project_id = None
    if project_filter:
        projects = db.list_projects()
        matched = [p for p in projects if project_filter.lower() in p.name.lower()]
        if matched:
            project_id = matched[0].id
    convs = db.list_conversations(project_id=project_id, limit=500)
    full_convs = [db.get_conversation(c.id) for c in convs[:100]]
    full_convs = [c for c in full_convs if c]
    blob = MemorySynthesizer().synthesize(full_convs, focus_topics=focus_topics or None)
    return blob.render()


async def _handle_explore_kg(db: Database, args: dict) -> str:
    query = args.get("query", "")
    limit = int(args.get("limit", 10))
    if query == "co_occurring_technologies":
        pairs = db.co_occurring_technologies(limit=limit)
        if not pairs:
            return "No co-occurrence data. Run `consciousness rebuild-graph` first."
        lines = ["## Technologies That Appear Together\n"]
        for t1, t2, w in pairs:
            lines.append(f"- **{t1}** + **{t2}** — {int(w)} conversations")
        return "\n".join(lines)
    if query == "revisited_topics":
        topics = db.revisited_topics(limit=limit)
        if not topics:
            return "No revisited decision topics found."
        lines = ["## Decision Topics Revisited Multiple Times\n"]
        for topic, count in topics:
            lines.append(f"- **{topic}** — {count} decisions recorded")
        return "\n".join(lines)
    if query == "technology_context":
        tech = args.get("technology", "").strip()
        if not tech:
            return "Provide a `technology` name for this query."
        node_id = f"tech:{tech.lower()}"
        node = db.get_kg_node(node_id)
        if not node:
            return f"No graph node found for '{tech}'. Run `consciousness rebuild-graph` first."
        neighbors = db.get_kg_neighbors(node_id)
        lines = [f"## Knowledge Graph: {node.label}\n"]
        co_occurs = [(e, n) for e, n in neighbors if e.relation == "co_occurs_with"]
        if co_occurs:
            lines.append("### Co-occurs with")
            for e, n in sorted(co_occurs, key=lambda x: -x[0].weight)[:limit]:
                lines.append(f"- **{n.label}** ({int(e.weight)} conversations)")
        return "\n".join(lines)
    return f"Unknown query '{query}'."
