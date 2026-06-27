"""Local read-only web UI — FastAPI + Jinja2, no external CDN dependencies."""

import re
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

from consciousness.store.db import Database

_TEMPLATES_DIR = Path(__file__).parent / "templates"
_PAGE_SIZE = 30
_FENCE_RE = re.compile(r"\*\*(.*?)\*\*")


def _highlight(text: str) -> str:
    """Replace **word** markers from FTS snippet with <mark> tags."""
    return _FENCE_RE.sub(r"<mark>\1</mark>", text)


def create_app(data_dir: Path) -> FastAPI:
    """Return a configured FastAPI application bound to data_dir."""
    db = Database(data_dir / "conversations.db").connect()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        yield
        db.close()

    app = FastAPI(title="Consciousness", lifespan=lifespan)
    templates = Jinja2Templates(directory=str(_TEMPLATES_DIR))
    templates.env.filters["highlight"] = _highlight

    @app.get("/", response_class=HTMLResponse)
    async def home(request: Request):
        projects = db.list_projects()
        s = db.stats()
        return templates.TemplateResponse(request, "projects.html", {
            "projects": projects, "stats": s,
        })

    @app.get("/conversations", response_class=HTMLResponse)
    async def conversations_list(
        request: Request, project_id: str = "", page: int = 1
    ):
        offset = (page - 1) * _PAGE_SIZE
        convs = db.list_conversations(
            project_id=project_id or None,
            limit=_PAGE_SIZE + 1,
            offset=offset,
        )
        has_next = len(convs) > _PAGE_SIZE
        page_convs = convs[:_PAGE_SIZE]
        projects = db.list_projects()
        project_map = {p.id: p.name for p in projects}
        summaries = db.get_summaries([c.id for c in page_convs])
        return templates.TemplateResponse(request, "conversations.html", {
            "conversations": page_convs,
            "summaries": summaries,
            "project_map": project_map,
            "projects": projects,
            "project_id": project_id,
            "page": page,
            "has_next": has_next,
        })

    @app.get("/conversations/{conv_id}", response_class=HTMLResponse)
    async def conversation_detail(request: Request, conv_id: str):
        conv = db.get_conversation(conv_id)
        if not conv:
            return templates.TemplateResponse(
                request, "error.html", {"message": "Conversation not found."}, status_code=404
            )
        return templates.TemplateResponse(request, "conversation.html", {"conv": conv})

    @app.get("/search", response_class=HTMLResponse)
    async def search(request: Request, q: str = ""):
        results: list[dict] = []
        if q.strip():
            fts_hits = db.fulltext_search(q.strip(), limit=20)
            seen: dict[str, dict] = {}
            for hit in fts_hits:
                cid = hit["conversation_id"]
                if cid not in seen:
                    seen[cid] = hit
            for cid, hit in seen.items():
                row = db.conn.execute(
                    "SELECT id, title FROM conversations WHERE id = ?", (cid,)
                ).fetchone()
                if row:
                    results.append({**hit, "title": row["title"]})
        return templates.TemplateResponse(request, "search.html", {"q": q, "results": results})

    @app.get("/decisions", response_class=HTMLResponse)
    async def decisions(request: Request):
        all_decisions = db.list_decisions(limit=200)
        preferences = db.list_preferences()
        tech_choices = db.list_tech_choices()
        return templates.TemplateResponse(request, "decisions.html", {
            "decisions": all_decisions,
            "preferences": preferences,
            "tech_choices": tech_choices,
        })

    return app
