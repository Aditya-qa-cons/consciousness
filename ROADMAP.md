# Roadmap

This document tracks the envisioned future state of `consciousness` — what we are building toward, in priority order.

---

## Vision

`consciousness` should be the standard layer between a user's AI conversation history and any AI assistant they use. The long-term goal is:

> **One ingest, any assistant, zero lock-in.**  
> Your history is yours — portable, private, searchable, and automatically surfaced as context wherever you work.

The current implementation covers the Claude.ai → Claude MCP path. Everything below extends that foundation.

---

## Near-term (next milestones)

### 1. Additional source adapters

The `SourceAdapter` protocol is already in place. Each new adapter adds support for a new conversation source without touching existing code.

| Source | Status | Notes |
|---|---|---|
| Claude.ai ZIP export | ✅ Done | Primary adapter |
| Claude.ai API (live sync) | Planned | Poll `/api/conversations` on a cron; needs API auth |
| ChatGPT export | Planned | `conversations.json` inside ZIP; different schema |
| Cursor AI | Planned | SQLite database at `~/.cursor/db/` |
| VS Code Copilot Chat | Planned | JSON logs in workspace `.vscode/` |
| Gemini Advanced | Planned | Google Takeout ZIP |

### 2. Incremental / watch-mode ingest

Currently `ingest` is a full scan on every run. Adding a `--watch` mode that polls for new export ZIPs (or directly watches a Claude.ai API endpoint) and only indexes new/changed conversations would allow near-real-time sync.

Design sketch:
- Store a `last_ingested_at` timestamp in a config table
- Filter `conversations` in the parser to only those with `updated_at > last_ingested_at`
- Run the normal ingest pipeline on the delta
- Update `last_ingested_at` on success

### 3. Better decision extraction

The current regex approach has ~60% recall on real conversations. Two improvements:

**Short-circuit LLM pass:** after ingest, run a cheap Claude Haiku call on each conversation to extract decisions as structured JSON. This would be optional (`--llm-extract` flag) and gated on `ANTHROPIC_API_KEY` being set, but would dramatically improve recall and reduce false positives.

**Confidence calibration:** tune the per-pattern confidence values based on a labeled test set of real conversations.

### 4. Full-text search alongside vector search

ChromaDB provides only vector (semantic) similarity. For exact queries ("find the conversation where I mentioned 'pgBouncer'"), a SQLite FTS5 index over `messages.content` would give instant, precise results.

Architecture:
- Add `CREATE VIRTUAL TABLE messages_fts USING fts5(content, tokenize="unicode61 remove_diacritics 1")`
- Add `db.fulltext_search(query)` returning message IDs
- `VectorStore.search()` and `db.fulltext_search()` results merged with RRF (Reciprocal Rank Fusion) in the MCP `search_history` tool

### 5. Web UI (read-only)

A lightweight local web interface for browsing history without opening Claude. Stack: FastAPI + HTMX or a simple Jinja2 template server. No JS build step, no external CDN dependencies.

Views needed:
- Project list
- Conversation list (paginated, filterable by project/date)
- Conversation reader (markdown-rendered messages)
- Search results page
- Decisions dashboard (all extracted decisions, grouped by topic)

```bash
consciousness ui --port 8080
# Opens http://localhost:8080
```

---

## Medium-term

### 6. Multi-account merge

Users with multiple Claude accounts (personal, work) should be able to ingest both into a single store. This requires:
- `account_id` field on `Conversation` and `Project`
- Deduplication by content hash (same conversation exported from two accounts)
- Source labels in search results

### 7. Hybrid search (BM25 + vector)

Replace pure cosine-similarity ranking with a hybrid of BM25 (keyword) and dense vector scores. BM25 handles exact-match queries better; dense vectors handle semantic queries better. Reciprocal Rank Fusion combines both without needing a learned reranker.

### 8. Conversation summarisation

Generate and store a 2–3 sentence summary of each conversation at ingest time (via Claude Haiku). These summaries would:
- Power `get_recent_context` (no need to load full message text)
- Improve `memory://context` resource quality
- Enable a "what did I work on last week" overview without message-level retrieval

### 9. Knowledge graph ✅ Done

The extracted decisions and tech choices are currently flat lists. Building a graph where "Postgres" is a node linked to multiple Decision and TechChoice nodes, with edges representing "same-project", "superseded-by", and "related-technology" relationships, would enable richer queries like:

> "What decisions have we revisited more than once?"  
> "Which technologies appear together most often in my projects?"

SQLite supports this natively via a simple `edges` table; no graph DB needed.

---

## Long-term / exploratory

### 10. Cross-assistant portability

The MCP server currently runs tools consumable by any MCP client. Extending the output layer to other protocols:

- **OpenAI function calling:** wrap tools as OpenAI function definitions
- **LangChain tool:** expose as a `BaseTool` for LangChain agents
- **REST API:** simple HTTP endpoints for any client that can make HTTP calls

### 11. Automatic memory maintenance

A background daemon that:
- Detects stale decisions (older than N months, no recent reinforcement) and marks them for review
- Surfaces "memory conflicts" (contradictory decisions on the same topic)
- Generates a weekly digest of what changed in your knowledge base

### 12. Sharing / collaboration

Selective sharing of conversations or project contexts:
- Export a project as a shareable bundle (stripped of sensitive data)
- Import a collaborator's bundle into a separate namespace in the same store
- Shared exclude rules for team-wide privacy policies

### 13. Plugin system for extractors

Allow third-party extractor plugins via Python entry points:

```toml
[project.entry-points."consciousness.extractors"]
my_extractor = "my_package.extractors:MyExtractor"
```

---

## Design constraints (non-negotiable)

These constraints are part of the product identity and should not be violated by any roadmap item:

1. **No mandatory cloud dependencies.** Every feature must work without any external service beyond Claude's own MCP transport.
2. **SQLite stays the source of truth.** Any new indexing structures (FTS5, graph edges, summaries) are derived and rebuildable.
3. **Privacy by default.** Sensitive content detection runs before any data is written. New data types (summaries, graph nodes) must also pass through redaction.
4. **Network-optional install.** The package must be installable and partially functional in an air-gapped environment. ML model downloads are lazy and can be replaced with `CONSCIOUSNESS_FAKE_ENCODER`.
5. **No ORM, no framework churn.** stdlib `sqlite3` + Pydantic is the data layer. This stays lean and avoids migration complexity.
