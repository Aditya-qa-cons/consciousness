# CLAUDE.md — consciousness codebase guide

This file is read by Claude Code at the start of every session. It describes the project structure, conventions, and key facts needed to work effectively in this codebase.

---

## What this project is

`consciousness` indexes a user's Claude.ai conversation history into a local SQLite + ChromaDB store and exposes it as an MCP server. Key goals: local-first (no API key needed for search), portable (single `.consciousness` ZIP bundle), privacy-preserving (sensitive content redacted at ingest).

---

## Repository structure

```
src/consciousness/
├── models.py              # All Pydantic domain models — start here
├── cli.py                 # Click commands: ingest serve stats export import-bundle rebuild-index exclude
├── parser/
│   ├── base.py            # SourceAdapter Protocol
│   ├── claude_export.py   # Claude.ai ZIP/JSON parser + ClaudeExportAdapter
│   └── __init__.py        # Auto-detect adapter registry; parse_export() convenience fn
├── store/
│   ├── db.py              # SQLite: all structured data (conversations, decisions, etc.)
│   └── vectors.py         # ChromaDB: exchange-level embeddings + search
├── extractors/
│   ├── knowledge.py       # Regex-based decision/preference/tech-choice extraction
│   └── sensitive.py       # API key / secret detection and redaction
├── mcp_server/
│   └── server.py          # MCP Server with 7 tools + memory://context resource
└── memory/
    └── synthesizer.py     # MemoryBlob generation (optionally calls Claude API)

tests/
├── conftest.py            # Shared unit test helpers: make_conversation, make_message, etc.
├── integration/
│   ├── conftest.py        # FakeEncoder + seeded DB/vector fixtures
│   └── ...                # Integration tests (marked @pytest.mark.integration)
└── test_*.py              # Unit tests
```

---

## Key conventions

### Python version and style
- Python 3.11+ — use `X | Y` union syntax, `match` statements, `datetime.now(timezone.utc)`
- Pydantic v2 — use `Field(default_factory=...)` not bare `default=`
- Line length 120, ruff selects E/F/I
- No comments unless the WHY is non-obvious; no docstrings on simple methods

### Models
- Domain models are in `models.py`. Do not add logic there; models are data containers.
- `Conversation.human_turns` and `Conversation.assistant_turns` are properties, not methods.
- All datetimes are timezone-aware UTC. Use `datetime.now(timezone.utc)`, never `datetime.utcnow()`.

### Database (db.py)
- No ORM. Plain `sqlite3` with `row_factory = sqlite3.Row`.
- Always call `db.commit()` after writes; it is not automatic.
- `db.connect()` returns `self` for chaining.
- Temporal pattern for decisions: `superseded_by IS NULL` = active; use `db.find_active_decisions()`.
- `is_excluded(conv)` does Python-side fnmatch for `title_glob` rules after loading all rules once.

### Vector store (vectors.py)
- `index_conversation(conv)` is the only public write method. Do not call `_index_exchange` directly.
- `search()` has `score_threshold=0.8` default. Lower = stricter (fewer but better results).
- `VectorHit.relevance_label` is `"high"/"medium"/"low"` based on cosine distance bands.
- `CONSCIOUSNESS_FAKE_ENCODER=1` env var bypasses HuggingFace download. Always set this in tests that call the CLI via CliRunner.

### Parser / adapters
- New source formats: implement `SourceAdapter` Protocol in `parser/`, register in `parser/__init__.py`.
- `parse_export(path)` is the public API; it auto-detects the adapter.
- `ClaudeExportAdapter.can_handle()` checks for ZIP + `conversations.json` inside.

### MCP server
- Business logic handlers are module-level async functions (not methods on the Server class) so tests can call them directly without going through the transport.
- `recall_decision` hits the DB first (structured, fast), then supplements with vector search.
- `memory://context` resource is auto-injected by MCP hosts at session start — it's not a tool.

### Extractors
- `extract_decisions()` scans `assistant_turns`; `extract_preferences()` scans `human_turns`.
- `apply_temporal_tracking(new, existing)` returns `(old_id, new_id)` pairs for supersession; call `db.supersede_decision()` for each.
- `redact(text)` returns `(clean_text, findings_list)`. Apply to `msg.content` before any DB/vector write.

---

## Running tests

```bash
# Fast unit tests (no ML, no network)
pytest tests/ -v

# Integration tests (FakeEncoder, ChromaDB)
pytest tests/ -m integration -v

# All tests
pytest tests/ -v

# Lint
python -m ruff check src/ tests/
python -m ruff check src/ tests/ --fix   # auto-fix what's fixable
```

Unit tests live in `tests/test_*.py`. Integration tests are in `tests/integration/` and are marked `@pytest.mark.integration`. The split means `pytest tests/` without `-m integration` runs only the fast unit suite.

The integration `conftest.py` defines:
- `FakeEncoder` — deterministic bag-of-words (MD5-based), no downloads
- `_make_store(path)` — creates VectorStore with FakeEncoder injected
- `seeded_vector_store` — fixture with 4 seeded messages across 2 conversations
- `full_stores` — both DB and VectorStore seeded, returned as `tuple[Database, VectorStore]`

---

## Common tasks

### Add a new CLI command
```python
@cli.command("my-command")
@click.argument("...")
@click.pass_context
def my_command(ctx, ...):
    data_dir: Path = ctx.obj["data_dir"]
    db = Database(data_dir / "conversations.db").connect()
    ...
    db.close()
```

### Add a new MCP tool
1. Add the `Tool(...)` definition in `list_tools()` in `server.py`
2. Add a `case "tool_name":` in `call_tool()`
3. Write the handler as a module-level async function: `async def tool_name(db, vectors, args)`
4. Add an integration test in `tests/integration/test_mcp_tools.py` using the `full_stores` fixture

### Add a new DB table
1. Add the `CREATE TABLE IF NOT EXISTS` statement in `Database._create_schema()`
2. Add model in `models.py`
3. Add upsert/list/find methods in `Database`
4. Add tests in `tests/test_database.py`
5. Add to `stats()` if it makes sense to count

### Add a new source adapter
1. Create `src/consciousness/parser/<format>.py` implementing `SourceAdapter`
2. Register in `parser/__init__.py`: append to `_ADAPTERS`
3. Add unit tests in `tests/test_parser.py`
4. Update `ROADMAP.md` to mark the source as Done

---

## Environment variables

| Variable | Effect |
|---|---|
| `CONSCIOUSNESS_DATA_DIR` | Override default `~/.consciousness` data directory |
| `CONSCIOUSNESS_FAKE_ENCODER` | Use deterministic bag-of-words encoder (no network) |
| `ANTHROPIC_API_KEY` | Enables `synthesize_memory` to call Claude for better output |

---

## Known gotchas

- **Datetime timezone:** `updated_at` on `Conversation` is timezone-aware. Any comparison must also be TZ-aware. Never use `datetime.utcnow()`.
- **FakeEncoder in CLI tests:** `CliRunner` must be constructed with `env={"CONSCIOUSNESS_FAKE_ENCODER": "1"}`, not just passed as an argument. A bare `CliRunner()` will attempt a HuggingFace download.
- **db.commit() is manual:** The database uses explicit transaction control. Forgetting `commit()` after writes is a common bug in tests that check DB state across fixture teardown.
- **ChromaDB upsert is idempotent:** re-ingesting the same conversation produces the same chunk IDs and is a no-op from ChromaDB's perspective. This is intentional.
- **avoid pattern regex:** The avoid/skip decision pattern requires `because` to appear within 26 characters of the avoided technology name. Sentences like "Avoid X for Y because Z" will not match; "Avoid X because Z" will.
