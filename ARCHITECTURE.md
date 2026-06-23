# Architecture

This document explains the design of `consciousness` — the key decisions, the data flow, and the component responsibilities.

---

## Guiding principles

1. **Local-first, network-optional.** Embeddings run on-device via `sentence-transformers`. No API key is needed to search your own history.
2. **SQLite is the source of truth.** ChromaDB holds derived embeddings that can always be regenerated from SQLite via `rebuild-index`. This makes the system portable: you only need to carry the `.db` file.
3. **Extraction without inference.** Knowledge extraction (decisions, preferences, tech choices) uses deterministic regex patterns. No LLM call is needed at ingest time.
4. **Progressive enhancement.** The system works without a Claude API key. The `synthesize_memory` tool calls Claude only if `ANTHROPIC_API_KEY` is set; otherwise it falls back to a template.

---

## Layer diagram

```
                 ┌─────────────────────────────────────────────┐
                 │                CLI (click)                   │
                 │  ingest  serve  stats  export  import  etc.  │
                 └────────────────────┬────────────────────────┘
                                      │
           ┌──────────────────────────┼───────────────────────┐
           │                          │                       │
    ┌──────▼──────┐         ┌─────────▼────────┐   ┌─────────▼───────┐
    │   Parser     │         │   Extractors      │   │   MCP Server    │
    │  (adapters)  │         │  knowledge.py     │   │   server.py     │
    │  SourceAdapter│         │  sensitive.py     │   │  7 tools        │
    │  Protocol    │         └─────────┬─────────┘   │  1 resource     │
    └──────┬───────┘                   │             └─────────┬───────┘
           │                           │                       │
           └───────────────┬───────────┘           ┌──────────┘
                           │                       │
                  ┌────────▼────────┐   ┌──────────▼────────┐
                  │   SQLite (db.py) │   │ ChromaDB (vectors) │
                  │  conversations   │   │  exchange chunks   │
                  │  messages        │   │  embeddings        │
                  │  projects        │   │  cosine search     │
                  │  decisions       │   └───────────────────┘
                  │  preferences     │
                  │  tech_choices    │
                  │  exclude_rules   │
                  └─────────────────┘
```

---

## Data flow: ingest

```
claude-export.zip
        │
        ▼
ClaudeExportAdapter.parse()
        │
        ├──▶  list[Conversation]  ──▶  db.upsert_conversation()
        │                               sensitive.redact()       (pre-clean)
        │                               db.is_excluded()         (skip check)
        │
        ├──▶  list[Project]       ──▶  db.upsert_project()
        │
        ├──▶  VectorStore.index_conversation()
        │       └── _pair_exchanges()   (human+assistant pairs)
        │           _smart_chunk()      (paragraph → sentence boundaries)
        │           FakeEncoder / SentenceTransformer
        │           ChromaDB upsert
        │
        └──▶  extract_decisions()   ──▶  db.upsert_decision()
              extract_preferences()      db.upsert_preference()
              extract_tech_choices()     db.upsert_tech_choice()
              apply_temporal_tracking()  db.supersede_decision()
```

---

## Data flow: MCP query

```
Claude session
      │
      │  (auto-injected at start)
      ├──▶  memory://context  ──▶  db.list_decisions()
      │                            db.list_tech_choices()
      │                            db.list_conversations()  (last 7 days)
      │
      │  search_history(query, ...)
      ├──▶  VectorStore.search()
      │       model.encode([query])
      │       collection.query()
      │       threshold filter (cosine distance < 0.8)
      │       exchange deduplication
      │       → list[VectorHit]
      │
      │  recall_decision(topic)
      └──▶  db.find_active_decisions(topic)   (structured, fast)
            VectorStore.search(topic, role=assistant)  (fallback context)
```

---

## Component responsibilities

### `models.py`

All domain types as Pydantic v2 models. Three tiers:

- **Raw history:** `Conversation`, `Message`, `Project`, `Attachment`
- **Extracted knowledge:** `Decision`, `Preference`, `TechChoice`, `ExcludeRule`
- **Search/output:** `SearchResult`, `MemoryBlob`

No business logic here; models are data containers only.

---

### `parser/` — SourceAdapter Protocol

```python
class SourceAdapter(Protocol):
    source_name: str
    def can_handle(self, path: Path) -> bool: ...
    def parse(self, path: Path) -> tuple[list[Conversation], list[Project]]: ...
```

The auto-detect registry in `__init__.py` iterates registered adapters and calls `can_handle()`. First match wins. Adding support for a new export format (Cursor AI, VS Code, ChatGPT) requires only a new class implementing this protocol — no changes to existing code.

`ClaudeExportAdapter` handles both ZIP archives (`conversations.json` inside) and bare JSON files. It normalises Claude's schema variants (flat array vs. `data.conversations` envelope) and handles missing/null fields.

---

### `store/db.py` — SQLite layer

WAL-mode SQLite via the stdlib `sqlite3` module. No ORM.

**Schema:**

```sql
-- core history
conversations   (id, title, project_id, created_at, updated_at)
messages        (id, conversation_id, role, content, timestamp, position)
projects        (id, name, created_at)

-- extracted knowledge
decisions       (id, topic, conclusion, confidence, conversation_id, extracted_at,
                 superseded_by REFERENCES decisions(id))
preferences     (id, area, preference, conversation_id, extracted_at)
tech_choices    (id, technology, verdict, rationale, conversation_id, extracted_at)

-- access control
exclude_rules   (pattern PRIMARY KEY, rule_type, created_at)
```

Key design choices:
- `superseded_by` is a self-referential FK on `decisions`; `find_active_decisions()` filters `WHERE superseded_by IS NULL`.
- `is_excluded()` runs all rule types in Python after a single `SELECT *` — simpler than SQL-side fnmatch for `title_glob`.
- All timestamps stored as ISO-8601 UTC strings.

---

### `store/vectors.py` — ChromaDB / embedding layer

**Exchange-level chunking** is the key design choice vs. raw character windows.

Each indexed unit is a Q+A *exchange* (one human message + one assistant reply), giving the embedding richer context. The human chunk is stored as `"Q: <human text>"` and the assistant chunk as `"[Q: <first 200 chars of human>]\nA: <assistant text>"`. Both carry `exchange_id` in metadata so the deduplication pass can collapse multiple chunks from the same exchange into a single search result.

```
Long message (>800 chars)
        │
        ▼
_smart_chunk()
    try paragraph splits (blank lines)
    then sentence splits (. ! ?)
    hard split as last resort
        │
        ▼
[chunk_0, chunk_1, ...]  each stored with same exchange_id
```

**Score threshold and deduplication:**

`search()` over-fetches `limit × 3` candidates from ChromaDB, then:
1. Drops any hit with cosine distance ≥ 0.8 (configurable `score_threshold`)
2. Keeps only the first occurrence per `exchange_id` (dedup)
3. Returns up to `limit` hits

This means a very long assistant response that was split into four chunks will appear once in results, not four times.

**Relevance labels** on `VectorHit`:

| Score range | Label |
|---|---|
| < 0.35 | `high` |
| 0.35–0.60 | `medium` |
| ≥ 0.60 | `low` |

**Fake encoder for CI:**

When `CONSCIOUSNESS_FAKE_ENCODER=1` is set, `VectorStore.model` returns a deterministic bag-of-words encoder (MD5-based index mapping, no download). Integration tests always set this env var. The same mechanism works in any environment — air-gapped machines, CI runners — by setting the env var before calling `consciousness ingest`.

---

### `extractors/knowledge.py` — pattern-based extraction

Five decision patterns (regex + confidence score), three preference patterns, and a 60+ term technology vocabulary with verdict-word proximity detection. All pure Python, no API calls.

**Temporal tracking:** `apply_temporal_tracking(new, existing)` checks for topic overlap (substring match, normalised). When a new decision's topic contains or is contained by an existing one, the existing decision is marked `superseded_by = new.id`. This keeps `find_active_decisions()` returning only the latest conclusion on each topic.

---

### `extractors/sensitive.py` — redaction

Nine compiled regex patterns covering:
`anthropic_key`, `openai_key`, `aws_access_key`, `aws_secret`, `github_pat` (two forms), `slack_token`, `generic_secret`, `bearer_token`

`redact(text)` returns `(clean_text, findings)`. Applied to every message at ingest time; `findings` is a list of `(name, matched_string)` tuples for audit logging. Original `content` is overwritten with the redacted version before DB or vector writes.

---

### `mcp_server/server.py` — MCP interface

Built on the official Python MCP SDK. One `Server` instance with lifespan context that holds the DB and VectorStore connections.

**Tool strategy:**

- `search_history` — pure vector search, optional project/role filter
- `recall_decision` — structured DB first (no embeddings, instant), vector search as supplementary context
- `get_project_context` — pure DB, no embeddings
- `get_recent_context`, `get_conversation`, `list_projects` — pure DB
- `synthesize_memory` — calls `MemorySynthesizer` which calls Claude API if key present

**Resource `memory://context`:**

This is the auto-inject mechanism. MCP hosts that support resources (Claude Desktop, Claude.ai) request this URI at session start. The response is a Markdown document with:
- Top 10 recent decisions (non-superseded)
- Tech choices (deduplicated by technology name)
- Conversations active in the last 7 days

This means Claude sees your context without you needing to call any tool.

---

## Portability model

```
Machine A
  ~/.consciousness/
    conversations.db     ← SQLite (source of truth)
    vectors/             ← ChromaDB (derived, not portable)

consciousness export backup.consciousness
  → ZIP of conversations.db + metadata.json
  → optional AES-256 (PBKDF2HMAC / Fernet, 480 000 iterations)

Machine B
consciousness import-bundle backup.consciousness
  → extracts conversations.db
  → runs rebuild-index (re-embeds locally using machine B's hardware)
```

The decision to exclude the vector index from the bundle is intentional:
- ChromaDB directories contain native binary format that may not be portable across OS/arch
- The index is fully reproducible from SQLite
- Re-embedding is fast for typical history sizes (< 2 min for 1 000 conversations on CPU)
- Bundles stay small (typically 1–15 MB vs. potentially 500+ MB for the full vector store)

---

## Database size estimates

| History size | SQLite | ChromaDB |
|---|---|---|
| 100 conversations, 1 000 messages | ~2 MB | ~20 MB |
| 500 conversations, 5 000 messages | ~8 MB | ~90 MB |
| 2 000 conversations, 20 000 messages | ~30 MB | ~350 MB |

SQLite growth is dominated by message content length. ChromaDB growth scales with the number of embedded chunks (each exchange creates 1–4 chunks depending on length). Typical exchange produces 2 chunks (1 human + 1 assistant).

---

## Testing strategy

```
tests/
├── test_models.py              # Pydantic model validation
├── test_parser.py              # Claude export parsing (unit, no DB)
├── test_parser_edge_cases.py   # Malformed / partial exports
├── test_database.py            # SQLite layer (tmp_path DB per test)
├── test_extractor.py           # Pattern extraction + sensitive redaction
├── test_memory_synthesizer.py  # MemorySynthesizer (mocked Claude API)
└── integration/
    ├── conftest.py             # FakeEncoder + seeded fixtures
    ├── test_vector_store.py    # VectorStore index + search
    ├── test_ingest_pipeline.py # parse → DB → vectors end-to-end
    ├── test_mcp_tools.py       # MCP tool handlers (no transport)
    └── test_export_import.py   # CLI export / import-bundle / rebuild-index
```

Unit tests run in milliseconds. Integration tests use FakeEncoder (no HuggingFace download) and are marked `@pytest.mark.integration`. The split allows fast feedback during development without network dependencies.
