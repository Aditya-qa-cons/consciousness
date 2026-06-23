# consciousness

**Your Claude conversation history as a portable, account-independent MCP server.**

Export your Claude.ai history once, index it locally, and query it from any Claude account via the Model Context Protocol — no API key required for search, no data ever leaves your machine.

```
┌──────────────────────────────────────────────────────────────────┐
│  Claude.ai  →  export .zip  →  consciousness ingest              │
│                                                                  │
│  consciousness serve  →  MCP tools in any Claude session         │
│                                                                  │
│  consciousness export  →  .consciousness bundle  →  any machine  │
└──────────────────────────────────────────────────────────────────┘
```

---

## What it does

| Capability | Detail |
|---|---|
| **Semantic search** | Find relevant Q+A exchanges across all your conversations by meaning, not just keywords |
| **Decision recall** | Pattern-extracted conclusions ("use Postgres", "avoid MongoDB") stored in a structured table, queryable without a vector lookup |
| **Preference memory** | Tracks recurring preferences expressed in your messages (languages, tools, styles) |
| **Tech choice history** | Every technology verdict reached across your projects |
| **Auto-context injection** | MCP Resource `memory://context` auto-feeds recent decisions and tech choices to Claude at session start — no tool call needed |
| **Portable export** | Single `.consciousness` ZIP bundle; optional AES-256 encryption; restore anywhere with `import-bundle` |
| **Sensitive-data redaction** | API keys, AWS credentials, GitHub PATs, Slack tokens, and passwords are stripped before indexing |
| **Exclude rules** | Mark conversations or entire projects as private; they are skipped at ingest time |

---

## Quick start

### 1. Install

```bash
git clone https://github.com/Aditya-qa-cons/consciousness
cd consciousness
pip install -e .
```

For encrypted export/import, also install the optional `cryptography` extra:

```bash
pip install -e ".[encrypt]"
```

### 2. Export your Claude history

Go to **claude.ai → Settings → Account → Export Data**. Download the `.zip` file.

### 3. Ingest

```bash
consciousness ingest ~/Downloads/claude-export.zip
```

Progress bars show database writes, vector indexing, and knowledge extraction.  
Data is stored in `~/.consciousness/` by default.

```
Parsing export: ~/Downloads/claude-export.zip
  Found 847 conversations across 12 projects
  Writing to database…   ████████████████ 100%
  Building vector index… ████████████████ 100%
  Extracting knowledge…  ████████████████ 100%

Done. 847 conversations, 14 231 messages, 28 904 vector chunks,
      312 decisions, 89 tech choices
```

### 4. Wire up MCP

```bash
consciousness mcp-config
```

Copy the printed JSON block into your Claude Desktop config:

- **macOS:** `~/Library/Application Support/Claude/claude_desktop_config.json`
- **Windows:** `%APPDATA%\Claude\claude_desktop_config.json`
- **Claude.ai (web):** Settings → Claude Code → MCP servers

### 5. Start the server

```bash
consciousness serve
```

Or let Claude Desktop manage it automatically via the MCP config (recommended).

---

## CLI reference

```
consciousness [--data-dir DIR] COMMAND [ARGS]
```

`--data-dir` defaults to `~/.consciousness`. Override with `CONSCIOUSNESS_DATA_DIR` env var.

| Command | Description |
|---|---|
| `ingest <file.zip>` | Parse a Claude.ai export and index everything |
| `ingest --skip-extraction` | Skip the knowledge extraction pass (faster) |
| `serve` | Start the MCP stdio server |
| `stats` | Show counts: conversations, messages, decisions, tech choices |
| `mcp-config` | Print the config block for `claude_desktop_config.json` |
| `export <output>` | Create a portable `.consciousness` bundle |
| `export --encrypt` | Same, but AES-256-encrypted (prompts for passphrase) |
| `import-bundle <file>` | Restore from a bundle, then rebuild the vector index |
| `import-bundle --no-rebuild` | Restore only the SQLite database, skip vectors |
| `rebuild-index` | Regenerate ChromaDB from SQLite (use after manual DB restore) |
| `exclude add --title "*private*"` | Exclude conversations by title glob |
| `exclude add --id <conv-id>` | Exclude a specific conversation by ID |
| `exclude add --project <proj-id>` | Exclude an entire project |
| `exclude list` | Show all active exclusion rules |
| `exclude remove <pattern>` | Delete an exclusion rule |

---

## MCP tools

When the server is running, these tools are available in any Claude session:

| Tool | Description |
|---|---|
| `search_history` | Semantic search over all conversations; optional `project` and `role` filters |
| `get_project_context` | All conversations in a named project; optional full message text |
| `recall_decision` | Looks up structured decision history first, then falls back to vector search |
| `get_recent_context` | Summaries of conversations from the last N days |
| `get_conversation` | Full text of a specific conversation by ID |
| `list_projects` | All projects with conversation counts |
| `synthesize_memory` | Generates a paste-ready memory import blob |

### MCP resource (auto-injected)

`memory://context` — a Markdown document containing your recent decisions, technology choices, and active conversations. Claude receives this automatically at session start without you having to call any tool.

---

## Portable export / import

Move your indexed history to another machine without re-downloading or re-ingesting:

```bash
# On machine A
consciousness export ~/Desktop/my-history.consciousness

# On machine B (just needs the package installed)
consciousness import-bundle ~/Desktop/my-history.consciousness
```

Encrypted:

```bash
consciousness export ~/Desktop/my-history.consciousness --encrypt
# → prompts for passphrase, stores AES-256 ciphertext

consciousness import-bundle ~/Desktop/my-history.consciousness
# → prompts for passphrase, decrypts, restores + rebuilds
```

The bundle is a ZIP of `conversations.db` (SQLite) plus `metadata.json`. The vector index (ChromaDB) is **not** included — it is derived data and is regenerated on import via `rebuild-index`. This keeps bundles small and means you always get a fresh, locally-embedded index tuned to your hardware.

---

## Exclusion rules

Keep private conversations out of the index:

```bash
# By title pattern (fnmatch glob, case-insensitive)
consciousness exclude add --title "*therapy*"
consciousness exclude add --title "*personal*"

# By specific conversation ID (copy from claude.ai URL)
consciousness exclude add --id "abc-123-def"

# By project (all conversations in a project)
consciousness exclude add --project "proj-uuid"
```

Rules are applied at ingest time. Re-ingest after adding a rule to remove previously-indexed content.

---

## Development

```bash
pip install -e ".[dev]"
pytest tests/                           # unit tests only (fast, no ML)
pytest tests/ -m integration            # integration tests (ChromaDB + FakeEncoder)
python -m ruff check src/ tests/        # lint
```

Integration tests use a deterministic bag-of-words `FakeEncoder` (no network, no model download). Set `CONSCIOUSNESS_FAKE_ENCODER=1` in any environment to use the same fake encoder in the real CLI — useful for CI or air-gapped machines.

---

## Project layout

```
src/consciousness/
├── models.py              # Pydantic domain models (Conversation, Decision, …)
├── cli.py                 # Click entry point for all CLI commands
├── parser/
│   ├── base.py            # SourceAdapter Protocol (pluggable parser interface)
│   ├── claude_export.py   # Parser for Claude.ai ZIP exports
│   └── __init__.py        # Auto-detect adapter registry
├── store/
│   ├── db.py              # SQLite layer — all structured data
│   └── vectors.py         # ChromaDB layer — exchange-level embeddings
├── extractors/
│   ├── knowledge.py       # Pattern-based decision / preference / tech-choice extraction
│   └── sensitive.py       # API key / secret detection and redaction
├── mcp_server/
│   └── server.py          # MCP tools + memory://context resource
└── memory/
    └── synthesizer.py     # MemoryBlob generation (calls Claude API if key present)
```
