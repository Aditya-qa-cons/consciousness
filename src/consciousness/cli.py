"""CLI entry point: consciousness ingest / serve / stats / export / import-bundle / rebuild-index / exclude."""

import asyncio
import io
import json
import time
import zipfile
from datetime import datetime, timezone
from pathlib import Path

import click
from rich.console import Console
from rich.progress import BarColumn, Progress, SpinnerColumn, TaskProgressColumn, TextColumn
from rich.table import Table

from consciousness.models import ExcludeRule
from consciousness.parser import parse_export
from consciousness.store.db import Database
from consciousness.store.vectors import VectorStore

console = Console()

DEFAULT_DATA_DIR = Path.home() / ".consciousness"


@click.group()
@click.option("--data-dir", default=str(DEFAULT_DATA_DIR), envvar="CONSCIOUSNESS_DATA_DIR", show_default=True)
@click.pass_context
def cli(ctx, data_dir):
    ctx.ensure_object(dict)
    ctx.obj["data_dir"] = Path(data_dir)
    ctx.obj["data_dir"].mkdir(parents=True, exist_ok=True)


# ── ingest ────────────────────────────────────────────────────────────────────


@cli.command()
@click.argument("export_path", type=click.Path(exists=True, path_type=Path))
@click.option("--skip-extraction/--no-skip-extraction", default=False, help="Skip knowledge extraction pass")
@click.option(
    "--force/--no-force", default=False,
    help="Re-process all conversations even if unchanged (default: skip unchanged)",
)
@click.option(
    "--llm-extract/--no-llm-extract", default=False,
    help="Use Claude Haiku for knowledge extraction (requires ANTHROPIC_API_KEY; slower but higher recall)",
)
@click.option(
    "--summarize/--no-summarize", default=False,
    help="Generate 2-3 sentence summaries per conversation (uses Haiku if ANTHROPIC_API_KEY set, else text extraction)",
)
@click.option(
    "--build-graph/--no-build-graph", default=False,
    help="Rebuild the knowledge graph after ingest (co-occurrence, supersession, relates-to edges)",
)
@click.option(
    "--account-id", default=None,
    help="Tag ingested conversations with this account ID (useful when merging exports from multiple accounts)",
)
@click.option(
    "--watch/--no-watch", default=False,
    help="Watch for new exports and re-ingest automatically (Ctrl+C to stop)",
)
@click.option(
    "--interval", default=300, show_default=True, type=int,
    help="Seconds between scans in watch mode",
)
@click.pass_context
def ingest(  # noqa: PLR0913
    ctx, export_path: Path, skip_extraction: bool, force: bool,
    llm_extract: bool, summarize: bool, build_graph: bool, account_id: str | None,
    watch: bool, interval: int,
):
    """Parse an export file or directory and index it into the local store.

    Supports Claude.ai exports (ZIP or JSON) and ChatGPT exports (ZIP).
    Pass a directory to ingest all ZIP files inside it.
    By default, conversations that haven't changed since the last ingest are
    skipped. Use --force to re-index everything unconditionally.

    Use --watch to run continuously, re-scanning every --interval seconds.
    In watch mode a directory path will only process ZIPs added or modified
    since the previous scan.

    Use --llm-extract to run Claude Haiku over each conversation for higher-
    quality knowledge extraction (requires ANTHROPIC_API_KEY).
    Use --summarize to store a 2-3 sentence summary for each conversation.
    Use --build-graph to rebuild the knowledge graph after ingest.

    EXPORT_PATH: path to the exported .zip / conversations.json, or a directory.
    """
    from consciousness.extractors.knowledge import (
        apply_temporal_tracking,
        extract_decisions,
        extract_preferences,
        extract_tech_choices,
    )
    from consciousness.extractors.llm import LLMExtractor
    from consciousness.extractors.sensitive import redact
    from consciousness.memory.summarizer import ConversationSummarizer
    from consciousness.models import ConversationSummary

    data_dir: Path = ctx.obj["data_dir"]

    llm_extractor = None
    if llm_extract and not skip_extraction:
        llm_extractor = LLMExtractor()
        if not llm_extractor.is_available():
            console.print(
                "[yellow]Warning:[/yellow] --llm-extract requires ANTHROPIC_API_KEY; falling back to regex extraction"
            )
            llm_extractor = None

    summarizer = None
    if summarize:
        summarizer = ConversationSummarizer()
        if not summarizer.is_available():
            console.print("[dim]Note:[/dim] No ANTHROPIC_API_KEY — summaries will use text extraction fallback")

    scan = 0
    while True:
        scan += 1
        if watch and scan > 1:
            console.print(f"\n[bold]Watch scan #{scan}[/bold]")

        db = Database(data_dir / "conversations.db").connect()
        vectors = VectorStore(data_dir / "vectors").connect()

        # Collect files to process. In directory mode, only consider ZIPs whose
        # mtime is newer than the last successful ingest (stored in config).
        if export_path.is_dir():
            since_str = db.get_config("last_ingested_at")
            since_ts: float | None = datetime.fromisoformat(since_str).timestamp() if since_str else None
            zip_files = sorted(
                f for f in export_path.glob("*.zip")
                if since_ts is None or f.stat().st_mtime > since_ts
            )
            if not zip_files:
                db.close()
                if not watch:
                    console.print("[red]No ZIP files found in directory.[/red]")
                    raise SystemExit(1)
                console.print(f"[dim]No new ZIPs in {export_path} — next scan in {interval}s…[/dim]")
                try:
                    time.sleep(interval)
                except KeyboardInterrupt:
                    console.print("\n[yellow]Watch stopped.[/yellow]")
                    return
                continue

            console.print(f"[bold]Parsing {len(zip_files)} export(s) from:[/bold] {export_path}")
            conversations: list = []
            projects: list = []
            for zf in zip_files:
                c, p = parse_export(zf, account_id=account_id)
                conversations.extend(c)
                projects.extend(p)
        else:
            console.print(f"[bold]Parsing export:[/bold] {export_path}")
            conversations, projects = parse_export(export_path, account_id=account_id)

        n_c, n_p = len(conversations), len(projects)
        console.print(f"  Found [green]{n_c}[/green] conversations across [green]{n_p}[/green] projects")

        for project in projects:
            db.upsert_project(project)

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            console=console,
        ) as progress:
            task_db = progress.add_task("Writing to database…", total=len(conversations))
            task_vec = progress.add_task("Building vector index…", total=len(conversations))
            task_ext = progress.add_task("Extracting knowledge…", total=len(conversations))
            task_sum = progress.add_task("Generating summaries…", total=len(conversations)) if summarize else None

            redaction_count = 0
            excluded_count = 0
            new_count = 0
            updated_count = 0
            skipped_count = 0
            dedup_count = 0

            for conv in conversations:
                if db.is_excluded(conv):
                    excluded_count += 1
                    progress.advance(task_db)
                    progress.advance(task_vec)
                    progress.advance(task_ext)
                    if task_sum is not None:
                        progress.advance(task_sum)
                    continue

                # Content-hash dedup: skip if an identical conversation already exists under a different ID
                if conv.content_hash:
                    existing_id = db.find_by_content_hash(conv.content_hash)
                    if existing_id and existing_id != conv.id:
                        dedup_count += 1
                        progress.advance(task_db)
                        progress.advance(task_vec)
                        progress.advance(task_ext)
                        if task_sum is not None:
                            progress.advance(task_sum)
                        continue

                stored_updated_at = db.get_conversation_updated_at(conv.id)
                is_new = stored_updated_at is None

                if not force and not is_new:
                    # Compare timestamps; skip if the stored version is current or newer.
                    # Both values are timezone-aware UTC datetimes.
                    conv_ts = conv.updated_at
                    if conv_ts is not None and stored_updated_at >= conv_ts:
                        # Still generate a summary if none exists yet.
                        if summarizer is not None and db.get_summary(conv.id) is None:
                            db.upsert_summary(ConversationSummary(
                                conversation_id=conv.id,
                                summary=summarizer.summarize(conv),
                                model=summarizer.model_used(),
                            ))
                        skipped_count += 1
                        progress.advance(task_db)
                        progress.advance(task_vec)
                        progress.advance(task_ext)
                        if task_sum is not None:
                            progress.advance(task_sum)
                        continue

                if is_new:
                    new_count += 1
                else:
                    updated_count += 1
                    db.delete_knowledge_for_conversation(conv.id)

                for msg in conv.messages:
                    clean, findings = redact(msg.content)
                    if findings:
                        msg.content = clean
                        redaction_count += len(findings)

                db.upsert_conversation(conv)
                progress.advance(task_db)

                vectors.index_conversation(conv)
                progress.advance(task_vec)

                if not skip_extraction:
                    existing = db.find_active_decisions(conv.title[:30])

                    llm_decisions, llm_prefs, llm_tcs = [], [], []
                    if llm_extractor is not None:
                        llm_decisions, llm_prefs, llm_tcs = llm_extractor.extract(conv)

                    new_decisions = llm_decisions or extract_decisions(conv)
                    new_prefs = llm_prefs or extract_preferences(conv)
                    new_tcs = llm_tcs or extract_tech_choices(conv)

                    for d in new_decisions:
                        supersessions = apply_temporal_tracking([d], existing)
                        for old_id, new_id in supersessions:
                            db.supersede_decision(old_id, new_id)
                        db.upsert_decision(d)
                    for pref in new_prefs:
                        db.upsert_preference(pref)
                    for tc in new_tcs:
                        db.upsert_tech_choice(tc)
                progress.advance(task_ext)

                if summarizer is not None:
                    db.upsert_summary(ConversationSummary(
                        conversation_id=conv.id,
                        summary=summarizer.summarize(conv),
                        model=summarizer.model_used(),
                    ))
                if task_sum is not None:
                    progress.advance(task_sum)

        if build_graph:
            from consciousness.memory.knowledge_graph import KnowledgeGraphBuilder
            with console.status("Building knowledge graph…"):
                kg_nodes, kg_edges = KnowledgeGraphBuilder().rebuild(db)
                db.commit()
            console.print(f"  Knowledge graph: {kg_nodes} nodes, {kg_edges} edges")

        s = db.stats()
        parts = []
        if new_count:
            parts.append(f"[green]{new_count} new[/green]")
        if updated_count:
            parts.append(f"[cyan]{updated_count} updated[/cyan]")
        if skipped_count:
            parts.append(f"[dim]{skipped_count} unchanged[/dim]")
        delta_summary = ", ".join(parts) if parts else f"{s['conversations']} total"
        console.print(
            f"\n[bold green]Done.[/bold green] {delta_summary} — "
            f"{s['conversations']} conversations, {s['messages']} messages, "
            f"{vectors.count()} vector chunks, "
            f"{s['decisions']} decisions, {s['tech_choices']} tech choices"
        )
        if dedup_count:
            console.print(f"  [dim]Deduped:[/dim] {dedup_count} conversations already indexed under another account")
        if excluded_count:
            console.print(f"  [yellow]Excluded:[/yellow] {excluded_count} conversations matched exclude rules")
        if redaction_count:
            console.print(f"  [yellow]Redacted:[/yellow] {redaction_count} sensitive values")
        if s.get("accounts", 0) > 1:
            accounts = db.list_accounts()
            console.print(f"  [bold]Accounts:[/bold] {', '.join(accounts)}")

        # Persist the timestamp of this successful ingest so directory watch mode
        # can filter ZIPs by mtime on the next scan.
        db.set_config("last_ingested_at", datetime.now(timezone.utc).isoformat())
        db.commit()
        console.print(f"Data directory: [dim]{data_dir}[/dim]")
        db.close()

        if not watch:
            break

        console.print(
            f"[dim]Watching [bold]{export_path}[/bold] — next scan in {interval}s (Ctrl+C to stop)[/dim]"
        )
        try:
            time.sleep(interval)
        except KeyboardInterrupt:
            console.print("\n[yellow]Watch stopped.[/yellow]")
            break


# ── serve ─────────────────────────────────────────────────────────────────────


@cli.command()
@click.pass_context
def serve(ctx):
    """Start the MCP server (stdio transport). Add to your Claude MCP config."""
    data_dir: Path = ctx.obj["data_dir"]
    if not (data_dir / "conversations.db").exists():
        console.print("[red]No data found.[/red] Run `consciousness ingest <export.zip>` first.")
        raise SystemExit(1)
    console.print(f"[bold]Starting consciousness MCP server[/bold] (data: {data_dir})")
    from consciousness.mcp_server.server import run
    asyncio.run(run(data_dir))


# ── stats ─────────────────────────────────────────────────────────────────────


@cli.command()
@click.pass_context
def stats(ctx):
    """Show statistics about the indexed data."""
    data_dir: Path = ctx.obj["data_dir"]
    db_path = data_dir / "conversations.db"
    if not db_path.exists():
        console.print("[red]No data found.[/red] Run `consciousness ingest <export.zip>` first.")
        raise SystemExit(1)

    db = Database(db_path).connect()
    s = db.stats()
    projects = db.list_projects()

    table = Table(title="Consciousness Store", show_header=True)
    table.add_column("Metric", style="bold")
    table.add_column("Value", justify="right")
    table.add_row("Conversations", str(s["conversations"]))
    table.add_row("Messages", str(s["messages"]))
    table.add_row("Projects", str(s["projects"]))
    table.add_row("Extracted decisions", str(s["decisions"]))
    table.add_row("Tech choices", str(s["tech_choices"]))
    console.print(table)

    if projects:
        ptable = Table(title="Projects", show_header=True)
        ptable.add_column("Name")
        ptable.add_column("Conversations", justify="right")
        for p in projects:
            ptable.add_row(p.name, str(p.conversation_count))
        console.print(ptable)

    db.close()


# ── export / import helpers ───────────────────────────────────────────────────


def _encrypt_bundle(data: bytes) -> bytes:
    try:
        import base64
        import os

        from cryptography.fernet import Fernet
        from cryptography.hazmat.primitives import hashes
        from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
    except ImportError:
        console.print("[red]cryptography package required.[/red] Install with: pip install cryptography")
        raise SystemExit(1)

    passphrase = click.prompt("Passphrase", hide_input=True, confirmation_prompt=True)
    salt = os.urandom(16)
    kdf = PBKDF2HMAC(algorithm=hashes.SHA256(), length=32, salt=salt, iterations=480_000)
    key = base64.urlsafe_b64encode(kdf.derive(passphrase.encode()))
    return b"CONSCIOUSNESS_ENC_V1\n" + salt + Fernet(key).encrypt(data)


def _decrypt_bundle(data: bytes) -> bytes:
    try:
        import base64

        from cryptography.fernet import Fernet
        from cryptography.hazmat.primitives import hashes
        from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
    except ImportError:
        console.print("[red]cryptography package required.[/red] Install with: pip install cryptography")
        raise SystemExit(1)

    header = b"CONSCIOUSNESS_ENC_V1\n"
    if not data.startswith(header):
        raise ValueError("File is not encrypted in the expected format.")
    payload = data[len(header):]
    salt, token = payload[:16], payload[16:]
    passphrase = click.prompt("Passphrase", hide_input=True)
    kdf = PBKDF2HMAC(algorithm=hashes.SHA256(), length=32, salt=salt, iterations=480_000)
    key = base64.urlsafe_b64encode(kdf.derive(passphrase.encode()))
    return Fernet(key).decrypt(token)


# ── export ────────────────────────────────────────────────────────────────────


@cli.command()
@click.argument("output", type=click.Path(path_type=Path))
@click.option("--encrypt/--no-encrypt", default=False, help="AES-encrypt the bundle (prompts for passphrase)")
@click.pass_context
def export(ctx, output: Path, encrypt: bool):
    """Export the local store to a portable .consciousness bundle.

    The bundle is a ZIP containing conversations.db and can be restored with
    import-bundle on any machine. Use --encrypt to protect with a passphrase.
    Sync the .consciousness file anywhere you like — iCloud, Dropbox, USB.
    """
    data_dir: Path = ctx.obj["data_dir"]
    db_path = data_dir / "conversations.db"
    if not db_path.exists():
        console.print("[red]No data found.[/red] Run `consciousness ingest <export.zip>` first.")
        raise SystemExit(1)

    if not output.suffix:
        output = output.with_suffix(".consciousness")

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.write(db_path, arcname="conversations.db")
        db = Database(db_path).connect()
        s = db.stats()
        db.close()
        zf.writestr("metadata.json", json.dumps({"version": 1, "stats": s}))

    bundle_bytes = buf.getvalue()
    if encrypt:
        bundle_bytes = _encrypt_bundle(bundle_bytes)

    output.write_bytes(bundle_bytes)
    console.print(
        f"[bold green]Exported[/bold green] {output} "
        f"({len(bundle_bytes)/1024:.1f} KB{'  [encrypted]' if encrypt else ''})"
    )


# ── import-bundle ─────────────────────────────────────────────────────────────


@cli.command("import-bundle")
@click.argument("bundle", type=click.Path(exists=True, path_type=Path))
@click.option("--rebuild/--no-rebuild", default=True, help="Rebuild vector index after restore (default: yes)")
@click.pass_context
def import_bundle(ctx, bundle: Path, rebuild: bool):
    """Restore from a .consciousness bundle and rebuild the vector index.

    On a new machine: consciousness import-bundle my-history.consciousness
    This restores the database and re-embeds everything locally.
    """
    data_dir: Path = ctx.obj["data_dir"]
    bundle_bytes = bundle.read_bytes()

    if bundle_bytes.startswith(b"CONSCIOUSNESS_ENC_V1\n"):
        bundle_bytes = _decrypt_bundle(bundle_bytes)

    with zipfile.ZipFile(io.BytesIO(bundle_bytes)) as zf:
        if "conversations.db" not in zf.namelist():
            console.print("[red]Invalid bundle:[/red] conversations.db not found.")
            raise SystemExit(1)
        zf.extract("conversations.db", path=data_dir)

    console.print(f"[bold green]Restored[/bold green] conversations.db → {data_dir}")

    if rebuild:
        ctx.invoke(rebuild_index)


# ── rebuild-index ─────────────────────────────────────────────────────────────


@cli.command("rebuild-index")
@click.pass_context
def rebuild_index(ctx):
    """Regenerate the vector index from SQLite.

    Use after import-bundle on a new machine, or after manually copying conversations.db.
    ChromaDB is derived data — this reconstructs it from the SQLite source of truth.
    """
    data_dir: Path = ctx.obj["data_dir"]
    db_path = data_dir / "conversations.db"
    if not db_path.exists():
        console.print("[red]No database found.[/red] Run `consciousness import-bundle` first.")
        raise SystemExit(1)

    db = Database(db_path).connect()
    vectors = VectorStore(data_dir / "vectors").connect()

    console.print("Clearing existing vector index…")
    vectors.clear()

    total = db.stats()["conversations"]
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        console=console,
    ) as progress:
        task = progress.add_task(f"Re-indexing {total} conversations…", total=total)
        offset = 0
        while True:
            stubs = db.list_conversations(limit=100, offset=offset)
            if not stubs:
                break
            for stub in stubs:
                full = db.get_conversation(stub.id)
                if full:
                    vectors.index_conversation(full)
                progress.advance(task)
            offset += 100

    console.print("Rebuilding full-text search index…")
    db.rebuild_fts()
    db.commit()
    db.close()
    console.print(f"[bold green]Done.[/bold green] {vectors.count()} chunks indexed.")


# ── rebuild-graph ─────────────────────────────────────────────────────────────


@cli.command("rebuild-graph")
@click.pass_context
def rebuild_graph(ctx):
    """Rebuild the knowledge graph from extracted decisions and tech choices.

    Run after ingest to update co-occurrence edges between technologies,
    superseded-decision chains, and relates-to links between topics and technologies.
    """
    from consciousness.memory.knowledge_graph import KnowledgeGraphBuilder

    data_dir: Path = ctx.obj["data_dir"]
    db_path = data_dir / "conversations.db"
    if not db_path.exists():
        console.print("[red]No database found.[/red] Run `consciousness ingest` first.")
        raise SystemExit(1)

    db = Database(db_path).connect()
    with console.status("Building knowledge graph…"):
        nodes, edges = KnowledgeGraphBuilder().rebuild(db)
    db.commit()
    db.close()
    console.print(f"[bold green]Done.[/bold green] {nodes} nodes, {edges} edges")


# ── maintenance ───────────────────────────────────────────────────────────────


@cli.command("maintenance")
@click.option(
    "--stale-days", default=90, show_default=True, type=int,
    help="Flag active decisions not updated in this many days",
)
@click.option(
    "--digest-days", default=7, show_default=True, type=int,
    help="Lookback window for the recent-activity digest",
)
@click.option("--json", "as_json", is_flag=True, default=False, help="Output as JSON")
@click.pass_context
def maintenance(ctx, stale_days: int, digest_days: int, as_json: bool):
    """Detect stale decisions, surface memory conflicts, and show a recent-changes digest.

    Stale decisions are active (non-superseded) decisions that have not been
    updated in more than --stale-days days. Memory conflicts are pairs of active
    decisions that share the same topic — they may represent unresolved tension
    that should be reviewed.

    Use --json to pipe results to other tools or save a report file.
    """
    data_dir: Path = ctx.obj["data_dir"]
    db_path = data_dir / "conversations.db"
    if not db_path.exists():
        console.print("[red]No data found.[/red] Run `consciousness ingest <export.zip>` first.")
        raise SystemExit(1)

    db = Database(db_path).connect()

    stale = db.list_stale_decisions(older_than_days=stale_days)
    conflicts = db.find_conflicting_decisions()
    digest = db.recent_changes(days=digest_days)
    db.close()

    if as_json:
        import json as _json

        out = {
            "stale_decisions": [
                {
                    "id": d.id, "topic": d.topic, "conclusion": d.conclusion,
                    "confidence": d.confidence, "extracted_at": d.extracted_at.isoformat() if d.extracted_at else None,
                    "conversation_id": d.conversation_id,
                }
                for d in stale
            ],
            "conflicting_pairs": [
                {
                    "decision_a": {"id": a.id, "topic": a.topic, "conclusion": a.conclusion},
                    "decision_b": {"id": b.id, "topic": b.topic, "conclusion": b.conclusion},
                }
                for a, b in conflicts
            ],
            "recent_changes": {
                "days": digest_days,
                "conversations": digest["conversations"],
                "decisions": [
                    {"id": d.id, "topic": d.topic, "conclusion": d.conclusion}
                    for d in digest["decisions"]
                ],
                "tech_choices": [
                    {"id": t.id, "technology": t.technology, "verdict": t.verdict}
                    for t in digest["tech_choices"]
                ],
            },
        }
        console.print(_json.dumps(out, indent=2))
        return

    # ── recent activity digest ─────────────────────────────────────────────────
    console.print(f"\n[bold]Recent Activity[/bold] (last {digest_days} days)")
    if not digest["conversations"] and not digest["decisions"] and not digest["tech_choices"]:
        console.print("  [dim]No activity in this period.[/dim]")
    else:
        if digest["conversations"]:
            console.print(f"  [green]{len(digest['conversations'])}[/green] conversation(s) updated")
            for c in digest["conversations"][:5]:
                console.print(f"    · {c['title']}")
            if len(digest["conversations"]) > 5:
                console.print(f"    … and {len(digest['conversations']) - 5} more")
        if digest["decisions"]:
            console.print(f"  [green]{len(digest['decisions'])}[/green] new decision(s) extracted")
            for d in digest["decisions"][:5]:
                console.print(f"    · [bold]{d.topic}[/bold]: {d.conclusion[:80]}")
            if len(digest["decisions"]) > 5:
                console.print(f"    … and {len(digest['decisions']) - 5} more")
        if digest["tech_choices"]:
            console.print(f"  [green]{len(digest['tech_choices'])}[/green] new tech choice(s) recorded")
            for t in digest["tech_choices"][:5]:
                console.print(f"    · [bold]{t.technology}[/bold] → {t.verdict}")

    # ── stale decisions ────────────────────────────────────────────────────────
    console.print(f"\n[bold]Stale Decisions[/bold] (not updated in >{stale_days} days)")
    if not stale:
        console.print("  [green]None.[/green] All active decisions are recent.")
    else:
        console.print(f"  [yellow]{len(stale)} decision(s) may be outdated:[/yellow]")
        table = Table(show_header=True, header_style="bold", box=None, padding=(0, 1))
        table.add_column("Topic")
        table.add_column("Conclusion", max_width=60)
        table.add_column("Age", justify="right")
        for d in stale:
            if d.extracted_at:
                age_days = (datetime.now(timezone.utc) - d.extracted_at).days
                age = f"{age_days}d"
            else:
                age = "?"
            table.add_row(d.topic, d.conclusion[:60], age)
        console.print(table)

    # ── memory conflicts ───────────────────────────────────────────────────────
    console.print("\n[bold]Potential Memory Conflicts[/bold]")
    if not conflicts:
        console.print("  [green]None.[/green] No duplicate-topic active decisions found.")
    else:
        console.print(
            f"  [yellow]{len(conflicts)} topic(s) with multiple active decisions[/yellow] "
            "(may or may not be genuine conflicts — review manually):"
        )
        for a, b in conflicts[:10]:
            console.print(f"\n  Topic: [bold]{a.topic}[/bold]")
            console.print(f"    A: {a.conclusion[:80]}")
            console.print(f"    B: {b.conclusion[:80]}")
        if len(conflicts) > 10:
            console.print(f"\n  … and {len(conflicts) - 10} more pairs")

    console.print()


# ── mcp-config ────────────────────────────────────────────────────────────────


@cli.command("mcp-config")
@click.pass_context
def mcp_config(ctx):
    """Print the MCP server config block to add to claude_desktop_config.json."""
    import sys
    data_dir: Path = ctx.obj["data_dir"]
    config = {
        "consciousness": {
            "command": sys.executable,
            "args": ["-m", "consciousness", "serve", "--data-dir", str(data_dir)],
            "env": {},
        }
    }
    console.print("\nAdd this to your [bold]claude_desktop_config.json[/bold] under [bold]mcpServers[/bold]:\n")
    console.print(json.dumps(config, indent=2))
    console.print("\nConfig file locations:")
    console.print("  macOS:   ~/Library/Application Support/Claude/claude_desktop_config.json")
    console.print("  Windows: %APPDATA%\\Claude\\claude_desktop_config.json")
    console.print("  Web:     Claude.ai → Settings → Claude Code → MCP servers\n")


# ── api ──────────────────────────────────────────────────────────────────────


@cli.command("api")
@click.option("--port", default=8765, show_default=True, help="Port to listen on")
@click.option("--host", default="127.0.0.1", show_default=True, help="Bind address")
@click.option("--no-open", is_flag=True, default=False, help="Do not open the browser automatically")
@click.pass_context
def api_server(ctx, port: int, host: str, no_open: bool):
    """Start the REST API server for cross-assistant portability.

    Exposes all consciousness tools as JSON HTTP endpoints.
    Any OpenAI-compatible assistant can import the tool definitions from:
      GET http://localhost:8765/api/v1/openai/tools

    Interactive API docs available at:
      http://localhost:8765/docs
    """
    try:
        import uvicorn

        from consciousness.api.app import create_api_app
    except ImportError:
        console.print(
            "[red]API server requires extra dependencies.[/red] "
            "Install with: pip install 'consciousness[web]'"
        )
        raise SystemExit(1)

    data_dir: Path = ctx.obj["data_dir"]
    if not (data_dir / "conversations.db").exists():
        console.print("[red]No data found.[/red] Run `consciousness ingest <export.zip>` first.")
        raise SystemExit(1)

    app = create_api_app(data_dir)
    url = f"http://{host}:{port}"
    console.print(f"[bold green]Consciousness API[/bold green] → {url}/api/v1/")
    console.print(f"OpenAI tools:  {url}/api/v1/openai/tools")
    console.print(f"Interactive docs: {url}/docs")
    console.print("Press Ctrl+C to stop.\n")

    if not no_open:
        import webbrowser
        webbrowser.open(f"{url}/docs")

    uvicorn.run(app, host=host, port=port, log_level="warning")


# ── ui ───────────────────────────────────────────────────────────────────────


@cli.command("ui")
@click.option("--port", default=8080, show_default=True, help="Port to listen on")
@click.option("--host", default="127.0.0.1", show_default=True, help="Bind address")
@click.option("--no-open", is_flag=True, default=False, help="Do not open the browser automatically")
@click.pass_context
def ui(ctx, port: int, host: str, no_open: bool):
    """Start the local web UI (read-only browser interface).

    Requires the web extra: pip install 'consciousness[web]'
    """
    try:
        import uvicorn

        from consciousness.web.app import create_app
    except ImportError:
        console.print(
            "[red]Web UI requires extra dependencies.[/red] "
            "Install with: pip install 'consciousness[web]'"
        )
        raise SystemExit(1)

    data_dir: Path = ctx.obj["data_dir"]
    if not (data_dir / "conversations.db").exists():
        console.print("[red]No data found.[/red] Run `consciousness ingest <export.zip>` first.")
        raise SystemExit(1)

    app = create_app(data_dir)
    url = f"http://{host}:{port}"
    console.print(f"[bold green]Consciousness UI[/bold green] → {url}")
    console.print("Press Ctrl+C to stop.\n")

    if not no_open:
        import webbrowser
        webbrowser.open(url)

    uvicorn.run(app, host=host, port=port, log_level="warning")


# ── exclude ───────────────────────────────────────────────────────────────────


@cli.group()
def exclude():
    """Manage conversation exclusion rules (applied during ingest)."""


@exclude.command("add")
@click.option("--id", "conv_id", default=None, help="Exclude a specific conversation by ID")
@click.option("--project", "project_id", default=None, help="Exclude all conversations in a project by project ID")
@click.option("--title", "title_glob", default=None, help="Exclude by title glob pattern (e.g. '*private*')")
@click.option(
    "--shared/--no-shared", default=False,
    help="Mark as a shared (team-wide) rule — bundled with share-export so collaborators can import it",
)
@click.pass_context
def exclude_add(ctx, conv_id, project_id, title_glob, shared):
    data_dir: Path = ctx.obj["data_dir"]
    db = Database(data_dir / "conversations.db").connect()

    if conv_id:
        rule = ExcludeRule(pattern=conv_id, rule_type="conversation_id", shared=shared)
    elif project_id:
        rule = ExcludeRule(pattern=project_id, rule_type="project_id", shared=shared)
    elif title_glob:
        rule = ExcludeRule(pattern=title_glob.lower(), rule_type="title_glob", shared=shared)
    else:
        console.print("[red]Specify one of --id, --project, or --title[/red]")
        raise SystemExit(1)

    db.add_exclude_rule(rule)
    db.commit()
    db.close()
    shared_label = " [shared]" if shared else ""
    console.print(f"[green]Added exclusion rule:[/green] {rule.rule_type} = {rule.pattern}{shared_label}")


@exclude.command("list")
@click.pass_context
def exclude_list(ctx):
    """List all active exclusion rules."""
    data_dir: Path = ctx.obj["data_dir"]
    db = Database(data_dir / "conversations.db").connect()
    rules = db.list_exclude_rules()
    db.close()

    if not rules:
        console.print("No exclusion rules defined.")
        return

    table = Table(title="Exclusion Rules", show_header=True)
    table.add_column("Type")
    table.add_column("Pattern")
    table.add_column("Shared")
    table.add_column("Created")
    for r in rules:
        table.add_row(
            r.rule_type, r.pattern,
            "yes" if r.shared else "",
            r.created_at.strftime("%Y-%m-%d") if r.created_at else "?",
        )
    console.print(table)


@exclude.command("remove")
@click.argument("pattern")
@click.pass_context
def exclude_remove(ctx, pattern: str):
    """Remove an exclusion rule by its pattern value."""
    data_dir: Path = ctx.obj["data_dir"]
    db = Database(data_dir / "conversations.db").connect()
    db.remove_exclude_rule(pattern)
    db.commit()
    db.close()
    console.print(f"[green]Removed exclusion rule:[/green] {pattern}")


# ── share-export ──────────────────────────────────────────────────────────────


@cli.command("share-export")
@click.argument("output", type=click.Path(path_type=Path))
@click.option("--project", "project_id", default=None, help="Export all conversations in this project")
@click.option("--conversation", "conv_ids", multiple=True, help="Specific conversation ID(s) to export (repeatable)")
@click.option("--namespace", default=None, help="Your name or team label (embedded in the bundle)")
@click.option("--encrypt/--no-encrypt", default=False, help="AES-encrypt the bundle (prompts for passphrase)")
@click.pass_context
def share_export(ctx, output: Path, project_id: str | None, conv_ids: tuple, namespace: str | None, encrypt: bool):
    """Export selected conversations as a shareable bundle.

    The bundle contains only the chosen project or conversations, strips sensitive
    content, and bundles any shared exclude rules so collaborators can adopt them.

    Collaborators import with: consciousness share-import BUNDLE --namespace YOURNAME

    Must specify at least one of --project or --conversation.
    """
    from consciousness.extractors.sensitive import redact

    if not project_id and not conv_ids:
        console.print("[red]Specify at least one of --project or --conversation.[/red]")
        raise SystemExit(1)

    data_dir: Path = ctx.obj["data_dir"]
    db_path = data_dir / "conversations.db"
    if not db_path.exists():
        console.print("[red]No data found.[/red] Run `consciousness ingest <export.zip>` first.")
        raise SystemExit(1)

    db = Database(db_path).connect()

    # Collect conversation stubs to export.
    selected: list = []
    if project_id:
        selected.extend(db.list_conversations(project_id=project_id, limit=10_000))
    for cid in conv_ids:
        conv = db.get_conversation(cid)
        if conv is None:
            console.print(f"[yellow]Warning:[/yellow] conversation {cid!r} not found — skipping")
        else:
            selected.append(conv)

    # Deduplicate by id (project + explicit ids may overlap).
    seen: set = set()
    unique: list = []
    for c in selected:
        if c.id not in seen:
            seen.add(c.id)
            unique.append(c)

    if not unique:
        console.print("[red]No matching conversations found to export.[/red]")
        db.close()
        raise SystemExit(1)

    # Apply all exclude rules so private conversations are never exported.
    excluded_count = 0
    exportable = []
    for conv in unique:
        if db.is_excluded(conv):
            excluded_count += 1
        else:
            exportable.append(conv)

    shared_rules = db.list_shared_exclude_rules()

    # Build the project set covering the exportable conversations.
    project_ids_needed = {c.project_id for c in exportable if c.project_id}
    all_projects = {p.id: p for p in db.list_projects()}
    projects_out = [all_projects[pid] for pid in project_ids_needed if pid in all_projects]

    # Collect full conversation data (with messages and knowledge).
    bundle_convs = []
    for stub in exportable:
        full = db.get_conversation(stub.id)
        if full is None:
            continue
        # Re-apply redaction on message content for safety.
        messages_out = []
        for msg in full.messages:
            clean, _ = redact(msg.content)
            messages_out.append({
                "id": msg.id,
                "role": msg.role.value,
                "content": clean,
                "timestamp": msg.timestamp.isoformat() if msg.timestamp else None,
                "position": msg.position,
            })
        decisions_out = [
            {
                "id": d.id, "topic": d.topic, "conclusion": d.conclusion,
                "confidence": d.confidence, "extracted_at": d.extracted_at.isoformat() if d.extracted_at else None,
                "superseded_by": d.superseded_by,
            }
            for d in db.list_decisions_for_conversation(full.id)
        ]
        prefs_out = [
            {"id": p.id, "area": p.area, "preference": p.preference,
             "extracted_at": p.extracted_at.isoformat() if p.extracted_at else None}
            for p in db.list_preferences_for_conversation(full.id)
        ]
        tcs_out = [
            {"id": t.id, "technology": t.technology, "verdict": t.verdict,
             "rationale": t.rationale,
             "extracted_at": t.extracted_at.isoformat() if t.extracted_at else None}
            for t in db.list_tech_choices_for_conversation(full.id)
        ]
        bundle_convs.append({
            "id": full.id,
            "title": full.title,
            "project_id": full.project_id,
            "created_at": full.created_at.isoformat() if full.created_at else None,
            "updated_at": full.updated_at.isoformat() if full.updated_at else None,
            "messages": messages_out,
            "decisions": decisions_out,
            "preferences": prefs_out,
            "tech_choices": tcs_out,
        })

    db.close()

    payload = {
        "version": 1,
        "format": "consciousness-share",
        "namespace": namespace or "",
        "exported_at": datetime.now(timezone.utc).isoformat(),
        "shared_exclude_rules": [
            {"pattern": r.pattern, "rule_type": r.rule_type, "created_at": r.created_at.isoformat()}
            for r in shared_rules
        ],
        "projects": [
            {"id": p.id, "name": p.name, "created_at": p.created_at.isoformat() if p.created_at else None}
            for p in projects_out
        ],
        "conversations": bundle_convs,
    }

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("share.json", json.dumps(payload, indent=2))

    bundle_bytes = buf.getvalue()
    if encrypt:
        bundle_bytes = _encrypt_bundle(bundle_bytes)

    if not output.suffix:
        output = output.with_suffix(".consciousness")
    output.write_bytes(bundle_bytes)

    console.print(
        f"[bold green]Shared bundle exported:[/bold green] {output} "
        f"({len(bundle_bytes)/1024:.1f} KB{'  [encrypted]' if encrypt else ''})"
    )
    console.print(f"  {len(bundle_convs)} conversation(s), {len(projects_out)} project(s)")
    if excluded_count:
        console.print(f"  [yellow]Excluded by rules:[/yellow] {excluded_count} conversation(s) not exported")
    if shared_rules:
        console.print(f"  [dim]{len(shared_rules)} shared exclude rule(s) bundled[/dim]")
    console.print(
        f"\nCollaborators can import with:\n"
        f"  [bold]consciousness share-import {output} --namespace {namespace or 'YOURNAME'}[/bold]"
    )


# ── share-import ──────────────────────────────────────────────────────────────


@cli.command("share-import")
@click.argument("bundle", type=click.Path(exists=True, path_type=Path))
@click.option("--namespace", required=True, help="Label for the imported conversations (e.g. collaborator's name)")
@click.option("--rebuild/--no-rebuild", default=True, help="Rebuild vector index after import (default: yes)")
@click.option(
    "--import-excludes/--no-import-excludes", default=False,
    help="Also import the shared exclude rules bundled in this file",
)
@click.pass_context
def share_import(ctx, bundle: Path, namespace: str, rebuild: bool, import_excludes: bool):
    """Import a collaborator's shared bundle into a separate namespace.

    Conversations are tagged with NAMESPACE (stored as account_id) so they
    stay isolated from your own history. Search results show the source label.

    Re-importing the same bundle is idempotent — existing namespace conversations
    are replaced in-place.

    Use --import-excludes to also adopt the shared exclude rules from the bundle.
    """
    data_dir: Path = ctx.obj["data_dir"]

    bundle_bytes = bundle.read_bytes()
    if bundle_bytes.startswith(b"CONSCIOUSNESS_ENC_V1\n"):
        bundle_bytes = _decrypt_bundle(bundle_bytes)

    with zipfile.ZipFile(io.BytesIO(bundle_bytes)) as zf:
        names = zf.namelist()
        if "share.json" not in names:
            console.print("[red]Invalid share bundle:[/red] share.json not found. "
                          "Use import-bundle for full database bundles.")
            raise SystemExit(1)
        payload = json.loads(zf.read("share.json"))

    if payload.get("format") != "consciousness-share":
        console.print("[red]Unrecognised bundle format.[/red]")
        raise SystemExit(1)

    db = Database(data_dir / "conversations.db").connect()

    # Upsert projects — prefix IDs with namespace to avoid collisions.
    def ns(orig_id: str) -> str:
        return f"{namespace}/{orig_id}"

    for p in payload.get("projects", []):
        from consciousness.models import Project
        db.upsert_project(Project(
            id=ns(p["id"]),
            name=p["name"],
            created_at=datetime.fromisoformat(p["created_at"]) if p.get("created_at") else None,
            account_id=namespace,
        ))

    from consciousness.models import Conversation, Decision, Message, Preference, Role, TechChoice

    imported = 0
    for c in payload.get("conversations", []):
        conv_id = ns(c["id"])
        proj_id = ns(c["project_id"]) if c.get("project_id") else None
        messages = [
            Message(
                id=ns(m["id"]),
                conversation_id=conv_id,
                role=Role(m["role"]),
                content=m["content"],
                timestamp=datetime.fromisoformat(m["timestamp"]) if m.get("timestamp") else datetime.now(timezone.utc),
                position=m["position"],
            )
            for m in c.get("messages", [])
        ]
        conv = Conversation(
            id=conv_id,
            title=c["title"],
            project_id=proj_id,
            created_at=datetime.fromisoformat(c["created_at"]) if c.get("created_at") else datetime.now(timezone.utc),
            updated_at=datetime.fromisoformat(c["updated_at"]) if c.get("updated_at") else datetime.now(timezone.utc),
            messages=messages,
            account_id=namespace,
        )
        db.upsert_conversation(conv)

        db.delete_knowledge_for_conversation(conv_id)
        _now = datetime.now(timezone.utc)
        for d in c.get("decisions", []):
            db.upsert_decision(Decision(
                id=ns(d["id"]), topic=d["topic"], conclusion=d["conclusion"],
                confidence=d.get("confidence", 0.75), conversation_id=conv_id,
                extracted_at=datetime.fromisoformat(d["extracted_at"]) if d.get("extracted_at") else _now,
            ))
        for p2 in c.get("preferences", []):
            db.upsert_preference(Preference(
                id=ns(p2["id"]), area=p2["area"], preference=p2["preference"],
                conversation_id=conv_id,
                extracted_at=datetime.fromisoformat(p2["extracted_at"]) if p2.get("extracted_at") else _now,
            ))
        for t in c.get("tech_choices", []):
            db.upsert_tech_choice(TechChoice(
                id=ns(t["id"]), technology=t["technology"], verdict=t["verdict"],
                rationale=t.get("rationale"), conversation_id=conv_id,
                extracted_at=datetime.fromisoformat(t["extracted_at"]) if t.get("extracted_at") else _now,
            ))
        imported += 1

    if import_excludes:
        _now = datetime.now(timezone.utc)
        for rule_data in payload.get("shared_exclude_rules", []):
            db.add_exclude_rule(ExcludeRule(
                pattern=rule_data["pattern"],
                rule_type=rule_data["rule_type"],
                created_at=datetime.fromisoformat(rule_data["created_at"]) if rule_data.get("created_at") else _now,
                shared=True,
            ))

    db.commit()
    db.close()

    bundle_ns = payload.get("namespace", "")
    source_label = f" from '{bundle_ns}'" if bundle_ns else ""
    console.print(
        f"[bold green]Imported[/bold green]{source_label} → namespace [bold]{namespace}[/bold]: "
        f"{imported} conversation(s)"
    )
    if import_excludes and payload.get("shared_exclude_rules"):
        console.print(f"  [dim]Imported {len(payload['shared_exclude_rules'])} shared exclude rule(s)[/dim]")

    if rebuild:
        from consciousness.store.vectors import VectorStore
        vectors = VectorStore(data_dir / "vectors").connect()
        db2 = Database(data_dir / "conversations.db").connect()
        with console.status(f"Building vector index for {imported} conversations…"):
            for c in payload.get("conversations", []):
                full = db2.get_conversation(ns(c["id"]))
                if full:
                    vectors.index_conversation(full)
        console.print("  Full-text search index updated…")
        db2.rebuild_fts()
        db2.commit()
        db2.close()
        console.print(f"  [dim]{vectors.count()} total vector chunks[/dim]")
