"""CLI entry point: consciousness ingest / serve / stats / export / import-bundle / rebuild-index / exclude."""

import asyncio
import io
import json
import zipfile
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
@click.pass_context
def ingest(ctx, export_path: Path, skip_extraction: bool):
    """Parse a Claude.ai export file and index it into the local store.

    EXPORT_PATH: path to the exported .zip or conversations.json file.
    """
    from consciousness.extractors.knowledge import (
        apply_temporal_tracking,
        extract_decisions,
        extract_preferences,
        extract_tech_choices,
    )
    from consciousness.extractors.sensitive import redact

    data_dir: Path = ctx.obj["data_dir"]
    console.print(f"[bold]Parsing export:[/bold] {export_path}")
    conversations, projects = parse_export(export_path)
    n_c, n_p = len(conversations), len(projects)
    console.print(f"  Found [green]{n_c}[/green] conversations across [green]{n_p}[/green] projects")

    db = Database(data_dir / "conversations.db").connect()
    vectors = VectorStore(data_dir / "vectors").connect()

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

        redaction_count = 0
        excluded_count = 0

        for conv in conversations:
            if db.is_excluded(conv):
                excluded_count += 1
                progress.advance(task_db)
                progress.advance(task_vec)
                progress.advance(task_ext)
                continue

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
                new_decisions = extract_decisions(conv)
                for d in new_decisions:
                    supersessions = apply_temporal_tracking([d], existing)
                    for old_id, new_id in supersessions:
                        db.supersede_decision(old_id, new_id)
                    db.upsert_decision(d)
                for pref in extract_preferences(conv):
                    db.upsert_preference(pref)
                for tc in extract_tech_choices(conv):
                    db.upsert_tech_choice(tc)
            progress.advance(task_ext)

        db.commit()

    s = db.stats()
    console.print(
        f"\n[bold green]Done.[/bold green] "
        f"{s['conversations']} conversations, {s['messages']} messages, "
        f"{vectors.count()} vector chunks, "
        f"{s['decisions']} decisions, {s['tech_choices']} tech choices"
    )
    if excluded_count:
        console.print(f"  [yellow]Excluded:[/yellow] {excluded_count} conversations matched exclude rules")
    if redaction_count:
        console.print(f"  [yellow]Redacted:[/yellow] {redaction_count} sensitive values")
    console.print(f"Data directory: [dim]{data_dir}[/dim]")
    db.close()


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

    db.close()
    console.print(f"[bold green]Done.[/bold green] {vectors.count()} chunks indexed.")


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


# ── exclude ───────────────────────────────────────────────────────────────────


@cli.group()
def exclude():
    """Manage conversation exclusion rules (applied during ingest)."""


@exclude.command("add")
@click.option("--id", "conv_id", default=None, help="Exclude a specific conversation by ID")
@click.option("--project", "project_id", default=None, help="Exclude all conversations in a project by project ID")
@click.option("--title", "title_glob", default=None, help="Exclude by title glob pattern (e.g. '*private*')")
@click.pass_context
def exclude_add(ctx, conv_id, project_id, title_glob):
    """Add an exclusion rule — matching conversations are skipped during ingest."""
    data_dir: Path = ctx.obj["data_dir"]
    db = Database(data_dir / "conversations.db").connect()

    if conv_id:
        rule = ExcludeRule(pattern=conv_id, rule_type="conversation_id")
    elif project_id:
        rule = ExcludeRule(pattern=project_id, rule_type="project_id")
    elif title_glob:
        rule = ExcludeRule(pattern=title_glob.lower(), rule_type="title_glob")
    else:
        console.print("[red]Specify one of --id, --project, or --title[/red]")
        raise SystemExit(1)

    db.add_exclude_rule(rule)
    db.commit()
    db.close()
    console.print(f"[green]Added exclusion rule:[/green] {rule.rule_type} = {rule.pattern}")


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
    table.add_column("Created")
    for r in rules:
        table.add_row(r.rule_type, r.pattern, r.created_at.strftime("%Y-%m-%d") if r.created_at else "?")
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
