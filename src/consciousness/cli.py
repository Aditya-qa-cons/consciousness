"""CLI entry point: consciousness ingest / serve / stats."""

import asyncio
from pathlib import Path

import click
from rich.console import Console
from rich.progress import BarColumn, Progress, SpinnerColumn, TaskProgressColumn, TextColumn
from rich.table import Table

from consciousness.parser.claude_export import parse_export
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


@cli.command()
@click.argument("export_path", type=click.Path(exists=True, path_type=Path))
@click.pass_context
def ingest(ctx, export_path: Path):
    """Parse a Claude.ai export file and index it into the local store.

    EXPORT_PATH: path to the exported .zip or conversations.json file.
    """
    data_dir: Path = ctx.obj["data_dir"]

    console.print(f"[bold]Parsing export:[/bold] {export_path}")
    conversations, projects = parse_export(export_path)
    n_conv, n_proj = len(conversations), len(projects)
    console.print(f"  Found [green]{n_conv}[/green] conversations across [green]{n_proj}[/green] projects")

    db = Database(data_dir / "conversations.db").connect()
    vectors = VectorStore(data_dir / "vectors").connect()

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        console=console,
    ) as progress:
        task_db = progress.add_task("Writing to database...", total=len(conversations) + len(projects))

        for project in projects:
            db.upsert_project(project)
            progress.advance(task_db)

        all_messages = []
        for conv in conversations:
            db.upsert_conversation(conv)
            all_messages.extend(conv.messages)
            progress.advance(task_db)

        db.commit()

        task_vec = progress.add_task("Building embeddings...", total=len(all_messages))
        vectors.index_messages_batch(all_messages, progress=progress.tasks[task_vec.id] if False else None)
        # Note: pass a simple progress callback
        for msg in all_messages:
            vectors.index_message(msg)
            progress.advance(task_vec)

    stats = db.stats()
    console.print(
        f"\n[bold green]Done.[/bold green] Store: "
        f"{stats['conversations']} conversations, {stats['messages']} messages, "
        f"{vectors.count()} vector chunks"
    )
    console.print(f"Data directory: [dim]{data_dir}[/dim]")
    db.close()


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

    console.print(table)

    if projects:
        ptable = Table(title="Projects", show_header=True)
        ptable.add_column("Name")
        ptable.add_column("Conversations", justify="right")
        for p in projects:
            ptable.add_row(p.name, str(p.conversation_count))
        console.print(ptable)

    db.close()


@cli.command("mcp-config")
@click.pass_context
def mcp_config(ctx):
    """Print the MCP server config block to add to claude_desktop_config.json."""
    data_dir: Path = ctx.obj["data_dir"]
    import json
    import sys

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
