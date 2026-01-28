#!/usr/bin/env python3
"""
Launch the Microscopy Copilot

Conversational AI agent for diSPIM microscope control.

Usage:
    python launch_copilot.py
    python launch_copilot.py --offline
    python launch_copilot.py --sessions          # List and select a session
    python launch_copilot.py --resume            # Interactive session picker
    python launch_copilot.py --resume latest     # Resume most recent session
    python launch_copilot.py --resume <id>       # Resume specific session
"""

import asyncio
import os
import argparse
from pathlib import Path
from datetime import datetime

from rich.console import Console
from rich.table import Table
from rich.text import Text
from rich import box

from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.application import Application
from prompt_toolkit.layout import Layout
from prompt_toolkit.layout.containers import Window, HSplit
from prompt_toolkit.layout.controls import FormattedTextControl
from prompt_toolkit.formatted_text import HTML

from gently.agent import MicroscopyCopilot, QueueServerClient, run_rich_cli
from gently.agent.startup import StartupSequence
from gently.agent.logger import CopilotLogger
from gently.agent.theme import get_theme
from gently.store import GentlyStore


async def show_session_picker(store: GentlyStore, console: Console) -> str:
    """Show interactive session picker before copilot starts."""
    theme = get_theme()

    # List sessions from GentlyStore (SQLite)
    sessions = store.list_sessions()

    if not sessions:
        console.print(f"\n[{theme.muted}]No saved sessions found. Starting new session.[/]")
        return None

    # Build session items
    items = []
    for session in sessions:
        session_id = session.get('session_id', 'unknown')
        name = session.get('name', '')
        # Get embryo count from store
        embryos = store.list_embryos(session_id)
        embryo_count = len(embryos) if embryos else 0

        # Format last active time
        last_active = session.get('last_active', '')
        time_str = ''
        if last_active:
            try:
                dt = datetime.fromisoformat(last_active)
                elapsed = (datetime.now() - dt).total_seconds()
                if elapsed < 60:
                    time_str = "just now"
                elif elapsed < 3600:
                    time_str = f"{int(elapsed / 60)}m ago"
                elif elapsed < 86400:
                    time_str = f"{int(elapsed / 3600)}h ago"
                else:
                    time_str = f"{int(elapsed / 86400)}d ago"
            except:
                pass

        items.append({
            'id': session_id,
            'name': name,
            'embryos': embryo_count,
            'time': time_str
        })

    # Add "New Session" option at top
    items.insert(0, {
        'id': None,
        'name': 'Start fresh',
        'embryos': 0,
        'time': '',
        'is_new': True
    })

    # State for the picker
    selected_idx = [0]
    result = [None]
    cancelled = [False]

    # Key bindings
    kb = KeyBindings()

    @kb.add('up')
    @kb.add('k')
    def move_up(event):
        selected_idx[0] = max(0, selected_idx[0] - 1)

    @kb.add('down')
    @kb.add('j')
    def move_down(event):
        selected_idx[0] = min(len(items) - 1, selected_idx[0] + 1)

    @kb.add('enter')
    def select(event):
        result[0] = items[selected_idx[0]]['id']
        event.app.exit()

    @kb.add('escape')
    @kb.add('q')
    def cancel(event):
        cancelled[0] = True
        event.app.exit()

    def get_formatted_text():
        text = '<b>Select a session:</b> (↑/↓ navigate, Enter select, Esc cancel)\n'
        text += '─' * 70 + '\n'

        for i, item in enumerate(items):
            is_selected = (i == selected_idx[0])
            marker = '▶ ' if is_selected else '  '

            if item.get('is_new'):
                # New session option
                if is_selected:
                    text += f'<style bg="#006600" fg="white"><b>{marker}+ New Session</b> (start fresh)</style>\n'
                else:
                    text += f'{marker}<style fg="#00aa00">+ New Session</style> <style fg="#666666">(start fresh)</style>\n'
            else:
                if is_selected:
                    line = f'<style bg="#0066cc" fg="white"><b>{marker}{item["id"]}</b>'
                    line += f' │ {item["embryos"]} embryos'
                    if item['time']:
                        line += f' │ {item["time"]}'
                    line += '</style>\n'
                    text += line
                else:
                    line = f'{marker}<style fg="#aaaaaa">{item["id"]}</style>'
                    line += f' <style fg="#666666">│ {item["embryos"]} embryos'
                    if item['time']:
                        line += f' │ {item["time"]}'
                    line += '</style>\n'
                    text += line

        text += '─' * 70
        return HTML(text)

    # Create the application
    layout = Layout(
        HSplit([
            Window(
                content=FormattedTextControl(get_formatted_text),
                wrap_lines=False
            )
        ])
    )

    app = Application(
        layout=layout,
        key_bindings=kb,
        full_screen=False,
        mouse_support=False
    )

    console.print(f"\n[bold {theme.primary}]Session Manager[/]\n")

    try:
        await app.run_async()
        if cancelled[0]:
            console.print(f"[{theme.muted}]Cancelled. Starting new session.[/]")
            return None
        return result[0]
    except Exception as e:
        console.print(f"[{theme.error}]Error: {e}[/]")
        return None


def list_sessions(store: GentlyStore, console: Console):
    """List all available sessions (non-interactive)."""
    theme = get_theme()

    sessions = store.list_sessions()

    if not sessions:
        console.print(f"\n[{theme.muted}]No saved sessions found.[/]")
        return

    table = Table(
        title="Available Sessions",
        box=box.ROUNDED,
        show_header=True,
        header_style=f"bold {theme.secondary}",
    )

    table.add_column("ID", style=theme.info)
    table.add_column("Embryos", justify="center")
    table.add_column("Last Active", style=theme.muted)

    for session in sessions:
        session_id = session.get('session_id', 'unknown')
        embryo_count = len(store.list_embryos(session_id))

        # Format last active time
        last_active = session.get('last_active', '')
        if last_active:
            try:
                dt = datetime.fromisoformat(last_active)
                elapsed = (datetime.now() - dt).total_seconds()
                if elapsed < 60:
                    last_active = "just now"
                elif elapsed < 3600:
                    last_active = f"{int(elapsed / 60)}m ago"
                elif elapsed < 86400:
                    last_active = f"{int(elapsed / 3600)}h ago"
                else:
                    last_active = f"{int(elapsed / 86400)}d ago"
            except:
                pass

        table.add_row(
            session_id,
            str(embryo_count),
            last_active,
        )

    console.print(table)
    console.print(f"\n[{theme.muted}]Use: python launch_copilot.py --resume <id>[/]")


async def main(offline: bool = False, resume_session: str = None, show_sessions: bool = False, pick_session: bool = False):
    theme = get_theme()
    console = Console()

    # Storage directory (unified with GentlyStore)
    storage_dir = Path("D:/Gently2")
    storage_dir.mkdir(exist_ok=True)

    # Create unified store (GentlyStore) early for session queries
    store = GentlyStore(storage_dir)

    # Handle --sessions (just list and exit)
    if show_sessions:
        list_sessions(store, console)
        store.close()
        return

    # Handle --resume (interactive picker, "latest", or specific session)
    session_to_resume = None
    if pick_session:
        # Show interactive session picker
        session_to_resume = await show_session_picker(store, console)
    elif resume_session == "latest":
        # Find the most recent session from GentlyStore
        sessions = store.list_sessions()
        if sessions:
            # Sessions are sorted by last_active (most recent first)
            latest = sessions[0]
            session_to_resume = latest.get('session_id')
            console.print(f"\n[{theme.info}]{theme.icon_info}[/] Resuming latest session: [bold]{session_to_resume}[/]")
        else:
            console.print(f"\n[{theme.warning}]{theme.icon_warning}[/] No sessions found - starting fresh")
    elif resume_session:
        session_to_resume = resume_session

    # Create log directory
    log_dir = storage_dir / "logs"
    log_dir.mkdir(exist_ok=True)

    # Initialize logger
    session_name = datetime.now().strftime("%Y%m%d")
    logger = CopilotLogger(log_dir, session_name=session_name)

    # Show startup banner
    startup = StartupSequence(console=console, logger=logger)
    startup.show_banner()

    client = None

    if not offline:
        # Connect to device layer server (unified HTTP API for hardware + SAM)
        client = QueueServerClient(
            http_url="http://127.0.0.1:60610"
        )

        connected = await client.connect()

        # Build status table
        status_lines = []

        if client.is_connected:
            status_lines.append((theme.icon_success, "Device Layer", "connected", theme.success))
            status = await client.get_status()
            qs_status = status.get('queue_server', {})
            manager_state = qs_status.get('manager_state', 'unknown')
            status_lines.append((theme.icon_info, "  Manager", manager_state, theme.muted))
        else:
            status_lines.append((theme.icon_error, "Device Layer", "not connected", theme.error))

        if client.has_sam:
            status_lines.append((theme.icon_success, "SAM Detection", "available", theme.success))
        else:
            status_lines.append((theme.icon_warning, "SAM Detection", "not available", theme.warning))

        # Print status table
        table = Table(show_header=False, box=None, padding=(0, 1))
        table.add_column("icon", width=2)
        table.add_column("service", min_width=14)
        table.add_column("status")

        for icon, service, status_text, style in status_lines:
            table.add_row(
                Text(icon, style=style),
                Text(service, style="bold" if not service.startswith(" ") else theme.muted),
                Text(status_text, style=style)
            )

        console.print()
        console.print(table)

        if not connected:
            console.print(f"\n[{theme.warning}]{theme.icon_warning} Running in offline mode[/]")
            # Close the session before discarding client
            await client.disconnect()
            client = None
    else:
        console.print(f"\n[{theme.muted}]{theme.icon_info} Offline mode[/]")

    # Store was created earlier for session queries
    console.print(f"  [{theme.muted}]Store: {storage_dir}[/]")

    # Configure device session for zero-copy volume transfer
    if client and client.is_connected:
        try:
            resp = await client.configure_device_session(str(store.incoming_dir))
            if resp.get("success"):
                console.print(f"  [{theme.muted}]Volume staging: {store.incoming_dir}[/]")
        except Exception as e:
            console.print(f"  [{theme.warning}]Could not configure device session: {e}[/]")

    # Create copilot
    copilot = MicroscopyCopilot(
        microscope_client=client,
        storage_path=storage_dir,
        session_id=session_to_resume,  # Resume specific session if provided
        store=store,
    )

    # Start visualization server for real-time feedback
    await copilot.start_viz_server(port=8080)
    if copilot.viz_server is not None:
        console.print(f"\n[{theme.info}]{theme.icon_info}[/] Viz dashboard: [link=http://localhost:8080]http://localhost:8080[/link]")

    # Report session status
    if session_to_resume:
        console.print(f"\n[{theme.success}]{theme.icon_success}[/] Resumed session: {session_to_resume}")
        embryo_count = len(store.list_embryos(session_to_resume))
        console.print(f"  [{theme.muted}]{embryo_count} embryos[/]")
    else:
        console.print(f"\n[{theme.info}]{theme.icon_info}[/] New session: {copilot.session_id}")

    # Show quick reference
    console.print(f"\n[bold {theme.primary}]Quick Commands[/]")
    console.print(f"  [{theme.info}]{theme.icon_info}[/] Type naturally: [italic]\"Find embryos\"[/], [italic]\"Calibrate embryo_1\"[/]")
    console.print(f"  [{theme.info}]{theme.icon_info}[/] Slash commands: [{theme.tool}]/embryos[/], [{theme.tool}]/status[/], [{theme.tool}]/help[/]")
    console.print(f"  [{theme.muted}]Log: {logger.log_file}[/]\n")

    # Run CLI
    await run_rich_cli(copilot, history_file=storage_dir / ".copilot_history")


if __name__ == "__main__":
    # Check for API key
    if not os.getenv("ANTHROPIC_API_KEY"):
        console = Console()
        theme = get_theme()
        console.print(f"[{theme.error}]{theme.icon_error}[/] ANTHROPIC_API_KEY not set")
        console.print(f"[{theme.muted}]Set with: set ANTHROPIC_API_KEY=your-key[/]")
        exit(1)

    parser = argparse.ArgumentParser(description="Launch Microscopy Copilot")
    parser.add_argument("--offline", action="store_true", help="Run without server connections")
    parser.add_argument("--sessions", action="store_true", help="List available sessions and exit")
    parser.add_argument("--resume", nargs="?", const="__PICK__", metavar="ID",
                        help="Resume a session. Without ID: shows picker. With ID: resumes that session.")
    args = parser.parse_args()

    # Determine resume mode
    pick_session = (args.resume == "__PICK__")
    resume_id = args.resume if args.resume and args.resume != "__PICK__" else None

    asyncio.run(main(
        offline=args.offline,
        show_sessions=args.sessions,
        resume_session=resume_id,
        pick_session=pick_session
    ))
