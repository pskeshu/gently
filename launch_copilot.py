#!/usr/bin/env python3
"""
Launch the Microscopy Copilot

Conversational AI agent for diSPIM microscope control.

Usage:
    python launch_copilot.py
    python launch_copilot.py --offline
    python launch_copilot.py --sessions          # List and select a session
    python launch_copilot.py --resume            # Interactive session picker
    python launch_copilot.py --resume <id>       # Resume specific session
"""

import asyncio
import os
import argparse
from pathlib import Path
from datetime import datetime

from rich.console import Console
from rich.table import Table
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
from gently.session import SessionManager


async def show_session_picker(storage_dir: Path, console: Console) -> str:
    """Show interactive session picker before copilot starts."""
    theme = get_theme()

    # Create a temporary session manager to list sessions
    session_mgr = SessionManager(sessions_dir=storage_dir / "sessions")
    sessions = session_mgr.list_sessions()

    if not sessions:
        console.print(f"\n[{theme.muted}]No saved sessions found. Starting new session.[/]")
        return None

    # Build session items
    items = []
    for session in sessions:
        session_id = session.get('session_id', session.get('id', 'unknown'))
        name = session.get('name', '')
        embryo_count = session.get('embryo_count', 0)
        message_count = session.get('message_count', 0)

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
            'messages': message_count,
            'time': time_str
        })

    # Add "New Session" option at top
    items.insert(0, {
        'id': None,
        'name': 'Start fresh',
        'embryos': 0,
        'messages': 0,
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
                    line += f' │ {item["embryos"]} embryos │ {item["messages"]} msgs'
                    if item['time']:
                        line += f' │ {item["time"]}'
                    line += '</style>\n'
                    text += line
                else:
                    line = f'{marker}<style fg="#aaaaaa">{item["id"]}</style>'
                    line += f' <style fg="#666666">│ {item["embryos"]} embryos │ {item["messages"]} msgs'
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


def list_sessions(storage_dir: Path, console: Console):
    """List all available sessions (non-interactive)."""
    theme = get_theme()

    session_mgr = SessionManager(sessions_dir=storage_dir / "sessions")
    sessions = session_mgr.list_sessions()

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
    table.add_column("Messages", justify="center")
    table.add_column("Last Active", style=theme.muted)

    for session in sessions:
        session_id = session.get('session_id', session.get('id', 'unknown'))

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
            str(session.get('embryo_count', 0)),
            str(session.get('message_count', 0)),
            last_active,
        )

    console.print(table)
    console.print(f"\n[{theme.muted}]Use: python launch_copilot.py --resume <id>[/]")


async def main(offline: bool = False, resume_session: str = None, show_sessions: bool = False, pick_session: bool = False):
    theme = get_theme()
    console = Console()

    # Storage directory
    storage_dir = Path("D:/Gently")
    storage_dir.mkdir(exist_ok=True)

    # Handle --sessions (just list and exit)
    if show_sessions:
        list_sessions(storage_dir, console)
        return

    # Handle --resume (interactive picker or specific session)
    session_to_resume = None
    if pick_session:
        # Show interactive session picker
        session_to_resume = await show_session_picker(storage_dir, console)
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
        # Connect to servers
        console.print(f"\n[bold {theme.primary}]Connecting to Servers[/]")

        client = QueueServerClient(
            http_url="http://127.0.0.1:60610",
            sam_host="localhost",
            sam_port=18862
        )

        connected = await client.connect()

        if client.is_connected:
            console.print(f"  [{theme.success}]{theme.icon_success}[/] Queue Server connected")
            status = await client.get_status()
            qs_status = status.get('queue_server', {})
            console.print(f"  [{theme.info}]{theme.icon_info}[/] Manager: {qs_status.get('manager_state', 'unknown')}")
        else:
            console.print(f"  [{theme.warning}]{theme.icon_warning}[/] Queue Server not connected")

        if client.has_sam:
            console.print(f"  [{theme.success}]{theme.icon_success}[/] SAM Server connected")
        else:
            console.print(f"  [{theme.warning}]{theme.icon_warning}[/] SAM Server not connected")

        # Databroker status
        if client.has_databroker:
            console.print(f"  [{theme.success}]{theme.icon_success}[/] Databroker connected")
        else:
            console.print(f"  [{theme.warning}]{theme.icon_warning}[/] Databroker not connected")

        if not connected:
            console.print(f"\n  [{theme.warning}]{theme.icon_warning}[/] Running in offline mode")
            client = None
    else:
        console.print(f"\n[{theme.info}]{theme.icon_info}[/] Running in offline mode")

    # Create copilot
    copilot = MicroscopyCopilot(
        microscope_client=client,
        storage_path=storage_dir,
        session_id=session_to_resume  # Resume specific session if provided
    )

    # Report session status
    if session_to_resume:
        console.print(f"\n[{theme.success}]{theme.icon_success}[/] Resumed session: {session_to_resume}")
        session = copilot.session_manager.current_session
        if session:
            console.print(f"  [{theme.muted}]{session.embryo_count} embryos, {session.message_count} messages[/]")
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
