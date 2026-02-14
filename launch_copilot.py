#!/usr/bin/env python3
"""
Launch the Microscopy Copilot

Conversational AI agent for diSPIM microscope control.

Usage:
    python launch_copilot.py                      # Ink TUI (default)
    python launch_copilot.py --offline
    python launch_copilot.py --sessions           # List sessions and exit
    python launch_copilot.py --resume             # Interactive session picker
    python launch_copilot.py --resume latest      # Resume most recent session
    python launch_copilot.py --resume <id>        # Resume specific session
"""

import asyncio
import json
import os
import sys
import shutil
import subprocess
import argparse
from pathlib import Path
from datetime import datetime

import yaml

from gently.agent import MicroscopyCopilot, QueueServerClient
from gently.agent.logger import CopilotLogger
from gently.organisms import load_organism
from gently.hardware import load_hardware
from gently.store import GentlyStore


def _format_elapsed(last_active: str) -> str:
    """Format an ISO timestamp into a human-readable relative time."""
    if not last_active:
        return ""
    try:
        dt = datetime.fromisoformat(last_active)
        elapsed = (datetime.now() - dt).total_seconds()
        if elapsed < 60:
            return "just now"
        elif elapsed < 3600:
            return f"{int(elapsed / 60)}m ago"
        elif elapsed < 86400:
            return f"{int(elapsed / 3600)}h ago"
        else:
            return f"{int(elapsed / 86400)}d ago"
    except Exception:
        return ""


def _build_session_items(store: GentlyStore) -> list:
    """Build a list of session dicts for the Ink picker."""
    sessions = store.list_sessions()
    items = []
    for session in sessions:
        session_id = session.get("session_id", "unknown")
        embryos = store.list_embryos(session_id)
        embryo_count = len(embryos) if embryos else 0
        items.append({
            "session_id": session_id,
            "embryo_count": embryo_count,
            "time": _format_elapsed(session.get("last_active", "")),
        })
    return items


def list_sessions(store: GentlyStore):
    """List all available sessions (plain text, no Rich)."""
    items = _build_session_items(store)
    if not items:
        print("\nNo saved sessions found.")
        return

    print("\nAvailable Sessions")
    print("-" * 50)
    for item in items:
        time_str = f"  ({item['time']})" if item["time"] else ""
        print(f"  {item['session_id']}  {item['embryo_count']} embryos{time_str}")
    print()
    print("Use: python launch_copilot.py --resume <id>")


def run_ink_picker(tui_dist: Path, sessions_json: str) -> str | None:
    """
    Spawn the Ink TUI in session-picker mode and capture the selection.

    Returns the selected session ID, or None for a new session.
    """
    proc = subprocess.run(
        ["node", str(tui_dist), "--pick-session", sessions_json],
        stdin=sys.stdin,
        stdout=subprocess.PIPE,
        stderr=sys.stderr,
        text=True,
    )

    # Clear the screen so the picker output doesn't linger
    # when the main TUI takes over.
    os.system("cls" if os.name == "nt" else "clear")

    # Parse the SESSION:<id> protocol line from stdout
    for line in (proc.stdout or "").splitlines():
        if line.startswith("SESSION:"):
            selected = line[len("SESSION:"):].strip()
            return selected if selected else None

    return None


async def main(offline: bool = False, resume_session: str = None, show_sessions: bool = False, pick_session: bool = False):
    # Load organism module from config
    config_path = Path(__file__).parent / "config" / "config.yml"
    if config_path.exists():
        with open(config_path) as f:
            config = yaml.safe_load(f) or {}
    else:
        config = {}
    load_organism(config.get("organism", "celegans"))
    load_hardware(config.get("hardware", "dispim"))

    # Storage directory (unified with GentlyStore)
    storage_dir = Path("D:/Gently2")
    storage_dir.mkdir(exist_ok=True)

    # Create unified store (GentlyStore) early for session queries
    store = GentlyStore(storage_dir)

    # Handle --sessions (just list and exit)
    if show_sessions:
        list_sessions(store)
        store.close()
        return

    # Ensure TUI is available
    tui_dist = Path(__file__).parent / "gently" / "tui" / "dist" / "index.js"
    if not tui_dist.exists() or not shutil.which("node"):
        print("Error: TUI not available.")
        if not tui_dist.exists():
            print("  Run: cd gently/tui && npm install && npm run build")
        if not shutil.which("node"):
            print("  Node.js not found in PATH")
        store.close()
        return

    # Handle --resume (interactive picker, "latest", or specific session)
    session_to_resume = None
    if pick_session:
        # Two-phase launch: spawn Ink picker to select a session
        items = _build_session_items(store)
        if not items:
            print("No saved sessions found. Starting new session.")
        else:
            session_to_resume = run_ink_picker(tui_dist, json.dumps(items))
    elif resume_session == "latest":
        sessions = store.list_sessions()
        if sessions:
            session_to_resume = sessions[0].get("session_id")
        else:
            print("No sessions found - starting fresh")
    elif resume_session:
        session_to_resume = resume_session

    # Create log directory
    log_dir = storage_dir / "logs"
    log_dir.mkdir(exist_ok=True)

    # Initialize logger
    session_name = datetime.now().strftime("%Y%m%d")
    logger = CopilotLogger(log_dir, session_name=session_name)

    # Connect to device layer
    client = None
    if not offline:
        client = QueueServerClient(http_url="http://127.0.0.1:60610")
        connected = await client.connect()
        if not connected:
            await client.disconnect()
            client = None

    # Configure device session for zero-copy volume transfer
    if client and client.is_connected:
        try:
            resp = await client.configure_device_session(str(store.incoming_dir))
        except Exception:
            pass

    # Create copilot
    copilot = MicroscopyCopilot(
        microscope_client=client,
        storage_path=storage_dir,
        session_id=session_to_resume,
        store=store,
    )

    # Start visualization server for real-time feedback
    await copilot.start_viz_server(port=8080)
    viz_url = "http://localhost:8080" if copilot.viz_server is not None else None

    # Attach the copilot bridge to the viz server
    from gently.agent.copilot_bridge import CopilotBridge
    bridge = CopilotBridge(copilot)

    bridge.set_launch_info({
        "device_connected": client.is_connected if client else False,
        "sam_available": client.has_sam if client else False,
        "offline": offline or (client is None),
        "store_path": str(storage_dir),
        "viz_url": viz_url,
        "log_path": str(logger.log_file),
        "resumed": session_to_resume is not None,
    })

    # Initialize startup wizard (gap-driven onboarding)
    from gently.context import ContextStore as CtxStore
    context_db = storage_dir / "context" / "agent_mind.db"
    context_store = CtxStore(context_db)
    copilot.set_context_store(context_store)
    bridge.init_wizard(context_store=context_store, claude_client=copilot.claude)

    if copilot.viz_server is not None:
        copilot.viz_server.copilot_bridge = bridge
        copilot.viz_server.set_context_store(context_store)

    ws_url = "ws://localhost:8080/ws/copilot"

    # Spawn the Node.js TUI — it inherits stdin/stdout/stderr so Ink
    # takes over the terminal.
    tui_proc = subprocess.Popen(
        ["node", str(tui_dist), "--ws-url", ws_url],
        stdin=sys.stdin,
        stdout=sys.stdout,
        stderr=sys.stderr,
    )

    try:
        # Wait for TUI to exit (blocks the event loop in a thread so
        # the asyncio loop stays responsive for the viz server).
        exit_code = await asyncio.get_event_loop().run_in_executor(
            None, tui_proc.wait
        )
    except (KeyboardInterrupt, asyncio.CancelledError):
        tui_proc.terminate()
        try:
            tui_proc.wait(timeout=5)
        except Exception:
            pass
    finally:
        # Suppress noisy CancelledError / overlapped IO errors from
        # uvicorn during shutdown on Windows.
        import logging as _logging
        _logging.getLogger("uvicorn.error").setLevel(_logging.CRITICAL)
        _logging.getLogger("uvicorn").setLevel(_logging.CRITICAL)
        # Cleanup: stop viz server gracefully
        if copilot.viz_server is not None:
            try:
                await copilot.viz_server.stop()
            except (asyncio.CancelledError, RuntimeError, OSError, Exception):
                pass


if __name__ == "__main__":
    # Check for API key
    if not os.getenv("ANTHROPIC_API_KEY"):
        print("Error: ANTHROPIC_API_KEY not set")
        print("Set with: set ANTHROPIC_API_KEY=your-key")
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

    try:
        asyncio.run(main(
            offline=args.offline,
            show_sessions=args.sessions,
            resume_session=resume_id,
            pick_session=pick_session,
        ))
    except KeyboardInterrupt:
        pass  # Clean exit on Ctrl+C
