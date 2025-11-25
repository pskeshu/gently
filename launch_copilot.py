#!/usr/bin/env python3
"""
Launch the Microscopy Copilot

Conversational AI agent for diSPIM microscope control.

Usage:
    python launch_copilot.py
    python launch_copilot.py --offline
"""

import asyncio
import os
import argparse
from pathlib import Path
from datetime import datetime

from rich.console import Console

from gently.agent import MicroscopyCopilot, QueueServerClient, run_rich_cli
from gently.agent.startup import StartupSequence
from gently.agent.logger import CopilotLogger
from gently.agent.theme import get_theme


async def main(offline: bool = False):
    theme = get_theme()
    console = Console()

    # Storage directory
    storage_dir = Path("D:/Gently")
    storage_dir.mkdir(exist_ok=True)

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

        if not connected:
            console.print(f"\n  [{theme.warning}]{theme.icon_warning}[/] Running in offline mode")
            client = None
    else:
        console.print(f"\n[{theme.info}]{theme.icon_info}[/] Running in offline mode")

    # Create copilot
    copilot = MicroscopyCopilot(
        microscope_client=client,
        storage_path=storage_dir
    )

    # Show quick reference
    console.print(f"\n[bold {theme.primary}]Quick Commands[/]")
    console.print(f"  [{theme.info}]{theme.icon_info}[/] Type naturally: [italic]\"Find embryos\"[/], [italic]\"Calibrate embryo_001\"[/]")
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
    args = parser.parse_args()

    asyncio.run(main(offline=args.offline))
