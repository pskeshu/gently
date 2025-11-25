#!/usr/bin/env python3
"""
Microscopy Agent for diSPIM Control

Conversational AI agent for microscope control using Bluesky Queue Server.

This script connects to:
1. Bluesky Queue Server (port 60610) - for RunEngine/hardware control
2. SAM Detection Server (port 18862) - for embryo detection

Usage:
    1. Start Queue Server: start_queue_server.bat
    2. Start SAM Server: python backend/sam_server.py
    3. Run this script: python run_microscope_agent.py
"""

import asyncio
import os
from pathlib import Path
from datetime import datetime

from rich.console import Console

from gently.agent import MicroscopyCopilot, QueueServerClient, run_rich_cli
from gently.agent.startup import StartupSequence
from gently.agent.logger import CopilotLogger
from gently.agent.theme import get_theme, set_theme


async def main():
    theme = get_theme()
    console = Console()

    # Create experiment directory
    experiment_dir = Path("./experiment_data")
    experiment_dir.mkdir(exist_ok=True)

    # Create log directory
    log_dir = experiment_dir / "logs"
    log_dir.mkdir(exist_ok=True)

    # Initialize logger for session
    session_name = datetime.now().strftime("%Y%m%d")
    logger = CopilotLogger(log_dir, session_name=session_name)

    # Show startup sequence
    startup = StartupSequence(console=console, logger=logger)
    startup.show_banner()

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
        # Get queue server status
        status = await client.get_status()
        qs_status = status.get('queue_server', {})
        console.print(f"  [{theme.info}]{theme.icon_info}[/] Manager state: {qs_status.get('manager_state', 'unknown')}")
        console.print(f"  [{theme.info}]{theme.icon_info}[/] RE state: {qs_status.get('re_state', 'unknown')}")
    else:
        console.print(f"  [{theme.warning}]{theme.icon_warning}[/] Queue Server not connected")
        console.print(f"  [{theme.muted}]Start with: start_queue_server.bat[/]")

    if client.has_sam:
        console.print(f"  [{theme.success}]{theme.icon_success}[/] SAM Server connected")
    else:
        console.print(f"  [{theme.warning}]{theme.icon_warning}[/] SAM Server not connected")
        console.print(f"  [{theme.muted}]Start with: python backend/sam_server.py[/]")
        console.print(f"  [{theme.muted}]Embryo detection will not be available[/]")

    if not connected:
        console.print(f"\n  [{theme.warning}]{theme.icon_warning}[/] Running in offline mode (limited functionality)")
        client = None

    # Create copilot with queue server client
    copilot = MicroscopyCopilot(
        microscope_client=client,
        storage_path=experiment_dir
    )

    # Show quick reference
    console.print(f"\n[bold {theme.primary}]Quick Commands[/]")
    console.print(f"  [{theme.info}]{theme.icon_info}[/] Type naturally: [italic]\"Find embryos\"[/], [italic]\"Calibrate embryo_001\"[/]")
    console.print(f"  [{theme.info}]{theme.icon_info}[/] Slash commands: [{theme.tool}]/embryos[/], [{theme.tool}]/status[/], [{theme.tool}]/theme[/], [{theme.tool}]/help[/]")
    console.print(f"  [{theme.muted}]Log file: {logger.log_file}[/]\n")

    # Run interactive CLI
    await run_rich_cli(
        copilot,
        history_file=experiment_dir / ".agent_history"
    )


if __name__ == "__main__":
    # Check for API key
    if not os.getenv("ANTHROPIC_API_KEY"):
        console = Console()
        theme = get_theme()
        console.print(f"[{theme.error}]{theme.icon_error}[/] [bold]Error:[/] ANTHROPIC_API_KEY not set")
        console.print(f"[{theme.muted}]Set it with: export ANTHROPIC_API_KEY='your-key-here'[/]")
        exit(1)

    asyncio.run(main())
