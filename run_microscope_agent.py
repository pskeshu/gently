#!/usr/bin/env python3
"""
Microscopy Agent for diSPIM Control

Simple script to run the conversational agent on the microscope.
"""

import asyncio
from pathlib import Path
from gently.agent import create_copilot_with_hardware, run_rich_cli


async def main():
    # Create experiment directory
    experiment_dir = Path("./experiment_data")
    experiment_dir.mkdir(exist_ok=True)

    print("="*70)
    print("MICROSCOPY COPILOT - diSPIM Control")
    print("="*70)
    print("\nInitializing hardware control...")

    # Create copilot with hardware
    copilot = create_copilot_with_hardware(
        storage_path=experiment_dir
    )

    print("\n✓ Ready! Hardware control enabled")
    print("\nAvailable commands:")
    print('  • "Find all embryos" - Detect embryos with SAM + Claude')
    print('  • "Calibrate embryo_000" - Run calibration')
    print('  • "Take a test image of embryo_001" - Acquire volume')
    print('  • "Move to embryo_002" - Position stage')
    print('  • "/embryos" - List all embryos')
    print('  • "/status" - Show experiment status')
    print('  • "/quit" - Exit')
    print()

    # Run interactive CLI
    await run_rich_cli(
        copilot,
        history_file=experiment_dir / ".agent_history"
    )


if __name__ == "__main__":
    import os

    # Check for API key
    if not os.getenv("ANTHROPIC_API_KEY"):
        print("Error: ANTHROPIC_API_KEY not set")
        print("Set it with: export ANTHROPIC_API_KEY='your-key-here'")
        exit(1)

    asyncio.run(main())
