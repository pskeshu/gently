#!/usr/bin/env python3
"""
Test Agent Hardware Control with Rich CLI

Interactive test of the conversational device control.
Just run and chat with the agent to control the microscope!

Usage:
    python test_agent_hardware_control.py
"""

import asyncio
from pathlib import Path
from gently.agent import create_copilot_with_hardware, run_rich_cli


async def main():
    """Run interactive test with Rich CLI"""

    print("="*70)
    print("AGENT HARDWARE CONTROL TEST")
    print("="*70)
    print("\nInitializing copilot with hardware control...")
    print("This will create devices from Micro-Manager core...\n")

    # Create copilot with full hardware control
    copilot = create_copilot_with_hardware(
        storage_path=Path("./test_agent_control")
    )

    # Load test embryos
    print("\nLoading test embryos...")
    database = {
        'embryos': {
            'embryo_001': {
                'embryo_number': 1,
                'stage_position_after_centering_um': {'x': 1000.0, 'y': 500.0},
                'calibration': {
                    'offset': 0.0,
                    'galvo_amplitude': 0.5,
                    'piezo_center': 50.0,
                    'piezo_amplitude': 25.0,
                }
            },
            'embryo_002': {
                'embryo_number': 2,
                'stage_position_after_centering_um': {'x': 1200.0, 'y': 600.0},
                'calibration': {
                    'offset': 0.0,
                    'galvo_amplitude': 0.5,
                    'piezo_center': 50.0,
                    'piezo_amplitude': 25.0,
                }
            }
        }
    }
    copilot.load_embryos_from_database(database)

    print(f"✓ Loaded {len(copilot.experiment.embryos)} embryos")
    print("\n" + "="*70)
    print("READY FOR INTERACTIVE CONTROL")
    print("="*70)
    print("\nTry these commands:")
    print('  • "Take a test image of embryo_001"')
    print('  • "Move to embryo_002"')
    print('  • "Acquire a 100-slice volume of embryo_001"')
    print('  • "What is the status of embryo_001?"')
    print('  • "/embryos" to see all embryos')
    print('  • "/quit" to exit')
    print("\n" + "="*70 + "\n")

    # Run Rich CLI
    await run_rich_cli(
        copilot,
        history_file=Path("./test_agent_control/.agent_history")
    )


if __name__ == "__main__":
    # Check for API key
    import os
    if not os.getenv("ANTHROPIC_API_KEY"):
        print("Error: ANTHROPIC_API_KEY environment variable not set")
        print("Please set your API key:")
        print("  export ANTHROPIC_API_KEY='your-key-here'")
        exit(1)

    # Run
    asyncio.run(main())
