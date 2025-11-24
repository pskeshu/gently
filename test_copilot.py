"""
Test script for Microscopy Copilot

Demonstrates the conversational AI interface for microscopy experiments.
"""

import asyncio
from pathlib import Path
from gently.agent import MicroscopyCopilot
from gently.agent.rich_cli import run_rich_cli


async def main():
    """Run interactive test of copilot with Rich CLI"""

    # Initialize copilot
    copilot = MicroscopyCopilot(
        storage_path=Path("./test_experiment_data"),
        model="claude-sonnet-4-5-20250929"
    )

    # Load example embryos
    example_database = {
        'embryos': {
            'embryo_001': {
                'embryo_number': 1,
                'stage_position_after_centering_um': {'x': 1000.0, 'y': 500.0},
                'calibration': {
                    'galvo_amplitude': 8.0,
                    'galvo_center': 0.0,
                    'piezo_amplitude': 50.0,
                    'piezo_center': 0.0,
                }
            },
            'embryo_002': {
                'embryo_number': 2,
                'stage_position_after_centering_um': {'x': 1200.0, 'y': 600.0},
                'calibration': {
                    'galvo_amplitude': 8.0,
                    'galvo_center': 0.0,
                    'piezo_amplitude': 55.0,
                    'piezo_center': 0.0,
                }
            },
            'embryo_003': {
                'embryo_number': 3,
                'stage_position_after_centering_um': {'x': 1400.0, 'y': 700.0},
                'calibration': {
                    'galvo_amplitude': 8.0,
                    'galvo_center': 0.0,
                    'piezo_amplitude': 48.0,
                    'piezo_center': 0.0,
                }
            },
        }
    }

    copilot.load_embryos_from_database(example_database)

    # Run Rich CLI
    await run_rich_cli(copilot, history_file=Path("./test_experiment_data/.copilot_history"))


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
