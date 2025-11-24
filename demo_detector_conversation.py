"""
Demo: Conversational Detector Management (Phase 2 & 3)

Demonstrates the full workflow:
1. Listing detectors
2. Generating detection prompts with Claude
3. Adding detectors via conversation
4. Testing detectors
5. Managing detector status

This showcases the natural language interface for detector management.
"""

import asyncio
from pathlib import Path
from gently.agent import MicroscopyCopilot
from gently.agent.rich_cli import run_rich_cli


async def main():
    # Initialize copilot
    copilot = MicroscopyCopilot(
        storage_path=Path("./demo_detector_data"),
        model="claude-sonnet-4-5-20250929"
    )

    # Load test embryos
    test_database = {
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
        }
    }

    copilot.load_embryos_from_database(test_database)

    # Check for API key
    import os
    if not os.getenv("ANTHROPIC_API_KEY"):
        print("=" * 80)
        print("Note: ANTHROPIC_API_KEY not set")
        print("=" * 80)
        print()
        print("This demo requires the API key to:")
        print("  1. Generate detection prompts with Claude")
        print("  2. Test detectors with Claude Vision")
        print()
        print("Set your key:")
        print("  export ANTHROPIC_API_KEY='sk-ant-...'")
        print()
        print("Press Enter to continue with Rich CLI demo...")
        input()

    # Run Rich CLI
    await run_rich_cli(copilot, history_file=Path("./demo_detector_data/.copilot_history"))


if __name__ == "__main__":
    print()
    print("This demo shows the conversational interface for detector management.")
    print("The copilot uses Claude's tool calling to:")
    print("  - List detectors")
    print("  - Generate detection prompts from descriptions")
    print("  - Add detectors with custom configurations")
    print("  - Test detectors on specific embryos")
    print("  - Enable/disable/remove detectors")
    print("  - Show detection summaries")
    print()
    print("All through natural language conversation!")
    print()

    asyncio.run(main())
