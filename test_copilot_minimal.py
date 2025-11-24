"""
Minimal test script for Microscopy Copilot

Tests the copilot without requiring ophyd/bluesky dependencies.
"""

import asyncio
import sys
from pathlib import Path

# Add gently/agent to path directly
sys.path.insert(0, str(Path(__file__).parent))

# Import agent components directly
from gently.agent.copilot import MicroscopyCopilot
from gently.agent.state import EmbryoState, ExperimentState


async def main():
    """Run minimal test of copilot"""

    print("=" * 70)
    print("Microscopy Copilot - Minimal Test")
    print("=" * 70)
    print()

    # Check for API key
    import os
    if not os.getenv("ANTHROPIC_API_KEY"):
        print("⚠️  Warning: ANTHROPIC_API_KEY not set")
        print("Set it with: export ANTHROPIC_API_KEY='sk-ant-...'")
        print()
        print("Continuing with demo (will fail on actual API calls)...")
        print()

    # Initialize copilot
    print("Initializing copilot...")
    try:
        copilot = MicroscopyCopilot(
            storage_path=Path("./test_experiment_data"),
            model="claude-sonnet-4-5-20250929"
        )
        print("✓ Copilot initialized")
    except Exception as e:
        print(f"✗ Failed to initialize: {e}")
        return
    print()

    # Load example embryos
    print("Loading example embryos...")
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
    print(f"✓ Loaded {len(copilot.experiment.embryos)} embryos")
    print()

    # Test state management
    print("Testing state management...")
    summary = copilot.experiment.get_summary()
    print(summary)
    print()

    # Test quick responses (no API call)
    print("=" * 70)
    print("Testing Quick Responses (no API call)")
    print("=" * 70)
    print()

    test_messages = [
        "status",
    ]

    for msg in test_messages:
        print(f"User: {msg}")
        response = copilot._try_quick_response(msg)
        if response:
            print(f"Copilot (quick): {response[:200]}...")
        else:
            print("(Would require API call)")
        print()

    # Test with API (if key available)
    if os.getenv("ANTHROPIC_API_KEY"):
        print("=" * 70)
        print("Testing with Claude API")
        print("=" * 70)
        print()

        conversations = [
            "What embryos do we have?",
            "Can you create a plan to monitor all embryos for hatching?",
        ]

        for user_message in conversations:
            print(f"User: {user_message}")
            print()

            try:
                # Get response
                response = await copilot.handle_message(user_message)
                print(f"Copilot: {response}")
            except Exception as e:
                print(f"Error: {e}")

            print()
            print("-" * 70)
            print()

        # Interactive mode
        print("=" * 70)
        print("Interactive Mode (type 'quit' to exit)")
        print("=" * 70)
        print()

        while True:
            try:
                user_input = input("You: ").strip()

                if user_input.lower() in ['quit', 'exit', 'q']:
                    print("Goodbye!")
                    break

                if not user_input:
                    continue

                # Get response
                response = await copilot.handle_message(user_input)
                print(f"\nCopilot: {response}\n")

            except KeyboardInterrupt:
                print("\n\nGoodbye!")
                break
            except Exception as e:
                print(f"\nError: {e}\n")

    else:
        print("=" * 70)
        print("Set ANTHROPIC_API_KEY to test full API functionality")
        print("=" * 70)


if __name__ == "__main__":
    print("Note: This is a minimal test that doesn't require ophyd/bluesky")
    print("For full functionality, use the complete gently package")
    print()

    asyncio.run(main())
