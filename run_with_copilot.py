"""
Run multi-embryo volume acquisition with Copilot

This demonstrates how the copilot integrates with actual microscope control.
The copilot can:
- Generate acquisition plans from natural language
- Monitor experiments in real-time
- Adapt parameters dynamically
- Answer questions during acquisition
"""

import asyncio
import json
from pathlib import Path
from datetime import datetime

from gently.agent import MicroscopyCopilot


async def run_with_copilot():
    """Run acquisition with copilot orchestration"""

    print("=" * 80)
    print("Multi-Embryo Volume Acquisition with AI Copilot")
    print("=" * 80)
    print()

    # Initialize copilot
    storage_path = Path("./experiment_data")
    copilot = MicroscopyCopilot(
        storage_path=storage_path,
        model="claude-sonnet-4-5-20250929"
    )

    print("✓ Copilot initialized")
    print()

    # Load embryo database
    database_path = Path("embryo_database.json")

    if not database_path.exists():
        print(f"Error: Embryo database not found at {database_path}")
        print("Please run calibration first to generate embryo_database.json")
        return

    with open(database_path, 'r') as f:
        database = json.load(f)

    copilot.load_embryos_from_database(database)
    print(f"✓ Loaded {len(copilot.experiment.embryos)} embryos from database")
    print()

    # Get user's experimental goal
    print("What would you like to do?")
    print()
    print("Examples:")
    print("  - 'Monitor all embryos for hatching'")
    print("  - 'Image embryos every 2 minutes with minimal photobleaching'")
    print("  - 'Take high-resolution volumes of embryo 1'")
    print("  - 'Track development and detect first cell division'")
    print()

    goal = input("Your goal: ").strip()

    if not goal:
        goal = "Monitor all embryos and detect when they hatch"
        print(f"Using default: {goal}")

    print()
    print("=" * 80)
    print()

    # Ask copilot to create plan
    response = await copilot.handle_message(goal)
    print(f"Copilot: {response}")
    print()

    # Confirm execution
    confirm = input("Proceed with this plan? [Y/n]: ").strip().lower()

    if confirm and confirm not in ['y', 'yes', '']:
        print("Cancelled.")
        return

    print()
    print("=" * 80)
    print("Starting Acquisition")
    print("=" * 80)
    print()

    # Get the generated plan
    if not copilot.experiment.plan_history:
        print("No plan generated. Please try again.")
        return

    latest_plan = copilot.experiment.plan_history[-1]
    plan_code = latest_plan['code']

    # In a real implementation, this would:
    # 1. Execute the generated Bluesky plan
    # 2. Stream volumes to copilot for analysis
    # 3. Allow user to query status during acquisition
    # 4. Adapt parameters based on copilot decisions

    print("Plan generated successfully!")
    print()
    print("Generated plan:")
    print("-" * 80)
    print(plan_code[:500] + "..." if len(plan_code) > 500 else plan_code)
    print("-" * 80)
    print()

    print("To execute this plan:")
    print("1. The plan code is stored in copilot.experiment.plan_history")
    print("2. Execute with Bluesky RunEngine:")
    print("   from bluesky import RunEngine")
    print("   RE = RunEngine()")
    print("   RE(adaptive_timelapse_plan(...))")
    print("3. During acquisition, call copilot.on_volume_acquired() for each volume")
    print("4. Users can query status: await copilot.handle_message('What is happening?')")
    print()

    # Simulate interactive monitoring
    print("=" * 80)
    print("Interactive Monitoring (simulation)")
    print("=" * 80)
    print()
    print("During acquisition, you can ask the copilot questions:")
    print("- 'What's the status?'")
    print("- 'Is embryo 2 hatching yet?'")
    print("- 'Focus more on embryo 3'")
    print("- 'How much longer until hatching?'")
    print()
    print("Type 'quit' to exit")
    print()

    copilot.experiment.acquisition_status = "running"
    copilot.experiment.start_time = datetime.now()

    while True:
        try:
            query = input("Ask copilot: ").strip()

            if query.lower() in ['quit', 'exit', 'q']:
                break

            if not query:
                continue

            response = await copilot.handle_message(query)
            print(f"\nCopilot: {response}\n")

        except KeyboardInterrupt:
            print("\n\nStopping...")
            break

    copilot.experiment.acquisition_status = "paused"
    print("\nExperiment paused.")


if __name__ == "__main__":
    import os

    # Check for API key
    if not os.getenv("ANTHROPIC_API_KEY"):
        print("Error: ANTHROPIC_API_KEY environment variable not set")
        print()
        print("Set your Anthropic API key:")
        print("  export ANTHROPIC_API_KEY='sk-ant-...'")
        print()
        print("Get your key from: https://console.anthropic.com/")
        exit(1)

    # Check for database
    if not Path("embryo_database.json").exists():
        print("Note: embryo_database.json not found")
        print("The script will create a demo database for testing")
        print()

        # Create demo database
        demo_db = {
            'embryos': {
                f'embryo_{i:03d}': {
                    'embryo_number': i,
                    'stage_position_after_centering_um': {
                        'x': 1000.0 + i * 200,
                        'y': 500.0 + i * 100
                    },
                    'calibration': {
                        'galvo_amplitude': 8.0,
                        'galvo_center': 0.0,
                        'piezo_amplitude': 50.0,
                        'piezo_center': 0.0,
                    }
                }
                for i in range(1, 7)  # 6 embryos
            }
        }

        with open("embryo_database.json", 'w') as f:
            json.dump(demo_db, f, indent=2)

        print("✓ Created demo embryo_database.json with 6 embryos")
        print()

    # Run
    asyncio.run(run_with_copilot())
