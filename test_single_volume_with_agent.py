#!/usr/bin/env python3
"""
Test Single Volume Acquisition with Agent Control

This script demonstrates using the MicroscopyCopilot to acquire a single volume.
You can use the conversational interface or directly call the tool.
"""

import asyncio
from pathlib import Path
from gently.agent import create_copilot_with_hardware

async def main():
    print("="*70)
    print("SINGLE VOLUME ACQUISITION TEST WITH AGENT")
    print("="*70)

    # Create copilot with hardware control
    print("\nInitializing copilot with hardware control...")
    copilot = create_copilot_with_hardware(
        storage_path=Path("./test_volume_data")
    )

    # Load embryos from database
    print("\nLoading embryos...")
    database = {
        'embryos': {
            'embryo_001': {
                'embryo_number': 1,
                'stage_position_after_centering_um': {'x': 1000.0, 'y': 500.0},
                'calibration': {
                    'offset': 0.0,  # galvo center
                    'galvo_amplitude': 0.5,
                    'piezo_center': 50.0,
                    'piezo_amplitude': 25.0,
                }
            }
        }
    }
    copilot.load_embryos_from_database(database)
    print(f"✓ Loaded {len(copilot.experiment.embryos)} embryo(s)")

    # Option 1: Use conversational interface
    print("\n" + "="*70)
    print("OPTION 1: Conversational Interface")
    print("="*70)
    print("\nYou can ask the agent to acquire a volume:")
    print('  await copilot.handle_message("Acquire a volume for embryo_001")')

    # Option 2: Direct tool call (useful for testing)
    print("\n" + "="*70)
    print("OPTION 2: Direct Tool Call")
    print("="*70)

    print("\nAcquiring volume directly...")
    result = await copilot._tool_acquire_volume({
        'embryo_id': 'embryo_001',
        'num_slices': 50,
        'exposure_ms': 10.0,
        'save': True
    })

    print("\n" + result)

    # Check what was saved
    print("\n" + "="*70)
    print("VERIFICATION")
    print("="*70)

    embryo = copilot.experiment.embryos['embryo_001']
    print(f"\nEmbryo state:")
    print(f"  Last timepoint: {embryo.last_acquisition_timepoint}")
    print(f"  Last acquisition: {embryo.last_acquisition_time}")

    # Check image manager
    latest = copilot.image_manager.get_latest_volume(embryo)
    if latest:
        print(f"\nStored volume:")
        print(f"  Shape: {latest.shape}")
        print(f"  Timepoint: {embryo.last_acquisition_timepoint}")
        print(f"  Location: {copilot.image_manager.storage_path}")

    print("\n" + "="*70)
    print("✓ TEST COMPLETE")
    print("="*70)
    print("\nThe volume has been acquired and stored!")
    print("You can now:")
    print("  1. View it with: copilot.image_manager.get_latest_volume(embryo)")
    print("  2. Analyze it with Claude Vision")
    print("  3. Run detectors on it")


async def test_conversational():
    """Alternative: Full conversational test"""
    print("\n" + "="*70)
    print("CONVERSATIONAL TEST")
    print("="*70)

    copilot = create_copilot_with_hardware(
        storage_path=Path("./test_volume_data")
    )

    # Load embryos
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
            }
        }
    }
    copilot.load_embryos_from_database(database)

    # Ask agent to acquire volume
    print("\nAsking agent: 'Take a test image of embryo_001'")
    response = await copilot.handle_message("Take a test image of embryo_001")
    print(f"\nAgent response:\n{response}")


if __name__ == "__main__":
    import sys

    # Check for API key
    import os
    if not os.getenv("ANTHROPIC_API_KEY"):
        print("Warning: ANTHROPIC_API_KEY not set")
        print("The direct tool call will work, but conversational interface needs the key")
        print()

    # Choose test mode
    if len(sys.argv) > 1 and sys.argv[1] == "conversational":
        asyncio.run(test_conversational())
    else:
        asyncio.run(main())
