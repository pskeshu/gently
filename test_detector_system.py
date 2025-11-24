"""
Test script for the generic detector system

Demonstrates creating detectors, running them, and checking results.
"""

import asyncio
import numpy as np
from pathlib import Path
from gently.agent import (
    MicroscopyCopilot,
    Detector,
    DetectorConditions,
    DetectorActions,
    DetectionMode,
    ConfidenceLevel
)


async def main():
    print("=" * 70)
    print("Detector System Test")
    print("=" * 70)
    print()

    # Initialize copilot
    print("1. Initializing copilot...")
    copilot = MicroscopyCopilot(
        storage_path=Path("./test_detector_data"),
        model="claude-sonnet-4-5-20250929"
    )
    print("[OK] Copilot initialized")
    print()

    # Load test embryos
    print("2. Loading test embryos...")
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
    print(f"[OK] Loaded {len(copilot.experiment.embryos)} embryos")
    print()

    # Add preset detectors
    print("3. Adding preset detectors...")

    # Hatching detector
    hatching = copilot.detector_registry.create_preset_detector('hatching')
    if hatching:
        hatching.conditions.min_timepoint = 10  # Lower for testing
        hatching.actions.mode = DetectionMode.RECOMMEND
        copilot.detector_registry.add(hatching)
        print(f"  [+] Added: {hatching.name} - {hatching.description}")

    # Comma stage detector
    comma = copilot.detector_registry.create_preset_detector('comma')
    if comma:
        comma.conditions.min_timepoint = 5
        comma.actions.mode = DetectionMode.AUTO
        comma.actions.parameter_changes = {
            "interval_seconds": 60,
            "num_slices": 80
        }
        copilot.detector_registry.add(comma)
        print(f"  [+] Added: {comma.name} - {comma.description}")

    print()

    # Create custom detector
    print("4. Creating custom detector...")
    custom_detector = Detector(
        name="test_detector",
        description="Test detector for demonstration",
        detection_prompt="""Analyze this image and always respond with:
DETECTED: YES
CONFIDENCE: MEDIUM
REASONING: This is a test detection""",
        enabled=True,
        use_temporal_context=False,
        confidence_threshold=ConfidenceLevel.LOW,
        conditions=DetectorConditions(
            min_timepoint=0,
            run_if_detected=True
        ),
        actions=DetectorActions(
            mode=DetectionMode.PASSIVE
        )
    )

    copilot.detector_registry.add(custom_detector)
    print(f"  [+] Added: {custom_detector.name}")
    print()

    # List all detectors
    print("5. Detector registry:")
    for detector in copilot.detector_registry.list_all():
        status = "enabled" if detector.enabled else "disabled"
        mode = detector.actions.mode.value
        print(f"  - {detector.name}: {status}, mode={mode}")
    print()

    # Get stats
    stats = copilot.detector_registry.get_stats()
    print(f"6. Registry stats:")
    print(f"  Total detectors: {stats['total_detectors']}")
    print(f"  Enabled: {stats['enabled_detectors']}")
    print(f"  Disabled: {stats['disabled_detectors']}")
    print()

    # Simulate volume acquisition (without actual API calls for demo)
    print("7. Simulating volume acquisition...")
    print("  (Note: Would need ANTHROPIC_API_KEY for actual detection)")
    print()

    # Show how detection results would be stored
    embryo = copilot.experiment.embryos['embryo_001']
    print(f"8. Embryo state structure:")
    print(f"  - ID: {embryo.id}")
    print(f"  - Detection results: {list(embryo.detection_results.keys())}")
    print(f"  - Recent images: {len(embryo.recent_images)}")
    print()

    # Show how to check detections
    print("9. Checking detection status:")
    for detector in copilot.detector_registry.list_all():
        detected = embryo.was_detected(detector.name)
        latest = embryo.get_latest_detection(detector.name)
        print(f"  - {detector.name}: detected={detected}, latest={latest}")
    print()

    # Test enable/disable
    print("10. Testing enable/disable:")
    copilot.detector_registry.disable('test_detector')
    print(f"  Disabled test_detector")
    print(f"  Enabled detectors: {len(copilot.detector_registry.list_enabled())}")
    copilot.detector_registry.enable('test_detector')
    print(f"  Re-enabled test_detector")
    print(f"  Enabled detectors: {len(copilot.detector_registry.list_enabled())}")
    print()

    # Save registry
    print("11. Saving detector registry...")
    copilot.detector_registry.save()
    print(f"  [OK] Saved to {copilot.detector_registry.storage_path}")
    print()

    print("=" * 70)
    print("Detector System Test Complete!")
    print("=" * 70)
    print()
    print("Summary:")
    print(f"  - Created {stats['total_detectors']} detectors (3 presets + 1 custom)")
    print(f"  - All detectors enabled and ready")
    print(f"  - Registry persisted to disk")
    print()
    print("Next steps:")
    print("  1. Set ANTHROPIC_API_KEY to test actual detection")
    print("  2. Acquire volumes and call copilot.on_volume_acquired()")
    print("  3. Check embryo.detection_results for results")
    print("  4. See DETECTOR_SYSTEM.md for full documentation")


if __name__ == "__main__":
    asyncio.run(main())
