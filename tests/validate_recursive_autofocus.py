#!/usr/bin/env python3
"""
Validate Recursive Autofocus Implementation
==========================================

Quick validation test for the recursive autofocus functionality using mock devices.
Tests the plan structure and parameter handling without requiring real hardware.
"""

import sys
import logging
from unittest.mock import Mock, MagicMock
import numpy as np

# Mock pymmcore for testing without hardware
class MockPyMMCore:
    class CMMCore:
        def __init__(self):
            self.current_position = 150.0  # Start at middle of new range

        def getPosition(self, device):
            return self.current_position

        def setPosition(self, device, position):
            # Simulate device movement
            self.current_position = position
            print(f"Mock {device} moved to {position:.2f} μm")

        def waitForDevice(self, device):
            pass

        def getImage(self):
            # Return mock image data
            return np.random.randint(0, 1000, (100, 100), dtype=np.uint16)

# Install mock
if 'pymmcore' not in sys.modules:
    sys.modules['pymmcore'] = MockPyMMCore()

# Import after mocking
from gently.devices import DiSPIMCamera, DiSPIMZstage
from gently.plans import recursive_focus_single_round
from bluesky import RunEngine
import bluesky.plan_stubs as bps


def test_device_limits():
    """Test that devices have correct limits and rounding"""
    print("Testing Device Limits and Rounding")
    print("=" * 40)

    mock_core = MockPyMMCore.CMMCore()

    # Test Z-stage limits
    z_stage = DiSPIMZstage("TestZStage", mock_core, name="test_z")
    print(f"Z-stage limits: {z_stage.limits}")

    # Test position rounding (this would be tested by trying problematic float values)
    test_positions = [160.85999999999999, 100.333333333333333, 75.0000000000001]

    for pos in test_positions:
        try:
            status = z_stage.set(pos)
            print(f"✓ Position {pos} handled successfully")
        except ValueError as e:
            print(f"✗ Position {pos} failed: {e}")

    print()


def test_recursive_autofocus_plan():
    """Test the recursive autofocus plan structure"""
    print("Testing Recursive Autofocus Plan")
    print("=" * 40)

    mock_core = MockPyMMCore.CMMCore()

    # Create mock devices
    z_stage = DiSPIMZstage("TestZStage", mock_core, name="test_z")
    camera = DiSPIMCamera("TestCamera", mock_core, name="test_camera")

    # Mock the camera's trigger method to avoid actual image acquisition
    camera.trigger = Mock(return_value=Mock(done=True))

    # Create RunEngine for testing
    RE = RunEngine()

    # Test parameters
    coarse_range = 50.0
    fine_range = 10.0
    coarse_steps = 5  # Fewer steps for quick test
    fine_steps = 7

    print(f"Testing with:")
    print(f"  Coarse range: ±{coarse_range/2} μm ({coarse_steps} steps)")
    print(f"  Fine range: ±{fine_range/2} μm ({fine_steps} steps)")
    print(f"  Starting position: {mock_core.current_position} μm")
    print()

    try:
        # Run the recursive autofocus plan
        plan = recursive_focus_single_round(
            z_stage, camera,
            coarse_range=coarse_range,
            fine_range=fine_range,
            coarse_steps=coarse_steps,
            fine_steps=fine_steps
        )

        # Execute plan
        result = RE(plan)

        print(f"✓ Recursive autofocus plan executed successfully")
        print(f"Final position: {mock_core.current_position:.2f} μm")

        # Verify position is within limits
        if z_stage.limits[0] <= mock_core.current_position <= z_stage.limits[1]:
            print(f"✓ Final position within limits {z_stage.limits}")
        else:
            print(f"✗ Final position outside limits {z_stage.limits}")

    except Exception as e:
        print(f"✗ Recursive autofocus plan failed: {e}")
        import traceback
        traceback.print_exc()

    print()


def test_position_bounds():
    """Test position bounds checking"""
    print("Testing Position Bounds")
    print("=" * 40)

    mock_core = MockPyMMCore.CMMCore()
    z_stage = DiSPIMZstage("TestZStage", mock_core, name="test_z")

    # Test positions at various points
    test_cases = [
        (25.0, "Below lower limit", False),
        (50.0, "At lower limit", True),
        (150.0, "Middle of range", True),
        (250.0, "At upper limit", True),
        (275.0, "Above upper limit", False),
    ]

    for pos, description, should_pass in test_cases:
        try:
            status = z_stage.set(pos)
            if should_pass:
                print(f"✓ {description} ({pos}): Accepted")
            else:
                print(f"✗ {description} ({pos}): Should have been rejected")
        except ValueError:
            if not should_pass:
                print(f"✓ {description} ({pos}): Correctly rejected")
            else:
                print(f"✗ {description} ({pos}): Should have been accepted")

    print()


def main():
    """Run all validation tests"""
    print("Recursive Autofocus Validation")
    print("=" * 50)
    print()

    # Reduce log noise
    logging.basicConfig(level=logging.WARNING)

    # Run tests
    test_device_limits()
    test_position_bounds()
    test_recursive_autofocus_plan()

    print("=" * 50)
    print("Validation Complete!")
    print()
    print("Usage Instructions:")
    print("=" * 20)
    print("1. Simple focus test:")
    print("   python test_embryo_focus.py")
    print()
    print("2. Recursive autofocus test:")
    print("   python test_embryo_focus.py recursive")
    print()
    print("The recursive autofocus performs:")
    print("  - Coarse scan (±50μm default)")
    print("  - Analysis to find best coarse position")
    print("  - Fine scan around best position (±10μm default)")
    print("  - Final move to optimal position from curve fit")


if __name__ == "__main__":
    main()