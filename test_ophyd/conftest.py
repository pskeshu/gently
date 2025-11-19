"""
pytest configuration and fixtures for DiSPIM ophyd device testing.

This module provides shared fixtures that create device instances and
set up the Bluesky RunEngine with databroker integration.
"""

import sys
from pathlib import Path
import pytest

# Add parent directory to path so we can import gently package
sys.path.insert(0, str(Path(__file__).parent.parent))

from client import get_mmc
from gently.devices import (
    DiSPIMScanner,
    DiSPIMPiezo,
    DiSPIMCamera,
    DiSPIMVolumeScanner,
    DiSPIMLaserControl,
    DiSPIMXYStage,
    DiSPIMZstage,
    DiSPIMBottomCamera,
    DiSPIMLightSheetSnap,
)
from bluesky import RunEngine
from databroker import Broker


@pytest.fixture(scope="session")
def core():
    """
    MMCore connection - shared across all tests in the session.

    Returns:
        pymmcore.CMMCore: Connected Micro-Manager core instance

    Note:
        This fixture has session scope, so the connection is established once
        and reused for all tests. Ensure Micro-Manager server is running.
    """
    print("\n🔌 Connecting to Micro-Manager...")
    mmc = get_mmc()
    print(f"✓ Connected to MM server")
    print(f"  Loaded config: {mmc.getConfigGroupState('System', 'Startup')}")
    return mmc


@pytest.fixture(scope="session")
def run_engine():
    """
    RunEngine with databroker subscription - shared across all tests.

    Returns:
        tuple: (RunEngine, Broker) - RE and databroker instance

    Note:
        Uses 'temp' catalog by default. All test runs are saved to databroker
        and can be retrieved for analysis.
    """
    print("\n⚙️ Creating RunEngine with databroker...")
    RE = RunEngine({})
    db = Broker.named('temp')
    RE.subscribe(db.insert)
    print(f"✓ RunEngine created with databroker catalog: {db.name}")
    return RE, db


# ============================================================================
# Individual Device Fixtures
# ============================================================================

@pytest.fixture
def scanner(core):
    """
    Scanner device for galvo control.

    Returns:
        DiSPIMScanner: Scanner device instance
    """
    return DiSPIMScanner("Scanner:AB:33", core)


@pytest.fixture
def piezo(core):
    """
    Piezo stage device for objective focus control.

    Returns:
        DiSPIMPiezo: Piezo device instance
    """
    return DiSPIMPiezo("PiezoStage:P:34", core)


@pytest.fixture
def camera(core):
    """
    SPIM camera device (Hamamatsu).

    Returns:
        DiSPIMCamera: Camera device instance
    """
    return DiSPIMCamera("HamCam1", core)


@pytest.fixture
def laser_control(core):
    """
    Laser control device (ConfigGroup-based).

    Returns:
        DiSPIMLaserControl: Laser control device instance
    """
    return DiSPIMLaserControl(core, "Laser")


@pytest.fixture
def xy_stage(core):
    """
    XY positioning stage device.

    Returns:
        DiSPIMXYStage: XY stage device instance
    """
    return DiSPIMXYStage("XYStage:XY:31", core)


@pytest.fixture
def z_stage(core):
    """
    Z stage (F-drive) device.

    Returns:
        DiSPIMZstage: Z stage device instance
    """
    return DiSPIMZstage("ZStage:V:37", core)


@pytest.fixture
def bottom_camera(core):
    """
    Bottom camera device with LED management.

    Returns:
        DiSPIMBottomCamera: Bottom camera device instance
    """
    led_name = "LED:X:31"
    return DiSPIMBottomCamera("Bottom PCO", led_name, core)


# ============================================================================
# Compound Device Fixtures
# ============================================================================

@pytest.fixture
def volume_scanner(scanner, camera, piezo, laser_control, core):
    """
    Full volume scanner device for hardware-triggered 3D acquisition.

    This compound device orchestrates scanner, camera, piezo, and laser
    for synchronized volume acquisition.

    Returns:
        DiSPIMVolumeScanner: Volume scanner device instance
    """
    return DiSPIMVolumeScanner(scanner, camera, piezo, laser_control, core)


@pytest.fixture
def lightsheet_snap(scanner, camera):
    """
    Light sheet snapshot device for single slice acquisition.

    Returns:
        DiSPIMLightSheetSnap: Light sheet snap device instance
    """
    return DiSPIMLightSheetSnap(scanner, camera)


# ============================================================================
# Helper Fixtures for Test Data
# ============================================================================

@pytest.fixture
def calibration_file():
    """
    Path to piezo-galvo calibration file.

    Returns:
        str: Path to calibration JSON file
    """
    return "backend/piezo_galvo_calibration_embryo.json"


@pytest.fixture
def output_dir():
    """
    Output directory for test images and data.

    Returns:
        Path: Output directory path
    """
    output_path = Path(__file__).parent / "outputs"
    output_path.mkdir(exist_ok=True)
    return output_path


# ============================================================================
# Test Session Hooks
# ============================================================================

def pytest_configure(config):
    """Called before test run begins."""
    print("\n" + "="*70)
    print("DiSPIM Ophyd Device Test Suite")
    print("="*70)


def pytest_sessionfinish(session, exitstatus):
    """Called after whole test run finished."""
    print("\n" + "="*70)
    print(f"Test session finished with status: {exitstatus}")
    print("="*70 + "\n")


# ============================================================================
# Custom Markers
# ============================================================================

def pytest_configure(config):
    """Register custom markers."""
    config.addinivalue_line(
        "markers", "hardware: mark test as requiring hardware connection"
    )
    config.addinivalue_line(
        "markers", "slow: mark test as slow running"
    )
    config.addinivalue_line(
        "markers", "acquisition: mark test as performing image acquisition"
    )
