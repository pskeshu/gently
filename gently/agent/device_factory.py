"""
Device Factory for MicroscopyCopilot

Helper functions to create Ophyd devices from Micro-Manager core for hardware control.
"""

from typing import Dict, Optional
from pathlib import Path
import pymmcore


def create_devices_from_mmcore(core: pymmcore.CMMCore,
                                config: Optional[Dict] = None) -> Dict:
    """
    Create all necessary Ophyd devices from Micro-Manager core

    Parameters
    ----------
    core : pymmcore.CMMCore
        Micro-Manager core instance
    config : dict, optional
        Device configuration with device names. If None, uses defaults.

    Returns
    -------
    dict
        Dictionary of device instances with keys:
        - xy_stage: DiSPIMXYStage
        - volume_scanner: DiSPIMVolumeScanner
        - bottom_camera: DiSPIMBottomCamera
        - lightsheet_snap: DiSPIMLightSheetSnap
        - scanner: DiSPIMScanner
        - piezo: DiSPIMPiezo

    Example
    -------
    >>> from client import get_mmc
    >>> from bluesky import RunEngine
    >>> from gently.agent import MicroscopyCopilot
    >>> from gently.agent.device_factory import create_devices_from_mmcore
    >>>
    >>> core = get_mmc()
    >>> devices = create_devices_from_mmcore(core)
    >>> RE = RunEngine({})
    >>>
    >>> copilot = MicroscopyCopilot(
    ...     run_engine=RE,
    ...     devices=devices
    ... )
    """
    # Import devices only when needed (avoids ophyd import issues)
    from gently.devices import (
        DiSPIMXYStage,
        DiSPIMVolumeScanner,
        DiSPIMBottomCamera,
        DiSPIMLightSheetSnap,
        DiSPIMScanner,
        DiSPIMPiezo
    )

    # Default device configuration (from MMConfig_tracking_screening.cfg)
    default_config = {
        'xy_stage_name': 'XYStage:XY:31',
        'camera_name': 'HamCam1',
        'scanner_name': 'Scanner:AB:33',
        'piezo_name': 'PiezoStage:P:34',
        'bottom_camera_name': 'Bottom PCO',
        'led_name': 'LED:X:31'
    }

    # Merge with user config
    if config:
        default_config.update(config)
    cfg = default_config

    devices = {}

    # Create individual devices first
    scanner = None
    camera = None
    piezo = None
    laser_control = None
    led = None

    try:
        # Scanner (for direct control)
        scanner = DiSPIMScanner(
            name=cfg['scanner_name'],
            core=core
        )
        devices['scanner'] = scanner
        print(f"  ✓ Created scanner: {cfg['scanner_name']}")
    except Exception as e:
        print(f"  ⚠ Could not create scanner: {e}")

    try:
        # Piezo (for direct control)
        piezo = DiSPIMPiezo(
            name=cfg['piezo_name'],
            core=core
        )
        devices['piezo'] = piezo
        print(f"  ✓ Created piezo: {cfg['piezo_name']}")
    except Exception as e:
        print(f"  ⚠ Could not create piezo: {e}")

    try:
        # Camera
        from gently.devices import DiSPIMCamera
        camera = DiSPIMCamera(
            name=cfg['camera_name'],
            core=core
        )
        devices['camera'] = camera
        print(f"  ✓ Created camera: {cfg['camera_name']}")
    except Exception as e:
        print(f"  ⚠ Could not create camera: {e}")

    try:
        # Laser Control
        from gently.devices import DiSPIMLaserControl
        laser_control = DiSPIMLaserControl(
            config_group_name="Laser",
            core=core,
            name='laser_control'
        )
        devices['laser_control'] = laser_control
        print(f"  ✓ Created laser control")
    except Exception as e:
        print(f"  ⚠ Could not create laser control: {e}")

    try:
        # XY Stage
        devices['xy_stage'] = DiSPIMXYStage(
            name=cfg['xy_stage_name'],
            core=core
        )
        print(f"  ✓ Created XY stage: {cfg['xy_stage_name']}")
    except Exception as e:
        print(f"  ⚠ Could not create XY stage: {e}")

    try:
        # LED (optional)
        if cfg.get('led_name'):
            from gently.devices import DiSPIMLED
            led = DiSPIMLED(
                name=cfg['led_name'],
                core=core
            )
            devices['led'] = led
            print(f"  ✓ Created LED: {cfg['led_name']}")
    except Exception as e:
        print(f"  ⚠ Could not create LED: {e}")

    # Now create compound devices
    try:
        # Volume Scanner (requires scanner, camera, piezo, laser_control)
        if scanner and camera and piezo and laser_control:
            devices['volume_scanner'] = DiSPIMVolumeScanner(
                scanner=scanner,
                camera=camera,
                piezo=piezo,
                laser_control=laser_control,
                core=core,
                name='volume_scanner'
            )
            print(f"  ✓ Created volume scanner")
        else:
            print(f"  ⚠ Skipping volume scanner (missing required devices)")
    except Exception as e:
        print(f"  ⚠ Could not create volume scanner: {e}")

    try:
        # Bottom Camera (with LED control)
        bottom_camera = DiSPIMBottomCamera(
            name=cfg['bottom_camera_name'],
            core=core,
            led_device=led,
            effective_pixel_size=0.65  # 6.5µm / 10x for 4x objective
        )
        devices['bottom_camera'] = bottom_camera
        print(f"  ✓ Created bottom camera: {cfg['bottom_camera_name']}")
    except Exception as e:
        print(f"  ⚠ Could not create bottom camera: {e}")

    try:
        # Light Sheet Snap (requires scanner and camera)
        if scanner and camera:
            devices['lightsheet_snap'] = DiSPIMLightSheetSnap(
                scanner=scanner,
                camera=camera,
                core=core,
                name='lightsheet_snap'
            )
            print(f"  ✓ Created lightsheet snap device")
        else:
            print(f"  ⚠ Skipping lightsheet snap (missing required devices)")
    except Exception as e:
        print(f"  ⚠ Could not create lightsheet snap: {e}")

    if not devices:
        raise RuntimeError("Failed to create any devices. Check your Micro-Manager configuration.")

    return devices


def create_copilot_with_hardware(storage_path: Path,
                                  core: Optional[pymmcore.CMMCore] = None,
                                  device_config: Optional[Dict] = None):
    """
    Convenience function to create MicroscopyCopilot with full hardware control

    Parameters
    ----------
    storage_path : Path
        Where to store experiment data
    core : pymmcore.CMMCore, optional
        Micro-Manager core. If None, will try to import from client.get_mmc()
    device_config : dict, optional
        Device configuration to pass to create_devices_from_mmcore()

    Returns
    -------
    MicroscopyCopilot
        Copilot with RunEngine and devices initialized

    Example
    -------
    >>> from pathlib import Path
    >>> from gently.agent.device_factory import create_copilot_with_hardware
    >>>
    >>> copilot = create_copilot_with_hardware(
    ...     storage_path=Path("./experiment_data")
    ... )
    >>>
    >>> # Now you can use hardware control tools
    >>> # e.g., "calibrate embryo_001", "acquire volume for embryo_002"
    """
    from bluesky import RunEngine
    from gently.agent import MicroscopyCopilot

    # Get or create core
    if core is None:
        try:
            from client import get_mmc
            core = get_mmc()
        except ImportError:
            raise ImportError("Could not import get_mmc from client. Please provide core explicitly.")

    # Create devices
    print("\nCreating devices from Micro-Manager core...")
    devices = create_devices_from_mmcore(core, device_config)
    print(f"✓ Created {len(devices)} devices\n")

    # Create RunEngine
    RE = RunEngine({})
    print("✓ Created RunEngine\n")

    # Create copilot
    copilot = MicroscopyCopilot(
        storage_path=storage_path,
        run_engine=RE,
        devices=devices
    )
    print("✓ Created MicroscopyCopilot with hardware control enabled\n")

    return copilot
