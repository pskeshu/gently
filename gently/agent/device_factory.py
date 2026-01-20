"""
Device Factory for MicroscopyCopilot

Helper functions to create Ophyd devices from Micro-Manager core for hardware control.
"""

from typing import Dict, Optional
from pathlib import Path
import pymmcore

from rich.console import Console

from .theme import get_theme

# Module-level console for styled output
_console = Console()


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

    theme = get_theme()

    try:
        # Scanner (for direct control)
        scanner = DiSPIMScanner(
            name=cfg['scanner_name'],
            core=core
        )
        devices['scanner'] = scanner
        _console.print(f"  [{theme.success}]{theme.icon_success}[/] Created [bold]scanner[/]: {cfg['scanner_name']}")
    except Exception as e:
        _console.print(f"  [{theme.warning}]{theme.icon_warning}[/] Could not create scanner: {e}")

    try:
        # Piezo (for direct control)
        piezo = DiSPIMPiezo(
            name=cfg['piezo_name'],
            core=core
        )
        devices['piezo'] = piezo
        _console.print(f"  [{theme.success}]{theme.icon_success}[/] Created [bold]piezo[/]: {cfg['piezo_name']}")
    except Exception as e:
        _console.print(f"  [{theme.warning}]{theme.icon_warning}[/] Could not create piezo: {e}")

    try:
        # Camera
        from gently.devices import DiSPIMCamera
        camera = DiSPIMCamera(
            device_name=cfg['camera_name'],
            core=core
        )
        devices['camera'] = camera
        _console.print(f"  [{theme.success}]{theme.icon_success}[/] Created [bold]camera[/]: {cfg['camera_name']}")
    except Exception as e:
        _console.print(f"  [{theme.warning}]{theme.icon_warning}[/] Could not create camera: {e}")

    try:
        # Laser Control
        from gently.devices import DiSPIMLaserControl
        laser_control = DiSPIMLaserControl(
            core=core,
            name='laser_control',
            group_name="Laser"
        )
        devices['laser_control'] = laser_control
        _console.print(f"  [{theme.success}]{theme.icon_success}[/] Created [bold]laser control[/]")
    except Exception as e:
        _console.print(f"  [{theme.warning}]{theme.icon_warning}[/] Could not create laser control: {e}")

    try:
        # XY Stage
        devices['xy_stage'] = DiSPIMXYStage(
            name=cfg['xy_stage_name'],
            core=core
        )
        _console.print(f"  [{theme.success}]{theme.icon_success}[/] Created [bold]XY stage[/]: {cfg['xy_stage_name']}")
    except Exception as e:
        _console.print(f"  [{theme.warning}]{theme.icon_warning}[/] Could not create XY stage: {e}")

    try:
        # LED (optional)
        if cfg.get('led_name'):
            from gently.devices import DiSPIMLED
            led = DiSPIMLED(
                core=core,
                name=cfg['led_name'],
                group_name="LED"
            )
            devices['led'] = led
            _console.print(f"  [{theme.success}]{theme.icon_success}[/] Created [bold]LED[/]: {cfg['led_name']}")
    except Exception as e:
        _console.print(f"  [{theme.warning}]{theme.icon_warning}[/] Could not create LED: {e}")

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
            _console.print(f"  [{theme.success}]{theme.icon_success}[/] Created [bold]volume scanner[/]")
        else:
            _console.print(f"  [{theme.warning}]{theme.icon_warning}[/] Skipping volume scanner (missing required devices)")
    except Exception as e:
        _console.print(f"  [{theme.warning}]{theme.icon_warning}[/] Could not create volume scanner: {e}")

    try:
        # Bottom Camera (with LED control)
        if led:
            bottom_camera = DiSPIMBottomCamera(
                device_name=cfg['bottom_camera_name'],
                core=core,
                led_control=led,
                pixel_size_um=6.5,
                magnification=10.0  # 6.5µm / 10x = 0.65µm effective
            )
            devices['bottom_camera'] = bottom_camera
            _console.print(f"  [{theme.success}]{theme.icon_success}[/] Created [bold]bottom camera[/]: {cfg['bottom_camera_name']}")
        else:
            _console.print(f"  [{theme.warning}]{theme.icon_warning}[/] Skipping bottom camera (missing LED device)")
    except Exception as e:
        _console.print(f"  [{theme.warning}]{theme.icon_warning}[/] Could not create bottom camera: {e}")

    try:
        # Light Sheet Snap (requires scanner and camera)
        if scanner and camera:
            devices['lightsheet_snap'] = DiSPIMLightSheetSnap(
                scanner=scanner,
                camera=camera,
                name='lightsheet_snap'
            )
            _console.print(f"  [{theme.success}]{theme.icon_success}[/] Created [bold]lightsheet snap[/] device")
        else:
            _console.print(f"  [{theme.warning}]{theme.icon_warning}[/] Skipping lightsheet snap (missing required devices)")
    except Exception as e:
        _console.print(f"  [{theme.warning}]{theme.icon_warning}[/] Could not create lightsheet snap: {e}")

    if not devices:
        raise RuntimeError("Failed to create any devices. Check your Micro-Manager configuration.")

    return devices


