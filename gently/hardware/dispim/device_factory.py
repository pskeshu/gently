"""
Device Factory for MicroscopyAgent

Helper functions to create Ophyd devices from Micro-Manager core for hardware control.
"""

import logging

import pymmcore

logger = logging.getLogger(__name__)


def create_devices_from_mmcore(core: pymmcore.CMMCore, config: dict | None = None) -> dict:
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
        - fdrive: DiSPIMFDrive (SPIM head Z / F axis)

    Example
    -------
    >>> from client import get_mmc
    >>> from bluesky import RunEngine
    >>> from gently.app.agent import MicroscopyAgent
    >>> from gently.device_factory import create_devices_from_mmcore
    >>>
    >>> core = get_mmc()
    >>> devices = create_devices_from_mmcore(core)
    >>> RE = RunEngine({})
    >>>
    >>> agent = MicroscopyAgent(
    ...     run_engine=RE,
    ...     devices=devices
    ... )
    """
    # Import devices only when needed (avoids ophyd import issues)
    from .devices import (
        DiSPIMBottomCamera,
        DiSPIMLightSheetSnap,
        DiSPIMPiezo,
        DiSPIMScanner,
        DiSPIMVolumeScanner,
        DiSPIMXYStage,
    )

    # Default device configuration (from MMConfig_tracking_screening.cfg)
    default_config = {
        "xy_stage_name": "XYStage:XY:31",
        "camera_name": "HamCam1",
        "scanner_name": "Scanner:AB:33",
        "piezo_name": "PiezoStage:P:34",
        "fdrive_name": "ZStage:V:37",
        "bottom_camera_name": "Bottom PCO",
        "led_name": "LED:X:31",
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
        scanner = DiSPIMScanner(name=cfg["scanner_name"], core=core)
        devices["scanner"] = scanner
        logger.info("Created scanner: %s", cfg["scanner_name"])
    except Exception as e:
        logger.warning("Could not create scanner: %s", e)

    try:
        piezo = DiSPIMPiezo(name=cfg["piezo_name"], core=core)
        devices["piezo"] = piezo
        logger.info("Created piezo: %s", cfg["piezo_name"])
    except Exception as e:
        logger.warning("Could not create piezo: %s", e)

    try:
        from .devices import DiSPIMCamera

        camera = DiSPIMCamera(device_name=cfg["camera_name"], core=core)
        devices["camera"] = camera
        logger.info("Created camera: %s", cfg["camera_name"])
    except Exception as e:
        logger.warning("Could not create camera: %s", e)

    try:
        from .devices import DiSPIMLightSource

        # Single instance, registered under both names: the ophyd name and
        # devices-dict key keep the historical "laser_control" identifier
        # so existing Bluesky plans (which take a `laser_control=...` kwarg)
        # continue to work; "light_source" is the new canonical alias for
        # callers that want the broader concept (power + channel).
        laser_control = DiSPIMLightSource(core=core, name="laser_control", group_name="Laser")
        devices["laser_control"] = laser_control
        devices["light_source"] = laser_control
        logger.info("Created light source (laser_control)")
    except Exception as e:
        logger.warning("Could not create light source: %s", e)

    try:
        devices["xy_stage"] = DiSPIMXYStage(name=cfg["xy_stage_name"], core=core)
        logger.info("Created XY stage: %s", cfg["xy_stage_name"])
    except Exception as e:
        logger.warning("Could not create XY stage: %s", e)

    try:
        from .devices import DiSPIMFDrive

        devices["fdrive"] = DiSPIMFDrive(name=cfg["fdrive_name"], core=core)
        logger.info("Created F-drive (SPIM head): %s", cfg["fdrive_name"])
    except Exception as e:
        logger.warning("Could not create F-drive (SPIM head): %s", e)

    try:
        if cfg.get("led_name"):
            from .devices import DiSPIMLED

            led = DiSPIMLED(core=core, name=cfg["led_name"], group_name="LED")
            devices["led"] = led
            logger.info("Created LED: %s", cfg["led_name"])
    except Exception as e:
        logger.warning("Could not create LED: %s", e)

    # Compound devices
    try:
        if scanner and camera and piezo and laser_control:
            devices["volume_scanner"] = DiSPIMVolumeScanner(
                scanner=scanner,
                camera=camera,
                piezo=piezo,
                laser_control=laser_control,
                core=core,
                name="volume_scanner",
            )
            logger.info("Created volume scanner")
        else:
            logger.warning("Skipping volume scanner (missing required devices)")
    except Exception as e:
        logger.warning("Could not create volume scanner: %s", e)

    try:
        if led:
            bottom_camera = DiSPIMBottomCamera(
                device_name=cfg["bottom_camera_name"],
                core=core,
                led_control=led,
                pixel_size_um=6.5,
                magnification=10.0,
            )
            devices["bottom_camera"] = bottom_camera
            logger.info("Created bottom camera: %s", cfg["bottom_camera_name"])
        else:
            logger.warning("Skipping bottom camera (missing LED device)")
    except Exception as e:
        logger.warning("Could not create bottom camera: %s", e)

    try:
        if scanner and camera:
            devices["lightsheet_snap"] = DiSPIMLightSheetSnap(
                scanner=scanner,
                camera=camera,
                name="lightsheet_snap",
            )
            logger.info("Created lightsheet snap device")
        else:
            logger.warning("Skipping lightsheet snap (missing required devices)")
    except Exception as e:
        logger.warning("Could not create lightsheet snap: %s", e)

    if not devices:
        raise RuntimeError("Failed to create any devices. Check your Micro-Manager configuration.")

    return devices
