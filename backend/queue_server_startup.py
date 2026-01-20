"""
Queue Server Startup Script for Gently DiSPIM

This script initializes the RE Worker namespace for bluesky-queueserver.
It is loaded via: start-re-manager --startup-script=backend/queue_server_startup.py

The script exports:
- All Ophyd devices to the namespace
- All Bluesky plans to the namespace

Usage:
    start-re-manager --startup-script=backend/queue_server_startup.py \
        --databroker-config=dispim_production

Note: MMCore must be accessible (Micro-Manager running)
"""

import os
import sys
from pathlib import Path

# Ensure gently package is importable
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# Check if running in RE Worker environment
try:
    from bluesky_queueserver import is_re_worker_active
    IN_RE_WORKER = is_re_worker_active()
except ImportError:
    IN_RE_WORKER = False


if IN_RE_WORKER:
    print("=" * 60)
    print("GENTLY DiSPIM - Queue Server Startup")
    print("=" * 60)

    # =========================================================================
    # CONFIGURATION
    # =========================================================================
    import yaml

    config_path = project_root / "config.yml"
    print(f"\n[1/4] Loading configuration from {config_path}")

    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    mm_dir = config['mmdirectory']
    mm_config = config['mmconfig']
    print(f"  MM Directory: {mm_dir}")
    print(f"  MM Config: {mm_config}")

    # =========================================================================
    # MMCORE INITIALIZATION
    # =========================================================================
    import pymmcore

    print("\n[2/4] Initializing Micro-Manager...")

    core = pymmcore.CMMCore()
    core.enableStderrLog(True)

    # Setup MM environment
    os.environ["PATH"] += os.pathsep.join(["", mm_dir])
    core.setDeviceAdapterSearchPaths([mm_dir])

    # Load configuration
    if not os.path.exists(mm_config):
        raise FileNotFoundError(f"Configuration file not found: {mm_config}")

    core.loadSystemConfiguration(mm_config)
    print(f"  MMCore loaded successfully")

    # =========================================================================
    # OPHYD DEVICES
    # =========================================================================
    print("\n[3/4] Creating Ophyd devices...")

    from gently.agent.device_factory import create_devices_from_mmcore
    devices = create_devices_from_mmcore(core)

    # Export devices to namespace (queueserver discovers these as global variables)
    xy_stage = devices.get('xy_stage')
    volume_scanner = devices.get('volume_scanner')
    bottom_camera = devices.get('bottom_camera')
    lightsheet_snap = devices.get('lightsheet_snap')
    scanner = devices.get('scanner')
    piezo = devices.get('piezo')
    camera = devices.get('camera')
    laser_control = devices.get('laser_control')
    led = devices.get('led')

    print(f"  Exported {len(devices)} devices to namespace:")
    for name in devices.keys():
        print(f"    - {name}")

    # =========================================================================
    # BLUESKY PLANS
    # =========================================================================
    print("\n[4/4] Importing Bluesky plans...")

    # Main acquisition and calibration plans
    try:
        from gently.plans import (
            calibrate_piezo_galvo_plan,
            acquire_single_volume_plan,
            timelapse_volume_plan,
        )
        print("  Imported main plans: calibrate_piezo_galvo_plan, acquire_single_volume_plan, timelapse_volume_plan")
    except ImportError as e:
        print(f"  Warning: Could not import main plans: {e}")

    # Queue server utility plans
    try:
        from gently.plans_qserver import (
            move_stage_plan,
            read_stage_plan,
            read_piezo_plan,
            capture_bottom_image_plan,
            capture_lightsheet_image_plan,
        )
        print("  Imported utility plans: move_stage_plan, read_stage_plan, read_piezo_plan, capture_bottom_image_plan, capture_lightsheet_image_plan")
    except ImportError as e:
        print(f"  Warning: Could not import utility plans: {e}")

    # Calibration plans (if available)
    try:
        from gently.calibration_plans import (
            calibrate_embryo_piezo_galvo,
            verify_embryo_centered,
        )
        print("  Imported calibration plans")
    except ImportError:
        print("  Calibration plans not available (optional)")

    # Multi-embryo plans (if available)
    try:
        from gently.multi_embryo_plans import (
            multi_embryo_calibration_workflow,
            mark_embryo_interactive_plan,
        )
        print("  Imported multi-embryo plans")
    except ImportError:
        print("  Multi-embryo plans not available (optional)")

    print("\n" + "=" * 60)
    print("Queue Server Startup Complete")
    print("=" * 60)
    print("\nAvailable devices:")
    for name in devices.keys():
        print(f"  {name}")
    print("\nReady to receive plans via Queue Server API")
    print("=" * 60 + "\n")

else:
    # Running interactively (not in queueserver)
    print("[QS Startup] Not running in RE Worker environment - skipping initialization")
    print("[QS Startup] This script is meant to be loaded by start-re-manager")
