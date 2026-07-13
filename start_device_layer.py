#!/usr/bin/env python3
"""
Start the Gently Device Layer

Launches the hardware control server for the configured hardware module.
The hardware module (e.g., dispim, twophoton) provides its own device layer
implementation via create_device_layer().

Usage:
    python start_device_layer.py
    python start_device_layer.py --port 60610
    python start_device_layer.py --sam-device cuda
    python start_device_layer.py --sam-device cpu
"""

import argparse
import asyncio
import logging
import signal
import sys
from pathlib import Path

import yaml

# Ensure project root is in path
project_root = Path(__file__).parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))


def _render_startup_failure(exc, log_file):
    """Turn a startup exception into a plain-language console panel.

    The full traceback goes to the log file; the operator (often a biologist)
    sees a diagnosis and concrete things to check. Recognises the common
    "hardware powered off / COM port" case from the MMCore error text.
    """
    import re
    import traceback

    from gently.hardware import console_ui as cui

    logging.getLogger("gently.device_layer").error(
        "Startup failed:\n%s", "".join(traceback.format_exception(exc))
    )

    text = str(exc)
    low = text.lower()
    first_line = next((ln.strip() for ln in text.splitlines() if ln.strip()), "Unknown error")

    summary = "The device layer could not start."
    details = first_line
    hints = []

    dev_m = re.search(r'device "([^"]+)"', text)
    device = dev_m.group(1) if dev_m else None

    if device and "initialize" in low:
        # MMCore failed to initialize a hardware device — almost always the
        # instrument is off, unplugged, or its COM port is held by another app.
        summary = "Can't reach the microscope - it looks powered off or disconnected."
        details = f'Device "{device}" failed to initialize.'
        if device.upper().startswith("COM"):
            hints = [
                "Is the microscope / stage controller powered on?",
                "Are the USB / serial cables connected?",
                f"Is another program using {device}? (e.g. Micro-Manager still open)",
            ]
        else:
            hints = [
                f'Check that "{device}" is powered on and connected.',
                "Is another program (e.g. Micro-Manager) holding the hardware?",
            ]
    elif "access is denied" in low or "system error code 5" in low or "already" in low:
        summary = "A hardware port is busy — another program may be holding it."
        hints = [
            "Close Micro-Manager or any other app using the microscope.",
            "Then start the device layer again.",
        ]

    # Structured failure event for the DeviceLayerSupervisor (rides the same
    # stdout pipe as progress) so the UI can show a plain-language reason + hints
    # instead of re-parsing the console panel. No-op on an interactive terminal.
    cui.progress_event(
        status="failed",
        stage=(device or "startup"),
        summary=summary,
        detail=details,
        hints=hints,
    )
    cui.error_panel("GENTLY DEVICE LAYER", summary, details, hints, log_file)


def main():
    parser = argparse.ArgumentParser(
        description="Gently Device Layer - Hardware Server",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python start_device_layer.py
    python start_device_layer.py --port 60610
    python start_device_layer.py --sam-device cpu

The server provides:
    - HTTP API on the specified port for plan submission
    - Hardware control via the configured hardware module
    - SAM embryo detection via /api/detect_embryos (if supported)
        """,
    )
    parser.add_argument("--port", type=int, default=60610, help="HTTP API port (default: 60610)")
    parser.add_argument(
        "--sam-device",
        default="cuda",
        choices=["cuda", "cpu"],
        help="Device for SAM model inference (default: cuda)",
    )
    parser.add_argument(
        "--config",
        default="config/config.yml",
        help="Path to config.yml (default: config/config.yml)",
    )

    args = parser.parse_args()

    from datetime import datetime

    from gently.settings import settings

    log_dir = Path(settings.storage.base_path) / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / f"device_layer_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

    # Windows consoles default to cp1252; force UTF-8 so Unicode in log
    # messages (e.g. '→') doesn't crash the StreamHandler.
    for _stream in (sys.stdout, sys.stderr):
        try:
            _stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, OSError):
            pass

    console = logging.StreamHandler(sys.stderr)
    console.setLevel(logging.WARNING)
    console.setFormatter(logging.Formatter("%(asctime)s %(levelname)s [%(name)s] %(message)s"))

    file_handler = logging.FileHandler(str(log_file), encoding="utf-8")
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(
        logging.Formatter("%(asctime)s %(name)s %(levelname)s %(funcName)s:%(lineno)d %(message)s")
    )

    logging.basicConfig(level=logging.DEBUG, handlers=[console, file_handler])
    logging.getLogger().info("Logging to %s", log_file)

    # Load config to determine hardware type
    config_path = Path(args.config)
    if config_path.exists():
        with open(config_path) as f:
            config = yaml.safe_load(f) or {}
    else:
        config = {}

    hardware_name = config.get("hardware", "dispim")

    from gently.hardware import console_ui as cui

    cui.out()
    cui.header(f"GENTLY{cui.MIDDOT}DEVICE LAYER", badge="starting", badge_style="cyan")
    cui.row("Hardware", str(hardware_name))
    cui.row("HTTP port", str(args.port))
    cui.row("SAM device", str(args.sam_device))
    cui.row("Config", str(args.config))
    cui.row("Log file", str(log_file))
    cui.rule(heavy=True)

    # Load hardware module and create device layer
    from gently.hardware import load_hardware

    hw = load_hardware(hardware_name)

    if not hasattr(hw, "create_device_layer"):
        print(f"Error: Hardware module '{hardware_name}' does not provide create_device_layer().")
        print("The hardware module must implement this factory function.")
        sys.exit(1)

    server = hw.create_device_layer(
        {
            "config_path": args.config,
            "sam_device": args.sam_device,
        }
    )

    async def run_server():
        # Set up signal handling within the async context
        loop = asyncio.get_running_loop()

        def request_shutdown():
            cui.out()
            cui.note("Interrupt received - stopping...", "yellow")
            logging.getLogger("gently.device_layer.signal").warning(
                "Interrupt signal received — initiating shutdown"
            )
            if hasattr(server, "_shutdown_event"):
                server._shutdown_event.set()

        # On Windows, use signal.signal; on Unix, use loop.add_signal_handler
        if sys.platform == "win32":
            # Windows: use signal module with thread-safe callback
            def win_signal_handler(sig, frame):
                loop.call_soon_threadsafe(request_shutdown)

            signal.signal(signal.SIGINT, win_signal_handler)
            signal.signal(signal.SIGTERM, win_signal_handler)
        else:
            # Unix: use asyncio's native signal handling
            loop.add_signal_handler(signal.SIGINT, request_shutdown)
            loop.add_signal_handler(signal.SIGTERM, request_shutdown)

        await server.run(port=args.port)

    try:
        asyncio.run(run_server())
    except KeyboardInterrupt:
        cui.note("Device layer stopped.", "grey")
    except Exception as exc:
        # Hardware/init failure — show a diagnosis, not a raw traceback.
        _render_startup_failure(exc, log_file)
        sys.exit(1)


if __name__ == "__main__":
    main()
