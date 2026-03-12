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

import asyncio
import argparse
import logging
import signal
import sys
from pathlib import Path

import yaml

# Ensure project root is in path
project_root = Path(__file__).parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))


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
        """
    )
    parser.add_argument(
        "--port",
        type=int,
        default=60610,
        help="HTTP API port (default: 60610)"
    )
    parser.add_argument(
        "--sam-device",
        default="cuda",
        choices=["cuda", "cpu"],
        help="Device for SAM model inference (default: cuda)"
    )
    parser.add_argument(
        "--config",
        default="config/config.yml",
        help="Path to config.yml (default: config/config.yml)"
    )

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
        stream=sys.stderr,
    )

    # Load config to determine hardware type
    config_path = Path(args.config)
    if config_path.exists():
        with open(config_path) as f:
            config = yaml.safe_load(f) or {}
    else:
        config = {}

    hardware_name = config.get("hardware", "dispim")

    print("\n" + "=" * 60)
    print("GENTLY DEVICE LAYER")
    print("=" * 60)
    print(f"\nConfiguration:")
    print(f"  Hardware:    {hardware_name}")
    print(f"  HTTP Port:   {args.port}")
    print(f"  SAM Device:  {args.sam_device}")
    print(f"  Config:      {args.config}")
    print()

    # Load hardware module and create device layer
    from gently.hardware import load_hardware
    hw = load_hardware(hardware_name)

    if not hasattr(hw, 'create_device_layer'):
        print(f"Error: Hardware module '{hardware_name}' does not provide create_device_layer().")
        print("The hardware module must implement this factory function.")
        sys.exit(1)

    server = hw.create_device_layer({
        'config_path': args.config,
        'sam_device': args.sam_device,
    })

    async def run_server():
        # Set up signal handling within the async context
        loop = asyncio.get_running_loop()

        def request_shutdown():
            print("\n\nReceived interrupt signal...")
            if hasattr(server, '_shutdown_event'):
                server._shutdown_event.set()

        # On Windows, use signal.signal; on Unix, use loop.add_signal_handler
        if sys.platform == 'win32':
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
        print("\n\nDevice layer stopped.")


if __name__ == "__main__":
    main()
