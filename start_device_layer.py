#!/usr/bin/env python3
"""
Start the Gently Device Layer

Unified server for all hardware control:
- Direct MMCore initialization (no external Micro-Manager process needed)
- Ophyd device abstraction
- Bluesky RunEngine for plan execution
- SAM embryo detection via HTTP

This replaces the previous architecture requiring 3 processes:
- start_server.py (MMCore RPyC) - ELIMINATED
- backend/simple_server.py (HTTP API) - REPLACED
- backend/sam_server.py (SAM RPyC) - REPLACED

Usage:
    python start_device_layer.py
    python start_device_layer.py --port 60610
    python start_device_layer.py --sam-device cuda
    python start_device_layer.py --sam-device cpu
"""

import asyncio
import argparse
import sys
from pathlib import Path

# Ensure project root is in path
project_root = Path(__file__).parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))


def main():
    parser = argparse.ArgumentParser(
        description="Gently Device Layer - Unified Hardware Server",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python start_device_layer.py
    python start_device_layer.py --port 60610
    python start_device_layer.py --sam-device cpu

The server provides:
    - HTTP API on port 60610 for plan submission
    - Direct MMCore control (no external Micro-Manager needed)
    - SAM embryo detection via /api/detect_embryos
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
        default="config.yml",
        help="Path to config.yml (default: config.yml)"
    )

    args = parser.parse_args()

    print("\n" + "=" * 60)
    print("GENTLY DEVICE LAYER")
    print("=" * 60)
    print(f"\nConfiguration:")
    print(f"  HTTP Port:   {args.port}")
    print(f"  SAM Device:  {args.sam_device}")
    print(f"  Config:      {args.config}")
    print()

    # Import and run server
    from backend.device_layer import DeviceLayerServer

    async def run_server():
        server = DeviceLayerServer(
            config_path=args.config,
            sam_device=args.sam_device
        )
        await server.run(port=args.port)

    try:
        asyncio.run(run_server())
    except KeyboardInterrupt:
        print("\n\nDevice layer stopped.")
        sys.exit(0)


if __name__ == "__main__":
    main()
