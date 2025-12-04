#!/usr/bin/env python3
"""
CV Subagent Service Entry Point

Starts the CV subagent service for C. elegans embryo analysis.

Usage:
    python start_cv_service.py [--host HOST] [--port PORT] [--debug]

Example:
    python start_cv_service.py --port 8100
    python start_cv_service.py --debug
"""

import argparse
import asyncio
import logging
import os
import signal
import sys

# Add package to path if running directly
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from gently.cv_subagent import CVSubagentService
from gently.cv_subagent.config import CVSubagentConfig


def setup_logging(debug: bool = False):
    """Configure logging"""
    level = logging.DEBUG if debug else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Reduce noise from some loggers
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="Start the CV Subagent Service",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python start_cv_service.py
  python start_cv_service.py --port 8100
  python start_cv_service.py --host 0.0.0.0 --port 8100
  python start_cv_service.py --debug

Environment Variables:
  ANTHROPIC_API_KEY     - API key for Claude (required for analysis)
  CV_SUBAGENT_HOST      - Service host (default: localhost)
  CV_SUBAGENT_PORT      - Service port (default: 8100)
  GENTLY_DATA_STORE_URL - URL for data store access
        """,
    )

    parser.add_argument(
        "--host",
        default=os.environ.get("CV_SUBAGENT_HOST", "localhost"),
        help="Host to bind to (default: localhost)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=int(os.environ.get("CV_SUBAGENT_PORT", "8100")),
        help="Port to bind to (default: 8100)",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug logging",
    )
    parser.add_argument(
        "--gpu-device",
        type=int,
        default=0,
        help="GPU device index (default: 0)",
    )

    return parser.parse_args()


async def main():
    """Main entry point"""
    args = parse_args()

    # Setup logging
    setup_logging(debug=args.debug)
    logger = logging.getLogger(__name__)

    # Log startup info
    logger.info("=" * 60)
    logger.info("CV Subagent Service")
    logger.info("=" * 60)
    logger.info(f"Host: {args.host}")
    logger.info(f"Port: {args.port}")
    logger.info(f"GPU Device: {args.gpu_device}")
    logger.info(f"Debug: {args.debug}")

    # Check for API key
    if not os.environ.get("ANTHROPIC_API_KEY"):
        logger.warning("ANTHROPIC_API_KEY not set - Claude Vision features will not work")

    # Create service
    service = CVSubagentService(
        host=args.host,
        port=args.port,
        anthropic_api_key=os.environ.get("ANTHROPIC_API_KEY"),
        data_store_url=os.environ.get("GENTLY_DATA_STORE_URL"),
        gpu_device=args.gpu_device,
    )

    # Setup signal handlers for graceful shutdown
    shutdown_event = asyncio.Event()

    def signal_handler(sig, frame):
        logger.info(f"Received signal {sig}, initiating shutdown...")
        shutdown_event.set()

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    try:
        # Start service
        await service.start()

        logger.info("=" * 60)
        logger.info(f"Service running at http://{args.host}:{args.port}")
        logger.info(f"API docs at http://{args.host}:{args.port}/docs")
        logger.info("Press Ctrl+C to stop")
        logger.info("=" * 60)

        # Wait for shutdown signal
        await shutdown_event.wait()

    except Exception as e:
        logger.error(f"Service error: {e}", exc_info=True)
        raise

    finally:
        # Stop service
        logger.info("Stopping service...")
        await service.stop()
        logger.info("Service stopped")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
