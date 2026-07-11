"""Logging configuration for the Gently system.

Reads its own GENTLY_LOG_* env vars directly — no dependency on settings.py
so logging can be bootstrapped before the rest of the app loads.

Env vars:
    GENTLY_LOG_LEVEL   — DEBUG, INFO (default), WARNING, ERROR
    GENTLY_LOG_FILE    — path to log file (disabled by default)
    GENTLY_LOG_FORMAT  — console format string
    GENTLY_LOG_DATEFMT — timestamp format (default: %H:%M:%S)
"""

import logging
import os
import sys

_DEFAULT_FORMAT = "%(asctime)s %(name)s %(levelname)s %(message)s"
_DEFAULT_FILE_FORMAT = "%(asctime)s %(name)s %(levelname)s %(funcName)s:%(lineno)d %(message)s"
_DEFAULT_DATEFMT = "%H:%M:%S"


def configure_logging(
    level: str | None = None,
    log_file: str | None = None,
):
    """Configure root logger for the Gently system.

    Call once at startup (launch_gently.py or gently.py).
    Explicit arguments take priority over env vars.
    """
    # Windows consoles default to cp1252, which crashes when log messages
    # contain Unicode (e.g. the arrow '→' used in plan-item titles). Force
    # the standard streams to UTF-8 with replacement so logging never raises.
    for _stream in (sys.stdout, sys.stderr):
        try:
            _stream.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[union-attr]
        except (AttributeError, OSError):
            pass

    level = level or os.environ.get("GENTLY_LOG_LEVEL", "INFO")
    log_file = log_file or os.environ.get("GENTLY_LOG_FILE") or None
    console_fmt = os.environ.get("GENTLY_LOG_FORMAT", _DEFAULT_FORMAT)
    datefmt = os.environ.get("GENTLY_LOG_DATEFMT", _DEFAULT_DATEFMT)

    log_level = getattr(logging, level.upper(), logging.INFO)

    # Configure both gently and gently_perception loggers
    for logger_name in ("gently", "gently_perception"):
        lgr = logging.getLogger(logger_name)
        lgr.setLevel(logging.DEBUG)

        console = logging.StreamHandler(sys.stderr)
        console.setLevel(log_level)
        console.setFormatter(logging.Formatter(console_fmt, datefmt=datefmt))
        lgr.addHandler(console)

    # Suppress noisy third-party loggers on console
    for name in (
        "uvicorn",
        "uvicorn.error",
        "uvicorn.access",
        "httpx",
        "httpcore",
        "anthropic",
        "aiohttp",
        "aiohttp.access",
        "bluesky",
        "bluesky.RE.state",
    ):
        logging.getLogger(name).setLevel(logging.WARNING)

    # The `websockets` library logs a full "data transfer failed" traceback at
    # ERROR level every time a client disconnects ungracefully (e.g. a browser
    # tab sleeping or dropping — Windows raises WinError 121, "semaphore timeout").
    # These are routine, not faults, and flood the console hundreds of lines deep,
    # burying real errors. Suppress below CRITICAL so a genuinely fatal WS fault
    # still surfaces. WARNING is not enough here because the noise is ERROR-level.
    for name in ("websockets", "websockets.server", "websockets.client"):
        logging.getLogger(name).setLevel(logging.CRITICAL)

    # File handler — always INFO+ regardless of console level
    if log_file:
        file_fmt = os.environ.get("GENTLY_LOG_FILE_FORMAT", _DEFAULT_FILE_FORMAT)
        fh = logging.FileHandler(log_file, encoding="utf-8")
        fh.setLevel(logging.INFO)
        fh.setFormatter(logging.Formatter(file_fmt))
        for logger_name in ("gently", "gently_perception"):
            logging.getLogger(logger_name).addHandler(fh)
