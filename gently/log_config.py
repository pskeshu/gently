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
    level: str = None,
    log_file: str = None,
):
    """Configure root logger for the Gently system.

    Call once at startup (launch_gently.py or gently.py).
    Explicit arguments take priority over env vars.
    """
    level = level or os.environ.get("GENTLY_LOG_LEVEL", "INFO")
    log_file = log_file or os.environ.get("GENTLY_LOG_FILE") or None
    console_fmt = os.environ.get("GENTLY_LOG_FORMAT", _DEFAULT_FORMAT)
    datefmt = os.environ.get("GENTLY_LOG_DATEFMT", _DEFAULT_DATEFMT)

    log_level = getattr(logging, level.upper(), logging.INFO)

    root = logging.getLogger("gently")
    root.setLevel(logging.DEBUG)  # Root accepts everything; handlers filter

    # Console handler — uses requested level
    console = logging.StreamHandler(sys.stderr)
    console.setLevel(log_level)
    console.setFormatter(logging.Formatter(console_fmt, datefmt=datefmt))
    root.addHandler(console)

    # Suppress noisy third-party loggers unless verbose/debug
    if log_level > logging.INFO:
        for name in ("uvicorn", "uvicorn.error", "uvicorn.access"):
            logging.getLogger(name).setLevel(logging.ERROR)

    # File handler — always INFO+ regardless of console level
    if log_file:
        file_fmt = os.environ.get("GENTLY_LOG_FILE_FORMAT", _DEFAULT_FILE_FORMAT)
        fh = logging.FileHandler(log_file)
        fh.setLevel(logging.INFO)
        fh.setFormatter(logging.Formatter(file_fmt))
        root.addHandler(fh)
