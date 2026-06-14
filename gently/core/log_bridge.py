"""Bridge Python logging into the EventBus so the Events page mirrors the
console.

A small ``LogToBusHandler`` subclasses ``logging.Handler``. Every record it
sees gets published as ``EventType.LOG_RECORD`` with a compact payload the
frontend can render. The handler attaches itself to whichever loggers
``configure_log_bridge`` is told to cover — by default only ``gently`` and
``gently_perception``, which keeps third-party noise (aiohttp access logs,
bluesky state transitions, anthropic SDK chatter) off the page unless the
operator opts in.

Env-configurable:
    GENTLY_LOG_BUS                — "on" / "off" (default: on)
    GENTLY_LOG_BUS_LEVEL          — DEBUG / INFO (default) / WARNING / ERROR
    GENTLY_LOG_BUS_INCLUDE_THIRDPARTY — "1"/"true" to include common third-
                                       party loggers (uvicorn, aiohttp,
                                       bluesky, anthropic, httpx, httpcore)

Re-entrancy is the only real subtlety: if a log call happens inside the
EventBus.publish path (e.g. from the dispatch loop's logger), republishing
it as another LOG_RECORD would loop forever. Guarded with a thread-local
re-entry flag.
"""

from __future__ import annotations

import logging
import os
import threading
from collections.abc import Iterable, Sequence

from .event_bus import EventBus, EventType, get_event_bus

logger = logging.getLogger(__name__)


# Loggers we never want on the Events page — they emit at the wrong layer
# (their own log lines describe bus dispatch / events page WebSocket frames)
# so republishing them would create feedback or infinite churn.
_NEVER_BRIDGE = frozenset(
    {
        "gently.core.event_bus",
        "gently.core.log_bridge",
    }
)

# Loggers that count as "third-party noise" — silenced by default, can be
# opted in with GENTLY_LOG_BUS_INCLUDE_THIRDPARTY=1.
_THIRDPARTY_DEFAULTS: Sequence[str] = (
    "uvicorn",
    "uvicorn.error",
    "uvicorn.access",
    "aiohttp",
    "aiohttp.access",
    "anthropic",
    "httpx",
    "httpcore",
    "bluesky",
    "bluesky.RE.state",
)


class LogToBusHandler(logging.Handler):
    """Publishes each record onto the EventBus as a LOG_RECORD event.

    Per-thread re-entry guard prevents infinite loops when something in
    the publish path itself logs.
    """

    def __init__(self, bus: EventBus, *, level: int = logging.INFO):
        super().__init__(level=level)
        self._bus = bus
        self._reentry = threading.local()

    def emit(self, record: logging.LogRecord) -> None:
        # Re-entry guard: if a downstream subscriber's handler logs, we
        # must not republish that log line.
        if getattr(self._reentry, "active", False):
            return
        # Never bridge our own machinery — those records describe the
        # bridge itself, would loop.
        if record.name in _NEVER_BRIDGE:
            return
        self._reentry.active = True
        try:
            try:
                # format() runs all configured formatters (incl. exc_info
                # serialisation). We send the formatted message + the
                # structured bits separately so the frontend can choose
                # how to render.
                message = record.getMessage()
            except Exception:
                message = "<log format error>"

            payload = {
                "level": int(record.levelno),
                "level_name": record.levelname,
                "logger": record.name,
                "message": message,
                "module": record.module,
                "func": record.funcName,
                "line": record.lineno,
                # Wall-clock ms since epoch — frontend uses this for its
                # own ordering / display, separate from the EventBus's
                # internal timestamp.
                "ts_ms": int(record.created * 1000),
            }
            if record.exc_info:
                try:
                    payload["exc_text"] = logging.Formatter().formatException(record.exc_info)
                except Exception:
                    pass

            self._bus.publish(
                event_type=EventType.LOG_RECORD,
                data=payload,
                source=f"log:{record.name}",
            )
        except Exception:
            # If we can't publish (shutdown, etc.), drop the record
            # silently — the live console + on-disk log still have it.
            pass
        finally:
            self._reentry.active = False


def configure_log_bridge(
    bus: EventBus | None = None,
    *,
    loggers: Iterable[str] | None = None,
    level: str | None = None,
    include_thirdparty: bool | None = None,
) -> LogToBusHandler | None:
    """Attach a LogToBusHandler to the requested loggers.

    Returns the installed handler (or None if the bridge is disabled).
    Idempotent: safe to call multiple times — only the first call attaches.

    Parameters honour env-var defaults so the launch script doesn't need
    to know the knobs:
      GENTLY_LOG_BUS                — "off" disables entirely
      GENTLY_LOG_BUS_LEVEL          — threshold (default INFO)
      GENTLY_LOG_BUS_INCLUDE_THIRDPARTY — adds aiohttp/uvicorn/bluesky/etc.
    """
    if os.environ.get("GENTLY_LOG_BUS", "on").lower() in ("off", "0", "false", "no"):
        return None

    if bus is None:
        bus = get_event_bus()

    if level is None:
        level = os.environ.get("GENTLY_LOG_BUS_LEVEL", "INFO")
    level_int = getattr(logging, level.upper(), logging.INFO)

    if include_thirdparty is None:
        env_val = os.environ.get("GENTLY_LOG_BUS_INCLUDE_THIRDPARTY", "")
        include_thirdparty = env_val.lower() in ("1", "true", "yes", "on")

    if loggers is None:
        loggers = ["gently", "gently_perception"]
        if include_thirdparty:
            loggers = list(loggers) + list(_THIRDPARTY_DEFAULTS)

    handler = LogToBusHandler(bus, level=level_int)

    attached = []
    for name in loggers:
        target = logging.getLogger(name)
        # Skip if already attached (idempotency for re-invocation).
        if any(isinstance(h, LogToBusHandler) for h in target.handlers):
            continue
        target.addHandler(handler)
        attached.append(name)

    if attached:
        # Surface the configuration once at startup — using our own logger
        # (which is in _NEVER_BRIDGE) so this announcement itself doesn't
        # become a LOG_RECORD event.
        logger.info(
            "Log bridge active: level=%s, loggers=%s, include_thirdparty=%s",
            logging.getLevelName(level_int),
            attached,
            include_thirdparty,
        )
    return handler
