"""EventCapture — wildcard-subscribe to an EventBus and append every event
to a per-session jsonl file.

The captured file is the substrate for replay and shadow-mode testing of
candidate orchestrators. High-volume telemetry types (DEVICE_STATE_UPDATE,
BOTTOM_CAMERA_FRAME) are filtered out by default — a 12-hour timelapse
would otherwise produce ~250 MB of polling noise and drown the meaningful
events (perception completions, operator actions, errors, plan boundaries).
Replay can reconstruct world state from the meaningful events plus the
state-snapshot model; it doesn't need the raw telemetry frames.

File format: one JSON object per line, mirroring Event.to_dict():
  {
    "event_type": "EMBRYOS_UPDATE",
    "data": {...},
    "source": "agent.experiment",
    "timestamp": "2026-05-15T15:32:55.123456",
    "event_id": "abc12345",
    "correlation_id": null
  }
"""

from __future__ import annotations

import json
import logging
import threading
from dataclasses import asdict, is_dataclass
from datetime import date, datetime
from enum import Enum
from pathlib import Path
from typing import Any

from gently.core.event_bus import _NO_HISTORY_TYPES, Event, EventBus, EventType

logger = logging.getLogger(__name__)


class EventCapture:
    """Append-only jsonl sink for an EventBus.

    Lifecycle:
      capture = EventCapture(path)
      capture.start(bus)   # opens file, subscribes
      ...
      capture.stop()       # unsubscribes, closes file

    Thread-safe — bus dispatch can come from any thread; writes are
    serialised through a lock.
    """

    # By default the same set of high-volume telemetry types the EventBus
    # itself skips for its history deque. The rationale carries over: at
    # 5 Hz over hours these would dominate the log without adding signal
    # that replay / diff can use.
    DEFAULT_SKIP: frozenset[EventType] = frozenset(_NO_HISTORY_TYPES)

    def __init__(self, path: Path, *, skip: set[EventType] | None = None):
        self.path = Path(path)
        self._skip = self.DEFAULT_SKIP if skip is None else frozenset(skip)
        self._fp: Any = None
        self._unsub: Any = None
        self._lock = threading.Lock()
        self._count = 0
        self._skipped = 0

    def start(self, bus: EventBus) -> None:
        """Open the capture file and subscribe to the bus (idempotent)."""
        if self._fp is not None:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._fp = self.path.open("a", encoding="utf-8")
        # Sync subscription on purpose — capture is fast (single file write)
        # and we want capture order to match dispatch order without async
        # scheduling ambiguity.
        self._unsub = bus.subscribe("*", self._on_event)
        logger.info("EventCapture: writing to %s", self.path)

    def stop(self) -> None:
        """Unsubscribe and close the file (idempotent)."""
        if self._unsub is not None:
            try:
                self._unsub()
            except Exception:
                logger.exception("EventCapture: unsubscribe failed")
            self._unsub = None
        with self._lock:
            if self._fp is not None:
                try:
                    self._fp.close()
                except Exception:
                    logger.exception("EventCapture: file close failed")
                self._fp = None
        logger.info("EventCapture: closed (%d captured, %d skipped)", self._count, self._skipped)

    def __del__(self):
        # Best-effort safety net for cases where the owner forgets to call
        # stop() — never let a forgotten file handle outlive the process'
        # capture object. We can't rely on this for correctness (GC timing
        # is undefined), but it makes tests and dev sessions tidier.
        try:
            self.stop()
        except Exception:
            pass

    @property
    def count(self) -> int:
        return self._count

    def _on_event(self, event: Event) -> None:
        if event.event_type in self._skip:
            self._skipped += 1
            return
        try:
            line = json.dumps(event.to_dict(), default=_json_default)
        except Exception:
            logger.exception("EventCapture: failed to serialise %s", event)
            return
        with self._lock:
            if self._fp is None:
                return
            try:
                self._fp.write(line + "\n")
                self._fp.flush()
                self._count += 1
            except Exception:
                logger.exception("EventCapture: write failed for %s", event)


def _json_default(obj):
    """Last-resort serialiser for types json.dumps can't natively handle.

    Designed to be lossy-but-useful: numpy arrays become lists, datetimes
    become ISO strings, dataclasses become dicts, anything else falls back
    to repr() so the line is at least valid JSON.
    """
    if isinstance(obj, (datetime, date)):
        return obj.isoformat()
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, Enum):
        return obj.name
    if is_dataclass(obj):
        try:
            return asdict(obj)
        except Exception:
            pass
    try:
        import numpy as np

        if isinstance(obj, np.generic):
            return obj.item()
        if isinstance(obj, np.ndarray):
            return obj.tolist()
    except ImportError:
        pass
    if isinstance(obj, set):
        return sorted(obj, key=str)
    if isinstance(obj, bytes):
        return obj.decode("utf-8", errors="replace")
    return repr(obj)
