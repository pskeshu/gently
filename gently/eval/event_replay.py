"""EventReplay — reads a captured events jsonl and republishes events to a
target EventBus.

Two modes:
  fast      events as fast as the bus can dispatch (default)
  real-time inserts sleep delays between events to preserve the original
            cadence — useful when a candidate's behaviour depends on
            time-since-last-event

Original Event timestamps are preserved by going through
EventBus.publish_event() (which keeps the dataclass instance untouched)
rather than EventBus.publish() (which constructs a fresh Event with
datetime.now()). Candidates can therefore reason about historical timing
as if they were live.
"""

from __future__ import annotations

import json
import logging
import time
from collections.abc import Callable, Iterator
from datetime import datetime
from pathlib import Path

from gently.core.event_bus import Event, EventBus

logger = logging.getLogger(__name__)


class EventReplay:
    """Stream-replays an events.jsonl into a target bus."""

    def __init__(self, path: Path):
        self.path = Path(path)
        if not self.path.exists():
            raise FileNotFoundError(f"event log not found: {self.path}")

    def events(self) -> Iterator[Event]:
        """Yield each Event from the captured log, in order.

        Lines that don't parse are skipped with a warning rather than
        aborting the whole replay — a partial log is better than no log.
        """
        with self.path.open("r", encoding="utf-8") as f:
            for line_no, raw in enumerate(f, start=1):
                raw = raw.strip()
                if not raw:
                    continue
                try:
                    record = json.loads(raw)
                except json.JSONDecodeError:
                    logger.warning("EventReplay: malformed line %d in %s", line_no, self.path)
                    continue
                try:
                    yield Event.from_dict(record)
                except KeyError:
                    # Unknown EventType — could be a newer enum the
                    # capturing process knew about. Skip rather than abort.
                    logger.warning("EventReplay: unknown event_type on line %d", line_no)
                except Exception:
                    logger.exception("EventReplay: parse failure on line %d", line_no)

    def replay(
        self,
        target: EventBus,
        *,
        real_time: bool = False,
        time_scale: float = 1.0,
        on_event: Callable[[Event], None] | None = None,
    ) -> int:
        """Replay the captured events to ``target``. Returns count emitted.

        Parameters
        ----------
        target:
            EventBus to publish into. The bus's existing subscribers (and
            any shadow candidates registered on it) will see the events.
        real_time:
            If True, sleep between events to reproduce the original
            cadence. If False, dispatch as fast as the bus can handle.
        time_scale:
            Only meaningful in real-time mode. ``time_scale=4`` runs the
            replay at 4× speed (sleep delays divided by 4). Must be > 0.
        on_event:
            Optional callback invoked after each event is published, for
            instrumentation / progress reporting. Exceptions are caught
            and logged.
        """
        if time_scale <= 0:
            raise ValueError("time_scale must be > 0")

        emitted = 0
        prev_ts: datetime | None = None
        wall_start = time.monotonic()
        for ev in self.events():
            if real_time and prev_ts is not None:
                delta = (ev.timestamp - prev_ts).total_seconds() / time_scale
                if delta > 0:
                    time.sleep(delta)
            target.publish_event(ev)
            emitted += 1
            if on_event is not None:
                try:
                    on_event(ev)
                except Exception:
                    logger.exception("EventReplay: on_event callback failed")
            prev_ts = ev.timestamp
        wall = time.monotonic() - wall_start
        logger.info(
            "EventReplay: emitted %d events in %.2fs (real_time=%s, time_scale=%g)",
            emitted,
            wall,
            real_time,
            time_scale,
        )
        return emitted

    def event_types(self) -> dict:
        """Return a {EventType.name: count} histogram of the log.

        Cheap pre-flight diagnostic before running an expensive replay.
        """
        counts: dict = {}
        for ev in self.events():
            counts[ev.event_type.name] = counts.get(ev.event_type.name, 0) + 1
        return counts
