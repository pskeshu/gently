"""
DeviceStateMonitor — bridges the device-layer SSE stream onto the EventBus.

The device layer (port 60610) polls MMCore and exposes a Server-Sent Events
stream at ``GET /api/devices/stream``. This service consumes that stream and
republishes each tick as a ``DEVICE_STATE_UPDATE`` event on the in-process
EventBus, where the visualization server (and any other listener) picks it up
through its wildcard subscription.

This keeps the streaming transport concerns in one place: the device layer
owns MMCore polling; the viz server owns browser delivery; this bridge is the
glue. If we ever swap SSE for WebSocket or split into a separate process, only
this file changes.

Watchdog
--------
The SSE iterator can silently stall in the agent process whenever a
half-open TCP path or a long synchronous tool call wedges the asyncio loop.
(Historically the worst offender was a Qt window — napari — blocking the
loop during a tool call; that path is gone now that all visualization is
in-browser, but the watchdog stays for general robustness.) aiohttp's
async iterator won't raise on a stalled socket; the ``async for`` just
waits forever. To recover, a sibling watchdog task tracks the timestamp of
the last received event; if no event arrives within ``stale_timeout_sec``
it cancels the reader, which triggers the normal reconnect path. The
generous default timeout (60 s) tolerates legitimate quiet windows during
heavy plan execution; tune via constructor args if needed.
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import TYPE_CHECKING

from gently.core.event_bus import EventType, get_event_bus
from gently.core.service import Service

if TYPE_CHECKING:
    from gently.hardware.dispim.client import DiSPIMMicroscope

logger = logging.getLogger(__name__)


class DeviceStateMonitor(Service):
    """Consumes the device-layer SSE stream and republishes events.

    Lifecycle: starts two background asyncio tasks — a reader that opens the
    SSE connection and republishes events, and a watchdog that forces
    reconnect if the reader goes quiet for ``stale_timeout_sec``. On stream
    drop or staleness-induced cancel the reader loops back through its
    reconnect delay.

    The device layer already rate-limits broadcasts (XY at 5 Hz, properties
    at ~0.07 Hz, callback-driven property changes coalesced over 50 ms), so
    this bridge is a pure passthrough — no further throttling.
    """

    # Default staleness threshold. Conservative because property polls during
    # heavy plan execution can run 2-3 s and we don't want to false-trip
    # on legitimate slowdowns. Position events at 5 Hz dominate the stream
    # so a real stall is obvious well within 60 s.
    DEFAULT_STALE_TIMEOUT_SEC = 60.0

    # Watchdog wakes every this often to check. Doesn't have to be precise;
    # the threshold is what matters for false-trip avoidance.
    DEFAULT_WATCHDOG_INTERVAL_SEC = 5.0

    def __init__(
        self,
        microscope: DiSPIMMicroscope,
        reconnect_delay_sec: float = 3.0,
        stale_timeout_sec: float = DEFAULT_STALE_TIMEOUT_SEC,
        watchdog_interval_sec: float = DEFAULT_WATCHDOG_INTERVAL_SEC,
    ):
        super().__init__(name="device-state-monitor", service_type="bridge")
        self.microscope = microscope
        self.reconnect_delay_sec = reconnect_delay_sec
        self.stale_timeout_sec = stale_timeout_sec
        self.watchdog_interval_sec = watchdog_interval_sec
        self._task: asyncio.Task | None = None
        self._watchdog_task: asyncio.Task | None = None
        self._stop_requested = False
        # Monotonic timestamp of the last successfully-received event. The
        # watchdog reads this; the reader writes it under no lock because
        # asyncio is single-threaded (datetime/float assignment is atomic
        # at the bytecode level on CPython).
        self._last_event_at: float | None = None
        # Counts of staleness-triggered reconnects, useful for diagnostics.
        self._watchdog_kicks: int = 0
        # Set by the watchdog right before it cancels the reader task, so
        # the reader's except-CancelledError block can tell a deliberate
        # staleness kick (reconnect) apart from an external cancellation
        # like Ctrl+C or on_stop (re-raise). Without this discrimination
        # the reader treats every cancel as a kick and reconnects forever
        # during shutdown.
        self._watchdog_kicked_reader = False

    async def on_start(self):
        self._stop_requested = False
        self._last_event_at = time.monotonic()
        self._task = asyncio.create_task(self._run(), name="device-state-monitor")
        self._watchdog_task = asyncio.create_task(
            self._watchdog(),
            name="device-state-watchdog",
        )

    async def on_stop(self):
        self._stop_requested = True
        # Cancel both tasks. Order doesn't matter — both honour the stop flag.
        for t in (self._watchdog_task, self._task):
            if t is not None and not t.done():
                t.cancel()
                try:
                    await t
                except asyncio.CancelledError:
                    pass
                except Exception:
                    logger.exception("DeviceStateMonitor: task cleanup error")
        self._task = None
        self._watchdog_task = None

    async def _run(self):
        bus = get_event_bus()
        while not self._stop_requested:
            try:
                logger.info("DeviceStateMonitor: opening device-state stream")
                async for payload in self.microscope.stream_device_states():
                    if self._stop_requested:
                        break
                    # Bump the staleness timer BEFORE publishing so the
                    # watchdog can't false-trip on a slow publish handler.
                    self._last_event_at = time.monotonic()
                    try:
                        bus.publish(
                            event_type=EventType.DEVICE_STATE_UPDATE,
                            data=payload,
                            source="device-state-monitor",
                        )
                    except Exception:
                        logger.exception("DeviceStateMonitor: failed to publish event")
            except asyncio.CancelledError:
                # Three sources of cancellation must be told apart:
                #   1. on_stop() set _stop_requested → re-raise so the
                #      stop's await self._task completes.
                #   2. Watchdog tagged _watchdog_kicked_reader → swallow,
                #      consume the flag, fall through to the reconnect
                #      delay.
                #   3. External cancel (Ctrl+C, event-loop teardown) →
                #      re-raise so the task actually exits; otherwise
                #      the reader would keep looping during shutdown.
                if self._stop_requested:
                    raise
                if self._watchdog_kicked_reader:
                    self._watchdog_kicked_reader = False
                    logger.warning(
                        "DeviceStateMonitor: reader cancelled by watchdog — reconnecting"
                    )
                else:
                    raise
            except Exception as exc:
                logger.debug(
                    "DeviceStateMonitor: stream ended (%s) — reconnecting in %.1fs",
                    exc,
                    self.reconnect_delay_sec,
                )
            if self._stop_requested:
                break
            try:
                await asyncio.sleep(self.reconnect_delay_sec)
            except asyncio.CancelledError:
                raise
            # Give the fresh connection room to deliver its first event
            # before the watchdog starts counting against it.
            self._last_event_at = time.monotonic()

    async def _watchdog(self):
        """Force-reconnect if the reader goes quiet for too long.

        We cancel the reader task rather than the underlying iterator
        because aiohttp's SSE iterator doesn't expose a clean cancel; the
        task cancel propagates through the awaiting read() and the reader
        loop's except-CancelledError block then re-enters the reconnect
        path.
        """
        while not self._stop_requested:
            try:
                await asyncio.sleep(self.watchdog_interval_sec)
            except asyncio.CancelledError:
                raise
            if self._stop_requested:
                break
            if self._last_event_at is None:
                continue
            age = time.monotonic() - self._last_event_at
            if age <= self.stale_timeout_sec:
                continue
            self._watchdog_kicks += 1
            logger.warning(
                "DeviceStateMonitor: stale stream (%.1fs since last event > "
                "%.1fs threshold) — forcing reconnect (kick #%d)",
                age,
                self.stale_timeout_sec,
                self._watchdog_kicks,
            )
            # Reset the timer FIRST so we don't trigger again before the
            # reader has a chance to reconnect and publish.
            self._last_event_at = time.monotonic()
            # Tag the cancel as ours BEFORE issuing it so the reader's
            # except-CancelledError block knows to swallow rather than
            # propagate. The reader consumes the flag on the way through.
            self._watchdog_kicked_reader = True
            if self._task is not None and not self._task.done():
                self._task.cancel()
