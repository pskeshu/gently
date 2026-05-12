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
"""

from __future__ import annotations

import asyncio
import logging
from typing import Optional, TYPE_CHECKING

from gently.core.event_bus import EventType, get_event_bus
from gently.core.service import Service

if TYPE_CHECKING:
    from gently.hardware.dispim.client import DiSPIMMicroscope

logger = logging.getLogger(__name__)


class DeviceStateMonitor(Service):
    """Consumes the device-layer SSE stream and republishes events.

    Lifecycle: starts a background asyncio Task that opens the SSE connection,
    iterates events, and publishes ``DEVICE_STATE_UPDATE`` for each. On stream
    drop the task waits ``reconnect_delay_sec`` and reconnects. ``stop()``
    cancels the task and closes the stream.

    The device layer already rate-limits broadcasts (XY at 5 Hz, properties at
    ~0.07 Hz, callback-driven property changes coalesced over 50 ms), so this
    bridge is a pure passthrough — no further throttling.
    """

    def __init__(
        self,
        microscope: "DiSPIMMicroscope",
        reconnect_delay_sec: float = 3.0,
    ):
        super().__init__(name="device-state-monitor", service_type="bridge")
        self.microscope = microscope
        self.reconnect_delay_sec = reconnect_delay_sec
        self._task: Optional[asyncio.Task] = None
        self._stop_requested = False

    async def on_start(self):
        self._stop_requested = False
        self._task = asyncio.create_task(self._run(), name="device-state-monitor")

    async def on_stop(self):
        self._stop_requested = True
        if self._task and not self._task.done():
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        self._task = None

    async def _run(self):
        bus = get_event_bus()
        while not self._stop_requested:
            try:
                logger.info("DeviceStateMonitor: opening device-state stream")
                async for payload in self.microscope.stream_device_states():
                    if self._stop_requested:
                        break
                    try:
                        bus.publish(
                            event_type=EventType.DEVICE_STATE_UPDATE,
                            data=payload,
                            source="device-state-monitor",
                        )
                    except Exception:
                        logger.exception("DeviceStateMonitor: failed to publish event")
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                logger.warning(
                    "DeviceStateMonitor: stream ended (%s) — reconnecting in %.1fs",
                    exc, self.reconnect_delay_sec,
                )
            if self._stop_requested:
                break
            try:
                await asyncio.sleep(self.reconnect_delay_sec)
            except asyncio.CancelledError:
                raise
