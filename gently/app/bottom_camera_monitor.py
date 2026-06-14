"""
BottomCameraStreamMonitor — bridges the device-layer bottom-camera SSE stream
onto the EventBus as ``BOTTOM_CAMERA_FRAME`` events.

Modelled on :class:`gently.app.device_state_monitor.DeviceStateMonitor`, with
one important difference: the bottom-camera stream is **opt-in**. Subscribing
costs MMCore lock time (the device layer's streamer task spins up on first
connect), so we don't start it on agent boot — only when the operator turns
the camera on from the UI. The agent's start/stop methods are the only path
that connects/disconnects this monitor.
"""

from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING

from gently.core.event_bus import EventType, get_event_bus
from gently.core.service import Service

if TYPE_CHECKING:
    from gently.hardware.dispim.client import DiSPIMMicroscope

logger = logging.getLogger(__name__)


class BottomCameraStreamMonitor(Service):
    """Consumes the bottom-camera SSE stream and republishes frames on the bus.

    The browser receives frames via the viz server's wildcard subscription;
    no additional plumbing is needed at the agent layer beyond starting the
    bridge when the operator asks for it.
    """

    def __init__(
        self,
        microscope: DiSPIMMicroscope,
        reconnect_delay_sec: float = 2.0,
    ):
        super().__init__(name="bottom-camera-monitor", service_type="bridge")
        self.microscope = microscope
        self.reconnect_delay_sec = reconnect_delay_sec
        self._task: asyncio.Task | None = None
        self._stop_requested = False
        self._last_frame_ts: float | None = None

    @property
    def running(self) -> bool:
        return self._task is not None and not self._task.done()

    async def on_start(self):
        if self.running:
            return
        self._stop_requested = False
        self._task = asyncio.create_task(self._run(), name="bottom-camera-monitor")
        logger.info("BottomCameraStreamMonitor: started")

    async def on_stop(self):
        self._stop_requested = True
        if self._task and not self._task.done():
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        self._task = None
        logger.info("BottomCameraStreamMonitor: stopped")

    async def _run(self):
        bus = get_event_bus()
        while not self._stop_requested:
            try:
                logger.info("BottomCameraStreamMonitor: opening stream")
                async for payload in self.microscope.stream_bottom_camera():
                    if self._stop_requested:
                        break
                    self._last_frame_ts = payload.get("t")
                    try:
                        bus.publish(
                            event_type=EventType.BOTTOM_CAMERA_FRAME,
                            data=payload,
                            source="bottom-camera-monitor",
                        )
                    except Exception:
                        logger.exception("Failed to publish frame")
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                logger.warning(
                    "BottomCameraStreamMonitor: stream ended (%s) — reconnecting in %.1fs",
                    exc,
                    self.reconnect_delay_sec,
                )
            if self._stop_requested:
                break
            try:
                await asyncio.sleep(self.reconnect_delay_sec)
            except asyncio.CancelledError:
                raise
