"""Session-scoped temperature sampler — polls the device layer, persists, emits.

Modeled on gently/app/device_state_monitor.py. While a session is active it polls
the microscope's temperature at a fixed cadence, appends each reading to the
session's temperature.jsonl, holds the latest reading (for acquisition stamping),
and publishes TEMPERATURE_UPDATE for the live graph. A failed poll is a gap, not a
crash; with no active session the loop idles.
"""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timezone

from gently.core.event_bus import EventType, get_event_bus
from gently.core.service import Service

logger = logging.getLogger(__name__)


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def temperature_stamp(latest: dict | None) -> dict | None:
    """Build a temperature meta block from a latest sample, or None if unavailable."""
    if not latest:
        return None
    return {
        "water_c": latest.get("water_c"),
        "setpoint_c": latest.get("setpoint_c"),
        "state": latest.get("state"),
        "sampled_at": latest.get("t"),
    }


class TemperatureSampler(Service):
    def __init__(self, microscope, store, session_id_getter, interval_sec: float = 1.0):
        super().__init__(name="temperature-sampler", service_type="monitor")
        self._microscope = microscope
        self._store = store
        self._session_id_getter = session_id_getter
        self._interval = interval_sec
        self._task: asyncio.Task | None = None
        self.latest: dict | None = None

    async def on_start(self) -> None:
        self._task = asyncio.create_task(self._run(), name="temperature-sampler-loop")

    async def on_stop(self) -> None:
        if self._task is not None:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None

    async def _run(self) -> None:
        bus = get_event_bus()
        fail_streak = 0
        while True:
            try:
                await self._tick(bus)
                if fail_streak:
                    logger.info(
                        "temperature sampler recovered after %d quiet failure(s)", fail_streak
                    )
                    fail_streak = 0
            except asyncio.CancelledError:
                raise
            except Exception as exc:  # a gap, never a crash
                fail_streak += 1
                if fail_streak == 1:
                    # Log once, then stay quiet (e.g. device layer not connected yet).
                    logger.warning(
                        "temperature sampler paused — %s (retrying quietly until connected)", exc
                    )
            # Back off while failing so we neither spam logs nor hammer a disconnected server.
            await asyncio.sleep(
                self._interval if fail_streak == 0 else min(self._interval * 10, 30.0)
            )

    async def _tick(self, bus) -> None:
        session_id = self._session_id_getter()
        if not session_id:
            self.latest = None
            return
        resp = await self._microscope.get_temperature()
        if not resp or not resp.get("success", True):
            self.latest = None
            return
        water = resp.get("temperature_c")
        if water is None:
            self.latest = None
            return
        sample = {
            "t": _now_iso(),
            "water_c": water,
            "setpoint_c": resp.get("setpoint_c"),
            "state": resp.get("state"),
        }
        self._store.append_temperature_sample(session_id, sample)
        self.latest = sample
        bus.publish(
            event_type=EventType.TEMPERATURE_UPDATE,
            data={"session_id": session_id, "sample": sample},
            source="temperature-sampler",
        )
