"""
Tests for LightSheetStreamMonitor.

Mirrors test_bottom_camera_monitor.py structure; uses a FakeScope that yields
three synthetic frames so the test never touches the device layer.
"""

import asyncio

import pytest

from gently.app.lightsheet_monitor import LightSheetStreamMonitor
from gently.core.event_bus import EventType, get_event_bus


class FakeScope:
    async def stream_lightsheet(self):
        for i in range(3):
            yield {"t": float(i), "jpeg_b64": f"f{i}"}
            await asyncio.sleep(0)


@pytest.mark.asyncio
async def test_monitor_publishes_frames():
    bus = get_event_bus()
    seen = []
    bus.subscribe(EventType.LIGHTSHEET_FRAME, lambda e: seen.append(e.data))
    mon = LightSheetStreamMonitor(FakeScope(), reconnect_delay_sec=0.01)
    await mon.start()
    await asyncio.sleep(0.05)
    await mon.stop()
    assert any(d.get("jpeg_b64") == "f0" for d in seen)
    assert mon.running is False
