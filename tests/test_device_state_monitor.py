"""Tests for DeviceStateMonitor — focus on the staleness watchdog.

Exercises:
  * normal pass-through publish on each event
  * watchdog fires reconnect when the stream stalls beyond the threshold
  * watchdog does NOT fire while events are flowing
  * stop cleanly cancels both reader and watchdog
"""

from __future__ import annotations

import asyncio

from gently.app.device_state_monitor import DeviceStateMonitor
from gently.core.event_bus import EventBus, EventType

# ---------------------------------------------------------------------------
# Fake microscope: yields events from an async generator that the test
# can advance, pause, or replace. Mirrors the shape DiSPIMMicroscope's
# stream_device_states() exposes (async iterator of payload dicts).
# ---------------------------------------------------------------------------


class _FakeMicroscope:
    """Replaceable stream source. ``connect_count`` tracks reconnects."""

    def __init__(self):
        # List of (delay_before_yield_sec, payload) tuples. Each call to
        # stream_device_states() returns an async iterator over a fresh
        # copy of this list.
        self.script = []
        self.connect_count = 0
        # If set, this duration is awaited before the FIRST yield each
        # connect, simulating a stalled stream (no events for X seconds).
        self.first_yield_delay = 0.0

    def stream_device_states(self):
        self.connect_count += 1
        script_copy = list(self.script)
        first_delay = self.first_yield_delay

        async def _gen():
            if first_delay > 0:
                await asyncio.sleep(first_delay)
            for delay, payload in script_copy:
                if delay > 0:
                    await asyncio.sleep(delay)
                yield payload
            # After the script ends, hang forever — simulating an SSE
            # stream that's connected but has no more data. The watchdog
            # is the only way out.
            while True:
                await asyncio.sleep(60)

        return _gen()


def _run(coro):
    """Run an async coroutine in a fresh event loop. Plain pytest works."""
    return asyncio.run(coro)


def _patch_bus(local_bus):
    """Context-manager style: swap module-level get_event_bus for a local one."""
    from gently.app import device_state_monitor as mod

    original = mod.get_event_bus
    mod.get_event_bus = lambda: local_bus

    def restore():
        mod.get_event_bus = original

    return restore


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_monitor_publishes_each_event():
    """Every payload from the stream becomes a DEVICE_STATE_UPDATE."""
    bus_seen = []
    bus = EventBus()
    bus.subscribe(EventType.DEVICE_STATE_UPDATE, lambda ev: bus_seen.append(ev))
    restore = _patch_bus(bus)

    fake = _FakeMicroscope()
    fake.script = [(0.0, {"i": 0}), (0.05, {"i": 1}), (0.05, {"i": 2})]

    async def go():
        mon = DeviceStateMonitor(
            microscope=fake,
            reconnect_delay_sec=0.1,
            stale_timeout_sec=10.0,
            watchdog_interval_sec=10.0,
        )
        await mon.on_start()
        await asyncio.sleep(0.3)
        await mon.on_stop()

    try:
        _run(go())
    finally:
        restore()

    payloads = [ev.data for ev in bus_seen]
    assert payloads == [{"i": 0}, {"i": 1}, {"i": 2}]


def test_watchdog_does_not_trip_while_events_flowing():
    """A steady stream of events keeps last_event_at fresh — no kicks."""
    bus = EventBus()
    restore = _patch_bus(bus)

    fake = _FakeMicroscope()
    fake.script = [(0.1, {"i": i}) for i in range(8)]

    result = {}

    async def go():
        mon = DeviceStateMonitor(
            microscope=fake,
            reconnect_delay_sec=0.1,
            stale_timeout_sec=1.0,
            watchdog_interval_sec=0.1,
        )
        await mon.on_start()
        await asyncio.sleep(1.0)
        result["kicks"] = mon._watchdog_kicks
        result["connects"] = fake.connect_count
        await mon.on_stop()

    try:
        _run(go())
    finally:
        restore()

    assert result["kicks"] == 0, (
        f"watchdog should not trip with healthy stream (kicks={result['kicks']})"
    )
    assert result["connects"] == 1, f"no spurious reconnects (connects={result['connects']})"


def test_watchdog_kicks_stalled_stream():
    """If the stream goes silent past the threshold the watchdog reconnects."""
    bus = EventBus()
    restore = _patch_bus(bus)

    fake = _FakeMicroscope()
    # Two events fast, then the generator's terminal sleep — stream stalls.
    fake.script = [(0.0, {"i": "first"}), (0.05, {"i": "second"})]

    result = {}

    async def go():
        mon = DeviceStateMonitor(
            microscope=fake,
            reconnect_delay_sec=0.05,
            stale_timeout_sec=0.4,
            watchdog_interval_sec=0.05,
        )
        await mon.on_start()
        await asyncio.sleep(1.5)
        result["kicks"] = mon._watchdog_kicks
        result["connects"] = fake.connect_count
        await mon.on_stop()

    try:
        _run(go())
    finally:
        restore()

    assert result["kicks"] >= 1, (
        f"watchdog should have fired at least once (kicks={result['kicks']})"
    )
    assert result["connects"] >= 2, (
        f"a kick should have triggered at least one reconnect (connects={result['connects']})"
    )


def test_watchdog_recovers_after_kick():
    """After a watchdog kick, the next connection re-publishes events."""
    bus_seen = []
    bus = EventBus()
    bus.subscribe(EventType.DEVICE_STATE_UPDATE, lambda ev: bus_seen.append(ev))
    restore = _patch_bus(bus)

    fake = _FakeMicroscope()
    # First generator stalls. Second generator yields a real event.
    state = {"swap_done": False}

    def patched_stream():
        if fake.connect_count == 0 and not state["swap_done"]:
            # First connect: hang forever
            fake.connect_count += 1

            async def _gen():
                while True:
                    await asyncio.sleep(60)

            return _gen()
        # Subsequent connects: yield a marker event then hang
        fake.connect_count += 1
        state["swap_done"] = True

        async def _gen2():
            yield {"i": "post-kick"}
            while True:
                await asyncio.sleep(60)

        return _gen2()

    fake.stream_device_states = patched_stream

    async def go():
        mon = DeviceStateMonitor(
            microscope=fake,
            reconnect_delay_sec=0.05,
            stale_timeout_sec=0.3,
            watchdog_interval_sec=0.05,
        )
        await mon.on_start()
        await asyncio.sleep(1.5)
        await mon.on_stop()

    try:
        _run(go())
    finally:
        restore()

    payloads = [ev.data for ev in bus_seen]
    assert {"i": "post-kick"} in payloads, (
        f"reconnect after kick should deliver a fresh event (got: {payloads})"
    )


def test_external_cancel_does_not_resurrect_reader():
    """A cancel NOT tagged by the watchdog must exit the reader, not loop.

    Regression: during agent shutdown, asyncio cancels every task. The
    reader's except-CancelledError used to swallow every cancel that
    arrived while _stop_requested was False — so during a graceful
    shutdown (where on_stop hadn't yet been called for the monitor) the
    reader would reconnect indefinitely, blocking process exit.
    """
    bus = EventBus()
    restore = _patch_bus(bus)
    fake = _FakeMicroscope()
    # Hang the iterator so the reader is firmly inside the async-for
    # await when we externally cancel it.
    fake.first_yield_delay = 60.0

    result = {}

    async def go():
        mon = DeviceStateMonitor(
            microscope=fake,
            reconnect_delay_sec=0.05,
            stale_timeout_sec=10.0,
            watchdog_interval_sec=10.0,
        )
        await mon.on_start()
        await asyncio.sleep(0.1)
        # External cancel (mimics event-loop teardown / Ctrl+C path):
        # neither _stop_requested nor _watchdog_kicked_reader is set.
        mon._task.cancel()
        # Give the reader a moment to propagate the cancel. If the bug
        # is present, the reader swallows the cancel and reconnects;
        # connect_count would climb past 1.
        await asyncio.sleep(0.3)
        result["task_done"] = mon._task.done()
        result["connects"] = fake.connect_count
        await mon.on_stop()

    try:
        _run(go())
    finally:
        restore()

    assert result["task_done"], "reader should exit after external cancel"
    assert result["connects"] == 1, (
        f"reader must NOT reconnect on external cancel (connects={result['connects']})"
    )


def test_stop_cancels_both_tasks_cleanly():
    """on_stop terminates the reader and watchdog without lingering tasks."""
    bus = EventBus()
    restore = _patch_bus(bus)

    fake = _FakeMicroscope()
    fake.first_yield_delay = 60.0  # Hang first iter to exercise cancel paths

    result = {}

    async def go():
        mon = DeviceStateMonitor(
            microscope=fake,
            reconnect_delay_sec=0.05,
            stale_timeout_sec=10.0,
            watchdog_interval_sec=0.05,
        )
        await mon.on_start()
        await asyncio.sleep(0.1)
        await mon.on_stop()
        result["task"] = mon._task
        result["watchdog"] = mon._watchdog_task

    try:
        _run(go())
    finally:
        restore()

    assert result["task"] is None
    assert result["watchdog"] is None
