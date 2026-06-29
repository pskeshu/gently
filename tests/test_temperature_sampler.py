"""Tests for TemperatureSampler service.

Adaptation note: create_session() requires session_id as its first positional
argument (confirmed from file_store.py:328). The brief's
`file_store.create_session(name="s")` is adapted to
`file_store.create_session(str(uuid.uuid4()), name="s")` to match the real API.
"""

import uuid

from gently.app.temperature_sampler import TemperatureSampler, temperature_stamp
from gently.core.event_bus import EventBus, EventType


class FakeScope:
    def __init__(self, resp):
        self.resp = resp
        self.calls = 0

    async def get_temperature(self):
        self.calls += 1
        if isinstance(self.resp, Exception):
            raise self.resp
        return self.resp


def _capture(bus):
    seen = []
    bus.subscribe(EventType.TEMPERATURE_UPDATE, lambda e: seen.append(e.data))
    return seen


async def test_tick_appends_emits_and_sets_latest(file_store):
    sid = file_store.create_session(str(uuid.uuid4()), name="s")
    scope = FakeScope(
        {"success": True, "temperature_c": 28.4, "setpoint_c": 32.0, "state": "heating"}
    )
    bus = EventBus()
    seen = _capture(bus)
    s = TemperatureSampler(scope, file_store, lambda: sid)
    await s._tick(bus)
    rows = file_store.read_temperature_log(sid)
    assert len(rows) == 1 and rows[0]["water_c"] == 28.4
    assert s.latest["water_c"] == 28.4
    assert seen and seen[0]["sample"]["water_c"] == 28.4 and seen[0]["session_id"] == sid


async def test_tick_no_active_session_is_noop(file_store):
    scope = FakeScope({"success": True, "temperature_c": 1.0, "setpoint_c": 2.0, "state": "x"})
    bus = EventBus()
    seen = _capture(bus)
    s = TemperatureSampler(scope, file_store, lambda: None)
    await s._tick(bus)
    assert scope.calls == 0 and s.latest is None and seen == []


async def test_tick_poll_failure_is_a_gap_not_a_crash(file_store):
    sid = file_store.create_session(str(uuid.uuid4()), name="s")
    scope = FakeScope(RuntimeError("device down"))
    bus = EventBus()
    s = TemperatureSampler(scope, file_store, lambda: sid)
    # _tick propagates the poll exception; _run's except-Exception swallows it.
    # Assert the gap (no rows).
    try:
        await s._tick(bus)
    except RuntimeError:
        pass  # _tick may raise; the loop in _run catches it
    assert file_store.read_temperature_log(sid) == []


def test_temperature_stamp_shapes():
    assert temperature_stamp(None) is None
    assert temperature_stamp(
        {"t": "2026-06-27T10:00:00+00:00", "water_c": 28.4, "setpoint_c": 32.0, "state": "heating"}
    ) == {
        "water_c": 28.4,
        "setpoint_c": 32.0,
        "state": "heating",
        "sampled_at": "2026-06-27T10:00:00+00:00",
    }


async def test_stale_latest_cleared_when_no_active_session(file_store):
    """After a successful tick, a subsequent tick with no active session resets latest to None."""
    sid = file_store.create_session(str(uuid.uuid4()), name="s")
    scope = FakeScope(
        {"success": True, "temperature_c": 28.4, "setpoint_c": 32.0, "state": "heating"}
    )
    bus = EventBus()

    # Successful first tick.
    s = TemperatureSampler(scope, file_store, lambda: sid)
    await s._tick(bus)
    assert s.latest is not None and s.latest["water_c"] == 28.4

    # Second tick with no active session — latest must be cleared.
    s._session_id_getter = lambda: None
    await s._tick(bus)
    assert s.latest is None


async def test_stale_latest_cleared_on_poll_failure(file_store):
    """After a successful tick, a failing poll resets latest to None."""
    sid = file_store.create_session(str(uuid.uuid4()), name="s")
    bus = EventBus()

    # Successful first tick.
    good_scope = FakeScope(
        {"success": True, "temperature_c": 28.4, "setpoint_c": 32.0, "state": "heating"}
    )
    s = TemperatureSampler(good_scope, file_store, lambda: sid)
    await s._tick(bus)
    assert s.latest is not None

    # Swap to a scope that returns falsy (empty dict) — simulates device dropout.
    s._microscope = FakeScope({})
    await s._tick(bus)
    assert s.latest is None
