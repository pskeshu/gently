"""TDD: OperationPlanUpdater — execution events transition plan tactics.

Covers:
- BURST_COMPLETE with tactic_id → state=done, bind values recorded
- TEMP_PROTOCOL_COMPLETED with tactic_id → state=done, bind values recorded
- EMBRYO_CADENCE_CHANGED with tactic_id → bind-only (no state change)
- TRIGGER_FIRED with tactic_id → bind-only + last_fired timestamp
- Event without tactic_id → skip (no transition_tactic call)
- Handler exception doesn't propagate to the caller
- on_stop unsubscribes (no further calls after stop)
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from gently.app.operation_plan_updater import OperationPlanUpdater
from gently.core.event_bus import EventBus, EventType

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class FakeContextStore:
    """Records all transition_tactic calls."""

    def __init__(self):
        self.calls: list[dict] = []

    def transition_tactic(
        self, session_id: str, tactic_id: str, state: str | None = None, **bind
    ) -> bool:
        self.calls.append(
            {"session_id": session_id, "tactic_id": tactic_id, "state": state, **bind}
        )
        return True


def make_event(event_type: EventType, data: dict):
    """Build a minimal Event-like object that _on_event expects."""
    evt = MagicMock()
    evt.event_type = event_type
    evt.data = data
    evt.timestamp = "2026-06-28T00:00:00+00:00"
    return evt


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def bus():
    """A fresh EventBus for each test (avoids cross-test leakage)."""
    return EventBus()


@pytest.fixture
def store():
    return FakeContextStore()


@pytest.fixture
def session_id():
    return "sess_abc"


@pytest.fixture
def updater(store, session_id):
    return OperationPlanUpdater(store, lambda: session_id)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_burst_complete_transitions_tactic_to_done(updater, store, bus, monkeypatch):
    """BURST_COMPLETE fires → tactic transitions to done with bind values."""
    monkeypatch.setattr("gently.app.operation_plan_updater.get_event_bus", lambda: bus)
    await updater.start()

    bus.publish(
        event_type=EventType.BURST_COMPLETE,
        data={
            "tactic_id": "t1",
            "embryo_id": "e1",
            "request_id": "req-001",
            "mp4_path": "/data/burst.mp4",
            "sustained_hz": 2.5,
            "frames_captured": 50,
        },
        source="test",
    )

    assert len(store.calls) == 1
    call = store.calls[0]
    assert call["tactic_id"] == "t1"
    assert call["state"] == "done"
    assert call["request_id"] == "req-001"
    assert call["mp4_path"] == "/data/burst.mp4"
    assert call["sustained_hz"] == 2.5
    assert call["frames_captured"] == 50

    await updater.stop()


@pytest.mark.asyncio
async def test_temp_protocol_completed_transitions_to_done(updater, store, bus, monkeypatch):
    """TEMP_PROTOCOL_COMPLETED fires → tactic done with locked/cancelled/error."""
    monkeypatch.setattr("gently.app.operation_plan_updater.get_event_bus", lambda: bus)
    await updater.start()

    bus.publish(
        event_type=EventType.TEMP_PROTOCOL_COMPLETED,
        data={
            "tactic_id": "t2",
            "embryo_id": "e1",
            "locked": True,
            "cancelled": False,
            "error": None,
        },
        source="test",
    )

    assert len(store.calls) == 1
    call = store.calls[0]
    assert call["tactic_id"] == "t2"
    assert call["state"] == "done"
    assert call["locked"] is True
    assert call["cancelled"] is False

    await updater.stop()


@pytest.mark.asyncio
async def test_embryo_cadence_changed_bind_only(updater, store, bus, monkeypatch):
    """EMBRYO_CADENCE_CHANGED → bind-only (state=None), cadence values bound."""
    monkeypatch.setattr("gently.app.operation_plan_updater.get_event_bus", lambda: bus)
    await updater.start()

    bus.publish(
        event_type=EventType.EMBRYO_CADENCE_CHANGED,
        data={
            "tactic_id": "t3",
            "embryo_id": "e1",
            "old_phase": "normal",
            "new_phase": "dense",
            "old_interval_s": 120,
            "new_interval_s": 30,
            "next_due_at": "2026-06-28T01:00:00+00:00",
        },
        source="test",
    )

    assert len(store.calls) == 1
    call = store.calls[0]
    assert call["tactic_id"] == "t3"
    assert call["state"] is None  # bind-only
    assert call["new_phase"] == "dense"
    assert call["new_interval_s"] == 30

    await updater.stop()


@pytest.mark.asyncio
async def test_trigger_fired_bind_only_with_last_fired(updater, store, bus, monkeypatch):
    """TRIGGER_FIRED → bind-only with last_fired timestamp injected."""
    monkeypatch.setattr("gently.app.operation_plan_updater.get_event_bus", lambda: bus)
    await updater.start()

    bus.publish(
        event_type=EventType.TRIGGER_FIRED,
        data={
            "tactic_id": "t4",
            "embryo_id": "e1",
            "rule_name": "dense_on_signal",
            "rule_kind": "interval",
        },
        source="test",
    )

    assert len(store.calls) == 1
    call = store.calls[0]
    assert call["tactic_id"] == "t4"
    assert call["state"] is None
    assert "last_fired" in call  # timestamp injected by updater

    await updater.stop()


@pytest.mark.asyncio
async def test_event_without_tactic_id_is_skipped(updater, store, bus, monkeypatch):
    """An event with no tactic_id must not call transition_tactic."""
    monkeypatch.setattr("gently.app.operation_plan_updater.get_event_bus", lambda: bus)
    await updater.start()

    bus.publish(
        event_type=EventType.BURST_COMPLETE,
        data={
            "embryo_id": "e1",
            "request_id": "req-999",
            # deliberately no tactic_id
        },
        source="test",
    )

    assert store.calls == []

    await updater.stop()


@pytest.mark.asyncio
async def test_handler_exception_does_not_propagate(updater, bus, monkeypatch):
    """An exception inside _handle must be swallowed — bus caller not affected."""
    monkeypatch.setattr("gently.app.operation_plan_updater.get_event_bus", lambda: bus)

    class BrokenStore:
        def transition_tactic(self, *a, **kw):
            raise RuntimeError("storage exploded")

    broken_updater = OperationPlanUpdater(BrokenStore(), lambda: "sess_xyz")
    await broken_updater.start()

    # Should not raise even though the store raises.
    bus.publish(
        event_type=EventType.BURST_COMPLETE,
        data={"tactic_id": "t1", "request_id": "r1"},
        source="test",
    )

    await broken_updater.stop()


@pytest.mark.asyncio
async def test_no_session_skips_transition(store, bus, monkeypatch):
    """If session_id_getter returns None, no transition_tactic call is made."""
    monkeypatch.setattr("gently.app.operation_plan_updater.get_event_bus", lambda: bus)
    no_session_updater = OperationPlanUpdater(store, lambda: None)
    await no_session_updater.start()

    bus.publish(
        event_type=EventType.BURST_COMPLETE,
        data={"tactic_id": "t1", "request_id": "r1"},
        source="test",
    )

    assert store.calls == []

    await no_session_updater.stop()


@pytest.mark.asyncio
async def test_stop_unsubscribes_from_bus(store, bus, monkeypatch):
    """After stop(), further bus events produce no transition_tactic calls."""
    monkeypatch.setattr("gently.app.operation_plan_updater.get_event_bus", lambda: bus)
    up = OperationPlanUpdater(store, lambda: "sess_z")
    await up.start()
    await up.stop()

    bus.publish(
        event_type=EventType.BURST_COMPLETE,
        data={"tactic_id": "t9", "request_id": "r9"},
        source="test",
    )

    assert store.calls == []
