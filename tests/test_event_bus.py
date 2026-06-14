"""
Tests for EventBus — publish/subscribe event system.

Tests cover:
- Subscribe and publish basic flow
- Wildcard subscriptions
- Unsubscribe
- Async handler dispatch
- Event history (storage, size limit, filtering)
- Event serialization round-trip (to_dict / from_dict)
- Handler counting
- Multiple handlers on same event type
"""

import asyncio

import pytest

from gently.core.event_bus import Event, EventBus, EventType


@pytest.fixture
def bus():
    """Fresh EventBus for each test."""
    return EventBus(history_size=10)


# =========================================================================
# Subscribe and publish
# =========================================================================


class TestSubscribeAndPublish:
    def test_subscribe_and_publish(self, bus):
        received = []
        bus.subscribe(EventType.IMAGE_ACQUIRED, lambda e: received.append(e))

        bus.publish(EventType.IMAGE_ACQUIRED, {"frame": 1}, source="camera")
        assert len(received) == 1
        assert received[0].event_type == EventType.IMAGE_ACQUIRED
        assert received[0].data["frame"] == 1
        assert received[0].source == "camera"

    def test_wildcard_subscription(self, bus):
        received = []
        bus.subscribe("*", lambda e: received.append(e))

        bus.publish(EventType.IMAGE_ACQUIRED, source="cam")
        bus.publish(EventType.STAGE_MOVED, source="stage")

        assert len(received) == 2
        assert received[0].event_type == EventType.IMAGE_ACQUIRED
        assert received[1].event_type == EventType.STAGE_MOVED

    def test_unsubscribe(self, bus):
        received = []
        unsub = bus.subscribe(EventType.IMAGE_ACQUIRED, lambda e: received.append(e))

        bus.publish(EventType.IMAGE_ACQUIRED)
        assert len(received) == 1

        unsub()
        bus.publish(EventType.IMAGE_ACQUIRED)
        assert len(received) == 1  # no new events after unsub

    def test_multiple_handlers_same_event(self, bus):
        results_a, results_b = [], []
        bus.subscribe(EventType.STAGE_MOVED, lambda e: results_a.append(e))
        bus.subscribe(EventType.STAGE_MOVED, lambda e: results_b.append(e))

        bus.publish(EventType.STAGE_MOVED, source="xy")

        assert len(results_a) == 1
        assert len(results_b) == 1

    def test_no_crosstalk(self, bus):
        received = []
        bus.subscribe(EventType.IMAGE_ACQUIRED, lambda e: received.append(e))

        bus.publish(EventType.STAGE_MOVED)
        assert len(received) == 0


# =========================================================================
# Async handlers
# =========================================================================


class TestAsyncHandlers:
    @pytest.mark.asyncio
    async def test_async_handler(self, bus):
        received = []

        async def handler(event):
            received.append(event)

        loop = asyncio.get_running_loop()
        bus.set_event_loop(loop)
        bus.subscribe_async(EventType.VOLUME_ACQUIRED, handler)

        bus.publish(EventType.VOLUME_ACQUIRED, {"vol": 42}, source="acq")

        # Give the loop a moment to dispatch
        await asyncio.sleep(0.05)
        assert len(received) == 1
        assert received[0].data["vol"] == 42


# =========================================================================
# Event history
# =========================================================================


class TestEventHistory:
    def test_event_history(self, bus):
        bus.publish(EventType.IMAGE_ACQUIRED, {"idx": 0})
        bus.publish(EventType.STAGE_MOVED, {"pos": 100})

        history = bus.get_history()
        assert len(history) == 2
        # Newest first
        assert history[0].event_type == EventType.STAGE_MOVED

    def test_history_size_limit(self, bus):
        # bus fixture has history_size=10
        for i in range(15):
            bus.publish(EventType.IMAGE_ACQUIRED, {"idx": i})

        history = bus.get_history(limit=100)
        assert len(history) == 10  # capped at history_size

    def test_get_history_filtered_by_type(self, bus):
        bus.publish(EventType.IMAGE_ACQUIRED)
        bus.publish(EventType.STAGE_MOVED)
        bus.publish(EventType.IMAGE_ACQUIRED)

        history = bus.get_history(event_type=EventType.IMAGE_ACQUIRED)
        assert len(history) == 2
        assert all(e.event_type == EventType.IMAGE_ACQUIRED for e in history)

    def test_get_history_filtered_by_source(self, bus):
        bus.publish(EventType.IMAGE_ACQUIRED, source="cam_A")
        bus.publish(EventType.IMAGE_ACQUIRED, source="cam_B")

        history = bus.get_history(source="cam_A")
        assert len(history) == 1
        assert history[0].source == "cam_A"

    def test_clear_history(self, bus):
        bus.publish(EventType.IMAGE_ACQUIRED)
        bus.publish(EventType.STAGE_MOVED)
        assert len(bus.get_history()) == 2

        bus.clear_history()
        assert len(bus.get_history()) == 0


# =========================================================================
# Serialization
# =========================================================================


class TestEventSerialization:
    def test_event_to_dict_from_dict(self):
        original = Event(
            event_type=EventType.EMBRYO_DETECTED,
            data={"embryo_id": "e1", "confidence": 0.95},
            source="detector",
            correlation_id="corr-123",
        )

        d = original.to_dict()
        assert d["event_type"] == "EMBRYO_DETECTED"
        assert d["source"] == "detector"
        assert d["correlation_id"] == "corr-123"

        restored = Event.from_dict(d)
        assert restored.event_type == EventType.EMBRYO_DETECTED
        assert restored.data["embryo_id"] == "e1"
        assert restored.source == "detector"
        assert restored.correlation_id == "corr-123"


# =========================================================================
# Handler count
# =========================================================================


class TestHandlerCount:
    def test_handler_count_specific(self, bus):
        bus.subscribe(EventType.IMAGE_ACQUIRED, lambda e: None)
        bus.subscribe(EventType.IMAGE_ACQUIRED, lambda e: None)
        assert bus.get_handler_count(EventType.IMAGE_ACQUIRED) == 2

    def test_handler_count_total(self, bus):
        bus.subscribe(EventType.IMAGE_ACQUIRED, lambda e: None)
        bus.subscribe(EventType.STAGE_MOVED, lambda e: None)
        bus.subscribe("*", lambda e: None)
        assert bus.get_handler_count() == 3
