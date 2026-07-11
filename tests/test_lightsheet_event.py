"""Tests for LIGHTSHEET_FRAME event type"""

from gently.core.event_bus import _NO_HISTORY_TYPES, EventBus, EventType


def test_lightsheet_frame_event_exists():
    """LIGHTSHEET_FRAME event type should exist with correct name"""
    assert EventType.LIGHTSHEET_FRAME.name == "LIGHTSHEET_FRAME"


def test_lightsheet_frame_excluded_from_history():
    """LIGHTSHEET_FRAME should be excluded from event history (high-volume telemetry)"""
    assert EventType.LIGHTSHEET_FRAME in _NO_HISTORY_TYPES


def test_lightsheet_frame_publishes_to_subscriber():
    """Publishing LIGHTSHEET_FRAME should reach subscribers"""
    bus = EventBus()
    seen = []
    bus.subscribe(EventType.LIGHTSHEET_FRAME, lambda e: seen.append(e.data))
    bus.publish(event_type=EventType.LIGHTSHEET_FRAME, data={"jpeg_b64": "x"}, source="t")
    assert seen == [{"jpeg_b64": "x"}]
