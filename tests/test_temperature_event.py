"""
Test suite for TEMPERATURE_UPDATE event type
"""

from gently.core.event_bus import EventBus, EventType


def test_temperature_update_event_exists():
    """Verify TEMPERATURE_UPDATE enum member exists with correct name"""
    assert hasattr(EventType, "TEMPERATURE_UPDATE")
    event_type = EventType.TEMPERATURE_UPDATE
    assert event_type.name == "TEMPERATURE_UPDATE"


def test_temperature_update_publishes_to_subscriber():
    """Verify publishing TEMPERATURE_UPDATE reaches subscribers"""
    bus = EventBus()
    seen = []
    bus.subscribe(EventType.TEMPERATURE_UPDATE, lambda e: seen.append(e.data))
    bus.publish(event_type=EventType.TEMPERATURE_UPDATE, data={"x": 1}, source="t")
    assert seen == [{"x": 1}]
