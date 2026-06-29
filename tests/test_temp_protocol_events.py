"""Test for temperature protocol event types (Task 1 of temp-change-burst-tactic)"""

from gently.core.event_bus import EventType


def test_new_event_types_exist():
    """Verify the three new EventType members exist"""
    for n in ("TEMPERATURE_SETPOINT_CHANGED", "TEMP_PROTOCOL_STARTED", "TEMP_PROTOCOL_COMPLETED"):
        assert getattr(EventType, n).name == n


def test_timeline_maps_the_subtypes():
    """Verify timeline.py contains the mapping entries with correct subtypes"""
    from gently.harness.session import timeline as tl

    src = open(tl.__file__, encoding="utf-8").read()
    for sub in ("temp_protocol_started", "temp_protocol_completed", "setpoint_changed"):
        assert sub in src
