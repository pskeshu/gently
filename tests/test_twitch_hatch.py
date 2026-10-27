from gently.tools.twitch_hatch import SULSTON_TWITCH_TO_HATCH_MIN, report


def test_minutes_and_temperature_from_frame_clocks():
    frames = [
        {"field": 0, "frame": 1, "captured_at": "2026-10-27T20:00:00"},
        {"field": 0, "frame": 2, "captured_at": "2026-10-27T20:05:00"},  # interval changed
        {"field": 0, "frame": 3, "captured_at": "2026-10-27T20:15:00"},
        {"field": 1, "frame": 3, "captured_at": "2026-10-27T20:16:00"},
    ]
    temps = [
        {"t": "2026-10-27T19:59:00", "water_c": 30.0},  # before: ignored
        {"t": "2026-10-27T20:03:00", "water_c": 20.0},
        {"t": "2026-10-27T20:10:00", "water_c": 21.0},
    ]
    rows = [
        {"field": 0, "x": 1, "y": 2, "twitch_frame": 1, "hatch_frame": 3},
        {"field": 1, "x": 1, "y": 2, "twitch_frame": 1, "hatch_frame": 3},  # no frame 1 in field 1
        {"field": 0, "x": 1, "y": 2, "twitch_frame": 2, "hatch_frame": None},
    ]
    out = report(rows, frames, temps)
    assert out[0]["minutes"] == 15.0
    assert out[0]["temp_c"] == 20.5
    assert out[0]["vs_sulston_min"] == 15.0 - SULSTON_TWITCH_TO_HATCH_MIN
    assert out[1]["minutes"] is None
    assert out[2]["minutes"] is None and out[2]["temp_c"] is None
