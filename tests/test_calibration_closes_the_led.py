"""Calibration closes the LED before it looks at anything.

#106. Ryan, on the 2026-08-07 walkthrough, watched the physical microscope
while the UI reported the laser was on. Two separate faults wore that one
symptom, and this is the second: `calibrate_embryo` never closed the LED.

A session finds embryos in brightfield — that is what the LED is for. Nothing
closed it afterwards, so every frame the edge detector saw was LED brightfield
with a 50 ms laser gate on top, and Claude was asked to find nuclei in a
DIC-like image. The calibration pane's own method text says the LED alone shows
no nuclei; the code did not act on it.

WHERE IT BELONGS

Both callers route through this tool — `POST /api/devices/embryos/{id}/calibrate`
and the agent's own `calibrate_embryo`. Closing the LED in the route would have
left the agent path broken, and vice versa.

WHY IT IS SAFE HERE

This path does no brightfield work of its own: every frame comes from
`capture_lightsheet_image`, and there is no head-focus phase (`spim_head_focus`
is the plan that legitimately wants the LED open, and it is not on this path).
So the close is unconditional.
"""

from __future__ import annotations

import re
from pathlib import Path

SRC = (
    Path(__file__).resolve().parents[1] / "gently" / "app" / "tools" / "calibration_tools.py"
).read_text(encoding="utf-8")


def _body() -> str:
    start = SRC.index("async def calibrate_embryo(")
    return SRC[start : SRC.index("\nasync def ", start + 10)]


def test_the_led_is_closed_before_the_first_frame() -> None:
    body = _body()
    assert 'client.set_led("Closed")' in body, (
        "calibration no longer closes the LED — edge detection will run on "
        "brightfield if the operator used it to find the embryos (#106)"
    )
    # Before any capture, or the first frames are the ruined ones.
    assert body.index("set_led") < body.index("capture_lightsheet_image"), (
        "the LED is closed after frames have already been captured"
    )


def test_a_failed_close_is_logged_not_fatal() -> None:
    """The cost is a poor fit; refusing to calibrate mid-session is worse."""
    body = _body()
    m = re.search(r"try:\s*\n\s*await client\.set_led\(\"Closed\"\)\s*\n\s*except", body)
    assert m, "the shutter close is unguarded — a timed-out status call aborts calibration"
    assert "logger.warning" in body[m.end() : m.end() + 400]
