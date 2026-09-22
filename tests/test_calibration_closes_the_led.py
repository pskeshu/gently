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

The capture moved once since: `observe_at_galvo` is now the single
capture-and-ask, used by the edge sweep and by the pre-calibration object
check. The check takes the first frame of a run, so it is the one the close
has to precede — which is why the ordering assertion below looks for every
call that can produce a frame, not one function name.
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
    # Before any frame, or the first ones are the ruined ones.
    #
    # The capture call itself now lives in `observe_at_galvo`, shared by the
    # edge sweep and the pre-calibration object check, so this function no
    # longer names `capture_lightsheet_image`. What it does name is every way
    # it can reach a frame — and the close has to precede all of them,
    # including the pre-flight probe, which is the FIRST frame of a run now.
    frame_calls = [
        m.start()
        for m in re.finditer(
            r"(observe_at_galvo|probe_for_object|capture_lightsheet_image)\(", body
        )
    ]
    assert frame_calls, (
        "calibrate_embryo no longer takes frames by any route this test knows; "
        "if the capture moved again, teach this test the new name rather than "
        "deleting the ordering check"
    )
    assert body.index("set_led") < min(frame_calls), (
        "the LED is closed after frames have already been captured"
    )


def test_a_failed_close_is_logged_not_fatal() -> None:
    """The cost is a poor fit; refusing to calibrate mid-session is worse."""
    body = _body()
    m = re.search(r"try:\s*\n\s*await client\.set_led\(\"Closed\"\)\s*\n\s*except", body)
    assert m, "the shutter close is unguarded — a timed-out status call aborts calibration"
    assert "logger.warning" in body[m.end() : m.end() + 400]
