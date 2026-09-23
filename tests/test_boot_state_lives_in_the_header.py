"""A boot is news, not furniture.

"Microscope warming up — step 2/5 · Initializing Micro-Manager core ... this
should be a different way of appearing to the user, not something that needs to
be dismissed explicitly, but rather a notification that shows up to notify, and
perhaps highlights the top Connected or Rig down indicator in the header."

Two things were wrong at once. The banner was a bar you had to dismiss — and
because the progress label changes every second while the state stays
`initializing`, it needed a per-state acknowledgement just to make the × stick.
Meanwhile the header chip, which is always on screen and costs nothing, said
"Scope offline" for the whole boot: true, and useless, because it does not say
whether to wait.

So the two swap roles. The notice announces and retires itself; the chip
carries the state, counts the steps, and says "Rig down" when the layer dies.

CI runs no JavaScript, so this is pinned in source.
"""

from __future__ import annotations

import re
from pathlib import Path

JS = Path(__file__).resolve().parents[1] / "gently" / "ui" / "web" / "static" / "js"
BANNER = (JS / "boot-banner.js").read_text(encoding="utf-8")
APP = (JS / "app.js").read_text(encoding="utf-8")
CSS = (
    Path(__file__).resolve().parents[1] / "gently" / "ui" / "web" / "static" / "css" / "main.css"
).read_text(encoding="utf-8")


def _booting_branch() -> str:
    m = re.search(
        r"if \(state === 'starting' \|\| state === 'initializing'\) \{(.*?)\n        \} else if",
        BANNER,
        re.S,
    )
    assert m, "the boot branch is gone"
    return m.group(1)


def test_the_notice_retires_itself_rather_than_asking_to_be_dismissed() -> None:
    body = _booting_branch()
    assert "close: false" in body, (
        "the warming-up notice offers a × again — something that leaves by "
        "itself does not need dismissing"
    )
    assert "setTimeout" in body and "hide()" in body, "the notice never retires on its own"


def test_the_notice_does_not_reappear_once_per_step() -> None:
    """Five reappearances of the same news is nagging."""
    body = _booting_branch()
    assert "_ackedState = state" in body, (
        "the notice is re-shown on every poll, so each step change flashes it back onto the screen"
    )


def test_details_survives_because_it_answers_a_real_question() -> None:
    assert "details: true" in _booting_branch()


def test_the_signal_carries_progress_and_fires_when_the_step_moves() -> None:
    """The chip shows "Warming up 2/5", which is only useful if it counts."""
    assert "state !== _lastState || step !== _lastStep" in BANNER, (
        "the device-layer signal fires only on state change, so the header's "
        "step counter would freeze at whatever it first saw"
    )
    assert re.search(r"progress: \{ i: progress\.i", BANNER), (
        "the signal no longer carries progress, so the header cannot count"
    )


def test_the_chip_shows_the_rig_coming_up_and_going_down() -> None:
    m = re.search(r"function rigChipOverride\(\) \{(.*?)\n\}", APP, re.S)
    assert m, "the header chip no longer knows about the rig"
    body = m.group(1)
    assert "Warming up" in body, "a boot reads as 'Scope offline' again"
    assert "Rig down" in body, "a crashed or failed device layer has no chip state"
    for state in ("starting", "initializing", "failed", "crashed"):
        assert state in body, f"the chip ignores the {state} state"


def test_the_chip_is_subscribed_to_the_rig() -> None:
    assert "ClientEventBus.on('DEVICE_LAYER_STATE'" in APP, "the chip never hears about the rig"
    assert "_rigState = d" in APP


def test_the_boot_state_outranks_scope_offline() -> None:
    """ "Scope offline" during a boot is true and says nothing worth knowing."""
    m = re.search(r"function renderConnectionUI\(s\) \{(.*?)\n\}", APP, re.S)
    assert m, "renderConnectionUI is gone"
    body = m.group(1)
    rig_at = body.index("rigChipOverride()")
    offline_at = body.index("Scope offline")
    assert rig_at < offline_at, "the connection snapshot overwrites the rig state"


def test_the_two_states_are_visible_as_states() -> None:
    assert ".status-dot.booting" in CSS and ".status-dot.down" in CSS, (
        "warming up and rig-down look identical to every other dot state"
    )
    assert "prefers-reduced-motion" in CSS, "the pulsing dot has no reduced-motion escape"
