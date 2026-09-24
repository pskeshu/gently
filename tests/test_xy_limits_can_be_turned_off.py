"""Two fences, and only one of them binds other people.

"in the joystick we have set the limits so the XY is not moved beyond that in
the joystick firmware. that is what we want to enable or disable from gently.
since it affects the larger users, who might use the system from micromanager"

`set_firmware_limits` writes `LowerLimX(mm)` and friends into the ASI Tiger,
which enforces them against EVERY motion source. A region Gently wrote on
Monday still stops the stage short for someone driving it from Micro-Manager
on Friday, with nothing on their screen to connect it to.

The envelope has since come apart into two fences, because they protect
against different things:

* the SOFTWARE fence bounds every move Gently commands — checked in `set()`
  before anything reaches the hardware, costs nothing, affects nobody else;
* the FIRMWARE fence is the same numbers in the controller, whose only
  advantage is stopping a hand on the joystick, and whose cost is binding
  every other client.

"the dispim is controlled by trained professionals who won't accidently move
the joystick" — so the second is opt-in and off by default, and the first
always holds.

What must stay true:

* off means the stage's FULL TRAVEL, because the controller always holds some
  box — there is no "no limits" to write;
* the region still binds Gently when the controller is left open, or turning
  enforcement off would quietly unbound every move Gently makes;
* the operator's region survives being switched off; and
* applying a region does not fence anybody else.
"""

from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1] / "gently"
DEVICE_LAYER = (ROOT / "hardware" / "dispim" / "device_layer.py").read_text(encoding="utf-8")
STAGE = (ROOT / "hardware" / "dispim" / "devices" / "stage.py").read_text(encoding="utf-8")
STORE = (ROOT / "ui" / "web" / "static" / "js" / "xy-limits-state.js").read_text(encoding="utf-8")
DEVICES_JS = (ROOT / "ui" / "web" / "static" / "js" / "devices.js").read_text(encoding="utf-8")
RIG_JS = (ROOT / "ui" / "web" / "static" / "js" / "rig-menu.js").read_text(encoding="utf-8")


def _block(src: str, marker: str, span: int = 3000) -> str:
    """The source following a marker — plain indexing, no regex.

    Deliberately not a regex: the patterns needed here are full of escaped
    parens and newlines, and every attempt to write one through tooling has
    mangled the file instead.
    """
    return src[src.index(marker) :][:span]


def _switch() -> str:
    # Wide enough to reach the sidecar write at the end of the handler.
    return _block(DEVICE_LAYER, "async def handle_set_envelope_enforced(self, request):", 5000)


def _apply() -> str:
    return _block(DEVICE_LAYER, "async def handle_set_envelope(self, request):")


def _boot() -> str:
    return _block(DEVICE_LAYER, 'enforced = env.get("enforced") is True', 1600)


# ---------------------------------------------------------------------------
# The switch
# ---------------------------------------------------------------------------


def test_off_writes_the_full_travel() -> None:
    """The Tiger always holds some box; full travel is what "off" means."""
    body = _switch()
    assert "self._full_travel()" in body, (
        "turning limits off no longer writes the stage's full range, so the "
        "controller keeps fencing whatever it last had"
    )
    assert "set_firmware_limits" in body, "the switch does not reach the firmware"


def test_turning_them_on_needs_a_region_someone_walked() -> None:
    """A fence nobody measured is not a safety feature."""
    body = _switch()
    assert "No saved region to enforce" in body
    assert "status=409" in body


def test_a_string_cannot_unfence_a_stage() -> None:
    assert "isinstance(enforced, bool)" in _switch(), (
        "a truthy string could now remove the limits on someone else's stage"
    )


def test_the_region_survives_being_switched_off() -> None:
    assert '{**saved, "enforced": enforced}' in _switch(), (
        "the saved region is dropped when limits are turned off, so turning "
        "them back on means walking the corners again"
    )


def test_switching_off_does_not_unbound_gently() -> None:
    """The trap in the split.

    `set_firmware_limits` sets the software envelope from its own numbers, so
    writing full travel to the controller ALSO widens what Gently commands —
    unless the region is re-applied straight after.
    """
    body = _switch()
    assert "set_software_limits" in body, (
        "turning the controller fence off silently unbounds every move Gently "
        "makes, because the software envelope follows the firmware write"
    )
    assert body.index("set_firmware_limits") < body.index("set_software_limits")


# ---------------------------------------------------------------------------
# Boot
# ---------------------------------------------------------------------------


def test_the_controller_is_left_alone_unless_enforcement_is_asked_for() -> None:
    """Off is the default now, not merely a state that survives."""
    assert 'enforced = env.get("enforced") is True' in DEVICE_LAYER, (
        "boot decides enforcement some other way; unless it is opt-in, a saved "
        "region silently fences every client at the next restart"
    )
    assert "box = region if enforced else self._full_travel()" in _boot()


def test_boot_still_binds_gently_to_the_region() -> None:
    body = _boot()
    assert "set_software_limits(" in body, (
        "boot writes the controller but never bounds Gently to the region, so "
        "every move Gently makes is limited only by the stage's travel"
    )
    assert body.index("set_firmware_limits") < body.index("set_software_limits"), (
        "the region is applied before the firmware write, which then overwrites "
        "it with its own numbers"
    )


def test_a_stage_parked_outside_a_saved_region_still_boots() -> None:
    """Refusing to start is worse than starting bounded."""
    assert "require_inside=False" in _boot(), (
        "boot applies the region with the operator-facing refusal enabled, so a "
        "stage left outside it stops the device layer from starting at all"
    )


# ---------------------------------------------------------------------------
# Applying a region
# ---------------------------------------------------------------------------


def test_defining_a_region_does_not_fence_the_other_clients() -> None:
    """Applying a region and fencing the controller are different asks.

    They used to be one action: walking the corners wrote the Tiger and stored
    `enforced: True`, so defining a working area for Gently also stopped a
    Micro-Manager user's stage.
    """
    body = _apply()
    assert '"enforced": enforced' in body, "applying a region rewrites the enforcement choice"
    assert "if enforced:" in body, "the controller is written on every apply, regardless"
    assert "set_software_limits" in body, "applying a region no longer binds Gently to it"


def test_a_region_must_fit_the_stage() -> None:
    """The firmware write used to catch this for free.

    The controller rejected or clamped an impossible value and the read-back
    check raised. A software-only region needs its own guard, or Gently
    commands moves the hardware refuses and reports them as mysterious errors.
    """
    body = _block(STAGE, "def set_software_limits(", 4000)
    assert "XY_STAGE_X_MAX_UM" in body, "a region wider than the stage's travel is accepted"


def test_the_software_fence_keeps_the_107_refusal() -> None:
    """An envelope that excludes where the stage stands is a mis-measurement."""
    body = _block(STAGE, "def set_software_limits(", 4000)
    assert "require_inside" in body and "outside" in body


# ---------------------------------------------------------------------------
# What the operator sees
# ---------------------------------------------------------------------------


def test_enforced_is_read_back_not_remembered() -> None:
    """A flag in a config file would say what we meant, not what is."""
    body = _block(DEVICE_LAYER, "def _envelope_payload(self, xy_stage) -> dict:")
    assert '"enforced": not self._is_full_travel(box)' in body, (
        "the payload reports a stored flag rather than what the controller "
        "actually holds — a write that did not take would read as applied"
    )


def test_both_surfaces_render_from_one_store() -> None:
    """The map and the rig menu must not disagree about who is fenced."""
    assert "XYLimitsState.subscribe(" in DEVICES_JS
    assert "XYLimitsState.subscribe(" in RIG_JS
    assert "/api/devices/stage/envelope" in STORE
    m = re.search(r"function setupLimitsSwitch\(\) \{(.*?)\n    \}\n", DEVICES_JS, re.S)
    assert m and "/api/devices/stage/envelope" not in m.group(1), (
        "the map fetches the endpoint itself again"
    )


def test_a_failed_write_re_reads() -> None:
    write = re.search(r"async function write\(enforced\) \{(.*?)\n    \}", STORE, re.S)
    assert write and write.group(1).count("await read()") >= 2, (
        "a failed write leaves the stored answer at a guess, and this one "
        "decides whether someone else's stage is fenced"
    )


def test_the_note_says_who_it_affects() -> None:
    """The reason this exists is people who are not looking at Gently."""
    m = re.search(r"function setupLimitsSwitch\(\) \{(.*?)\n    \}\n", DEVICES_JS, re.S)
    assert m and "Micro-Manager" in m.group(1), (
        "the switch no longer says that it affects every client of this "
        "controller, which is the only reason it exists"
    )


def test_the_store_does_not_depend_on_hearing_one_event() -> None:
    """Reported from the rig: "Microscope not connected" under a live readout.

    The route was fine — a GET returned 200 with `enforced: false` while X/Y
    streamed. The store had asked once, early, been told no, and had no way to
    ask again: it re-read on DEVICE_LAYER_STATE, which fires ONCE per state
    change and is emitted by boot-banner.js, loaded BEFORE this file. A rig
    already up when the page loads announces itself into an empty room.
    """
    index = (ROOT / "ui" / "web" / "templates" / "index.html").read_text(encoding="utf-8")
    assert index.index("boot-banner.js") < index.index("xy-limits-state.js"), (
        "if this order ever flips the race below is gone, but so is the "
        "reason this test reads the way it does"
    )
    assert "gentlyDeviceReady" in STORE, (
        "the store has no way to learn the rig is up except an event it can "
        "be loaded too late to hear"
    )
    assert "scheduleRetry" in STORE, "an unanswered question is kept as an answer"


def test_it_does_not_contradict_the_readout_beside_it() -> None:
    """ "Microscope not connected" is a claim about hardware, not about us.

    Printed under a live X/Y the operator can watch moving, it sends them to
    check a cable. When the rig is up, an unreadable fence is our problem.
    """
    m = re.search(r"const offlineReason = \(\) =>\n?\s*(.*?);", STORE, re.S)
    assert m, "the store has only one reason for not knowing"
    assert "rigReady()" in m.group(1), (
        "the message does not depend on whether the rig is actually down"
    )
