"""The controller's fence can be taken down, and stays down.

"in the joystick we have set the limits so the XY is not moved beyond that in
the joystick firmware. that is what we want to enable or disable from gently.
since it affects the larger users, who might use the system from micromanager"

`set_firmware_limits` writes `LowerLimX(mm)` and friends into the ASI Tiger,
and the Tiger enforces them against EVERY motion source — the code comment
says so. A region Gently wrote on Monday still stops the stage short for
someone driving it from Micro-Manager on Friday, with nothing on their screen
to connect it to. There was no way to turn it off: the envelope was always
some box.

Three things make this honest rather than cosmetic:

* off means the stage's FULL TRAVEL, because the controller always holds some
  box — there is no "no limits" to write;
* the operator's region survives being switched off, so turning it back on
  does not mean walking the corners again; and
* OFF PERSISTS. A boot that silently re-applied the region would re-fence the
  Micro-Manager user days later, which is the version of this bug nobody would
  connect to Gently.
"""

from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1] / "gently"
DEVICE_LAYER = (ROOT / "hardware" / "dispim" / "device_layer.py").read_text(encoding="utf-8")
STORE = (ROOT / "ui" / "web" / "static" / "js" / "xy-limits-state.js").read_text(encoding="utf-8")
DEVICES_JS = (ROOT / "ui" / "web" / "static" / "js" / "devices.js").read_text(encoding="utf-8")
RIG_JS = (ROOT / "ui" / "web" / "static" / "js" / "rig-menu.js").read_text(encoding="utf-8")


def _handler() -> str:
    m = re.search(
        r"async def handle_set_envelope_enforced\(self, request\):(.*?)\n    async def ",
        DEVICE_LAYER,
        re.S,
    )
    assert m, "the enforcement switch is gone"
    return m.group(1)


def test_off_writes_the_full_travel() -> None:
    """The Tiger always holds some box; full travel is what "off" means."""
    body = _handler()
    assert "self._full_travel()" in body, (
        "turning limits off no longer writes the stage's full range, so the "
        "controller keeps fencing whatever it last had"
    )
    assert "set_firmware_limits" in body, "the switch does not reach the firmware"


def test_turning_them_on_needs_a_region_someone_walked() -> None:
    """A fence nobody measured is not a safety feature."""
    body = _handler()
    assert "No saved region to enforce" in body
    assert "status=409" in body


def test_a_string_cannot_unfence_a_stage() -> None:
    body = _handler()
    assert "isinstance(enforced, bool)" in body, (
        "a truthy string could now remove the limits on someone else's stage"
    )


def test_the_region_survives_being_switched_off() -> None:
    body = _handler()
    assert '{**saved, "enforced": enforced}' in body, (
        "the saved region is dropped when limits are turned off, so turning "
        "them back on means walking the corners again"
    )


def test_off_survives_a_restart() -> None:
    """The whole point: the next boot must not re-fence the other user."""
    assert 'if env.get("enforced") is False:' in DEVICE_LAYER, (
        "boot re-applies the region regardless, so a device-layer restart "
        "silently re-fences every client days after someone turned it off"
    )
    boot = DEVICE_LAYER[DEVICE_LAYER.index('if env.get("enforced") is False:') :][:400]
    assert "_full_travel()" in boot


def test_defining_a_region_turns_enforcement_on() -> None:
    """Nobody walks the corners of a fence they want removed."""
    m = re.search(
        r"async def handle_set_envelope\(self, request\):(.*?)\n    async def ", DEVICE_LAYER, re.S
    )
    assert m and '"enforced": True' in m.group(1)


def test_enforced_is_read_back_not_remembered() -> None:
    """A flag in a config file would say what we meant, not what is."""
    m = re.search(
        r"def _envelope_payload\(self, xy_stage\) -> dict:(.*?)\n    async def ", DEVICE_LAYER, re.S
    )
    assert m, "the envelope payload is gone"
    assert '"enforced": not self._is_full_travel(box)' in m.group(1), (
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
