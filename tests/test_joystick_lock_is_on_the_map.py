"""The joystick lock is reachable where the stage is.

It existed — `GET/POST /api/devices/stage/joystick`, written to the Tiger
controller and read back, persisted so a boot re-applies the operator's choice
rather than forcing the joystick on — but the only control for it was a
checkbox in Settings. That is not where anyone is standing when they wonder
why the physical controller does nothing, or when they want it to stop working
while a run is on.

What must stay true:

* the state shown is READ BACK from the controller, never the command that was
  sent (PANELS rule 3) — this one matters more than usual, because the failure
  mode is a UI claiming a lock the hardware refused while someone leans on the
  joystick; and
* a failed write re-reads rather than guessing, so the button never settles
  into a state nobody confirmed.
"""

from __future__ import annotations

import re
from pathlib import Path

WEB = Path(__file__).resolve().parents[1] / "gently" / "ui" / "web"
DEVICES_JS = WEB / "static" / "js" / "devices.js"
INDEX = WEB / "templates" / "index.html"


def _setup() -> str:
    js = DEVICES_JS.read_text(encoding="utf-8")
    m = re.search(r"function setupJoystickLock\(\) \{(.*?)\n    \}\n", js, re.S)
    assert m, "the map lost its joystick control"
    return m.group(1)


def test_the_control_is_on_the_map() -> None:
    html = INDEX.read_text(encoding="utf-8")
    assert 'id="devices-js-toggle"' in html, "no joystick control on the map"
    js = DEVICES_JS.read_text(encoding="utf-8")
    assert "setupJoystickLock();" in js, "the control is never wired up"


def test_what_is_shown_is_read_back_not_commanded() -> None:
    body = _setup()
    # The rendered state comes from the response, not from the request.
    assert "show(!!d.enabled)" in body, (
        "the button renders the value it sent instead of the one the controller "
        "reported — a lock the hardware refused would look applied"
    )
    assert "read()" in body, "the control never reads the controller at all"


def test_a_failed_write_re_reads_rather_than_guessing() -> None:
    body = _setup()
    failure_paths = body.count("read();")
    assert failure_paths >= 3, (
        "a failed or forbidden write leaves the button in a state nobody "
        "confirmed; every failure path must re-read"
    )


def test_the_locked_state_is_visible_as_a_state() -> None:
    """ "Why is the joystick dead?" should be answerable from across the room."""
    body = _setup()
    assert "aria-pressed" in body, "the lock is not exposed as a pressed state"
    css = (WEB / "static" / "css" / "devices.css").read_text(encoding="utf-8")
    assert '.devices-js-btn[aria-pressed="true"]' in css, (
        "a locked joystick looks identical to an enabled one"
    )


def test_settings_and_the_map_drive_the_same_endpoint() -> None:
    """Two controls for one hardware flag must not diverge."""
    settings = (WEB / "templates" / "settings.html").read_text(encoding="utf-8")
    assert "/api/devices/stage/joystick" in settings
    assert "/api/devices/stage/joystick" in _setup()
