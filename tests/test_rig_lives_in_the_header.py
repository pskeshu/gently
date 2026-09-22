"""The rig is chrome, not a page — and the camera keeps the height that bought.

The device layer's state and its Start/Stop/Log, the water temperature and the
room light all lived on the Devices tab. That put the one question an operator
asks from anywhere ("is the microscope up?") behind a tab switch, and it spent
~274px of the Operate pane on furniture: the bottom-camera frame was 648x116 at
1440x920 while the contrast panel below it took 4.7x that area. The rig moved
into the header's chip-and-menu; the camera got the height.

Pinned here because CI runs no JavaScript. Nothing else would notice the card
growing back on the tab, a second device-layer poll appearing beside the one in
boot-banner.js, or Log sprouting its own tail instead of opening the console
drawer that already has a Device layer tab.
"""

from __future__ import annotations

import re
from pathlib import Path

WEB = Path(__file__).resolve().parents[1] / "gently" / "ui" / "web"
INDEX = WEB / "templates" / "index.html"
HEADER = WEB / "templates" / "_header.html"
RIG_JS = WEB / "static" / "js" / "rig-menu.js"
SHELL_JS = WEB / "static" / "js" / "shell.js"
BANNER_JS = WEB / "static" / "js" / "boot-banner.js"


def test_the_devices_tab_no_longer_carries_the_rig() -> None:
    """The card and the tab's temperature/room-light controls are gone."""
    index = INDEX.read_text(encoding="utf-8")
    for gone in (
        'id="devices-layer-card"',
        'id="devices-layer-start"',
        'id="devices-temp-readout"',
        'id="devices-room-light-toggle"',
    ):
        assert gone not in index, (
            f"{gone} is back on the Devices tab — the rig belongs to the header's "
            "rig menu, and that markup costs the camera its height"
        )
    assert "device-layer.js" not in index, (
        "device-layer.js is loaded again; rig-menu.js supersedes it"
    )


def test_the_header_carries_the_whole_rig() -> None:
    """State, the two verbs, the log, water and the room light — all in the menu."""
    header = HEADER.read_text(encoding="utf-8")
    for needed in (
        'id="rig-dl-state"',
        'id="rig-dl-start"',
        'id="rig-dl-stop"',
        'id="rig-dl-log"',
        'id="rig-temp-input"',
        'id="rig-light-toggle"',
        'id="rig-chip-temp"',
    ):
        assert needed in header, f"the rig menu lost {needed}"
    assert 'aria-haspopup="true"' in header, "the rig chip no longer announces its menu"


def test_the_rig_menu_rides_the_existing_poll_and_the_existing_console() -> None:
    """One device-layer poll in the app, one log surface.

    boot-banner.js polls /api/device-layer/status globally and publishes
    DEVICE_LAYER_STATE; the menu subscribes. A second poll here would double the
    request rate on every page for the same fact.
    """
    js = RIG_JS.read_text(encoding="utf-8")
    assert "ClientEventBus.on('DEVICE_LAYER_STATE'" in js, (
        "the rig menu no longer rides boot-banner.js's device-layer signal"
    )
    # One repeating timer, and it is the water/room-light read that only runs
    # while the menu is open. A second interval here would mean the device
    # layer is polled twice on every page, for the same fact.
    assert js.count("setInterval(") == 1 and "setInterval(refreshAmbient" in js, (
        "the rig menu has grown a timer of its own — the device-layer poll "
        "belongs to boot-banner.js, and the ambient read follows the menu"
    )
    assert "LogConsole.open()" in js and "selectSource('device')" in js, (
        "Log no longer opens the shared console on its Device layer tab"
    )


def test_the_rail_collapses_and_does_not_offer_the_agent_twice() -> None:
    """The agent is the docked right rail and Ctrl/Cmd+J; the left rail folds."""
    index = INDEX.read_text(encoding="utf-8")
    shell = SHELL_JS.read_text(encoding="utf-8")
    assert "v2-rail-chat" not in index and "v2-rail-chat" not in shell, (
        "the third agent entry point is back in the left rail"
    )
    assert 'id="v2-rail-collapse"' in index, "the rail lost its collapse control"
    assert "rail-collapsed" in shell, "shell.js no longer applies the collapsed state"
    assert re.search(r"localStorage\.setItem\(\s*COLLAPSE_KEY", shell), (
        "the rail no longer remembers the choice"
    )
    # Collapsed, the rail is icons only — so every destination needs one.
    items = re.findall(r'class="v2-nav-item"[^>]*>(.*?)</button>', index, re.S)
    assert len(items) >= 10, f"expected the ten destinations, found {len(items)}"
    for item in items:
        assert "v2-nav-icon" in item, (
            "a rail destination has no icon — it would be blank when collapsed"
        )


def test_the_display_range_is_one_row_under_the_camera() -> None:
    """Label, histogram, Auto, Reset — on a line, not a stacked panel.

    As three rows (heading, 60px histogram, its own button row) the display
    range took ~132px directly beneath the frame: more area than the image it
    describes, on the page whose subject is that image.
    """
    js = (WEB / "static" / "js" / "panels" / "imageview.js").read_text(encoding="utf-8")
    css = (WEB / "static" / "css" / "operate.css").read_text(encoding="utf-8")
    assert '<div class="iv-row">' in js, (
        "the display range is no longer a single row — it is back to costing the camera its height"
    )
    assert "lp-head" not in js, "the display panel grew its heading row back"
    assert re.search(r"\.iv-hist\s*\{[^}]*height:\s*30px", css, re.S), (
        "the histogram is no longer the compact 30px strip"
    )
    # The parts must still be there: this is a shrink, not a removal.
    for part in ("data-hist", 'data-h="lo"', 'data-h="hi"', "data-auto", "data-reset"):
        assert part in js, f"the display range lost {part} in the shrink"


def test_a_rig_that_was_asked_for_and_is_not_running_says_so() -> None:
    """The banner offers Start, takes "not now" for an answer, and never nags a
    session that chose to work without the microscope."""
    js = BANNER_JS.read_text(encoding="utf-8")
    assert "state === 'stopped' && _wantsHardware !== false" in js, (
        "the not-running banner no longer checks what the gate was asked for"
    )
    assert "The microscope is not running." in js
    assert "/api/launch/prefs" in js, "the banner no longer reads the gate's choice"
    assert 'id="boot-banner-notnow"' in INDEX.read_text(encoding="utf-8"), (
        "the banner lost its 'Not now'"
    )
