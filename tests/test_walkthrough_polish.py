"""Three things from the first walkthrough of the acquisition surface.

* "the DIC overview image is not viewable … more like an icon, than a
  clickable image" — the strip's frames open a viewer with the full frame.
* "the embryos left layout has more space than needed, and the center and
  right one feels a bit crowded" — the columns are sized to what they hold.
* "why does the embryos tab say 2 embryos · Connected … once I hard
  refreshed it, it changed to 4" — the strip counted the image store's
  list, not the roster, and re-rendered only on a connection change.
"""

from __future__ import annotations

import re
from pathlib import Path

WEB = Path(__file__).resolve().parents[1] / "gently" / "ui" / "web"
HTML = (WEB / "templates" / "index.html").read_text(encoding="utf-8")
EMBRYOS = (WEB / "static" / "js" / "embryos.js").read_text(encoding="utf-8")
SHELL = (WEB / "static" / "js" / "shell.js").read_text(encoding="utf-8")
OPERATE_CSS = (WEB / "static" / "css" / "operate.css").read_text(encoding="utf-8")
MAIN_CSS = (WEB / "static" / "css" / "main.css").read_text(encoding="utf-8")
ROUTES_INIT = (WEB / "routes" / "__init__.py").read_text(encoding="utf-8")


def test_a_frame_on_the_strip_opens_the_stage_not_a_modal():
    """ "It appears more like an icon than a clickable image." A frame on the
    strip used to open a lightbox. Now it opens the overview stage in place —
    the newest frame large, the series beneath — on that frame. There is no
    modal: a frame you can see does not need a lightbox."""
    assert 'id="dic-viewer"' not in HTML
    assert 'id="embryos-overview"' in HTML and "panels/overview-stage.js" in HTML
    assert "openDicViewer(index) {" in EMBRYOS
    opener = EMBRYOS[EMBRYOS.index("openDicViewer(index) {") :][:400]
    assert "this._overviewOpen = true;" in opener and "OverviewStage.go(index, true)" in opener
    stage = (WEB / "static" / "js" / "panels" / "overview-stage.js").read_text(encoding="utf-8")
    for needle in (
        "ArrowRight",
        "ArrowLeft",
        "Home",
        "End",
        "' '",
        "setPointerCapture",
        "corrected=1",
        "FOLLOWING NEWEST",
    ):
        assert needle in stage, needle


def test_the_stage_walks_the_series_from_the_keyboard():
    """The series is walked on the stage, not in a modal: arrows step, Shift
    is ten, Home/End, Space plays, and a frame on the folded strip opens it."""
    stage = (WEB / "static" / "js" / "panels" / "overview-stage.js").read_text(encoding="utf-8")
    keys = stage[stage.index("function onKey(ev)") :][:1200]
    for key in ("'ArrowLeft'", "'ArrowRight'", "'Home'", "'End'", "' '", "ev.shiftKey ? 10 : 1"):
        assert key in keys, key
    wiring = EMBRYOS[EMBRYOS.index("_wireDicStrip() {") :][:900]
    assert "this.openDicViewer(Number(b.dataset.dicIndex))" in wiring
    assert "dic-strip-open" in wiring


def test_the_strip_hydrates_from_disk_for_a_page_that_opened_late():
    assert "/api/dic/frames" in EMBRYOS
    assert "this.refreshDicStrip();" in EMBRYOS
    assert "ClientEventBus.on('ACQUISITION_STARTED', () => this.refreshDicStrip())" in EMBRYOS
    assert "create_dic_router" in ROUTES_INIT, "the DIC routes are not registered"


def test_a_restored_session_brings_its_own_frames_to_the_strip():
    """A restored session fires no ACQUISITION_STARTED, and the page may have
    opened before the agent had a session at all. The session boundary is the
    one event both paths share: there the old session's frames go and the new
    session's are read from disk."""
    boundary = EMBRYOS[
        EMBRYOS.index("if (serverHasNewSession || serverIsIdleButClientHasData) {") :
    ][:600]
    assert "this._dicFrames = [];" in boundary
    assert "this.refreshDicStrip();" in boundary


def test_frames_on_the_strip_read_as_pictures_not_icons():
    m = re.search(r"\.dic-frame img[^{]*\{[^}]*height:\s*(\d+)px", MAIN_CSS)
    assert m and int(m.group(1)) >= 96, "strip frames are icon-sized again"
    assert re.search(r"\.dic-frame:hover[^{]*\{[^}]*border-color", MAIN_CSS), (
        "no hover, so nothing says 'clickable'"
    )


def test_the_columns_are_sized_to_what_they_hold():
    m = re.search(r"\.op-main-cols \{[^}]*grid-template-columns:\s*([^;]+);", OPERATE_CSS)
    assert m, "no column template"
    cols = m.group(1)
    frs = [float(x) for x in re.findall(r"([\d.]+)fr", cols)]
    assert len(frs) == 3
    assert frs[0] < frs[1] and frs[0] < frs[2], f"the roster column is not the narrowest: {cols}"
    assert "minmax(220px" in cols, "the roster can be squeezed below a readable row"


def test_the_strip_counts_the_roster_and_hears_it_change():
    assert "ClientEventBus.on('EMBRYOS_UPDATE'" in SHELL, "the strip never hears the roster"
    body = SHELL[SHELL.index("function renderStrip(") :][:900]
    assert "_rosterCount" in body, "the strip still counts the image store's list"


def test_hidden_actually_hides_the_viewer():
    assert ".overview-stage[hidden] { display: none; }" in MAIN_CSS


def test_the_interface_knows_when_it_is_acquiring():
    """ "some way the gently interface should appear different when an
    acquisition is running … as opposed to when we are idle, or setting up."
    The run is a state on <body>; the strip and the rail style from it."""
    assert "document.body.dataset.run = status" in SHELL
    for ev in (
        "ACQUISITION_STARTED",
        "ACQUISITION_COMPLETED",
        "ACQUISITION_STOPPED",
        "TIMELAPSE_STATE",
    ):
        assert f"'{ev}'" in SHELL, f"the shell does not hear {ev}"
    assert "setInterval(pollRun, 10000)" in SHELL, "a page opened mid-run would never learn of it"
    shell_css = (WEB / "static" / "css" / "shell.css").read_text(encoding="utf-8")
    assert 'body[data-run="running"] .v2-strip-dot' in shell_css
    assert 'body[data-run="running"] .v2-nav-item[data-tab="devices"]::after' in shell_css
    assert 'body[data-run="running"] .v2-nav-item[data-tab="embryos"]::after' in shell_css
    assert "prefers-reduced-motion" in shell_css


def test_idle_is_the_absence_of_a_rule():
    shell_css = (WEB / "static" / "css" / "shell.css").read_text(encoding="utf-8")
    assert 'body[data-run="idle"]' not in shell_css, (
        "idle should look like nothing is happening, because nothing is"
    )


def test_every_method_the_embryos_init_calls_exists():
    """A patch that replaced the DIC handler sliced from its comment to the next
    landmark — a thousand lines later — and took loadDashboardConfig() and
    everything else in between with it. init() then threw on its first line,
    on every page load, and the Embryos tab never subscribed to anything.
    Caught before it shipped; pinned so the class of mistake cannot ship."""
    init = re.search(r"\n    init\(\) \{(.*?)\n    \},", EMBRYOS, re.S)
    assert init, "no init()"
    called = set(re.findall(r"this\.(\w+)\(", init.group(1)))
    assert called, "init() calls nothing?"
    missing = [m for m in called if not re.search(rf"\n    (?:async )?{m}\(", EMBRYOS)]
    assert not missing, f"init() calls methods that do not exist: {missing}"


def test_the_embryos_module_kept_its_body():
    """A blunt size check, because the mistake above deleted ~1000 lines and
    every syntax check still passed."""
    assert EMBRYOS.count("\n") > 2400, f"embryos.js is {EMBRYOS.count(chr(10))} lines; was ~2600"
