"""The run, embryo by embryo, on the Acquisition pane; the DIC strip on Embryos.

CI runs no JavaScript and no browser, so what the pane and the tab need in
order to work is pinned as source. The status route the pane reads is tested
in tests/test_timelapse_start_route.py; the thumbnail on the event in
tests/test_dic_overview.py.
"""

from __future__ import annotations

from pathlib import Path

WEB = Path(__file__).resolve().parents[1] / "gently" / "ui" / "web"
HTML = (WEB / "templates" / "index.html").read_text(encoding="utf-8")
OPERATE = (WEB / "static" / "js" / "operate.js").read_text(encoding="utf-8")
EMBRYOS = (WEB / "static" / "js" / "embryos.js").read_text(encoding="utf-8")
MAIN_CSS = (WEB / "static" / "css" / "main.css").read_text(encoding="utf-8")


def _wire_body() -> str:
    start = OPERATE.index("if (_wired) return;")
    end = OPERATE.index("\n    async function ", start)
    return OPERATE[start:end]


def test_the_run_is_read_from_the_status_route():
    body = OPERATE[OPERATE.index("async function renderRun()") :][:2500]
    assert "/api/devices/timelapse/status" in body, (
        "the pane still guesses the run from tactic cards"
    )
    assert "st.embryos" in body


def test_each_embryo_gets_a_row_with_its_own_ending_and_stop():
    row = OPERATE[OPERATE.index("function runRow(") :][:2200]
    for field in ("t${r.timepoints}", "next ${fmtWhen(due)}", "data-run-stop=", "data-run-halt="):
        assert field in row, f"a run row lost {field}"
    assert "is_complete || !live ? '<span></span><span></span>'" in row, (
        "a finished embryo still offers Stop"
    )


def test_per_embryo_controls_are_wired_in_wire():
    """The rule test_operate_controls_are_wired.py pins: listeners live in wire()."""
    wiring = _wire_body()
    assert "$('op-runspine')" in wiring
    assert "[data-run-halt]" in wiring and "stopEmbryo(" in wiring
    assert "[data-run-stop]" in wiring and "changeEmbryoEnding(" in wiring


def test_stopping_one_embryo_is_confirmed_and_leaves_the_rest_alone():
    fn = OPERATE[OPERATE.index("async function stopEmbryo(") :][:700]
    assert "window.confirm(" in fn
    assert "/timelapse/embryo/${encodeURIComponent(id)}/stop" in fn
    assert "/api/devices/timelapse/stop'" not in fn, "stopping one embryo stops the run"


def test_the_run_is_watched_only_while_the_pane_is_open():
    pane = OPERATE[OPERATE.index("        acquire: {") :][:700]
    assert "setInterval(renderRun, 5000)" in pane
    assert "onLeave() { clearInterval(_runPoll)" in pane, "the poll outlives the pane"


def test_the_moments_of_a_run_redraw_it():
    wiring = _wire_body()
    for ev in (
        "ACQUISITION_STARTED",
        "ACQUISITION_COMPLETED",
        "IMAGE_ACQUIRED",
        "EMBRYO_TERMINATED",
    ):
        assert f"'{ev}'" in wiring, f"{ev} does not redraw the run"
    assert "scheduleRenderRun()" in wiring


def test_the_dic_strip_exists_and_is_fed_by_the_event():
    assert 'id="dic-strip"' in HTML and 'id="dic-strip-frames"' in HTML
    assert "ClientEventBus.on('IMAGE_ACQUIRED', (data) => this.handleDicFrame(data))" in EMBRYOS
    fn = EMBRYOS[EMBRYOS.index("handleDicFrame(data) {") :][:1400]
    assert "data.source !== 'dic'" in fn, "any IMAGE_ACQUIRED would land on the DIC strip"
    assert "data:image/png;base64," in fn
    render = EMBRYOS[EMBRYOS.index("renderDicStrip() {") :][:2400]
    assert "slice(-12)" in render, "the strip keeps every frame in the DOM"


def test_hidden_actually_hides_the_strip():
    assert ".dic-strip[hidden] { display: none; }" in MAIN_CSS
