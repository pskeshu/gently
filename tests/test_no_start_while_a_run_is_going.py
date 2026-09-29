"""Start is not offered while a run is going.

"when a tactic is running in operate > acquisition, then why do i see the
option to run tactic again? it should be disabled right?"
"""

from __future__ import annotations

from pathlib import Path

OPERATE = (
    Path(__file__).resolve().parents[1] / "gently" / "ui" / "web" / "static" / "js" / "operate.js"
).read_text(encoding="utf-8")


def test_the_button_is_disabled_and_says_why_while_a_run_is_going():
    fn = OPERATE[OPERATE.index("    function renderRunButton() {") :][:500]
    assert "b.disabled = _runBusy;" in fn
    assert "_runBusy ? 'A run is going'" in fn
    assert "Stop the run before starting another" in fn


def test_whether_a_run_is_going_comes_from_the_server_not_a_flag_the_page_keeps():
    fn = OPERATE[OPERATE.index("    async function renderRun() {") :][:1200]
    running = fn.index("const running = st && (st.status === 'running' || st.status === 'paused');")
    assert fn.index("_runBusy = !!running;") > running
    assert "renderRunButton();" in fn
    # nothing else sets it
    assert OPERATE.count("_runBusy = ") == 2, "set somewhere besides its declaration and renderRun"


def test_a_paused_run_is_still_a_run():
    assert "st.status === 'running' || st.status === 'paused'" in OPERATE


def test_start_refuses_by_itself_too():
    fn = OPERATE[OPERATE.index("    async function startRun() {") :][:500]
    assert "if (_runBusy) {" in fn and "return; }" in fn
    assert fn.index("if (_runBusy)") < fn.index("_starting = true;")


def test_a_start_in_flight_keeps_its_own_label():
    fn = OPERATE[OPERATE.index("    function renderRunButton() {") :][:300]
    assert "if (!b || _starting) return;" in fn
    start = OPERATE[OPERATE.index("    async function startRun() {") :][:500]
    assert "const done = () => { _starting = false; renderRunButton(); renderRun(); };" in start
