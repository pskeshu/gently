"""The Adaptive pane is a form for one object, said as one sentence.

CI runs no JavaScript and no browser, so what the page has to have for the
configurator to work is pinned as source. The plan's own logic is tested in
tests/js/acquisition-plan.test.mjs.
"""

from __future__ import annotations

import re
from pathlib import Path

WEB = Path(__file__).resolve().parents[1] / "gently" / "ui" / "web"
HTML = (WEB / "templates" / "index.html").read_text(encoding="utf-8")
OPERATE = (WEB / "static" / "js" / "operate.js").read_text(encoding="utf-8")
PLAN = (WEB / "static" / "js" / "acquisition-plan.js").read_text(encoding="utf-8")
CSS = (WEB / "static" / "css" / "operate.css").read_text(encoding="utf-8")


def test_the_plan_module_is_loaded_before_the_pane_that_uses_it():
    assert '<script src="/static/js/acquisition-plan.js"></script>' in HTML
    assert HTML.index("acquisition-plan.js") < HTML.index('src="/static/js/operate.js"')


def test_the_pane_asks_only_for_what_the_instrument_cannot_answer():
    """Positions are the roster; z is each embryo's calibration. Not asked."""
    panel = HTML[HTML.index('id="op-panel-adaptive"') : HTML.index('id="op-panel-library"')]
    for needed in (
        "op-tl-interval",
        "op-plan-unit",
        "op-plan-slices",
        "op-plan-exposure",
        "op-plan-laser",
        "op-plan-dic",
        "op-plan-dic-every",
        "op-plan-dic-pos",
        "op-tl-stop",
        "op-tl-condval",
        "op-plan-overrides",
        "op-tl-monitor",
    ):
        assert f'id="{needed}"' in panel, f"the pane lost {needed}"
    for not_asked in ("galvo", "piezo", "position_x", "z_start"):
        assert not_asked not in panel.lower(), (
            f"the pane asks for {not_asked}, which calibration answers"
        )


def test_the_sentence_is_on_the_pane_and_heads_the_run():
    assert 'id="op-plan-say"' in HTML
    assert 'id="op-run-plan"' in HTML, "Start has nowhere to land"


def test_start_sends_exactly_what_was_said():
    branch = OPERATE[
        OPERATE.index("if (_mode === 'adaptive') {") : OPERATE.index("if (_mode === 'library') {")
    ]
    assert "AcquisitionPlan.toPayload(plan, ids)" in branch, (
        "Start builds its own request beside the sentence"
    )
    # With the device layer's power limits, so a laser power the hardware
    # would refuse is refused here and not at the first volume.
    assert "AcquisitionPlan.validate(plan, ids, _laserLimits)" in branch, (
        "an invalid plan can be started"
    )
    assert "landOnRun(" in branch, "Start does not land on the run"
    assert "op-tl-condval" not in branch, "the old hand-built stop string is back"


def _wire_body() -> str:
    """wire() — where every Operate listener lives (see
    test_operate_controls_are_wired.py for why that rule is pinned)."""
    start = OPERATE.index("if (_wired) return;")
    end = OPERATE.index("\n    async function ", start)
    return OPERATE[start:end]


def test_every_input_re_says_the_plan():
    wiring = _wire_body()
    assert "$('op-panel-adaptive')" in wiring, "the plan's listeners are not attached in wire()"
    assert "addEventListener('input', () => { _planDirty = true; renderPlan(); })" in wiring
    assert "addEventListener('change'" in wiring


def test_taken_from_here_captures_the_stage_when_chosen_not_at_start():
    wiring = _wire_body()
    assert "_dicPin = { x: _xy.x, y: _xy.y }" in wiring
    assert "No stage position known yet" in wiring


def test_the_sentence_sits_beside_start_not_atop_a_scrolling_panel():
    panel = HTML[HTML.index('id="op-panel-adaptive"') : HTML.index('id="op-panel-library"')]
    assert 'id="op-plan-say"' not in panel, "the sentence is back at the top of the panel"
    tail = HTML[HTML.index('id="op-plan-say"') : HTML.index('id="op-run-start"')]
    assert len(tail) < 600, "the sentence and Start are not next to each other"


def test_the_roster_keeps_the_override_rows_in_step():
    assert "renderTargetScope(); renderPlan();" in OPERATE, (
        "a roster change leaves stale per-embryo rows"
    )


def test_the_plan_is_pure():
    """No DOM, no network: the same object a saved tactic stores and the agent seeds."""
    body = PLAN[PLAN.index("const AcquisitionPlan") :]
    for forbidden in ("document.", "fetch(", "window.", "localStorage"):
        assert forbidden not in body, f"acquisition-plan.js reaches for {forbidden}"
    assert "module.exports = AcquisitionPlan" in PLAN


def test_hidden_actually_hides_the_plans_panels():
    """`[hidden]` is specificity 0,1,0; a class rule that sets display beats it."""
    for cls in (".op-plan-say", ".op-plan-dic-body"):
        rule = re.search(re.escape(cls) + r"\s*\{[^}]*display:", CSS)
        if rule:
            assert re.search(re.escape(cls) + r"\[hidden\]\s*\{[^}]*display:\s*none", CSS), (
                f"{cls} sets display without a [hidden] guard"
            )


def test_the_stop_vocabulary_is_what_the_orchestrator_parses():
    """Every kind the pane offers must be a spec _parse_stop_condition accepts."""
    kinds = re.search(r"const STOP_KINDS = \{(.*?)\n    \};", PLAN, re.S)
    assert kinds
    offered = set(re.findall(r"^\s*(\w+): \{", kinds.group(1), re.M))
    assert offered == {"manual", "timepoints", "duration", "hatching", "comma", "all_test_hatched"}


def test_a_detect_clears_the_last_attempts_note_before_it_starts():
    """Reported: "Automatic detection is unavailable on this rig" sitting under
    a detect that was, at that moment, succeeding. The note was from a press
    made in the half-minute between the page connecting and the device layer
    attaching, and nothing ever cleared it."""
    fn = OPERATE[OPERATE.index("async function runDetect(") :][:3200]
    assert fn.index("setDetectNote('')") < fn.index("postJSON('/api/devices/detect_embryos'")


def test_the_two_503s_are_told_apart():
    """ "Not connected" is a wait; "no SAM" is this rig's shape."""
    branch = OPERATE[OPERATE.index("if (e.status === 503) {") :][:900]
    assert "/not connected/i.test(detail)" in branch
    assert "still coming up" in branch
    assert "unavailable on this rig" in branch
