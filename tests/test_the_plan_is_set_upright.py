"""What is to be run is set upright.

"i do not like the italics in the what to run section... on description of
a run."
"""

from __future__ import annotations

import re
from pathlib import Path

CSS = (
    Path(__file__).resolve().parents[1] / "gently" / "ui" / "web" / "static" / "css" / "operate.css"
).read_text(encoding="utf-8")


def _rule(selector: str) -> str:
    start = CSS.index(selector + " {")
    return CSS[start : CSS.index("}", start)]


def test_the_plans_sentence_is_upright():
    assert "font-style: normal;" in _rule(".op-plan-say")


def test_a_saved_plans_sentence_is_upright():
    assert "font-style: normal;" in _rule(".op-libitem .op-lib-say")


def test_a_waiting_row_is_upright():
    assert "italic" not in _rule(".op-runrow.is-waiting .op-runrow-state")


def test_nothing_in_the_acquisition_pane_is_in_italics():
    # Every rule for the plan, the library and the run's rows.
    for m in re.finditer(r"(?m)^(\.op-(?:plan|lib|libitem|run|runrow|tcard)[^{]*)\{([^}]*)\}", CSS):
        assert "italic" not in m.group(2), m.group(1).strip()
