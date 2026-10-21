"""A reporter that has not switched on is not an empty field.

2026-10-21, four nuclear-marker embryos imaged as role ``test``: the reporter
detector read "no puncta" five rounds running, the orchestrator filed each
reading as ``no_object``, and the role's patience for empty fields stopped
three of the four embryos as "likely hatched / out of FOV". They were there.
"""

from __future__ import annotations

import re
from pathlib import Path

from gently.app.orchestration.timelapse import reporter_pseudo_stage

ORCH = (
    Path(__file__).resolve().parents[1] / "gently" / "app" / "orchestration" / "timelapse.py"
).read_text(encoding="utf-8")


def test_a_dark_reporter_is_no_signal_never_no_object():
    assert reporter_pseudo_stage("NONE", False) == "no_signal"
    for level in ("NONE", "WEAK", "MEDIUM", "STRONG", "SATURATING", "UNCERTAIN", None, "odd"):
        assert reporter_pseudo_stage(level, False) != "no_object", level


def test_the_other_readings_keep_their_words():
    assert reporter_pseudo_stage("SATURATING", False) == "lit_up_saturating"
    assert reporter_pseudo_stage("MEDIUM", False) == "lit_up"
    assert reporter_pseudo_stage("STRONG", False) == "lit_up"
    assert reporter_pseudo_stage("WEAK", False) == "weak_signal"
    assert reporter_pseudo_stage("UNCERTAIN", False) == "uncertain"
    assert reporter_pseudo_stage("NONE", True) == "hatched"
    assert reporter_pseudo_stage(None, False) == "unknown"


def test_the_reporter_path_does_not_touch_the_empty_field_counter():
    """Only the perception path, which can actually see an empty field, counts."""
    start = ORCH.index("pseudo_stage = reporter_pseudo_stage(")
    end = ORCH.index("def ", start)
    assert "consecutive_no_object" not in ORCH[start:end]
    # The counter is still driven by perception's own empty-field call.
    assert re.search(
        r'if result\.stage == "no_object":\s*\n\s*embryo_state\.consecutive_no_object \+= 1',
        ORCH,
    )
