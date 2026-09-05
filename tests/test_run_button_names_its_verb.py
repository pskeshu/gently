"""`Start` must say which of four things it will do.

`startRun` branches on `_mode` into four different verbs — `single` acquires
one volume and finishes, `adaptive` starts a timelapse, `library` runs a saved
tactic, `agent` hands over a prompt. The button said "Start" for all four, and
the mode selector lives in a block above it, so an operator who chose a mode
and then looked away had nothing on the button to read.

Two invariants worth pinning, because both rot silently:

1. Every mode has a verb. A mode added to `setMode`'s list but not to
   `RUN_VERB` falls back to "Start" and quietly reintroduces the ambiguity.
2. `subjectIds()` keeps no fallback. The old `subs.length ? subs : all` could
   only ever fire when EVERY embryo was marked `calibration` — an embryo with
   no role, or 'test', or 'unassigned' already passes the filter. So it fired
   exactly when the operator had said "these are all references" and answered
   by imaging all of them as subjects.

ponytail: source assertions, because operate.js is an IIFE with no export
surface and CI runs no JavaScript. They check the shape, not the behaviour.
"""

from __future__ import annotations

import re
from pathlib import Path

OPERATE = (
    Path(__file__).resolve().parents[1] / "gently" / "ui" / "web" / "static" / "js" / "operate.js"
)


def _src() -> str:
    return OPERATE.read_text(encoding="utf-8")


def test_every_run_mode_has_a_verb() -> None:
    src = _src()

    block = src[src.index("const RUN_VERB = {") : src.index("function renderRunButton")]
    verbs = set(re.findall(r"(\w+):\s*'", block))

    assert re.search(r"\['single', 'adaptive', 'library', 'agent'\]", src), (
        "setMode's mode list changed shape — update this test alongside it"
    )
    modes = {"single", "adaptive", "library", "agent"}

    assert modes <= verbs, (
        f"modes with no verb in RUN_VERB: {sorted(modes - verbs)} — the button "
        "falls back to 'Start' for those and the ambiguity is back"
    )


def test_single_mode_does_not_claim_to_start_anything() -> None:
    """It acquires one volume and finishes. The label must not imply a run."""
    src = _src()
    block = src[src.index("const RUN_VERB = {") : src.index("function renderRunButton")]
    single = re.search(r"single:\s*'([^']+)'", block)
    assert single, "single mode lost its verb"
    label = single.group(1).lower()
    assert "start" not in label, (
        f"single mode's label is {label!r} — it acquires one volume and stops, "
        "so it must not read as starting an experiment"
    )


def test_subject_ids_keeps_no_fallback() -> None:
    src = _src()
    body = src[src.index("function subjectIds()") :]
    body = body[: body.index("}")]
    assert "_embryos.map" not in body, (
        "subjectIds() has a fallback again — it can only fire when every embryo "
        "is a reference, and it answers by imaging all of them as subjects"
    )


def test_a_reference_only_roster_is_refused_rather_than_imaged() -> None:
    src = _src()
    assert "function haveSubjects()" in src
    assert src.count("haveSubjects()") >= 3, (
        "haveSubjects is defined but not guarding both roster-driven run modes"
    )
    # "no embryos" and "no subjects among your embryos" need different fixes,
    # so the guard must distinguish them rather than emitting one message.
    guard = src[src.index("function haveSubjects()") :]
    guard = guard[: guard.index("\n    }")]
    assert "No embryos registered" in guard
    assert "marked as a reference" in guard
