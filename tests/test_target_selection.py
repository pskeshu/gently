"""A run's targets are said, not inferred.

The orchestrator and the route have always accepted an explicit `embryo_ids`
list — `orchestrator.start(embryo_ids=[...])` images exactly those. The UI never
offered it: `adaptive` and `library` sent every non-reference embryo, and there
was no way to say "just this one".

The only workaround was to mark the others as references, and that is not a
selection mechanism. `role` decides what an embryo is *for*:
`expression_monitoring` scopes to `role == 'test'`, and the orchestrator's
`_is_eligible` reads `role.photodose_budget_multiplier` for the dose budget. So
excluding an embryo from tonight's run by demoting it also changed its
monitoring and its exposure allowance.

Selection is now a set, and the run's scope is explicit.

THE SAFETY PROPERTY, and the reason for the two-state control: targets must
never follow the selection implicitly. A plain click in the roster would
otherwise narrow a timelapse from every subject to one, silently. `all` is the
default and is what this pane has always done; narrowing is a thing you say.
"""

from __future__ import annotations

import re
from pathlib import Path

JS = Path(__file__).resolve().parents[1] / "gently" / "ui" / "web" / "static" / "js"
INDEX = Path(__file__).resolve().parents[1] / "gently" / "ui" / "web" / "templates" / "index.html"


def _operate() -> str:
    return (JS / "operate.js").read_text(encoding="utf-8")


def test_the_default_scope_is_every_subject() -> None:
    """Anything else is a silent narrowing of an existing workflow."""
    src = _operate()
    assert re.search(r"let _targetScope = 'all'", src), (
        "the default run scope is no longer 'all' — a plain click in the roster "
        "would narrow a timelapse without saying so"
    )


def test_scope_selects_between_two_explicit_sets() -> None:
    src = _operate()
    body = src[src.index("function subjectIds()") :]
    body = body[: body.index("\n    }")]
    assert "_targetScope !== 'selected'" in body
    assert "allSubjectIds()" in body


def test_a_reference_is_never_a_target_however_it_is_selected() -> None:
    """Role still decides what an embryo is for, whatever is highlighted."""
    src = _operate()
    body = src[src.index("function subjectIds()") :]
    body = body[: body.index("\n    }")]
    assert "'calibration'" in body, (
        "subjectIds no longer excludes references — selecting one would image it as a subject"
    )


def test_the_primary_and_the_set_are_both_published() -> None:
    """Instrument panes act on the primary; runs act on the set."""
    src = _operate()
    assert "SharedState.set('selectedEmbryoIds'" in src
    assert "SharedState.set('selectedEmbryoId'" in src
    store = (JS / "status-store.js").read_text(encoding="utf-8")
    assert "selectedEmbryoIds" in store


def test_all_three_click_modes_exist() -> None:
    panel = (JS / "panels" / "roster.js").read_text(encoding="utf-8")
    assert "metaKey" in panel and "ctrlKey" in panel, "no toggle modifier"
    assert "shiftKey" in panel, "no range modifier"
    assert "'replace'" in panel, "a plain click must still replace the set"

    src = _operate()
    sel = src[src.index("function selectEmbryo(id, mode)") :]
    sel = sel[: sel.index("\n    }")]
    for mode in ("toggle", "range"):
        assert f"'{mode}'" in sel, f"selectEmbryo does not handle {mode}"


def test_the_set_drops_embryos_that_no_longer_exist() -> None:
    """A roster refresh must not leave the set targeting ghosts (#126's class)."""
    src = _operate()
    upd = src[src.index("function onEmbryosUpdate(p)") :]
    upd = upd[: upd.index("\n    }")]
    assert "_targets = _targets.filter" in upd


def test_the_pane_states_which_set_it_will_use() -> None:
    """The modifier keys need not be discovered for the scope to be readable."""
    html = INDEX.read_text(encoding="utf-8")
    assert 'id="op-target-scope"' in html
    assert 'data-scope="all"' in html and 'data-scope="selected"' in html
    src = _operate()
    body = src[src.index("function renderTargetScope()") :]
    body = body[: body.index("\n    }")]
    # Counts, so "Selected" is never an unknown quantity.
    assert "All subjects (${all})" in body
    assert "Selected (${" in body


def test_selected_cannot_be_chosen_with_nothing_selected() -> None:
    src = _operate()
    body = src[src.index("function renderTargetScope()") :]
    body = body[: body.index("\n    }")]
    assert "disabled = !_targets.length" in body
