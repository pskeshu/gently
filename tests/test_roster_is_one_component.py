"""The embryo roster is rendered once, and published as an independent copy.

There were three renderings of one list: `renderEmbryoRail` and `renderRoster`
in operate.js — ~80% the same code, each with its own count element, which is
the root of #129 — plus a badge in embryos.js. Worse than the duplication, the
action sets differed arbitrarily: delete on Bottom cam, Centre and role on
Acquisition, for no reason either pane justified.

Both are now `panels/roster.js`, mounted twice with actions declared per mount.

The second assertion here is the more interesting one, and it is a bug this
change introduced and then fixed. `SharedState.set` only emits on a real change
and compares by value. `toggleRole` mutates `emb.role` **in place**, so
publishing `_embryos.slice()` stored an array of the very objects being
compared against — the change was invisible and the role button silently did
nothing. Same lesson as #126: never hand out a reference to mutable state.
"""

from __future__ import annotations

import re
from pathlib import Path

JS = Path(__file__).resolve().parents[1] / "gently" / "ui" / "web" / "static" / "js"


def test_operate_no_longer_renders_the_roster_itself() -> None:
    src = (JS / "operate.js").read_text(encoding="utf-8")
    for gone in ("function renderEmbryoRail", "function renderRoster"):
        assert gone not in src, (
            f"{gone} is back — the roster is a panel now, and two renderers of "
            "one list is what #129 is about"
        )


def test_the_published_snapshot_is_independent() -> None:
    src = (JS / "operate.js").read_text(encoding="utf-8")
    body = src[src.index("function publishRoster()") :]
    body = body[: body.index("\n    }")]

    assert "structuredClone" in body, (
        "publishRoster no longer deep-copies. SharedState compares by value and "
        "emits only on change, so publishing a shallow copy of elements that are "
        "mutated in place stores the objects it is comparing against — and the "
        "change becomes invisible. toggleRole mutates emb.role in place."
    )
    assert "_embryos.slice()" not in body, "a shallow copy is the bug, not the fix"


def test_both_mounts_declare_their_actions() -> None:
    """The difference between the two lists must be a decision, not an accident."""
    src = (JS / "operate.js").read_text(encoding="utf-8")
    mounts = re.findall(r"RosterPanel\.mount\('([^']+)',\s*\{([^}]*)\}", src, re.S)
    hosts = {h for h, _ in mounts}
    assert hosts == {"op-erail-list", "op-roster"}, f"unexpected mounts: {hosts}"
    for host, opts in mounts:
        assert "actions:" in opts, f"{host} mounts the roster without declaring actions"


def test_the_panel_dispatches_rather_than_reimplementing() -> None:
    """The list, the selection and the endpoints stay in operate.js."""
    panel = (JS / "panels" / "roster.js").read_text(encoding="utf-8")
    assert "OperateManager.roster" in panel
    # A panel that fetches is a panel that has opinions about endpoints.
    assert "fetch(" not in panel, "the roster panel should call verbs, not endpoints"

    src = (JS / "operate.js").read_text(encoding="utf-8")
    exported = src[src.index("        roster: {") :]
    exported = exported[: exported.index("\n        }")]
    for verb in ("select", "remove", "centre", "toggleRole", "goTo"):
        assert f"{verb}:" in exported, f"operate.js stopped exporting the {verb} verb"


def test_every_mount_offers_a_way_out_of_the_empty_state() -> None:
    """The actionable empty state used to be on the pane you reached second.

    Bottom cam described the fix without offering it; Acquisition carried the
    button. The panel's empty state now names the fix everywhere, and the CTA
    is opt-in per mount.
    """
    panel = (JS / "panels" / "roster.js").read_text(encoding="utf-8")
    empty = panel[panel.index("function empty(opts)") :]
    empty = empty[: empty.index("\n    }")]
    assert "detect on the bottom camera" in empty
    assert "emptyAction" in empty
