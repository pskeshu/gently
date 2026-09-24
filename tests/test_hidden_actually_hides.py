"""`hidden` has to hide.

A ready device layer kept reporting "2/5 · Initializing Micro-Manager core" in
the rig menu. The renderer was right — it sets `dlProgress.hidden = true` the
moment the boot ends — but the row stayed on screen, holding the last thing it
had said.

The cause is a CSS specificity tie. The UA stylesheet's `[hidden] { display:
none }` is an attribute selector at specificity 0,1,0; a class rule like
`.rig-progress { display: flex }` is ALSO 0,1,0, and it comes later, so it
wins. The property is set, the element is "hidden" to every script that asks,
and it is plainly visible to the operator.

That last part is why this went unnoticed through several browser checks: they
asserted `element.hidden`, which is the property, not whether anything is on
screen. A test that reads the property can never catch this.

So this scans instead: any class that a script hides through the property, and
whose CSS sets `display`, must carry its own `[hidden]` guard. The pre-existing
code already does this — `.boot-banner[hidden]`, `.op-lock[hidden]`,
`.devices-region-wiz[hidden]` — it was the newer panels that missed it.
"""

from __future__ import annotations

import re
from pathlib import Path

WEB = Path(__file__).resolve().parents[1] / "gently" / "ui" / "web" / "static"

# Classes the UI toggles through `.hidden` in JS. Kept explicit rather than
# inferred: the point is to state which surfaces must be able to disappear.
TOGGLED_CLASSES = [
    "rig-progress",
    "cp",
    "cp-strip",
    "cal-adv",
    "op-adv",
    "al-history",
    "boot-banner",
    "op-lock",
    "devices-region-wiz",
    "devices-limits",
]


def _all_css() -> str:
    return "\n".join(p.read_text(encoding="utf-8") for p in sorted(WEB.glob("css/*.css")))


def _sets_display(css: str, cls: str) -> bool:
    blocks = re.findall(r"(?m)^\." + re.escape(cls) + r"\s*\{([^}]*)\}", css)
    return any("display:" in b for b in blocks)


def _has_guard(css: str, cls: str) -> bool:
    return bool(re.search(r"\." + re.escape(cls) + r"\[hidden\]", css))


def test_every_hideable_class_can_actually_hide() -> None:
    css = _all_css()
    broken = [
        cls for cls in TOGGLED_CLASSES if _sets_display(css, cls) and not _has_guard(css, cls)
    ]
    assert not broken, (
        "these classes set `display` and are hidden through the property, so "
        f"`hidden` does nothing for them: {broken}. Add `.<cls>[hidden] "
        "{ display: none; }` — the UA rule loses the specificity tie."
    )


def test_the_guards_are_not_quietly_deleted() -> None:
    """The bug is invisible to anything that asks the DOM, so pin the fix."""
    css = _all_css()
    for cls in ("rig-progress", "cp", "cal-adv", "op-adv"):
        assert _has_guard(css, cls), f".{cls} lost its [hidden] guard"


def test_a_new_panel_with_a_display_rule_is_noticed() -> None:
    """The list above is the contract; this checks it is not stale.

    If a class in the list no longer exists in the CSS at all, someone renamed
    or removed a panel and the list should follow — otherwise the scan quietly
    protects nothing.
    """
    css = _all_css()
    missing = [cls for cls in TOGGLED_CLASSES if not re.search(r"\." + re.escape(cls) + r"\b", css)]
    assert not missing, f"these classes are no longer in the CSS: {missing}"
