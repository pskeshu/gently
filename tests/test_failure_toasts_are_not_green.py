"""A failure toast must not render with success styling.

`showGentlyToast` had no error variant, so `Delete failed (405)` was drawn with
the same green border as `Volume acquired`. An operator scanning for "did that
work?" got green either way — observed at 05:45 in the 2026-08-07 walkthrough.

The fix is an optional `level` on the single funnel plus a `toastFail` wrapper
at each caller. Nothing in CI runs JavaScript, so the thing that rots is a *new*
failure path added through the plain `toast(...)` wrapper. This reads the real
sources and fails on exactly that.

ponytail: failure is detected by the wording of the message, not by dataflow —
a call whose text says "failed" or "blocked" but whose level is success. That
catches the copy-paste case, which is the one that actually happens. A failure
worded without any of those tokens still slips through; upgrade to parsing the
enclosing `catch` block if one ever does.
"""

from __future__ import annotations

import re
from pathlib import Path

JS = Path(__file__).resolve().parents[1] / "gently" / "ui" / "web" / "static" / "js"
CSS = Path(__file__).resolve().parents[1] / "gently" / "ui" / "web" / "static" / "css" / "main.css"

# Wording that means the action did not happen.
FAILURE_WORDS = re.compile(
    r"\b(failed|failure|blocked|unavailable|cannot|could not|couldn't"
    r"|denied|refused|too close|unknown)\b",
    re.IGNORECASE,
)

# A call through a wrapper that does NOT pass the error level.
SUCCESS_CALL = re.compile(r"(?<![A-Za-z0-9_])toast\s*\((.+)$")


def _js_sources() -> list[Path]:
    return sorted(p for p in JS.glob("*.js") if p.name != "atrium.js")


def test_error_variant_exists_and_cannot_lose_on_specificity() -> None:
    """The rule must outrank `.gently-toast`, not merely follow it."""
    css = CSS.read_text(encoding="utf-8")
    assert ".gently-toast.gently-toast--error" in css, (
        "no doubled-class error rule — a single .gently-toast--error ties with "
        ".gently-toast at (0,1,0) and wins only by source order"
    )


def test_funnel_takes_a_level() -> None:
    src = (JS / "gallery.js").read_text(encoding="utf-8")
    signature = (
        "function showGentlyToast(message, actionLabel, actionFn, "
        "duration = 6000, level = 'success')"
    )
    assert signature in src, (
        "showGentlyToast lost its `level` parameter — every failure toast is green again"
    )
    assert "gently-toast--error" in src, "the funnel never applies the error class"


def test_no_failure_message_goes_out_as_a_success_toast() -> None:
    offenders: list[str] = []
    for path in _js_sources():
        for n, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            m = SUCCESS_CALL.search(line)
            if not m:
                continue
            if "function toast" in line or "showGentlyToast" in line:
                continue  # the wrapper's own definition
            if FAILURE_WORDS.search(m.group(1)):
                offenders.append(f"{path.name}:{n}: {line.strip()}")

    assert not offenders, (
        "these report a failure through the success toast — use toastFail(...):\n  "
        + "\n  ".join(offenders)
    )


if __name__ == "__main__":
    test_error_variant_exists_and_cannot_lose_on_specificity()
    test_funnel_takes_a_level()
    test_no_failure_message_goes_out_as_a_success_toast()
    print("ok")
