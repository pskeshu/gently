"""The build id on the launch gate can be copied in one click.

`CONTRIBUTING.md` asks for the launch-gate string in every bug report, and the
gate is the one screen a reviewer sees before logging in — so the id has to
leave the screen easily. As rendered text it had to be selected by dragging
across a 12px line, which is exactly the friction that turns "1.0.0.dev1
+g0b8da49-dirty" into "the latest one, I think" and costs a reviewer round trip.

Pinned here because CI runs no JavaScript: nothing else would notice if the
control went back to being a `<span>`, or if the handler stopped copying the
version actually on screen.
"""

from __future__ import annotations

import re
from pathlib import Path

GATE = Path(__file__).resolve().parents[1] / "gently" / "ui" / "web" / "templates" / "launch.html"


def _gate() -> str:
    return GATE.read_text(encoding="utf-8")


def test_the_version_is_a_control_not_a_label() -> None:
    """A button: focusable, keyboard-activatable, and announced as pressable."""
    html = _gate()
    button = re.search(
        r'<button[^>]*id="foot-version"[^>]*>\s*v\{\{\s*gently_version\s*\}\}\s*</button>',
        html,
        re.S,
    )
    assert button, (
        "the launch gate's build id is no longer a <button id='foot-version'> "
        "rendering v{{ gently_version }} — it cannot be copied in one click"
    )
    assert 'type="button"' in button.group(0), (
        "the build-id button has no type='button' — inside a form it would submit"
    )


def test_clicking_it_copies_what_is_on_screen() -> None:
    """The handler copies the button's own text, so copied == displayed."""
    html = _gate()
    handler = re.search(
        r'getElementById\("foot-version"\).*?\}\)\(\);',
        html,
        re.S,
    )
    assert handler, "no click handler is bound to the build-id button"
    body = handler.group(0)
    assert re.search(r"btn\.textContent", body), (
        "the copy handler no longer reads the button's own text — the copied "
        "string could drift from the one displayed"
    )
    assert "clipboard.writeText" in body, "the copy handler no longer writes to the clipboard"


def test_copying_still_works_without_a_secure_context() -> None:
    """A LAN http:// visit has no navigator.clipboard; the fallback must hold.

    Two rungs: execCommand('copy'), then selecting the id so Ctrl+C works and
    saying so. Without them the gate would silently do nothing on exactly the
    machines a reviewer is most likely to be using.
    """
    body = _gate()
    assert "isSecureContext" in body, "the copy path no longer checks for a secure context"
    assert 'execCommand("copy")' in body, "the execCommand fallback is gone"
    assert "Press Ctrl+C to copy" in body, (
        "nothing tells the operator what to do when both copy paths fail"
    )


def test_the_id_itself_never_moves() -> None:
    """Feedback goes to the note beside it, not into the button's own label.

    Swapping the id for 'Copied' would reflow the line under the cursor — and a
    second click would then copy the word 'Copied'.
    """
    html = _gate()
    assert re.search(r'id="foot-note"[^>]*aria-live="polite"', html), (
        "the launch gate's copy feedback is no longer a polite live region beside the build id"
    )
    assert re.search(r"note\.textContent\s*=\s*msg", html), (
        "copy feedback no longer goes to the note element"
    )
