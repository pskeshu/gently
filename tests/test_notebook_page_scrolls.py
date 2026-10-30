"""The Notebook panel scrolls.

"when i click agents' view, it opens the notebook. but that page has no
ability to scroll..." The shell gives every tab a fixed-height box with
overflow hidden and leaves the scroll to the tab; the notebook had not
arranged one, so anything below the first screen was unreachable.
"""

from __future__ import annotations

from pathlib import Path

CSS = (
    Path(__file__).resolve().parents[1]
    / "gently"
    / "ui"
    / "web"
    / "static"
    / "css"
    / "notebook.css"
).read_text(encoding="utf-8")


def test_the_notebook_panel_is_its_own_scroll_container():
    assert "#notebook-content { overflow-y: auto; }" in CSS
