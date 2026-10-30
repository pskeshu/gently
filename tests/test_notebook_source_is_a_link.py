"""A notebook card's session chip goes to the line it came from.

"currently the notebook entry has session id mentioned, but nothing is
clickable, not even the source of the comment." A note written on a session
is a line of that session's stream. So the chip opens the session in the
Sessions tab (by URL: /#sessions:<sid>:<noteId>), every stream line carries
its note id, and the stream lands on that line.

CI drives no browser, so this reads the sources for the three ends of the
link; the click itself is checked by hand.
"""

from __future__ import annotations

from pathlib import Path

JS = Path(__file__).resolve().parents[1] / "gently" / "ui" / "web" / "static" / "js"


def test_the_card_links_the_session_chip_to_the_note_on_the_review_page():
    src = (JS / "notebook.js").read_text(encoding="utf-8")
    assert "/#sessions:${encodeURIComponent(ch.sid)}:${encodeURIComponent(n.id)}" in src
    assert "ReviewApp.openNote(ch.sid, n.id)" in src


def test_every_stream_line_carries_its_note_id():
    src = (JS / "utils.js").read_text(encoding="utf-8")
    assert "data-note-id=\"${escapeHtml(n.id || '')}\"" in src


def test_the_review_page_lands_on_the_line_in_the_hash():
    src = (JS / "review.js").read_text(encoding="utf-8")
    assert "this.landOnNote()" in src, "mountStream must call landOnNote after drawing the stream"
    assert "async openNote(sessionId, noteId)" in src
    assert "scrollIntoView" in src
    app = (JS / "app.js").read_text(encoding="utf-8")
    assert "ReviewApp.openNote(param, item || null)" in app, (
        "/#sessions:<sid>:<noteId> must route here"
    )
