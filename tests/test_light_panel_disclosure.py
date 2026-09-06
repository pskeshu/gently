"""The Light panel reveals the laser's settings once a line is routed.

Beam and power only MATTER once the config routes something, so they are
hidden until then. That is scope, not dependency — and the distinction is the
whole point, because #106 is what happens when someone assumes `BeamEnabled`
follows from the config. Emission is conjunctive:

    emitting = armed AND routed AND power > 0

So the indent groups the laser's own settings, and the contradiction between
them gets a line of its own rather than being softened by the nesting.

Four states, and which detail appears in each is the design:

    config unknown        nothing revealed  (it used to render FOUR disabled
                                             sliders reading an em dash)
    ALL OFF, beam armed   nothing revealed, but a note keeps the fact visible
    routed, beam off      revealed, and the contradiction stated
    routed, armed, power  revealed, EMITTING
"""

from __future__ import annotations

from pathlib import Path

JS = Path(__file__).resolve().parents[1] / "gently" / "ui" / "web" / "static" / "js"
CSS = Path(__file__).resolve().parents[1] / "gently" / "ui" / "web" / "static" / "css"


def _light() -> str:
    return (JS / "panels" / "light.js").read_text(encoding="utf-8")


def test_display_asks_a_narrower_question_than_emission() -> None:
    """`routedLines` is for rendering; `wavelengthsOf` is for safety.

    `wavelengthsOf(null)` returns every line, which is right for `emitting()` —
    an unread config with unread power must come out unknown, not safe. It is
    wrong for the panel, which would render four disabled sliders saying
    nothing.
    """
    src = _light()
    assert "function routedLines(s)" in src
    body = src[src.index("function routedLines(s)") :]
    body = body[: body.index("\n    }")]
    assert "s.config ?" in body, "routedLines no longer distinguishes unknown from empty"

    # emitting() must keep the safe answer
    em = src[src.index("function emitting(s)") :]
    em = em[: em.index("\n    }")]
    assert "wavelengthsOf(s.config)" in em


def test_the_detail_is_revealed_not_always_present() -> None:
    src = _light()
    assert "lines.length ? laserDetail(" in src, (
        "the laser detail is no longer conditional — an unknown config renders empty controls again"
    )


def test_the_contradiction_is_stated_rather_than_inferred() -> None:
    """Routed lines with the beam off is the most misleading state possible."""
    src = _light()
    body = src[src.index("function laserDetail(") :]
    body = body[: body.index("\n    }")]
    assert "contradicts" in body
    assert "will not emit" in body
    # Name the cause: it is the state every acquisition leaves behind.
    assert "volume acquisition" in body


def test_an_armed_beam_is_never_hidden_by_the_disclosure() -> None:
    """Armed with nothing routed is safe, surprising, and the resting state.

    Hiding the detail must not hide the fact, or the disclosure would have made
    the panel less honest than the flat version it replaced.
    """
    src = _light()
    assert "function idleBeamNote(" in src
    body = src[src.index("function idleBeamNote(") :]
    body = body[: body.index("\n    }")]
    assert "armed" in body
    assert "nothing emits" in body


def test_the_config_select_shows_what_is_actually_set() -> None:
    """The preset list 503s without a device layer; the read-back config does not.

    Otherwise the select reads an em dash while the detail below it shows routed
    lines and live power — the two halves of one panel contradicting each other.
    """
    src = _light()
    assert "function configOptions(current)" in src
    body = src[src.index("function configOptions(current)") :]
    body = body[: body.index("\n    }")]
    assert "opts.includes(current)" in body


def test_the_device_layer_buttons_do_not_borrow_marking_classes() -> None:
    """They did, and deleting the marking surface took their styling with it.

    The liveness check behind that deletion used a regex requiring a character
    before "marking", so class names STARTING with it were never seen as live.
    `.marking-action-btn` was in use by this strip and its rules were removed.
    """
    html = (
        Path(__file__).resolve().parents[1] / "gently" / "ui" / "web" / "templates" / "index.html"
    ).read_text(encoding="utf-8")
    assert "marking-action-btn" not in html
    assert "marking-done-btn" not in html

    css = (CSS / "main.css").read_text(encoding="utf-8")
    assert ".devices-layer-btn" in css, "the device-layer buttons have no styling"
    assert ".devices-layer-btn.is-primary" in css, "Start lost its primary treatment"
