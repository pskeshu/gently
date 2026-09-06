"""There is one marking surface, invoked two ways.

There were two complete implementations. `static/js/marking.js` — 452 lines
with its own canvas, its own hit-test, its own marker list — lived in the
Embryos tab behind a "Marking" subtab, and `websocket.js` switched the operator
into it when the agent sent a `marking_image` frame. `operate.js` had the
other, on the bottom-camera pane, for an operator marking unprompted.

So every improvement to marking landed on exactly one of them. The zoom, the
display range, #105's hit-test radius and #126's roster aliasing were all on
the Operate side; none of it existed on the agent's side. Whether an operator
got the corrected behaviour depended on who had asked them to mark.

The agent's request now lands on the Operate pane. The pushed image is adapted
into the payload shape a live camera frame already has, so the existing
geometry applies to it unchanged.

WHAT MUST NOT BREAK

The contract is a websocket request/response, not an HTTP call — the agent
blocks on `session["complete"]` in `routes/websocket.py` and reads a role per
marker out of the answer. So the reply must carry `marking_done` with the
session id and pixel coordinates, and per-marker roles must remain settable
before it is sent.
"""

from __future__ import annotations

from pathlib import Path

WEB = Path(__file__).resolve().parents[1] / "gently" / "ui" / "web"
JS = WEB / "static" / "js"
INDEX = WEB / "templates" / "index.html"


def test_the_second_implementation_is_gone() -> None:
    assert not (JS / "marking.js").exists(), (
        "static/js/marking.js is back — a second marking surface means every "
        "fix lands on one of two, and which one an operator gets depends on who "
        "asked them to mark"
    )
    html = INDEX.read_text(encoding="utf-8")
    assert "MarkingManager" not in html
    assert 'id="embryos-marking"' not in html
    # The old path specifically. `panels/marking.js` is the panel and stays —
    # the two files having near-identical names was itself part of the mess.
    assert "/static/js/marking.js" not in html


def test_the_agent_request_reaches_the_operate_surface() -> None:
    ws = (JS / "websocket.js").read_text(encoding="utf-8")
    assert "MARKING_IMAGE" in ws, "marking_image no longer reaches any surface"
    assert "MarkingManager" not in ws

    src = (JS / "operate.js").read_text(encoding="utf-8")
    assert "ClientEventBus.on('MARKING_IMAGE'" in src
    assert "function onMarkingImage(d)" in src


def test_the_pushed_image_is_adapted_not_special_cased() -> None:
    """It becomes a frame, so the existing geometry applies to it unchanged.

    That is the whole point: the agent's request inherits the zoom, the
    corrected hit-test and the display range because it is the same surface,
    not because any of them were reimplemented.
    """
    src = (JS / "operate.js").read_text(encoding="utf-8")
    body = src[src.index("function onMarkingImage(d)") :]
    body = body[: body.index("\n    }")]
    # The live-frame payload shape: shape/downsample/stage_position/jpeg_b64.
    for key in ("shape:", "downsample:", "stage_position:", "jpeg_b64:"):
        assert key in body, f"the adapted frame is missing {key}"
    # PNG, not JPEG — start_marking_session sends PNG.
    assert "image/png" in body


def test_the_session_declares_its_own_scale() -> None:
    """`pixel_size_um` is a session parameter, not the rig default."""
    src = (JS / "operate.js").read_text(encoding="utf-8")
    assert "function pxBase()" in src
    assert "pixelSizeUm" in src
    # Threaded into every geometry call, or a session with a different scale
    # would silently place markers wrong.
    assert src.count("pxBase()") >= 5, (
        "pxBase is not threaded through the geometry — a session declaring a "
        "different µm/px would place markers at the rig default instead"
    )


def test_the_answer_keeps_the_contract() -> None:
    src = (JS / "operate.js").read_text(encoding="utf-8")
    body = src[src.index("function finishMarkingSession()") :]
    body = body[: body.index("\n    }")]
    assert "'marking_done'" in body
    assert "session_id" in body
    # Back out to pixels: the server and the agent speak pixel coordinates.
    assert "stageToFrame" in body
    for field in ("number:", "pixelX:", "pixelY:", "role:", "source:"):
        assert field in body, f"the answer is missing {field}"


def test_per_marker_roles_stay_settable() -> None:
    """`marking_done` carries a role per marker and the agent reads them.

    Registered embryos get roles in the Acquisition roster; these are not
    registered yet, so the session needs its own way to say which is a
    reference.
    """
    src = (JS / "operate.js").read_text(encoding="utf-8")
    assert "function cycleMarkerRole(index)" in src
    assert "cycleRole:" in src, "the panel cannot reach the role toggle"

    panel = (JS / "panels" / "marking.js").read_text(encoding="utf-8")
    assert 'data-act="cycleRole"' in panel
    assert "data-index=" in panel


def test_the_session_ui_is_present_only_while_the_agent_waits() -> None:
    """A standing "the agent is waiting" panel would be a lie most of the time."""
    panel = (JS / "panels" / "marking.js").read_text(encoding="utf-8")
    body = panel[panel.index("function session(s)") :]
    body = body[: body.index("\n    }")]
    assert "if (!s.session) return ''" in body
