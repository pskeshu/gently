"""Four one-line fixes from the rc1 issue triage, pinned because CI runs no browser.

Each was a sentence in an issue from the August walkthroughs whose fix was one
line, and each stayed open for two months because nothing would have failed if
the line was never written — or if it is reverted. These tests fail then.

- #117 Stop on the device layer blocked the event loop for the whole grace
  period, so the status polls stalled and the button looked dead.
- #140 A roster push that no longer carried the selected embryo moved the
  selection to embryo 1 with nobody touching the list.
- #133 One close of the agent panel was remembered, so "open by default" was
  false on that machine for good.
- #131 The agent's question card was in normal flow, so asking a question
  pushed the instrument's controls down and off the panel. (The stage this
  first fixed was then replaced by the slot and the overlay; see
  test_the_agent_asks_in_one_place.py.)
"""

from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
WEB = ROOT / "gently" / "ui" / "web"
ROUTES = WEB / "routes" / "device_layer.py"
OPERATE = WEB / "static" / "js" / "operate.js"
CHAT = WEB / "static" / "js" / "agent-chat.js"


def _js_function(src: str, name: str) -> str:
    body = src[src.index(f"function {name}(") :]
    return body[: body.index("\n    }")]


def test_stopping_the_device_layer_is_kept_off_the_event_loop() -> None:
    """#117: the handler awaits stop() in a thread, as /api/shutdown already did."""
    src = ROUTES.read_text(encoding="utf-8")
    handler = src[src.index("async def device_layer_stop(") :]
    handler = handler[: handler.index("@router.")]
    assert "await asyncio.to_thread(sup.stop," in handler, (
        "device_layer_stop calls sup.stop() on the event loop again; it blocks in "
        "proc.wait() for the grace period and every other request stalls with it"
    )
    assert re.search(r"return\s+sup\.stop\(", src) is None


def test_a_selection_is_only_picked_for_the_operator_on_first_population() -> None:
    """#140: a push that dropped the selected embryo leaves none selected."""
    src = OPERATE.read_text(encoding="utf-8")
    upd = _js_function(src, "onEmbryosUpdate")
    assert "const hadAny = _embryos.length > 0;" in upd
    assert "if (!_selected && !hadAny && _embryos.length) _selected = _embryos[0].id;" in upd
    assert "if (!_selected && _embryos.length) _selected = _embryos[0].id;" not in src, (
        "the unguarded auto-select is back: a server push hops the selection"
    )
    # Removing the selected embryo yourself is the same: the pick is gone, not moved.
    assert "_selected = _embryos.length ? _embryos[0].id : null;" not in src


def test_the_agent_panel_opens_on_every_load() -> None:
    """#133: the collapse is for the session; nothing remembers it."""
    src = CHAT.read_text(encoding="utf-8")
    assert "gently-chat-open" not in src, (
        "the panel's collapse state is persisted again; one close on the scope PC "
        "defeats 'open by default' for good"
    )
    assert "togglePanel(true);" in _js_function(src, "restorePrefs")
