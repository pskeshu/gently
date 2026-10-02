"""The agent asks in one place, and that place follows the operator.

A question used to render twice: as a card at the top of the workspace and as
a faint pointer in the chat that said "answer above" while the card was in a
different column. The eye is in the chat, because that is where the operator
just typed, so the question was missed and the agent waited in silence.

Now the card is the sticky slot above the composer. While the panel is
collapsed a header beacon says what is being waited for, and clicking it
shows the same card over the workspace, on demand. Nothing times out and
nothing is answered by default: the waiting is made visible, not resolved.

Pinned here because CI runs no browser.
"""

from __future__ import annotations

from pathlib import Path

WEB = Path(__file__).resolve().parents[1] / "gently" / "ui" / "web"
INDEX = (WEB / "templates" / "index.html").read_text(encoding="utf-8")
HEADER = (WEB / "templates" / "_header.html").read_text(encoding="utf-8")
CHAT = (WEB / "static" / "js" / "agent-chat.js").read_text(encoding="utf-8")
OVERLAY = (WEB / "static" / "js" / "ask-overlay.js").read_text(encoding="utf-8")
CSS = (WEB / "static" / "css" / "ask-overlay.css").read_text(encoding="utf-8")


def _fn(src: str, name: str) -> str:
    body = src[src.index(f"function {name}(") :]
    return body[: body.index("\n    }")]


def test_the_card_lives_above_the_composer_and_nowhere_on_the_page() -> None:
    assert 'id="ask-stage"' not in INDEX, "the page-top stage is back"
    body = _fn(CHAT, "renderChoice")
    assert "pendingSlot.appendChild(card)" in body, (
        "the card no longer goes to the slot above the composer"
    )
    assert "ac-ask-pointer" not in body and "answer above" not in CHAT, (
        "the chat shows a pointer instead of the card again"
    )


def test_the_beacon_shows_only_while_a_question_waits_and_the_panel_is_collapsed() -> None:
    assert 'id="ask-beacon"' in HEADER
    assert "const collapsed = pending && !panelOpen();" in OVERLAY
    assert "beacon.classList.toggle('hidden', !collapsed);" in OVERLAY
    # The panel tells the overlay when it opens or closes, and can be asked.
    assert "ClientEventBus.emit('AGENT_PANEL', { open: panelOpen });" in _fn(CHAT, "togglePanel")
    assert "isPanelOpen: () => panelOpen" in CHAT
    assert "ClientEventBus.on('AGENT_PANEL', render);" in OVERLAY


def test_the_overlay_is_on_demand_and_dismisses_to_the_beacon() -> None:
    assert 'id="ask-overlay"' in INDEX and 'id="ask-overlay-card"' in INDEX
    assert "const show = collapsed && wanted;" in OVERLAY
    # Arrival never opens it; the beacon does.
    assert "wanted = false;   // never opens by itself" in OVERLAY
    assert "beacon.addEventListener('click', () => { wanted = true; render(); });" in OVERLAY
    # Esc, the ×, and the backdrop put it away; the question stays pending.
    assert "e.key === 'Escape'" in OVERLAY and "dismiss()" in OVERLAY
    assert INDEX.count("data-dismiss") == 2, "both the × and the backdrop dismiss"
    # The way into the chat is on the card.
    assert "AgentChat.togglePanel(true)" in OVERLAY
    # It dims the workspace but sits under toasts.
    assert "z-index: 5000" in CSS


def test_no_default_and_no_timeout_answers_for_the_operator() -> None:
    assert "setTimeout" not in _fn(CHAT, "answerChoice")
    # The only timer in the overlay changes the tab title; it answers nothing.
    assert OVERLAY.count("setTimeout(") == 1
    assert (
        "answerChoice"
        not in OVERLAY[
            OVERLAY.index("titleTimer = setTimeout(") : OVERLAY.index("}, TITLE_DELAY_MS);")
        ]
    )


def test_an_answer_keeps_its_record_in_the_transcript() -> None:
    assert "keepTheRecord(reqId, selected);" in _fn(CHAT, "answerChoice")
    body = _fn(CHAT, "keepTheRecord")
    assert "log.appendChild(card);" in body, "the answered card is dropped instead of kept"
    assert "classList.add('ac-choice-picked')" in body, "the pick is not marked"
    assert "btn.dataset.optId = String(opt.id);" in _fn(CHAT, "buildAskCard")


def test_the_tab_title_says_so_after_a_moment_and_is_restored() -> None:
    assert "document.title = ASKING_TITLE;" in OVERLAY
    assert "document.title = savedTitle;" in OVERLAY
    assert "const TITLE_DELAY_MS = 3000;" in OVERLAY
