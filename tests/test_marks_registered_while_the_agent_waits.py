"""Random testing of 2026-10-21, session ce91600d, from a project's plan item.

"Start imaging for plan item b8f48d34": the agent detected four embryos and
put them on the map; the operator pressed the pane's own Register, which
filed them as `unassigned` and answered the agent with an empty list; the tool
said "0 embryo(s)"; the agent detected again. Along the way the agent asked
for a plan's status by the item's id and was told no such campaign, and the
pane's LED command at boot left a traceback in the log.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import numpy as np
from fastapi import FastAPI
from fastapi.testclient import TestClient

from gently.app.tools import detection_tools
from gently.harness.plan_mode.tools import planning
from gently.ui.web import auth
from gently.ui.web.routes.data import create_router

_JS = Path(__file__).resolve().parents[1] / "gently" / "ui" / "web" / "static" / "js"
OPERATE_JS = (_JS / "operate.js").read_text(encoding="utf-8")
CHAT_JS = (_JS / "agent-chat.js").read_text(encoding="utf-8")


def _fn(src: str, name: str, n: int = 2400) -> str:
    return src[src.index(name) :][:n]


# ── the pane ────────────────────────────────────────────────────────────────


def test_register_while_the_agent_waits_is_the_answer_to_the_agent():
    body = _fn(OPERATE_JS, "async function confirmMarks() {")
    guard = body.index("if (_markSession) { finishMarkingSession(); return; }")
    post = body.index("postJSON('/api/devices/embryos/confirm'")
    assert guard < post, "the pane's own Register ran first and the agent got nothing"


# ── the tool ────────────────────────────────────────────────────────────────


def _agent_and_client():
    agent = MagicMock()
    agent.experiment.embryos = {}
    agent.viz_server = object()
    client = MagicMock()
    client.has_sam = True
    client.detect_embryos = AsyncMock(
        return_value={
            "success": True,
            "embryos": [{"pixel_x": 4, "pixel_y": 4, "confidence": 0.9, "embryo_id": "embryo_1"}],
            "image": np.zeros((8, 8), dtype=np.uint8),
            "stage_position": [0.0, 0.0],
        }
    )
    return agent, client


def test_an_empty_answer_names_the_embryos_the_pane_registered_meanwhile(monkeypatch):
    agent, client = _agent_and_client()

    async def the_operator_pressed_register(**_kw):
        agent.experiment.embryos["embryo_1"] = MagicMock(role="unassigned")
        agent.experiment.embryos["embryo_2"] = MagicMock(role="unassigned")
        return [], None

    monkeypatch.setattr(detection_tools, "_route_to_map_view", the_operator_pressed_register)
    fn = getattr(detection_tools.detect_embryos, "__wrapped__", detection_tools.detect_embryos)
    msg = asyncio.run(fn(context={"agent": agent, "client": client}))
    assert "0 embryo" not in msg
    assert "registered from the pane" in msg and "embryo_1, embryo_2" in msg
    assert "do not detect again" in msg


def test_an_empty_answer_with_nothing_registered_is_still_zero(monkeypatch):
    agent, client = _agent_and_client()

    async def nothing(**_kw):
        return [], None

    monkeypatch.setattr(detection_tools, "_route_to_map_view", nothing)
    fn = getattr(detection_tools.detect_embryos, "__wrapped__", detection_tools.detect_embryos)
    msg = asyncio.run(fn(context={"agent": agent, "client": client}))
    assert msg.startswith("Detection complete: 0 embryo(s)")


# ── the plan ────────────────────────────────────────────────────────────────


def _store():
    campaign = SimpleNamespace(id="092fef99-full", description="brightfield datasets")
    item = SimpleNamespace(id="b8f48d34-full", campaign_id="092fef99-full")
    store = MagicMock()
    store.resolve_campaign = lambda ref: campaign if ref in ("092fef99", "092fef99-full") else None
    store.resolve_plan_item = lambda ref, campaign_id=None: item if ref == "b8f48d34" else None
    store.get_plan_status = lambda cid: {
        "completed": 0,
        "total": 1,
        "in_progress": 0,
        "by_type": {"imaging": {"completed": 0, "total": 1}},
        "next_actions": [],
        "pending_decisions": [],
    }
    return store


def test_a_plan_items_reference_answers_for_its_campaign():
    agent = MagicMock()
    agent.context_store = _store()
    fn = getattr(planning.get_plan_status, "__wrapped__", planning.get_plan_status)
    msg = asyncio.run(fn("b8f48d34", context={"agent": agent}))
    assert msg.startswith("Plan: brightfield datasets")
    assert "not found" not in msg


def test_a_reference_that_is_neither_says_so():
    agent = MagicMock()
    agent.context_store = _store()
    fn = getattr(planning.get_plan_status, "__wrapped__", planning.get_plan_status)
    msg = asyncio.run(fn("nothing-here", context={"agent": agent}))
    assert "neither a campaign nor a plan item" in msg


# ── the LED at boot ─────────────────────────────────────────────────────────


def _app(client):
    server = MagicMock()
    server.agent_bridge.agent.client = client
    app = FastAPI()
    app.include_router(create_router(server))
    app.dependency_overrides[auth.require_control] = lambda: True
    return TestClient(app)


def test_the_led_before_the_device_layer_is_a_503_not_a_traceback():
    client = MagicMock()
    client.is_connected = False
    client.set_led = AsyncMock()
    r = _app(client).post("/api/devices/led/set", json={"state": "Closed"})
    assert r.status_code == 503
    assert "not connected" in r.json()["detail"].lower()
    client.set_led.assert_not_awaited()


# ── the chat ────────────────────────────────────────────────────────────────


def test_the_agents_marking_request_shows_up_in_the_chat_with_its_buttons():
    """ "i suppose the register button can show up in the chat window -- it was
    hard to notice that." The request lands under the tool that asked, with
    Open the map and Register; the count follows the map; the answer goes
    through the pane's own Done so there is still one door."""
    assert "ClientEventBus.on('MARKING_IMAGE', (d) => renderMarkingCard(d));" in CHAT_JS
    assert "SharedState.on('marking', updateMarkingCard)" in CHAT_JS
    card = _fn(CHAT_JS, "function renderMarkingCard(d) {", 2200)
    assert "OperateManager.marking.done()" in card
    assert "OperateManager.roster.goTo('bottom')" in card
    assert "querySelectorAll('.ac-tool-running')" in card, "the card sits under the tool that asked"
    done = _fn(CHAT_JS, "function updateMarkingCard(s) {", 1200)
    assert "Marks sent to the agent." in done
