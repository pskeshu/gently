"""
Agent WebSocket route — /ws/agent

Provides the TUI ↔ agent bridge. The Ink TUI connects here
to send chat messages and receive streaming responses, tool calls,
choice pickers, command results, and notifications.
"""

import asyncio
import json
import logging
from collections.abc import Callable
from datetime import datetime

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from gently.settings import settings

logger = logging.getLogger(__name__)


def create_router(server) -> APIRouter:
    """Create the /ws/agent router.

    Parameters
    ----------
    server : VisualizationServer
        Must have a ``agent_bridge`` attribute (set by launch_gently.py
        after the bridge is created).
    """
    router = APIRouter()

    # Pending choice futures keyed by request_id
    _choice_futures: dict[str, asyncio.Future] = {}

    # ── Single-driver control arbitration ─────────────────────
    # Shared across all /ws/agent clients (the router is created once).
    # Only the control holder may drive the agent (chat/command/cancel);
    # everyone else is an observer until they take control. This is the
    # seed of the multi-user control lock and also prevents the shared
    # agent conversation from being corrupted when >1 client connects.
    _control: dict[str, str | None] = {"holder": None}
    _clients: dict[str, Callable] = {}
    _client_labels: dict[str, str] = {}
    _client_counter = {"n": 0}
    _raw_clients: dict[str, WebSocket] = {}  # client_id -> websocket (broadcast)

    # ── Uniform display transcript ────────────────────────────
    # A single conversation history shared by every client of this session.
    # Persisted to <session>/chat_display.json so it survives reconnects and
    # restarts; broadcast live so all instances stay in sync.
    _history: list = []
    _history_state = {"sid": None, "path": None, "agent_buf": None, "autonomous": False}

    async def _broadcast_control_status():
        """Tell every connected agent client who currently holds control."""
        holder = _control["holder"]
        holder_label = _client_labels.get(holder) if holder else None
        for cid, fn in list(_clients.items()):
            try:
                await fn(
                    {
                        "type": "control_status",
                        "holder": holder,
                        "holder_label": holder_label,
                        "you_have_control": (cid == holder),
                    }
                )
            except Exception:
                pass

    def _load_history_for_session(bridge):
        """Load the current session's display history, reloading if the
        session changed (e.g. after a resume from the Sessions tab)."""
        try:
            agent = bridge.agent
            store = getattr(agent, "store", None)
            sid = getattr(agent, "session_id", None)
        except Exception:
            return
        if sid == _history_state["sid"]:
            return  # already loaded for this session
        # Session changed (or first load): reset and reload from disk.
        _history.clear()
        _history_state["sid"] = sid
        _history_state["path"] = None
        _history_state["agent_buf"] = None
        _history_state["autonomous"] = False
        try:
            if store and sid:
                sdir = store._session_dir(sid)
                if sdir:
                    p = sdir / "chat_display.json"
                    _history_state["path"] = p
                    if p.exists():
                        loaded = json.loads(p.read_text(encoding="utf-8")) or []
                        if isinstance(loaded, list):
                            _history.extend(loaded)
        except Exception:
            logger.debug("Could not load chat history", exc_info=True)

        # Fallback: sessions created before chat_display.json existed (or any
        # session resumed for the first time) — derive a best-effort transcript
        # from the saved Claude conversation so the chat still shows history.
        if not _history and store and sid:
            try:
                snap = store.load_session_snapshot(sid) or {}
                for m in snap.get("conversation_history") or []:
                    role = m.get("role")
                    content = m.get("content")
                    if isinstance(content, list):
                        text = "".join(
                            b.get("text", "")
                            for b in content
                            if isinstance(b, dict) and b.get("type") == "text"
                        )
                    else:
                        text = content if isinstance(content, str) else ""
                    text = (text or "").strip()
                    if not text:
                        continue
                    if role == "user":
                        _history.append({"role": "user", "text": text})
                    elif role == "assistant":
                        _history.append({"role": "agent", "text": text})
            except Exception:
                logger.debug("Could not derive history from conversation", exc_info=True)

    def _save_history():
        p = _history_state["path"]
        if not p:
            return
        try:
            tmp = p.with_suffix(".json.tmp")
            tmp.write_text(json.dumps(_history[-500:]), encoding="utf-8")
            tmp.replace(p)
        except Exception:
            pass

    def _record(item):
        _history.append(item)
        if len(_history) > 500:
            del _history[: len(_history) - 500]
        _save_history()

    def _flush_agent_buf():
        buf = _history_state["agent_buf"]
        if buf:
            # An autonomous (wake) turn's text is recorded distinctly so replay
            # shows it as "Gently · autonomous", not an ordinary agent reply.
            role = "autonomous" if _history_state.get("autonomous") else "agent"
            _record({"role": role, "text": buf})
        _history_state["agent_buf"] = None

    def _record_display(msg):
        """Fold a streamed chunk into the persistent display history."""
        t = msg.get("type")
        if t == "user_message":
            _flush_agent_buf()
            _history_state["autonomous"] = False
            _record(
                {
                    "role": "user",
                    "text": msg.get("text", ""),
                    "author": msg.get("author"),
                    "author_id": msg.get("author_id"),
                }
            )
        elif t == "autonomous_start":
            # An autonomous wake turn is beginning — record the trigger banner
            # and mark following text as autonomous until stream_end.
            _flush_agent_buf()
            _history_state["autonomous"] = True
            _record({"role": "autonomous_start", "trigger": msg.get("trigger", "")})
        elif t == "text":
            _history_state["agent_buf"] = (_history_state["agent_buf"] or "") + msg.get("text", "")
        elif t == "tool_call":
            _flush_agent_buf()
            _record(
                {
                    "role": "tool",
                    "name": msg.get("tool_name"),
                    "duration": msg.get("duration"),
                    "summary": msg.get("result_summary"),
                }
            )
        elif t == "stream_end":
            _flush_agent_buf()
            _history_state["autonomous"] = False

    async def _broadcast(msg):
        """Record to history + send a display message to ALL clients."""
        _record_display(msg)
        for _cid, ws in list(_raw_clients.items()):
            try:
                await ws.send_json(msg)
            except Exception:
                pass

    async def _run_wizard(
        wizard, websocket, send_fn, _choice_futures, bridge=None, log_transcript=None
    ):
        """Run the wizard's interactive loop.

        Returns the wizard task so callers can check for exceptions.
        Used both at startup and for the /wizard command.
        """
        _wizard_input_future: asyncio.Future | None = None

        async def _wizard_wait_for_input() -> str:
            nonlocal _wizard_input_future
            loop = asyncio.get_event_loop()
            _wizard_input_future = loop.create_future()
            return await _wizard_input_future

        async def _wizard_wait_for_choice(choice_data: dict) -> str:
            request_id = _make_request_id()
            choice_data["request_id"] = request_id
            await send_fn(
                {
                    "type": "choice_request",
                    "choice_data": choice_data,
                    "request_id": request_id,
                }
            )
            loop = asyncio.get_event_loop()
            future = loop.create_future()
            _choice_futures[request_id] = future
            return await future

        wizard_task = asyncio.create_task(
            wizard.run(send_fn, _wizard_wait_for_input, _wizard_wait_for_choice)
        )

        while not wizard_task.done():
            try:
                raw = await asyncio.wait_for(
                    websocket.receive_text(),
                    timeout=60.0,
                )
            except asyncio.TimeoutError:
                await websocket.send_json({"type": "ping"})
                continue

            try:
                data = json.loads(raw)
            except json.JSONDecodeError:
                continue

            if log_transcript:
                log_transcript("in", data)

            msg_type = data.get("type")

            if msg_type == "chat":
                text = data.get("text", "").strip()
                if text and _wizard_input_future and not _wizard_input_future.done():
                    _wizard_input_future.set_result(text)

            elif msg_type == "choice_response":
                request_id = data.get("request_id", "")
                selected = data.get("selected", "")
                future = _choice_futures.pop(request_id, None)
                if future and not future.done():
                    future.set_result(selected)

            elif msg_type == "command":
                # Forward commands even during wizard (e.g. /reset-context)
                command = data.get("command", "").strip()
                if command and bridge is not None:
                    await bridge.handle_command(command, send_fn)
                    # /reset-context kills the wizard — context is gone
                    if command.strip().lower() == "/reset-context":
                        wizard_task.cancel()
                        await send_fn(
                            {
                                "type": "stream_end",
                                "tokens": {
                                    "input_tokens": 0,
                                    "output_tokens": 0,
                                    "total_tokens": 0,
                                    "api_calls": 0,
                                },
                                "wizard_complete": True,
                            }
                        )
                        return wizard_task

            elif msg_type == "ping":
                await websocket.send_json({"type": "pong"})

            elif msg_type == "pong":
                pass

        return wizard_task

    def _handle_wizard_result(wizard_task):
        """Check wizard task for errors, return exception or None."""
        if wizard_task.done() and not wizard_task.cancelled():
            return wizard_task.exception()
        return None

    @router.websocket("/ws/agent")
    async def agent_websocket(websocket: WebSocket):
        await websocket.accept()

        bridge = getattr(server, "agent_bridge", None)
        if bridge is None:
            await websocket.send_json(
                {
                    "type": "error",
                    "error": "Agent bridge not initialized",
                }
            )
            await websocket.close()
            return

        # Route autonomous (wake-router) turns through this router's _broadcast so
        # they stream to all chat clients + persist to the display transcript.
        # Idempotent; _broadcast is router-scoped and fans out to whoever is live.
        bridge.register_display_broadcaster(_broadcast)

        # ── Authenticate the connection (account mode) ────────────
        # When user accounts are configured, identity comes from the signed
        # session cookie (set at login). Viewers may watch but not drive;
        # operators/admins may take the control lock. With no accounts
        # configured we fall back to the legacy "anyone connected can drive".
        from gently.ui.web.accounts import CONTROL_ROLES, get_account_store
        from gently.ui.web.auth import SESSION_COOKIE

        _acct = get_account_store()
        username = None
        can_control = True  # legacy default when no accounts are configured
        if _acct is not None and _acct.has_users():
            # Viewing is open: anonymous clients may connect and *watch* the
            # conversation. Only authenticated operators/admins can hold or
            # take the control lock (enforced on the drive actions below).
            _token = websocket.cookies.get(SESSION_COOKIE)
            username = _acct.verify_session(_token) if _token else None
            role = _acct.get_role(username) if username else None
            can_control = role in CONTROL_ROLES

        # Assign a stable id for control arbitration. The label shown to other
        # clients is the username when authenticated, else "Anonymous". The UI
        # renders "You" for the viewer's own messages by matching client_id, so
        # anonymous participants don't need disambiguating numbers.
        _client_counter["n"] += 1
        client_id = f"agent_client_{_client_counter['n']}"
        client_label = username or "Anonymous"

        # Send connection metadata (version, tokens, embryo count, commands).
        # you_id lets the client label its own messages "You".
        meta = bridge.get_connect_metadata()
        _connected_msg = {
            "type": "connected",
            **meta,
            "you_id": client_id,
            "timestamp": datetime.now().isoformat(),
        }
        await websocket.send_json(_connected_msg)

        # Subscribe to mesh peer events so the TUI gets live peer counts
        _mesh_unsubs = []
        mesh_svc = getattr(server, "mesh_service", None)
        if mesh_svc is not None and server.event_bus is not None:
            from gently.core.event_bus import EventType as _ET

            async def _push_peer_count(event):
                try:
                    msg = {
                        "type": "state_update",
                        "state": {"peer_count": mesh_svc.peer_count},
                    }
                    _log_transcript("out", msg)
                    await websocket.send_json(msg)
                except Exception:
                    pass

            async def _push_peer_discovered(event):
                try:
                    is_trusted = event.data.get("is_trusted", True)
                    hostname = event.data.get("hostname", "unknown")
                    msg = {
                        "type": "state_update",
                        "state": {"peer_count": mesh_svc.peer_count},
                    }
                    _log_transcript("out", msg)
                    await websocket.send_json(msg)
                    if is_trusted:
                        msg = {
                            "type": "notification",
                            "level": "info",
                            "title": f"Peer joined: {hostname}",
                        }
                        _log_transcript("out", msg)
                        await websocket.send_json(msg)
                    else:
                        # Interactive prompt for unpaired peers
                        request_id = f"mesh_discover:{hostname}"
                        msg = {
                            "type": "choice_request",
                            "choice_data": {
                                "_type": "single",
                                "question": f"New peer discovered: {hostname}",
                                "options": [
                                    {
                                        "id": "pair",
                                        "label": "Pair",
                                        "description": f"Start pairing with {hostname}",
                                    },
                                    {
                                        "id": "ignore",
                                        "label": "Ignore",
                                        "description": "Dismiss (you can pair later via /pair)",
                                    },
                                ],
                                "allow_multiple": False,
                            },
                            "request_id": request_id,
                        }
                        _log_transcript("out", msg)
                        await websocket.send_json(msg)
                        loop = asyncio.get_event_loop()
                        future = loop.create_future()
                        _choice_futures[request_id] = future
                        selected = await future
                        if selected == "pair":
                            await bridge.handle_command(f"/pair {hostname}", send_fn)
                except asyncio.CancelledError:
                    pass
                except Exception:
                    pass

            async def _push_peer_lost(event):
                try:
                    hostname = event.data.get("hostname", "unknown")
                    msg = {
                        "type": "state_update",
                        "state": {"peer_count": mesh_svc.peer_count},
                    }
                    _log_transcript("out", msg)
                    await websocket.send_json(msg)
                    msg = {
                        "type": "notification",
                        "level": "warning",
                        "title": f"Peer offline: {hostname}",
                    }
                    _log_transcript("out", msg)
                    await websocket.send_json(msg)
                except Exception:
                    pass

            async def _push_pairing_requested(event):
                try:
                    hostname = event.data.get("initiator_hostname", "unknown")
                    pin = event.data.get("pin", "??????")
                    pairing_id = event.data.get("pairing_id", "")

                    request_id = f"mesh_pair:{pairing_id}"
                    msg = {
                        "type": "choice_request",
                        "choice_data": {
                            "_type": "single",
                            "question": (
                                f"{hostname} wants to pair\nVerify this code matches: {pin}"
                            ),
                            "options": [
                                {
                                    "id": "accept",
                                    "label": "Accept pairing",
                                    "description": f"Trust {hostname} and allow mesh communication",
                                },
                                {
                                    "id": "reject",
                                    "label": "Reject",
                                    "description": "Decline this pairing request",
                                },
                            ],
                            "allow_multiple": False,
                        },
                        "request_id": request_id,
                    }
                    _log_transcript("out", msg)
                    await websocket.send_json(msg)

                    # Register future — REPL loop resolves it on choice_response
                    loop = asyncio.get_event_loop()
                    future = loop.create_future()
                    _choice_futures[request_id] = future

                    selected = await future
                    if selected == "accept":
                        await bridge.handle_command("/pair accept", send_fn)
                    else:
                        await bridge.handle_command("/pair reject", send_fn)
                except asyncio.CancelledError:
                    pass
                except Exception:
                    pass

            async def _push_pairing_completed(event):
                try:
                    hostname = event.data.get("peer_hostname", "unknown")
                    msg = {
                        "type": "notification",
                        "level": "success",
                        "title": f"Paired with {hostname}",
                    }
                    _log_transcript("out", msg)
                    await websocket.send_json(msg)
                except Exception:
                    pass

            async def _push_auth_failure(event):
                try:
                    ip = event.data.get("ip", "unknown")
                    msg = {
                        "type": "notification",
                        "level": "warning",
                        "title": "Auth failed",
                        "body": f"from {ip}",
                    }
                    _log_transcript("out", msg)
                    await websocket.send_json(msg)
                except Exception:
                    pass

            async def _push_cert_pin_failure(event):
                try:
                    peer_id = event.data.get("peer_id", "unknown")
                    msg = {
                        "type": "notification",
                        "level": "error",
                        "title": "Certificate mismatch",
                        "body": f"{peer_id} \u2014 possible MITM",
                    }
                    _log_transcript("out", msg)
                    await websocket.send_json(msg)
                except Exception:
                    pass

            async def _push_scope_denied(event):
                try:
                    peer_id = event.data.get("peer_id", "unknown")
                    scope = event.data.get("scope", "unknown")
                    msg = {
                        "type": "notification",
                        "level": "warning",
                        "title": "Access denied",
                        "body": f"{peer_id} missing scope: {scope}",
                    }
                    _log_transcript("out", msg)
                    await websocket.send_json(msg)
                except Exception:
                    pass

            # Peer discovery
            unsub = server.event_bus.subscribe_async(
                _ET.MESH_PEER_DISCOVERED, _push_peer_discovered
            )
            _mesh_unsubs.append(unsub)
            unsub = server.event_bus.subscribe_async(_ET.MESH_PEER_LOST, _push_peer_lost)
            _mesh_unsubs.append(unsub)
            unsub = server.event_bus.subscribe_async(_ET.MESH_PEER_UPDATED, _push_peer_count)
            _mesh_unsubs.append(unsub)

            # Pairing events
            unsub = server.event_bus.subscribe_async(
                _ET.MESH_PAIRING_REQUESTED, _push_pairing_requested
            )
            _mesh_unsubs.append(unsub)
            unsub = server.event_bus.subscribe_async(
                _ET.MESH_PAIRING_COMPLETED, _push_pairing_completed
            )
            _mesh_unsubs.append(unsub)

            # Security events
            unsub = server.event_bus.subscribe_async(_ET.MESH_AUTH_FAILURE, _push_auth_failure)
            _mesh_unsubs.append(unsub)
            unsub = server.event_bus.subscribe_async(
                _ET.MESH_CERT_PIN_FAILURE, _push_cert_pin_failure
            )
            _mesh_unsubs.append(unsub)
            unsub = server.event_bus.subscribe_async(_ET.MESH_SCOPE_DENIED, _push_scope_denied)
            _mesh_unsubs.append(unsub)

        # Active streaming task (so we can cancel on disconnect)
        active_task: asyncio.Task | None = None
        wizard_task = None
        bootstrap_task: asyncio.Task | None = None

        # ── Session transcript ────────────────────────────────
        # Log every WebSocket message (both directions) to a JSONL
        # file in the session directory. This captures the full
        # conversation as it appeared on screen — every text chunk,
        # tool call, choice picker, and user message.
        _transcript_file = None
        try:
            agent = bridge.agent
            store = getattr(agent, "store", None)
            sid = getattr(agent, "session_id", None)
            if store and sid:
                sdir = store._session_dir(sid)
                if sdir and sdir.exists():
                    _transcript_file = open(
                        sdir / "transcript.jsonl",
                        "a",
                        encoding="utf-8",
                    )
                    logger.info("Transcript logging to %s", sdir / "transcript.jsonl")
        except Exception as e:
            logger.debug("Could not open transcript file: %s", e)

        def _log_transcript(direction: str, data: dict):
            """Append a timestamped message to the transcript."""
            if _transcript_file is None:
                return
            try:
                entry = {
                    "ts": datetime.now().isoformat(),
                    "dir": direction,
                    **data,
                }
                _transcript_file.write(json.dumps(entry, default=str) + "\n")
                _transcript_file.flush()
            except Exception:
                pass

        # Retroactively log the connected message that was sent before
        # the transcript file was opened.
        _log_transcript("out", _connected_msg)

        async def send_fn(data: dict):
            """Send a JSON message to the TUI client."""
            _log_transcript("out", data)
            try:
                await websocket.send_json(data)
            except Exception:
                logger.debug("Failed to send to TUI client")

        def choice_future_factory(choice_data: dict) -> asyncio.Future:
            """Create a future for a choice request and send it to the client."""
            request_id = choice_data.get("request_id") or _make_request_id()
            # Write request_id back so the bridge can attach it to the chunk
            choice_data["request_id"] = request_id
            future = asyncio.get_event_loop().create_future()
            _choice_futures[request_id] = future
            return future

        def _discard_choice(request_id: str) -> None:
            _choice_futures.pop(request_id, None)

        # Give the bridge the choice-factory + discard too, so ASK-mode autonomous
        # turns can round-trip an approval picker through this connection's channel
        # and clean up the future on timeout/cancel.
        bridge.register_display_broadcaster(_broadcast, choice_future_factory, _discard_choice)

        # Register this client for control arbitration; grant control if free
        # (only to clients allowed to drive — viewers never auto-hold).
        _clients[client_id] = send_fn
        _client_labels[client_id] = client_label
        _raw_clients[client_id] = websocket
        if _control["holder"] is None and can_control:
            _control["holder"] = client_id
        await _broadcast_control_status()

        # Replay the uniform session transcript so every client (and every
        # reconnect/refresh) shows the same conversation.
        _load_history_for_session(bridge)
        if _history:
            try:
                await websocket.send_json({"type": "history", "items": list(_history)})
            except Exception:
                pass

        try:
            # ── Wizard phase ──────────────────────────────────────
            # The startup wizard no longer auto-pops in the chat — setup is now
            # launched on demand from the Home page (which sends /wizard) or via
            # the /wizard command. Re-enable auto-run by setting
            # server.wizard_autorun = True. NOTE: wizard_ran below is still
            # derived from wizard.needed, so the briefing/resolution path is
            # unaffected by this gate.
            wizard = getattr(bridge, "_wizard", None)
            if wizard is not None and wizard.needed and getattr(server, "wizard_autorun", False):
                wizard_task = await _run_wizard(
                    wizard,
                    websocket,
                    send_fn,
                    _choice_futures,
                    bridge,
                    log_transcript=_log_transcript,
                )
                exc = _handle_wizard_result(wizard_task)
                if exc:
                    logger.error(f"Wizard error: {exc}", exc_info=exc)
                    await send_fn(
                        {
                            "type": "stream_end",
                            "tokens": {
                                "input_tokens": 0,
                                "output_tokens": 0,
                                "total_tokens": 0,
                                "api_calls": 0,
                            },
                            "wizard_complete": True,
                        }
                    )

            # ── Auto-briefing or resolution picker ────────────────
            # New sessions with multiple unblocked imaging candidates
            # open into a structured resolution picker. The picker is
            # deterministic (no LLM call) and dispatches the user's
            # pick server-side. It runs concurrently with the REPL so
            # its awaited choice future can be resolved by incoming
            # ``choice_response`` messages. All other launches fall
            # back to the deterministic briefing text.
            wizard_ran = wizard is not None and wizard.needed

            async def _run_resolution_bootstrap():
                try:
                    await bridge.bootstrap_resolution_picker(
                        send_fn,
                        choice_future_factory,
                    )
                except asyncio.CancelledError:
                    raise
                except Exception as exc:
                    logger.error(
                        "Resolution picker failed; falling back to static briefing: %s",
                        exc,
                        exc_info=exc,
                    )
                    try:
                        briefing = bridge.get_session_briefing()
                        if briefing:
                            await send_fn({"type": "stream_start"})
                            await send_fn({"type": "text", "text": briefing})
                            await send_fn(
                                {
                                    "type": "stream_end",
                                    "tokens": {
                                        "input_tokens": 0,
                                        "output_tokens": 0,
                                        "total_tokens": 0,
                                        "api_calls": 0,
                                    },
                                }
                            )
                    except Exception:
                        pass

            if not wizard_ran:
                enter_resolution = bridge.should_enter_resolution()
                # Under ux_v2 the agent-first landing owns the session-entry
                # decision ("Plan an experiment" / "Take a quick look"), so the
                # legacy connect-time resolution picker would just duplicate it —
                # and contradict it, by offering "Standalone" after the user has
                # already chosen to plan. Stay quiet on connect for new sessions;
                # the landing drives plan-mode (/plan) or standalone instead.
                if enter_resolution and not settings.ui.ux_v2:
                    bootstrap_task = asyncio.create_task(_run_resolution_bootstrap())
                elif not enter_resolution:
                    # Resume / already-resolved sessions still get their briefing
                    # (it sits behind the landing overlay until dismissed).
                    briefing = bridge.get_session_briefing()
                    if briefing:
                        await send_fn({"type": "stream_start"})
                        await send_fn({"type": "text", "text": briefing})
                        await send_fn(
                            {
                                "type": "stream_end",
                                "tokens": {
                                    "input_tokens": 0,
                                    "output_tokens": 0,
                                    "total_tokens": 0,
                                    "api_calls": 0,
                                },
                            }
                        )

            # ── Main REPL loop ────────────────────────────────────
            while True:
                try:
                    raw = await asyncio.wait_for(
                        websocket.receive_text(),
                        timeout=60.0,
                    )
                except asyncio.TimeoutError:
                    await websocket.send_json({"type": "ping"})
                    continue

                try:
                    data = json.loads(raw)
                except json.JSONDecodeError:
                    logger.warning(f"Invalid JSON from TUI: {raw[:100]}")
                    continue

                _log_transcript("in", data)
                msg_type = data.get("type")

                # ── Control arbitration ───────────────────────────
                # A client requesting the wheel.
                if msg_type == "take_control":
                    if not can_control:
                        await send_fn(
                            {
                                "type": "notification",
                                "level": "warning",
                                "title": "View-only role",
                                "body": "Your account can watch but not control the microscope.",
                            }
                        )
                        await _broadcast_control_status()
                        continue
                    prev = _control["holder"]
                    _control["holder"] = client_id
                    if prev and prev != client_id and prev in _clients:
                        try:
                            await _clients[prev](
                                {
                                    "type": "notification",
                                    "level": "warning",
                                    "title": f"Control taken by {client_label}",
                                    "body": "You are now viewing.",
                                }
                            )
                        except Exception:
                            pass
                    await _broadcast_control_status()
                    continue

                # Only the holder may drive the agent. Observers are told
                # to take control rather than silently corrupting the
                # single shared conversation.
                if msg_type in ("chat", "command", "cancel") and client_id != _control["holder"]:
                    holder_label = _client_labels.get(_control["holder"] or "") or "another client"
                    await send_fn(
                        {
                            "type": "notification",
                            "level": "info",
                            "title": f"Viewing only — control is held by {holder_label}",
                            "body": "Take control to drive the microscope.",
                        }
                    )
                    await _broadcast_control_status()
                    continue

                if msg_type == "chat":
                    text = data.get("text", "").strip()
                    if not text:
                        continue

                    # Cancel any previous stream
                    if active_task and not active_task.done():
                        active_task.cancel()

                    # Echo the user's message to ALL clients (so observers see
                    # what was asked), then stream the reply to everyone. author
                    # is the display name (username or "Anonymous"); author_id
                    # lets each client render its own messages as "You".
                    await _broadcast(
                        {
                            "type": "user_message",
                            "text": text,
                            "author": client_label,
                            "author_id": client_id,
                        }
                    )
                    active_task = asyncio.create_task(
                        bridge.stream_response(text, _broadcast, choice_future_factory)
                    )

                elif msg_type == "choice_response":
                    # Only the control holder answers pickers (observers see
                    # them read-only).
                    if _control["holder"] != client_id:
                        continue
                    request_id = data.get("request_id", "")
                    selected = data.get("selected", "")
                    # Check if bridge owns this choice (e.g. /import-embryos picker)
                    if await bridge.handle_choice_response(request_id, selected, send_fn):
                        pass  # bridge handled it
                    else:
                        future = _choice_futures.pop(request_id, None)
                        if future and not future.done():
                            future.set_result(selected)

                elif msg_type == "cancel":
                    if active_task and not active_task.done():
                        active_task.cancel()
                        active_task = None
                        # A cancelled stream emits no stream_end of its own, so
                        # tell every client the turn is over — otherwise their
                        # "Working…" indicator spins forever after Stop.
                        await _broadcast({"type": "stream_end"})

                elif msg_type == "command":
                    command = data.get("command", "").strip()
                    if not command:
                        continue

                    # /wizard — run the wizard inline
                    if command.lower() in ("/wizard",):
                        w = getattr(bridge, "_wizard", None)
                        if w is None:
                            await send_fn(
                                {
                                    "type": "command_result",
                                    "command": "/wizard",
                                    "error": "Wizard not available",
                                }
                            )
                        else:
                            # Re-create wizard so it re-assesses gaps
                            cs = getattr(bridge, "_context_store", None)
                            cc = w.claude_client
                            bridge.init_wizard(cs, cc)
                            w = bridge._wizard

                            # Tell TUI we're entering wizard mode
                            await send_fn(
                                {
                                    "type": "command_result",
                                    "command": "/wizard",
                                    "content": {"wizard_active": True},
                                }
                            )

                            wizard_task = await _run_wizard(
                                w,
                                websocket,
                                send_fn,
                                _choice_futures,
                                bridge,
                                log_transcript=_log_transcript,
                            )
                            exc = _handle_wizard_result(wizard_task)
                            if exc:
                                logger.error(f"Wizard error: {exc}", exc_info=exc)
                                await send_fn(
                                    {
                                        "type": "stream_end",
                                        "tokens": {
                                            "input_tokens": 0,
                                            "output_tokens": 0,
                                            "total_tokens": 0,
                                            "api_calls": 0,
                                        },
                                        "wizard_complete": True,
                                    }
                                )
                    else:
                        try:
                            await bridge.handle_command(
                                command, send_fn, choice_futures=_choice_futures
                            )
                        except Exception as e:
                            logger.error("Command '%s' failed: %s", command, e, exc_info=True)
                            await send_fn(
                                {
                                    "type": "command_result",
                                    "command": command,
                                    "error": str(e),
                                }
                            )

                elif msg_type == "browse":
                    target = data.get("target", "")
                    await _handle_browse(
                        target,
                        data,
                        server,
                        bridge,
                        send_fn,
                    )

                elif msg_type == "ping":
                    await websocket.send_json({"type": "pong"})

                elif msg_type == "pong":
                    pass  # response to our ping

        except WebSocketDisconnect:
            logger.info("Agent websocket client disconnected")
        except asyncio.CancelledError:
            pass
        except Exception as e:
            logger.error(f"Agent WS error: {e}", exc_info=True)
        finally:
            # Close transcript file
            if _transcript_file is not None:
                try:
                    _transcript_file.close()
                except Exception:
                    pass
            # Unsubscribe from mesh events
            for unsub in _mesh_unsubs:
                try:
                    unsub()
                except Exception:
                    pass
            if wizard_task and not wizard_task.done():
                wizard_task.cancel()
            if active_task and not active_task.done():
                active_task.cancel()
                # This connection was mid-stream — persist whatever the agent
                # generated so far, otherwise a reload loses the in-progress
                # reply (it's only committed on stream_end). Guarded to the
                # owning connection so an observer's disconnect can't split a
                # still-streaming reply into two history entries.
                try:
                    _flush_agent_buf()
                except Exception:
                    logger.debug("Could not flush agent buffer on disconnect", exc_info=True)
            if bootstrap_task is not None and not bootstrap_task.done():
                bootstrap_task.cancel()
            # Release control arbitration for this client; hand the wheel
            # to any remaining client (or free it) and resync everyone.
            _clients.pop(client_id, None)
            _client_labels.pop(client_id, None)
            _raw_clients.pop(client_id, None)
            if _control["holder"] == client_id:
                _control["holder"] = next(iter(_clients), None)
                try:
                    await _broadcast_control_status()
                except Exception:
                    pass
            # Clean up pending futures only when the last client leaves —
            # otherwise we'd cancel another connected client's pending choices.
            if not _clients:
                for future in _choice_futures.values():
                    if not future.done():
                        future.cancel()
                _choice_futures.clear()

    return router


async def _handle_browse(target, data, server, bridge, send_fn):
    """Handle browse requests from the TUI browser panel."""
    try:
        if target == "campaigns":
            cs = getattr(server, "context_store", None)
            if cs is None:
                await send_fn({"type": "browse_result", "target": "campaigns", "data": []})
                return

            def serialize_campaign(c):
                status_val = c.status.value if hasattr(c.status, "value") else str(c.status)
                plan_status = cs.get_plan_status(c.id)
                subcampaigns = cs.get_subcampaigns(c.id)
                if subcampaigns:
                    children = [serialize_campaign(s) for s in subcampaigns]
                    items = []
                else:
                    children = []
                    items_raw = cs.get_plan_items(campaign_id=c.id)
                    items = [
                        {
                            "id": item.id,
                            "title": item.title,
                            "status": item.status.value
                            if hasattr(item.status, "value")
                            else str(item.status),
                            "type": item.type.value
                            if hasattr(item.type, "value")
                            else str(item.type),
                            "claimed_by_hostname": getattr(item, "claimed_by_hostname", None),
                        }
                        for item in items_raw
                    ]
                return {
                    "id": c.id,
                    "shorthand": c.shorthand or "",
                    "description": c.description or "",
                    "target": c.target or "",
                    "status": status_val,
                    "is_shared": bool(c.is_shared),
                    "total": plan_status["total"],
                    "completed": plan_status["completed"],
                    "in_progress": plan_status["in_progress"],
                    "subcampaigns": children,
                    "items": items,
                }

            roots = cs.get_root_campaigns()
            result = [serialize_campaign(c) for c in roots]
            await send_fn({"type": "browse_result", "target": "campaigns", "data": result})

        elif target == "peers":
            mesh_svc = getattr(server, "mesh_service", None)
            if mesh_svc is None:
                await send_fn({"type": "browse_result", "target": "peers", "data": []})
                return
            peers = mesh_svc.get_peers()
            result = []
            for p in peers:
                result.append(
                    {
                        "instance_id": p.instance_id,
                        "hostname": p.hostname,
                        "ip_address": p.ip_address,
                        "viz_port": p.viz_port,
                        "mode": p.status.agent_mode if p.status else "unknown",
                        "embryo_count": p.status.embryo_count if p.status else 0,
                        "is_trusted": p.is_trusted,
                        "tls_enabled": p.tls_enabled,
                        "shared_campaigns": [],
                    }
                )
            await send_fn({"type": "browse_result", "target": "peers", "data": result})

        elif target == "peer_campaigns":
            hostname = data.get("hostname", "")
            mesh_svc = getattr(server, "mesh_service", None)
            if not mesh_svc or not hostname:
                await send_fn({"type": "browse_result", "target": "peer_campaigns", "data": []})
                return
            peer = mesh_svc.find_peer_by_hostname(hostname)
            if not peer or not mesh_svc.peer_client:
                await send_fn({"type": "browse_result", "target": "peer_campaigns", "data": []})
                return
            info = await mesh_svc.peer_client.fetch_peer_info(peer)
            shared = (info or {}).get("shared_campaigns", [])
            # Return peers list with this peer's campaigns populated
            peers = mesh_svc.get_peers()
            result = []
            for p in peers:
                campaigns = []
                if p.instance_id == peer.instance_id:
                    for c in shared:
                        campaigns.append(
                            {
                                "id": c.get("id", ""),
                                "shorthand": c.get("shorthand", ""),
                                "description": c.get("description", ""),
                                "total": c.get("item_count", 0),
                                "completed": c.get("completed_count", 0),
                                "items": [],
                            }
                        )
                result.append(
                    {
                        "instance_id": p.instance_id,
                        "hostname": p.hostname,
                        "ip_address": p.ip_address,
                        "viz_port": p.viz_port,
                        "mode": p.status.agent_mode if p.status else "unknown",
                        "embryo_count": p.status.embryo_count if p.status else 0,
                        "is_trusted": p.is_trusted,
                        "tls_enabled": p.tls_enabled,
                        "shared_campaigns": campaigns,
                    }
                )
            await send_fn({"type": "browse_result", "target": "peer_campaigns", "data": result})

        elif target == "peer_campaign_items":
            hostname = data.get("hostname", "")
            campaign_id = data.get("campaign_id", "")
            mesh_svc = getattr(server, "mesh_service", None)
            if not mesh_svc or not hostname or not campaign_id:
                await send_fn(
                    {
                        "type": "browse_result",
                        "target": "peer_campaign_items",
                        "data": [],
                    }
                )
                return
            peer = mesh_svc.find_peer_by_hostname(hostname)
            if not peer or not mesh_svc.peer_client:
                await send_fn(
                    {
                        "type": "browse_result",
                        "target": "peer_campaign_items",
                        "data": [],
                    }
                )
                return
            export = await mesh_svc.peer_client.fetch_campaign_export(peer, campaign_id)
            if not export:
                await send_fn(
                    {
                        "type": "browse_result",
                        "target": "peer_campaign_items",
                        "data": [],
                    }
                )
                return
            items = []
            for item in export.get("items", []):
                items.append(
                    {
                        "id": item.get("id", ""),
                        "title": item.get("title", ""),
                        "status": item.get("status", "planned"),
                        "claimed_by_hostname": item.get("claimed_by_hostname"),
                    }
                )
            await send_fn(
                {
                    "type": "browse_result",
                    "target": "peer_campaign_items",
                    "data": items,
                    "campaign_id": campaign_id,
                    "hostname": hostname,
                }
            )

    except Exception as e:
        logger.debug(f"Browse error ({target}): {e}")
        await send_fn({"type": "browse_result", "target": target, "data": []})


_request_counter = 0


def _make_request_id() -> str:
    """Generate a simple unique request ID."""
    global _request_counter
    _request_counter += 1
    return f"req_{_request_counter}"
