"""
Agent WebSocket route — /ws/agent

Provides the TUI ↔ agent bridge. The Ink TUI connects here
to send chat messages and receive streaming responses, tool calls,
choice pickers, command results, and notifications.
"""

import asyncio
import json
import logging
from datetime import datetime
from typing import Dict, Optional

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

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
    _choice_futures: Dict[str, asyncio.Future] = {}

    async def _run_wizard(wizard, websocket, send_fn, _choice_futures, bridge=None, log_transcript=None):
        """Run the wizard's interactive loop.

        Returns the wizard task so callers can check for exceptions.
        Used both at startup and for the /wizard command.
        """
        _wizard_input_future: Optional[asyncio.Future] = None

        async def _wizard_wait_for_input() -> str:
            nonlocal _wizard_input_future
            loop = asyncio.get_event_loop()
            _wizard_input_future = loop.create_future()
            return await _wizard_input_future

        async def _wizard_wait_for_choice(choice_data: dict) -> str:
            request_id = _make_request_id()
            choice_data["request_id"] = request_id
            await send_fn({
                "type": "choice_request",
                "choice_data": choice_data,
                "request_id": request_id,
            })
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
                    websocket.receive_text(), timeout=60.0,
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
                        await send_fn({
                            "type": "stream_end",
                            "tokens": {"input_tokens": 0, "output_tokens": 0,
                                       "total_tokens": 0, "api_calls": 0},
                            "wizard_complete": True,
                        })
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
            await websocket.send_json({
                "type": "error",
                "error": "Agent bridge not initialized",
            })
            await websocket.close()
            return

        # Send connection metadata (version, tokens, embryo count, commands)
        meta = bridge.get_connect_metadata()
        _connected_msg = {
            "type": "connected",
            **meta,
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
                                    {"id": "pair", "label": "Pair",
                                     "description": f"Start pairing with {hostname}"},
                                    {"id": "ignore", "label": "Ignore",
                                     "description": "Dismiss (you can pair later via /pair)"},
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
                            "question": f"{hostname} wants to pair\nVerify this code matches: {pin}",
                            "options": [
                                {"id": "accept", "label": "Accept pairing",
                                 "description": f"Trust {hostname} and allow mesh communication"},
                                {"id": "reject", "label": "Reject",
                                 "description": "Decline this pairing request"},
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
            unsub = server.event_bus.subscribe_async(_ET.MESH_PEER_DISCOVERED, _push_peer_discovered)
            _mesh_unsubs.append(unsub)
            unsub = server.event_bus.subscribe_async(_ET.MESH_PEER_LOST, _push_peer_lost)
            _mesh_unsubs.append(unsub)
            unsub = server.event_bus.subscribe_async(_ET.MESH_PEER_UPDATED, _push_peer_count)
            _mesh_unsubs.append(unsub)

            # Pairing events
            unsub = server.event_bus.subscribe_async(_ET.MESH_PAIRING_REQUESTED, _push_pairing_requested)
            _mesh_unsubs.append(unsub)
            unsub = server.event_bus.subscribe_async(_ET.MESH_PAIRING_COMPLETED, _push_pairing_completed)
            _mesh_unsubs.append(unsub)

            # Security events
            unsub = server.event_bus.subscribe_async(_ET.MESH_AUTH_FAILURE, _push_auth_failure)
            _mesh_unsubs.append(unsub)
            unsub = server.event_bus.subscribe_async(_ET.MESH_CERT_PIN_FAILURE, _push_cert_pin_failure)
            _mesh_unsubs.append(unsub)
            unsub = server.event_bus.subscribe_async(_ET.MESH_SCOPE_DENIED, _push_scope_denied)
            _mesh_unsubs.append(unsub)

        # Active streaming task (so we can cancel on disconnect)
        active_task: Optional[asyncio.Task] = None
        wizard_task = None

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
                        sdir / "transcript.jsonl", "a", encoding="utf-8",
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

        try:
            # ── Wizard phase ──────────────────────────────────────
            # Run startup wizard (if needed) before entering the REPL.
            wizard = getattr(bridge, "_wizard", None)
            if wizard is not None and wizard.needed:
                wizard_task = await _run_wizard(
                    wizard, websocket, send_fn, _choice_futures, bridge,
                    log_transcript=_log_transcript,
                )
                exc = _handle_wizard_result(wizard_task)
                if exc:
                    logger.error(f"Wizard error: {exc}", exc_info=exc)
                    await send_fn({
                        "type": "stream_end",
                        "tokens": {"input_tokens": 0, "output_tokens": 0,
                                   "total_tokens": 0, "api_calls": 0},
                        "wizard_complete": True,
                    })

            # ── Auto-briefing for new sessions with existing context ──
            wizard_ran = wizard is not None and wizard.needed
            if not wizard_ran:
                briefing = bridge.get_session_briefing()
                if briefing:
                    await send_fn({"type": "stream_start"})
                    await send_fn({"type": "text", "text": briefing})
                    await send_fn({
                        "type": "stream_end",
                        "tokens": {"input_tokens": 0, "output_tokens": 0,
                                   "total_tokens": 0, "api_calls": 0},
                    })

            # ── Main REPL loop ────────────────────────────────────
            while True:
                try:
                    raw = await asyncio.wait_for(
                        websocket.receive_text(), timeout=60.0,
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

                if msg_type == "chat":
                    text = data.get("text", "").strip()
                    if not text:
                        continue

                    # Cancel any previous stream
                    if active_task and not active_task.done():
                        active_task.cancel()

                    active_task = asyncio.create_task(
                        bridge.stream_response(text, send_fn, choice_future_factory)
                    )

                elif msg_type == "choice_response":
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

                elif msg_type == "command":
                    command = data.get("command", "").strip()
                    if not command:
                        continue

                    # /wizard — run the wizard inline
                    if command.lower() in ("/wizard",):
                        w = getattr(bridge, "_wizard", None)
                        if w is None:
                            await send_fn({
                                "type": "command_result",
                                "command": "/wizard",
                                "error": "Wizard not available",
                            })
                        else:
                            # Re-create wizard so it re-assesses gaps
                            cs = getattr(bridge, "_context_store", None)
                            cc = w.claude_client
                            bridge.init_wizard(cs, cc)
                            w = bridge._wizard

                            # Tell TUI we're entering wizard mode
                            await send_fn({
                                "type": "command_result",
                                "command": "/wizard",
                                "content": {"wizard_active": True},
                            })

                            wizard_task = await _run_wizard(
                                w, websocket, send_fn, _choice_futures, bridge,
                                log_transcript=_log_transcript,
                            )
                            exc = _handle_wizard_result(wizard_task)
                            if exc:
                                logger.error(f"Wizard error: {exc}", exc_info=exc)
                                await send_fn({
                                    "type": "stream_end",
                                    "tokens": {"input_tokens": 0, "output_tokens": 0,
                                               "total_tokens": 0, "api_calls": 0},
                                    "wizard_complete": True,
                                })
                    else:
                        try:
                            await bridge.handle_command(command, send_fn, choice_futures=_choice_futures)
                        except Exception as e:
                            logger.error("Command '%s' failed: %s", command, e, exc_info=True)
                            await send_fn({
                                "type": "command_result",
                                "command": command,
                                "error": str(e),
                            })

                elif msg_type == "browse":
                    target = data.get("target", "")
                    await _handle_browse(
                        target, data, server, bridge, send_fn,
                    )

                elif msg_type == "ping":
                    await websocket.send_json({"type": "pong"})

                elif msg_type == "pong":
                    pass  # response to our ping

        except WebSocketDisconnect:
            logger.info("TUI client disconnected")
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
            # Clean up pending futures
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
                    items = [{
                        "id": item.id,
                        "title": item.title,
                        "status": item.status.value if hasattr(item.status, "value") else str(item.status),
                        "type": item.type.value if hasattr(item.type, "value") else str(item.type),
                        "claimed_by_hostname": getattr(item, "claimed_by_hostname", None),
                    } for item in items_raw]
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
                result.append({
                    "instance_id": p.instance_id,
                    "hostname": p.hostname,
                    "ip_address": p.ip_address,
                    "viz_port": p.viz_port,
                    "mode": p.status.agent_mode if p.status else "unknown",
                    "embryo_count": p.status.embryo_count if p.status else 0,
                    "is_trusted": p.is_trusted,
                    "tls_enabled": p.tls_enabled,
                    "shared_campaigns": [],
                })
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
                        campaigns.append({
                            "id": c.get("id", ""),
                            "shorthand": c.get("shorthand", ""),
                            "description": c.get("description", ""),
                            "total": c.get("item_count", 0),
                            "completed": c.get("completed_count", 0),
                            "items": [],
                        })
                result.append({
                    "instance_id": p.instance_id,
                    "hostname": p.hostname,
                    "ip_address": p.ip_address,
                    "viz_port": p.viz_port,
                    "mode": p.status.agent_mode if p.status else "unknown",
                    "embryo_count": p.status.embryo_count if p.status else 0,
                    "is_trusted": p.is_trusted,
                    "tls_enabled": p.tls_enabled,
                    "shared_campaigns": campaigns,
                })
            await send_fn({"type": "browse_result", "target": "peer_campaigns", "data": result})

        elif target == "peer_campaign_items":
            hostname = data.get("hostname", "")
            campaign_id = data.get("campaign_id", "")
            mesh_svc = getattr(server, "mesh_service", None)
            if not mesh_svc or not hostname or not campaign_id:
                await send_fn({"type": "browse_result", "target": "peer_campaign_items", "data": []})
                return
            peer = mesh_svc.find_peer_by_hostname(hostname)
            if not peer or not mesh_svc.peer_client:
                await send_fn({"type": "browse_result", "target": "peer_campaign_items", "data": []})
                return
            export = await mesh_svc.peer_client.fetch_campaign_export(peer, campaign_id)
            if not export:
                await send_fn({"type": "browse_result", "target": "peer_campaign_items", "data": []})
                return
            items = []
            for item in export.get("items", []):
                items.append({
                    "id": item.get("id", ""),
                    "title": item.get("title", ""),
                    "status": item.get("status", "planned"),
                    "claimed_by_hostname": item.get("claimed_by_hostname"),
                })
            await send_fn({
                "type": "browse_result",
                "target": "peer_campaign_items",
                "data": items,
                "campaign_id": campaign_id,
                "hostname": hostname,
            })

    except Exception as e:
        logger.debug(f"Browse error ({target}): {e}")
        await send_fn({"type": "browse_result", "target": target, "data": []})


_request_counter = 0


def _make_request_id() -> str:
    """Generate a simple unique request ID."""
    global _request_counter
    _request_counter += 1
    return f"req_{_request_counter}"
