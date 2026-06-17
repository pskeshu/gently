"""WebSocket route - real-time streaming and message handling."""

import asyncio
import json
import logging
from datetime import datetime
from typing import Any

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from ..models import ClientInfo

logger = logging.getLogger(__name__)

# /ws message types that mutate experiment state (define what gets imaged).
# These are control actions and are gated by role; pure read/presence
# messages stay open so anyone can watch.
_MARKING_TYPES = frozenset(
    {
        "embryo_marked",
        "marking_update",
        "marking_done",
        "marking_redetect",
    }
)


def _ws_can_control(websocket: WebSocket) -> bool:
    """Whether this /ws client may perform control actions (marking).

    Account mode: operators/admins (by session cookie) only. Legacy mode
    (no accounts configured): open, preserving prior behavior.
    """
    from gently.ui.web.accounts import CONTROL_ROLES, get_account_store
    from gently.ui.web.auth import SESSION_COOKIE

    store = get_account_store()
    if store is None or not store.has_users():
        return True
    token = websocket.cookies.get(SESSION_COOKIE)
    user = store.verify_session(token) if token else None
    role = store.get_role(user) if user else None
    return role in CONTROL_ROLES


def create_router(server) -> APIRouter:
    router = APIRouter()

    @router.websocket("/ws")
    async def websocket_endpoint(websocket: WebSocket):
        """WebSocket endpoint for real-time updates"""
        # Connect with temporary defaults - client will send 'join' message with real info
        await server.manager.connect(websocket)
        try:
            # Send current status on connect
            stats = server.store.get_stats()
            await websocket.send_json(
                {"type": "connected", **stats, "timestamp": datetime.now().isoformat()}
            )

            # Always send timelapse state on connect so client can reconcile
            # (if IDLE with no session_id, client will clear stale cached state)
            timelapse_state = server.timelapse_tracker.to_dict()
            # The header's session id is driven by this payload; the tracker's
            # session_id goes stale after a resume with no active timelapse, so
            # override it with the live agent session (the source of truth).
            try:
                bridge = getattr(server, "agent_bridge", None)
                if bridge is not None and getattr(bridge, "agent", None) is not None:
                    live_sid = bridge.agent.session_id
                    if live_sid:
                        timelapse_state["session_id"] = live_sid
            except Exception:
                pass
            await websocket.send_json({"type": "timelapse_state", "data": timelapse_state})

            # Keep connection alive and handle incoming messages
            while True:
                try:
                    data = await asyncio.wait_for(websocket.receive_text(), timeout=30.0)
                    # Handle client messages (e.g., requests)
                    await _handle_ws_message(server, websocket, data)
                except asyncio.TimeoutError:
                    # Send ping to keep connection alive
                    await websocket.send_json({"type": "ping"})

        except WebSocketDisconnect:
            try:
                await server.manager.disconnect(websocket)
            except Exception:
                pass
        except asyncio.CancelledError:
            try:
                await server.manager.disconnect(websocket)
            except Exception:
                pass
        except Exception as e:
            logger.error(f"WebSocket error: {e}")
            try:
                await server.manager.disconnect(websocket)
            except Exception:
                pass

    return router


async def _handle_ws_message(server, websocket: WebSocket, message: str):
    """Handle incoming WebSocket message"""
    try:
        data = json.loads(message)
        msg_type = data.get("type")
        embryo_id = data.get("embryo_id")

        # Gate control actions (marking) by role; viewing/presence stays open.
        if msg_type in _MARKING_TYPES and not _ws_can_control(websocket):
            logger.warning("Ignored %s from a view-only /ws client", msg_type)
            return

        if msg_type == "get_calibration":
            images = server.store.get_all_calibration(embryo_id)
            await websocket.send_json(
                {"type": "calibration", "data": [img.to_dict() for img in images]}
            )

        elif msg_type == "get_volumes":
            images = server.store.get_all_volumes(embryo_id)
            await websocket.send_json(
                {"type": "volumes", "data": [img.to_dict() for img in images]}
            )

        elif msg_type == "get_snapshots":
            images = server.store.get_all_snapshots(embryo_id)
            await websocket.send_json(
                {"type": "snapshots", "data": [img.to_dict() for img in images]}
            )

        elif msg_type == "get_embryos":
            await websocket.send_json({"type": "embryos", "data": server.store.get_embryo_ids()})

        elif msg_type == "get_image":
            uid = data.get("uid")
            image = server.store.get_image_by_uid(uid)
            if image:
                await websocket.send_json({"type": "image", "data": image.to_dict()})

        elif msg_type == "pong":
            pass  # Client responding to ping

        # Presence-related messages
        elif msg_type == "join":
            # Client joining with identity info
            client_id = data.get("client_id")
            name = data.get("name")
            if client_id:
                # Sanitize name: strip HTML tags, limit length
                if name:
                    import re

                    name = re.sub(r"<[^>]+>", "", name)[:50]
                # Update the client's info
                async with server.manager._lock:
                    if websocket in server.manager.active_connections:
                        old_info = server.manager.active_connections[websocket]
                        server.manager.active_connections[websocket] = ClientInfo(
                            client_id=client_id,
                            name=name or old_info.name,
                            color=server.manager._generate_color(client_id),
                            connected_at=old_info.connected_at,
                        )
                await server.manager.broadcast_presence()

        elif msg_type == "set_name":
            # Client updating their display name
            name = data.get("name")
            if name:
                # Sanitize name: strip HTML tags, limit length
                import re

                name = re.sub(r"<[^>]+>", "", name)[:50]
                await server.manager.update_client_name(websocket, name)

        elif msg_type == "get_presence":
            # Client requesting current presence list
            await server.manager.broadcast_presence()

        # Embryo marking messages
        elif msg_type == "embryo_marked":
            session_id = data.get("session_id")
            marker = data.get("marker")
            if session_id and marker and hasattr(server, "_marking_sessions"):
                session = server._marking_sessions.get(session_id)
                if session:
                    session["markers"].append(marker)
                    logger.info(
                        f"Embryo marked: #{marker['number']}"
                        f" at ({marker['pixelX']}, {marker['pixelY']})"
                    )

        elif msg_type == "marking_update":
            session_id = data.get("session_id")
            markers = data.get("markers", [])
            if session_id and hasattr(server, "_marking_sessions"):
                session = server._marking_sessions.get(session_id)
                if session:
                    session["markers"] = markers
                    logger.info(f"Marking updated: {len(markers)} embryo(s)")

        elif msg_type == "marking_done":
            session_id = data.get("session_id")
            markers = data.get("markers", [])
            if session_id and hasattr(server, "_marking_sessions"):
                session = server._marking_sessions.get(session_id)
                if session:
                    session["markers"] = markers
                    session["complete"].set()
                    role_summary: dict[str, Any] = {}
                    for m in markers:
                        r = m.get("role", "test")
                        role_summary[r] = role_summary.get(r, 0) + 1
                    logger.info(
                        f"Marking complete: {len(markers)} embryo(s) (roles: {role_summary})"
                    )

        elif msg_type == "marking_redetect":
            # Client requested recapture + re-run SAM. The agent that started
            # the marking session is responsible for handling this; we mark
            # the request on the session and emit an event the agent can
            # listen for. Once recapture lands, the agent calls
            # start_marking_session again with the new image + markers.
            session_id = data.get("session_id")
            if session_id and hasattr(server, "_marking_sessions"):
                session = server._marking_sessions.get(session_id)
                if session is not None:
                    session["redetect_requested"] = True
                    logger.info(f"Marking redetect requested for session {session_id}")
                    try:
                        from gently.core import EventType, get_event_bus

                        get_event_bus().publish(
                            event_type=EventType.STATUS_CHANGED,
                            data={
                                "change": "marking_redetect_requested",
                                "session_id": session_id,
                            },
                            source="marking_ws",
                        )
                    except Exception as e:
                        logger.warning(f"Failed to publish redetect event: {e}")

    except json.JSONDecodeError:
        logger.warning(f"Invalid JSON received: {message[:100]}")
