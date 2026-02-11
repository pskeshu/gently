"""WebSocket route - real-time streaming and message handling."""

import asyncio
import json
import logging
from datetime import datetime

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from ..models import ClientInfo

logger = logging.getLogger(__name__)


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
            await websocket.send_json({
                "type": "connected",
                **stats,
                "timestamp": datetime.now().isoformat()
            })

            # Always send timelapse state on connect so client can reconcile
            # (if IDLE with no session_id, client will clear stale cached state)
            timelapse_state = server.timelapse_tracker.to_dict()
            await websocket.send_json({
                "type": "timelapse_state",
                "data": timelapse_state
            })

            # Keep connection alive and handle incoming messages
            while True:
                try:
                    data = await asyncio.wait_for(
                        websocket.receive_text(),
                        timeout=30.0
                    )
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

        if msg_type == "get_calibration":
            images = server.store.get_all_calibration(embryo_id)
            await websocket.send_json({
                "type": "calibration",
                "data": [img.to_dict() for img in images]
            })

        elif msg_type == "get_volumes":
            images = server.store.get_all_volumes(embryo_id)
            await websocket.send_json({
                "type": "volumes",
                "data": [img.to_dict() for img in images]
            })

        elif msg_type == "get_snapshots":
            images = server.store.get_all_snapshots(embryo_id)
            await websocket.send_json({
                "type": "snapshots",
                "data": [img.to_dict() for img in images]
            })

        elif msg_type == "get_embryos":
            await websocket.send_json({
                "type": "embryos",
                "data": server.store.get_embryo_ids()
            })

        elif msg_type == "get_image":
            uid = data.get("uid")
            image = server.store.get_image_by_uid(uid)
            if image:
                await websocket.send_json({
                    "type": "image",
                    "data": image.to_dict()
                })

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
                    name = re.sub(r'<[^>]+>', '', name)[:50]
                # Update the client's info
                async with server.manager._lock:
                    if websocket in server.manager.active_connections:
                        old_info = server.manager.active_connections[websocket]
                        server.manager.active_connections[websocket] = ClientInfo(
                            client_id=client_id,
                            name=name or old_info.name,
                            color=server.manager._generate_color(client_id),
                            connected_at=old_info.connected_at
                        )
                await server.manager.broadcast_presence()

        elif msg_type == "set_name":
            # Client updating their display name
            name = data.get("name")
            if name:
                # Sanitize name: strip HTML tags, limit length
                import re
                name = re.sub(r'<[^>]+>', '', name)[:50]
                await server.manager.update_client_name(websocket, name)

        elif msg_type == "get_presence":
            # Client requesting current presence list
            await server.manager.broadcast_presence()

    except json.JSONDecodeError:
        logger.warning(f"Invalid JSON received: {message[:100]}")
