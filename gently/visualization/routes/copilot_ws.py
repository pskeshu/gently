"""
Copilot WebSocket route — /ws/copilot

Provides the TUI ↔ Python copilot bridge. The Ink TUI connects here
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
    """Create the /ws/copilot router.

    Parameters
    ----------
    server : VisualizationServer
        Must have a ``copilot_bridge`` attribute (set by launch_copilot.py
        after the bridge is created).
    """
    router = APIRouter()

    # Pending choice futures keyed by request_id
    _choice_futures: Dict[str, asyncio.Future] = {}

    @router.websocket("/ws/copilot")
    async def copilot_websocket(websocket: WebSocket):
        await websocket.accept()

        bridge = getattr(server, "copilot_bridge", None)
        if bridge is None:
            await websocket.send_json({
                "type": "error",
                "error": "Copilot bridge not initialized",
            })
            await websocket.close()
            return

        # Send connection metadata (version, tokens, embryo count, commands)
        meta = bridge.get_connect_metadata()
        await websocket.send_json({
            "type": "connected",
            **meta,
            "timestamp": datetime.now().isoformat(),
        })

        # Active streaming task (so we can cancel on disconnect)
        active_task: Optional[asyncio.Task] = None

        async def send_fn(data: dict):
            """Send a JSON message to the TUI client."""
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
                    future = _choice_futures.pop(request_id, None)
                    if future and not future.done():
                        future.set_result(selected)

                elif msg_type == "cancel":
                    if active_task and not active_task.done():
                        active_task.cancel()
                        active_task = None

                elif msg_type == "command":
                    command = data.get("command", "").strip()
                    if command:
                        await bridge.handle_command(command, send_fn)

                elif msg_type == "ping":
                    await websocket.send_json({"type": "pong"})

                elif msg_type == "pong":
                    pass  # response to our ping

        except WebSocketDisconnect:
            logger.info("TUI client disconnected")
        except asyncio.CancelledError:
            pass
        except Exception as e:
            logger.error(f"Copilot WS error: {e}", exc_info=True)
        finally:
            if active_task and not active_task.done():
                active_task.cancel()
            # Clean up pending futures
            for future in _choice_futures.values():
                if not future.done():
                    future.cancel()
            _choice_futures.clear()

    return router


_request_counter = 0


def _make_request_id() -> str:
    """Generate a simple unique request ID."""
    global _request_counter
    _request_counter += 1
    return f"req_{_request_counter}"
