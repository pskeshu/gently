"""
Connection Manager for the Visualization Server
=================================================

Manages WebSocket connections for broadcasting updates with presence tracking.
"""

import asyncio
import json
import logging
from datetime import datetime

from .models import ClientInfo, ImageData

logger = logging.getLogger(__name__)

# Optional imports
try:
    from fastapi import WebSocket

    FASTAPI_AVAILABLE = True
except ImportError:
    FASTAPI_AVAILABLE = False


class ConnectionManager:
    """Manages WebSocket connections for broadcasting updates with presence tracking"""

    # Colors for avatar backgrounds (pleasant, distinct colors)
    AVATAR_COLORS = [
        "#4a9eff",
        "#ff6b6b",
        "#51cf66",
        "#ffd43b",
        "#cc5de8",
        "#ff922b",
        "#20c997",
        "#748ffc",
        "#f06595",
        "#69db7c",
        "#ffa94d",
        "#9775fa",
        "#38d9a9",
        "#e599f7",
        "#74c0fc",
    ]

    def __init__(self):
        self.active_connections: dict[WebSocket, ClientInfo] = {}
        self._lock = asyncio.Lock()

    def _generate_color(self, client_id: str) -> str:
        """Generate consistent color from client_id"""
        hash_val = sum(ord(c) for c in client_id)
        return self.AVATAR_COLORS[hash_val % len(self.AVATAR_COLORS)]

    async def connect(
        self, websocket: WebSocket, client_id: str | None = None, name: str | None = None
    ):
        await websocket.accept()

        # Generate defaults if not provided
        if not client_id:
            import uuid

            client_id = str(uuid.uuid4())[:8]
        if not name:
            name = f"Anonymous {client_id[:4]}"

        client_info = ClientInfo(
            client_id=client_id,
            name=name,
            color=self._generate_color(client_id),
            connected_at=datetime.now().isoformat(),
        )

        async with self._lock:
            self.active_connections[websocket] = client_info
        logger.info(
            f"WebSocket connected: {name} ({client_id}). Total: {len(self.active_connections)}"
        )

        # Broadcast updated presence to all clients
        await self.broadcast_presence()

    async def disconnect(self, websocket: WebSocket):
        async with self._lock:
            client_info = self.active_connections.pop(websocket, None)
        if client_info:
            logger.info(
                f"WebSocket disconnected: {client_info.name}. Total: {len(self.active_connections)}"
            )
        else:
            logger.info(f"WebSocket disconnected. Total: {len(self.active_connections)}")

        # Broadcast updated presence to remaining clients
        await self.broadcast_presence()

    async def update_client_name(self, websocket: WebSocket, name: str):
        """Update a client's display name"""
        async with self._lock:
            if websocket in self.active_connections:
                old_info = self.active_connections[websocket]
                self.active_connections[websocket] = ClientInfo(
                    client_id=old_info.client_id,
                    name=name,
                    color=old_info.color,
                    connected_at=old_info.connected_at,
                )
        await self.broadcast_presence()

    def get_client_info(self, websocket: WebSocket) -> ClientInfo | None:
        """Get client info for a websocket"""
        return self.active_connections.get(websocket)

    async def broadcast_presence(self):
        """Broadcast current presence list to all clients"""
        if not self.active_connections:
            return

        # Deduplicate by client_id (same user in multiple tabs = one avatar)
        async with self._lock:
            seen_clients = {}
            for _ws, info in self.active_connections.items():
                # Keep the most recent entry for each client_id
                seen_clients[info.client_id] = {
                    "client_id": info.client_id,
                    "name": info.name,
                    "color": info.color,
                }
            clients_list = list(seen_clients.values())

        # Send personalized presence to each client (with is_you flag)
        disconnected = []
        for ws, info in list(self.active_connections.items()):
            try:
                personalized = []
                for client in clients_list:
                    personalized.append({**client, "is_you": client["client_id"] == info.client_id})
                await ws.send_json({"type": "presence", "clients": personalized})
            except Exception:
                disconnected.append(ws)

        # Remove stale connections
        async with self._lock:
            for ws in disconnected:
                self.active_connections.pop(ws, None)

    async def broadcast(self, message: dict):
        """Broadcast message to all connected clients"""
        if not self.active_connections:
            return

        message_json = json.dumps(message)
        async with self._lock:
            disconnected = []
            for connection in self.active_connections.keys():
                try:
                    await connection.send_text(message_json)
                except Exception as e:
                    # Expected when a client disconnects/reloads mid-broadcast
                    # (send after websocket.close). The connection is dropped
                    # below, so this is debug-level, not a warning.
                    logger.debug(
                        "Dropping a websocket that errored on send (client likely gone): %s", e
                    )
                    disconnected.append(connection)

            # Remove disconnected clients
            for conn in disconnected:
                self.active_connections.pop(conn, None)

    async def send_image(self, image_data: ImageData):
        """Send image data to all connected clients"""
        await self.broadcast({"type": "image", "data": image_data.to_dict()})

    async def send_event(
        self, event_type: str, data: dict, source: str | None = None, event_id: str | None = None
    ):
        """Send event notification to all clients"""
        await self.broadcast(
            {
                "type": "event",
                "event_type": event_type,
                "data": data,
                "source": source or "unknown",
                "event_id": event_id or "",
                "timestamp": datetime.now().isoformat(),
            }
        )
