"""
UDP broadcast discovery layer for Gently mesh.

Uses asyncio DatagramProtocol for non-blocking broadcast/listen
on port 19547. Zero dependencies beyond Python stdlib.
"""

import asyncio
import json
import logging
import socket
from typing import Callable, Optional

logger = logging.getLogger(__name__)

MESH_PORT = 19547
MESH_MAGIC = "GENTLY_MESH"
MESH_PROTOCOL_VERSION = 1
BROADCAST_INTERVAL = 5.0  # seconds


class _MeshProtocol(asyncio.DatagramProtocol):
    """asyncio datagram protocol for mesh heartbeats."""

    def __init__(
        self,
        instance_id: str,
        on_peer_discovered: Callable[[dict, str], None],
        on_peer_heartbeat: Callable[[str, str], None],
    ):
        self.instance_id = instance_id
        self._on_peer_discovered = on_peer_discovered
        self._on_peer_heartbeat = on_peer_heartbeat
        self._known_ids: set = set()
        self.transport: Optional[asyncio.DatagramTransport] = None

    def connection_made(self, transport: asyncio.DatagramTransport):
        self.transport = transport

    def datagram_received(self, data: bytes, addr: tuple):
        try:
            msg = json.loads(data.decode("utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError):
            return

        if msg.get("magic") != MESH_MAGIC:
            return

        peer_id = msg.get("instance_id", "")
        if not peer_id or peer_id == self.instance_id:
            return

        sender_ip = addr[0]

        if peer_id not in self._known_ids:
            self._known_ids.add(peer_id)
            self._on_peer_discovered(msg, sender_ip)
        else:
            self._on_peer_heartbeat(peer_id, sender_ip)

    def error_received(self, exc: Exception):
        logger.debug(f"Mesh UDP error: {exc}")

    def connection_lost(self, exc: Optional[Exception]):
        pass

    def forget_peer(self, instance_id: str):
        """Remove a peer from the known set (e.g. after it's declared dead)."""
        self._known_ids.discard(instance_id)


class MeshDiscovery:
    """
    UDP broadcast-based mesh discovery.

    Sends periodic heartbeats and listens for peers on the LAN.
    """

    def __init__(
        self,
        instance_id: str,
        hostname: str,
        viz_port: int = 8080,
        mesh_port: int = MESH_PORT,
    ):
        self.instance_id = instance_id
        self.hostname = hostname
        self.viz_port = viz_port
        self.mesh_port = mesh_port

        self._protocol: Optional[_MeshProtocol] = None
        self._transport: Optional[asyncio.DatagramTransport] = None
        self._broadcast_task: Optional[asyncio.Task] = None
        self._running = False

        # Callbacks — set by MeshService before start()
        self.on_peer_discovered: Callable[[dict, str], None] = lambda d, ip: None
        self.on_peer_heartbeat: Callable[[str, str], None] = lambda id, ip: None

    async def start(self):
        """Bind UDP socket and start broadcast loop."""
        loop = asyncio.get_running_loop()

        # Create a UDP socket with broadcast enabled
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.setblocking(False)

        try:
            sock.bind(("", self.mesh_port))
        except OSError as e:
            logger.warning(f"Mesh discovery: cannot bind port {self.mesh_port}: {e}")
            sock.close()
            return

        self._protocol = _MeshProtocol(
            instance_id=self.instance_id,
            on_peer_discovered=self.on_peer_discovered,
            on_peer_heartbeat=self.on_peer_heartbeat,
        )

        self._transport, _ = await loop.create_datagram_endpoint(
            lambda: self._protocol,
            sock=sock,
        )

        self._running = True
        self._broadcast_task = asyncio.create_task(self._broadcast_loop())
        logger.info(
            f"Mesh discovery started on port {self.mesh_port} "
            f"(instance={self.instance_id[:8]})"
        )

    async def stop(self):
        """Stop discovery and release socket."""
        self._running = False

        if self._broadcast_task and not self._broadcast_task.done():
            self._broadcast_task.cancel()
            try:
                await self._broadcast_task
            except asyncio.CancelledError:
                pass

        if self._transport:
            self._transport.close()
            self._transport = None

        logger.info("Mesh discovery stopped")

    def forget_peer(self, instance_id: str):
        """Tell the protocol to forget a peer so it can be re-discovered."""
        if self._protocol:
            self._protocol.forget_peer(instance_id)

    async def _broadcast_loop(self):
        """Send heartbeat every BROADCAST_INTERVAL seconds."""
        heartbeat = json.dumps({
            "magic": MESH_MAGIC,
            "version": MESH_PROTOCOL_VERSION,
            "instance_id": self.instance_id,
            "hostname": self.hostname,
            "viz_port": self.viz_port,
        }).encode("utf-8")

        while self._running:
            try:
                if self._transport:
                    self._transport.sendto(
                        heartbeat, ("255.255.255.255", self.mesh_port)
                    )
            except OSError as e:
                logger.debug(f"Mesh broadcast failed: {e}")

            await asyncio.sleep(BROADCAST_INTERVAL)
