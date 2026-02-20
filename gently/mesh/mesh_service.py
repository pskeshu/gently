"""
MeshService — main orchestrator for peer discovery and status exchange.

Subclasses the existing Service base class, managing:
- UDP discovery (broadcast + listen via MeshDiscovery)
- Peer reaping (remove dead peers)
- Status refresh (HTTP fetch from live peers)
"""

import asyncio
import logging
import time
from typing import Callable, Dict, List, Optional

from gently.core.event_bus import EventType
from gently.core.service import Service

from .discovery import MeshDiscovery
from .models import PeerCapability, PeerInfo, PeerStatus
from .peer_client import PeerClient

logger = logging.getLogger(__name__)

REAPER_INTERVAL = 10.0  # seconds
STATUS_REFRESH_INTERVAL = 30.0  # seconds


class MeshService(Service):
    """
    Manages peer discovery, status exchange, and the local peer registry.

    Parameters
    ----------
    instance_id : str
        Stable UUID for this Gently instance.
    viz_port : int
        Port of the local viz server (default 8080).
    capability_provider : callable
        Returns a PeerCapability dict for this node.
    status_provider : callable
        Returns a PeerStatus dict for this node.
    mesh_port : int
        UDP port for broadcast discovery (default 19547).
    """

    def __init__(
        self,
        instance_id: str,
        viz_port: int = 8080,
        capability_provider: Callable[[], dict] = lambda: {},
        status_provider: Callable[[], dict] = lambda: {},
        mesh_port: int = 19547,
    ):
        import socket as _socket

        super().__init__(
            name="mesh",
            service_type="mesh",
            host="0.0.0.0",
            port=mesh_port,
        )
        self.instance_id = instance_id
        self.viz_port = viz_port
        self._capability_provider = capability_provider
        self._status_provider = status_provider
        self._mesh_port = mesh_port

        self._hostname = _socket.gethostname()
        self._peers: Dict[str, PeerInfo] = {}
        self._discovery: Optional[MeshDiscovery] = None
        self._peer_client: Optional[PeerClient] = None
        self._reaper_task: Optional[asyncio.Task] = None
        self._refresh_task: Optional[asyncio.Task] = None

    # ------------------------------------------------------------------
    # Service lifecycle
    # ------------------------------------------------------------------

    async def on_start(self):
        self._peer_client = PeerClient()

        self._discovery = MeshDiscovery(
            instance_id=self.instance_id,
            hostname=self._hostname,
            viz_port=self.viz_port,
            mesh_port=self._mesh_port,
        )
        self._discovery.on_peer_discovered = self._on_peer_discovered
        self._discovery.on_peer_heartbeat = self._on_peer_heartbeat

        await self._discovery.start()

        self._reaper_task = asyncio.create_task(self._reaper_loop())
        self._refresh_task = asyncio.create_task(self._refresh_loop())

    async def on_stop(self):
        for task in (self._reaper_task, self._refresh_task):
            if task and not task.done():
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass

        if self._discovery:
            await self._discovery.stop()

        if self._peer_client:
            await self._peer_client.close()

        self._peers.clear()

    # ------------------------------------------------------------------
    # Discovery callbacks
    # ------------------------------------------------------------------

    def _on_peer_discovered(self, data: dict, sender_ip: str):
        """Called when a new instance_id is first seen."""
        peer_id = data.get("instance_id", "")
        now = time.time()

        peer = PeerInfo(
            instance_id=peer_id,
            hostname=data.get("hostname", ""),
            ip_address=sender_ip,
            viz_port=data.get("viz_port", 8080),
            first_seen=now,
            last_seen=now,
        )
        self._peers[peer_id] = peer

        self._emit_event(EventType.MESH_PEER_DISCOVERED, {
            "instance_id": peer_id,
            "hostname": peer.hostname,
            "ip_address": sender_ip,
        })

        logger.info(f"Mesh: discovered peer {peer.hostname} ({peer_id[:8]}) at {sender_ip}")

        # Kick off an immediate status fetch (fire-and-forget)
        asyncio.ensure_future(self._fetch_and_update_peer(peer))

    def _on_peer_heartbeat(self, instance_id: str, sender_ip: str):
        """Called on subsequent heartbeats from a known peer."""
        peer = self._peers.get(instance_id)
        if peer:
            peer.last_seen = time.time()
            peer.ip_address = sender_ip  # may change if DHCP renews

    # ------------------------------------------------------------------
    # Background loops
    # ------------------------------------------------------------------

    async def _reaper_loop(self):
        """Periodically remove dead peers."""
        while True:
            await asyncio.sleep(REAPER_INTERVAL)
            dead = [pid for pid, p in self._peers.items() if p.is_dead]
            for pid in dead:
                peer = self._peers.pop(pid, None)
                if peer:
                    # Let discovery re-discover this peer if it comes back
                    if self._discovery:
                        self._discovery.forget_peer(pid)
                    self._emit_event(EventType.MESH_PEER_LOST, {
                        "instance_id": pid,
                        "hostname": peer.hostname,
                    })
                    logger.info(f"Mesh: lost peer {peer.hostname} ({pid[:8]})")

    async def _refresh_loop(self):
        """Periodically fetch full status from all live peers."""
        while True:
            await asyncio.sleep(STATUS_REFRESH_INTERVAL)
            for peer in list(self._peers.values()):
                if not peer.is_dead:
                    await self._fetch_and_update_peer(peer)

    async def _fetch_and_update_peer(self, peer: PeerInfo):
        """Fetch status from a peer and update local record."""
        if not self._peer_client:
            return

        data = await self._peer_client.fetch_peer_info(peer)
        if data is None:
            return

        caps_data = data.get("capabilities", {})
        status_data = data.get("status", {})

        peer.capabilities = PeerCapability.from_dict(caps_data)
        peer.status = PeerStatus.from_dict(status_data)

        self._emit_event(EventType.MESH_PEER_UPDATED, {
            "instance_id": peer.instance_id,
            "hostname": peer.hostname,
        })

    # ------------------------------------------------------------------
    # Public query API
    # ------------------------------------------------------------------

    def get_peers(self) -> List[PeerInfo]:
        """Return all live (non-dead) peers."""
        return [p for p in self._peers.values() if not p.is_dead]

    def get_all_peers(self) -> List[PeerInfo]:
        """Return all tracked peers including stale/dead ones."""
        return list(self._peers.values())

    def get_peer(self, instance_id: str) -> Optional[PeerInfo]:
        """Get a specific peer by instance_id."""
        return self._peers.get(instance_id)

    def find_peers_with(self, capability: str) -> List[PeerInfo]:
        """
        Find live peers that have a given capability flag.

        E.g. find_peers_with("has_gpu") returns peers where
        capabilities.has_gpu is True.
        """
        results = []
        for p in self.get_peers():
            if getattr(p.capabilities, capability, False):
                results.append(p)
        return results

    def get_local_info(self) -> dict:
        """
        Build this node's full info dict (capabilities + status).

        Called by the /api/mesh/status route to respond to peer queries.
        """
        caps = self._capability_provider()
        status = self._status_provider()

        return {
            "instance_id": self.instance_id,
            "hostname": self._hostname,
            "viz_port": self.viz_port,
            "capabilities": caps,
            "status": status,
        }

    @property
    def peer_client(self) -> Optional[PeerClient]:
        """Expose the peer client for direct campaign operations."""
        return self._peer_client

    def find_peer_by_hostname(self, hostname: str) -> Optional[PeerInfo]:
        """Find a live peer by hostname (case-insensitive)."""
        hostname_lower = hostname.lower()
        for p in self.get_peers():
            if p.hostname.lower() == hostname_lower:
                return p
        return None

    @property
    def peer_count(self) -> int:
        """Number of live peers."""
        return len(self.get_peers())
