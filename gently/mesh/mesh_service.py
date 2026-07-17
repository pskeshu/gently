"""
MeshService — main orchestrator for peer discovery and status exchange.

Subclasses the existing Service base class, managing:
- UDP discovery (broadcast + listen via MeshDiscovery)
- Persistent verse map (topology survives restarts)
- Peer reaping (mark offline, don't delete trusted peers)
- Status refresh (HTTP fetch from live peers)
"""

import asyncio
import logging
import time
from collections.abc import Callable
from pathlib import Path

from gently.core.event_bus import EventType
from gently.core.service import Service

from ..settings import settings
from .discovery import MeshDiscovery
from .models import PeerCapability, PeerInfo, PeerStatus
from .peer_client import PeerClient
from .verse_map import VerseMap

logger = logging.getLogger(__name__)

REAPER_INTERVAL = settings.mesh.reaper_interval_s
STATUS_REFRESH_INTERVAL = settings.mesh.status_refresh_s


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
    config_dir : Path, optional
        Directory for persistent config files (verse map).
    """

    def __init__(
        self,
        instance_id: str,
        viz_port: int = settings.network.viz_port,
        capability_provider: Callable[[], dict] = lambda: {},
        status_provider: Callable[[], dict] = lambda: {},
        mesh_port: int = settings.network.mesh_port,
        pairing_manager=None,
        audit_log=None,
        config_dir: Path | None = None,
    ):
        import socket as _socket

        super().__init__(
            name="mesh",
            service_type="mesh",
            host=settings.network.mesh_bind,
            port=mesh_port,
        )
        self.instance_id = instance_id
        self.viz_port = viz_port
        self._capability_provider = capability_provider
        self._status_provider = status_provider
        self._mesh_port = mesh_port
        self._pairing_manager = pairing_manager
        self._audit_log = audit_log

        self._hostname = _socket.gethostname()
        self._peers: dict[str, PeerInfo] = {}
        self._discovery: MeshDiscovery | None = None
        self._peer_client: PeerClient | None = None
        self._reaper_task: asyncio.Task | None = None
        self._refresh_task: asyncio.Task | None = None
        self._cleanup_task: asyncio.Task | None = None

        # Persistent verse map
        if config_dir is None:
            config_dir = settings.storage.base_path / "config"
        self._verse_map = VerseMap(config_dir)

    # ------------------------------------------------------------------
    # Service lifecycle
    # ------------------------------------------------------------------

    async def on_start(self):
        self._peer_client = PeerClient(
            pairing_manager=self._pairing_manager,
            audit_log=self._audit_log,
        )

        self._discovery = MeshDiscovery(
            instance_id=self.instance_id,
            hostname=self._hostname,
            viz_port=self.viz_port,
            mesh_port=self._mesh_port,
            pairing_manager=self._pairing_manager,
            audit_log=self._audit_log,
        )
        self._discovery.on_peer_discovered = self._on_peer_discovered
        self._discovery.on_peer_heartbeat = self._on_peer_heartbeat
        self._discovery.on_nudge_received = self._on_nudge_received

        await self._discovery.start()

        self._reaper_task = asyncio.create_task(self._reaper_loop())
        self._refresh_task = asyncio.create_task(self._refresh_loop())
        if self._pairing_manager:
            self._cleanup_task = asyncio.create_task(self._pairing_cleanup_loop())

        # When our own status changes, broadcast a nudge to all peers
        self._status_unsub = self._event_bus.subscribe(
            EventType.STATUS_CHANGED,
            self._on_local_status_changed,
        )

    async def on_stop(self):
        if hasattr(self, "_status_unsub"):
            self._status_unsub()

        for task in (self._reaper_task, self._refresh_task, self._cleanup_task):
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

    def _on_peer_discovered(self, data: dict, sender_ip: str, verified: bool = False):
        """Called when a new instance_id is first seen."""
        peer_id = data.get("instance_id", "")
        now = time.time()

        # Check if this peer is already trusted
        trusted = (
            self._pairing_manager.is_trusted(peer_id)
            if self._pairing_manager
            else True  # no manager = trust all (backward compat)
        )

        # Determine TLS status — trusted peers with a cert fingerprint use TLS
        tls_enabled = False
        if trusted and self._pairing_manager:
            cert_fp = self._pairing_manager.get_cert_fingerprint_for_peer(peer_id)
            tls_enabled = bool(cert_fp)

        # Check if this is a returning peer (was in verse map as offline)
        was_offline = self._verse_map.was_online(peer_id)

        peer = PeerInfo(
            instance_id=peer_id,
            hostname=data.get("hostname", ""),
            ip_address=sender_ip,
            viz_port=data.get("viz_port", settings.network.viz_port),
            first_seen=now,
            last_seen=now,
            is_trusted=trusted,
            tls_enabled=tls_enabled,
            udp_verified=verified,
        )
        self._peers[peer_id] = peer

        # Update verse map
        self._verse_map.on_peer_discovered(peer)

        if was_offline:
            # Previously offline peer returned
            self._verse_map.on_peer_returned(peer_id)
            self._emit_event(
                EventType.MESH_PEER_RETURNED,
                {
                    "instance_id": peer_id,
                    "hostname": peer.hostname,
                    "ip_address": sender_ip,
                    "is_trusted": trusted,
                },
            )
            logger.info(f"Mesh: peer returned {peer.hostname} ({peer_id[:8]}) at {sender_ip}")
        else:
            self._emit_event(
                EventType.MESH_PEER_DISCOVERED,
                {
                    "instance_id": peer_id,
                    "hostname": peer.hostname,
                    "ip_address": sender_ip,
                    "is_trusted": trusted,
                    "udp_verified": verified,
                    "tls_enabled": tls_enabled,
                },
            )
            logger.info(
                f"Mesh: discovered peer {peer.hostname} ({peer_id[:8]}) at {sender_ip} "
                f"[trusted={trusted}, udp_verified={verified}, tls={tls_enabled}]"
            )

        # Only fetch status from trusted peers
        if trusted:
            self._schedule_status_fetch(peer)

    def _on_peer_heartbeat(self, instance_id: str, sender_ip: str, verified: bool = False):
        """Called on subsequent heartbeats from a known peer."""
        peer = self._peers.get(instance_id)
        if peer:
            peer.last_seen = time.time()
            peer.ip_address = sender_ip  # may change if DHCP renews
            peer.udp_verified = verified

    def _on_nudge_received(self, peer_id: str, sender_ip: str):
        """Called when a peer broadcasts a status-changed nudge."""
        peer = self._peers.get(peer_id)
        if peer:
            peer.last_seen = time.time()
            peer.ip_address = sender_ip
            self._schedule_status_fetch(peer)
            logger.debug(f"Mesh: nudge from {peer.hostname} ({peer_id[:8]}), refetching")

    def _on_local_status_changed(self, event):
        """Our own status changed — nudge all peers to refetch."""
        if self._discovery:
            self._discovery.send_nudge()

    # ------------------------------------------------------------------
    # Background loops
    # ------------------------------------------------------------------

    async def _reaper_loop(self):
        """Periodically mark dead peers as offline (trusted) or remove (untrusted)."""
        while True:
            await asyncio.sleep(REAPER_INTERVAL)
            dead = [pid for pid, p in self._peers.items() if p.is_dead]
            for pid in dead:
                peer = self._peers.get(pid)
                if not peer:
                    continue

                if peer.is_trusted:
                    # Trusted peer: mark offline in verse map, remove from live
                    # registry, but let discovery re-discover if it returns
                    self._peers.pop(pid, None)
                    self._verse_map.on_peer_offline(pid)
                    if self._discovery:
                        self._discovery.forget_peer(pid)
                    self._emit_event(
                        EventType.MESH_PEER_OFFLINE,
                        {
                            "instance_id": pid,
                            "hostname": peer.hostname,
                        },
                    )
                    logger.info(
                        f"Mesh: peer offline {peer.hostname} ({pid[:8]}) — kept in verse map"
                    )
                else:
                    # Untrusted peer: fully remove
                    self._peers.pop(pid, None)
                    if self._discovery:
                        self._discovery.forget_peer(pid)
                    self._emit_event(
                        EventType.MESH_PEER_LOST,
                        {
                            "instance_id": pid,
                            "hostname": peer.hostname,
                        },
                    )
                    logger.info(f"Mesh: lost peer {peer.hostname} ({pid[:8]})")

    async def _refresh_loop(self):
        """Periodically fetch full status from all live peers."""
        while True:
            await asyncio.sleep(STATUS_REFRESH_INTERVAL)
            for peer in list(self._peers.values()):
                if not peer.is_dead and peer.is_trusted:
                    await self._fetch_and_update_peer(peer)

    async def _pairing_cleanup_loop(self):
        """Periodically clean up expired pairing sessions."""
        while True:
            await asyncio.sleep(30.0)
            if self._pairing_manager:
                self._pairing_manager.cleanup_expired()

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

        # Update verse map with latest capabilities
        self._verse_map.on_peer_updated(peer)

        self._emit_event(
            EventType.MESH_PEER_UPDATED,
            {
                "instance_id": peer.instance_id,
                "hostname": peer.hostname,
            },
        )

    def _schedule_status_fetch(self, peer: PeerInfo) -> None:
        """Schedule a best-effort peer status fetch when the service is running."""
        if not self._peer_client:
            return
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            logger.debug(
                "Mesh: skipping status fetch for %s because no event loop is running",
                peer.instance_id[:8],
            )
            return
        loop.create_task(self._fetch_and_update_peer(peer))

    # ------------------------------------------------------------------
    # Pairing integration
    # ------------------------------------------------------------------

    @property
    def pairing_manager(self):
        """Expose the pairing manager for routes and commands."""
        return self._pairing_manager

    @property
    def audit_log(self):
        """Expose the audit log for routes."""
        return self._audit_log

    def mark_peer_trusted(self, instance_id: str):
        """Mark a peer as trusted (after pairing completes)."""
        peer = self._peers.get(instance_id)
        if peer:
            peer.is_trusted = True
            # Check if this peer has a cert fingerprint → enable TLS
            if self._pairing_manager:
                cert_fp = self._pairing_manager.get_cert_fingerprint_for_peer(instance_id)
                if cert_fp:
                    peer.tls_enabled = True
            # Kick off an immediate status fetch now that we trust them
            self._schedule_status_fetch(peer)
            logger.info(f"Mesh: peer {peer.hostname} ({instance_id[:8]}) now trusted")

    # ------------------------------------------------------------------
    # Public query API
    # ------------------------------------------------------------------

    def get_peers(self) -> list[PeerInfo]:
        """Return all live (non-dead) peers."""
        return [p for p in self._peers.values() if not p.is_dead]

    def get_all_peers(self) -> list[PeerInfo]:
        """Return all tracked peers including stale/dead ones."""
        return list(self._peers.values())

    def get_peer(self, instance_id: str) -> PeerInfo | None:
        """Get a specific peer by instance_id."""
        return self._peers.get(instance_id)

    def find_peers_with(self, capability: str) -> list[PeerInfo]:
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
    def peer_client(self) -> PeerClient | None:
        """Expose the peer client for direct campaign operations."""
        return self._peer_client

    def find_peer_by_hostname(self, hostname: str) -> PeerInfo | None:
        """Find a live peer by hostname (case-insensitive)."""
        hostname_lower = hostname.lower()
        for p in self.get_peers():
            if p.hostname.lower() == hostname_lower:
                return p
        return None

    @property
    def verse_map(self) -> VerseMap:
        """Expose the verse map for routes and tools."""
        return self._verse_map

    @property
    def peer_count(self) -> int:
        """Number of live peers."""
        return len(self.get_peers())
