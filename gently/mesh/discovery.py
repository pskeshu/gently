"""
UDP broadcast discovery layer for Gently mesh.

Uses asyncio DatagramProtocol for non-blocking broadcast/listen
on port 19547. Zero dependencies beyond Python stdlib.

Phase 2: heartbeats are HMAC-signed with a per-instance UDP key.
Replay protection via timestamp (30s window).
"""

import asyncio
import hashlib
import hmac as _hmac
import json
import logging
import socket
import time
from collections.abc import Callable

from ..settings import settings

logger = logging.getLogger(__name__)

MESH_PORT = settings.network.mesh_port
MESH_MAGIC = "GENTLY_MESH"
MESH_PROTOCOL_VERSION = 2  # bumped for signed heartbeats
BROADCAST_INTERVAL = settings.mesh.broadcast_interval_s
REPLAY_WINDOW = settings.mesh.replay_window_s


def _sign_packet(payload: dict, udp_sign_key: str) -> bytes:
    """JSON-encode a payload with HMAC signature."""
    # Serialize without sig field
    raw = json.dumps(payload, separators=(",", ":"), sort_keys=True)
    sig = _hmac.new(
        udp_sign_key.encode(),
        raw.encode(),
        hashlib.sha256,
    ).hexdigest()
    payload["sig"] = sig
    return json.dumps(payload).encode("utf-8")


def _verify_sig(msg: dict, udp_sign_key: str) -> bool:
    """Verify HMAC signature on a received packet."""
    sig = msg.pop("sig", "")
    if not sig:
        return False
    raw = json.dumps(msg, separators=(",", ":"), sort_keys=True)
    expected = _hmac.new(
        udp_sign_key.encode(),
        raw.encode(),
        hashlib.sha256,
    ).hexdigest()
    return _hmac.compare_digest(sig, expected)


class _MeshProtocol(asyncio.DatagramProtocol):
    """asyncio datagram protocol for mesh heartbeats."""

    def __init__(
        self,
        instance_id: str,
        on_peer_discovered: Callable[[dict, str, bool], None],
        on_peer_heartbeat: Callable[[str, str, bool], None],
        on_nudge_received: Callable[[str, str], None] = lambda pid, ip: None,
        pairing_manager=None,
        audit_log=None,
    ):
        self.instance_id = instance_id
        self._on_peer_discovered = on_peer_discovered
        self._on_peer_heartbeat = on_peer_heartbeat
        self._on_nudge_received = on_nudge_received
        self._pairing_manager = pairing_manager
        self._audit_log = audit_log
        self._known_ids: set = set()
        self.transport: asyncio.DatagramTransport | None = None

    def connection_made(self, transport: asyncio.BaseTransport) -> None:
        # DatagramProtocol always receives a DatagramTransport here; narrow for
        # the typed attribute (and matches the asyncio.BaseProtocol signature).
        assert isinstance(transport, asyncio.DatagramTransport)
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
        msg_type = msg.get("msg_type", "heartbeat")

        # Replay protection: reject packets with stale timestamps
        ts = msg.get("ts", 0)
        if ts and abs(time.time() - ts) > REPLAY_WINDOW:
            logger.debug(f"Mesh: rejected stale packet from {peer_id[:8]} (ts={ts})")
            if self._audit_log:
                from .audit import AuditEvent

                self._audit_log.log(
                    AuditEvent.REPLAY_REJECTED,
                    outcome="deny",
                    peer_id=peer_id,
                    ip=sender_ip,
                    detail=f"ts_delta={abs(time.time() - ts):.1f}s",
                )
            return

        # Verify HMAC signature if we have a key for this peer
        verified = False
        if self._pairing_manager and "sig" in msg:
            udp_key = self._pairing_manager.get_udp_key_for_peer(peer_id)
            if udp_key:
                # Make a copy to verify (pop removes sig)
                msg_copy = dict(msg)
                verified = _verify_sig(msg_copy, udp_key)
                if not verified:
                    logger.debug(f"Mesh: bad signature from {peer_id[:8]}")
                    if self._audit_log:
                        from .audit import AuditEvent

                        self._audit_log.log(
                            AuditEvent.SIG_INVALID,
                            outcome="deny",
                            peer_id=peer_id,
                            ip=sender_ip,
                        )

        if msg_type == "nudge":
            self._on_nudge_received(peer_id, sender_ip)
            return

        if peer_id not in self._known_ids:
            self._known_ids.add(peer_id)
            self._on_peer_discovered(msg, sender_ip, verified)
        else:
            self._on_peer_heartbeat(peer_id, sender_ip, verified)

    def error_received(self, exc: Exception):
        logger.debug(f"Mesh UDP error: {exc}")

    def connection_lost(self, exc: Exception | None):
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
        viz_port: int = settings.network.viz_port,
        mesh_port: int = MESH_PORT,
        pairing_manager=None,
        audit_log=None,
    ):
        self.instance_id = instance_id
        self.hostname = hostname
        self.viz_port = viz_port
        self.mesh_port = mesh_port
        self._pairing_manager = pairing_manager
        self._audit_log = audit_log

        self._protocol: _MeshProtocol | None = None
        self._transport: asyncio.DatagramTransport | None = None
        self._broadcast_task: asyncio.Task | None = None
        self._running = False

        # Callbacks — set by MeshService before start()
        self.on_peer_discovered: Callable[[dict, str, bool], None] = lambda d, ip, v: None
        self.on_peer_heartbeat: Callable[[str, str, bool], None] = lambda id, ip, v: None
        self.on_nudge_received: Callable[[str, str], None] = lambda pid, ip: None

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
            on_nudge_received=self.on_nudge_received,
            pairing_manager=self._pairing_manager,
            audit_log=self._audit_log,
        )

        self._transport, _ = await loop.create_datagram_endpoint(
            lambda: self._protocol,
            sock=sock,
        )

        self._running = True
        self._broadcast_task = asyncio.create_task(self._broadcast_loop())
        logger.info(
            f"Mesh discovery started on port {self.mesh_port} (instance={self.instance_id[:8]})"
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

    def send_nudge(self):
        """Broadcast a nudge packet telling peers to refetch our status."""
        if not self._transport:
            return
        payload = {
            "magic": MESH_MAGIC,
            "version": MESH_PROTOCOL_VERSION,
            "msg_type": "nudge",
            "instance_id": self.instance_id,
            "ts": time.time(),
        }

        # Sign if we have a UDP key
        udp_key = self._pairing_manager.udp_sign_key if self._pairing_manager else ""
        if udp_key:
            packet = _sign_packet(payload, udp_key)
        else:
            packet = json.dumps(payload).encode("utf-8")

        try:
            self._transport.sendto(packet, ("255.255.255.255", self.mesh_port))
        except OSError as e:
            logger.debug(f"Mesh nudge broadcast failed: {e}")

    async def _broadcast_loop(self):
        """Send heartbeat every BROADCAST_INTERVAL seconds."""
        udp_key = self._pairing_manager.udp_sign_key if self._pairing_manager else ""

        while self._running:
            payload = {
                "magic": MESH_MAGIC,
                "version": MESH_PROTOCOL_VERSION,
                "instance_id": self.instance_id,
                "hostname": self.hostname,
                "viz_port": self.viz_port,
                "ts": time.time(),
            }

            if udp_key:
                heartbeat = _sign_packet(payload, udp_key)
            else:
                heartbeat = json.dumps(payload).encode("utf-8")

            try:
                if self._transport:
                    self._transport.sendto(heartbeat, ("255.255.255.255", self.mesh_port))
            except OSError as e:
                logger.debug(f"Mesh broadcast failed: {e}")

            await asyncio.sleep(BROADCAST_INTERVAL)
