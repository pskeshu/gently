"""
VerseMap — Persistent topology map of the gently mesh.

Peers survive offline/restarts. Backed by a JSON file in the config directory
(follows the mesh_trusted_peers.json pattern).
"""

import json
import logging
import time
from pathlib import Path

from .models import (
    PeerInfo,
    PersistedPeer,
)

logger = logging.getLogger(__name__)


class VerseMap:
    """Persistent topology map of the gently mesh. Survives restarts."""

    def __init__(self, config_dir: Path):
        self._config_dir = config_dir
        self._map_file = config_dir / "mesh_verse_map.json"
        self._peers: dict[str, PersistedPeer] = {}
        self._load()

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def _load(self):
        """Load verse map from disk."""
        if not self._map_file.exists():
            return
        try:
            data = json.loads(self._map_file.read_text())
            for entry in data:
                peer = PersistedPeer.from_dict(entry)
                if peer.instance_id:
                    self._peers[peer.instance_id] = peer
            logger.info(f"VerseMap: loaded {len(self._peers)} peers from disk")
        except Exception as e:
            logger.warning(f"VerseMap: failed to load: {e}")

    def _save(self):
        """Persist verse map to disk."""
        data = [peer.to_dict() for peer in self._peers.values()]
        try:
            self._config_dir.mkdir(parents=True, exist_ok=True)
            self._map_file.write_text(json.dumps(data, indent=2))
        except Exception as e:
            logger.error(f"VerseMap: failed to save: {e}")

    # ------------------------------------------------------------------
    # Peer lifecycle events (called by MeshService)
    # ------------------------------------------------------------------

    def on_peer_discovered(self, peer: PeerInfo):
        """A new peer was discovered on the mesh (or a known one re-appeared)."""
        existing = self._peers.get(peer.instance_id)
        if existing:
            # Known peer returned — update fields, mark online
            existing.online = True
            existing.last_online = time.time()
            existing.last_seen = peer.last_seen
            existing.ip_address = peer.ip_address
            existing.viz_port = peer.viz_port
            existing.is_trusted = peer.is_trusted
            existing.tls_enabled = peer.tls_enabled
        else:
            # Brand new peer
            self._peers[peer.instance_id] = PersistedPeer.from_peer_info(peer)
        self._save()

    def on_peer_updated(self, peer: PeerInfo):
        """Peer capabilities/status refreshed via HTTP fetch."""
        existing = self._peers.get(peer.instance_id)
        if existing:
            existing.capabilities = peer.capabilities
            existing.status = peer.status
            existing.last_seen = peer.last_seen
            existing.roles = peer.capabilities.roles
            existing.datasets = peer.capabilities.datasets
            self._save()

    def on_peer_offline(self, instance_id: str):
        """Peer went offline (dead timeout). Keep in map, mark offline."""
        existing = self._peers.get(instance_id)
        if existing:
            existing.online = False
            self._save()

    def on_peer_returned(self, instance_id: str):
        """A previously offline peer came back online."""
        existing = self._peers.get(instance_id)
        if existing:
            existing.online = True
            existing.last_online = time.time()
            self._save()

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    def get_all_peers(self) -> list[PersistedPeer]:
        """All peers, online and offline."""
        return list(self._peers.values())

    def get_online_peers(self) -> list[PersistedPeer]:
        """Only online peers."""
        return [p for p in self._peers.values() if p.online]

    def get_offline_peers(self) -> list[PersistedPeer]:
        """Only offline peers."""
        return [p for p in self._peers.values() if not p.online]

    def get_peer(self, instance_id: str) -> PersistedPeer | None:
        """Get a specific peer by instance_id."""
        return self._peers.get(instance_id)

    def is_known_peer(self, instance_id: str) -> bool:
        """Check if we've ever seen this peer."""
        return instance_id in self._peers

    def was_online(self, instance_id: str) -> bool:
        """Check if a known peer was previously online (now offline)."""
        peer = self._peers.get(instance_id)
        return peer is not None and not peer.online

    # ------------------------------------------------------------------
    # Route-finding: sorted online-first, then by last_seen recency
    # ------------------------------------------------------------------

    def _sorted_peers(self, peers: list[PersistedPeer]) -> list[PersistedPeer]:
        """Sort peers: online first, then by last_seen descending."""
        return sorted(peers, key=lambda p: (not p.online, -p.last_seen))

    def find_gpu_peers(self) -> list[PersistedPeer]:
        """Find peers with GPU capability, best candidates first."""
        results = [p for p in self._peers.values() if p.capabilities.has_gpu or p.capabilities.gpus]
        return self._sorted_peers(results)

    def find_microscope_peers(self) -> list[PersistedPeer]:
        """Find peers with microscope capability."""
        results = [
            p
            for p in self._peers.values()
            if p.capabilities.has_microscope or p.capabilities.microscope_connected
        ]
        return self._sorted_peers(results)

    def find_data_peers(self, session_id: str | None = None) -> list[PersistedPeer]:
        """Find peers with data, optionally filtering by session."""
        results = []
        for p in self._peers.values():
            if not p.datasets and not p.capabilities.datasets:
                continue
            if session_id:
                all_datasets = p.datasets + p.capabilities.datasets
                if not any(d.session_id == session_id for d in all_datasets):
                    continue
            results.append(p)
        return self._sorted_peers(results)

    def find_resource(self, capability: str) -> list[PersistedPeer]:
        """Find peers matching a generic capability attribute.

        The capability string is checked against:
        - PeerCapability boolean flags (e.g. "has_gpu", "has_microscope")
        - PeerRole values in the roles list (e.g. "ml_trainer")
        """
        results = []
        for p in self._peers.values():
            # Check bool attrs on capabilities
            if getattr(p.capabilities, capability, False):
                results.append(p)
                continue
            # Check roles
            if capability in p.roles or capability in p.capabilities.roles:
                results.append(p)
        return self._sorted_peers(results)
