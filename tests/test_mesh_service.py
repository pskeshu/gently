"""
Tests for MeshService — peer lifecycle, reaper, queries.
"""

import time
from unittest.mock import MagicMock

import pytest

from gently.mesh.mesh_service import MeshService
from gently.mesh.models import PeerCapability


@pytest.fixture
def mesh_service(tmp_path):
    """Create a MeshService with mock dependencies."""
    config_dir = tmp_path / "config"
    config_dir.mkdir()
    svc = MeshService(
        instance_id="test-instance-001",
        viz_port=8080,
        capability_provider=lambda: {"has_gpu": True, "organism": "c_elegans"},
        status_provider=lambda: {"active_session": None, "mode": "idle"},
        mesh_port=19999,  # non-default to avoid conflicts
        pairing_manager=None,
        config_dir=config_dir,
    )
    return svc


class TestMeshServiceInit:
    def test_instance_id(self, mesh_service):
        assert mesh_service.instance_id == "test-instance-001"

    def test_initial_empty_peers(self, mesh_service):
        assert mesh_service.get_peers() == []
        assert mesh_service.peer_count == 0

    def test_verse_map_created(self, mesh_service):
        assert mesh_service.verse_map is not None


class TestPeerDiscovery:
    def test_on_peer_discovered_creates_peer(self, mesh_service):
        data = {
            "instance_id": "peer-abc-001",
            "hostname": "microscope-1",
            "viz_port": 8080,
        }
        mesh_service._on_peer_discovered(data, "192.168.1.10")
        peer = mesh_service.get_peer("peer-abc-001")
        assert peer is not None
        assert peer.hostname == "microscope-1"
        assert peer.ip_address == "192.168.1.10"

    def test_on_peer_discovered_trusted_without_manager(self, mesh_service):
        """No pairing manager = trust all (backward compat)."""
        data = {"instance_id": "peer-abc-002", "hostname": "host2"}
        mesh_service._on_peer_discovered(data, "10.0.0.1")
        peer = mesh_service.get_peer("peer-abc-002")
        assert peer.is_trusted is True

    def test_on_peer_discovered_with_pairing_manager(self, tmp_path):
        config_dir = tmp_path / "config"
        config_dir.mkdir()
        mgr = MagicMock()
        mgr.is_trusted.return_value = False
        mgr.get_cert_fingerprint_for_peer.return_value = None
        svc = MeshService(
            instance_id="test-id",
            pairing_manager=mgr,
            config_dir=config_dir,
        )
        data = {"instance_id": "untrusted-peer", "hostname": "stranger"}
        svc._on_peer_discovered(data, "10.0.0.2")
        peer = svc.get_peer("untrusted-peer")
        assert peer.is_trusted is False


class TestPeerHeartbeat:
    def test_heartbeat_updates_last_seen(self, mesh_service):
        data = {"instance_id": "peer-hb-001", "hostname": "host1"}
        mesh_service._on_peer_discovered(data, "10.0.0.1")
        old_seen = mesh_service.get_peer("peer-hb-001").last_seen
        time.sleep(0.01)
        mesh_service._on_peer_heartbeat("peer-hb-001", "10.0.0.1")
        assert mesh_service.get_peer("peer-hb-001").last_seen >= old_seen

    def test_heartbeat_updates_ip(self, mesh_service):
        data = {"instance_id": "peer-hb-002", "hostname": "host2"}
        mesh_service._on_peer_discovered(data, "10.0.0.1")
        mesh_service._on_peer_heartbeat("peer-hb-002", "10.0.0.99")
        assert mesh_service.get_peer("peer-hb-002").ip_address == "10.0.0.99"

    def test_heartbeat_unknown_peer_ignored(self, mesh_service):
        # Should not crash when heartbeat arrives for unknown peer
        mesh_service._on_peer_heartbeat("nonexistent", "1.2.3.4")


class TestPeerQueries:
    def test_get_peers_excludes_dead(self, mesh_service):
        data = {"instance_id": "peer-q-001", "hostname": "host1"}
        mesh_service._on_peer_discovered(data, "10.0.0.1")
        # Peer is alive — should appear
        assert len(mesh_service.get_peers()) == 1
        # Make it dead by setting last_seen far in the past
        mesh_service._peers["peer-q-001"].last_seen = time.time() - 300
        assert len(mesh_service.get_peers()) == 0

    def test_get_all_peers_includes_dead(self, mesh_service):
        data = {"instance_id": "peer-q-002", "hostname": "host2"}
        mesh_service._on_peer_discovered(data, "10.0.0.1")
        mesh_service._peers["peer-q-002"].last_seen = time.time() - 300
        assert len(mesh_service.get_all_peers()) == 1

    def test_find_peers_with_capability(self, mesh_service):
        data = {"instance_id": "gpu-peer", "hostname": "gpu-host"}
        mesh_service._on_peer_discovered(data, "10.0.0.1")
        peer = mesh_service.get_peer("gpu-peer")
        peer.capabilities = PeerCapability(has_gpu=True, gpu_name="A5000")
        results = mesh_service.find_peers_with("has_gpu")
        assert len(results) == 1
        assert results[0].instance_id == "gpu-peer"

    def test_find_peers_with_no_match(self, mesh_service):
        data = {"instance_id": "cpu-peer", "hostname": "cpu-host"}
        mesh_service._on_peer_discovered(data, "10.0.0.1")
        results = mesh_service.find_peers_with("has_gpu")
        assert len(results) == 0

    def test_find_peer_by_hostname(self, mesh_service):
        data = {"instance_id": "host-peer", "hostname": "MY-WORKSTATION"}
        mesh_service._on_peer_discovered(data, "10.0.0.1")
        # Case-insensitive search
        result = mesh_service.find_peer_by_hostname("my-workstation")
        assert result is not None
        assert result.instance_id == "host-peer"

    def test_find_peer_by_hostname_not_found(self, mesh_service):
        result = mesh_service.find_peer_by_hostname("nonexistent")
        assert result is None


class TestLocalInfo:
    def test_get_local_info(self, mesh_service):
        info = mesh_service.get_local_info()
        assert info["instance_id"] == "test-instance-001"
        assert info["capabilities"]["has_gpu"] is True
        assert info["status"]["mode"] == "idle"


class TestMarkTrusted:
    def test_mark_peer_trusted(self, mesh_service):
        data = {"instance_id": "untrusted-peer", "hostname": "stranger"}
        mesh_service._on_peer_discovered(data, "10.0.0.1")
        peer = mesh_service.get_peer("untrusted-peer")
        peer.is_trusted = False
        mesh_service.mark_peer_trusted("untrusted-peer")
        assert mesh_service.get_peer("untrusted-peer").is_trusted is True
