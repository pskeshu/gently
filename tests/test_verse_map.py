"""
Tests for VerseMap — persistent topology map.
"""

import pytest

from gently.mesh.models import (
    DatasetAdvertisement,
    PeerCapability,
    PeerInfo,
)
from gently.mesh.verse_map import VerseMap


@pytest.fixture
def verse_map(config_dir):
    return VerseMap(config_dir)


def _make_peer(
    instance_id="p1",
    hostname="lab-pc",
    ip="192.168.1.10",
    has_gpu=False,
    has_microscope=False,
    roles=None,
    datasets=None,
):
    caps = PeerCapability(
        has_gpu=has_gpu,
        has_microscope=has_microscope,
        roles=roles or [],
        datasets=datasets or [],
    )
    return PeerInfo(
        instance_id=instance_id,
        hostname=hostname,
        ip_address=ip,
        capabilities=caps,
        is_trusted=True,
    )


class TestVerseMapPersistence:
    def test_save_and_load(self, config_dir):
        vm1 = VerseMap(config_dir)
        peer = _make_peer("p1", "lab-pc")
        vm1.on_peer_discovered(peer)
        assert len(vm1.get_all_peers()) == 1

        # Create new instance from same config dir (simulate restart)
        vm2 = VerseMap(config_dir)
        assert len(vm2.get_all_peers()) == 1
        assert vm2.get_peer("p1").hostname == "lab-pc"

    def test_json_file_exists(self, config_dir):
        vm = VerseMap(config_dir)
        peer = _make_peer()
        vm.on_peer_discovered(peer)
        assert (config_dir / "mesh_verse_map.json").exists()

    def test_empty_load(self, config_dir):
        vm = VerseMap(config_dir)
        assert len(vm.get_all_peers()) == 0


class TestPeerTracking:
    def test_discover_new_peer(self, verse_map):
        peer = _make_peer("p1")
        verse_map.on_peer_discovered(peer)
        assert len(verse_map.get_all_peers()) == 1
        assert verse_map.get_peer("p1").online is True

    def test_mark_offline_keeps_peer(self, verse_map):
        peer = _make_peer("p1")
        verse_map.on_peer_discovered(peer)
        verse_map.on_peer_offline("p1")
        assert len(verse_map.get_all_peers()) == 1  # still in map
        assert verse_map.get_peer("p1").online is False

    def test_peer_returned(self, verse_map):
        peer = _make_peer("p1")
        verse_map.on_peer_discovered(peer)
        verse_map.on_peer_offline("p1")
        assert verse_map.get_peer("p1").online is False

        verse_map.on_peer_returned("p1")
        assert verse_map.get_peer("p1").online is True

    def test_rediscover_known_peer(self, verse_map):
        peer = _make_peer("p1", ip="192.168.1.10")
        verse_map.on_peer_discovered(peer)
        verse_map.on_peer_offline("p1")

        # Peer comes back with new IP
        peer2 = _make_peer("p1", ip="192.168.1.20")
        verse_map.on_peer_discovered(peer2)
        assert verse_map.get_peer("p1").online is True
        assert verse_map.get_peer("p1").ip_address == "192.168.1.20"

    def test_update_capabilities(self, verse_map):
        peer = _make_peer("p1", has_gpu=False)
        verse_map.on_peer_discovered(peer)

        # Update with GPU capabilities
        peer.capabilities = PeerCapability(has_gpu=True, gpu_name="A5000", roles=["ml_trainer"])
        verse_map.on_peer_updated(peer)

        pp = verse_map.get_peer("p1")
        assert pp.capabilities.has_gpu is True
        assert pp.roles == ["ml_trainer"]

    def test_is_known_peer(self, verse_map):
        assert not verse_map.is_known_peer("p1")
        verse_map.on_peer_discovered(_make_peer("p1"))
        assert verse_map.is_known_peer("p1")

    def test_was_online(self, verse_map):
        peer = _make_peer("p1")
        verse_map.on_peer_discovered(peer)
        assert not verse_map.was_online("p1")  # currently online, not "was"
        verse_map.on_peer_offline("p1")
        assert verse_map.was_online("p1")  # now offline, "was" online


class TestRouting:
    def test_find_gpu_peers_online_first(self, verse_map):
        # Online GPU peer
        verse_map.on_peer_discovered(_make_peer("p1", "online-gpu", has_gpu=True))
        # Offline GPU peer
        verse_map.on_peer_discovered(_make_peer("p2", "offline-gpu", has_gpu=True))
        verse_map.on_peer_offline("p2")

        gpu_peers = verse_map.find_gpu_peers()
        assert len(gpu_peers) == 2
        assert gpu_peers[0].instance_id == "p1"  # online first
        assert gpu_peers[1].instance_id == "p2"

    def test_find_microscope_peers(self, verse_map):
        verse_map.on_peer_discovered(_make_peer("p1", has_microscope=True))
        verse_map.on_peer_discovered(_make_peer("p2", has_gpu=True))

        micro_peers = verse_map.find_microscope_peers()
        assert len(micro_peers) == 1
        assert micro_peers[0].instance_id == "p1"

    def test_find_data_peers_by_session(self, verse_map):
        ds = DatasetAdvertisement(session_id="s1", embryo_count=10)
        verse_map.on_peer_discovered(_make_peer("p1", datasets=[ds]))
        verse_map.on_peer_discovered(_make_peer("p2"))

        # Find peers with session s1
        found = verse_map.find_data_peers(session_id="s1")
        assert len(found) == 1
        assert found[0].instance_id == "p1"

        # Find any data peers
        found_all = verse_map.find_data_peers()
        assert len(found_all) == 1

    def test_find_resource_by_role(self, verse_map):
        verse_map.on_peer_discovered(_make_peer("p1", roles=["ml_trainer"]))
        verse_map.on_peer_discovered(_make_peer("p2"))

        found = verse_map.find_resource("ml_trainer")
        assert len(found) == 1
        assert found[0].instance_id == "p1"

    def test_find_resource_by_capability_flag(self, verse_map):
        verse_map.on_peer_discovered(_make_peer("p1", has_gpu=True))
        found = verse_map.find_resource("has_gpu")
        assert len(found) == 1

    def test_get_online_and_offline(self, verse_map):
        verse_map.on_peer_discovered(_make_peer("p1"))
        verse_map.on_peer_discovered(_make_peer("p2"))
        verse_map.on_peer_offline("p2")

        assert len(verse_map.get_online_peers()) == 1
        assert len(verse_map.get_offline_peers()) == 1
        assert len(verse_map.get_all_peers()) == 2
