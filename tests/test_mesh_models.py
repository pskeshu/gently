"""
Tests for mesh data models — PeerCapability, PeerStatus, PeerInfo,
GpuInfo, DatasetAdvertisement, PersistedPeer.
"""

import time

from gently.mesh.models import (
    DatasetAdvertisement,
    GpuInfo,
    PeerCapability,
    PeerInfo,
    PeerRole,
    PeerStatus,
    PersistedPeer,
)


class TestGpuInfo:
    def test_round_trip(self):
        gpu = GpuInfo(
            device_index=0,
            name="A5000",
            vram_gb=24.0,
            compute_capability="8.6",
            utilization_pct=45.0,
            memory_used_gb=8.2,
        )
        d = gpu.to_dict()
        gpu2 = GpuInfo.from_dict(d)
        assert gpu2.name == "A5000"
        assert gpu2.vram_gb == 24.0
        assert gpu2.compute_capability == "8.6"

    def test_defaults(self):
        gpu = GpuInfo.from_dict({})
        assert gpu.device_index == 0
        assert gpu.name == ""
        assert gpu.vram_gb == 0.0


class TestDatasetAdvertisement:
    def test_round_trip(self):
        ds = DatasetAdvertisement(
            session_id="s1",
            session_name="Run 1",
            embryo_count=12,
            volume_count=120,
            has_ground_truth=True,
            ground_truth_count=50,
            stages_covered=["early", "comma"],
            total_size_gb=2.5,
        )
        d = ds.to_dict()
        ds2 = DatasetAdvertisement.from_dict(d)
        assert ds2.session_id == "s1"
        assert ds2.stages_covered == ["early", "comma"]
        assert ds2.ground_truth_count == 50


class TestPeerCapability:
    def test_round_trip_legacy_fields(self):
        cap = PeerCapability(has_microscope=True, has_gpu=True, gpu_name="A5000", gpu_vram_gb=24.0)
        d = cap.to_dict()
        cap2 = PeerCapability.from_dict(d)
        assert cap2.has_microscope is True
        assert cap2.has_gpu is True
        assert cap2.gpu_name == "A5000"

    def test_backward_compat_missing_new_fields(self):
        """Old peers that don't send new fields should get defaults."""
        old_dict = {"has_microscope": True, "has_gpu": False}
        cap = PeerCapability.from_dict(old_dict)
        assert cap.gpus == []
        assert cap.roles == []
        assert cap.datasets == []
        assert cap.microscope_connected is False
        assert cap.cpu_cores == 0

    def test_enhanced_fields(self):
        cap = PeerCapability(
            gpus=[GpuInfo(name="A5000", vram_gb=24.0)],
            roles=["ml_trainer", "planner"],
            datasets=[DatasetAdvertisement(session_id="s1", embryo_count=12)],
            microscope_connected=True,
            cpu_cores=16,
            ram_gb=64.0,
        )
        d = cap.to_dict()
        cap2 = PeerCapability.from_dict(d)
        assert len(cap2.gpus) == 1
        assert cap2.gpus[0].name == "A5000"
        assert cap2.roles == ["ml_trainer", "planner"]
        assert len(cap2.datasets) == 1
        assert cap2.microscope_connected is True
        assert cap2.cpu_cores == 16


class TestPeerStatus:
    def test_round_trip(self):
        status = PeerStatus(
            session_id="s1",
            acquisition_status="running",
            embryo_count=5,
            version="0.9.2",
        )
        d = status.to_dict()
        status2 = PeerStatus.from_dict(d)
        assert status2.session_id == "s1"
        assert status2.acquisition_status == "running"

    def test_defaults(self):
        status = PeerStatus.from_dict({})
        assert status.acquisition_status == "idle"
        assert status.agent_mode == "run"


class TestPeerInfo:
    def test_round_trip(self):
        peer = PeerInfo(
            instance_id="abc123",
            hostname="lab-pc",
            ip_address="192.168.1.10",
            viz_port=8080,
        )
        d = peer.to_dict()
        peer2 = PeerInfo.from_dict(d)
        assert peer2.instance_id == "abc123"
        assert peer2.hostname == "lab-pc"

    def test_base_url_http(self):
        peer = PeerInfo(ip_address="10.0.0.1", viz_port=8080, tls_enabled=False)
        assert peer.base_url == "http://10.0.0.1:8080"

    def test_base_url_https(self):
        peer = PeerInfo(ip_address="10.0.0.1", viz_port=8080, tls_enabled=True)
        assert peer.base_url == "https://10.0.0.1:8080"

    def test_is_stale_and_dead(self):
        peer = PeerInfo(last_seen=time.time() - 100)
        assert peer.is_stale
        assert peer.is_dead


class TestPeerRole:
    def test_values(self):
        assert PeerRole.MICROSCOPE_CONTROLLER.value == "microscope_controller"
        assert PeerRole.ML_TRAINER.value == "ml_trainer"
        assert PeerRole.DATA_SERVER.value == "data_server"
        assert PeerRole.PLANNER.value == "planner"


class TestPersistedPeer:
    def test_round_trip(self):
        peer = PersistedPeer(
            instance_id="abc123",
            hostname="lab-pc",
            ip_address="192.168.1.10",
            online=True,
            roles=["ml_trainer"],
            datasets=[DatasetAdvertisement(session_id="s1")],
        )
        d = peer.to_dict()
        peer2 = PersistedPeer.from_dict(d)
        assert peer2.instance_id == "abc123"
        assert peer2.online is True
        assert peer2.roles == ["ml_trainer"]
        assert len(peer2.datasets) == 1

    def test_from_peer_info(self):
        info = PeerInfo(
            instance_id="abc123",
            hostname="lab-pc",
            ip_address="192.168.1.10",
            is_trusted=True,
            capabilities=PeerCapability(
                has_gpu=True,
                roles=["ml_trainer"],
                datasets=[DatasetAdvertisement(session_id="s1")],
            ),
        )
        pp = PersistedPeer.from_peer_info(info)
        assert pp.instance_id == "abc123"
        assert pp.online is True
        assert pp.roles == ["ml_trainer"]
        assert len(pp.datasets) == 1

    def test_offline_backward_compat(self):
        """Old verse map JSON without new fields loads gracefully."""
        old_dict = {
            "instance_id": "old-peer",
            "hostname": "old-pc",
        }
        pp = PersistedPeer.from_dict(old_dict)
        assert pp.instance_id == "old-peer"
        assert pp.online is True  # default
        assert pp.roles == []
        assert pp.datasets == []
