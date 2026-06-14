"""
Tests for DynamicCapabilityProvider.
"""

from unittest.mock import MagicMock, patch

from gently.mesh.capability_provider import DynamicCapabilityProvider
from gently.mesh.models import PeerRole


class TestDynamicCapabilities:
    def test_basic_static_caps(self):
        provider = DynamicCapabilityProvider(
            static_caps={"has_microscope": True, "organism": "c_elegans"},
        )
        caps = provider()
        assert caps["has_microscope"] is True
        assert caps["organism"] == "c_elegans"
        assert isinstance(caps["roles"], list)
        assert isinstance(caps["datasets"], list)

    def test_no_gpu_detection(self):
        """Without torch, gpus list should be empty."""
        provider = DynamicCapabilityProvider()
        caps = provider()
        assert isinstance(caps["gpus"], list)
        # has_gpu depends on whether torch + CUDA is available
        # In test environment, likely no GPU
        assert isinstance(caps["has_gpu"], bool)

    @patch("gently.mesh.capability_provider._detect_gpus")
    def test_gpu_detected(self, mock_detect):
        from gently.mesh.models import GpuInfo

        mock_detect.return_value = [
            GpuInfo(device_index=0, name="A5000", vram_gb=24.0),
        ]
        provider = DynamicCapabilityProvider()
        provider._gpus = mock_detect()
        caps = provider()
        assert caps["has_gpu"] is True
        assert caps["gpu_name"] == "A5000"
        assert caps["gpu_vram_gb"] == 24.0
        assert len(caps["gpus"]) == 1

    def test_cpu_cores_populated(self):
        provider = DynamicCapabilityProvider()
        caps = provider()
        assert caps["cpu_cores"] > 0


class TestRoleLogic:
    @patch("gently.mesh.capability_provider._detect_gpus")
    def test_ml_trainer_role_with_gpu(self, mock_detect):
        from gently.mesh.models import GpuInfo

        mock_detect.return_value = [GpuInfo(name="A5000")]
        provider = DynamicCapabilityProvider()
        provider._gpus = mock_detect()
        caps = provider()
        assert PeerRole.ML_TRAINER.value in caps["roles"]

    def test_microscope_role_from_static(self):
        provider = DynamicCapabilityProvider(
            static_caps={"has_microscope": True},
        )
        caps = provider()
        assert PeerRole.MICROSCOPE_CONTROLLER.value in caps["roles"]

    def test_planner_always_present(self):
        provider = DynamicCapabilityProvider()
        caps = provider()
        assert PeerRole.PLANNER.value in caps["roles"]

    def test_data_server_with_datasets(self):
        mock_store = MagicMock()
        mock_session = MagicMock()
        mock_session.session_id = "s1"
        mock_session.name = "Test"
        mock_store.list_sessions.return_value = [mock_session]

        mock_embryo = MagicMock()
        mock_embryo.embryo_id = "e1"
        mock_store.list_embryos.return_value = [mock_embryo]
        mock_store.list_volumes.return_value = [MagicMock()]
        mock_store.get_ground_truth.return_value = []

        provider = DynamicCapabilityProvider(gently_store=mock_store)
        caps = provider()
        assert PeerRole.DATA_SERVER.value in caps["roles"]
        assert len(caps["datasets"]) == 1
