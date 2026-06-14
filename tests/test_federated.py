"""
Tests for federated averaging — weighted state dict averaging + orchestrator.
"""

from unittest.mock import MagicMock

import pytest

try:
    import torch

    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

from gently.ml.federated import FederatedOrchestrator, federated_average


class TestFederatedAverage:
    @pytest.mark.skipif(not HAS_TORCH, reason="PyTorch not installed")
    def test_equal_weights(self):
        """Two equal-weight state dicts → simple mean."""
        sd1 = {"layer.weight": torch.tensor([2.0, 4.0])}
        sd2 = {"layer.weight": torch.tensor([6.0, 8.0])}
        result = federated_average([sd1, sd2], [1.0, 1.0])
        expected = torch.tensor([4.0, 6.0])
        assert torch.allclose(result["layer.weight"].float(), expected)

    @pytest.mark.skipif(not HAS_TORCH, reason="PyTorch not installed")
    def test_weighted_average(self):
        """Weighted average: dataset A has 3x the data."""
        sd1 = {"w": torch.tensor([10.0])}
        sd2 = {"w": torch.tensor([0.0])}
        result = federated_average([sd1, sd2], [3.0, 1.0])
        # Expected: (10*3 + 0*1) / 4 = 7.5
        assert abs(result["w"].item() - 7.5) < 0.01

    @pytest.mark.skipif(not HAS_TORCH, reason="PyTorch not installed")
    def test_single_worker_passthrough(self):
        """Single worker → result is a copy of that worker's dict."""
        sd = {"w": torch.tensor([42.0])}
        result = federated_average([sd], [1.0])
        assert torch.equal(result["w"], sd["w"])
        # Should be a copy, not the same object
        result["w"][0] = 999.0
        assert sd["w"][0] == 42.0

    def test_empty_state_dicts(self):
        result = federated_average([], [])
        assert result == {}

    @pytest.mark.skipif(not HAS_TORCH, reason="PyTorch not installed")
    def test_zero_weights_fallback(self):
        """Zero total weight → treats all weights as equal."""
        sd1 = {"w": torch.tensor([2.0])}
        sd2 = {"w": torch.tensor([4.0])}
        result = federated_average([sd1, sd2], [0.0, 0.0])
        assert abs(result["w"].item() - 3.0) < 0.01

    @pytest.mark.skipif(not HAS_TORCH, reason="PyTorch not installed")
    def test_preserves_dtype(self):
        """Output dtype should match input dtype."""
        sd1 = {"w": torch.tensor([1.0], dtype=torch.float16)}
        sd2 = {"w": torch.tensor([3.0], dtype=torch.float16)}
        result = federated_average([sd1, sd2], [1.0, 1.0])
        assert result["w"].dtype == torch.float16

    @pytest.mark.skipif(not HAS_TORCH, reason="PyTorch not installed")
    def test_multiple_layers(self):
        """Average across multiple layers."""
        sd1 = {
            "conv.weight": torch.ones(3, 3),
            "fc.bias": torch.tensor([1.0, 2.0]),
        }
        sd2 = {
            "conv.weight": torch.ones(3, 3) * 3,
            "fc.bias": torch.tensor([3.0, 4.0]),
        }
        result = federated_average([sd1, sd2], [1.0, 1.0])
        assert torch.allclose(result["conv.weight"].float(), torch.ones(3, 3) * 2.0)
        assert torch.allclose(result["fc.bias"].float(), torch.tensor([2.0, 3.0]))

    @pytest.mark.skipif(not HAS_TORCH, reason="PyTorch not installed")
    def test_three_workers(self):
        """Three workers with different weights."""
        sd1 = {"w": torch.tensor([10.0])}
        sd2 = {"w": torch.tensor([20.0])}
        sd3 = {"w": torch.tensor([30.0])}
        result = federated_average([sd1, sd2, sd3], [1.0, 2.0, 3.0])
        # (10*1 + 20*2 + 30*3) / 6 = (10 + 40 + 90) / 6 = 23.33...
        expected = (10 * 1 + 20 * 2 + 30 * 3) / 6.0
        assert abs(result["w"].item() - expected) < 0.01


class TestFederatedOrchestrator:
    @pytest.mark.asyncio
    async def test_no_workers_returns_empty(self):
        """No workers → immediate return with 0 rounds."""
        vm = MagicMock()
        orch = FederatedOrchestrator(verse_map=vm)
        result = await orch.run_federated_training(
            pipeline_id="p1",
            worker_peers=[],
            initial_weights_path=MagicMock(),
            max_rounds=3,
        )
        assert result["rounds_completed"] == 0

    @pytest.mark.asyncio
    @pytest.mark.skipif(not HAS_TORCH, reason="PyTorch not installed")
    async def test_convergence_detection(self):
        """Orchestrator should detect convergence and stop early."""
        vm = MagicMock()
        peer = MagicMock()
        peer.hostname = "test-host"
        peer.instance_id = "peer-1"
        peer.ip_address = "10.0.0.1"
        peer.viz_port = 8080
        peer.is_trusted = True
        peer.tls_enabled = False

        orch = FederatedOrchestrator(verse_map=vm, peer_client=MagicMock())

        # Mock _train_workers to return results with state_dict and same accuracy
        # (convergence threshold will trigger after round 2)
        async def fake_train(*args, **kwargs):
            return [
                {
                    "val_accuracy": 0.90,
                    "dataset_size": 100,
                    "state_dict": {"w": torch.tensor([1.0])},
                }
            ]

        orch._train_workers = fake_train

        result = await orch.run_federated_training(
            pipeline_id="p1",
            worker_peers=[peer],
            initial_weights_path=MagicMock(),
            max_rounds=10,
            convergence_threshold=0.01,
        )
        # Should have stopped early (improvement < threshold after round 2)
        assert result["convergence_reached"] is True
        assert result["rounds_completed"] < 10

    @pytest.mark.asyncio
    @pytest.mark.skipif(not HAS_TORCH, reason="PyTorch not installed")
    async def test_max_rounds_limit(self):
        """Orchestrator respects max_rounds."""
        vm = MagicMock()
        orch = FederatedOrchestrator(verse_map=vm, peer_client=MagicMock())

        call_count = [0]

        async def fake_train(*args, **kwargs):
            call_count[0] += 1
            return [
                {
                    "val_accuracy": 0.5 + call_count[0] * 0.1,
                    "dataset_size": 100,
                    "state_dict": {"w": torch.tensor([float(call_count[0])])},
                }
            ]

        orch._train_workers = fake_train

        peer = MagicMock()
        peer.hostname = "host"
        result = await orch.run_federated_training(
            pipeline_id="p1",
            worker_peers=[peer],
            initial_weights_path=MagicMock(),
            max_rounds=3,
            convergence_threshold=0.0001,
        )
        assert result["rounds_completed"] == 3
