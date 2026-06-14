"""
Tests for ML data loader — GentlyDataset and data splits.
"""

from pathlib import Path
from unittest.mock import MagicMock

import pytest

try:
    import torch

    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

pytestmark = pytest.mark.skipif(not HAS_TORCH, reason="PyTorch not installed")


class TestGentlyDataset:
    def test_length(self):
        from gently.ml.data_loader import GentlyDataset

        samples = [("/fake/img1.png", 0), ("/fake/img2.png", 1)]
        ds = GentlyDataset(samples, input_size=64, augment=False)
        assert len(ds) == 2

    def test_getitem_returns_tensor_and_label(self):
        from gently.ml.data_loader import GentlyDataset

        samples = [("/nonexistent/img.png", 3)]
        ds = GentlyDataset(samples, input_size=64, augment=False)
        tensor, label = ds[0]
        assert tensor.shape == (1, 64, 64)  # [C, H, W] grayscale
        assert label == 3
        assert tensor.dtype == torch.float32

    def test_augmentations(self):
        from gently.ml.data_loader import GentlyDataset

        samples = [("/nonexistent/img.png", 0)]
        ds = GentlyDataset(samples, input_size=32, augment=True)
        # Should not crash
        tensor, label = ds[0]
        assert tensor.shape == (1, 32, 32)

    def test_empty_dataset(self):
        from gently.ml.data_loader import GentlyDataset

        ds = GentlyDataset([], input_size=64)
        assert len(ds) == 0


class TestCreateDataSplits:
    def test_basic_split(self):
        from gently.ml.data_loader import create_data_splits

        labels_data = {
            "class_names": ["early", "comma"],
            "samples": [{"path": f"img_{i}.png", "label": i % 2} for i in range(20)],
        }
        train, val, test = create_data_splits(
            labels_data,
            data_root=Path("/fake"),
            train_ratio=0.7,
            val_ratio=0.15,
            random_seed=42,
        )
        total = len(train) + len(val) + len(test)
        assert total == 20
        assert len(train) > len(val)
        assert len(train) > len(test)

    def test_stratified_preserves_labels(self):
        from gently.ml.data_loader import create_data_splits

        # 10 samples of class 0, 10 of class 1
        labels_data = {
            "class_names": ["A", "B"],
            "samples": (
                [{"path": f"a_{i}.png", "label": 0} for i in range(10)]
                + [{"path": f"b_{i}.png", "label": 1} for i in range(10)]
            ),
        }
        train, val, test = create_data_splits(
            labels_data,
            data_root=Path("/fake"),
            random_seed=42,
        )
        # Both classes should be present in train set
        train_labels = set(label for _, label in train.samples)
        assert 0 in train_labels
        assert 1 in train_labels

    def test_empty_dataset(self):
        from gently.ml.data_loader import create_data_splits

        labels_data = {"class_names": [], "samples": []}
        train, val, test = create_data_splits(
            labels_data,
            data_root=Path("/fake"),
        )
        assert len(train) == 0
        assert len(val) == 0
        assert len(test) == 0

    def test_no_overlap_between_splits(self):
        from gently.ml.data_loader import create_data_splits

        labels_data = {
            "class_names": ["X"],
            "samples": [{"path": f"x_{i}.png", "label": 0} for i in range(30)],
        }
        train, val, test = create_data_splits(
            labels_data,
            data_root=Path("/fake"),
            random_seed=42,
        )
        train_paths = {p for p, _ in train.samples}
        val_paths = {p for p, _ in val.samples}
        test_paths = {p for p, _ in test.samples}
        assert len(train_paths & val_paths) == 0
        assert len(train_paths & test_paths) == 0
        assert len(val_paths & test_paths) == 0


class TestBuildLabelsFromStore:
    def test_build_labels(self):
        from gently.ml.data_loader import build_labels_from_store

        store = MagicMock()
        sess = MagicMock()
        sess.session_id = "s1"
        store.list_sessions.return_value = [sess]

        emb = MagicMock()
        emb.embryo_id = "e1"
        store.list_embryos.return_value = [emb]

        gt = MagicMock()
        gt.stage = "early"
        gt.start_tp = 0
        store.get_ground_truth.return_value = [gt]
        store.get_projection_path.return_value = "/path/to/proj.png"

        result = build_labels_from_store(store)
        assert "early" in result["class_names"]
        assert len(result["samples"]) == 1
        assert result["samples"][0]["stage"] == "early"

    def test_build_labels_no_ground_truth(self):
        from gently.ml.data_loader import build_labels_from_store

        store = MagicMock()
        sess = MagicMock()
        sess.session_id = "s1"
        store.list_sessions.return_value = [sess]
        emb = MagicMock()
        emb.embryo_id = "e1"
        store.list_embryos.return_value = [emb]
        store.get_ground_truth.return_value = []

        result = build_labels_from_store(store)
        assert len(result["samples"]) == 0
        assert len(result["class_names"]) == 0
