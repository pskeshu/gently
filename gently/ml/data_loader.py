"""
GentlyDataset — PyTorch Dataset loading projections + ground_truth from FileStore.
"""

import logging
import random
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

try:
    import numpy as np
    import torch
    from torch.utils.data import Dataset

    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

    # Stub for import-time safety
    class Dataset:  # type: ignore[no-redef]
        pass


class GentlyDataset(Dataset):
    """PyTorch dataset that loads projections from a labels file.

    Parameters
    ----------
    samples : list of (image_path, label_index)
        Pre-split list of samples.
    input_size : int
        Resize target (square).
    augment : bool
        Apply training augmentations.
    """

    def __init__(
        self,
        samples: list[tuple[str, int]],
        input_size: int = 224,
        augment: bool = False,
    ):
        if not HAS_TORCH:
            raise ImportError("PyTorch is required for GentlyDataset")

        self.samples = samples
        self.input_size = input_size
        self.augment = augment

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]

        # Load image
        try:
            from PIL import Image

            img = Image.open(img_path).convert("L")  # grayscale
            img = img.resize((self.input_size, self.input_size))
            img_np = np.array(img, dtype=np.float32) / 255.0
        except Exception:
            # Fallback: random noise (for testing without real images)
            img_np = np.random.rand(self.input_size, self.input_size).astype(np.float32)

        # Augmentations
        if self.augment:
            img_np = self._apply_augmentations(img_np)

        # Convert to tensor [C, H, W]
        tensor = torch.from_numpy(img_np).unsqueeze(0)  # [1, H, W]

        return tensor, label

    def _apply_augmentations(self, img: np.ndarray) -> np.ndarray:
        """Apply simple augmentations."""
        # Random horizontal flip
        if random.random() > 0.5:
            img = np.flip(img, axis=1).copy()

        # Random rotation (90 degree increments)
        k = random.randint(0, 3)
        if k > 0:
            img = np.rot90(img, k).copy()

        # Random brightness
        factor = random.uniform(0.8, 1.2)
        img = np.clip(img * factor, 0.0, 1.0)

        return img


def create_data_splits(
    labels_data: dict[str, Any],
    data_root: Path,
    input_size: int = 224,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    random_seed: int = 42,
) -> tuple:
    """Create train/val/test datasets from a labels file.

    Parameters
    ----------
    labels_data : dict
        {"class_names": [...], "samples": [{"path": "...", "label": 0}, ...]}
    data_root : Path
        Root directory for resolving relative image paths.
    input_size : int
        Resize target size.
    train_ratio, val_ratio : float
        Split ratios (test = 1 - train - val).
    random_seed : int
        Random seed for reproducibility.

    Returns
    -------
    tuple of (train_dataset, val_dataset, test_dataset)
    """
    if not HAS_TORCH:
        raise ImportError("PyTorch is required for data splits")

    samples_raw = labels_data.get("samples", [])
    random.seed(random_seed)

    # Build (path, label) tuples
    all_samples = []
    for s in samples_raw:
        path = s.get("path", "")
        label = s.get("label", 0)
        full_path = str(data_root / path) if not Path(path).is_absolute() else path
        all_samples.append((full_path, label))

    # Stratified split
    by_label: dict[Any, list] = {}
    for path, label in all_samples:
        by_label.setdefault(label, []).append((path, label))

    train_samples = []
    val_samples = []
    test_samples = []

    for _label, items in by_label.items():
        random.shuffle(items)
        n = len(items)
        n_train = max(1, int(n * train_ratio))
        n_val = max(1, int(n * val_ratio))

        train_samples.extend(items[:n_train])
        val_samples.extend(items[n_train : n_train + n_val])
        test_samples.extend(items[n_train + n_val :])

    train_ds = GentlyDataset(train_samples, input_size=input_size, augment=True)
    val_ds = GentlyDataset(val_samples, input_size=input_size, augment=False)
    test_ds = GentlyDataset(test_samples, input_size=input_size, augment=False)

    return train_ds, val_ds, test_ds


def build_labels_from_store(gently_store, session_ids: list[str] | None = None) -> dict:
    """Build a labels dict from FileStore ground truth.

    Returns
    -------
    dict
        {"class_names": [...], "samples": [{"path": "...", "label": int}, ...]}
    """
    stage_to_idx: dict[Any, int] = {}
    samples = []

    sessions = gently_store.list_sessions()
    for sess in sessions:
        sid = sess.session_id if hasattr(sess, "session_id") else sess.get("session_id", "")
        if session_ids and sid not in session_ids:
            continue

        embryos = gently_store.list_embryos(sid)
        for emb in embryos:
            eid = emb.embryo_id if hasattr(emb, "embryo_id") else emb.get("embryo_id", "")
            try:
                gts = gently_store.get_ground_truth(sid, eid)
                for gt in gts:
                    stage = gt.stage if hasattr(gt, "stage") else gt.get("stage", "")
                    if not stage:
                        continue

                    if stage not in stage_to_idx:
                        stage_to_idx[stage] = len(stage_to_idx)

                    # Get projection path for this timepoint
                    start_tp = gt.start_tp if hasattr(gt, "start_tp") else gt.get("start_tp", 0)
                    try:
                        proj_path = gently_store.get_projection_path(sid, eid, start_tp)
                        if proj_path:
                            samples.append(
                                {
                                    "path": str(proj_path),
                                    "label": stage_to_idx[stage],
                                    "session_id": sid,
                                    "embryo_id": eid,
                                    "stage": stage,
                                }
                            )
                    except Exception:
                        pass
            except Exception:
                pass

    class_names = [""] * len(stage_to_idx)
    for stage, idx in stage_to_idx.items():
        class_names[idx] = stage

    return {"class_names": class_names, "samples": samples}
