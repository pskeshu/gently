"""
Data models for the Visualization Server
=========================================

Dataclasses and type constants used across the visualization package.
"""

from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np

# Data types for routing to tabs
CALIBRATION_TYPES = {
    "focus_sweep",
    "focus_plot",
    "edge_detection",
    "calibration_summary",
    "focus_snap",
    "focus_coarse",
    "focus_curve",
    "focus_assess",
}

VOLUME_TYPES = {"volume", "volume_projection", "z_stack", "timelapse"}

# CV/Analysis types - shown in a separate "Analysis" category within Calibration
ANALYSIS_TYPES = {
    "segmentation",
    "detection",
    "classification",
    "tracking",
    # CV agent visualization types
    "roi_detection",
    "cropped_roi",
    "vision_prepared",
    "timeline",
    "cv_visualization",
}

# 3D types that support Z-slider browsing
VOLUME_3D_TYPES = {"segmentation_3d"}


@dataclass
class ClientInfo:
    """Information about a connected WebSocket client for presence tracking"""

    client_id: str
    name: str
    color: str  # Hex color for avatar background
    connected_at: str


@dataclass
class Volume3DData:
    """Container for 3D volume data with segmentation overlay"""

    uid: str
    data_type: str
    timestamp: str
    volume: np.ndarray  # Original volume (Z, H, W)
    masks: np.ndarray  # Segmentation masks (Z, H, W)
    colors: np.ndarray  # Cell colors (num_labels, 3)
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def num_slices(self) -> int:
        return self.volume.shape[0]

    @property
    def shape(self) -> tuple:
        return self.volume.shape

    def get_slice_overlay(self, z: int, alpha: float = 0.4) -> np.ndarray:
        """Get RGB overlay for a specific Z-slice"""
        z = max(0, min(z, self.num_slices - 1))

        vol_slice = self.volume[z]
        mask_slice = self.masks[z]

        # Normalize volume slice to 0-255
        vol_norm = vol_slice.astype(np.float32)
        vmin, vmax = vol_norm.min(), vol_norm.max()
        if vmax > vmin:
            vol_norm = (vol_norm - vmin) / (vmax - vmin) * 255
        vol_norm = vol_norm.astype(np.uint8)

        # Create RGB from grayscale
        rgb = np.stack([vol_norm, vol_norm, vol_norm], axis=-1)

        # Blend colored masks
        if mask_slice.max() > 0:
            mask_colored = self.colors[mask_slice.astype(int)]
            mask_region = mask_slice > 0
            rgb[mask_region] = (
                (1 - alpha) * rgb[mask_region] + alpha * mask_colored[mask_region]
            ).astype(np.uint8)

        return rgb

    def to_info_dict(self) -> dict:
        """Return metadata without the heavy arrays"""
        return {
            "uid": self.uid,
            "data_type": self.data_type,
            "timestamp": self.timestamp,
            "shape": list(self.shape),
            "num_slices": self.num_slices,
            "num_cells": int(self.masks.max()),
            "metadata": self.metadata,
        }


@dataclass
class ImageData:
    """Container for image data sent to clients"""

    uid: str
    data_type: str  # 'volume', 'projection', 'snapshot', 'detection', 'focus_sweep', etc.
    timestamp: str
    metadata: dict[str, Any] = field(default_factory=dict)
    base64_png: str | None = None
    shape: tuple | None = None

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class EmbryoImageCache:
    """Per-embryo image organization"""

    embryo_id: str
    volumes: list[ImageData] = field(default_factory=list)
    calibration: list[ImageData] = field(default_factory=list)
    snapshots: list[ImageData] = field(default_factory=list)
