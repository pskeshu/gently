"""Shared helpers for hardware tools."""

import numpy as np


def select_best_view(image: np.ndarray) -> np.ndarray:
    """Select the brighter half of a dual-view lightsheet image."""
    if image.ndim != 2:
        return image
    h, w = image.shape
    if w < 100:
        return image
    mid = w // 2
    left = image[:, :mid]
    right = image[:, mid:]
    if np.mean(left) >= np.mean(right):
        return left
    return right


def crop_to_embryo_roi(image: np.ndarray, padding_percent: float = 20.0) -> np.ndarray:
    """
    Detect embryo in image and crop to ROI.
    Returns cropped image for more accurate focus scoring.
    """
    try:
        from scipy import ndimage

        # Threshold to find embryo
        threshold = np.percentile(image, 75)
        mask = image > threshold

        # Label connected components
        labeled, num_features = ndimage.label(mask)
        if num_features == 0:
            return image  # No embryo found, return full image

        # Find largest component
        sizes = ndimage.sum(mask, labeled, range(1, num_features + 1))
        largest_label = np.argmax(sizes) + 1
        embryo_mask = labeled == largest_label

        # Get bounding box
        coords = np.argwhere(embryo_mask)
        if len(coords) < 100:  # Too small, probably noise
            return image

        y_min, x_min = coords.min(axis=0)
        y_max, x_max = coords.max(axis=0)

        # Add padding
        h, w = image.shape
        y_pad = int((y_max - y_min) * padding_percent / 100)
        x_pad = int((x_max - x_min) * padding_percent / 100)

        y1 = max(0, y_min - y_pad)
        x1 = max(0, x_min - x_pad)
        y2 = min(h, y_max + y_pad + 1)
        x2 = min(w, x_max + x_pad + 1)

        cropped = image[y1:y2, x1:x2]
        return cropped

    except Exception:
        return image  # Fallback to full image


def select_view_and_crop_roi(image: np.ndarray) -> np.ndarray:
    """Select best view and crop to embryo ROI for focus scoring."""
    view = select_best_view(image)
    return crop_to_embryo_roi(view)
