"""
Gently DiSPIM Detection
=======================

Embryo detection and object identification utilities for DiSPIM microscopy.
Provides ROI detection and object finding for sparse bottom camera images.

Separated from plans to maintain clean architecture - plans orchestrate devices,
detection modules handle image analysis.
"""

import numpy as np
from typing import Optional, Tuple, List
from scipy.ndimage import uniform_filter, gaussian_filter, sobel
from scipy.ndimage.measurements import center_of_mass, label
from scipy.ndimage import binary_opening, binary_closing


def detect_embryo_roi(image: np.ndarray,
                     kernel_size_fraction: float = 0.1,
                     min_kernel_size: int = 24,
                     variance_weight: float = 0.4,
                     gradient_weight: float = 0.4,
                     intensity_weight: float = 0.2,
                     threshold_std: float = 1.5,
                     min_area_fraction: float = 0.05,
                     max_area_fraction: float = 0.5) -> Optional[Tuple[int, int, int, int]]:
    """
    Detect embryo region of interest for sparse bottom camera images

    Uses multiple detection methods to find the most likely embryo location:
    1. Local variance for texture detection
    2. Gradient magnitude for edge detection
    3. Intensity thresholding for object detection

    Parameters
    ----------
    image : np.ndarray
        Input image array
    kernel_size_fraction : float
        Kernel size as fraction of min(image dimensions)
    min_kernel_size : int
        Minimum kernel size in pixels
    variance_weight : float
        Weight for texture/variance detection
    gradient_weight : float
        Weight for edge/gradient detection
    intensity_weight : float
        Weight for intensity contrast detection
    threshold_std : float
        Threshold in standard deviations above mean
    min_area_fraction : float
        Minimum ROI area as fraction of total image
    max_area_fraction : float
        Maximum ROI area as fraction of total image

    Returns
    -------
    Optional[Tuple[int, int, int, int]]
        ROI as (x, y, width, height) or None if no good ROI found
    """
    try:
        # Normalize image to 0-1 range
        img_norm = (image - np.min(image)) / (np.max(image) - np.min(image) + 1e-8)

        # Adaptive kernel size
        kernel_size = int(min(image.shape) * kernel_size_fraction)
        kernel_size = max(kernel_size, min_kernel_size)

        # Method 1: Local variance (texture detection)
        local_mean = uniform_filter(img_norm, size=kernel_size)
        local_var = uniform_filter(img_norm**2, size=kernel_size) - local_mean**2

        # Method 2: Gradient magnitude (edge detection)
        img_smooth = gaussian_filter(img_norm, sigma=1.0)
        grad_x = sobel(img_smooth, axis=1)
        grad_y = sobel(img_smooth, axis=0)
        gradient_mag = np.sqrt(grad_x**2 + grad_y**2)

        # Method 3: Intensity contrast detection
        background_level = np.median(img_norm)
        intensity_contrast = np.abs(img_norm - background_level)

        # Combine detection methods
        combined_score = (variance_weight * local_var +
                         gradient_weight * gradient_mag +
                         intensity_weight * intensity_contrast)

        # Find regions above threshold
        threshold = np.mean(combined_score) + threshold_std * np.std(combined_score)
        embryo_mask = combined_score > threshold

        # Clean up mask with morphological operations
        embryo_mask = binary_opening(embryo_mask, iterations=1)
        embryo_mask = binary_closing(embryo_mask, iterations=2)

        # Find connected components
        labeled_regions, num_regions = label(embryo_mask)

        if num_regions == 0:
            return None

        # Find largest connected component (most likely embryo)
        region_sizes = [(labeled_regions == i).sum() for i in range(1, num_regions + 1)]
        largest_region_idx = np.argmax(region_sizes) + 1
        largest_region_mask = (labeled_regions == largest_region_idx)

        # Get bounding box with padding
        rows, cols = np.where(largest_region_mask)
        if len(rows) == 0:
            return None

        padding = kernel_size // 2
        y_min = max(0, np.min(rows) - padding)
        y_max = min(image.shape[0], np.max(rows) + padding)
        x_min = max(0, np.min(cols) - padding)
        x_max = min(image.shape[1], np.max(cols) + padding)

        w = x_max - x_min
        h = y_max - y_min
        roi_area = w * h
        total_area = image.shape[0] * image.shape[1]

        # Check if ROI is reasonable size
        if min_area_fraction <= roi_area/total_area <= max_area_fraction:
            return (x_min, y_min, w, h)
        else:
            return None

    except Exception as e:
        print(f"Embryo ROI detection failed: {e}")
        return None


def detect_multiple_embryos(image: np.ndarray,
                           max_embryos: int = 5,
                           min_separation: int = 50) -> List[Tuple[int, int, int, int]]:
    """
    Detect multiple embryo regions in bottom camera images

    Parameters
    ----------
    image : np.ndarray
        Input image
    max_embryos : int
        Maximum number of embryos to detect
    min_separation : int
        Minimum separation between embryo centers in pixels

    Returns
    -------
    List[Tuple[int, int, int, int]]
        List of ROIs as (x, y, width, height) tuples
    """
    try:
        # Use same detection method as single embryo
        img_norm = (image - np.min(image)) / (np.max(image) - np.min(image) + 1e-8)

        kernel_size = max(min(image.shape) // 10, 24)

        # Combined detection score
        local_mean = uniform_filter(img_norm, size=kernel_size)
        local_var = uniform_filter(img_norm**2, size=kernel_size) - local_mean**2

        img_smooth = gaussian_filter(img_norm, sigma=1.0)
        grad_x = sobel(img_smooth, axis=1)
        grad_y = sobel(img_smooth, axis=0)
        gradient_mag = np.sqrt(grad_x**2 + grad_y**2)

        combined_score = 0.6 * local_var + 0.4 * gradient_mag

        # Find multiple regions above threshold
        threshold = np.mean(combined_score) + 1.2 * np.std(combined_score)
        embryo_mask = combined_score > threshold

        # Clean up
        embryo_mask = binary_opening(embryo_mask, iterations=1)
        embryo_mask = binary_closing(embryo_mask, iterations=2)

        # Find all connected components
        labeled_regions, num_regions = label(embryo_mask)

        if num_regions == 0:
            return []

        # Get all regions with their sizes and centers
        regions = []
        for i in range(1, num_regions + 1):
            region_mask = (labeled_regions == i)
            region_size = region_mask.sum()

            # Skip tiny regions
            if region_size < 100:  # Minimum embryo size
                continue

            # Get bounding box
            rows, cols = np.where(region_mask)
            if len(rows) == 0:
                continue

            y_center = (np.min(rows) + np.max(rows)) // 2
            x_center = (np.min(cols) + np.max(cols)) // 2

            padding = kernel_size // 2
            y_min = max(0, np.min(rows) - padding)
            y_max = min(image.shape[0], np.max(rows) + padding)
            x_min = max(0, np.min(cols) - padding)
            x_max = min(image.shape[1], np.max(cols) + padding)

            w = x_max - x_min
            h = y_max - y_min

            regions.append({
                'roi': (x_min, y_min, w, h),
                'center': (x_center, y_center),
                'size': region_size
            })

        # Sort by size (largest first)
        regions.sort(key=lambda x: x['size'], reverse=True)

        # Filter out overlapping regions
        selected_regions = []
        for region in regions:
            if len(selected_regions) >= max_embryos:
                break

            center = region['center']

            # Check separation from already selected regions
            too_close = False
            for selected in selected_regions:
                selected_center = selected['center']
                distance = np.sqrt((center[0] - selected_center[0])**2 +
                                 (center[1] - selected_center[1])**2)
                if distance < min_separation:
                    too_close = True
                    break

            if not too_close:
                selected_regions.append(region)

        return [r['roi'] for r in selected_regions]

    except Exception as e:
        print(f"Multiple embryo detection failed: {e}")
        return []


def get_embryo_focus_roi(image: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
    """
    Simple interface for getting embryo ROI for focus analysis

    This is the main function that should be called from focus analysis code.

    Parameters
    ----------
    image : np.ndarray
        Input image from bottom camera

    Returns
    -------
    Optional[Tuple[int, int, int, int]]
        ROI as (x, y, width, height) or None to use full image
    """
    return detect_embryo_roi(image)