"""
Napari-based visualization for microscopy images and embryo positions.

Provides interactive viewing, marking, and editing of embryo positions
on camera images. All functions accept data directly — no dependency
on the device client or network.
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from gently.core.coordinates import (
    pixel_to_stage_position,
    stage_to_pixel_position,
    get_um_per_pixel,
    DEFAULT_PIXEL_SIZE_UM,
    DEFAULT_OBJECTIVE_MAG,
)

logger = logging.getLogger(__name__)


def _save_image(image: np.ndarray, save_path: str):
    """Save an image to disk, choosing format by extension."""
    from PIL import Image as PILImage

    if image.dtype != np.uint8:
        img_min, img_max = image.min(), image.max()
        if img_max > img_min:
            image = ((image - img_min) / (img_max - img_min) * 255).astype(np.uint8)
        else:
            image = np.zeros_like(image, dtype=np.uint8)

    ext = Path(save_path).suffix.lower()
    if ext in ('.jpg', '.jpeg'):
        PILImage.fromarray(image).save(save_path, 'JPEG', quality=70, optimize=True)
    elif ext == '.png':
        PILImage.fromarray(image).save(save_path, 'PNG', optimize=True)
    else:
        import tifffile
        tifffile.imwrite(save_path, image)

    logger.info("Saved image to: %s", save_path)


def view_image(
    image: np.ndarray,
    title: str = "Image View",
    save_path: Optional[str] = None,
    show: bool = True,
    embryo_annotations: Optional[List[Dict]] = None,
) -> Dict:
    """
    View an image using napari with optional embryo annotations.

    Parameters
    ----------
    image : np.ndarray
        Image to display.
    title : str
        Window title.
    save_path : str, optional
        Path to save the image.
    show : bool
        Whether to open napari window (blocking).
    embryo_annotations : list, optional
        Embryo dicts with 'pixel_x', 'pixel_y' for overlay markers.
    """
    result = {'success': True}

    if save_path:
        _save_image(image, save_path)
        result['saved_to'] = save_path

    if show:
        import napari

        viewer = napari.Viewer(title=title)
        viewer.add_image(image, name='Image', colormap='gray')

        if embryo_annotations:
            in_view, out_of_view = [], []
            for emb in embryo_annotations:
                px = emb.get('pixel_x')
                py = emb.get('pixel_y')
                if px is not None and py is not None:
                    if 0 <= px < image.shape[1] and 0 <= py < image.shape[0]:
                        in_view.append([py, px])
                    else:
                        out_of_view.append([py, px])

            if in_view:
                viewer.add_points(
                    in_view, name='Embryos (in view)',
                    face_color='lime', border_color='white', size=30, symbol='cross',
                )
            if out_of_view:
                viewer.add_points(
                    out_of_view, name='Embryos (out of view)',
                    face_color='orange', border_color='white', size=30, symbol='cross',
                )

        napari.run()

    return result


def view_embryos(
    image: np.ndarray,
    embryos: List[Dict],
    title: str = "Embryos",
    save_path: Optional[str] = None,
    show: bool = True,
) -> Dict:
    """
    View embryos with colored markers on an image using napari.

    Parameters
    ----------
    image : np.ndarray
        Image to display.
    embryos : list of dict
        Embryo dicts with 'pixel_x'/'center_x' and 'pixel_y'/'center_y'.
    title : str
        Window title.
    save_path : str, optional
        Path to save the image.
    show : bool
        Whether to open napari window (blocking).
    """
    if image is None:
        return {'error': 'No image provided'}
    if not embryos:
        return {'error': 'No embryos to display'}

    result = {'success': True, 'num_embryos': len(embryos)}

    if save_path:
        _save_image(image, save_path)
        result['saved_to'] = save_path

    if show:
        import napari

        viewer = napari.Viewer(title=title)
        viewer.add_image(image, name='Image', colormap='gray')

        color_palette = [
            'red', 'blue', 'green', 'yellow', 'magenta',
            'cyan', 'orange', 'purple', 'pink', 'lime',
        ]
        points, colors = [], []
        for i, embryo in enumerate(embryos):
            px = embryo.get('center_x', embryo.get('pixel_x', 0))
            py = embryo.get('center_y', embryo.get('pixel_y', 0))
            points.append([py, px])
            colors.append(color_palette[i % len(color_palette)])

        if points:
            viewer.add_points(
                points, name=f'Embryos ({len(embryos)})',
                face_color=colors, border_color='white', size=40, symbol='disc',
            )

        napari.run()

    return result


def _points_to_embryos(
    points: np.ndarray,
    image_center: Tuple[float, float],
    stage_position: Tuple[float, float],
    pixel_size_um: float,
    objective_mag: float,
    start_index: int = 0,
) -> List[Dict]:
    """Convert napari point coordinates to embryo dicts with stage positions."""
    um_per_px = get_um_per_pixel(pixel_size_um, objective_mag)
    embryos = []
    for i, point in enumerate(points):
        py, px = float(point[0]), float(point[1])
        stage_x, stage_y = pixel_to_stage_position(
            px, py, image_center[0], image_center[1],
            stage_position[0], stage_position[1], um_per_px,
        )
        embryos.append({
            'embryo_id': f'embryo_{start_index + i + 1}',
            'center_x': px,
            'center_y': py,
            'stage_x_um': stage_x,
            'stage_y_um': stage_y,
            'confidence': 1.0,
            'source': 'manual',
        })
    return embryos


def edit_embryos_in_napari(
    image: np.ndarray,
    embryos: List[Dict],
    stage_position: Tuple[float, float],
    pixel_size_um: float = DEFAULT_PIXEL_SIZE_UM,
    objective_mag: float = DEFAULT_OBJECTIVE_MAG,
) -> List[Dict]:
    """
    Open napari for interactive embryo position editing.

    User can add, delete, or move embryo markers. Closing the window
    applies changes.

    Parameters
    ----------
    image : np.ndarray
        Camera image to display.
    embryos : list of dict
        Existing embryo dicts with 'center_x'/'pixel_x' and 'center_y'/'pixel_y'.
    stage_position : tuple
        Current (x, y) stage position in micrometers.
    pixel_size_um : float
        Camera pixel size in micrometers.
    objective_mag : float
        Objective magnification.

    Returns
    -------
    list of dict
        Updated embryo list with stage coordinates.
    """
    import napari

    um_per_px = get_um_per_pixel(pixel_size_um, objective_mag)
    image_center = (image.shape[1] / 2, image.shape[0] / 2)

    # Build initial points from existing embryos
    initial_points = []
    for emb in embryos:
        px = emb.get('center_x', emb.get('pixel_x', 0))
        py = emb.get('center_y', emb.get('pixel_y', 0))
        initial_points.append([py, px])

    viewer = napari.Viewer(title="Edit Embryo Positions")
    viewer.add_image(image, name='Camera Image', colormap='gray')

    points_layer = viewer.add_points(
        initial_points if initial_points else np.empty((0, 2)),
        name='Embryos',
        face_color='lime',
        border_color='white',
        size=30,
        symbol='disc',
    )

    napari.run()

    # Convert final points back to embryo dicts with stage coordinates
    final_points = points_layer.data
    return _points_to_embryos(
        final_points, image_center, stage_position,
        pixel_size_um, objective_mag,
    )


def mark_embryos_in_napari(
    image: np.ndarray,
    stage_position: Tuple[float, float],
    pixel_size_um: float = DEFAULT_PIXEL_SIZE_UM,
    objective_mag: float = DEFAULT_OBJECTIVE_MAG,
    existing_embryos: Optional[List[Dict]] = None,
) -> List[Dict]:
    """
    Open napari for manual embryo marking on a camera image.

    Existing embryos are shown in green (non-editable reference layer).
    New marks go on a separate red layer.

    Parameters
    ----------
    image : np.ndarray
        Camera image to display.
    stage_position : tuple
        Current (x, y) stage position in micrometers.
    pixel_size_um : float
        Camera pixel size in micrometers.
    objective_mag : float
        Objective magnification.
    existing_embryos : list, optional
        Previously detected embryos to show as reference.

    Returns
    -------
    list of dict
        All embryos (existing + newly marked) with stage coordinates.
    """
    import napari

    um_per_px = get_um_per_pixel(pixel_size_um, objective_mag)
    image_center = (image.shape[1] / 2, image.shape[0] / 2)

    viewer = napari.Viewer(title="Mark Embryo Positions")
    viewer.add_image(image, name='Camera Image', colormap='gray')

    # Show existing embryos as reference (green, non-editable)
    if existing_embryos:
        ref_points = []
        for emb in existing_embryos:
            px = emb.get('center_x', emb.get('pixel_x', 0))
            py = emb.get('center_y', emb.get('pixel_y', 0))
            ref_points.append([py, px])
        if ref_points:
            viewer.add_points(
                ref_points, name='Existing Embryos',
                face_color='lime', border_color='white', size=30, symbol='disc',
            )

    # Editable layer for new marks
    new_points_layer = viewer.add_points(
        np.empty((0, 2)),
        name='New Marks (click to add)',
        face_color='red',
        border_color='white',
        size=30,
        symbol='cross',
    )

    napari.run()

    # Combine existing + new embryos
    all_embryos = []

    # Keep existing embryos with their original data
    if existing_embryos:
        all_embryos.extend(existing_embryos)

    # Convert new marks to embryo dicts
    new_marks = new_points_layer.data
    if len(new_marks) > 0:
        start_idx = len(all_embryos)
        new_embryos = _points_to_embryos(
            new_marks, image_center, stage_position,
            pixel_size_um, objective_mag, start_index=start_idx,
        )
        all_embryos.extend(new_embryos)

    return all_embryos
