#!/usr/bin/env python3
"""
Web-Based Interactive Embryo Marking
====================================

Sends a bottom-camera image to the web map view, waits for the user to
mark / edit embryo positions (and assign roles), and returns the result.

Replaces the deprecated napari-based marker. The napari implementation
was retired in the Phase 1 consolidation — the web map view is now the
single spatial GUI for detection, marking, role assignment, and live
monitoring.
"""

import logging
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)


async def mark_embryos_web(
    viz_server,
    image: np.ndarray,
    initial_stage_position: tuple[float, float],
    pixel_size_um: float = 0.65,
    timeout: float | None = None,
    save_image_path: Path | None = None,
    initial_markers: list[dict] | None = None,
    default_role: str = "test",
) -> list[dict]:
    """
    Interactive embryo marking via the web map view.

    Sends the bottom camera image to all connected browser clients and waits
    for the user to mark/edit embryo positions and assign Test/Calibration
    roles per marker.

    Parameters
    ----------
    viz_server : VisualizationServer
        Running visualization server instance
    image : np.ndarray
        Bottom camera overview image (2D grayscale or RGB)
    initial_stage_position : tuple of float
        Initial XY stage position in micrometers (x, y)
    pixel_size_um : float, optional
        Pixel size in micrometers/pixel (default: 0.65)
    timeout : float, optional
        Timeout in seconds (None = wait indefinitely)
    save_image_path : Path, optional
        If provided, saves the marked image to disk
    initial_markers : list of dict, optional
        Pre-populate the map view with editable markers (e.g. from SAM
        auto-detection or existing embryos). Each dict should have
        ``pixel_x``/``pixel_y`` and optionally ``role``, ``source``,
        ``embryo_id``, ``confidence``.
    default_role : str
        Role assigned to markers without an explicit role.

    Returns
    -------
    list of dict
        Marked embryos with ``pixel_position``, ``role``, ``source``,
        ``initial_stage_position``, ``marking_timestamp``.
    """
    session_id = await viz_server.start_marking_session(
        image=image,
        initial_stage_position=initial_stage_position,
        pixel_size_um=pixel_size_um,
        initial_markers=initial_markers,
        default_role=default_role,
    )
    logger.info("Waiting for embryo marking via map view (session %s)...", session_id)

    embryos = await viz_server.wait_for_marking(session_id, timeout=timeout)

    logger.info("Map-view marking complete: %d embryo(s) marked", len(embryos))

    if save_image_path and embryos:
        _save_marked_image(image, embryos, save_image_path)

    return embryos


def _save_marked_image(image: np.ndarray, embryos: list[dict], output_path: Path):
    """Save image with embryo markers drawn on it."""
    from PIL import Image as PILImage
    from PIL import ImageDraw, ImageFont

    output_path = Path(output_path)

    if image.dtype != np.uint8:
        img_normalized = ((image - image.min()) / max(image.max() - image.min(), 1) * 255).astype(
            np.uint8
        )
    else:
        img_normalized = image

    pil_image = PILImage.fromarray(img_normalized)
    if pil_image.mode != "RGB":
        pil_image = pil_image.convert("RGB")

    draw = ImageDraw.Draw(pil_image)

    for embryo in embryos:
        pixel_x, pixel_y = embryo["pixel_position"]
        embryo_num = embryo.get("embryo_number") or embryo.get("embryo_id") or "?"

        marker_size = 20
        draw.line(
            [(pixel_x - marker_size, pixel_y), (pixel_x + marker_size, pixel_y)],
            fill=(0, 255, 255),
            width=3,
        )
        draw.line(
            [(pixel_x, pixel_y - marker_size), (pixel_x, pixel_y + marker_size)],
            fill=(0, 255, 255),
            width=3,
        )

        circle_radius = 40
        draw.ellipse(
            [
                pixel_x - circle_radius,
                pixel_y - circle_radius,
                pixel_x + circle_radius,
                pixel_y + circle_radius,
            ],
            outline=(0, 255, 255),
            width=2,
        )

        font: ImageFont.FreeTypeFont | ImageFont.ImageFont
        try:
            font = ImageFont.truetype("arial.ttf", 24)
        except Exception:
            font = ImageFont.load_default()

        draw.text(
            (pixel_x - 10, pixel_y + circle_radius + 5),
            str(embryo_num),
            fill=(0, 255, 255),
            font=font,
        )

    pil_image.save(output_path)
    logger.info("Saved marked image: %s", output_path)
