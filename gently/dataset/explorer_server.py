"""
Web Explorer Server with Annotation Support.

FastAPI server providing:
- Browse sessions, embryos, images
- View images with stage annotations
- Create/edit ground truth labels
- View perception run results

Usage:
    python -m gently.dataset serve --port 8765
"""

import base64
import io
import json
import logging
from pathlib import Path
from typing import Optional, List, Tuple

import numpy as np
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import HTMLResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from .schema import get_connection, get_database_stats, DEFAULT_DB_PATH
from .embryo_dataset import EmbryoDataset

# Lazy imports for projection functions
tifffile = None
PIL_Image = None


def ensure_projection_deps():
    """Ensure projection dependencies are available."""
    global tifffile, PIL_Image
    if tifffile is None:
        import tifffile as _tifffile
        tifffile = _tifffile
    if PIL_Image is None:
        from PIL import Image as _Image
        PIL_Image = _Image


# =============================================================================
# Projection Functions (from projection_explorer.py)
# =============================================================================

def normalize_image(img: np.ndarray, p_low: float = 1, p_high: float = 99) -> np.ndarray:
    """Normalize image to 0-255 uint8 using percentile scaling."""
    img = img.astype(np.float32)
    vmin = np.percentile(img, p_low)
    vmax = np.percentile(img, p_high)
    if vmax > vmin:
        img = np.clip((img - vmin) / (vmax - vmin), 0, 1)
    else:
        img = np.zeros_like(img)
    return (img * 255).astype(np.uint8)


def image_to_base64(img: np.ndarray, format: str = "JPEG", quality: int = 90) -> str:
    """Convert numpy array to base64-encoded image."""
    ensure_projection_deps()
    pil_img = PIL_Image.fromarray(img)
    if pil_img.mode not in ('RGB', 'RGBA'):
        pil_img = pil_img.convert('RGB')
    buffer = io.BytesIO()
    pil_img.save(buffer, format=format, quality=quality)
    return base64.b64encode(buffer.getvalue()).decode()


def load_volume(path: Path) -> np.ndarray:
    """Load a volume from TIFF file, extract View A."""
    ensure_projection_deps()
    vol = tifffile.imread(str(path))
    vol = np.squeeze(vol)
    if vol.ndim == 3:
        z_depth, height, width = vol.shape
        # Extract View A (left half) if dual-view format
        if width > height * 2:
            vol = vol[:, :, :width // 2]
    return vol


def compute_crop_bounds(volume: np.ndarray, padding: int = 20, sigma_mult: float = 3.5) -> Tuple[int, int, int, int]:
    """Compute crop bounds for 3D volume using center-of-mass of bright pixels."""
    if volume.ndim != 3:
        return (0, volume.shape[0], 0, volume.shape[1])
    max_proj = np.max(volume, axis=0).astype(np.float32)
    threshold = np.percentile(max_proj, 95)
    mask = max_proj > threshold
    y_coords, x_coords = np.where(mask)
    if len(y_coords) < 10:
        return (0, volume.shape[1], 0, volume.shape[2])
    cy, cx = np.mean(y_coords), np.mean(x_coords)
    y_std = max(np.std(y_coords), 20)
    x_std = max(np.std(x_coords), 20)
    y_min = int(max(0, cy - sigma_mult * y_std - padding))
    y_max = int(min(volume.shape[1], cy + sigma_mult * y_std + padding))
    x_min = int(max(0, cx - sigma_mult * x_std - padding))
    x_max = int(min(volume.shape[2], cx + sigma_mult * x_std + padding))
    return (y_min, y_max, x_min, x_max)


def apply_crop_bounds(volume: np.ndarray, bounds: Tuple[int, int, int, int]) -> np.ndarray:
    """Apply pre-computed crop bounds to a volume."""
    y_min, y_max, x_min, x_max = bounds
    return volume[:, max(0, y_min):min(volume.shape[1], y_max),
                     max(0, x_min):min(volume.shape[2], x_max)]


def find_outer_boundary(img: np.ndarray, percentile: float = 50) -> np.ndarray:
    """Find outer boundary of embryo by thresholding and extracting mask edge."""
    thresh = np.percentile(img, percentile)
    mask = img > thresh
    padded = np.pad(mask, 1, mode='constant', constant_values=True)
    eroded = (padded[:-2, :-2] & padded[:-2, 1:-1] & padded[:-2, 2:] &
              padded[1:-1, :-2] & padded[1:-1, 1:-1] & padded[1:-1, 2:] &
              padded[2:, :-2] & padded[2:, 1:-1] & padded[2:, 2:])
    boundary = (mask & ~eroded).astype(np.uint8) * 255
    return boundary


def overlay_edges(img: np.ndarray, edges: np.ndarray, color: Tuple[int, int, int] = (255, 200, 0)) -> np.ndarray:
    """Overlay edge contours on image in specified color."""
    if img.ndim == 2:
        rgb = np.stack([img, img, img], axis=-1)
    else:
        rgb = img.copy()
    edge_mask = edges > 0
    rgb[edge_mask] = color
    return rgb


def projection_dual_view(volume: np.ndarray) -> Tuple[np.ndarray, str]:
    """Dual-view projection: TOP above, SIDE below with boundary overlay."""
    ensure_projection_deps()
    if volume.ndim != 3:
        return normalize_image(volume), "2D input"
    z_depth, height, width = volume.shape
    top_proj = np.max(volume, axis=0)
    side_proj = np.max(volume, axis=1)
    top_norm = normalize_image(top_proj)
    side_norm = normalize_image(side_proj)
    top_edges = find_outer_boundary(top_norm, percentile=50)
    side_edges = find_outer_boundary(side_norm, percentile=50)
    top_rgb = overlay_edges(top_norm, top_edges)
    side_rgb = overlay_edges(side_norm, side_edges)
    target_width = top_rgb.shape[1]
    side_new_h = max(height // 3, int(z_depth * 3))
    pil_side = PIL_Image.fromarray(side_rgb)
    pil_side = pil_side.resize((target_width, side_new_h), PIL_Image.Resampling.LANCZOS)
    side_scaled = np.array(pil_side)
    sep = np.ones((3, target_width, 3), dtype=np.uint8) * 128
    combined = np.concatenate([top_rgb, sep, side_scaled], axis=0)
    return combined, "Dual-view MIP with boundary: TOP (XY) + SIDE (XZ)"


def projection_depth_colored(volume: np.ndarray, colormap: str = 'turbo') -> Tuple[np.ndarray, str]:
    """Depth-colored max intensity projection."""
    ensure_projection_deps()
    if volume.ndim != 3:
        gray = normalize_image(volume)
        return np.stack([gray, gray, gray], axis=-1), "2D input"
    z_depth, height, width = volume.shape
    try:
        import matplotlib.pyplot as plt
        cmap = plt.get_cmap(colormap)
    except:
        cmap = None
    colored_volume = np.zeros((z_depth, height, width, 3), dtype=np.float32)
    for z in range(z_depth):
        z_norm = z / max(1, z_depth - 1)
        if cmap is not None:
            color = np.array(cmap(z_norm)[:3])
        else:
            color = np.array([z_norm, 0.5, 1 - z_norm])
        slice_data = volume[z].astype(np.float32)
        slice_norm = (slice_data - slice_data.min()) / max(1, slice_data.max() - slice_data.min())
        colored_volume[z] = slice_norm[:, :, np.newaxis] * color
    top_rgb = (np.max(colored_volume, axis=0) * 255).astype(np.uint8)
    side_rgb = (np.max(colored_volume, axis=1) * 255).astype(np.uint8)
    pil_side = PIL_Image.fromarray(side_rgb)
    side_new_h = max(height // 3, int(z_depth * 3))
    pil_side = pil_side.resize((width, side_new_h), PIL_Image.Resampling.LANCZOS)
    side_scaled = np.array(pil_side)
    sep = np.ones((3, width, 3), dtype=np.uint8) * 128
    combined = np.concatenate([top_rgb, sep, side_scaled], axis=0)
    return combined, f"Z-depth colored MIP ({colormap}): TOP + SIDE"


def projection_multi_slice(volume: np.ndarray, n_slices: int = 6) -> Tuple[np.ndarray, str]:
    """Montage of N representative z-slices."""
    if volume.ndim != 3:
        return normalize_image(volume), "2D input"
    z_depth, height, width = volume.shape
    if z_depth <= n_slices:
        indices = list(range(z_depth))
    else:
        indices = np.linspace(0, z_depth - 1, n_slices, dtype=int).tolist()
    slices = [normalize_image(volume[i]) for i in indices]
    n_cols = (len(slices) + 1) // 2
    n_rows = 2
    while len(slices) < n_rows * n_cols:
        slices.append(np.zeros_like(slices[0]))
    rows = []
    for r in range(n_rows):
        row_slices = slices[r * n_cols:(r + 1) * n_cols]
        sep = np.ones((height, 2), dtype=np.uint8) * 64
        row_with_sep = []
        for i, s in enumerate(row_slices):
            if i > 0:
                row_with_sep.append(sep)
            row_with_sep.append(s)
        rows.append(np.concatenate(row_with_sep, axis=1))
    row_width = rows[0].shape[1]
    h_sep = np.ones((2, row_width), dtype=np.uint8) * 64
    montage = np.concatenate([rows[0], h_sep, rows[1]], axis=0)
    return montage, f"Multi-slice montage ({n_slices} slices)"


def projection_three_view(volume: np.ndarray) -> Tuple[np.ndarray, str]:
    """Three orthogonal views with axis alignment."""
    ensure_projection_deps()
    if volume.ndim != 3:
        return normalize_image(volume), "2D input"
    z_depth, height, width = volume.shape
    xy_proj = normalize_image(np.max(volume, axis=0))
    xz_proj = normalize_image(np.max(volume, axis=1))
    yz_proj = normalize_image(np.max(volume, axis=2))
    xy_h, xy_w = xy_proj.shape
    z_display_h = max(xy_h // 3, int(z_depth * 3))
    pil_xz = PIL_Image.fromarray(xz_proj)
    pil_xz = pil_xz.resize((xy_w, z_display_h), PIL_Image.Resampling.LANCZOS)
    xz_scaled = np.array(pil_xz)
    yz_rotated = yz_proj.T
    z_display_w = z_display_h
    pil_yz = PIL_Image.fromarray(yz_rotated)
    pil_yz = pil_yz.resize((z_display_w, xy_h), PIL_Image.Resampling.LANCZOS)
    yz_scaled = np.array(pil_yz)
    sep = 3
    v_sep = np.ones((xy_h, sep), dtype=np.uint8) * 128
    top_row = np.concatenate([xy_proj, v_sep, yz_scaled], axis=1)
    total_width = top_row.shape[1]
    if xz_scaled.shape[1] < total_width:
        pad = np.zeros((xz_scaled.shape[0], total_width - xz_scaled.shape[1]), dtype=np.uint8)
        bottom_row = np.concatenate([xz_scaled, pad], axis=1)
    else:
        bottom_row = xz_scaled[:, :total_width]
    h_sep = np.ones((sep, total_width), dtype=np.uint8) * 128
    combined = np.concatenate([top_row, h_sep, bottom_row], axis=0)
    return combined, "Three-view: [XY|YZ] top, [XZ] bottom"


def render_volume_rotated(volume: np.ndarray, angle_y: float, angle_x: float = -0.5,
                          threshold: float = 0.12, num_slices: int = 48,
                          perspective: float = 0.4) -> np.ndarray:
    """
    Render volume from a rotated viewpoint with parallax and perspective.

    Simulates the 3D viewer by shifting and scaling slices based on rotation,
    creating depth through parallax and perspective foreshortening.

    Parameters
    ----------
    volume : np.ndarray
        3D volume (Z, Y, X)
    angle_y : float
        Rotation around Y axis in radians (horizontal spin)
    angle_x : float
        Rotation around X axis in radians (vertical tilt)
    threshold : float
        Intensity threshold for transparency (0-1)
    num_slices : int
        Number of slices to composite
    perspective : float
        Perspective strength (0 = orthographic, higher = more perspective)

    Returns
    -------
    np.ndarray
        RGB image (H, W, 3) uint8
    """
    from scipy import ndimage

    # Normalize volume to 0-1
    vol = volume.astype(np.float32)
    p1, p99 = np.percentile(vol, [1, 99])
    vol = np.clip((vol - p1) / (p99 - p1 + 1e-8), 0, 1)

    # Apply Gaussian blur along Z axis to reduce banding at side views (Z interpolation)
    vol = ndimage.gaussian_filter1d(vol, sigma=1.0, axis=0)

    z_depth, height, width = vol.shape

    # Calculate the z-scale factor (exaggerate depth like in 3D viewer)
    z_scale = (z_depth / width) * 3.0

    # Output canvas size (larger to accommodate shifts and scaling)
    margin = int(max(width, height) * 0.5)
    out_h = height + 2 * margin
    out_w = width + 2 * margin
    center_y, center_x = out_h // 2, out_w // 2

    # Initialize output
    result = np.zeros((out_h, out_w, 3), dtype=np.float32)

    # Compute rotation components
    sin_y, cos_y = np.sin(angle_y), np.cos(angle_y)
    sin_x, cos_x = np.sin(angle_x), np.cos(angle_x)

    # Sample slices - sort by depth after rotation for proper compositing
    slice_data_list = []
    for i in range(num_slices):
        z_idx = int(i * z_depth / num_slices)
        z_pos = (i / num_slices - 0.5) * z_scale  # Normalized z position

        # Apply rotation to get the projected depth and shifts
        rotated_z = z_pos * cos_y
        shift_x = z_pos * sin_y * width * 1.2  # Horizontal parallax (increased for more depth)

        # Apply X rotation (tilt)
        shift_y = -rotated_z * sin_x * height * 1.0  # Vertical parallax (increased for more depth)
        depth_after_rotation = rotated_z * cos_x

        # Perspective scale: closer slices are larger, further are smaller
        # Increase perspective effect for more 3D depth
        perspective_scale = 1.0 + (depth_after_rotation * perspective * 1.5 / z_scale)
        perspective_scale = np.clip(perspective_scale, 0.5, 1.5)

        slice_data_list.append({
            'z_idx': z_idx,
            'shift_x': shift_x,
            'shift_y': shift_y,
            'depth': depth_after_rotation,
            'scale': perspective_scale,
        })

    # Sort by depth (back to front for proper alpha compositing)
    slice_data_list.sort(key=lambda s: s['depth'])

    # Composite slices
    for slice_info in slice_data_list:
        z_idx = slice_info['z_idx']
        shift_x = slice_info['shift_x']
        shift_y = slice_info['shift_y']
        scale = slice_info['scale']

        # Get slice
        slice_img = vol[z_idx, :, :]

        # Apply slight blur for smoothness
        slice_img = ndimage.gaussian_filter(slice_img, sigma=0.5)

        # Compute alpha - match Three.js formula: (val - threshold) * 2, capped at 1
        alpha = np.clip((slice_img - threshold) * 2, 0, 1)

        curr_h, curr_w = height, width

        # Scale the slice for perspective effect
        if abs(scale - 1.0) > 0.01:
            new_h = int(height * scale)
            new_w = int(width * scale)
            pil_slice = PIL_Image.fromarray((slice_img * 255).astype(np.uint8))
            pil_alpha = PIL_Image.fromarray((alpha * 255).astype(np.uint8))
            pil_slice = pil_slice.resize((new_w, new_h), PIL_Image.Resampling.BILINEAR)
            pil_alpha = pil_alpha.resize((new_w, new_h), PIL_Image.Resampling.BILINEAR)
            slice_img = np.array(pil_slice).astype(np.float32) / 255
            alpha = np.array(pil_alpha).astype(np.float32) / 255
            curr_h, curr_w = new_h, new_w

        # Calculate paste position (centered, then shifted)
        y_start = center_y - curr_h // 2 + int(shift_y)
        x_start = center_x - curr_w // 2 + int(shift_x)

        # Bounds checking
        src_y_start = max(0, -y_start)
        src_x_start = max(0, -x_start)
        dst_y_start = max(0, y_start)
        dst_x_start = max(0, x_start)

        src_y_end = min(curr_h, out_h - y_start)
        src_x_end = min(curr_w, out_w - x_start)
        dst_y_end = dst_y_start + (src_y_end - src_y_start)
        dst_x_end = dst_x_start + (src_x_end - src_x_start)

        if dst_y_end <= dst_y_start or dst_x_end <= dst_x_start:
            continue

        # Get the slice region
        src_slice = slice_img[src_y_start:src_y_end, src_x_start:src_x_end]
        src_alpha = alpha[src_y_start:src_y_end, src_x_start:src_x_end]

        # Alpha composite
        for c in range(3):
            result[dst_y_start:dst_y_end, dst_x_start:dst_x_end, c] = (
                src_slice * src_alpha +
                result[dst_y_start:dst_y_end, dst_x_start:dst_x_end, c] * (1 - src_alpha)
            )

    # Crop to content (remove empty margins)
    gray = np.mean(result, axis=2)
    rows = np.any(gray > 0.01, axis=1)
    cols = np.any(gray > 0.01, axis=0)
    if np.any(rows) and np.any(cols):
        y_min, y_max = np.where(rows)[0][[0, -1]]
        x_min, x_max = np.where(cols)[0][[0, -1]]
        # Add small padding
        pad = 15
        y_min = max(0, y_min - pad)
        y_max = min(out_h, y_max + pad)
        x_min = max(0, x_min - pad)
        x_max = min(out_w, x_max + pad)
        result = result[y_min:y_max, x_min:x_max]

    # Convert to uint8
    result = (np.clip(result, 0, 1) * 255).astype(np.uint8)

    return result


def projection_spin_3d(volume: np.ndarray) -> Tuple[np.ndarray, str]:
    """Multiple 3D perspective views from different angles (2x3 grid)."""
    ensure_projection_deps()
    if volume.ndim != 3:
        return normalize_image(volume), "2D input"

    py_threshold = 0.12
    base_tilt = 0.21
    angles_y = [-0.05, 0.5, 1.0, 1.57, 2.1, 2.6]

    views = []
    for angle_y in angles_y:
        view = render_volume_rotated(
            volume, angle_y=angle_y, angle_x=base_tilt,
            threshold=py_threshold, perspective=0.5
        )
        views.append(view)

    # Arrange in 2x3 grid
    target_h = max(v.shape[0] for v in views)
    target_w = max(v.shape[1] for v in views)

    padded = []
    for v in views:
        h, w = v.shape[:2]
        pad_h = (target_h - h) // 2
        pad_w = (target_w - w) // 2
        if v.ndim == 3:
            p = np.zeros((target_h, target_w, 3), dtype=v.dtype)
            p[pad_h:pad_h+h, pad_w:pad_w+w] = v
        else:
            p = np.zeros((target_h, target_w), dtype=v.dtype)
            p[pad_h:pad_h+h, pad_w:pad_w+w] = v
        padded.append(p)

    row1 = np.hstack(padded[0:3])
    row2 = np.hstack(padded[3:6])
    grid = np.vstack([row1, row2])

    return grid, "6 perspective views rotating around volume"


PROJECTION_METHODS = {
    'dual_view': projection_dual_view,
    'depth_colored': projection_depth_colored,
    'multi_slice': projection_multi_slice,
    'three_view': projection_three_view,
    'spin_3d': projection_spin_3d,
}


logger = logging.getLogger(__name__)

# Pydantic models for API
class GroundTruthCreate(BaseModel):
    session_id: str
    embryo_id: str
    stage: str
    start_timepoint: int
    annotator: Optional[str] = None
    notes: Optional[str] = None


class GroundTruthDelete(BaseModel):
    session_id: str
    embryo_id: str
    stage: Optional[str] = None


class DatasetExplorer:
    """
    FastAPI-based dataset explorer with annotation support.

    Parameters
    ----------
    db_path : Path
        Path to SQLite database
    port : int
        Port to serve on
    host : str
        Host to bind to
    """

    def __init__(
        self,
        db_path: Path = DEFAULT_DB_PATH,
        port: int = 8765,
        host: str = "0.0.0.0",
    ):
        self.db_path = db_path
        self.port = port
        self.host = host
        self.dataset = EmbryoDataset(db_path=db_path)

        self.app = FastAPI(
            title="Embryo Dataset Explorer",
            description="Browse and annotate embryo imaging data",
            version="1.0.0",
        )
        self._setup_routes()

    def _setup_routes(self):
        """Configure all API routes."""
        app = self.app

        # =====================================================================
        # Static Files & Main Page
        # =====================================================================

        @app.get("/", response_class=HTMLResponse)
        async def index():
            """Serve the main explorer page."""
            return self._get_explorer_html()

        # =====================================================================
        # Stats & Overview
        # =====================================================================

        @app.get("/api/stats")
        async def get_stats():
            """Get database statistics."""
            conn = get_connection(self.db_path)
            stats = get_database_stats(conn)
            conn.close()
            return stats

        # =====================================================================
        # Sessions
        # =====================================================================

        @app.get("/api/sessions")
        async def list_sessions():
            """List all sessions with summary info."""
            return self.dataset.get_sessions()

        @app.get("/api/sessions/{session_id}")
        async def get_session(session_id: str):
            """Get session details."""
            conn = get_connection(self.db_path)
            session = conn.execute(
                "SELECT * FROM sessions WHERE session_id = ?",
                (session_id,)
            ).fetchone()
            conn.close()

            if not session:
                raise HTTPException(status_code=404, detail="Session not found")

            return dict(session)

        # =====================================================================
        # Embryos
        # =====================================================================

        @app.get("/api/embryos")
        async def list_embryos(
            session_id: Optional[str] = None,
            has_ground_truth: Optional[bool] = None,
        ):
            """List embryos with optional filters."""
            embryos = []
            for embryo in self.dataset.iter_embryos(
                session_id=session_id,
                has_ground_truth=has_ground_truth,
            ):
                embryos.append({
                    "embryo_id": embryo.embryo_id,
                    "session_id": embryo.session_id,
                    "num_images": embryo.num_images,
                    "num_volumes": embryo.num_volumes,
                    "timepoint_range": embryo.timepoint_range,
                    "has_ground_truth": embryo.has_ground_truth,
                    "ground_truth_stages": embryo.ground_truth_stages,
                })
            return embryos

        @app.get("/api/embryos/{session_id}/{embryo_id}")
        async def get_embryo(session_id: str, embryo_id: str):
            """Get embryo details including ground truth."""
            for embryo in self.dataset.iter_embryos(session_id=session_id):
                if embryo.embryo_id == embryo_id:
                    ground_truth = self.dataset.get_ground_truth(session_id, embryo_id)
                    return {
                        "embryo_id": embryo.embryo_id,
                        "session_id": embryo.session_id,
                        "num_images": embryo.num_images,
                        "num_volumes": embryo.num_volumes,
                        "timepoint_range": embryo.timepoint_range,
                        "has_ground_truth": embryo.has_ground_truth,
                        "ground_truth": ground_truth,
                    }

            raise HTTPException(status_code=404, detail="Embryo not found")

        # =====================================================================
        # Images
        # =====================================================================

        @app.get("/api/images/{session_id}/{embryo_id}")
        async def list_images(
            session_id: str,
            embryo_id: str,
            start_tp: Optional[int] = None,
            end_tp: Optional[int] = None,
        ):
            """List images for an embryo (without image data)."""
            timepoint_range = None
            if start_tp is not None and end_tp is not None:
                timepoint_range = (start_tp, end_tp)

            images = []
            for img in self.dataset.iter_images(
                embryo_id=embryo_id,
                session_id=session_id,
                timepoint_range=timepoint_range,
                load_image_data=False,
            ):
                images.append(img.to_dict())

            return images

        @app.get("/api/image/{session_id}/{embryo_id}/{index}")
        async def get_image(
            session_id: str,
            embryo_id: str,
            index: int,
            include_data: bool = True,
        ):
            """Get a single image by index (sequential order)."""
            # Use index-based lookup since timepoints may be NULL
            img = self.dataset.get_image_by_index(
                embryo_id=embryo_id,
                index=index,
                session_id=session_id,
            )

            if not img:
                raise HTTPException(status_code=404, detail="Image not found")

            result = img.to_dict()
            result["index"] = index  # Add index to response
            if include_data:
                result["image_b64"] = img.image_b64

            return result

        @app.get("/api/image_by_uid/{uid}")
        async def get_image_by_uid(
            uid: str,
            include_data: bool = True,
        ):
            """Get a single image by its UID."""
            img = self.dataset.get_image_by_uid(uid)

            if not img:
                raise HTTPException(status_code=404, detail="Image not found")

            result = img.to_dict()
            if include_data:
                result["image_b64"] = img.image_b64

            return result

        # =====================================================================
        # Ground Truth / Annotations
        # =====================================================================

        @app.get("/api/ground_truth/{session_id}/{embryo_id}")
        async def get_ground_truth(session_id: str, embryo_id: str):
            """Get ground truth annotations for an embryo."""
            return self.dataset.get_ground_truth(session_id, embryo_id)

        @app.post("/api/ground_truth")
        async def create_ground_truth(data: GroundTruthCreate):
            """Create or update a ground truth annotation."""
            self.dataset.set_ground_truth(
                session_id=data.session_id,
                embryo_id=data.embryo_id,
                stage=data.stage,
                start_timepoint=data.start_timepoint,
                annotator=data.annotator,
                notes=data.notes,
            )
            return {"status": "ok", "message": f"Set {data.stage} @ t={data.start_timepoint}"}

        @app.delete("/api/ground_truth")
        async def delete_ground_truth(data: GroundTruthDelete):
            """Delete ground truth annotation(s)."""
            self.dataset.delete_ground_truth(
                session_id=data.session_id,
                embryo_id=data.embryo_id,
                stage=data.stage,
            )
            return {"status": "ok"}

        # =====================================================================
        # Perception Runs
        # =====================================================================

        @app.get("/api/runs")
        async def list_runs():
            """List perception runs."""
            return self.dataset.get_perception_runs()

        @app.get("/api/runs/{run_id}/metrics")
        async def get_run_metrics(run_id: int):
            """Get metrics for a perception run."""
            return self.dataset.compute_run_metrics(run_id)

        @app.get("/api/runs/{run_id}/predictions")
        async def get_run_predictions(
            run_id: int,
            embryo_id: Optional[str] = None,
            limit: int = Query(100, le=1000),
            offset: int = 0,
        ):
            """Get predictions for a perception run."""
            conn = get_connection(self.db_path)

            query = """
                SELECT * FROM predictions
                WHERE perception_run_id = ?
            """
            params = [run_id]

            if embryo_id:
                query += " AND embryo_id = ?"
                params.append(embryo_id)

            query += " ORDER BY embryo_id, timepoint LIMIT ? OFFSET ?"
            params.extend([limit, offset])

            rows = conn.execute(query, params).fetchall()
            conn.close()

            return [dict(r) for r in rows]

        # =====================================================================
        # Timeline View Data
        # =====================================================================

        @app.get("/api/timeline/{session_id}/{embryo_id}")
        async def get_timeline(session_id: str, embryo_id: str):
            """
            Get timeline data for an embryo.

            Returns timepoints with ground truth stages and any predictions.
            """
            # Get all images
            images = []
            for idx, img in enumerate(self.dataset.iter_images(
                embryo_id=embryo_id,
                session_id=session_id,
                load_image_data=False,
            )):
                images.append({
                    "index": idx,
                    "timepoint": img.timepoint,
                    "timestamp": img.timestamp,
                    "ground_truth_stage": img.ground_truth_stage,
                    "uid": img.uid,
                    "volume_path": img.volume_path,
                })

            # Get ground truth transitions
            ground_truth = self.dataset.get_ground_truth(session_id, embryo_id)

            # Get predictions if any
            conn = get_connection(self.db_path)
            predictions = conn.execute("""
                SELECT timepoint, predicted_stage, confidence, perception_run_id, reasoning
                FROM predictions
                WHERE session_id = ? AND embryo_id = ?
                ORDER BY perception_run_id DESC, timepoint
            """, (session_id, embryo_id)).fetchall()
            conn.close()

            return {
                "embryo_id": embryo_id,
                "session_id": session_id,
                "images": images,
                "ground_truth": ground_truth,
                "predictions": [dict(p) for p in predictions],
            }

        # =====================================================================
        # Projections API (for View More feature)
        # =====================================================================

        @app.get("/api/volume/{session_id}/{embryo_id}/{index}")
        async def get_volume_data(session_id: str, embryo_id: str, index: int):
            """Get volume data for 3D rendering."""
            img = self.dataset.get_image_by_index(
                embryo_id=embryo_id,
                index=index,
                session_id=session_id,
            )
            if not img or not img.volume_path:
                raise HTTPException(status_code=404, detail="Volume not found")

            vol_path = Path(img.volume_path)
            if not vol_path.exists():
                raise HTTPException(status_code=404, detail="Volume file not found")

            # Load volume
            vol = load_volume(vol_path)

            # Auto-crop to embryo region
            bounds = compute_crop_bounds(vol)
            vol = apply_crop_bounds(vol, bounds)

            z, h, w = vol.shape

            # Normalize to float
            vol_norm = vol.astype(np.float32)
            p1, p99 = np.percentile(vol_norm, [1, 99])
            vol_norm = np.clip((vol_norm - p1) / (p99 - p1 + 1e-8), 0, 1)

            # Apply Gaussian blur along Z axis to reduce banding at side views
            from scipy import ndimage
            vol_norm = ndimage.gaussian_filter1d(vol_norm, sigma=1.0, axis=0)

            vol_uint8 = (vol_norm * 255).astype(np.uint8)

            # Encode as base64
            vol_bytes = vol_uint8.tobytes()
            vol_b64 = base64.b64encode(vol_bytes).decode()

            return {
                "shape": [z, h, w],
                "data": vol_b64,
                "bounds": bounds,
            }

        @app.get("/api/projections/{session_id}/{embryo_id}/{index}")
        async def get_projections(session_id: str, embryo_id: str, index: int):
            """Get all projection types for a volume."""
            img = self.dataset.get_image_by_index(
                embryo_id=embryo_id,
                index=index,
                session_id=session_id,
            )
            if not img or not img.volume_path:
                raise HTTPException(status_code=404, detail="Volume not found")

            vol_path = Path(img.volume_path)
            if not vol_path.exists():
                raise HTTPException(status_code=404, detail="Volume file not found")

            # Load volume
            vol = load_volume(vol_path)

            # Auto-crop
            bounds = compute_crop_bounds(vol)
            vol = apply_crop_bounds(vol, bounds)

            # Generate all projections
            projections = []
            for method_name, method_func in PROJECTION_METHODS.items():
                try:
                    proj_img, desc = method_func(vol)
                    projections.append({
                        "method": method_name,
                        "description": desc,
                        "data": image_to_base64(proj_img),
                    })
                except Exception as e:
                    logger.warning(f"Projection {method_name} failed: {e}")

            return {
                "session_id": session_id,
                "embryo_id": embryo_id,
                "index": index,
                "volume_shape": list(vol.shape),
                "projections": projections,
                "ground_truth_stage": img.ground_truth_stage,
            }

        @app.get("/projections/{session_id}/{embryo_id}/{index}", response_class=HTMLResponse)
        async def projections_page(session_id: str, embryo_id: str, index: int):
            """Serve the full projection viewer page."""
            return self._get_projections_html(session_id, embryo_id, index)

    def _get_explorer_html(self) -> str:
        """Generate the explorer HTML page."""
        return """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Embryo Dataset Explorer</title>
    <style>
        * { box-sizing: border-box; margin: 0; padding: 0; }
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: #1a1a2e;
            color: #eee;
            min-height: 100vh;
        }
        .container { max-width: 1400px; margin: 0 auto; padding: 20px; }
        h1 { color: #00d4ff; margin-bottom: 20px; }
        h2 { color: #fff; margin: 20px 0 10px; font-size: 1.2em; }

        /* Stats */
        .stats {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
            gap: 15px;
            margin-bottom: 30px;
        }
        .stat-card {
            background: #16213e;
            padding: 20px;
            border-radius: 10px;
            text-align: center;
        }
        .stat-value { font-size: 2em; color: #00d4ff; font-weight: bold; }
        .stat-label { color: #888; font-size: 0.9em; }

        /* Layout */
        .main-grid {
            display: grid;
            grid-template-columns: 300px 1fr;
            gap: 20px;
        }

        /* Sidebar */
        .sidebar {
            background: #16213e;
            border-radius: 10px;
            padding: 15px;
            max-height: calc(100vh - 200px);
            overflow-y: auto;
        }
        .session-item, .embryo-item {
            padding: 10px;
            margin: 5px 0;
            background: #1a1a2e;
            border-radius: 5px;
            cursor: pointer;
            transition: background 0.2s;
        }
        .session-item:hover, .embryo-item:hover { background: #0f3460; }
        .session-item.selected, .embryo-item.selected {
            background: #0f3460;
            border-left: 3px solid #00d4ff;
        }
        .badge {
            display: inline-block;
            padding: 2px 8px;
            border-radius: 10px;
            font-size: 0.8em;
            margin-left: 5px;
        }
        .badge-gt { background: #2ecc71; color: #000; }
        .badge-count { background: #3498db; }

        /* Content area */
        .content {
            background: #16213e;
            border-radius: 10px;
            padding: 20px;
            min-height: 500px;
        }

        /* Timeline */
        .timeline {
            display: flex;
            gap: 2px;
            flex-wrap: wrap;
            margin: 20px 0;
        }
        .timeline-item {
            width: 30px;
            height: 40px;
            border-radius: 3px;
            cursor: pointer;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 0.7em;
            transition: transform 0.1s;
        }
        .timeline-item:hover { transform: scale(1.2); z-index: 10; }
        .timeline-item.selected { outline: 2px solid #fff; }

        /* Stage colors */
        .stage-early { background: #3498db; }
        .stage-bean { background: #9b59b6; }
        .stage-comma { background: #e74c3c; }
        .stage-1_5fold { background: #e67e22; }
        .stage-2fold { background: #f1c40f; color: #000; }
        .stage-pretzel { background: #2ecc71; }
        .stage-hatching { background: #1abc9c; }
        .stage-hatched { background: #ecf0f1; color: #000; }
        .stage-unknown { background: #555; }

        /* Image viewer */
        .image-viewer {
            display: flex;
            gap: 20px;
            margin-top: 20px;
        }
        .image-container {
            flex: 1;
            background: #000;
            border-radius: 10px;
            padding: 10px;
            text-align: center;
        }
        .image-container img {
            max-width: 100%;
            max-height: 400px;
            border-radius: 5px;
        }

        /* Annotation panel */
        .annotation-panel {
            width: 300px;
            background: #1a1a2e;
            border-radius: 10px;
            padding: 15px;
        }
        .stage-btn {
            display: block;
            width: 100%;
            padding: 10px;
            margin: 5px 0;
            border: none;
            border-radius: 5px;
            cursor: pointer;
            font-size: 1em;
            transition: opacity 0.2s;
        }
        .stage-btn:hover { opacity: 0.8; }
        .stage-btn.active { outline: 3px solid #fff; }

        /* Loading */
        .loading {
            text-align: center;
            padding: 40px;
            color: #888;
        }

        /* Info text */
        .info { color: #888; font-size: 0.9em; margin: 10px 0; }
        .current-gt { color: #2ecc71; font-weight: bold; }

        /* Embryo grid */
        .embryo-grid {
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(180px, 1fr));
            gap: 15px;
            margin-top: 15px;
        }
        .embryo-card {
            background: #1a1a2e;
            padding: 15px;
            border-radius: 10px;
            cursor: pointer;
            transition: all 0.2s;
            border: 2px solid transparent;
        }
        .embryo-card:hover { background: #0f3460; }
        .embryo-card.selected {
            border-color: #00d4ff;
            background: #0f3460;
        }
        .embryo-card-title {
            font-size: 1.2em;
            font-weight: bold;
            color: #00d4ff;
            margin-bottom: 5px;
        }
        .embryo-card-info {
            font-size: 0.9em;
            color: #888;
        }
        .embryo-card-stages {
            font-size: 0.8em;
            color: #2ecc71;
            margin-top: 8px;
            word-wrap: break-word;
        }

        /* Content layout - master-detail */
        .content-layout {
            display: grid;
            grid-template-columns: 200px 1fr;
            gap: 20px;
            height: 100%;
        }
        .embryo-panel {
            background: #1a1a2e;
            border-radius: 10px;
            padding: 15px;
            max-height: calc(100vh - 300px);
            overflow-y: auto;
        }
        .embryo-panel h2 {
            font-size: 1em;
            margin-bottom: 10px;
        }
        .embryo-list {
            display: flex;
            flex-direction: column;
            gap: 8px;
        }
        .embryo-list .embryo-card {
            padding: 10px;
        }
        .embryo-list .embryo-card-title {
            font-size: 1em;
        }
        .timeline-panel {
            flex: 1;
        }
    </style>
</head>
<body>
    <div class="container">
        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 20px;">
            <h1 style="margin-bottom: 0;">Embryo Dataset Explorer</h1>
            <div id="server-status" style="font-size: 0.9em;">Checking...</div>
        </div>

        <div class="stats" id="stats">
            <div class="stat-card">
                <div class="stat-value" id="stat-sessions">-</div>
                <div class="stat-label">Sessions</div>
            </div>
            <div class="stat-card">
                <div class="stat-value" id="stat-embryos">-</div>
                <div class="stat-label">Embryos</div>
            </div>
            <div class="stat-card">
                <div class="stat-value" id="stat-volumes">-</div>
                <div class="stat-label">Volumes</div>
            </div>
            <div class="stat-card">
                <div class="stat-value" id="stat-images">-</div>
                <div class="stat-label">Images</div>
            </div>
            <div class="stat-card">
                <div class="stat-value" id="stat-ground_truth">-</div>
                <div class="stat-label">Ground Truth</div>
            </div>
        </div>

        <div class="main-grid">
            <div class="sidebar">
                <h2>Sessions</h2>
                <div style="margin-bottom: 10px;">
                    <label style="font-size: 0.9em; cursor: pointer;">
                        <input type="checkbox" id="filter-with-embryos" onchange="filterSessions()" checked>
                        With embryos only
                    </label>
                </div>
                <div id="sessions-list">
                    <div class="loading">Loading...</div>
                </div>
            </div>

            <div class="content">
                <div id="content-area">
                    <div class="info">Select a session to see embryos</div>
                </div>
            </div>
        </div>
    </div>

    <script>
        // State
        let state = {
            sessions: [],
            embryos: [],
            selectedSession: null,
            selectedEmbryo: null,
            timeline: null,
            selectedIndex: null,
            currentImage: null,
        };

        const STAGES = ['early', 'bean', 'comma', '1.5fold', '2fold', 'pretzel', 'hatching', 'hatched'];

        // API helpers
        async function api(endpoint) {
            const res = await fetch('/api/' + endpoint);
            return res.json();
        }

        async function apiPost(endpoint, data) {
            const res = await fetch('/api/' + endpoint, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(data),
            });
            return res.json();
        }

        // Load stats
        async function loadStats() {
            const stats = await api('stats');
            document.getElementById('stat-sessions').textContent = stats.sessions?.toLocaleString() || 0;
            document.getElementById('stat-embryos').textContent = stats.embryos?.toLocaleString() || 0;
            document.getElementById('stat-volumes').textContent = stats.volumes?.toLocaleString() || 0;
            document.getElementById('stat-images').textContent = stats.images?.toLocaleString() || 0;
            document.getElementById('stat-ground_truth').textContent = stats.ground_truth?.toLocaleString() || 0;
        }

        // Load sessions
        async function loadSessions() {
            state.sessions = await api('sessions');
            renderSessions();
        }

        function renderSessions() {
            const container = document.getElementById('sessions-list');
            const filterWithEmbryos = document.getElementById('filter-with-embryos').checked;

            const filteredSessions = filterWithEmbryos
                ? state.sessions.filter(s => s.embryo_count > 0)
                : state.sessions;

            container.innerHTML = filteredSessions.map(s => {
                const date = s.created_at ? s.created_at.split('T')[0] : '';
                return `
                <div class="session-item ${state.selectedSession === s.session_id ? 'selected' : ''}"
                     onclick="selectSession('${s.session_id}')">
                    <div><strong>${s.session_id}</strong></div>
                    <div style="font-size: 0.8em; color: #888;">
                        ${date} | ${s.embryo_count} embryos
                        ${s.has_ground_truth ? '<span class="badge badge-gt">GT</span>' : ''}
                    </div>
                </div>
            `}).join('');
        }

        function filterSessions() {
            renderSessions();
        }

        // Select session
        async function selectSession(sessionId) {
            state.selectedSession = sessionId;
            state.selectedEmbryo = null;
            state.timeline = null;
            renderSessions();

            const embryos = await api(`embryos?session_id=${sessionId}`);
            state.embryos = embryos;
            renderEmbryoGrid();
        }

        function renderEmbryoGrid() {
            const container = document.getElementById('content-area');
            if (!state.embryos || state.embryos.length === 0) {
                container.innerHTML = '<div class="info">No embryos in this session</div>';
                return;
            }

            container.innerHTML = `
                <div class="content-layout">
                    <div class="embryo-panel">
                        <h2>Session ${state.selectedSession}</h2>
                        <div class="embryo-list">
                            ${state.embryos.map(e => `
                                <div class="embryo-card ${state.selectedEmbryo === e.embryo_id ? 'selected' : ''}"
                                     onclick="selectEmbryo('${e.embryo_id}')">
                                    <div class="embryo-card-title">${e.embryo_id}</div>
                                    <div class="embryo-card-info">
                                        ${e.num_volumes} vol
                                        ${e.has_ground_truth ? '<span class="badge badge-gt">GT</span>' : ''}
                                    </div>
                                </div>
                            `).join('')}
                        </div>
                    </div>
                    <div class="timeline-panel" id="timeline-panel">
                        <div class="info">Select an embryo to view timeline</div>
                    </div>
                </div>
            `;
        }

        // Select embryo
        async function selectEmbryo(embryoId) {
            state.selectedEmbryo = embryoId;
            state.selectedIndex = null;

            // Update embryo selection visually
            document.querySelectorAll('.embryo-card').forEach(el => {
                el.classList.toggle('selected', el.querySelector('.embryo-card-title')?.textContent === embryoId);
            });

            const timeline = await api(`timeline/${state.selectedSession}/${embryoId}`);
            state.timeline = timeline;

            renderTimeline();
        }

        function renderTimeline() {
            if (!state.timeline) return;

            const t = state.timeline;
            const container = document.getElementById('timeline-panel');
            if (!container) return;

            container.innerHTML = `
                <h2>${t.embryo_id}</h2>

                <div class="info">
                    ${t.images.length} images |
                    GT: ${t.ground_truth.length > 0 ? t.ground_truth.map(g => g.stage).join(' → ') : 'None'}
                </div>

                <div class="timeline" id="timeline">
                    ${t.images.map((img, idx) => `
                        <div class="timeline-item stage-${(img.ground_truth_stage || 'unknown').replace('.', '_')} ${state.selectedIndex === idx ? 'selected' : ''}"
                             onclick="selectIndex(${idx})"
                             title="#${idx}: ${img.ground_truth_stage || 'unlabeled'}">
                        </div>
                    `).join('')}
                </div>

                <div class="image-viewer" id="image-viewer">
                    <div class="info">Click a timeline dot to view image</div>
                </div>
            `;
        }

        // Select image by index
        async function selectIndex(idx) {
            state.selectedIndex = idx;

            // Update timeline selection
            document.querySelectorAll('.timeline-item').forEach((el, i) => {
                el.classList.toggle('selected', i === idx);
            });

            // Load image by index
            try {
                const img = await api(`image/${state.selectedSession}/${state.selectedEmbryo}/${idx}`);
                state.currentImage = img;
                state.currentImage.index = idx;

                // Look up prediction for this index if available
                if (state.timeline && state.timeline.predictions) {
                    const pred = state.timeline.predictions.find(p => p.timepoint === idx);
                    state.currentImage.prediction = pred || null;
                }

                renderImageViewer();
            } catch (err) {
                console.error('Failed to load image:', err);
            }
        }

        function renderImageViewer() {
            const img = state.currentImage;
            if (!img) return;

            const idx = state.selectedIndex;
            const pred = img.prediction;
            const viewer = document.getElementById('image-viewer');

            // Build prediction display if available
            let predictionHtml = '';
            if (pred) {
                const confidence = pred.confidence ? (pred.confidence * 100).toFixed(0) + '%' : 'N/A';
                const isCorrect = pred.predicted_stage === img.ground_truth_stage;
                const predColor = isCorrect ? '#2ecc71' : '#e74c3c';
                predictionHtml = `
                    <div style="margin-top: 15px; padding: 10px; background: #1a1a2e; border-radius: 8px;">
                        <h3 style="color: #00d4ff; margin-bottom: 8px;">Prediction</h3>
                        <div style="display: flex; gap: 15px; margin-bottom: 10px;">
                            <div>Stage: <span style="color: ${predColor}; font-weight: bold;">${pred.predicted_stage}</span></div>
                            <div>Confidence: <span style="color: #f1c40f;">${confidence}</span></div>
                            ${img.ground_truth_stage ? `<div>${isCorrect ? '✓ Correct' : '✗ Wrong'}</div>` : ''}
                        </div>
                        ${pred.reasoning ? `
                            <details style="margin-top: 10px;">
                                <summary style="cursor: pointer; color: #888;">View Reasoning</summary>
                                <div style="margin-top: 8px; padding: 10px; background: #0d1117; border-radius: 5px; font-size: 0.9em; white-space: pre-wrap; max-height: 200px; overflow-y: auto;">
                                    ${pred.reasoning}
                                </div>
                            </details>
                        ` : ''}
                    </div>
                `;
            }

            viewer.innerHTML = `
                <div class="image-container">
                    <div style="margin-bottom: 10px; display: flex; justify-content: space-between; align-items: center;">
                        <div>
                            <strong>Image #${idx}</strong>
                            ${img.ground_truth_stage ? `<span class="current-gt"> - GT: ${img.ground_truth_stage}</span>` : ''}
                            <span style="color: #666; font-size: 0.9em;"> ${img.timestamp || ''}</span>
                        </div>
                        <button onclick="openProjections()" style="padding: 5px 12px; background: #3498db; border: none; border-radius: 5px; color: #fff; cursor: pointer; font-size: 0.9em;">
                            View More →
                        </button>
                    </div>
                    ${img.image_b64 ? `<img src="data:image/jpeg;base64,${img.image_b64}" alt="Image ${idx}">` : '<div class="info">Image not available</div>'}
                    ${predictionHtml}
                </div>

                <div class="annotation-panel">
                    <h2>Set Stage Transition</h2>
                    <p class="info">Click to mark when this stage STARTS at index ${idx}</p>
                    ${STAGES.map(stage => `
                        <button class="stage-btn stage-${stage.replace('.', '_')} ${img.ground_truth_stage === stage ? 'active' : ''}"
                                onclick="setGroundTruth('${stage}', ${idx})">
                            ${stage}
                        </button>
                    `).join('')}

                    <div style="margin-top: 20px;">
                        <h2>Navigation</h2>
                        <button class="stage-btn" style="background: #555;" onclick="prevImage()">← Previous</button>
                        <button class="stage-btn" style="background: #555;" onclick="nextImage()">Next →</button>
                    </div>
                </div>
            `;
        }

        function openProjections() {
            const url = `/projections/${state.selectedSession}/${state.selectedEmbryo}/${state.selectedIndex}`;
            window.open(url, '_blank');
        }

        // Set ground truth
        async function setGroundTruth(stage, idx) {
            await apiPost('ground_truth', {
                session_id: state.selectedSession,
                embryo_id: state.selectedEmbryo,
                stage: stage,
                start_timepoint: idx,  // Using index as timepoint
                annotator: 'web_explorer',
            });

            // Reload timeline
            await selectEmbryo(state.selectedEmbryo);
            await selectIndex(idx);

            // Refresh stats
            loadStats();
        }

        // Navigation
        function prevImage() {
            if (!state.timeline || state.selectedIndex === null) return;
            if (state.selectedIndex > 0) {
                selectIndex(state.selectedIndex - 1);
            }
        }

        function nextImage() {
            if (!state.timeline || state.selectedIndex === null) return;
            if (state.selectedIndex < state.timeline.images.length - 1) {
                selectIndex(state.selectedIndex + 1);
            }
        }

        // Keyboard navigation
        document.addEventListener('keydown', (e) => {
            if (e.key === 'ArrowLeft') prevImage();
            if (e.key === 'ArrowRight') nextImage();
        });

        // Server status indicator
        async function checkServerStatus() {
            const indicator = document.getElementById('server-status');
            try {
                const start = Date.now();
                await api('stats');
                const latency = Date.now() - start;
                indicator.innerHTML = `<span style="color: #2ecc71;">● Online</span> <span style="color: #666; font-size: 0.8em;">(${latency}ms)</span>`;
                indicator.title = 'Server is responding';
            } catch (err) {
                indicator.innerHTML = '<span style="color: #e74c3c;">● Offline</span>';
                indicator.title = 'Cannot reach server: ' + err.message;
            }
        }

        // Init
        loadStats();
        loadSessions();
        checkServerStatus();
        // Periodically check server status
        setInterval(checkServerStatus, 30000);
    </script>
</body>
</html>"""

    def _get_projections_html(self, session_id: str, embryo_id: str, index: int) -> str:
        """Generate the projections viewer page."""
        return f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Projections - {embryo_id} #{index}</title>
    <style>
        * {{ box-sizing: border-box; margin: 0; padding: 0; }}
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: #1a1a2e;
            color: #eee;
            min-height: 100vh;
            padding: 20px;
        }}
        .header {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 20px;
        }}
        h1 {{ color: #00d4ff; }}
        .info {{ color: #888; font-size: 0.9em; }}
        .grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(400px, 1fr));
            gap: 20px;
        }}
        .card {{
            background: #16213e;
            border-radius: 10px;
            padding: 15px;
        }}
        .card h3 {{ color: #fff; margin-bottom: 10px; }}
        .card img {{
            width: 100%;
            border-radius: 5px;
            background: #000;
        }}
        .loading {{
            text-align: center;
            padding: 40px;
            color: #888;
        }}
        .back-link {{
            color: #00d4ff;
            text-decoration: none;
        }}
        .back-link:hover {{ text-decoration: underline; }}
        /* 3D viewer container */
        #viewer3d-container {{
            background: #000;
            border-radius: 5px;
            height: 500px;
            position: relative;
        }}
        #viewer3d {{ width: 100%; height: 100%; }}
        .controls {{
            position: absolute;
            top: 10px;
            right: 10px;
            background: rgba(0,0,0,0.7);
            padding: 10px;
            border-radius: 5px;
        }}
        .controls label {{ display: block; margin: 5px 0; font-size: 0.9em; }}
        .controls input {{ width: 100px; }}
    </style>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>
</head>
<body>
    <div class="header">
        <div>
            <h1>Projections Viewer</h1>
            <div class="info">
                Session: {session_id} | Embryo: {embryo_id} | Index: #{index}
            </div>
        </div>
        <a href="/" class="back-link">← Back to Explorer</a>
    </div>

    <div id="content">
        <div class="loading">Loading projections...</div>
    </div>

    <script>
        const SESSION = '{session_id}';
        const EMBRYO = '{embryo_id}';
        const INDEX = {index};

        async function loadProjections() {{
            try {{
                const res = await fetch(`/api/projections/${{SESSION}}/${{EMBRYO}}/${{INDEX}}`);
                const data = await res.json();
                renderProjections(data);
            }} catch (err) {{
                document.getElementById('content').innerHTML = `
                    <div class="card">
                        <h3>Error</h3>
                        <p>Failed to load projections: ${{err.message}}</p>
                    </div>
                `;
            }}
        }}

        function renderProjections(data) {{
            const container = document.getElementById('content');

            let html = `
                <div class="info" style="margin-bottom: 15px;">
                    Volume shape: ${{data.volume_shape.join(' x ')}}
                    ${{data.ground_truth_stage ? ` | Ground Truth: <strong>${{data.ground_truth_stage}}</strong>` : ''}}
                </div>
                <div class="grid">
            `;

            // Add each projection
            for (const proj of data.projections) {{
                html += `
                    <div class="card">
                        <h3>${{proj.method.replace('_', ' ').toUpperCase()}}</h3>
                        <p class="info">${{proj.description}}</p>
                        <img src="data:image/jpeg;base64,${{proj.data}}" alt="${{proj.method}}">
                    </div>
                `;
            }}

            // Add 3D viewer card
            html += `
                <div class="card" style="grid-column: span 2;">
                    <h3>3D VOLUME VIEWER</h3>
                    <p class="info">Drag: rotate X/Y | Shift+Drag: rotate Z | Scroll: zoom |
                        Threshold: <input type="range" id="thresh3d" min="0" max="100" value="30" style="width:80px;vertical-align:middle;">
                        <span id="thresh-display" style="font-family:monospace;color:#58a6ff;min-width:35px;display:inline-block;">0.30</span>
                        | <span id="angle-display" style="font-family:monospace;color:#58a6ff;">angle_y: 0.50, angle_x: -0.50</span>
                    </p>
                    <div id="viewer3d-container" style="height:500px;background:#000;"></div>
                </div>
            `;

            html += '</div>';
            container.innerHTML = html;

            // Load 3D volume data
            load3DViewer();
        }}

        let scene3d, camera3d, renderer3d, sliceGroup;
        let volumeData, volumeShape;
        let savedRotation = {{ x: -0.5, y: 0.5 }};
        let savedZoom = 0.9;

        async function load3DViewer() {{
            try {{
                const res = await fetch(`/api/volume/${{SESSION}}/${{EMBRYO}}/${{INDEX}}`);
                const volData = await res.json();
                volumeShape = volData.shape;

                const raw = atob(volData.data);
                volumeData = new Uint8Array(raw.length);
                for (let i = 0; i < raw.length; i++) {{
                    volumeData[i] = raw.charCodeAt(i);
                }}

                init3DViewer();
            }} catch (err) {{
                console.error('Failed to load 3D volume:', err);
            }}
        }}

        function createSliceTex(zIndex, threshold) {{
            const [zd, h, w] = volumeShape;
            const sliceSize = w * h;
            const offset = zIndex * sliceSize;
            const rgba = new Uint8Array(w * h * 4);
            for (let i = 0; i < sliceSize; i++) {{
                const val = volumeData[offset + i];
                if (val > threshold) {{
                    rgba[i * 4] = val;
                    rgba[i * 4 + 1] = val;
                    rgba[i * 4 + 2] = val;
                    rgba[i * 4 + 3] = Math.min(255, (val - threshold) * 2);
                }} else {{
                    rgba[i * 4 + 3] = 0;
                }}
            }}
            const tex = new THREE.DataTexture(rgba, w, h, THREE.RGBAFormat);
            tex.needsUpdate = true;
            return tex;
        }}

        function buildSlices3d(numSlices, threshold) {{
            if (!volumeShape) return;
            const [zd, h, w] = volumeShape;

            // Clear old slices
            while (sliceGroup.children.length > 0) {{
                const c = sliceGroup.children[0];
                c.geometry.dispose();
                c.material.dispose();
                sliceGroup.remove(c);
            }}

            const aspect = w / h;
            const zScale = (zd / w) * 3;  // Exaggerate Z for depth

            for (let i = 0; i < numSlices; i++) {{
                const zIndex = Math.floor(i * zd / numSlices);
                const zPos = (i / numSlices - 0.5) * zScale;
                const tex = createSliceTex(zIndex, threshold);
                const mat = new THREE.MeshBasicMaterial({{
                    map: tex,
                    transparent: true,
                    side: THREE.DoubleSide,
                    depthWrite: false
                }});
                const geo = new THREE.PlaneGeometry(1, 1 / aspect);
                const mesh = new THREE.Mesh(geo, mat);
                mesh.position.z = zPos;
                sliceGroup.add(mesh);
            }}
        }}

        function init3DViewer() {{
            const container = document.getElementById('viewer3d-container');
            const w = container.clientWidth || 600;
            const h = container.clientHeight || 500;

            scene3d = new THREE.Scene();
            camera3d = new THREE.PerspectiveCamera(50, w / h, 0.1, 100);
            camera3d.position.z = savedZoom;

            renderer3d = new THREE.WebGLRenderer({{ antialias: true }});
            renderer3d.setSize(w, h);
            renderer3d.setClearColor(0x000000);
            container.appendChild(renderer3d.domElement);

            sliceGroup = new THREE.Group();
            sliceGroup.rotation.x = savedRotation.x;
            sliceGroup.rotation.y = savedRotation.y;
            sliceGroup.rotation.z = 0;
            sliceGroup.scale.y = -1;  // Flip Y to match dual_view orientation
            scene3d.add(sliceGroup);

            buildSlices3d(32, 30);

            // Threshold control
            document.getElementById('thresh3d').addEventListener('input', (e) => {{
                const threshVal = parseInt(e.target.value);
                buildSlices3d(32, threshVal);
                document.getElementById('thresh-display').textContent = (threshVal / 100).toFixed(2);
            }});

            // Mouse controls
            let isDragging = false;
            let prevMouse = {{ x: 0, y: 0 }};

            renderer3d.domElement.addEventListener('mousedown', (e) => {{
                isDragging = true;
                prevMouse = {{ x: e.clientX, y: e.clientY }};
            }});

            window.addEventListener('mouseup', () => isDragging = false);

            renderer3d.domElement.addEventListener('mousemove', (e) => {{
                if (!isDragging) return;
                if (e.shiftKey) {{
                    // Shift+drag: rotate around Z axis
                    sliceGroup.rotation.z += (e.clientX - prevMouse.x) * 0.01;
                }} else {{
                    // Normal drag: rotate around X and Y
                    sliceGroup.rotation.y += (e.clientX - prevMouse.x) * 0.01;
                    sliceGroup.rotation.x += (e.clientY - prevMouse.y) * 0.01;
                }}
                savedRotation.x = sliceGroup.rotation.x;
                savedRotation.y = sliceGroup.rotation.y;

                document.getElementById('angle-display').textContent =
                    'angle_y: ' + sliceGroup.rotation.y.toFixed(2) +
                    ', angle_x: ' + sliceGroup.rotation.x.toFixed(2);
                prevMouse = {{ x: e.clientX, y: e.clientY }};
            }});

            renderer3d.domElement.addEventListener('wheel', (e) => {{
                e.preventDefault();
                camera3d.position.z = Math.max(0.5, Math.min(5, camera3d.position.z + e.deltaY * 0.002));
                savedZoom = camera3d.position.z;
            }});

            // Animation loop
            function animate() {{
                requestAnimationFrame(animate);
                renderer3d.render(scene3d, camera3d);
            }}
            animate();

            // Handle resize
            window.addEventListener('resize', () => {{
                const w = container.clientWidth || 600;
                const h = container.clientHeight || 500;
                camera3d.aspect = w / h;
                camera3d.updateProjectionMatrix();
                renderer3d.setSize(w, h);
            }});

            console.log('3D slice viewer initialized');
        }}

        loadProjections();
    </script>
</body>
</html>"""

    def run(self):
        """Start the server."""
        import uvicorn
        print(f"\n=== Embryo Dataset Explorer ===")
        print(f"Database: {self.db_path}")
        print(f"Open http://localhost:{self.port} in your browser\n")
        uvicorn.run(self.app, host=self.host, port=self.port)


def main():
    """CLI entry point for explorer."""
    import argparse

    parser = argparse.ArgumentParser(description="Embryo Dataset Explorer")
    parser.add_argument("--db", default=str(DEFAULT_DB_PATH), help="Database path")
    parser.add_argument("--port", type=int, default=8765, help="Port")
    parser.add_argument("--host", default="0.0.0.0", help="Host")

    args = parser.parse_args()

    explorer = DatasetExplorer(
        db_path=Path(args.db),
        port=args.port,
        host=args.host,
    )
    explorer.run()


if __name__ == "__main__":
    main()
