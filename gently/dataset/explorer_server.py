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

        # Set up template and static file paths
        self.base_dir = Path(__file__).parent
        self.templates_dir = self.base_dir / "templates"
        self.static_dir = self.base_dir / "static"

        self.app = FastAPI(
            title="Embryo Dataset Explorer",
            description="Browse and annotate embryo imaging data",
            version="1.0.0",
        )

        # Mount static files
        self.app.mount("/static", StaticFiles(directory=str(self.static_dir)), name="static")

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
        # Cross-Session Embryo UID Endpoints
        # =====================================================================

        @app.get("/api/embryo_by_uid/{uid}")
        async def get_embryo_by_uid(uid: str):
            """Get all instances of an embryo across sessions by its UID."""
            instances = self.dataset.get_embryo_by_uid(uid)
            if not instances:
                raise HTTPException(status_code=404, detail="Embryo UID not found")
            return instances

        @app.get("/api/embryo_timeline_by_uid/{uid}")
        async def get_embryo_timeline_by_uid(uid: str):
            """Get complete cross-session timeline for an embryo."""
            timeline = self.dataset.get_embryo_timeline_by_uid(uid)
            if not timeline.get("sessions"):
                raise HTTPException(status_code=404, detail="Embryo UID not found")
            return timeline

        @app.get("/api/embryos_with_multiple_sessions")
        async def get_embryos_with_multiple_sessions():
            """Get embryos that appear in multiple sessions (imported)."""
            return self.dataset.get_embryos_with_multiple_sessions()

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
        """Load and return the explorer HTML page from template."""
        template_path = self.templates_dir / "explorer.html"
        return template_path.read_text(encoding="utf-8")

    def _get_projections_html(self, session_id: str, embryo_id: str, index: int) -> str:
        """Load and return the projections HTML page from template."""
        template_path = self.templates_dir / "projections.html"
        html = template_path.read_text(encoding="utf-8")
        # Replace template placeholders
        html = html.replace("{{session_id}}", session_id)
        html = html.replace("{{embryo_id}}", embryo_id)
        html = html.replace("{{index}}", str(index))
        return html

    # =========================================================================
    # OLD INLINE HTML REMOVED - Now in templates/ and static/ directories
    # =========================================================================
    # - templates/explorer.html
    # - templates/projections.html
    # - static/css/explorer.css
    # - static/css/projections.css
    # - static/js/explorer.js
    # - static/js/projections.js
    # =========================================================================

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
