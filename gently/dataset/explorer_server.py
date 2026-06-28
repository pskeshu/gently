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

import asyncio
import base64
import json
import logging
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
from fastapi import FastAPI, HTTPException, Query, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from gently.core.imaging import (
    apply_crop_bounds,
    compute_crop_bounds,
    load_volume,
    normalize_to_uint8,
    projection_three_view,
)
from gently.core.imaging import (
    image_to_base64 as _image_to_base64,
)

from .embryo_dataset import EmbryoDataset
from .schema import DEFAULT_DB_PATH, get_connection, get_database_stats

logger = logging.getLogger(__name__)
# Lazy imports for explorer-specific projection functions
tifffile: Any = None
PIL_Image: Any = None


def ensure_projection_deps():
    """Ensure projection dependencies are available."""
    global tifffile, PIL_Image
    if tifffile is None:
        import tifffile as _tifffile

        tifffile = _tifffile
    if PIL_Image is None:
        from PIL import Image as _Image

        PIL_Image = _Image


def normalize_image(img: np.ndarray, p_low: float = 1, p_high: float = 99) -> np.ndarray:
    """Normalize image to 0-255 uint8 using percentile scaling."""
    return normalize_to_uint8(img, method="percentile", p_low=p_low, p_high=p_high)


def image_to_base64(img: np.ndarray, format: str = "JPEG", quality: int = 90) -> str:
    """Convert numpy array to base64-encoded image."""
    return _image_to_base64(img, format=format, quality=quality, ensure_rgb=True)


def find_outer_boundary(img: np.ndarray, percentile: float = 50) -> np.ndarray:
    """Find outer boundary of embryo by thresholding and extracting mask edge."""
    thresh = np.percentile(img, percentile)
    mask = img > thresh
    padded = np.pad(mask, 1, mode="constant", constant_values=True)
    eroded = (
        padded[:-2, :-2]
        & padded[:-2, 1:-1]
        & padded[:-2, 2:]
        & padded[1:-1, :-2]
        & padded[1:-1, 1:-1]
        & padded[1:-1, 2:]
        & padded[2:, :-2]
        & padded[2:, 1:-1]
        & padded[2:, 2:]
    )
    boundary = (mask & ~eroded).astype(np.uint8) * 255
    return boundary


def overlay_edges(
    img: np.ndarray, edges: np.ndarray, color: tuple[int, int, int] = (255, 200, 0)
) -> np.ndarray:
    """Overlay edge contours on image in specified color."""
    if img.ndim == 2:
        rgb = np.stack([img, img, img], axis=-1)
    else:
        rgb = img.copy()
    edge_mask = edges > 0
    rgb[edge_mask] = color
    return rgb


def projection_dual_view(
    volume: np.ndarray,
    voxel_size: tuple[float, float, float] = (1.0, 0.1625, 0.1625),
) -> tuple[np.ndarray, str]:
    """Dual-view projection: TOP above, SIDE below with boundary overlay.

    voxel_size: (dz, dy, dx) in microns. Used to compute the physically
    correct height of the side view so the XZ panel is isometric with the
    XY panel. Default matches projection_three_view.
    """
    ensure_projection_deps()
    if volume.ndim != 3:
        return normalize_image(volume), "2D input"
    z_depth, height, width = volume.shape
    dz, dy, dx = voxel_size
    top_proj = np.max(volume, axis=0)
    side_proj = np.max(volume, axis=1)
    top_norm = normalize_image(top_proj)
    side_norm = normalize_image(side_proj)
    top_edges = find_outer_boundary(top_norm, percentile=50)
    side_edges = find_outer_boundary(side_norm, percentile=50)
    top_rgb = overlay_edges(top_norm, top_edges)
    side_rgb = overlay_edges(side_norm, side_edges)
    target_width = top_rgb.shape[1]
    # Isometric: each Z voxel covers dz microns, each X display pixel covers
    # dx microns, so a side view with 1:1 aspect needs height = z_depth*dz/dx.
    # Floor at height//3 so very thin embryos still have a visible side view.
    z_display_h = max(1, int(round(z_depth * dz / dx)))
    side_new_h = max(height // 3, z_display_h)
    pil_side = PIL_Image.fromarray(side_rgb)
    pil_side = pil_side.resize((target_width, side_new_h), PIL_Image.Resampling.LANCZOS)
    side_scaled = np.array(pil_side)
    sep = np.ones((3, target_width, 3), dtype=np.uint8) * 128
    combined = np.concatenate([top_rgb, sep, side_scaled], axis=0)
    return combined, "Dual-view MIP with boundary: TOP (XY) + SIDE (XZ)"


def projection_depth_colored(
    volume: np.ndarray,
    colormap: str = "turbo",
    voxel_size: tuple[float, float, float] = (1.0, 0.1625, 0.1625),
) -> tuple[np.ndarray, str]:
    """Depth-colored max intensity projection.

    voxel_size: (dz, dy, dx) in microns. The side view is rescaled so the
    XZ panel is isometric with the XY panel above it.
    """
    ensure_projection_deps()
    if volume.ndim != 3:
        gray = normalize_image(volume)
        return np.stack([gray, gray, gray], axis=-1), "2D input"
    z_depth, height, width = volume.shape
    dz, dy, dx = voxel_size
    try:
        import matplotlib.pyplot as plt

        cmap = plt.get_cmap(colormap)
    except Exception:
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
    # Isometric side height - see projection_dual_view for the same math.
    z_display_h = max(1, int(round(z_depth * dz / dx)))
    side_new_h = max(height // 3, z_display_h)
    pil_side = pil_side.resize((width, side_new_h), PIL_Image.Resampling.LANCZOS)
    side_scaled = np.array(pil_side)
    sep = np.ones((3, width, 3), dtype=np.uint8) * 128
    combined = np.concatenate([top_rgb, sep, side_scaled], axis=0)
    return combined, f"Z-depth colored MIP ({colormap}): TOP + SIDE"


def projection_multi_slice(volume: np.ndarray, n_slices: int = 6) -> tuple[np.ndarray, str]:
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
        row_slices = slices[r * n_cols : (r + 1) * n_cols]
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


def render_volume_rotated(
    volume: np.ndarray,
    angle_y: float,
    angle_x: float = -0.5,
    threshold: float = 0.12,
    num_slices: int = 48,
    perspective: float = 0.4,
    voxel_size: tuple[float, float, float] = (1.0, 0.1625, 0.1625),
) -> np.ndarray:
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
    voxel_size : tuple of (float, float, float)
        Physical voxel dimensions (dz, dy, dx) in microns. Used to compute
        the correct physical Z extent so the rotated view is isometric.

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
    dz, dy, dx = voxel_size

    # Physical Z extent normalized to width units. Previously this was a
    # hardcoded (z_depth / width) * 3.0, which under-scaled Z by a factor
    # of ~2 for the default diSPIM voxel (1.0 um Z step vs 0.1625 um XY).
    z_scale = (z_depth * dz) / (width * dx)

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

        slice_data_list.append(
            {
                "z_idx": z_idx,
                "shift_x": shift_x,
                "shift_y": shift_y,
                "depth": depth_after_rotation,
                "scale": perspective_scale,
            }
        )

    # Sort by depth (back to front for proper alpha compositing)
    slice_data_list.sort(key=lambda s: s["depth"])

    # Composite slices
    for slice_info in slice_data_list:
        z_idx = slice_info["z_idx"]
        shift_x = slice_info["shift_x"]
        shift_y = slice_info["shift_y"]
        scale = slice_info["scale"]

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
                src_slice * src_alpha
                + result[dst_y_start:dst_y_end, dst_x_start:dst_x_end, c] * (1 - src_alpha)
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


def projection_spin_3d(
    volume: np.ndarray,
    voxel_size: tuple[float, float, float] = (1.0, 0.1625, 0.1625),
) -> tuple[np.ndarray, str]:
    """Multiple 3D perspective views from different angles (2x3 grid).

    voxel_size: (dz, dy, dx) in microns. Forwarded to render_volume_rotated
    so the spin views are Z-isometric.
    """
    ensure_projection_deps()
    if volume.ndim != 3:
        return normalize_image(volume), "2D input"

    py_threshold = 0.12
    base_tilt = 0.21
    angles_y = [-0.05, 0.5, 1.0, 1.57, 2.1, 2.6]

    views = []
    for angle_y in angles_y:
        view = render_volume_rotated(
            volume,
            angle_y=angle_y,
            angle_x=base_tilt,
            threshold=py_threshold,
            perspective=0.5,
            voxel_size=voxel_size,
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
            p[pad_h : pad_h + h, pad_w : pad_w + w] = v
        else:
            p = np.zeros((target_h, target_w), dtype=v.dtype)
            p[pad_h : pad_h + h, pad_w : pad_w + w] = v
        padded.append(p)

    row1 = np.hstack(padded[0:3])
    row2 = np.hstack(padded[3:6])
    grid = np.vstack([row1, row2])

    return grid, "6 perspective views rotating around volume"


PROJECTION_METHODS: dict[str, Any] = {
    "dual_view": projection_dual_view,
    "depth_colored": projection_depth_colored,
    "multi_slice": projection_multi_slice,
    "three_view": projection_three_view,
    "spin_3d": projection_spin_3d,
}


logger = logging.getLogger(__name__)


# =============================================================================
# Presence Tracking (Collaborative Feature)
# =============================================================================


@dataclass
class ClientInfo:
    """Information about a connected WebSocket client for presence tracking"""

    client_id: str
    name: str
    color: str  # Hex color for avatar background
    connected_at: str


class ConnectionManager:
    """Manages WebSocket connections for broadcasting updates with presence tracking"""

    # Colors for avatar backgrounds (pleasant, distinct colors)
    AVATAR_COLORS = [
        "#4a9eff",
        "#ff6b6b",
        "#51cf66",
        "#ffd43b",
        "#cc5de8",
        "#ff922b",
        "#20c997",
        "#748ffc",
        "#f06595",
        "#69db7c",
        "#ffa94d",
        "#9775fa",
        "#38d9a9",
        "#e599f7",
        "#74c0fc",
    ]

    def __init__(self):
        self.active_connections: dict[WebSocket, ClientInfo] = {}
        self._lock = asyncio.Lock()

    def _generate_color(self, client_id: str) -> str:
        """Generate consistent color from client_id"""
        hash_val = sum(ord(c) for c in client_id)
        return self.AVATAR_COLORS[hash_val % len(self.AVATAR_COLORS)]

    async def connect(
        self, websocket: WebSocket, client_id: str | None = None, name: str | None = None
    ):
        await websocket.accept()

        # Generate defaults if not provided
        if not client_id:
            import uuid

            client_id = str(uuid.uuid4())[:8]
        if not name:
            name = f"Anonymous {client_id[:4]}"

        client_info = ClientInfo(
            client_id=client_id,
            name=name,
            color=self._generate_color(client_id),
            connected_at=datetime.now().isoformat(),
        )

        async with self._lock:
            self.active_connections[websocket] = client_info
        logger.info(
            f"WebSocket connected: {name} ({client_id}). Total: {len(self.active_connections)}"
        )

        # Broadcast updated presence to all clients
        await self.broadcast_presence()

    async def disconnect(self, websocket: WebSocket):
        async with self._lock:
            client_info = self.active_connections.pop(websocket, None)
        if client_info:
            logger.info(
                f"WebSocket disconnected: {client_info.name}. Total: {len(self.active_connections)}"
            )
        else:
            logger.info(f"WebSocket disconnected. Total: {len(self.active_connections)}")

        # Broadcast updated presence to remaining clients
        await self.broadcast_presence()

    async def update_client_name(self, websocket: WebSocket, name: str):
        """Update a client's display name"""
        async with self._lock:
            if websocket in self.active_connections:
                old_info = self.active_connections[websocket]
                self.active_connections[websocket] = ClientInfo(
                    client_id=old_info.client_id,
                    name=name,
                    color=old_info.color,
                    connected_at=old_info.connected_at,
                )
        await self.broadcast_presence()

    def get_client_info(self, websocket: WebSocket) -> ClientInfo | None:
        """Get client info for a websocket"""
        return self.active_connections.get(websocket)

    async def broadcast_presence(self):
        """Broadcast current presence list to all clients"""
        if not self.active_connections:
            return

        # Deduplicate by client_id (same user in multiple tabs = one avatar)
        async with self._lock:
            seen_clients = {}
            for _ws, info in self.active_connections.items():
                # Keep the most recent entry for each client_id
                seen_clients[info.client_id] = {
                    "client_id": info.client_id,
                    "name": info.name,
                    "color": info.color,
                }
            clients_list = list(seen_clients.values())

        # Send personalized presence to each client (with is_you flag)
        disconnected = []
        for ws, info in list(self.active_connections.items()):
            try:
                personalized = []
                for client in clients_list:
                    personalized.append({**client, "is_you": client["client_id"] == info.client_id})
                await ws.send_json({"type": "presence", "clients": personalized})
            except Exception:
                disconnected.append(ws)

        # Remove stale connections
        async with self._lock:
            for ws in disconnected:
                self.active_connections.pop(ws, None)


# Pydantic models for API
class GroundTruthCreate(BaseModel):
    session_id: str
    embryo_id: str
    stage: str
    start_timepoint: int
    end_timepoint: int | None = None
    annotator: str | None = None
    notes: str | None = None


class GroundTruthDelete(BaseModel):
    session_id: str
    embryo_id: str
    stage: str | None = None


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

        # Presence tracking for collaborative features
        self.manager = ConnectionManager()

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
                "SELECT * FROM sessions WHERE session_id = ?", (session_id,)
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
            session_id: str | None = None,
            has_ground_truth: bool | None = None,
        ):
            """List embryos with optional filters."""
            embryos = []
            for embryo in self.dataset.iter_embryos(
                session_id=session_id,
                has_ground_truth=has_ground_truth,
            ):
                embryos.append(
                    {
                        "embryo_id": embryo.embryo_id,
                        "session_id": embryo.session_id,
                        "num_images": embryo.num_images,
                        "num_volumes": embryo.num_volumes,
                        "timepoint_range": embryo.timepoint_range,
                        "has_ground_truth": embryo.has_ground_truth,
                        "ground_truth_stages": embryo.ground_truth_stages,
                    }
                )
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
            if not timeline.get("timeline"):
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
            start_tp: int | None = None,
            end_tp: int | None = None,
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
                end_timepoint=data.end_timepoint,
                annotator=data.annotator,
                notes=data.notes,
            )
            end_str = f"-{data.end_timepoint}" if data.end_timepoint else ""
            return {
                "status": "ok",
                "message": f"Set {data.stage} @ t={data.start_timepoint}{end_str}",
            }

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
            embryo_id: str | None = None,
            limit: int = Query(100, le=1000),
            offset: int = 0,
        ):
            """Get predictions for a perception run."""
            conn = get_connection(self.db_path)

            query = """
                SELECT * FROM predictions
                WHERE perception_run_id = ?
            """
            params: list[Any] = [run_id]

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
            for idx, img in enumerate(
                self.dataset.iter_images(
                    embryo_id=embryo_id,
                    session_id=session_id,
                    load_image_data=False,
                )
            ):
                images.append(
                    {
                        "index": idx,
                        "timepoint": img.timepoint,
                        "timestamp": img.timestamp,
                        "ground_truth_stage": img.ground_truth_stage,
                        "uid": img.uid,
                        "volume_path": img.volume_path,
                    }
                )

            # Get ground truth transitions
            ground_truth = self.dataset.get_ground_truth(session_id, embryo_id)

            # Get predictions if any
            conn = get_connection(self.db_path)
            predictions = conn.execute(
                """
                SELECT timepoint, predicted_stage, confidence, perception_run_id, reasoning
                FROM predictions
                WHERE session_id = ? AND embryo_id = ?
                ORDER BY perception_run_id DESC, timepoint
            """,
                (session_id, embryo_id),
            ).fetchall()
            conn.close()

            return {
                "embryo_id": embryo_id,
                "session_id": session_id,
                "images": images,
                "ground_truth": ground_truth,
                "predictions": [dict(p) for p in predictions],
            }

        # =====================================================================
        # Perception Traces API (from JSON files)
        # =====================================================================

        @app.get("/api/trace/{session_id}/{embryo_id}/{timepoint}")
        async def get_perception_trace(session_id: str, embryo_id: str, timepoint: int):
            """Get perception trace from JSON file."""
            # Look for trace file in traces directory
            traces_dir = self.db_path.parent / "traces" / session_id
            trace_file = traces_dir / f"{embryo_id}_T{timepoint:04d}.json"

            if not trace_file.exists():
                raise HTTPException(status_code=404, detail="Trace not found")

            try:
                trace_data = json.loads(trace_file.read_text(encoding="utf-8"))
                return trace_data
            except Exception as e:
                logger.error(f"Failed to read trace file: {e}")
                raise HTTPException(status_code=500, detail=str(e)) from e

        @app.get("/api/traces/{session_id}/{embryo_id}")
        async def list_perception_traces(session_id: str, embryo_id: str):
            """List available perception traces for an embryo."""
            traces_dir = self.db_path.parent / "traces" / session_id
            if not traces_dir.exists():
                return []

            traces = []
            for trace_file in sorted(traces_dir.glob(f"{embryo_id}_T*.json")):
                # Extract timepoint from filename
                try:
                    tp_str = trace_file.stem.split("_T")[-1]
                    timepoint = int(tp_str)
                    traces.append({"timepoint": timepoint, "file": trace_file.name})
                except (ValueError, IndexError):
                    continue

            return traces

        @app.get("/api/unified_timeline/{embryo_uid}")
        async def get_unified_timeline(embryo_uid: str):
            """Get unified timeline of all images across sessions for an embryo UID."""
            # Get all volumes for this embryo UID
            # Sort by session (using min timestamp per session) then by timepoint within session
            conn = get_connection(self.db_path)
            rows = conn.execute(
                """
                SELECT
                    v.uid,
                    v.session_id,
                    v.embryo_id,
                    v.timepoint,
                    v.timestamp,
                    v.file_path,
                    (SELECT MIN(v2.uid) FROM volumes v2
                     WHERE v2.session_id = v.session_id) as session_min_uid
                FROM volumes v
                WHERE v.embryo_uid = ?
                ORDER BY session_min_uid ASC, v.uid ASC
            """,
                (embryo_uid,),
            ).fetchall()

            if not rows:
                conn.close()
                raise HTTPException(status_code=404, detail="Embryo UID not found")

            # Get ground truth for all session/embryo combinations
            # Ground truth start_timepoint/end_timepoint are ROW INDICES, not v.timepoint!
            session_embryos = set((r[1], r[2]) for r in rows)
            gt_maps = {}
            for session_id, embryo_id in session_embryos:
                gt_rows = conn.execute(
                    """
                    SELECT stage, start_timepoint, end_timepoint
                    FROM ground_truth
                    WHERE session_id = ? AND embryo_id = ?
                    ORDER BY start_timepoint
                """,
                    (session_id, embryo_id),
                ).fetchall()
                gt_maps[(session_id, embryo_id)] = gt_rows
            conn.close()

            # Build unified image list with ground truth based on row index
            # within each session/embryo
            images = []
            # keyed by (session_id, embryo_id); value is the row index within that group
            session_embryo_counts: dict[tuple[Any, Any], int] = {}

            for idx, r in enumerate(rows):
                session_id = r[1]
                embryo_id = r[2]
                key = (session_id, embryo_id)

                # Get the row index within this session/embryo
                row_idx = session_embryo_counts.get(key, 0)
                session_embryo_counts[key] = row_idx + 1

                # Look up ground truth using row index (not v.timepoint!)
                gt_stage = None
                for stage, start_tp, end_tp in gt_maps.get(key, []):
                    # start_timepoint and end_timepoint are row indices
                    if row_idx >= start_tp and (end_tp is None or row_idx < end_tp):
                        gt_stage = stage
                        break

                images.append(
                    {
                        "index": idx,
                        "uid": r[0],
                        "session_id": session_id,
                        "embryo_id": embryo_id,
                        "timepoint": r[3],
                        "timestamp": r[4],
                        "file_path": r[5],
                        "ground_truth_stage": gt_stage,
                    }
                )

            # Get unique sessions
            sessions = list(set(img["session_id"] for img in images))

            return {
                "embryo_uid": embryo_uid,
                "total_images": len(images),
                "sessions": sessions,
                "images": images,
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
                    projections.append(
                        {
                            "method": method_name,
                            "description": desc,
                            "data": image_to_base64(proj_img),
                        }
                    )
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

        # =====================================================================
        # WebSocket for Presence
        # =====================================================================

        @app.websocket("/ws")
        async def websocket_endpoint(websocket: WebSocket):
            """WebSocket endpoint for presence tracking."""
            await self.manager.connect(websocket)
            try:
                while True:
                    data = await websocket.receive_json()
                    msg_type = data.get("type")

                    if msg_type == "join":
                        # Client joining with ID and name
                        client_id = data.get("client_id")
                        name = data.get("name")
                        # Update client info
                        async with self.manager._lock:
                            if websocket in self.manager.active_connections:
                                old_info = self.manager.active_connections[websocket]
                                self.manager.active_connections[websocket] = ClientInfo(
                                    client_id=client_id or old_info.client_id,
                                    name=name or old_info.name,
                                    color=self.manager._generate_color(
                                        client_id or old_info.client_id
                                    ),
                                    connected_at=old_info.connected_at,
                                )
                        await self.manager.broadcast_presence()

                    elif msg_type == "update_name":
                        # Client updating their display name
                        name = data.get("name")
                        if name:
                            await self.manager.update_client_name(websocket, name)

                    elif msg_type == "get_presence":
                        # Client requesting current presence list
                        await self.manager.broadcast_presence()

            except WebSocketDisconnect:
                await self.manager.disconnect(websocket)
            except Exception as e:
                logger.warning(f"WebSocket error: {e}")
                await self.manager.disconnect(websocket)

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

        logger.info("=== Embryo Dataset Explorer ===")
        logger.info("Database: %s", self.db_path)
        logger.info("Open http://localhost:%d in your browser", self.port)
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
