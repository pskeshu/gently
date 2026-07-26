"""
Visualization Server for Gently Microscopy System
==================================================

Web-based visualization server providing:
- REST API for image retrieval
- WebSocket streaming for real-time updates
- Tabbed interface: Main, Volumes, Calibration
- Embryo-specific filtering
- Calibration gallery view
- Integration with EventBus for automatic notifications

Usage:
    from gently.visualization import VisualizationServer

    server = VisualizationServer(port=8080)
    await server.start()
"""

import asyncio
import base64
import io
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

from gently.core.service import Service
from gently.settings import settings

logger = logging.getLogger(__name__)

# Optional imports
try:
    import uvicorn
    from fastapi import FastAPI
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.staticfiles import StaticFiles
    from fastapi.templating import Jinja2Templates

    FASTAPI_AVAILABLE = True
except ImportError:
    FASTAPI_AVAILABLE = False
    logger.warning("FastAPI not available. Install with: pip install fastapi uvicorn")


class _InvalidHttpFilter(logging.Filter):
    """Suppress uvicorn's "Invalid HTTP request received." warning.

    Fires when something speaks non-HTTP at the viz port — typically a
    mesh peer with a TLS/plain mismatch (one side advertised
    tls_enabled=True while the other is plain http), or an opportunistic
    network scanner. uvicorn rejects the connection cleanly; the warning
    is cosmetic. We downgrade it to a single DEBUG line so the log stays
    readable but the diagnostic info is still reachable with -v.
    """

    def filter(self, record: logging.LogRecord) -> bool:
        msg = record.getMessage()
        if "Invalid HTTP request received" in msg:
            logger.debug("uvicorn dropped non-HTTP bytes on viz port (probable TLS/peer mismatch)")
            return False
        return True


if FASTAPI_AVAILABLE:
    logging.getLogger("uvicorn.error").addFilter(_InvalidHttpFilter())

# Web asset paths (templates and static files live alongside this module)
_WEB_DIR = Path(__file__).parent
TEMPLATES_DIR = _WEB_DIR / "templates"
STATIC_DIR = _WEB_DIR / "static"

try:
    from PIL import Image

    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False

# Import data models and components
from .connection_manager import ConnectionManager  # noqa: E402
from .image_store import ImageStore  # noqa: E402
from .models import (  # noqa: E402
    ANALYSIS_TYPES,
    CALIBRATION_TYPES,
    VOLUME_3D_TYPES,
    VOLUME_TYPES,
    ClientInfo,
    EmbryoImageCache,
    ImageData,
    Volume3DData,
)
from .timelapse_tracker import TimelapseStateTracker  # noqa: E402

# Re-export for backward compatibility
__all__ = [
    "VisualizationServer",
    "create_visualization_server",
    "ClientInfo",
    "Volume3DData",
    "ImageData",
    "EmbryoImageCache",
    "CALIBRATION_TYPES",
    "VOLUME_TYPES",
    "ANALYSIS_TYPES",
    "VOLUME_3D_TYPES",
    "ImageStore",
]


class VisualizationServer(Service):
    """
    Web-based visualization server for microscopy data

    Features:
    - REST API for image retrieval by UID
    - WebSocket streaming for real-time updates
    - Tabbed interface: Main, Volumes, Calibration
    - Embryo-specific filtering
    - Calibration gallery view
    - EventBus integration for automatic notifications

    Parameters
    ----------
    host : str
        Server host (default: "0.0.0.0")
    port : int
        Server port (default: 8080)
    data_store : DataStore, optional
        Data store for retrieving images by UID
    event_bus : EventBus, optional
        Event bus for subscribing to updates
    """

    def __init__(
        self,
        host: str = settings.network.viz_host,
        port: int = settings.network.viz_port,
        data_store=None,
        event_bus=None,
        sessions_dir: str = str(settings.storage.sessions_dir),
        gently_store=None,
        ssl_certfile: str | None = None,
        ssl_keyfile: str | None = None,
    ):
        super().__init__(name="visualization", service_type="http", host=host, port=port)
        if not FASTAPI_AVAILABLE:
            raise ImportError(
                "FastAPI is required for visualization server. "
                "Install with: pip install fastapi uvicorn"
            )

        self.data_store = data_store
        self.event_bus = event_bus
        self._ssl_certfile = ssl_certfile
        self._ssl_keyfile = ssl_keyfile
        self.sessions_dir = Path(sessions_dir)
        self.gently_store = gently_store  # FileStore for persistent volume/projection access
        self.context_store = None  # FileContextStore — set via set_context_store()
        # Wired in by launch_gently after construction (optional subsystems).
        self.agent_bridge: Any = None
        self.mesh_service: Any = None
        self.device_supervisor: Any = None  # DeviceLayerSupervisor (RFC #78)
        # Callable that stops the WHOLE backend (launcher keep-alive included);
        # POST /api/shutdown uses it for the desktop shell handshake (issue #85).
        self.request_shutdown: Any = None
        # False until the launch gate is submitted this session; while False, /
        # bounces to /launch so the gate is the entry point (RFC #78).
        self.gate_passed: bool = False

        # Connection manager for WebSocket clients
        self.manager = ConnectionManager()

        # Organized image storage (unlimited)
        self.store = ImageStore()

        # Timelapse state tracker for client sync
        self.timelapse_tracker = TimelapseStateTracker()

        # Create FastAPI app
        self.app = FastAPI(
            title="Gently Visualization Server",
            description="Real-time microscopy visualization",
            version="2.0.0",
        )

        # Setup templates and static files
        self.templates = Jinja2Templates(directory=str(TEMPLATES_DIR))
        self.app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

        # Static assets are served live (CLAUDE.md: "refresh the window — served
        # live by Python, no rebuild"). Default StaticFiles sends ETag +
        # Last-Modified but no Cache-Control, so browsers apply *heuristic*
        # freshness and can serve a stale .js/.css after an edit — breaking that
        # promise. Force revalidation on every load: with the ETag still present
        # an unchanged file returns a cheap 304, a changed one returns fresh bytes.
        @self.app.middleware("http")
        async def _revalidate_static(request, call_next):
            response = await call_next(request)
            if request.url.path.startswith("/static"):
                response.headers["Cache-Control"] = "no-cache"
            return response

        # Add CORS middleware
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )

        # Register route groups
        from .routes import register_all_routes

        register_all_routes(self)

        # Subscribe to events if event bus provided
        if self.event_bus:
            self._subscribe_to_events()

        # Server instance
        self._server = None
        self._server_task = None

    def set_context_store(self, context_store) -> None:
        """Set the FileContextStore for campaign/plan data access."""
        self.context_store = context_store

    def _resolve_volume_path(self, embryo_id: str, timepoint: int) -> str | None:
        """Resolve volume file path from timelapse tracker or FileStore."""
        # 1. Try timelapse tracker (in-memory, fastest)
        if embryo_id in self.timelapse_tracker.volume_paths:
            path = self.timelapse_tracker.volume_paths[embryo_id].get(timepoint)
            if path:
                return path

        # 2. Try FileStore (file-based, persistent). Key on the LIVE agent
        # session, not the tracker's (which goes stale after a resume with no
        # active timelapse) — mirrors _resolve_projection_path so an agent-driven
        # open_volume hand-off doesn't 404 after a /resume.
        sid = self._current_session_id()
        if self.gently_store and sid:
            try:
                vol_path = self.gently_store.get_volume_path(
                    sid,
                    embryo_id,
                    timepoint,
                )
                if vol_path and vol_path.exists():
                    return str(vol_path)
            except Exception as e:
                logger.debug(f"FileStore volume path lookup failed: {e}")

        return None

    def _current_session_id(self) -> str | None:
        """The live agent session (source of truth), falling back to the
        timelapse tracker. The tracker's session_id goes stale after a resume
        with no active timelapse, so the live agent session is preferred."""
        bridge = getattr(self, "agent_bridge", None)
        if bridge is not None and getattr(bridge, "agent", None) is not None:
            sid = getattr(bridge.agent, "session_id", None)
            if sid:
                return sid
        return self.timelapse_tracker.session_id

    def _resolve_projection_path(self, embryo_id: str, timepoint: int) -> Path | None:
        """Resolve projection file path from FileStore (current session)."""
        sid = self._current_session_id()
        if self.gently_store and sid:
            try:
                proj_path = self.gently_store.get_projection_path(
                    sid,
                    embryo_id,
                    timepoint,
                )
                if proj_path and proj_path.exists():
                    return proj_path
            except Exception as e:
                logger.debug(f"FileStore projection path lookup failed: {e}")
        return None

    def rehydrate_session(self, session_id: str) -> int:
        """Repopulate the in-memory image store with the FileStore's persisted
        projections for a (resumed) session, so galleries and filmstrips show
        its historical data.

        Lightweight: only metadata-bearing ImageData entries are created (uid
        ``volume_{embryo}_t{NNNN}``); the JPEG pixels load lazily on demand via
        /api/images/{uid}/png (which falls back to the FileStore projection).
        Resets the store first so the previous session's images don't linger.
        Returns the number of projection entries added.
        """
        if self.gently_store is None or not session_id:
            return 0
        self.store = ImageStore()  # drop the previous session's images
        added = 0
        try:
            embryos = self.gently_store.list_embryos(session_id) or []
        except Exception:
            embryos = []
        for emb in embryos:
            eid = emb.get("embryo_id") if isinstance(emb, dict) else getattr(emb, "embryo_id", None)
            if not eid:
                continue
            try:
                tps = self.gently_store.list_projection_timepoints(session_id, eid)
            except Exception:
                tps = []
            for tp in tps:
                self.store.add_image(
                    ImageData(
                        uid=f"volume_{eid}_t{tp:04d}",
                        data_type="volume_projection",
                        timestamp=f"{tp:06d}",  # monotonic with timepoint for ordering
                        metadata={"embryo_id": eid, "timepoint": tp},
                    )
                )
                added += 1

        # Rehydrate the timelapse tracker's per-embryo perception state from
        # predictions.jsonl so the Default / Film / reasoning views populate
        # (those are driven by detection_reasoning, not the raw image store).
        # Thumbnails resolve via the projection uids added above.
        tracker = self.timelapse_tracker
        try:
            tracker.session_id = session_id
            tracker.detection_reasoning = {}
            tracker.projection_uids = {}
            for emb in embryos:
                eid = (
                    emb.get("embryo_id")
                    if isinstance(emb, dict)
                    else getattr(emb, "embryo_id", None)
                )
                if not eid:
                    continue
                try:
                    preds = self.gently_store.get_predictions(session_id, eid) or []
                except Exception:
                    preds = []
                if not preds:
                    continue
                items, puids, last_stage = [], {}, None
                for p in preds:
                    tp = p.get("timepoint")
                    if tp is None:
                        continue
                    uid = f"volume_{eid}_t{tp:04d}"
                    puids[tp] = uid
                    stage = p.get("predicted_stage")
                    last_stage = stage or last_stage
                    items.append(
                        {
                            "timepoint": tp,
                            "stage": stage,
                            "detected_stage": stage,
                            "reasoning": p.get("reasoning"),
                            "confidence": p.get("confidence"),
                            "projection_uid": uid,
                            "image_uid": uid,
                            "detector_name": "perception",
                        }
                    )
                tracker.detection_reasoning[eid] = items
                tracker.projection_uids[eid] = puids
                entry = tracker.embryos.setdefault(
                    eid,
                    {
                        "embryo_id": eid,
                        "timepoints": 0,
                        "is_complete": False,
                        "detections": {},
                        "current_stage": None,
                    },
                )
                entry["timepoints"] = max((it["timepoint"] for it in items), default=0)
                entry["current_stage"] = last_stage
            tracker.total_timepoints = sum(len(v) for v in tracker.detection_reasoning.values())
        except Exception:
            logger.exception("Tracker perception rehydration failed")

        logger.info("Rehydrated %d projections for session %s", added, session_id)
        return added

    def _subscribe_to_events(self):
        """Subscribe to EventBus for automatic updates - broadcasts ALL events"""

        # Initialize timelapse tracker from event history
        # This catches SESSION_STARTED/SESSION_RESTORED that happened before we subscribed
        self._init_from_event_history()

        async def on_event_async(event):
            """Async handler for all events - broadcasts to WebSocket clients"""
            event_type_str = (
                event.event_type.name
                if hasattr(event.event_type, "name")
                else str(event.event_type)
            )

            # Update timelapse state tracker
            self.timelapse_tracker.handle_event(event_type_str, event.data)

            # Broadcast to all clients
            await self.manager.send_event(
                event_type=event_type_str,
                data=event.data,
                source=event.source,
                event_id=event.event_id,
            )

            # For session events, also broadcast updated timelapse_state so clients can sync
            if event_type_str in ("SESSION_STARTED", "SESSION_RESTORED"):
                await self.manager.broadcast(
                    {
                        "type": "timelapse_state",
                        "data": self.timelapse_tracker.to_dict(),
                    }
                )

        # Subscribe to ALL events using wildcard with async handler
        self.event_bus.subscribe_async("*", on_event_async)

        logger.info("Subscribed to ALL event types via wildcard")

    def _init_from_event_history(self):
        """Initialize timelapse tracker state from event bus history"""
        if not self.event_bus:
            return

        try:
            # Get recent history and replay relevant events to build current state
            history = self.event_bus.get_history(limit=500)

            # Process events in chronological order (history is newest-first)
            for event in reversed(history):
                event_type_str = (
                    event.event_type.name
                    if hasattr(event.event_type, "name")
                    else str(event.event_type)
                )
                self.timelapse_tracker.handle_event(event_type_str, event.data)

            if self.timelapse_tracker.session_id:
                logger.info(
                    f"Initialized timelapse state from history:"
                    f" session={self.timelapse_tracker.session_id},"
                    f" status={self.timelapse_tracker.status}"
                )
        except Exception as e:
            logger.warning(f"Failed to initialize from event history: {e}")

    def _array_to_image_data(
        self,
        array: np.ndarray,
        uid: str,
        data_type: str,
        metadata: dict | None = None,
    ) -> ImageData:
        """Convert numpy array to ImageData with base64 PNG"""
        from gently.core.imaging import (
            apply_crop_bounds,
            compute_crop_bounds,
            projection_three_view,
        )

        # Handle 4D arrays (Views, Z, Y, X) - select View A only
        if array.ndim == 4:
            array = array[0]  # View A

        # Handle 3D arrays - distinguish RGB images from volumes
        if array.ndim == 3:
            if array.shape[2] in (3, 4):
                # It's an RGB(A) image
                pass
            else:
                # It's a volume (Z, H, W) - generate three-view projection
                # View selection already happened via the 4D branch above; a 3D
                # array is one view. Do not split by aspect ratio (2048x512
                # native frames would be halved).
                # Auto-crop to embryo region
                bounds = compute_crop_bounds(array)
                array = apply_crop_bounds(array, bounds)
                # Generate three-view projection
                array, _ = projection_three_view(array)

        # Normalize to 0-255 (skip for RGB uint8 images)
        if array.dtype != np.uint8:
            arr_min, arr_max = array.min(), array.max()
            if arr_max > arr_min:
                array = ((array - arr_min) / (arr_max - arr_min) * 255).astype(np.uint8)
            else:
                array = np.zeros_like(array, dtype=np.uint8)

        # Convert to PNG base64
        base64_png = None
        if PIL_AVAILABLE:
            img = Image.fromarray(array)
            buffer = io.BytesIO()
            img.save(buffer, format="PNG")
            base64_png = base64.b64encode(buffer.getvalue()).decode("utf-8")

        return ImageData(
            uid=uid,
            data_type=data_type,
            timestamp=datetime.now().isoformat(),
            metadata=metadata or {},
            base64_png=base64_png,
            shape=array.shape,
        )

    async def push_image(
        self,
        array: np.ndarray,
        uid: str,
        data_type: str = "image",
        metadata: dict | None = None,
    ):
        """
        Push an image to connected clients

        Parameters
        ----------
        array : np.ndarray
            Image array (2D or 3D)
        uid : str
            Unique identifier
        data_type : str
            Type of image (volume_projection, focus_sweep, etc.)
        metadata : dict, optional
            Additional metadata (should include embryo_id for organization)
        """
        image_data = self._array_to_image_data(array, uid, data_type, metadata)

        # Add to organized storage
        self.store.add_image(image_data)

        # Broadcast to clients
        await self.manager.send_image(image_data)

        logger.debug(
            f"Pushed image {uid} ({data_type}) to {len(self.manager.active_connections)} clients"
        )

    async def start_marking_session(
        self,
        image: np.ndarray,
        initial_stage_position: tuple = (0.0, 0.0),
        pixel_size_um: float = 0.65,
        initial_markers: list | None = None,
        default_role: str = "test",
    ) -> str:
        """
        Start an embryo marking session in the web UI (map view).

        Broadcasts the bottom camera image to all clients and waits
        for the user to mark/edit embryo positions via clicks. Each
        marker carries a role (test / calibration / unassigned).

        Parameters
        ----------
        image : np.ndarray
            Bottom camera overview image (2D grayscale or RGB)
        initial_stage_position : tuple
            Initial XY stage position in micrometers
        pixel_size_um : float
            Pixel size in micrometers/pixel
        initial_markers : list of dict, optional
            Pre-populate the map view with editable markers (e.g. from
            SAM auto-detection). Each entry should have ``pixel_x``,
            ``pixel_y``; may also include ``role``, ``source``,
            ``embryo_id``, ``confidence``.
        default_role : str
            Role assigned to markers that don't specify one
            (e.g. user-clicked new markers). Default ``"test"`` matches
            the EmbryoState default — accidentally treating Calibration
            as Test only over-protects.

        Returns
        -------
        str
            Session ID for this marking session
        """
        import uuid

        if not hasattr(self, "_marking_sessions"):
            self._marking_sessions: dict[str, dict[str, Any]] = {}

        session_id = str(uuid.uuid4())[:8]

        # Normalize initial markers to the same shape the frontend uses.
        normalized = []
        for i, m in enumerate(initial_markers or []):
            px = m.get("pixel_x", m.get("pixelX"))
            py = m.get("pixel_y", m.get("pixelY"))
            if px is None or py is None:
                continue
            normalized.append(
                {
                    "number": i + 1,
                    "pixelX": round(float(px), 1),
                    "pixelY": round(float(py), 1),
                    "role": m.get("role", default_role),
                    "source": m.get("source", "sam"),
                    "embryo_id": m.get("embryo_id"),
                    "confidence": m.get("confidence"),
                }
            )

        self._marking_sessions[session_id] = {
            "markers": list(normalized),
            "complete": asyncio.Event(),
            "initial_stage_position": initial_stage_position,
            "pixel_size_um": pixel_size_um,
            "image_shape": image.shape,
            "default_role": default_role,
        }

        # Encode image as base64 PNG
        from PIL import Image as PILImage

        img = image
        if img.dtype != np.uint8:
            img = ((img - img.min()) / max(img.max() - img.min(), 1) * 255).astype(np.uint8)
        pil_img = PILImage.fromarray(img)
        buf = io.BytesIO()
        pil_img.save(buf, format="PNG")
        b64 = base64.b64encode(buf.getvalue()).decode("ascii")

        h, w = image.shape[:2]

        # Broadcast to all clients
        await self.manager.broadcast(
            {
                "type": "marking_image",
                "data": {
                    "session_id": session_id,
                    "image_b64": b64,
                    "width": w,
                    "height": h,
                    "initial_markers": normalized,
                    "default_role": default_role,
                    "stage_x_um": float(initial_stage_position[0]),
                    "stage_y_um": float(initial_stage_position[1]),
                    "pixel_size_um": pixel_size_um,
                },
            }
        )

        logger.info(
            f"Marking session {session_id} started, image {w}x{h}, "
            f"{len(normalized)} initial markers, "
            f"{len(self.manager.active_connections)} clients"
        )
        return session_id

    async def wait_for_marking(self, session_id: str, timeout: float | None = None) -> list:
        """
        Wait for a marking session to complete.

        Parameters
        ----------
        session_id : str
            Session ID from start_marking_session
        timeout : float, optional
            Timeout in seconds (None = wait forever)

        Returns
        -------
        list of dict
            Marked embryos with pixel positions and stage positions
        """
        session = self._marking_sessions.get(session_id)
        if not session:
            return []

        try:
            await asyncio.wait_for(session["complete"].wait(), timeout=timeout)
        except asyncio.TimeoutError:
            logger.warning(f"Marking session {session_id} timed out")

        markers = session["markers"]
        initial_pos = session["initial_stage_position"]
        session["pixel_size_um"]
        h, w = session["image_shape"][:2]
        _center_x, _center_y = w / 2, h / 2

        # Convert to embryo entries. Carries role + source so callers can
        # register each embryo with the right experimental classification.
        default_role = session.get("default_role", "test")
        embryos = []
        for m in markers:
            px, py = m["pixelX"], m["pixelY"]
            embryos.append(
                {
                    "embryo_number": m["number"],
                    # Unpadded to match the live convention used everywhere else
                    # (detection_tools registers embryos as f"embryo_{n}"). A
                    # zero-padded fallback here produced ids like "embryo_002"
                    # that never matched the stored "embryo_2".
                    "embryo_id": m.get("embryo_id") or f"embryo_{m['number']}",
                    "pixel_position": (px, py),
                    "pixel_x": px,
                    "pixel_y": py,
                    "initial_stage_position": initial_pos,
                    "role": m.get("role", default_role),
                    "source": m.get("source", "manual"),
                    "confidence": m.get("confidence"),
                    "marking_timestamp": m.get("timestamp", datetime.now().isoformat()),
                }
            )

        # Clean up
        del self._marking_sessions[session_id]

        return embryos

    async def push_volume_3d(
        self,
        volume: np.ndarray,
        masks: np.ndarray,
        uid: str,
        metadata: dict | None = None,
    ):
        """
        Push a 3D segmentation volume to connected clients

        Parameters
        ----------
        volume : np.ndarray
            Original volume (Z, H, W)
        masks : np.ndarray
            Segmentation masks (Z, H, W) with integer labels
        uid : str
            Unique identifier
        metadata : dict, optional
            Additional metadata
        """
        # Generate consistent colors for all cells
        np.random.seed(42)
        num_labels = int(masks.max()) + 1
        colors = np.random.randint(100, 255, size=(num_labels, 3), dtype=np.uint8)
        colors[0] = [0, 0, 0]  # Background is black

        volume_data = Volume3DData(
            uid=uid,
            data_type="segmentation_3d",
            timestamp=datetime.now().isoformat(),
            volume=volume,
            masks=masks,
            colors=colors,
            metadata=metadata or {},
        )

        # Store the 3D volume
        self.store.add_volume_3d(volume_data)

        # Broadcast notification to clients (without the heavy data)
        await self.manager.broadcast({"type": "volume_3d", "data": volume_data.to_info_dict()})

        logger.info(
            f"Pushed 3D volume {uid} ({volume.shape}) to"
            f" {len(self.manager.active_connections)} clients"
        )

    async def open_volume_in_browser(
        self,
        embryo_id: str,
        timepoint: int,
        view: str = "3d_viewer",
    ) -> int:
        """Ask every connected browser to open the in-browser volume viewer.

        This is the web-native replacement for the old napari ``view_volume``:
        the agent triggers the existing ProjectionViewer (WebGL raymarcher +
        projections) instead of launching a desktop Qt window that would block
        the shared agent/web event loop. Returns the number of clients notified.
        """
        await self.manager.broadcast(
            {
                "type": "open_volume",
                "embryo_id": embryo_id,
                "timepoint": timepoint,
                "view": view,
            }
        )
        n = len(self.manager.active_connections)
        logger.info(
            "Requested browser open_volume for %s t%s (%d client(s))",
            embryo_id,
            timepoint,
            n,
        )
        return n

    async def on_start(self):
        """Start the visualization server"""
        # Set the event loop on the event bus so async handlers work
        # even when events are published from sync code
        if self.event_bus:
            self.event_bus.set_event_loop(asyncio.get_running_loop())

        # Pre-flight check: verify the port is available before handing
        # off to uvicorn (whose bind error surfaces inside a background
        # task and produces an unhelpful log line).
        import socket

        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        # Match uvicorn's own bind semantics. uvicorn sets SO_REUSEADDR before it
        # binds, so a bare preflight bind WITHOUT it is *stricter* than the real
        # server: when a previous instance has just exited, its browser/websocket
        # connections linger in TIME_WAIT holding this local port, and a plain
        # bind() fails with EADDRINUSE even though uvicorn would bind fine. That
        # false positive was the recurring "port in use" on quick restarts. With
        # SO_REUSEADDR the preflight now fails only on a genuine live listener
        # (a real second instance) — exactly when uvicorn would also fail.
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            sock.bind((self.host, self.port))
        except OSError:
            raise OSError(
                f"Port {self.port} is already in use — another instance may be running. "
                f"Free it with:  fuser -k {self.port}/tcp  "
                f"(or: lsof -ti:{self.port} | xargs -r kill), then try again."
            ) from None
        finally:
            sock.close()

        # On Windows, suppress noisy ConnectionResetError from the
        # ProactorEventLoop when remote connections drop abruptly.
        if sys.platform == "win32":
            loop = asyncio.get_running_loop()
            _original_handler = loop.get_exception_handler()

            def _quiet_connection_reset(loop, context):
                exc = context.get("exception")
                if isinstance(exc, ConnectionResetError):
                    return  # suppress — these are harmless on Windows
                if _original_handler:
                    _original_handler(loop, context)
                else:
                    loop.default_exception_handler(context)

            loop.set_exception_handler(_quiet_connection_reset)

        config_kwargs = dict(
            host=self.host,
            port=self.port,
            log_level="warning",
        )
        if self._ssl_certfile and self._ssl_keyfile:
            config_kwargs["ssl_certfile"] = self._ssl_certfile
            config_kwargs["ssl_keyfile"] = self._ssl_keyfile

        config = uvicorn.Config(self.app, **config_kwargs)
        self._server = uvicorn.Server(config)

        scheme = "https" if self._ssl_certfile else "http"
        logger.info(f"Starting visualization server on {scheme}://{self.host}:{self.port}")

        # Run server in background task
        self._server_task = asyncio.create_task(self._server.serve())

        # Wait a moment for server to start
        await asyncio.sleep(0.5)

        logger.info(f"Visualization server running at {scheme}://{self.host}:{self.port}")

    async def on_stop(self):
        """Stop the visualization server"""
        if self._server:
            self._server.should_exit = True
            if self._server_task:
                try:
                    await asyncio.wait_for(self._server_task, timeout=5.0)
                except asyncio.TimeoutError:
                    logger.warning("Server task did not complete in time, cancelling")
                    self._server_task.cancel()
                    try:
                        await self._server_task
                    except asyncio.CancelledError:
                        pass
            self._server = None
            self._server_task = None
        logger.info("Visualization server stopped")

    async def health_check(self) -> dict:
        """Return health status with connected client count."""
        base = await super().health_check()
        base["connected_clients"] = len(self.manager.active_connections)
        return base

    async def run_forever(self):
        """Run the server until interrupted."""
        import signal
        import sys

        stop_event = asyncio.Event()

        def signal_handler(*args):
            logger.info("Received shutdown signal")
            stop_event.set()

        loop = asyncio.get_running_loop()

        signals_installed = False
        if hasattr(signal, "SIGINT"):
            try:
                loop.add_signal_handler(signal.SIGINT, signal_handler)
                signals_installed = True
            except NotImplementedError:
                pass

        if hasattr(signal, "SIGTERM"):
            try:
                loop.add_signal_handler(signal.SIGTERM, signal_handler)
            except NotImplementedError:
                pass

        if sys.platform == "win32" and not signals_installed:
            signal.signal(signal.SIGINT, signal_handler)
            signal.signal(signal.SIGTERM, signal_handler)

        await self.start()

        logger.info(f"Server running at http://{self.host}:{self.port} - Press Ctrl+C to stop")

        try:
            if sys.platform == "win32":
                while not stop_event.is_set():
                    try:
                        await asyncio.wait_for(asyncio.shield(self._server_task), timeout=0.5)
                        break
                    except asyncio.TimeoutError:
                        continue
            else:
                stop_task = asyncio.create_task(stop_event.wait())
                done, pending = await asyncio.wait(
                    [self._server_task, stop_task], return_when=asyncio.FIRST_COMPLETED
                )
                for task in pending:
                    task.cancel()
                    try:
                        await task
                    except asyncio.CancelledError:
                        pass

        except asyncio.CancelledError:
            logger.info("Server cancelled")
        finally:
            await self.stop()


# Convenience function
def create_visualization_server(
    port: int = settings.network.viz_port,
    data_store=None,
    event_bus=None,
    gently_store=None,
) -> VisualizationServer:
    """
    Create a visualization server instance

    Parameters
    ----------
    port : int
        Server port (default: 8080)
    data_store : DataStore, optional
        Data store for image retrieval
    event_bus : EventBus, optional
        Event bus for real-time updates
    gently_store : FileStore, optional
        Unified store for persistent volume/projection access

    Returns
    -------
    VisualizationServer
        Configured server instance
    """
    return VisualizationServer(
        port=port,
        data_store=data_store,
        event_bus=event_bus,
        gently_store=gently_store,
    )
