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
from typing import Any, Dict, List, Optional

import numpy as np

from gently.core.service import Service
from gently.settings import settings

logger = logging.getLogger(__name__)

# Optional imports
try:
    from fastapi import FastAPI
    from fastapi.staticfiles import StaticFiles
    from fastapi.templating import Jinja2Templates
    from fastapi.middleware.cors import CORSMiddleware
    import uvicorn
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
            logger.debug("uvicorn dropped non-HTTP bytes on viz port "
                         "(probable TLS/peer mismatch)")
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
from .models import (
    ClientInfo, Volume3DData, ImageData, EmbryoImageCache,
    CALIBRATION_TYPES, VOLUME_TYPES, ANALYSIS_TYPES, VOLUME_3D_TYPES,
)
from .image_store import ImageStore
from .timelapse_tracker import TimelapseStateTracker
from .connection_manager import ConnectionManager

# Re-export for backward compatibility
__all__ = [
    'VisualizationServer', 'create_visualization_server',
    'ClientInfo', 'Volume3DData', 'ImageData', 'EmbryoImageCache',
    'CALIBRATION_TYPES', 'VOLUME_TYPES', 'ANALYSIS_TYPES', 'VOLUME_3D_TYPES',
    'ImageStore',
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
        ssl_certfile: str = None,
        ssl_keyfile: str = None,
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
            version="2.0.0"
        )

        # Setup templates and static files
        self.templates = Jinja2Templates(directory=str(TEMPLATES_DIR))
        self.app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

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

    def _resolve_volume_path(self, embryo_id: str, timepoint: int) -> Optional[str]:
        """Resolve volume file path from timelapse tracker or FileStore."""
        # 1. Try timelapse tracker (in-memory, fastest)
        if embryo_id in self.timelapse_tracker.volume_paths:
            path = self.timelapse_tracker.volume_paths[embryo_id].get(timepoint)
            if path:
                return path

        # 2. Try FileStore (file-based, persistent)
        if self.gently_store and self.timelapse_tracker.session_id:
            try:
                vol_path = self.gently_store.get_volume_path(
                    self.timelapse_tracker.session_id, embryo_id, timepoint,
                )
                if vol_path and vol_path.exists():
                    return str(vol_path)
            except Exception as e:
                logger.debug(f"FileStore volume path lookup failed: {e}")

        return None

    def _resolve_projection_path(self, embryo_id: str, timepoint: int) -> Optional[Path]:
        """Resolve projection file path from FileStore."""
        if self.gently_store and self.timelapse_tracker.session_id:
            try:
                proj_path = self.gently_store.get_projection_path(
                    self.timelapse_tracker.session_id, embryo_id, timepoint,
                )
                if proj_path and proj_path.exists():
                    return proj_path
            except Exception as e:
                logger.debug(f"FileStore projection path lookup failed: {e}")
        return None

    def _subscribe_to_events(self):
        """Subscribe to EventBus for automatic updates - broadcasts ALL events"""

        # Initialize timelapse tracker from event history
        # This catches SESSION_STARTED/SESSION_RESTORED that happened before we subscribed
        self._init_from_event_history()

        async def on_event_async(event):
            """Async handler for all events - broadcasts to WebSocket clients"""
            event_type_str = event.event_type.name if hasattr(event.event_type, 'name') else str(event.event_type)

            # Update timelapse state tracker
            self.timelapse_tracker.handle_event(event_type_str, event.data)

            # Broadcast to all clients
            await self.manager.send_event(
                event_type=event_type_str,
                data=event.data,
                source=event.source,
                event_id=event.event_id
            )

            # For session events, also broadcast updated timelapse_state so clients can sync
            if event_type_str in ("SESSION_STARTED", "SESSION_RESTORED"):
                await self.manager.broadcast({
                    "type": "timelapse_state",
                    "data": self.timelapse_tracker.to_dict()
                })

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
                event_type_str = event.event_type.name if hasattr(event.event_type, 'name') else str(event.event_type)
                self.timelapse_tracker.handle_event(event_type_str, event.data)

            if self.timelapse_tracker.session_id:
                logger.info(f"Initialized timelapse state from history: session={self.timelapse_tracker.session_id}, status={self.timelapse_tracker.status}")
        except Exception as e:
            logger.warning(f"Failed to initialize from event history: {e}")

    def _array_to_image_data(
        self,
        array: np.ndarray,
        uid: str,
        data_type: str,
        metadata: Optional[Dict] = None
    ) -> ImageData:
        """Convert numpy array to ImageData with base64 PNG"""
        from gently.core.imaging import (
            projection_three_view,
            compute_crop_bounds,
            apply_crop_bounds,
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
                z_depth, height, width = array.shape
                # Handle dual-view format (width > 2*height)
                if width > height * 2:
                    array = array[:, :, :width // 2]
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
            img.save(buffer, format='PNG')
            base64_png = base64.b64encode(buffer.getvalue()).decode('utf-8')

        return ImageData(
            uid=uid,
            data_type=data_type,
            timestamp=datetime.now().isoformat(),
            metadata=metadata or {},
            base64_png=base64_png,
            shape=array.shape
        )

    async def push_image(
        self,
        array: np.ndarray,
        uid: str,
        data_type: str = "image",
        metadata: Optional[Dict] = None,
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

        logger.debug(f"Pushed image {uid} ({data_type}) to {len(self.manager.active_connections)} clients")

    async def start_marking_session(
        self,
        image: np.ndarray,
        initial_stage_position: tuple = (0.0, 0.0),
        pixel_size_um: float = 0.65,
    ) -> str:
        """
        Start an embryo marking session in the web UI.

        Broadcasts the bottom camera image to all clients and waits
        for the user to mark embryo positions via clicks.

        Parameters
        ----------
        image : np.ndarray
            Bottom camera overview image (2D grayscale or RGB)
        initial_stage_position : tuple
            Initial XY stage position in micrometers
        pixel_size_um : float
            Pixel size in micrometers/pixel

        Returns
        -------
        str
            Session ID for this marking session
        """
        import uuid

        if not hasattr(self, '_marking_sessions'):
            self._marking_sessions = {}

        session_id = str(uuid.uuid4())[:8]
        self._marking_sessions[session_id] = {
            "markers": [],
            "complete": asyncio.Event(),
            "initial_stage_position": initial_stage_position,
            "pixel_size_um": pixel_size_um,
            "image_shape": image.shape,
        }

        # Encode image as base64 PNG
        from PIL import Image as PILImage
        img = image
        if img.dtype != np.uint8:
            img = ((img - img.min()) / max(img.max() - img.min(), 1) * 255).astype(np.uint8)
        pil_img = PILImage.fromarray(img)
        buf = io.BytesIO()
        pil_img.save(buf, format='PNG')
        b64 = base64.b64encode(buf.getvalue()).decode('ascii')

        h, w = image.shape[:2]

        # Broadcast to all clients
        await self.manager.broadcast({
            "type": "marking_image",
            "data": {
                "session_id": session_id,
                "image_b64": b64,
                "width": w,
                "height": h,
            }
        })

        logger.info(f"Marking session {session_id} started, image {w}x{h} sent to {len(self.manager.active_connections)} clients")
        return session_id

    async def wait_for_marking(self, session_id: str, timeout: float = None) -> list:
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
        pixel_size = session["pixel_size_um"]
        h, w = session["image_shape"][:2]
        center_x, center_y = w / 2, h / 2

        # Convert to embryo entries compatible with EmbryoMarker format
        embryos = []
        for m in markers:
            px, py = m["pixelX"], m["pixelY"]
            embryos.append({
                "embryo_number": m["number"],
                "embryo_id": f"embryo_{m['number']:03d}",
                "pixel_position": (px, py),
                "initial_stage_position": initial_pos,
                "marking_timestamp": m.get("timestamp", datetime.now().isoformat()),
            })

        # Clean up
        del self._marking_sessions[session_id]

        return embryos

    async def push_volume_3d(
        self,
        volume: np.ndarray,
        masks: np.ndarray,
        uid: str,
        metadata: Optional[Dict] = None,
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
            data_type='segmentation_3d',
            timestamp=datetime.now().isoformat(),
            volume=volume,
            masks=masks,
            colors=colors,
            metadata=metadata or {}
        )

        # Store the 3D volume
        self.store.add_volume_3d(volume_data)

        # Broadcast notification to clients (without the heavy data)
        await self.manager.broadcast({
            'type': 'volume_3d',
            'data': volume_data.to_info_dict()
        })

        logger.info(f"Pushed 3D volume {uid} ({volume.shape}) to {len(self.manager.active_connections)} clients")

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
        try:
            sock.bind((self.host, self.port))
        except OSError:
            raise OSError(
                f"Port {self.port} is already in use. "
                "Is another instance of the agent running? "
                "Close it first and try again."
            )
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

    async def health_check(self) -> Dict:
        """Return health status with connected client count."""
        base = await super().health_check()
        base['connected_clients'] = len(self.manager.active_connections)
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
        if hasattr(signal, 'SIGINT'):
            try:
                loop.add_signal_handler(signal.SIGINT, signal_handler)
                signals_installed = True
            except NotImplementedError:
                pass

        if hasattr(signal, 'SIGTERM'):
            try:
                loop.add_signal_handler(signal.SIGTERM, signal_handler)
            except NotImplementedError:
                pass

        if sys.platform == 'win32' and not signals_installed:
            signal.signal(signal.SIGINT, signal_handler)
            signal.signal(signal.SIGTERM, signal_handler)

        await self.start()

        logger.info(f"Server running at http://{self.host}:{self.port} - Press Ctrl+C to stop")

        try:
            if sys.platform == 'win32':
                while not stop_event.is_set():
                    try:
                        await asyncio.wait_for(
                            asyncio.shield(self._server_task),
                            timeout=0.5
                        )
                        break
                    except asyncio.TimeoutError:
                        continue
            else:
                stop_task = asyncio.create_task(stop_event.wait())
                done, pending = await asyncio.wait(
                    [self._server_task, stop_task],
                    return_when=asyncio.FIRST_COMPLETED
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
