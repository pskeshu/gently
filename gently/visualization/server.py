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
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Set
from dataclasses import dataclass, field, asdict

import numpy as np

logger = logging.getLogger(__name__)

# Optional imports
try:
    from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException, Request
    from fastapi.responses import HTMLResponse, JSONResponse, Response
    from fastapi.staticfiles import StaticFiles
    from fastapi.templating import Jinja2Templates
    from fastapi.middleware.cors import CORSMiddleware
    import uvicorn
    FASTAPI_AVAILABLE = True
except ImportError:
    FASTAPI_AVAILABLE = False
    logger.warning("FastAPI not available. Install with: pip install fastapi uvicorn")

# Import web asset paths
from .web import TEMPLATES_DIR, STATIC_DIR

try:
    from PIL import Image
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False


# Data types for routing to tabs
CALIBRATION_TYPES = {
    'focus_sweep', 'focus_plot', 'edge_detection', 'calibration_summary',
    'focus_snap', 'focus_coarse', 'focus_curve', 'focus_assess'
}

VOLUME_TYPES = {
    'volume', 'volume_projection', 'z_stack', 'timelapse'
}

# CV/Analysis types - shown in a separate "Analysis" category within Calibration
ANALYSIS_TYPES = {
    'segmentation', 'detection', 'classification', 'tracking',
    # CV agent visualization types
    'roi_detection', 'cropped_roi', 'vision_prepared', 'timeline', 'cv_visualization'
}

# 3D types that support Z-slider browsing
VOLUME_3D_TYPES = {
    'segmentation_3d'
}


@dataclass
class Volume3DData:
    """Container for 3D volume data with segmentation overlay"""
    uid: str
    data_type: str
    timestamp: str
    volume: np.ndarray  # Original volume (Z, H, W)
    masks: np.ndarray   # Segmentation masks (Z, H, W)
    colors: np.ndarray  # Cell colors (num_labels, 3)
    metadata: Dict[str, Any] = field(default_factory=dict)

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

    def to_info_dict(self) -> Dict:
        """Return metadata without the heavy arrays"""
        return {
            'uid': self.uid,
            'data_type': self.data_type,
            'timestamp': self.timestamp,
            'shape': list(self.shape),
            'num_slices': self.num_slices,
            'num_cells': int(self.masks.max()),
            'metadata': self.metadata
        }


@dataclass
class ImageData:
    """Container for image data sent to clients"""
    uid: str
    data_type: str  # 'volume', 'projection', 'snapshot', 'detection', 'focus_sweep', etc.
    timestamp: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    base64_png: Optional[str] = None
    shape: Optional[tuple] = None

    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass
class EmbryoImageCache:
    """Per-embryo image organization"""
    embryo_id: str
    volumes: List[ImageData] = field(default_factory=list)
    calibration: List[ImageData] = field(default_factory=list)
    snapshots: List[ImageData] = field(default_factory=list)


class ImageStore:
    """Organized storage for images by type and embryo"""

    def __init__(self, max_per_category: int = 50):
        self.max_per_category = max_per_category
        self._embryo_caches: Dict[str, EmbryoImageCache] = {}
        self._global_images: List[ImageData] = []  # Images without embryo_id
        self._calibration_images: List[ImageData] = []  # Global calibration
        self._volume_images: List[ImageData] = []  # Global volumes
        self._volumes_3d: Dict[str, Volume3DData] = {}  # 3D volumes by UID

    def _get_embryo_cache(self, embryo_id: str) -> EmbryoImageCache:
        if embryo_id not in self._embryo_caches:
            self._embryo_caches[embryo_id] = EmbryoImageCache(embryo_id=embryo_id)
        return self._embryo_caches[embryo_id]

    def add_image(self, image: ImageData):
        """Add image to appropriate storage based on type and embryo"""
        embryo_id = image.metadata.get('embryo_id')
        data_type = image.data_type

        if data_type in CALIBRATION_TYPES or data_type in ANALYSIS_TYPES:
            # Both calibration and analysis go to calibration tab
            if embryo_id:
                cache = self._get_embryo_cache(embryo_id)
                cache.calibration.append(image)
                self._trim_list(cache.calibration)
            else:
                self._calibration_images.append(image)
                self._trim_list(self._calibration_images)

        elif data_type in VOLUME_TYPES:
            if embryo_id:
                cache = self._get_embryo_cache(embryo_id)
                cache.volumes.append(image)
                self._trim_list(cache.volumes)
            else:
                self._volume_images.append(image)
                self._trim_list(self._volume_images)
        else:
            # General snapshot/other
            if embryo_id:
                cache = self._get_embryo_cache(embryo_id)
                cache.snapshots.append(image)
                self._trim_list(cache.snapshots)
            else:
                self._global_images.append(image)
                self._trim_list(self._global_images)

    def _trim_list(self, lst: List):
        while len(lst) > self.max_per_category:
            lst.pop(0)

    def get_all_calibration(self, embryo_id: Optional[str] = None) -> List[ImageData]:
        """Get calibration images, optionally filtered by embryo"""
        if embryo_id:
            cache = self._embryo_caches.get(embryo_id)
            return cache.calibration if cache else []
        # Return all calibration images
        all_cal = list(self._calibration_images)
        for cache in self._embryo_caches.values():
            all_cal.extend(cache.calibration)
        return sorted(all_cal, key=lambda x: x.timestamp)

    def get_all_volumes(self, embryo_id: Optional[str] = None) -> List[ImageData]:
        """Get volume images, optionally filtered by embryo"""
        if embryo_id:
            cache = self._embryo_caches.get(embryo_id)
            return cache.volumes if cache else []
        all_vol = list(self._volume_images)
        for cache in self._embryo_caches.values():
            all_vol.extend(cache.volumes)
        return sorted(all_vol, key=lambda x: x.timestamp)

    def get_all_snapshots(self, embryo_id: Optional[str] = None) -> List[ImageData]:
        """Get snapshot images, optionally filtered by embryo"""
        if embryo_id:
            cache = self._embryo_caches.get(embryo_id)
            return cache.snapshots if cache else []
        all_snap = list(self._global_images)
        for cache in self._embryo_caches.values():
            all_snap.extend(cache.snapshots)
        return sorted(all_snap, key=lambda x: x.timestamp)

    def get_embryo_ids(self) -> List[str]:
        """Get list of all embryo IDs with images"""
        return list(self._embryo_caches.keys())

    def get_image_by_uid(self, uid: str) -> Optional[ImageData]:
        """Find image by UID across all storage"""
        for img in self._global_images:
            if img.uid == uid:
                return img
        for img in self._calibration_images:
            if img.uid == uid:
                return img
        for img in self._volume_images:
            if img.uid == uid:
                return img
        for cache in self._embryo_caches.values():
            for img in cache.volumes + cache.calibration + cache.snapshots:
                if img.uid == uid:
                    return img
        return None

    def add_volume_3d(self, volume_data: Volume3DData):
        """Add a 3D volume with segmentation"""
        self._volumes_3d[volume_data.uid] = volume_data
        # Keep only last 10 3D volumes to manage memory
        if len(self._volumes_3d) > 10:
            oldest_uid = next(iter(self._volumes_3d))
            del self._volumes_3d[oldest_uid]

    def get_volume_3d(self, uid: str) -> Optional[Volume3DData]:
        """Get a 3D volume by UID"""
        return self._volumes_3d.get(uid)

    def get_all_volumes_3d(self) -> List[Dict]:
        """Get info for all 3D volumes (without heavy data)"""
        return [v.to_info_dict() for v in self._volumes_3d.values()]

    def get_sequence(
        self,
        embryo_id: str,
        start: int = 0,
        end: Optional[int] = None,
        data_type: Optional[str] = None
    ) -> List[ImageData]:
        """Get ordered sequence of images for an embryo within a timepoint range.

        Args:
            embryo_id: The embryo to get images for
            start: Starting timepoint (inclusive)
            end: Ending timepoint (inclusive), None for all
            data_type: Filter by data type (e.g., 'volume_projection')

        Returns:
            List of ImageData sorted by timepoint
        """
        cache = self._embryo_caches.get(embryo_id)
        if not cache:
            return []

        # Get all images for this embryo (volumes + snapshots)
        all_images = list(cache.volumes) + list(cache.snapshots)

        # Filter by data type if specified
        if data_type:
            all_images = [img for img in all_images if img.data_type == data_type]

        # Filter by timepoint range
        def get_timepoint(img: ImageData) -> Optional[int]:
            tp = img.metadata.get('timepoint')
            if tp is not None:
                return int(tp)
            return None

        filtered = []
        for img in all_images:
            tp = get_timepoint(img)
            if tp is None:
                continue
            if tp < start:
                continue
            if end is not None and tp > end:
                continue
            filtered.append(img)

        # Sort by timepoint
        filtered.sort(key=lambda x: get_timepoint(x) or 0)
        return filtered

    def get_stats(self) -> Dict:
        """Get storage statistics"""
        total_cal = len(self._calibration_images)
        total_vol = len(self._volume_images)
        total_snap = len(self._global_images)

        for cache in self._embryo_caches.values():
            total_cal += len(cache.calibration)
            total_vol += len(cache.volumes)
            total_snap += len(cache.snapshots)

        return {
            'embryo_count': len(self._embryo_caches),
            'calibration_count': total_cal,
            'volume_count': total_vol,
            'snapshot_count': total_snap,
            'volumes_3d_count': len(self._volumes_3d),
            'embryo_ids': list(self._embryo_caches.keys()),
        }


class TimelapseStateTracker:
    """
    Tracks timelapse state from events for client synchronization.

    Maintains state from EventBus events so new WebSocket clients
    can receive current timelapse status on connect.

    Uses session_id to help clients identify session boundaries and
    clear stale data when a new experiment starts.
    """

    def __init__(self):
        self.session_id: Optional[str] = None  # Unique ID per experiment
        self.status = "IDLE"  # IDLE, RUNNING, PAUSED, COMPLETED
        self.started_at: Optional[str] = None
        self.embryos: Dict[str, dict] = {}  # embryo_id -> state
        self.total_timepoints = 0
        self.base_interval = 120
        self.detection_reasoning: Dict[str, List[dict]] = {}  # embryo_id -> list of detections

    def handle_event(self, event_type: str, data: dict):
        """Update state based on incoming event"""
        if event_type == "ACQUISITION_STARTED":
            # Generate new session ID for this experiment
            import uuid
            self.session_id = f"exp_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"
            self.status = "RUNNING"
            self.started_at = datetime.now().isoformat()
            self.base_interval = data.get("interval_seconds", 120)
            self.embryos = {}
            self.detection_reasoning = {}
            self.total_timepoints = 0
            for eid in data.get("embryo_ids", []):
                self.embryos[eid] = {
                    "embryo_id": eid,
                    "stop_condition": data.get("stop_condition", "manual"),
                    "interval_seconds": self.base_interval,
                    "timepoints": 0,
                    "is_complete": False,
                    "last_acquired": None,
                    "detections": {}
                }
                self.detection_reasoning[eid] = []

        elif event_type == "VOLUME_ACQUIRED":
            eid = data.get("embryo_id")
            if eid:
                # Create embryo if not exists (late join)
                if eid not in self.embryos:
                    self.embryos[eid] = {
                        "embryo_id": eid,
                        "stop_condition": "unknown",
                        "interval_seconds": self.base_interval,
                        "timepoints": 0,
                        "is_complete": False,
                        "last_acquired": None,
                        "detections": {}
                    }
                    self.detection_reasoning[eid] = []
                    if self.status == "IDLE":
                        self.status = "RUNNING"
                        self.started_at = datetime.now().isoformat()

                self.embryos[eid]["timepoints"] = data.get("timepoint", 0) + 1
                self.embryos[eid]["last_acquired"] = datetime.now().isoformat()
                self.total_timepoints += 1

        elif event_type == "ACQUISITION_COMPLETED":
            self.status = "COMPLETED"
            for embryo in self.embryos.values():
                embryo["is_complete"] = True

        elif event_type == "DETECTOR_EVALUATED":
            # All detector evaluations (with reasoning) - populates reasoning panel
            eid = data.get("embryo_id")
            if eid:
                detection = {
                    "detector_name": data.get("detector_name", "unknown"),
                    "detected": data.get("detected", False),
                    "confidence": data.get("confidence"),
                    "reasoning": data.get("reasoning"),
                    "timepoint": data.get("timepoint"),
                    "volume_uid": data.get("volume_uid"),
                    "projection_uid": data.get("projection_uid"),
                    "timestamp": datetime.now().isoformat()
                }
                if eid not in self.detection_reasoning:
                    self.detection_reasoning[eid] = []
                self.detection_reasoning[eid].append(detection)

        elif event_type in ("DETECTION_TRIGGERED", "HATCHING_DETECTED"):
            # Positive detection events - update embryo status
            eid = data.get("embryo_id")
            if eid and eid in self.embryos:
                detector_name = data.get("detector_name", "unknown")
                self.embryos[eid]["detections"][detector_name] = {
                    "detected": True,
                    "confidence": data.get("confidence")
                }
                if detector_name == "hatching":
                    self.embryos[eid]["is_complete"] = True

        elif event_type == "STATUS_CHANGED":
            if data.get("status"):
                self.status = data["status"]
            # Handle interval changes
            if data.get("embryo_id") and data.get("new_interval_seconds"):
                eid = data["embryo_id"]
                if eid in self.embryos:
                    self.embryos[eid]["interval_seconds"] = data["new_interval_seconds"]

    def to_dict(self) -> dict:
        """Serialize for WebSocket transmission"""
        return {
            "session_id": self.session_id,
            "status": self.status,
            "started_at": self.started_at,
            "embryos": self.embryos,
            "total_timepoints": self.total_timepoints,
            "base_interval": self.base_interval,
            "detection_reasoning": self.detection_reasoning
        }

    def reset(self):
        """Clear state for new timelapse"""
        self.__init__()


class ConnectionManager:
    """Manages WebSocket connections for broadcasting updates"""

    def __init__(self):
        self.active_connections: Set[WebSocket] = set()
        self._lock = asyncio.Lock()

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        async with self._lock:
            self.active_connections.add(websocket)
        logger.info(f"WebSocket connected. Total: {len(self.active_connections)}")

    async def disconnect(self, websocket: WebSocket):
        async with self._lock:
            self.active_connections.discard(websocket)
        logger.info(f"WebSocket disconnected. Total: {len(self.active_connections)}")

    async def broadcast(self, message: Dict):
        """Broadcast message to all connected clients"""
        if not self.active_connections:
            return

        message_json = json.dumps(message)
        async with self._lock:
            disconnected = set()
            for connection in self.active_connections:
                try:
                    await connection.send_text(message_json)
                except Exception as e:
                    logger.warning(f"Failed to send to websocket: {e}")
                    disconnected.add(connection)

            # Remove disconnected clients
            self.active_connections -= disconnected

    async def send_image(self, image_data: ImageData):
        """Send image data to all connected clients"""
        await self.broadcast({
            'type': 'image',
            'data': image_data.to_dict()
        })

    async def send_event(self, event_type: str, data: Dict, source: str = None, event_id: str = None):
        """Send event notification to all clients"""
        await self.broadcast({
            'type': 'event',
            'event_type': event_type,
            'data': data,
            'source': source or 'unknown',
            'event_id': event_id or '',
            'timestamp': datetime.now().isoformat()
        })


class VisualizationServer:
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
        host: str = "0.0.0.0",
        port: int = 8080,
        data_store=None,
        event_bus=None,
    ):
        if not FASTAPI_AVAILABLE:
            raise ImportError(
                "FastAPI is required for visualization server. "
                "Install with: pip install fastapi uvicorn"
            )

        self.host = host
        self.port = port
        self.data_store = data_store
        self.event_bus = event_bus

        # Connection manager for WebSocket clients
        self.manager = ConnectionManager()

        # Organized image storage
        self.store = ImageStore(max_per_category=50)

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

        # Setup routes
        self._setup_routes()

        # Subscribe to events if event bus provided
        if self.event_bus:
            self._subscribe_to_events()

        # Server instance
        self._server = None
        self._server_task = None

    def _setup_routes(self):
        """Setup FastAPI routes"""

        @self.app.get("/", response_class=HTMLResponse)
        async def index(request: Request):
            """Serve the main visualization page"""
            return self.templates.TemplateResponse(
                "index.html",
                {"request": request}
            )

        @self.app.get("/api/status")
        async def get_status():
            """Get server status"""
            stats = self.store.get_stats()
            return {
                "status": "running",
                "connections": len(self.manager.active_connections),
                **stats,
                "timestamp": datetime.now().isoformat()
            }

        @self.app.get("/api/calibration")
        async def list_calibration(embryo_id: Optional[str] = None):
            """Get calibration images"""
            images = self.store.get_all_calibration(embryo_id)
            return {
                "calibration": [img.to_dict() for img in images],
                "count": len(images)
            }

        @self.app.get("/api/volumes")
        async def list_volumes(embryo_id: Optional[str] = None):
            """Get volume images"""
            images = self.store.get_all_volumes(embryo_id)
            return {
                "volumes": [img.to_dict() for img in images],
                "count": len(images)
            }

        @self.app.get("/api/snapshots")
        async def list_snapshots(embryo_id: Optional[str] = None):
            """Get snapshot images"""
            images = self.store.get_all_snapshots(embryo_id)
            return {
                "snapshots": [img.to_dict() for img in images],
                "count": len(images)
            }

        @self.app.get("/api/embryos")
        async def list_embryos():
            """Get list of embryos with images"""
            return {
                "embryos": self.store.get_embryo_ids()
            }

        @self.app.get("/api/sequence/{embryo_id}")
        async def get_image_sequence(
            embryo_id: str,
            start: int = 0,
            end: Optional[int] = None,
            data_type: str = "volume_projection",
            buffer_percent: float = 0.15
        ):
            """Get ordered sequence of images for timepoint range.

            Returns a list of image metadata (without base64 data) for lazy loading.
            Use /api/images/{uid}/png to load individual frames.

            Args:
                embryo_id: The embryo to get images for
                start: Starting timepoint (inclusive)
                end: Ending timepoint (inclusive)
                data_type: Filter by data type (default: volume_projection)
                buffer_percent: Extend range by this percentage on each side

            Returns:
                Sequence metadata with UIDs for lazy loading
            """
            # Calculate buffered range
            if end is not None:
                range_size = end - start
                buffer = int(range_size * buffer_percent)
                buffered_start = max(0, start - buffer)
                buffered_end = end + buffer
            else:
                buffered_start = start
                buffered_end = None

            images = self.store.get_sequence(
                embryo_id=embryo_id,
                start=buffered_start,
                end=buffered_end,
                data_type=data_type
            )

            # Return lightweight metadata (no base64 data)
            sequence = []
            for img in images:
                sequence.append({
                    "uid": img.uid,
                    "timepoint": img.metadata.get("timepoint"),
                    "timestamp": img.timestamp,
                    "data_type": img.data_type,
                    "shape": img.shape,
                    "embryo_id": img.metadata.get("embryo_id")
                })

            return {
                "embryo_id": embryo_id,
                "requested_range": {"start": start, "end": end},
                "buffered_range": {"start": buffered_start, "end": buffered_end},
                "sequence": sequence,
                "count": len(sequence)
            }

        @self.app.get("/api/events")
        async def list_events(
            event_type: Optional[str] = None,
            source: Optional[str] = None,
            limit: int = 100
        ):
            """Get event history from EventBus"""
            if not self.event_bus:
                return {"events": [], "total": 0}

            # Get history from event bus
            from ..core import EventType
            et = None
            if event_type:
                try:
                    et = EventType[event_type]
                except KeyError:
                    pass

            events = self.event_bus.get_history(
                event_type=et,
                source=source,
                limit=limit
            )

            return {
                "events": [
                    {
                        "event_type": e.event_type.name if hasattr(e.event_type, 'name') else str(e.event_type),
                        "data": e.data,
                        "source": e.source,
                        "timestamp": e.timestamp.isoformat(),
                        "event_id": e.event_id
                    }
                    for e in events
                ],
                "total": len(events)
            }

        @self.app.get("/api/images/{uid}")
        async def get_image(uid: str):
            """Get image by UID"""
            image = self.store.get_image_by_uid(uid)
            if image:
                return image.to_dict()
            raise HTTPException(status_code=404, detail=f"Image {uid} not found")

        @self.app.get("/api/images/{uid}/png")
        async def get_image_png(uid: str):
            """Get image as PNG"""
            image = self.store.get_image_by_uid(uid)
            if image and image.base64_png:
                png_bytes = base64.b64decode(image.base64_png)
                return Response(content=png_bytes, media_type="image/png")
            raise HTTPException(status_code=404, detail=f"Image {uid} not found")

        @self.app.get("/api/volumes3d")
        async def list_volumes_3d():
            """Get list of 3D volumes (without heavy data)"""
            return {
                "volumes_3d": self.store.get_all_volumes_3d(),
                "count": len(self.store._volumes_3d)
            }

        @self.app.get("/api/volumes3d/{uid}")
        async def get_volume_3d_info(uid: str):
            """Get 3D volume info by UID"""
            vol = self.store.get_volume_3d(uid)
            if vol:
                return vol.to_info_dict()
            raise HTTPException(status_code=404, detail=f"3D Volume {uid} not found")

        @self.app.get("/api/volumes3d/{uid}/slice/{z}")
        async def get_volume_3d_slice(uid: str, z: int):
            """Get a specific Z-slice as PNG with segmentation overlay"""
            vol = self.store.get_volume_3d(uid)
            if not vol:
                raise HTTPException(status_code=404, detail=f"3D Volume {uid} not found")

            rgb = vol.get_slice_overlay(z)

            if PIL_AVAILABLE:
                img = Image.fromarray(rgb)
                buffer = io.BytesIO()
                img.save(buffer, format='PNG')
                return Response(content=buffer.getvalue(), media_type="image/png")

            raise HTTPException(status_code=500, detail="PIL not available")

        @self.app.post("/api/volumes3d")
        async def push_volume_3d_http(request: Request):
            """Push a 3D volume with segmentation via HTTP (for CV subagent)"""
            try:
                data = await request.json()

                # Decode the volume and masks from base64
                volume_b64 = data.get('volume_b64')
                masks_b64 = data.get('masks_b64')
                uid = data.get('uid')
                shape = data.get('shape')
                dtype_vol = data.get('dtype_vol', 'uint16')
                dtype_mask = data.get('dtype_mask', 'uint16')
                metadata = data.get('metadata', {})

                if not all([volume_b64, masks_b64, uid, shape]):
                    raise HTTPException(status_code=400, detail="Missing required fields")

                # Decode arrays
                volume = np.frombuffer(
                    base64.b64decode(volume_b64),
                    dtype=np.dtype(dtype_vol)
                ).reshape(shape)

                masks = np.frombuffer(
                    base64.b64decode(masks_b64),
                    dtype=np.dtype(dtype_mask)
                ).reshape(shape)

                # Push using the existing method
                await self.push_volume_3d(volume, masks, uid, metadata)

                return {"status": "ok", "uid": uid, "shape": shape}

            except Exception as e:
                logger.error(f"Failed to push 3D volume via HTTP: {e}")
                raise HTTPException(status_code=500, detail=str(e))

        @self.app.post("/api/images")
        async def push_image_http(request: Request):
            """Push a 2D image via HTTP (for CV subagent visualizations)"""
            try:
                data = await request.json()

                # Decode the image from base64
                image_b64 = data.get('image_b64')
                uid = data.get('uid')
                shape = data.get('shape')
                dtype = data.get('dtype', 'uint8')
                data_type = data.get('data_type', 'cv_visualization')
                metadata = data.get('metadata', {})

                if not all([image_b64, uid, shape]):
                    raise HTTPException(status_code=400, detail="Missing required fields")

                # Decode array
                array = np.frombuffer(
                    base64.b64decode(image_b64),
                    dtype=np.dtype(dtype)
                ).reshape(shape)

                # Push using the existing method
                await self.push_image(array, uid, data_type, metadata)

                return {"status": "ok", "uid": uid, "data_type": data_type}

            except Exception as e:
                logger.error(f"Failed to push image via HTTP: {e}")
                raise HTTPException(status_code=500, detail=str(e))

        @self.app.get("/api/narrative")
        async def get_narrative(since: Optional[str] = None):
            """Generate experiment narrative summary.

            This endpoint generates an AI-powered summary of the experiment state.
            Currently returns a local summary; can be extended to use Claude Haiku.

            Args:
                since: Optional ISO timestamp to get differential summary
            """
            return self._generate_narrative_summary(since)

        @self.app.websocket("/ws")
        async def websocket_endpoint(websocket: WebSocket):
            """WebSocket endpoint for real-time updates"""
            await self.manager.connect(websocket)
            try:
                # Send current status on connect
                stats = self.store.get_stats()
                await websocket.send_json({
                    "type": "connected",
                    **stats,
                    "timestamp": datetime.now().isoformat()
                })

                # Always send timelapse state on connect so client can reconcile
                # (if IDLE with no session_id, client will clear stale cached state)
                timelapse_state = self.timelapse_tracker.to_dict()
                await websocket.send_json({
                    "type": "timelapse_state",
                    "data": timelapse_state
                })

                # Keep connection alive and handle incoming messages
                while True:
                    try:
                        data = await asyncio.wait_for(
                            websocket.receive_text(),
                            timeout=30.0
                        )
                        # Handle client messages (e.g., requests)
                        await self._handle_ws_message(websocket, data)
                    except asyncio.TimeoutError:
                        # Send ping to keep connection alive
                        await websocket.send_json({"type": "ping"})

            except WebSocketDisconnect:
                try:
                    await self.manager.disconnect(websocket)
                except Exception:
                    pass
            except asyncio.CancelledError:
                try:
                    await self.manager.disconnect(websocket)
                except Exception:
                    pass
            except Exception as e:
                logger.error(f"WebSocket error: {e}")
                try:
                    await self.manager.disconnect(websocket)
                except Exception:
                    pass

    async def _handle_ws_message(self, websocket: WebSocket, message: str):
        """Handle incoming WebSocket message"""
        try:
            data = json.loads(message)
            msg_type = data.get("type")
            embryo_id = data.get("embryo_id")

            if msg_type == "get_calibration":
                images = self.store.get_all_calibration(embryo_id)
                await websocket.send_json({
                    "type": "calibration",
                    "data": [img.to_dict() for img in images]
                })

            elif msg_type == "get_volumes":
                images = self.store.get_all_volumes(embryo_id)
                await websocket.send_json({
                    "type": "volumes",
                    "data": [img.to_dict() for img in images]
                })

            elif msg_type == "get_snapshots":
                images = self.store.get_all_snapshots(embryo_id)
                await websocket.send_json({
                    "type": "snapshots",
                    "data": [img.to_dict() for img in images]
                })

            elif msg_type == "get_embryos":
                await websocket.send_json({
                    "type": "embryos",
                    "data": self.store.get_embryo_ids()
                })

            elif msg_type == "get_image":
                uid = data.get("uid")
                image = self.store.get_image_by_uid(uid)
                if image:
                    await websocket.send_json({
                        "type": "image",
                        "data": image.to_dict()
                    })

            elif msg_type == "pong":
                pass  # Client responding to ping

        except json.JSONDecodeError:
            logger.warning(f"Invalid JSON received: {message[:100]}")

    def _subscribe_to_events(self):
        """Subscribe to EventBus for automatic updates - broadcasts ALL events"""

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

        # Subscribe to ALL events using wildcard with async handler
        self.event_bus.subscribe_async("*", on_event_async)

        logger.info("Subscribed to ALL event types via wildcard")

    def _array_to_image_data(
        self,
        array: np.ndarray,
        uid: str,
        data_type: str,
        metadata: Optional[Dict] = None
    ) -> ImageData:
        """Convert numpy array to ImageData with base64 PNG"""

        # Handle 4D arrays (Views, Z, Y, X) - select View A only
        if array.ndim == 4:
            array = array[0]  # View A

        # Handle 3D arrays - distinguish RGB images from volumes
        if array.ndim == 3:
            if array.shape[2] in (3, 4):
                # It's an RGB(A) image
                pass
            else:
                # It's a volume (Z, H, W) - take max projection
                array = np.max(array, axis=0)

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

    def _generate_narrative_summary(self, since: Optional[str] = None) -> Dict[str, Any]:
        """Generate a narrative summary of the experiment state.

        This generates a local summary based on the timelapse tracker state.
        Can be extended to use Claude Haiku for AI-powered narratives.

        Args:
            since: Optional ISO timestamp to generate differential summary

        Returns:
            Dict with status, headline, summary, and details
        """
        tracker = self.timelapse_tracker

        # If no active experiment
        if tracker.status == "IDLE" and not tracker.embryos:
            return {
                "status": "normal",
                "headline": "No Active Experiment",
                "summary": None,
                "details": ["Start a timelapse to see experiment summaries here."],
                "generated_at": datetime.now().isoformat()
            }

        # Count embryos
        embryo_count = len(tracker.embryos)
        active_embryos = [e for e in tracker.embryos.values() if not e.get("is_complete")]
        completed_embryos = [e for e in tracker.embryos.values() if e.get("is_complete")]

        # Count detections
        total_detections = 0
        detection_details = []
        for embryo_id, reasoning_list in tracker.detection_reasoning.items():
            positives = [r for r in reasoning_list if r.get("detected")]
            total_detections += len(positives)
            for d in positives:
                detector = d.get("detector_name", "unknown")
                tp = d.get("timepoint", "?")
                detection_details.append(f"{embryo_id}: {detector.title()} at T{tp}")

        # Build details list
        details = []

        if len(active_embryos) > 0:
            details.append(f"{len(active_embryos)} embryo{'s' if len(active_embryos) != 1 else ''} actively imaging")

        if len(completed_embryos) > 0:
            details.append(f"{len(completed_embryos)} embryo{'s' if len(completed_embryos) != 1 else ''} completed")

        details.append(f"{tracker.total_timepoints} total timepoints acquired")

        if tracker.base_interval:
            interval_str = f"{tracker.base_interval // 60} min" if tracker.base_interval >= 60 else f"{tracker.base_interval}s"
            details.append(f"Imaging interval: {interval_str}")

        if detection_details:
            if len(detection_details) <= 3:
                details.append(f"{total_detections} detection{'s' if total_detections != 1 else ''}: {', '.join(detection_details)}")
            else:
                details.append(f"{total_detections} detection{'s' if total_detections != 1 else ''}: {', '.join(detection_details[:3])}...")

        # Calculate duration
        if tracker.started_at:
            started = datetime.fromisoformat(tracker.started_at) if isinstance(tracker.started_at, str) else tracker.started_at
            duration_sec = (datetime.now() - started).total_seconds()
            hours = int(duration_sec // 3600)
            minutes = int((duration_sec % 3600) // 60)
            duration_str = f"{hours}h {minutes}m" if hours > 0 else f"{minutes}m"
            details.append(f"Running for {duration_str}")

        # Determine status and headline
        if total_detections > 0:
            status = "notable"
            headline = f"{total_detections} Detection{'s' if total_detections != 1 else ''} Found"
        elif len(completed_embryos) > 0:
            status = "normal"
            headline = f"{len(completed_embryos)}/{embryo_count} Embryos Complete"
        else:
            status = "normal"
            headline = "Experiment In Progress"

        # Build summary text
        summary = None
        if total_detections > 0:
            latest = detection_details[-1] if detection_details else None
            summary = f"Positive detections have been identified. {latest}. All imaging continues normally."
        elif len(completed_embryos) > 0:
            summary = f"{len(completed_embryos)} embryo{'s have' if len(completed_embryos) != 1 else ' has'} reached their stop condition. {len(active_embryos)} still being imaged."

        return {
            "status": status,
            "headline": headline,
            "summary": summary,
            "details": details,
            "generated_at": datetime.now().isoformat()
        }

    async def start(self):
        """Start the visualization server"""
        # Set the event loop on the event bus so async handlers work
        # even when events are published from sync code
        if self.event_bus:
            self.event_bus.set_event_loop(asyncio.get_running_loop())

        config = uvicorn.Config(
            self.app,
            host=self.host,
            port=self.port,
            log_level="warning",
        )
        self._server = uvicorn.Server(config)

        logger.info(f"Starting visualization server on http://{self.host}:{self.port}")

        # Run server in background task
        self._server_task = asyncio.create_task(self._server.serve())

        # Wait a moment for server to start
        await asyncio.sleep(0.5)

        logger.info(f"Visualization server running at http://{self.host}:{self.port}")

    async def stop(self):
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


# NOTE: HTML/CSS/JS has been extracted to gently/visualization/web/
# - templates/index.html
# - static/css/main.css
# - static/js/app.js, websocket.js, viewer.js, gallery.js


# Convenience function
def create_visualization_server(
    port: int = 8080,
    data_store=None,
    event_bus=None,
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

    Returns
    -------
    VisualizationServer
        Configured server instance
    """
    return VisualizationServer(
        port=port,
        data_store=data_store,
        event_bus=event_bus,
    )
