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
    from fastapi.responses import HTMLResponse, JSONResponse, Response, FileResponse
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
class ClientInfo:
    """Information about a connected WebSocket client for presence tracking"""
    client_id: str
    name: str
    color: str  # Hex color for avatar background
    connected_at: str


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
    """Organized storage for images by type and embryo (unlimited)"""

    def __init__(self):
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
            else:
                self._calibration_images.append(image)

        elif data_type in VOLUME_TYPES:
            if embryo_id:
                cache = self._get_embryo_cache(embryo_id)
                cache.volumes.append(image)
            else:
                self._volume_images.append(image)
        else:
            # General snapshot/other
            if embryo_id:
                cache = self._get_embryo_cache(embryo_id)
                cache.snapshots.append(image)
            else:
                self._global_images.append(image)

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
        """Get snapshot images (including volume projections), optionally filtered by embryo"""
        if embryo_id:
            cache = self._embryo_caches.get(embryo_id)
            if not cache:
                return []
            # Include both snapshots and volumes for the embryo
            return sorted(cache.snapshots + cache.volumes, key=lambda x: x.timestamp)
        # Include all snapshots and volumes
        all_snap = list(self._global_images) + list(self._volume_images)
        for cache in self._embryo_caches.values():
            all_snap.extend(cache.snapshots)
            all_snap.extend(cache.volumes)
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
        self.projection_uids: Dict[str, Dict[int, str]] = {}  # embryo_id -> {timepoint -> projection_uid}
        self.volume_paths: Dict[str, Dict[int, str]] = {}  # embryo_id -> {timepoint -> volume_path}

    def handle_event(self, event_type: str, data: dict):
        """Update state based on incoming event"""
        if event_type == "SESSION_STARTED":
            # New session - clear all state from previous session
            self.session_id = data.get("session_id")
            self.status = "IDLE"
            self.started_at = None
            self.embryos = {}
            self.detection_reasoning = {}
            self.projection_uids = {}
            self.volume_paths = {}
            self.total_timepoints = 0

        elif event_type == "SESSION_RESTORED":
            # Capture session ID when copilot resumes a session
            self.session_id = data.get("session_id")

        elif event_type == "ACQUISITION_STARTED":
            # Use session_id from prior SESSION_STARTED/SESSION_RESTORED event
            # (session_id should already be set before acquisition starts)
            self.status = "RUNNING"
            self.started_at = datetime.now().isoformat()
            self.base_interval = data.get("interval_seconds", 120)
            self.embryos = {}
            self.detection_reasoning = {}
            self.projection_uids = {}
            self.volume_paths = {}
            self.total_timepoints = 0
            for eid in data.get("embryo_ids", []):
                self.embryos[eid] = {
                    "embryo_id": eid,
                    "stop_condition": data.get("stop_condition", "manual"),
                    "interval_seconds": self.base_interval,
                    "timepoints": 0,
                    "is_complete": False,
                    "first_acquired": None,
                    "last_acquired": None,
                    "detections": {},
                    "current_stage": None,  # Updated by perception system
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
                        "first_acquired": None,
                        "last_acquired": None,
                        "detections": {},
                        "current_stage": None,  # Updated by perception system
                    }
                    self.detection_reasoning[eid] = []
                    if self.status == "IDLE":
                        self.status = "RUNNING"
                        self.started_at = datetime.now().isoformat()

                now = datetime.now().isoformat()
                # timepoint is already the count (timepoints_acquired), not 0-indexed
                timepoint = data.get("timepoint", 1)
                self.embryos[eid]["timepoints"] = timepoint
                if self.embryos[eid]["first_acquired"] is None:
                    self.embryos[eid]["first_acquired"] = now
                self.embryos[eid]["last_acquired"] = now
                self.total_timepoints += 1

                # Track projection UID for image lookup
                projection_uid = data.get("projection_uid")
                if projection_uid:
                    if eid not in self.projection_uids:
                        self.projection_uids[eid] = {}
                    self.projection_uids[eid][timepoint] = projection_uid

                # Track volume path for direct file access (projection generation)
                volume_path = data.get("volume_path")
                if volume_path:
                    if eid not in self.volume_paths:
                        self.volume_paths[eid] = {}
                    self.volume_paths[eid][timepoint] = volume_path

        elif event_type == "ACQUISITION_COMPLETED":
            self.status = "COMPLETED"
            for embryo in self.embryos.values():
                embryo["is_complete"] = True

        elif event_type == "ACQUISITION_STOPPED":
            self.status = "STOPPED"
            # Don't mark embryos as complete - they were stopped, not finished

        elif event_type == "DETECTOR_EVALUATED":
            # All detector/perception evaluations (with reasoning) - populates reasoning panel
            eid = data.get("embryo_id")
            if eid:
                timepoint = data.get("timepoint")
                # Look up projection UID for this timepoint
                projection_uid = None
                if eid in self.projection_uids and timepoint in self.projection_uids.get(eid, {}):
                    projection_uid = self.projection_uids[eid][timepoint]

                detection = {
                    "detector_name": data.get("detector_name", "unknown"),
                    "detected": data.get("detected", data.get("is_hatching", False)),
                    "confidence": data.get("confidence"),
                    "reasoning": data.get("reasoning"),
                    "timepoint": timepoint,
                    "volume_uid": data.get("volume_uid"),
                    "projection_uid": data.get("projection_uid") or projection_uid,  # Use stored UID as fallback
                    "timestamp": datetime.now().isoformat(),
                    # Perception-specific fields
                    "stage": data.get("stage"),
                    "is_hatching": data.get("is_hatching", False),
                    # Full reasoning trace from VLM (for detail panel)
                    "reasoning_trace": data.get("reasoning_trace"),
                    "is_transitional": data.get("is_transitional"),
                    "transition_between": data.get("transition_between"),
                    "observed_features": data.get("observed_features"),
                    "shape": data.get("shape"),
                }
                if eid not in self.detection_reasoning:
                    self.detection_reasoning[eid] = []
                self.detection_reasoning[eid].append(detection)

                # Update embryo's current stage if perception result
                if data.get("stage") and eid in self.embryos:
                    self.embryos[eid]["current_stage"] = data.get("stage")

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

        elif event_type == "VERIFICATION_STARTED":
            # Verification round started for embryo
            eid = data.get("embryo_id")
            if eid and eid in self.embryos:
                self.embryos[eid]["verification"] = {
                    "status": "running",
                    "consecutive_count": data.get("consecutive_count", 0),
                    "required_count": data.get("required_count", 5),
                    "strategies_complete": 0,
                    "total_strategies": 5,
                    "strategies": {},
                }

        elif event_type == "VERIFICATION_STRATEGY":
            # Individual strategy result
            eid = data.get("embryo_id")
            if eid and eid in self.embryos and "verification" in self.embryos[eid]:
                strategy = data.get("strategy")
                self.embryos[eid]["verification"]["strategies"][strategy] = {
                    "passed": data.get("passed"),
                    "summary": data.get("summary"),
                }

        elif event_type == "VERIFICATION_PROGRESS":
            # Progress update
            eid = data.get("embryo_id")
            if eid and eid in self.embryos and "verification" in self.embryos[eid]:
                self.embryos[eid]["verification"]["strategies_complete"] = data.get("strategies_complete", 0)
                self.embryos[eid]["verification"]["total_strategies"] = data.get("total_strategies", 5)

        elif event_type == "VERIFICATION_COMPLETED":
            # Final verification result
            eid = data.get("embryo_id")
            if eid and eid in self.embryos:
                self.embryos[eid]["verification"] = {
                    "status": "completed",
                    "consensus": data.get("consensus"),
                    "reasoning": data.get("reasoning"),
                    "strategies": data.get("strategies", {}),
                    "ensemble_votes": data.get("ensemble_votes"),
                    "duration_seconds": data.get("duration_seconds"),
                }
                # Update consecutive count display
                if data.get("consensus"):
                    current = self.embryos[eid].get("consecutive_verified", 0)
                    self.embryos[eid]["consecutive_verified"] = current + 1
                else:
                    self.embryos[eid]["consecutive_verified"] = 0

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
    """Manages WebSocket connections for broadcasting updates with presence tracking"""

    # Colors for avatar backgrounds (pleasant, distinct colors)
    AVATAR_COLORS = [
        '#4a9eff', '#ff6b6b', '#51cf66', '#ffd43b', '#cc5de8',
        '#ff922b', '#20c997', '#748ffc', '#f06595', '#69db7c',
        '#ffa94d', '#9775fa', '#38d9a9', '#e599f7', '#74c0fc'
    ]

    def __init__(self):
        self.active_connections: Dict[WebSocket, ClientInfo] = {}
        self._lock = asyncio.Lock()

    def _generate_color(self, client_id: str) -> str:
        """Generate consistent color from client_id"""
        hash_val = sum(ord(c) for c in client_id)
        return self.AVATAR_COLORS[hash_val % len(self.AVATAR_COLORS)]

    async def connect(self, websocket: WebSocket, client_id: str = None, name: str = None):
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
            connected_at=datetime.now().isoformat()
        )

        async with self._lock:
            self.active_connections[websocket] = client_info
        logger.info(f"WebSocket connected: {name} ({client_id}). Total: {len(self.active_connections)}")

        # Broadcast updated presence to all clients
        await self.broadcast_presence()

    async def disconnect(self, websocket: WebSocket):
        async with self._lock:
            client_info = self.active_connections.pop(websocket, None)
        if client_info:
            logger.info(f"WebSocket disconnected: {client_info.name}. Total: {len(self.active_connections)}")
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
                    connected_at=old_info.connected_at
                )
        await self.broadcast_presence()

    def get_client_info(self, websocket: WebSocket) -> Optional[ClientInfo]:
        """Get client info for a websocket"""
        return self.active_connections.get(websocket)

    async def broadcast_presence(self):
        """Broadcast current presence list to all clients"""
        if not self.active_connections:
            return

        # Deduplicate by client_id (same user in multiple tabs = one avatar)
        async with self._lock:
            seen_clients = {}
            for ws, info in self.active_connections.items():
                # Keep the most recent entry for each client_id
                seen_clients[info.client_id] = {
                    'client_id': info.client_id,
                    'name': info.name,
                    'color': info.color
                }
            clients_list = list(seen_clients.values())

        # Send personalized presence to each client (with is_you flag)
        for ws, info in list(self.active_connections.items()):
            try:
                personalized = []
                for client in clients_list:
                    personalized.append({
                        **client,
                        'is_you': client['client_id'] == info.client_id
                    })
                await ws.send_json({
                    'type': 'presence',
                    'clients': personalized
                })
            except Exception as e:
                logger.warning(f"Failed to send presence to client: {e}")

    async def broadcast(self, message: Dict):
        """Broadcast message to all connected clients"""
        if not self.active_connections:
            return

        message_json = json.dumps(message)
        async with self._lock:
            disconnected = []
            for connection in self.active_connections.keys():
                try:
                    await connection.send_text(message_json)
                except Exception as e:
                    logger.warning(f"Failed to send to websocket: {e}")
                    disconnected.append(connection)

            # Remove disconnected clients
            for conn in disconnected:
                self.active_connections.pop(conn, None)

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
        sessions_dir: str = "D:/Gently/sessions",
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
        self.sessions_dir = Path(sessions_dir)

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

        @self.app.get("/review", response_class=HTMLResponse)
        async def review_page(request: Request):
            """Serve the session review page"""
            return self.templates.TemplateResponse(
                "review.html",
                {"request": request}
            )

        @self.app.get("/api/sessions")
        async def list_sessions():
            """List available sessions with metadata"""
            sessions = []
            if self.sessions_dir.exists():
                for path in self.sessions_dir.glob("*.json"):
                    try:
                        with open(path) as f:
                            data = json.load(f)
                        sessions.append({
                            'session_id': data.get('session_id', path.stem),
                            'name': data.get('name', path.stem),
                            'created_at': data.get('created_at', ''),
                            'last_active': data.get('last_active', ''),
                            'embryo_count': len(data.get('embryo_states', {})),
                            'description': data.get('description', '')
                        })
                    except Exception as e:
                        logger.warning(f"Failed to read session {path}: {e}")
            # Sort by created_at descending (newest first)
            sessions.sort(key=lambda x: x.get('created_at', ''), reverse=True)
            return {'sessions': sessions}

        @self.app.get("/api/sessions/{session_id}")
        async def get_session(session_id: str):
            """Get full session state for review"""
            path = self.sessions_dir / f"{session_id}.json"
            if not path.exists():
                raise HTTPException(status_code=404, detail="Session not found")
            try:
                with open(path) as f:
                    return json.load(f)
            except Exception as e:
                raise HTTPException(status_code=500, detail=f"Failed to load session: {e}")

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
            seen_uids = set()
            for img in images:
                seen_uids.add(img.uid)
                sequence.append({
                    "uid": img.uid,
                    "timepoint": img.metadata.get("timepoint"),
                    "timestamp": img.timestamp,
                    "data_type": img.data_type,
                    "shape": img.shape,
                    "embryo_id": img.metadata.get("embryo_id")
                })

            # Fallback to persistent DataStore for missing timepoints
            if self.data_store and (len(sequence) == 0 or buffered_end is not None):
                try:
                    refs = self.data_store.query(
                        data_type=data_type,
                        embryo_id=embryo_id
                    )
                    for ref in refs:
                        if ref.uid in seen_uids:
                            continue
                        tp = ref.metadata.get('timepoint')
                        if tp is None:
                            continue
                        tp = int(tp)
                        if tp < buffered_start:
                            continue
                        if buffered_end is not None and tp > buffered_end:
                            continue
                        seen_uids.add(ref.uid)
                        sequence.append({
                            "uid": ref.uid,
                            "timepoint": tp,
                            "timestamp": ref.metadata.get('timestamp', ''),
                            "data_type": ref.data_type,
                            "shape": ref.metadata.get('shape'),
                            "embryo_id": embryo_id
                        })
                    # Re-sort by timepoint
                    sequence.sort(key=lambda x: x.get('timepoint') or 0)
                except Exception as e:
                    logger.warning(f"DataStore fallback failed: {e}")

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
            # Fallback to persistent DataStore
            if self.data_store:
                try:
                    data = self.data_store.retrieve(uid)
                    if data is not None:
                        return {"uid": uid, "data": "loaded_from_store"}
                except Exception:
                    pass
            raise HTTPException(status_code=404, detail=f"Image {uid} not found")

        @self.app.get("/api/images/{uid}/png")
        async def get_image_png(uid: str):
            """Get image as PNG (cached - images are immutable)"""
            # Cache headers - images are immutable so cache aggressively
            cache_headers = {
                "Cache-Control": "public, max-age=86400, immutable",  # 24 hours
                "ETag": f'"{uid}"',
            }

            image = self.store.get_image_by_uid(uid)

            # Fallback: If UID follows volume_EMBRYOID_tNNNN pattern, try looking up real UID
            if not image and uid.startswith("volume_"):
                import re
                match = re.match(r"volume_(.+)_t(\d+)$", uid)
                if match:
                    embryo_id, timepoint_str = match.groups()
                    timepoint = int(timepoint_str)
                    # Look up real projection UID from timelapse tracker
                    if embryo_id in self.timelapse_tracker.projection_uids:
                        real_uid = self.timelapse_tracker.projection_uids[embryo_id].get(timepoint)
                        if real_uid:
                            image = self.store.get_image_by_uid(real_uid)

            if image and image.base64_png:
                png_bytes = base64.b64decode(image.base64_png)
                return Response(content=png_bytes, media_type="image/png", headers=cache_headers)
            # Fallback to persistent DataStore
            if self.data_store:
                try:
                    # Try with original UID first
                    data = self.data_store.retrieve(uid)
                    # If not found and this is a fallback pattern, try with real UID
                    if data is None and uid.startswith("volume_"):
                        import re
                        match = re.match(r"volume_(.+)_t(\d+)$", uid)
                        if match:
                            embryo_id, timepoint_str = match.groups()
                            timepoint = int(timepoint_str)
                            if embryo_id in self.timelapse_tracker.projection_uids:
                                real_uid = self.timelapse_tracker.projection_uids[embryo_id].get(timepoint)
                                if real_uid:
                                    data = self.data_store.retrieve(real_uid)
                    if data is not None:
                        import numpy as np
                        from io import BytesIO
                        from PIL import Image
                        from gently.agent.perception.projection import (
                            projection_three_view,
                            compute_crop_bounds,
                            apply_crop_bounds,
                        )
                        # Handle numpy array
                        if isinstance(data, np.ndarray):
                            # Handle 4D volumes (Views, Z, Y, X) - take View A
                            if data.ndim == 4:
                                data = data[0]
                            # Handle 3D volumes - generate three-view projection
                            if data.ndim == 3:
                                z_depth, height, width = data.shape
                                # Handle dual-view format
                                if width > height * 2:
                                    data = data[:, :, :width // 2]
                                # Auto-crop and project
                                bounds = compute_crop_bounds(data)
                                data = apply_crop_bounds(data, bounds)
                                data, _ = projection_three_view(data)
                            # Normalize to uint8 if needed
                            if data.dtype != np.uint8:
                                data = ((data - data.min()) / (data.max() - data.min() + 1e-8) * 255).astype(np.uint8)
                            img = Image.fromarray(data)
                            buf = BytesIO()
                            img.save(buf, format='PNG')
                            return Response(content=buf.getvalue(), media_type="image/png", headers=cache_headers)
                except Exception as e:
                    logger.warning(f"Failed to load image {uid} from DataStore: {e}")
            raise HTTPException(status_code=404, detail=f"Image {uid} not found")

        @self.app.get("/api/projections/{embryo_id}/{timepoint}")
        async def get_projections(embryo_id: str, timepoint: int, method: str = "all"):
            """
            Generate projections from volume file on disk.

            Args:
                embryo_id: Embryo identifier
                timepoint: Timepoint number (1-indexed)
                method: Projection method - 'all', 'three_view', 'dual_view', 'depth_colored', 'multi_slice'

            Returns:
                List of projections with method name, description, and base64 PNG data
            """
            import numpy as np
            from io import BytesIO
            from PIL import Image
            import tifffile
            from gently.agent.perception.projection import (
                projection_three_view,
                compute_crop_bounds,
                apply_crop_bounds,
            )

            # Look up volume path
            if embryo_id not in self.timelapse_tracker.volume_paths:
                raise HTTPException(status_code=404, detail=f"No volumes for embryo {embryo_id}")

            volume_path = self.timelapse_tracker.volume_paths[embryo_id].get(timepoint)
            if not volume_path:
                raise HTTPException(status_code=404, detail=f"No volume for {embryo_id} at timepoint {timepoint}")

            # Load volume from disk
            try:
                from pathlib import Path
                path = Path(volume_path)
                if not path.exists():
                    raise HTTPException(status_code=404, detail=f"Volume file not found: {volume_path}")

                vol = tifffile.imread(str(path))
                vol = np.squeeze(vol)

                # Handle dual-view format (diSPIM)
                if vol.ndim == 3:
                    z_depth, height, width = vol.shape
                    if width > height * 2:
                        vol = vol[:, :, :width // 2]

                # Auto-crop to embryo region
                bounds = compute_crop_bounds(vol)
                vol = apply_crop_bounds(vol, bounds)

                # Normalize to 0-1 float
                vol = vol.astype(np.float32)
                vol = (vol - vol.min()) / (vol.max() - vol.min() + 1e-8)

            except HTTPException:
                raise
            except Exception as e:
                logger.error(f"Failed to load volume: {e}")
                raise HTTPException(status_code=500, detail=f"Failed to load volume: {e}")

            # Define projection methods
            def image_to_base64(img_array):
                """Convert numpy array to base64 PNG"""
                if img_array.dtype != np.uint8:
                    img_array = (img_array * 255).astype(np.uint8)
                img = Image.fromarray(img_array)
                buf = BytesIO()
                img.save(buf, format='PNG')
                return base64.b64encode(buf.getvalue()).decode('utf-8')

            PROJECTION_METHODS = {
                'three_view': projection_three_view,
            }

            # Try to import additional projection methods from explorer
            try:
                from gently.dataset.explorer_server import (
                    projection_dual_view,
                    projection_depth_colored,
                    projection_multi_slice,
                    projection_spin_3d,
                )
                PROJECTION_METHODS.update({
                    'dual_view': projection_dual_view,
                    'depth_colored': projection_depth_colored,
                    'multi_slice': projection_multi_slice,
                    'spin_3d': projection_spin_3d,
                })
            except ImportError:
                pass  # Explorer projections not available

            projections = []

            if method == "all":
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
            else:
                if method not in PROJECTION_METHODS:
                    raise HTTPException(status_code=400, detail=f"Unknown method: {method}. Available: {list(PROJECTION_METHODS.keys())}")
                proj_img, desc = PROJECTION_METHODS[method](vol)
                projections.append({
                    "method": method,
                    "description": desc,
                    "data": image_to_base64(proj_img),
                })

            return {
                "embryo_id": embryo_id,
                "timepoint": timepoint,
                "volume_shape": list(vol.shape),
                "projections": projections,
            }

        @self.app.get("/api/volume-raw/{embryo_id}/{timepoint}")
        async def get_volume_raw(embryo_id: str, timepoint: int):
            """
            Get raw volume data for 3D viewer.

            Returns the volume as base64-encoded uint8 bytes with shape info.
            """
            import numpy as np
            import tifffile
            from gently.agent.perception.projection import (
                compute_crop_bounds,
                apply_crop_bounds,
            )

            # Look up volume path
            if embryo_id not in self.timelapse_tracker.volume_paths:
                raise HTTPException(status_code=404, detail=f"No volumes for embryo {embryo_id}")

            volume_path = self.timelapse_tracker.volume_paths[embryo_id].get(timepoint)
            if not volume_path:
                raise HTTPException(status_code=404, detail=f"No volume for {embryo_id} at timepoint {timepoint}")

            try:
                from pathlib import Path
                path = Path(volume_path)
                if not path.exists():
                    raise HTTPException(status_code=404, detail=f"Volume file not found")

                vol = tifffile.imread(str(path))
                vol = np.squeeze(vol)

                # Handle dual-view format (diSPIM)
                if vol.ndim == 3:
                    z_depth, height, width = vol.shape
                    if width > height * 2:
                        vol = vol[:, :, :width // 2]

                # Auto-crop to embryo region
                bounds = compute_crop_bounds(vol)
                vol = apply_crop_bounds(vol, bounds)

                # Normalize to uint8
                vol = vol.astype(np.float32)
                vol = (vol - vol.min()) / (vol.max() - vol.min() + 1e-8)
                vol_uint8 = (vol * 255).astype(np.uint8)

                # Encode as base64
                vol_bytes = vol_uint8.tobytes()
                vol_b64 = base64.b64encode(vol_bytes).decode('utf-8')

                return {
                    "embryo_id": embryo_id,
                    "timepoint": timepoint,
                    "shape": list(vol_uint8.shape),
                    "data": vol_b64,
                }

            except HTTPException:
                raise
            except Exception as e:
                logger.error(f"Failed to load volume: {e}")
                raise HTTPException(status_code=500, detail=f"Failed to load volume: {e}")

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

        @self.app.get("/api/volume-data/{uid}")
        async def get_volume_data_for_3d_viewer(uid: str):
            """Get raw volume data as base64 for 3D viewer (projection viewer)"""
            # First check if it's a 3D segmented volume
            vol = self.store.get_volume_3d(uid)
            if vol:
                # Return the raw volume data (normalized to uint8)
                volume = vol.volume
                if volume.dtype != np.uint8:
                    vmin, vmax = volume.min(), volume.max()
                    if vmax > vmin:
                        volume = ((volume - vmin) / (vmax - vmin) * 255).astype(np.uint8)
                    else:
                        volume = np.zeros(volume.shape, dtype=np.uint8)
                return {
                    "shape": list(volume.shape),
                    "data": base64.b64encode(volume.tobytes()).decode('utf-8'),
                    "uid": uid
                }

            # Check if it's a regular image with stored volume data
            image = self.store.get_image_by_uid(uid)
            if image and image.shape and len(image.shape) == 3:
                # Try to retrieve the original volume from metadata or stored data
                # For now, return error - volume data not stored for regular images
                raise HTTPException(status_code=404, detail=f"Volume data for {uid} not available - only segmented volumes supported")

            raise HTTPException(status_code=404, detail=f"Volume {uid} not found")

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
            # Connect with temporary defaults - client will send 'join' message with real info
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

            # Presence-related messages
            elif msg_type == "join":
                # Client joining with identity info
                client_id = data.get("client_id")
                name = data.get("name")
                if client_id:
                    # Update the client's info
                    async with self.manager._lock:
                        if websocket in self.manager.active_connections:
                            old_info = self.manager.active_connections[websocket]
                            self.manager.active_connections[websocket] = ClientInfo(
                                client_id=client_id,
                                name=name or old_info.name,
                                color=self.manager._generate_color(client_id),
                                connected_at=old_info.connected_at
                            )
                    await self.manager.broadcast_presence()

            elif msg_type == "set_name":
                # Client updating their display name
                name = data.get("name")
                if name:
                    await self.manager.update_client_name(websocket, name)

            elif msg_type == "get_presence":
                # Client requesting current presence list
                await self.manager.broadcast_presence()

        except json.JSONDecodeError:
            logger.warning(f"Invalid JSON received: {message[:100]}")

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
        from gently.agent.perception.projection import (
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

        # Check if using perception system (has stage data) or legacy detectors
        is_perception = False
        stage_info = {}  # embryo_id -> current stage
        hatching_embryos = []
        stage_order = ['early', 'bean', 'comma', '1.5fold', '2fold', '3fold', 'pretzel', 'hatching', 'hatched']

        for embryo_id, reasoning_list in tracker.detection_reasoning.items():
            # Check for stage data (perception system)
            stages = [r.get("stage") for r in reasoning_list if r.get("stage")]
            if stages:
                is_perception = True
                stage_info[embryo_id] = stages[-1]  # Latest stage
                # Check for hatching
                if any(r.get("is_hatching") for r in reasoning_list):
                    hatching_embryos.append(embryo_id)

        # Count detections (legacy) or stage progression (perception)
        total_detections = 0
        detection_details = []

        if is_perception:
            # Group embryos by stage for perception
            for embryo_id, stage in stage_info.items():
                detection_details.append(f"{embryo_id}: {stage.replace('fold', '-fold').title()}")
        else:
            # Legacy detection counting
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

        if is_perception and stage_info:
            # Show stage distribution for perception
            stage_counts = {}
            for stage in stage_info.values():
                stage_counts[stage] = stage_counts.get(stage, 0) + 1
            # Sort by stage order
            sorted_stages = sorted(stage_counts.items(),
                                   key=lambda x: stage_order.index(x[0].lower()) if x[0].lower() in stage_order else 99)
            stage_summary = ", ".join(f"{count} {stage}" for stage, count in sorted_stages)
            details.append(f"Stages: {stage_summary}")
            if hatching_embryos:
                details.append(f"Hatching detected: {', '.join(hatching_embryos)}")
        elif detection_details:
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
        if is_perception:
            if hatching_embryos:
                status = "notable"
                headline = f"Hatching Detected in {len(hatching_embryos)} Embryo{'s' if len(hatching_embryos) != 1 else ''}"
            elif stage_info:
                # Find most advanced stage
                max_stage_idx = max(stage_order.index(s.lower()) if s.lower() in stage_order else 0
                                    for s in stage_info.values())
                max_stage = stage_order[max_stage_idx].replace('fold', '-fold').title()
                status = "normal"
                headline = f"Most Advanced: {max_stage}"
            else:
                status = "normal"
                headline = "Experiment In Progress"
        elif total_detections > 0:
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
        if is_perception:
            if hatching_embryos:
                summary = f"Hatching has been detected in {', '.join(hatching_embryos)}. Monitoring continues for all embryos."
            elif stage_info:
                # Summarize stage distribution
                unique_stages = set(stage_info.values())
                if len(unique_stages) == 1:
                    summary = f"All {len(stage_info)} embryos are at {list(unique_stages)[0].replace('fold', '-fold').title()} stage."
                else:
                    summary = f"Embryos are progressing through developmental stages. {len(stage_info)} embryos tracked."
        elif total_detections > 0:
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
