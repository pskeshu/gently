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
    from fastapi.middleware.cors import CORSMiddleware
    import uvicorn
    FASTAPI_AVAILABLE = True
except ImportError:
    FASTAPI_AVAILABLE = False
    logger.warning("FastAPI not available. Install with: pip install fastapi uvicorn")

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

    async def send_event(self, event_type: str, data: Dict):
        """Send event notification to all clients"""
        await self.broadcast({
            'type': 'event',
            'event_type': event_type,
            'data': data,
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

        # Create FastAPI app
        self.app = FastAPI(
            title="Gently Visualization Server",
            description="Real-time microscopy visualization",
            version="2.0.0"
        )

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
        async def index():
            """Serve the main visualization page"""
            return self._get_html_page()

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
        """Subscribe to EventBus for automatic updates"""
        from ..core import EventType

        events_to_broadcast = [
            EventType.VOLUME_ACQUIRED,
            EventType.DETECTION_TRIGGERED,
            EventType.HATCHING_DETECTED,
            EventType.EMBRYO_CENTERED,
            EventType.ANALYSIS_COMPLETED,
            EventType.SESSION_STARTED,
            EventType.SESSION_RESTORED,
            # CV Agent events
            EventType.CV_TASK_QUEUED,
            EventType.CV_TASK_COMPLETED,
            EventType.CV_TASK_FAILED,
            EventType.CV_AGENT_THINKING,
            EventType.SEGMENTATION_COMPLETED,
            EventType.STAGE_DETECTED,
        ]

        for event_type in events_to_broadcast:
            self.event_bus.subscribe(
                event_type,
                lambda e, et=event_type: asyncio.create_task(
                    self.manager.send_event(et.name, e.data)
                )
            )

        logger.info(f"Subscribed to {len(events_to_broadcast)} event types")

    def _array_to_image_data(
        self,
        array: np.ndarray,
        uid: str,
        data_type: str,
        metadata: Optional[Dict] = None
    ) -> ImageData:
        """Convert numpy array to ImageData with base64 PNG"""

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

    async def start(self):
        """Start the visualization server"""
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

    def _get_html_page(self) -> str:
        """Generate the main HTML page with tabs"""
        return '''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Gently - Microscopy Visualization</title>
    <style>
        :root {
            --bg-dark: #0d1117;
            --bg-card: #161b22;
            --bg-hover: #21262d;
            --border: #30363d;
            --text: #c9d1d9;
            --text-muted: #8b949e;
            --accent: #58a6ff;
            --accent-green: #3fb950;
            --accent-purple: #a371f7;
            --accent-orange: #d29922;
        }

        * { margin: 0; padding: 0; box-sizing: border-box; }

        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: var(--bg-dark);
            color: var(--text);
            min-height: 100vh;
        }

        /* Header */
        .header {
            background: var(--bg-card);
            padding: 0.75rem 1.5rem;
            display: flex;
            justify-content: space-between;
            align-items: center;
            border-bottom: 1px solid var(--border);
            position: sticky;
            top: 0;
            z-index: 100;
        }

        .header h1 {
            color: var(--accent);
            font-size: 1.25rem;
            font-weight: 600;
        }

        .status-bar {
            display: flex;
            gap: 1.5rem;
            align-items: center;
            font-size: 0.85rem;
        }

        .status-item {
            display: flex;
            align-items: center;
            gap: 0.5rem;
        }

        .status-dot {
            width: 8px;
            height: 8px;
            border-radius: 50%;
            background: #f85149;
        }

        .status-dot.connected { background: var(--accent-green); }

        .embryo-filter {
            background: var(--bg-hover);
            border: 1px solid var(--border);
            border-radius: 6px;
            padding: 0.35rem 0.75rem;
            color: var(--text);
            font-size: 0.85rem;
            cursor: pointer;
        }

        /* Tabs */
        .tabs {
            display: flex;
            gap: 0;
            background: var(--bg-card);
            border-bottom: 1px solid var(--border);
            padding: 0 1.5rem;
        }

        .tab {
            padding: 0.75rem 1.25rem;
            cursor: pointer;
            color: var(--text-muted);
            border-bottom: 2px solid transparent;
            transition: all 0.2s;
            font-size: 0.9rem;
        }

        .tab:hover {
            color: var(--text);
            background: var(--bg-hover);
        }

        .tab.active {
            color: var(--accent);
            border-bottom-color: var(--accent);
        }

        .tab-badge {
            background: var(--bg-hover);
            padding: 0.1rem 0.5rem;
            border-radius: 10px;
            font-size: 0.75rem;
            margin-left: 0.5rem;
        }

        /* Main Layout */
        .main {
            display: grid;
            grid-template-columns: 1fr 280px;
            height: calc(100vh - 100px);
        }

        /* Viewer */
        .viewer {
            padding: 1rem;
            display: flex;
            flex-direction: column;
            gap: 1rem;
        }

        .viewer-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
        }

        .image-title {
            font-size: 0.9rem;
            color: var(--text-muted);
        }

        .image-container {
            flex: 1;
            background: #000;
            border-radius: 8px;
            display: flex;
            align-items: center;
            justify-content: center;
            overflow: hidden;
            min-height: 500px;
        }

        .image-container img {
            width: 100%;
            height: 100%;
            object-fit: contain;
        }

        .placeholder {
            color: var(--text-muted);
            font-size: 1rem;
        }

        /* Sidebar */
        .sidebar {
            background: var(--bg-card);
            border-left: 1px solid var(--border);
            display: flex;
            flex-direction: column;
            overflow: hidden;
        }

        .panel {
            padding: 1rem;
            border-bottom: 1px solid var(--border);
        }

        .panel-title {
            color: var(--text-muted);
            font-size: 0.75rem;
            text-transform: uppercase;
            letter-spacing: 0.05em;
            margin-bottom: 0.75rem;
        }

        .info-grid {
            display: grid;
            grid-template-columns: auto 1fr;
            gap: 0.5rem 1rem;
            font-size: 0.85rem;
        }

        .info-label { color: var(--text-muted); }
        .info-value { color: var(--text); }
        .info-value.accent { color: var(--accent-green); font-weight: 500; }

        /* Event Log */
        .event-log {
            flex: 1;
            overflow-y: auto;
            padding: 0.5rem 1rem;
        }

        .event-item {
            padding: 0.5rem 0;
            border-bottom: 1px solid var(--border);
            font-size: 0.8rem;
        }

        .event-time { color: var(--text-muted); margin-right: 0.5rem; }
        .event-type { color: var(--accent); }
        .event-type.cv-event { color: #7C3AED; font-weight: 500; }  /* Purple for CV/AI events */

        /* Gallery Grid */
        .gallery {
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(140px, 1fr));
            gap: 0.75rem;
            padding: 1rem;
            overflow-y: auto;
        }

        .gallery-item {
            background: var(--bg-card);
            border: 1px solid var(--border);
            border-radius: 8px;
            overflow: hidden;
            cursor: pointer;
            transition: all 0.2s;
        }

        .gallery-item:hover {
            border-color: var(--accent);
            transform: translateY(-2px);
        }

        .gallery-item.selected {
            border-color: var(--accent);
            box-shadow: 0 0 0 1px var(--accent);
        }

        .gallery-img {
            width: 100%;
            aspect-ratio: 1;
            object-fit: cover;
            background: #000;
        }

        .gallery-info {
            padding: 0.5rem;
            font-size: 0.75rem;
        }

        .gallery-type {
            color: var(--accent);
            font-weight: 500;
        }

        .gallery-meta {
            color: var(--text-muted);
            margin-top: 0.25rem;
        }

        /* Tab Content */
        .tab-content {
            display: none;
            height: 100%;
        }

        .tab-content.active {
            display: flex;
        }

        /* Volumes/Calibration full gallery */
        #volumes-content, #calibration-content {
            flex-direction: column;
        }

        .empty-state {
            display: flex;
            align-items: center;
            justify-content: center;
            height: 200px;
            color: var(--text-muted);
            font-size: 0.9rem;
        }

        /* Scrollbar */
        ::-webkit-scrollbar { width: 8px; }
        ::-webkit-scrollbar-track { background: var(--bg-dark); }
        ::-webkit-scrollbar-thumb { background: var(--border); border-radius: 4px; }
        ::-webkit-scrollbar-thumb:hover { background: var(--text-muted); }

        /* Z-Slider for 3D volumes */
        .z-slider-container {
            display: none;
            flex-direction: column;
            align-items: center;
            padding: 0.5rem;
            background: var(--bg-card);
            border-left: 1px solid var(--border);
            min-width: 50px;
        }

        .z-slider-container.active {
            display: flex;
        }

        .z-slider-label {
            font-size: 0.7rem;
            color: var(--text-muted);
            margin-bottom: 0.5rem;
            text-transform: uppercase;
        }

        .z-slider-value {
            font-size: 0.85rem;
            color: var(--accent);
            font-weight: 600;
            margin-bottom: 0.5rem;
        }

        .z-slider {
            -webkit-appearance: slider-vertical;
            writing-mode: bt-lr;
            width: 20px;
            flex: 1;
            min-height: 200px;
            cursor: pointer;
        }

        /* Firefox vertical slider */
        @-moz-document url-prefix() {
            .z-slider {
                width: 20px;
            }
        }

        .z-slider-info {
            font-size: 0.7rem;
            color: var(--text-muted);
            margin-top: 0.5rem;
            text-align: center;
        }

        /* Image container needs to accommodate slider */
        .viewer-with-slider {
            display: flex;
            flex: 1;
            overflow: hidden;
        }

        .viewer-with-slider .image-container {
            flex: 1;
        }
    </style>
</head>
<body>
    <div class="header">
        <h1>Gently Microscopy</h1>
        <div class="status-bar">
            <div class="status-item">
                <span id="current-embryo" style="color: var(--accent-green); font-weight: 500;"></span>
            </div>
            <select class="embryo-filter" id="embryo-filter">
                <option value="">All Embryos</option>
            </select>
            <div class="status-item">
                <span id="status-text">Connecting...</span>
                <div class="status-dot" id="status-dot"></div>
            </div>
        </div>
    </div>

    <div class="tabs">
        <div class="tab active" data-tab="main">
            Main
            <span class="tab-badge" id="main-count">0</span>
        </div>
        <div class="tab" data-tab="volumes">
            Volumes
            <span class="tab-badge" id="volumes-count">0</span>
        </div>
        <div class="tab" data-tab="calibration">
            Calibration
            <span class="tab-badge" id="calibration-count">0</span>
        </div>
    </div>

    <!-- Main Tab -->
    <div id="main-content" class="tab-content active">
        <div class="main">
            <div class="viewer">
                <div class="viewer-header">
                    <span class="image-title" id="image-info">No image selected</span>
                    <span class="image-title" id="image-time"></span>
                </div>
                <div class="viewer-with-slider">
                    <div class="image-container">
                        <img id="main-image" style="display: none;">
                        <div class="placeholder" id="placeholder">Waiting for images...</div>
                    </div>
                    <div class="z-slider-container" id="z-slider-container">
                        <div class="z-slider-label">Z</div>
                        <div class="z-slider-value" id="z-slider-value">0</div>
                        <input type="range" class="z-slider" id="z-slider" min="0" max="0" value="0" orient="vertical">
                        <div class="z-slider-info" id="z-slider-info">0 / 0</div>
                    </div>
                </div>
            </div>

            <div class="sidebar">
                <div class="panel">
                    <div class="panel-title">Image Info</div>
                    <div class="info-grid">
                        <span class="info-label">Embryo</span>
                        <span class="info-value accent" id="info-embryo">-</span>
                        <span class="info-label">Type</span>
                        <span class="info-value" id="info-type">-</span>
                        <span class="info-label">Shape</span>
                        <span class="info-value" id="info-shape">-</span>
                        <span class="info-label">UID</span>
                        <span class="info-value" id="info-uid">-</span>
                    </div>
                </div>

                <div class="panel">
                    <div class="panel-title">Recent Images</div>
                    <div id="recent-list" style="max-height: 200px; overflow-y: auto;"></div>
                </div>

                <div class="panel" style="flex: 1; display: flex; flex-direction: column; overflow: hidden;">
                    <div class="panel-title">Event Log</div>
                    <div class="event-log" id="event-log"></div>
                </div>
            </div>
        </div>
    </div>

    <!-- Volumes Tab -->
    <div id="volumes-content" class="tab-content">
        <div class="gallery" id="volumes-gallery">
            <div class="empty-state">No volume images yet</div>
        </div>
    </div>

    <!-- Calibration Tab -->
    <div id="calibration-content" class="tab-content">
        <div style="padding: 0.5rem 1rem; color: var(--accent); font-weight: 500;">3D Segmentations</div>
        <div class="gallery" id="volumes3d-gallery" style="max-height: 200px; min-height: 80px;">
            <div class="empty-state">No 3D segmentations yet</div>
        </div>
        <div style="padding: 0.5rem 1rem; color: var(--accent); font-weight: 500; border-top: 1px solid var(--border);">Calibration Images</div>
        <div class="gallery" id="calibration-gallery">
            <div class="empty-state">No calibration images yet</div>
        </div>
    </div>

    <script>
        // State
        const state = {
            ws: null,
            connected: false,
            tab: 'main',
            embryoFilter: '',
            snapshots: [],
            volumes: [],
            calibration: [],
            embryos: [],
            volumes3d: [],
            currentImage: null,
            current3dVolume: null,  // Currently displayed 3D volume
            currentZ: 0
        };

        // Data type classification
        const CALIBRATION_TYPES = ['focus_sweep', 'focus_plot', 'edge_detection', 'calibration_summary',
                                   'focus_snap', 'focus_coarse', 'focus_curve', 'focus_assess'];
        const ANALYSIS_TYPES = ['segmentation', 'detection', 'classification', 'tracking',
                                'roi_detection', 'cropped_roi', 'vision_prepared', 'timeline', 'cv_visualization'];
        const VOLUME_TYPES = ['volume', 'volume_projection', 'z_stack', 'timelapse'];

        // Connect WebSocket
        function connect() {
            const protocol = location.protocol === 'https:' ? 'wss:' : 'ws:';
            state.ws = new WebSocket(`${protocol}//${location.host}/ws`);

            state.ws.onopen = () => {
                state.connected = true;
                document.getElementById('status-text').textContent = 'Connected';
                document.getElementById('status-dot').classList.add('connected');
                logEvent('system', 'Connected to server');

                // Request initial data
                state.ws.send(JSON.stringify({type: 'get_embryos'}));
                state.ws.send(JSON.stringify({type: 'get_snapshots'}));
                state.ws.send(JSON.stringify({type: 'get_volumes'}));
                state.ws.send(JSON.stringify({type: 'get_calibration'}));
            };

            state.ws.onclose = () => {
                state.connected = false;
                document.getElementById('status-text').textContent = 'Disconnected';
                document.getElementById('status-dot').classList.remove('connected');
                logEvent('system', 'Disconnected');
                setTimeout(connect, 3000);
            };

            state.ws.onerror = () => logEvent('error', 'Connection error');

            state.ws.onmessage = (event) => {
                const msg = JSON.parse(event.data);
                handleMessage(msg);
            };
        }

        function handleMessage(msg) {
            if (msg.type === 'image') {
                handleNewImage(msg.data);
            } else if (msg.type === 'volume_3d') {
                handleNew3DVolume(msg.data);
            } else if (msg.type === 'snapshots') {
                state.snapshots = msg.data || [];
                updateMainCount();
                renderRecentList();
            } else if (msg.type === 'volumes') {
                state.volumes = msg.data || [];
                updateVolumesCount();
                if (state.tab === 'volumes') renderVolumesGallery();
            } else if (msg.type === 'calibration') {
                state.calibration = msg.data || [];
                updateCalibrationCount();
                if (state.tab === 'calibration') renderCalibrationGallery();
            } else if (msg.type === 'embryos') {
                state.embryos = msg.data || [];
                updateEmbryoFilter();
            } else if (msg.type === 'event') {
                // Format CV events nicely
                let eventMsg;
                if (msg.event_type === 'CV_AGENT_THINKING') {
                    const thinking = msg.data.thinking || '';
                    const preview = thinking.length > 40 ? thinking.slice(0, 40) + '...' : thinking;
                    eventMsg = `iter ${msg.data.iteration}: ${preview}`;
                } else if (msg.event_type === 'CV_TASK_QUEUED') {
                    eventMsg = `${msg.data.intent} (${msg.data.embryo_id})`;
                } else if (msg.event_type === 'CV_TASK_COMPLETED') {
                    eventMsg = `${msg.data.intent} done in ${(msg.data.processing_time_ms/1000).toFixed(1)}s`;
                } else if (msg.event_type === 'CV_TASK_FAILED') {
                    eventMsg = `${msg.data.intent} failed: ${msg.data.error?.slice(0, 30) || 'unknown'}`;
                } else {
                    eventMsg = JSON.stringify(msg.data).slice(0, 50);
                }
                logEvent(msg.event_type, eventMsg);
            } else if (msg.type === 'ping') {
                state.ws.send(JSON.stringify({type: 'pong'}));
            }
        }

        function handleNew3DVolume(data) {
            // Add to volumes3d list
            state.volumes3d.push(data);
            logEvent('segmentation', `3D: ${data.num_cells} cells, ${data.num_slices} slices`);

            // Update counts and galleries
            updateCalibrationCount();
            if (state.tab === 'calibration') {
                render3DVolumesGallery();
            }

            // Auto-display the new 3D volume
            if (state.tab === 'main') {
                display3DVolume(data);
            }
        }

        function display3DVolume(data) {
            state.current3dVolume = data;
            state.currentZ = Math.floor(data.num_slices / 2);  // Start in middle

            // Setup slider
            const slider = document.getElementById('z-slider');
            const sliderContainer = document.getElementById('z-slider-container');

            slider.min = 0;
            slider.max = data.num_slices - 1;
            slider.value = state.currentZ;

            // Show slider
            sliderContainer.classList.add('active');

            // Update info
            updateZSliderDisplay();

            // Load the slice
            loadZSlice(data.uid, state.currentZ);

            // Update image info
            document.getElementById('info-type').textContent = '3D Segmentation';
            document.getElementById('info-shape').textContent = data.shape.join(' × ');
            document.getElementById('info-uid').textContent = data.uid.slice(0, 16) + '...';
            document.getElementById('info-embryo').textContent = data.metadata?.embryo_id || '-';
            document.getElementById('image-info').textContent = `3D Seg: ${data.num_cells} cells`;
            document.getElementById('image-time').textContent = new Date(data.timestamp).toLocaleTimeString();
        }

        function loadZSlice(uid, z) {
            const img = document.getElementById('main-image');
            const placeholder = document.getElementById('placeholder');

            // Add cache buster to force reload
            img.src = `/api/volumes3d/${uid}/slice/${z}?t=${Date.now()}`;
            img.style.display = 'block';
            placeholder.style.display = 'none';
        }

        function updateZSliderDisplay() {
            const data = state.current3dVolume;
            if (!data) return;

            document.getElementById('z-slider-value').textContent = state.currentZ;
            document.getElementById('z-slider-info').textContent = `${state.currentZ + 1} / ${data.num_slices}`;
        }

        function hideZSlider() {
            document.getElementById('z-slider-container').classList.remove('active');
            state.current3dVolume = null;
        }

        function handleNewImage(data) {
            const dataType = data.data_type;
            const embryoId = data.metadata?.embryo_id;

            // Route to appropriate list
            if (CALIBRATION_TYPES.includes(dataType) || ANALYSIS_TYPES.includes(dataType)) {
                state.calibration.push(data);
                updateCalibrationCount();
                if (state.tab === 'calibration') renderCalibrationGallery();
                const eventType = ANALYSIS_TYPES.includes(dataType) ? 'analysis' : 'calibration';
                logEvent(eventType, `${dataType}${embryoId ? ' ' + embryoId : ''}`);
            } else if (VOLUME_TYPES.includes(dataType)) {
                state.volumes.push(data);
                updateVolumesCount();
                if (state.tab === 'volumes') renderVolumesGallery();
                logEvent('volume', `${dataType}${embryoId ? ' ' + embryoId : ''}`);
            } else {
                state.snapshots.push(data);
                updateMainCount();
                renderRecentList();
                logEvent('image', `${dataType}${embryoId ? ' ' + embryoId : ''}`);
            }

            // Update embryo list if new
            if (embryoId && !state.embryos.includes(embryoId)) {
                state.embryos.push(embryoId);
                updateEmbryoFilter();
            }

            // Show on main viewer if main tab
            if (state.tab === 'main') {
                displayImage(data);
            }
        }

        function displayImage(data) {
            state.currentImage = data;

            // Hide Z slider when showing regular 2D images
            hideZSlider();

            const img = document.getElementById('main-image');
            const placeholder = document.getElementById('placeholder');

            if (data.base64_png) {
                img.src = 'data:image/png;base64,' + data.base64_png;
                img.style.display = 'block';
                placeholder.style.display = 'none';
            }

            const embryoId = data.metadata?.embryo_id || '-';
            document.getElementById('current-embryo').textContent = embryoId !== '-' ? embryoId : '';
            document.getElementById('info-embryo').textContent = embryoId;
            document.getElementById('info-type').textContent = data.data_type;
            document.getElementById('info-shape').textContent = data.shape ? data.shape.join(' × ') : '-';
            document.getElementById('info-uid').textContent = data.uid.slice(0, 16) + '...';
            document.getElementById('image-info').textContent = data.data_type;
            document.getElementById('image-time').textContent = new Date(data.timestamp).toLocaleTimeString();
        }

        function renderRecentList() {
            const list = document.getElementById('recent-list');
            const filtered = filterByEmbryo(state.snapshots);

            list.innerHTML = filtered.slice(-15).reverse().map(img => `
                <div class="gallery-item" style="margin-bottom: 0.5rem;" onclick="displayImage(state.snapshots.find(i => i.uid === '${img.uid}'))">
                    <div class="gallery-info">
                        <div class="gallery-type">${img.data_type}</div>
                        <div class="gallery-meta">${img.uid.slice(0, 8)}...</div>
                    </div>
                </div>
            `).join('');
        }

        function renderVolumesGallery() {
            const gallery = document.getElementById('volumes-gallery');
            const filtered = filterByEmbryo(state.volumes);

            if (filtered.length === 0) {
                gallery.innerHTML = '<div class="empty-state">No volume images yet</div>';
                return;
            }

            gallery.innerHTML = filtered.slice(-50).reverse().map(img => `
                <div class="gallery-item" onclick="showInModal('${img.uid}', 'volumes')">
                    <img class="gallery-img" src="data:image/png;base64,${img.base64_png}" alt="${img.data_type}">
                    <div class="gallery-info">
                        <div class="gallery-type">${img.data_type}</div>
                        <div class="gallery-meta">${img.metadata?.embryo_id || 'unknown'}</div>
                    </div>
                </div>
            `).join('');
        }

        function renderCalibrationGallery() {
            const gallery = document.getElementById('calibration-gallery');
            const filtered = filterByEmbryo(state.calibration);

            if (filtered.length === 0) {
                gallery.innerHTML = '<div class="empty-state">No calibration images yet</div>';
                return;
            }

            gallery.innerHTML = filtered.slice(-50).reverse().map(img => `
                <div class="gallery-item" onclick="showInModal('${img.uid}', 'calibration')">
                    <img class="gallery-img" src="data:image/png;base64,${img.base64_png}" alt="${img.data_type}">
                    <div class="gallery-info">
                        <div class="gallery-type">${img.data_type}</div>
                        <div class="gallery-meta">${img.metadata?.embryo_id || ''} ${formatMeta(img.metadata)}</div>
                    </div>
                </div>
            `).join('');

            // Also render 3D volumes
            render3DVolumesGallery();
        }

        function render3DVolumesGallery() {
            const gallery = document.getElementById('volumes3d-gallery');
            if (!gallery) return;

            if (state.volumes3d.length === 0) {
                gallery.innerHTML = '<div class="empty-state">No 3D segmentations yet</div>';
                return;
            }

            gallery.innerHTML = state.volumes3d.slice(-20).reverse().map(vol => `
                <div class="gallery-item" onclick="show3DVolume('${vol.uid}')" style="min-width: 120px;">
                    <div class="gallery-info" style="padding: 0.75rem; text-align: center;">
                        <div class="gallery-type">3D Seg</div>
                        <div class="gallery-meta" style="font-size: 1.1rem; color: var(--accent-green);">${vol.num_cells} cells</div>
                        <div class="gallery-meta">${vol.num_slices} slices</div>
                        <div class="gallery-meta" style="font-size: 0.65rem;">${vol.uid.slice(0, 12)}...</div>
                    </div>
                </div>
            `).join('');
        }

        function show3DVolume(uid) {
            const vol = state.volumes3d.find(v => v.uid === uid);
            if (vol) {
                display3DVolume(vol);
                switchTab('main');
            }
        }

        function formatMeta(meta) {
            if (!meta) return '';
            if (meta.focus_score) return `score: ${meta.focus_score.toFixed(2)}`;
            if (meta.piezo_um) return `${meta.piezo_um.toFixed(1)}µm`;
            return '';
        }

        function showInModal(uid, source) {
            const list = source === 'volumes' ? state.volumes : state.calibration;
            const img = list.find(i => i.uid === uid);
            if (img) displayImage(img);
            // Switch to main tab to show the image
            switchTab('main');
        }

        function filterByEmbryo(list) {
            if (!state.embryoFilter) return list;
            return list.filter(img => img.metadata?.embryo_id === state.embryoFilter);
        }

        function updateEmbryoFilter() {
            const select = document.getElementById('embryo-filter');
            const currentValue = select.value;
            select.innerHTML = '<option value="">All Embryos</option>' +
                state.embryos.map(e => `<option value="${e}">${e}</option>`).join('');
            select.value = currentValue;
        }

        function updateMainCount() {
            document.getElementById('main-count').textContent = filterByEmbryo(state.snapshots).length;
        }

        function updateVolumesCount() {
            document.getElementById('volumes-count').textContent = filterByEmbryo(state.volumes).length;
        }

        function updateCalibrationCount() {
            const calCount = filterByEmbryo(state.calibration).length;
            const vol3dCount = state.volumes3d.length;
            document.getElementById('calibration-count').textContent = calCount + vol3dCount;
        }

        function switchTab(tabName) {
            state.tab = tabName;

            // Update tab styling
            document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
            document.querySelector(`.tab[data-tab="${tabName}"]`).classList.add('active');

            // Show/hide content
            document.querySelectorAll('.tab-content').forEach(c => c.classList.remove('active'));
            document.getElementById(`${tabName}-content`).classList.add('active');

            // Render galleries
            if (tabName === 'volumes') renderVolumesGallery();
            if (tabName === 'calibration') renderCalibrationGallery();
        }

        function logEvent(type, message) {
            const log = document.getElementById('event-log');
            const div = document.createElement('div');
            div.className = 'event-item';
            // CV events get special styling
            const isCvEvent = type.startsWith('CV_') || type === 'SEGMENTATION_COMPLETED' || type === 'STAGE_DETECTED';
            const typeClass = isCvEvent ? 'event-type cv-event' : 'event-type';
            div.innerHTML = `<span class="event-time">${new Date().toLocaleTimeString()}</span>
                            <span class="${typeClass}">${type}</span>: ${message}`;
            log.insertBefore(div, log.firstChild);
            while (log.children.length > 50) log.removeChild(log.lastChild);
        }

        // Event listeners
        document.querySelectorAll('.tab').forEach(tab => {
            tab.addEventListener('click', () => switchTab(tab.dataset.tab));
        });

        document.getElementById('embryo-filter').addEventListener('change', (e) => {
            state.embryoFilter = e.target.value;
            updateMainCount();
            updateVolumesCount();
            updateCalibrationCount();
            renderRecentList();
            if (state.tab === 'volumes') renderVolumesGallery();
            if (state.tab === 'calibration') renderCalibrationGallery();
        });

        // Z-slider event listener
        document.getElementById('z-slider').addEventListener('input', (e) => {
            if (!state.current3dVolume) return;
            state.currentZ = parseInt(e.target.value);
            updateZSliderDisplay();
            loadZSlice(state.current3dVolume.uid, state.currentZ);
        });

        // Start
        connect();
    </script>
</body>
</html>'''


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
