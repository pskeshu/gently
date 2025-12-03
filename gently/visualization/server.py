"""
Visualization Server for Gently Microscopy System
==================================================

Web-based visualization server providing:
- REST API for image retrieval
- WebSocket streaming for real-time updates
- Integration with EventBus for automatic notifications
- Simple web UI for viewing microscope data

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
    from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
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


@dataclass
class ImageData:
    """Container for image data sent to clients"""
    uid: str
    data_type: str  # 'volume', 'projection', 'snapshot', 'detection'
    timestamp: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    base64_png: Optional[str] = None
    shape: Optional[tuple] = None

    def to_dict(self) -> Dict:
        return asdict(self)


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
    - EventBus integration for automatic notifications
    - Simple web UI

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

        # Image cache for recent images
        self._image_cache: Dict[str, ImageData] = {}
        self._cache_limit = 100

        # Create FastAPI app
        self.app = FastAPI(
            title="Gently Visualization Server",
            description="Real-time microscopy visualization",
            version="1.0.0"
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
            return {
                "status": "running",
                "connections": len(self.manager.active_connections),
                "cached_images": len(self._image_cache),
                "timestamp": datetime.now().isoformat()
            }

        @self.app.get("/api/images")
        async def list_images():
            """List cached images"""
            return {
                "images": [
                    {
                        "uid": uid,
                        "data_type": img.data_type,
                        "timestamp": img.timestamp,
                        "shape": img.shape,
                    }
                    for uid, img in self._image_cache.items()
                ]
            }

        @self.app.get("/api/images/{uid}")
        async def get_image(uid: str):
            """Get image by UID"""
            # Check cache first
            if uid in self._image_cache:
                return self._image_cache[uid].to_dict()

            # Try data store
            if self.data_store:
                try:
                    data = self.data_store.retrieve(uid)
                    if data is not None:
                        image_data = self._array_to_image_data(
                            data, uid, "retrieved"
                        )
                        return image_data.to_dict()
                except Exception as e:
                    logger.error(f"Failed to retrieve {uid}: {e}")

            raise HTTPException(status_code=404, detail=f"Image {uid} not found")

        @self.app.get("/api/images/{uid}/png")
        async def get_image_png(uid: str):
            """Get image as PNG"""
            if uid in self._image_cache:
                img_data = self._image_cache[uid]
                if img_data.base64_png:
                    png_bytes = base64.b64decode(img_data.base64_png)
                    return Response(content=png_bytes, media_type="image/png")

            raise HTTPException(status_code=404, detail=f"Image {uid} not found")

        @self.app.websocket("/ws")
        async def websocket_endpoint(websocket: WebSocket):
            """WebSocket endpoint for real-time updates"""
            await self.manager.connect(websocket)
            try:
                # Send current status on connect
                await websocket.send_json({
                    "type": "connected",
                    "cached_images": len(self._image_cache),
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
                    pass  # Ignore errors during disconnect
            except asyncio.CancelledError:
                # Graceful shutdown - don't log as error
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

            if msg_type == "get_image":
                uid = data.get("uid")
                if uid and uid in self._image_cache:
                    await websocket.send_json({
                        "type": "image",
                        "data": self._image_cache[uid].to_dict()
                    })

            elif msg_type == "get_latest":
                # Send most recent image
                if self._image_cache:
                    latest = list(self._image_cache.values())[-1]
                    await websocket.send_json({
                        "type": "image",
                        "data": latest.to_dict()
                    })

            elif msg_type == "pong":
                pass  # Client responding to ping

        except json.JSONDecodeError:
            logger.warning(f"Invalid JSON received: {message[:100]}")

    def _subscribe_to_events(self):
        """Subscribe to EventBus for automatic updates"""
        from ..core import EventType

        # Events to broadcast to clients
        events_to_broadcast = [
            EventType.VOLUME_ACQUIRED,
            EventType.DETECTION_TRIGGERED,
            EventType.HATCHING_DETECTED,
            EventType.EMBRYO_CENTERED,
            EventType.ANALYSIS_COMPLETED,
            EventType.SESSION_STARTED,
            EventType.SESSION_RESTORED,
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
            # Check if it's an RGB/RGBA image (last dim is 3 or 4)
            if array.shape[2] in (3, 4):
                # It's an RGB(A) image - keep as is, PIL will handle it
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
            Type of image (volume, projection, snapshot, etc.)
        metadata : dict, optional
            Additional metadata
        """
        image_data = self._array_to_image_data(array, uid, data_type, metadata)

        # Add to cache
        self._image_cache[uid] = image_data

        # Trim cache if needed
        while len(self._image_cache) > self._cache_limit:
            oldest_uid = next(iter(self._image_cache))
            del self._image_cache[oldest_uid]

        # Broadcast to clients
        await self.manager.send_image(image_data)

        logger.debug(f"Pushed image {uid} to {len(self.manager.active_connections)} clients")

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
        """
        Run the server until interrupted.

        This method handles Ctrl+C properly on both Windows and Unix.
        Use this instead of start() + run_forever() pattern.
        """
        import signal
        import sys

        stop_event = asyncio.Event()

        def signal_handler(*args):
            logger.info("Received shutdown signal")
            stop_event.set()

        # Setup signal handlers
        loop = asyncio.get_running_loop()

        # Try asyncio signal handlers first (works on Unix)
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

        # On Windows, use traditional signal handler as fallback
        if sys.platform == 'win32' and not signals_installed:
            signal.signal(signal.SIGINT, signal_handler)
            signal.signal(signal.SIGTERM, signal_handler)

        # Start the server
        await self.start()

        logger.info(f"Server running at http://{self.host}:{self.port} - Press Ctrl+C to stop")

        # Wait for stop signal or server task completion
        try:
            # On Windows, we need to periodically check stop_event
            # because signal handlers may not wake up the event loop
            if sys.platform == 'win32':
                while not stop_event.is_set():
                    try:
                        # Check every 0.5 seconds if we should stop
                        await asyncio.wait_for(
                            asyncio.shield(self._server_task),
                            timeout=0.5
                        )
                        # Server task completed (crashed or stopped)
                        break
                    except asyncio.TimeoutError:
                        # Check if stop was requested
                        continue
            else:
                # On Unix, wait for either server task or stop event
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
        """Generate the main HTML page"""
        return '''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Gently - Microscopy Visualization</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: #1a1a2e;
            color: #eee;
            min-height: 100vh;
        }
        .header {
            background: #16213e;
            padding: 1rem 2rem;
            display: flex;
            justify-content: space-between;
            align-items: center;
            border-bottom: 1px solid #0f3460;
        }
        .header h1 {
            color: #00d9ff;
            font-size: 1.5rem;
        }
        .status {
            display: flex;
            gap: 1rem;
            align-items: center;
        }
        .status-dot {
            width: 10px;
            height: 10px;
            border-radius: 50%;
            background: #ff4444;
        }
        .status-dot.connected { background: #00ff88; }
        .main {
            display: grid;
            grid-template-columns: 1fr 300px;
            gap: 1rem;
            padding: 1rem;
            height: calc(100vh - 60px);
        }
        .viewer {
            background: #0f0f23;
            border-radius: 8px;
            padding: 1rem;
            display: flex;
            flex-direction: column;
        }
        .viewer-header {
            display: flex;
            justify-content: space-between;
            margin-bottom: 1rem;
        }
        .image-container {
            flex: 1;
            display: flex;
            align-items: center;
            justify-content: center;
            background: #000;
            border-radius: 4px;
            overflow: hidden;
        }
        .image-container img {
            max-width: 100%;
            max-height: 100%;
            object-fit: contain;
        }
        .placeholder {
            color: #666;
            font-size: 1.2rem;
        }
        .sidebar {
            display: flex;
            flex-direction: column;
            gap: 1rem;
        }
        .panel {
            background: #0f0f23;
            border-radius: 8px;
            padding: 1rem;
        }
        .panel h3 {
            color: #00d9ff;
            margin-bottom: 0.5rem;
            font-size: 0.9rem;
            text-transform: uppercase;
        }
        .event-log {
            flex: 1;
            overflow-y: auto;
            max-height: 300px;
        }
        .event-item {
            padding: 0.5rem;
            border-bottom: 1px solid #1a1a2e;
            font-size: 0.85rem;
        }
        .event-item .time {
            color: #666;
            font-size: 0.75rem;
        }
        .event-item .type {
            color: #00d9ff;
        }
        .image-list {
            max-height: 200px;
            overflow-y: auto;
        }
        .image-item {
            padding: 0.5rem;
            cursor: pointer;
            border-radius: 4px;
            margin-bottom: 0.25rem;
        }
        .image-item:hover {
            background: #1a1a2e;
        }
        .image-item.selected {
            background: #0f3460;
        }
        .info-row {
            display: flex;
            justify-content: space-between;
            padding: 0.25rem 0;
            font-size: 0.85rem;
        }
        .info-label { color: #888; }
        .info-value { color: #fff; }
    </style>
</head>
<body>
    <div class="header">
        <h1>Gently Microscopy</h1>
        <div class="status">
            <span id="current-embryo" style="color:#00ff88;font-weight:bold;margin-right:1rem;"></span>
            <span id="status-text">Disconnected</span>
            <div class="status-dot" id="status-dot"></div>
        </div>
    </div>

    <div class="main">
        <div class="viewer">
            <div class="viewer-header">
                <span id="image-info">No image</span>
                <span id="image-time"></span>
            </div>
            <div class="image-container">
                <img id="main-image" style="display:none;">
                <div class="placeholder" id="placeholder">Waiting for images...</div>
            </div>
        </div>

        <div class="sidebar">
            <div class="panel">
                <h3>Image Info</h3>
                <div id="image-details">
                    <div class="info-row">
                        <span class="info-label">Embryo:</span>
                        <span class="info-value" id="info-embryo" style="color:#00ff88;">-</span>
                    </div>
                    <div class="info-row">
                        <span class="info-label">UID:</span>
                        <span class="info-value" id="info-uid">-</span>
                    </div>
                    <div class="info-row">
                        <span class="info-label">Type:</span>
                        <span class="info-value" id="info-type">-</span>
                    </div>
                    <div class="info-row">
                        <span class="info-label">Shape:</span>
                        <span class="info-value" id="info-shape">-</span>
                    </div>
                </div>
            </div>

            <div class="panel">
                <h3>Recent Images</h3>
                <div class="image-list" id="image-list"></div>
            </div>

            <div class="panel" style="flex:1;">
                <h3>Event Log</h3>
                <div class="event-log" id="event-log"></div>
            </div>
        </div>
    </div>

    <script>
        let ws;
        let images = {};
        let currentUid = null;

        function connect() {
            const protocol = location.protocol === 'https:' ? 'wss:' : 'ws:';
            ws = new WebSocket(`${protocol}//${location.host}/ws`);

            ws.onopen = () => {
                document.getElementById('status-text').textContent = 'Connected';
                document.getElementById('status-dot').classList.add('connected');
                logEvent('system', 'Connected to server');
            };

            ws.onclose = () => {
                document.getElementById('status-text').textContent = 'Disconnected';
                document.getElementById('status-dot').classList.remove('connected');
                logEvent('system', 'Disconnected from server');
                setTimeout(connect, 3000);
            };

            ws.onerror = (error) => {
                logEvent('error', 'WebSocket error');
            };

            ws.onmessage = (event) => {
                const msg = JSON.parse(event.data);
                handleMessage(msg);
            };
        }

        function handleMessage(msg) {
            if (msg.type === 'image') {
                const data = msg.data;
                images[data.uid] = data;
                updateImageList();
                displayImage(data);
                logEvent('image', `New ${data.data_type}: ${data.uid.slice(0,8)}...`);
            } else if (msg.type === 'event') {
                logEvent(msg.event_type, JSON.stringify(msg.data).slice(0,50));
            } else if (msg.type === 'ping') {
                ws.send(JSON.stringify({type: 'pong'}));
            }
        }

        function displayImage(data) {
            currentUid = data.uid;
            const img = document.getElementById('main-image');
            const placeholder = document.getElementById('placeholder');

            if (data.base64_png) {
                img.src = 'data:image/png;base64,' + data.base64_png;
                img.style.display = 'block';
                placeholder.style.display = 'none';
            }

            // Update embryo display
            const embryoId = data.metadata?.embryo_id || '-';
            document.getElementById('current-embryo').textContent = embryoId !== '-' ? embryoId : '';
            document.getElementById('info-embryo').textContent = embryoId;

            document.getElementById('image-info').textContent = data.data_type;
            document.getElementById('image-time').textContent = new Date(data.timestamp).toLocaleTimeString();
            document.getElementById('info-uid').textContent = data.uid.slice(0,12) + '...';
            document.getElementById('info-type').textContent = data.data_type;
            document.getElementById('info-shape').textContent = data.shape ? data.shape.join(' x ') : '-';

            // Update selection
            document.querySelectorAll('.image-item').forEach(el => {
                el.classList.toggle('selected', el.dataset.uid === data.uid);
            });
        }

        function updateImageList() {
            const list = document.getElementById('image-list');
            list.innerHTML = '';

            Object.values(images).slice(-20).reverse().forEach(img => {
                const div = document.createElement('div');
                div.className = 'image-item' + (img.uid === currentUid ? ' selected' : '');
                div.dataset.uid = img.uid;
                div.innerHTML = `
                    <div>${img.data_type}</div>
                    <div style="color:#666;font-size:0.75rem">${img.uid.slice(0,8)}...</div>
                `;
                div.onclick = () => displayImage(img);
                list.appendChild(div);
            });
        }

        function logEvent(type, message) {
            const log = document.getElementById('event-log');
            const div = document.createElement('div');
            div.className = 'event-item';
            div.innerHTML = `
                <span class="time">${new Date().toLocaleTimeString()}</span>
                <span class="type">${type}</span>: ${message}
            `;
            log.insertBefore(div, log.firstChild);

            // Keep only last 50 events
            while (log.children.length > 50) {
                log.removeChild(log.lastChild);
            }
        }

        // Start connection
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
