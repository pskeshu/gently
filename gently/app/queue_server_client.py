"""
Queue Server Client for MicroscopyAgent

Provides async interface to the Gently Device Layer:
- Hardware control via Bluesky plans (HTTP API)
- SAM embryo detection (HTTP API)

This client connects to the unified device_layer.py server which provides
both hardware control and SAM detection in a single HTTP process.
"""

import asyncio
import logging
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import aiohttp

logger = logging.getLogger(__name__)

from gently.core.coordinates import (
    pixel_to_stage_position,
    stage_to_pixel_position,
    get_um_per_pixel,
    DEFAULT_PIXEL_SIZE_UM,
    DEFAULT_OBJECTIVE_MAG,
)
from gently.settings import settings
from gently.exceptions import DeviceLayerError, NetworkError, AcquisitionError


class QueueServerClient:
    """
    Client for Gently Device Layer Server.

    Provides async methods for:
    - Submitting Bluesky plans to the server
    - Waiting for plan completion
    - Retrieving results
    - Running SAM embryo detection (via HTTP, same server)

    Example
    -------
    >>> client = QueueServerClient(http_url="http://127.0.0.1:60610")
    >>> await client.connect()
    >>>
    >>> # Move stage
    >>> await client.move_to_position(1000.0, 500.0)
    >>>
    >>> # Detect embryos
    >>> results = await client.capture_and_detect_embryos()
    >>>
    >>> # Acquire volume
    >>> volume_data = await client.acquire_volume(num_slices=50)
    """

    def __init__(
        self,
        http_url: str = f"http://{settings.network.device_host}:{settings.network.device_port}",
    ):
        """
        Parameters
        ----------
        http_url : str
            Device Layer HTTP API URL (provides both hardware and SAM)
        """
        self.http_url = http_url

        self._session = None  # aiohttp session
        self._qs_connected = False  # Track actual queue server connection
        self._sam_available = False  # Track SAM availability via HTTP

        # Store last detection results for visualization
        self._last_detection = None  # {image, embryos, stage_position}

    async def connect(self) -> bool:
        """
        Connect to Device Layer Server.

        Returns
        -------
        bool
            True if connection successful
        """
        import aiohttp

        self._qs_connected = False
        self._sam_available = False

        # Create aiohttp session
        self._session = aiohttp.ClientSession()

        # Connect to Device Layer Server (HTTP API)
        try:
            async with self._session.get(f"{self.http_url}/api/status") as resp:
                if resp.status == 200:
                    await resp.json()  # Validate response
                    self._qs_connected = True
                else:
                    raise NetworkError(f"HTTP {resp.status}")
        except (aiohttp.ClientError, NetworkError):
            self._qs_connected = False

        # Check SAM availability (via HTTP, same server)
        if self._qs_connected:
            try:
                async with self._session.get(f"{self.http_url}/api/sam/status") as resp:
                    if resp.status == 200:
                        sam_status = await resp.json()
                        self._sam_available = sam_status.get('available', False)
            except aiohttp.ClientError:
                self._sam_available = False

        return self._qs_connected

    async def disconnect(self):
        """Disconnect from server"""
        if self._session:
            await self._session.close()
            self._session = None
        self._sam_available = False

    @property
    def is_connected(self) -> bool:
        """Check if connected to Device Layer Server"""
        return self._qs_connected

    @property
    def has_sam(self) -> bool:
        """Check if SAM detection is available (via HTTP)"""
        return self._sam_available

    def _ensure_connected(self):
        """Raise error if not connected"""
        if not self.is_connected:
            raise ConnectionError(
                "Not connected to Microscope Server. Call connect() first."
            )

    # =========================================================================
    # Session Configuration (GentlyStore integration)
    # =========================================================================

    async def configure_device_session(self, volume_dir: str) -> dict:
        """Tell the device server where to write staging TIFFs.

        After this call, the server's ``serialize_value()`` will write large
        numpy arrays as TIFF files into *volume_dir* and return a lightweight
        file-reference dict instead of serializing the full data to JSON.

        Parameters
        ----------
        volume_dir : str
            Absolute path to the staging directory
            (e.g. ``"D:/Gently2/incoming"``).

        Returns
        -------
        dict
            ``{"success": True, "volume_dir": "..."}`` on success.
        """
        self._ensure_connected()
        async with self._session.post(
            f"{self.http_url}/session/configure",
            json={"volume_dir": volume_dir},
        ) as resp:
            return await resp.json()

    @staticmethod
    def _is_file_ref(obj) -> bool:
        """Check if *obj* is a file-reference dict from the device server."""
        return isinstance(obj, dict) and obj.get("__file_ref__") is True

    @staticmethod
    def _resolve_file_ref(ref: dict) -> tuple:
        """Read a single file reference and return ``(np.ndarray, Path)``.

        Parameters
        ----------
        ref : dict
            ``{"__file_ref__": True, "path": "...", "shape": [...], "dtype": "..."}``

        Returns
        -------
        tuple of (np.ndarray, Path)
        """
        import tifffile
        from pathlib import Path

        path = Path(ref["path"])
        arr = tifffile.imread(str(path))
        return arr, path

    @classmethod
    def _resolve_file_refs(cls, data: dict) -> dict:
        """Walk *data* and replace every file-ref dict with its numpy array.

        Also stores resolved file paths in ``data["__resolved_paths__"]``
        for downstream use (e.g. ``register_volume``).
        """
        resolved_paths = {}

        def _walk(obj, key_path=""):
            if cls._is_file_ref(obj):
                arr, path = cls._resolve_file_ref(obj)
                resolved_paths[key_path] = path
                return arr
            if isinstance(obj, dict):
                return {k: _walk(v, f"{key_path}.{k}") for k, v in obj.items()}
            if isinstance(obj, list):
                return [_walk(v, f"{key_path}[{i}]") for i, v in enumerate(obj)]
            return obj

        result = _walk(data)
        if isinstance(result, dict):
            result["__resolved_paths__"] = resolved_paths
        return result

    # =========================================================================
    # Server Operations
    # =========================================================================

    async def _submit_plan_and_wait(
        self,
        plan_name: str,
        kwargs: Dict = None,
        timeout: float = settings.timeouts.plan_execution
    ) -> Dict:
        """
        Submit a plan and wait for completion.

        The simple_server runs plans synchronously - the POST request blocks
        until the plan completes and returns the full result directly.
        No polling needed!

        Parameters
        ----------
        plan_name : str
            Name of the plan function
        kwargs : dict
            Plan keyword arguments
        timeout : float
            Maximum wait time in seconds

        Returns
        -------
        dict
            Result with 'success', 'run_uid', 'documents', and any error info
        """
        self._ensure_connected()
        kwargs = kwargs or {}

        try:
            # Submit plan via HTTP POST - server runs it synchronously and returns result
            payload = {
                'item': {
                    'name': plan_name,
                    'kwargs': kwargs
                }
            }

            async with self._session.post(
                f"{self.http_url}/api/queue/item/add",
                json=payload,
                timeout=timeout  # Use full timeout for plan execution
            ) as resp:
                result = await resp.json()

                if result.get('success', False):
                    # Extract run UID from response
                    run_uid = (
                        result.get('uid') or
                        result.get('documents', {}).get('start', {}).get('uid')
                    )
                    return {
                        'success': True,
                        'run_uid': run_uid,
                        'documents': result.get('documents', {}),
                    }
                else:
                    return {
                        'success': False,
                        'error': result.get('error', result.get('msg', 'Plan failed'))
                    }

        except asyncio.TimeoutError:
            return {'success': False, 'error': f'Plan timed out after {timeout}s'}
        except aiohttp.ClientError as e:
            import traceback
            return {
                'success': False,
                'error': str(NetworkError(f"Device layer request failed: {e}")),
                'traceback': traceback.format_exc()
            }
        except Exception as e:
            import traceback
            return {
                'success': False,
                'error': str(e),
                'traceback': traceback.format_exc()
            }

    # =========================================================================
    # Stage Operations
    # =========================================================================

    async def move_to_position(self, x: float, y: float) -> Dict:
        """
        Move XY stage to position.

        Parameters
        ----------
        x : float
            X position in micrometers
        y : float
            Y position in micrometers

        Returns
        -------
        dict
            Result with new position
        """
        result = await self._submit_plan_and_wait(
            'move_stage_plan',
            kwargs={'xy_stage': 'xy_stage', 'x': x, 'y': y}
        )

        if result.get('success'):
            return {'x': x, 'y': y, 'success': True}
        return {'error': result.get('error', 'Move failed'), 'success': False}

    async def get_stage_position(self) -> Tuple[float, float]:
        """
        Get current stage position.

        Returns
        -------
        tuple
            (x, y) position in micrometers
        """
        result = await self._submit_plan_and_wait(
            'read_stage_plan',
            kwargs={'xy_stage': 'xy_stage'}
        )

        if result.get('success'):
            docs = result.get('documents', {})
            events = docs.get('events', [])
            if events:
                data = events[0].get('data', {})
                # DiSPIMXYStage.read() returns {device_name: [x, y]}
                for key in ['xy_stage', 'XYStage:XY:31', 'xy_stage_position']:
                    if key in data:
                        val = data[key]
                        if isinstance(val, (list, tuple)) and len(val) >= 2:
                            return (float(val[0]), float(val[1]))

        return (0.0, 0.0)

    async def get_piezo_position(self) -> float:
        """
        Get current piezo position.

        Returns
        -------
        float
            Piezo position in micrometers
        """
        result = await self._submit_plan_and_wait(
            'read_piezo_plan',
            kwargs={'piezo': 'piezo'}
        )

        if result.get('success'):
            docs = result.get('documents', {})
            events = docs.get('events', [])
            if events:
                data = events[0].get('data', {})
                # Look for piezo position in various possible keys
                for key in data.keys():
                    if 'piezo' in key.lower():
                        val = data[key]
                        if isinstance(val, (int, float)):
                            return float(val)
                        elif isinstance(val, (list, tuple)) and len(val) > 0:
                            return float(val[0])

        return 0.0

    # =========================================================================
    # Calibration Operations
    # =========================================================================

    async def calibrate_piezo_galvo(
        self,
        piezo_positions: List[float] = None
    ) -> Dict:
        """
        Run piezo-galvo calibration.

        Parameters
        ----------
        piezo_positions : list of float, optional
            Piezo positions to use for calibration

        Returns
        -------
        dict
            Calibration results
        """
        if piezo_positions is None:
            piezo_positions = [40.0, 60.0]

        result = await self._submit_plan_and_wait(
            'calibrate_piezo_galvo_plan',
            kwargs={
                'lightsheet_snap': 'lightsheet_snap',
                'piezo_positions': piezo_positions
            },
            timeout=settings.timeouts.plan_execution
        )

        if result.get('success'):
            docs = result.get('documents', {})
            start = docs.get('start', {})
            return {
                'calibration': start.get('calibration', {}),
                'success': True
            }

        return {'error': result.get('error', 'Calibration failed'), 'success': False}

    async def capture_lightsheet_image(
        self,
        piezo_position: float = 50.0,
        galvo_position: float = 0.0
    ) -> Dict:
        """
        Capture a single lightsheet image at specified piezo/galvo positions.

        Parameters
        ----------
        piezo_position : float
            Piezo position in micrometers. Default: 50.0
        galvo_position : float
            Galvo position in volts. Default: 0.0

        Returns
        -------
        dict
            Contains 'image' (numpy array) and 'success' status
        """
        result = await self._submit_plan_and_wait(
            'capture_lightsheet_image_plan',
            kwargs={
                'lightsheet_snap': 'lightsheet_snap',
                'scanner': 'scanner',
                'piezo': 'piezo',
                'laser_control': 'laser_control',
                'piezo_position': piezo_position,
                'galvo_position': galvo_position
            },
            timeout=settings.timeouts.rpc_call
        )

        if result.get('success'):
            run_uid = result.get('run_uid')
            docs = result.get('documents', {})

            # Try to get image from the response documents first
            events = docs.get('events', [])
            if events:
                data = events[0].get('data', {})
                # Look for image data under various possible keys
                for key in ['HamCam1', 'lightsheet_snap', 'camera']:
                    if key in data:
                        image_data = data[key]
                        # Handle file-ref protocol
                        volume_path = None
                        if self._is_file_ref(image_data):
                            image_data, volume_path = self._resolve_file_ref(image_data)
                        else:
                            image_data = np.array(image_data)
                        ret = {
                            'image': image_data,
                            'piezo_position': piezo_position,
                            'galvo_position': galvo_position,
                            'run_uid': run_uid,
                            'success': True,
                        }
                        if volume_path is not None:
                            ret['volume_path'] = volume_path
                        return ret

            # Plan succeeded but no image data available
            return {
                'image': None,
                'piezo_position': piezo_position,
                'galvo_position': galvo_position,
                'run_uid': run_uid,
                'success': True,
                'note': 'Plan completed but image not in response'
            }

        return {'error': result.get('error', 'Lightsheet snap failed'), 'success': False}

    # =========================================================================
    # Acquisition Operations
    # =========================================================================

    async def acquire_volume(
        self,
        num_slices: int = 50,
        exposure_ms: float = 10.0,
        galvo_amplitude: float = 0.5,
        galvo_center: float = 0.0,
        piezo_amplitude: float = 25.0,
        piezo_center: float = 50.0,
    ) -> Dict:
        """
        Acquire a single volume.

        Parameters
        ----------
        num_slices : int
            Number of Z slices
        exposure_ms : float
            Camera exposure time in milliseconds
        galvo_amplitude : float
            Galvo sweep amplitude in degrees
        galvo_center : float
            Galvo center position in degrees
        piezo_amplitude : float
            Piezo sweep amplitude in micrometers
        piezo_center : float
            Piezo center position in micrometers

        Returns
        -------
        dict
            Results with 'volume' (numpy array) and metadata
        """
        result = await self._submit_plan_and_wait(
            'acquire_single_volume_plan',
            kwargs={
                'volume_scanner': 'volume_scanner',
                'num_slices': num_slices,
                'exposure_ms': exposure_ms,
                'galvo_amplitude': galvo_amplitude,
                'galvo_center': galvo_center,
                'piezo_amplitude': piezo_amplitude,
                'piezo_center': piezo_center,
            },
            timeout=120
        )

        if result.get('success'):
            # Extract volume data from documents
            docs = result.get('documents', {})
            events = docs.get('events', [])
            if events:
                # Try to reconstruct volume from events
                images = []
                volume_path = None
                for event in events:
                    data = event.get('data', {})
                    for key in ['volume_scanner', 'camera', 'camera_image']:
                        if key in data:
                            val = data[key]
                            # Handle file-ref protocol
                            if self._is_file_ref(val):
                                arr, fpath = self._resolve_file_ref(val)
                                images.append(arr)
                                volume_path = fpath
                            else:
                                images.append(val)
                            break
                if images:
                    # If a file ref gave us the complete volume, use it directly
                    if len(images) == 1 and isinstance(images[0], np.ndarray) and images[0].ndim >= 3:
                        volume = images[0]
                    else:
                        volume = np.array(images)
                    ret = {
                        'volume': volume,
                        'shape': volume.shape,
                        'success': True,
                    }
                    if volume_path is not None:
                        ret['volume_path'] = volume_path
                    return ret

        return {'error': result.get('error', 'Acquisition failed'), 'success': False}

    async def set_led(self, state: str = 'Closed') -> Dict:
        """
        Set LED state directly (bypasses plan queue for immediate effect).

        Parameters
        ----------
        state : str
            'Open' (on) or 'Closed' (off) - Micro-Manager ConfigGroup presets
        """
        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{self.http_url}/api/led/set",
                json={'state': state}
            ) as response:
                return await response.json()

    async def get_led_status(self) -> Dict:
        """
        Get current LED status and available configurations.

        Returns
        -------
        dict
            Contains 'current_state', 'available_configs', 'group_name'
        """
        async with aiohttp.ClientSession() as session:
            async with session.get(f"{self.http_url}/api/led/status") as response:
                return await response.json()

    async def set_camera_led_mode(self, use_led: bool = False) -> Dict:
        """
        Enable/disable automatic LED control for bottom camera captures.

        Parameters
        ----------
        use_led : bool
            True to enable LED during capture, False to disable (ambient light only)
        """
        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{self.http_url}/api/camera/led_mode",
                json={'use_led': use_led}
            ) as response:
                return await response.json()

    async def set_bottom_camera_exposure(self, exposure_ms: float) -> Dict:
        """
        Set bottom camera exposure time.

        Parameters
        ----------
        exposure_ms : float
            Exposure time in milliseconds
        """
        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{self.http_url}/api/camera/exposure",
                json={'exposure_ms': exposure_ms}
            ) as response:
                return await response.json()

    async def get_bottom_camera_exposure(self) -> Dict:
        """Get current bottom camera exposure time."""
        async with aiohttp.ClientSession() as session:
            async with session.get(f"{self.http_url}/api/camera/exposure") as response:
                return await response.json()

    async def capture_bottom_image(self, use_led: bool = False, exposure_ms: float = None) -> np.ndarray:
        """
        Capture image from bottom camera.

        Parameters
        ----------
        use_led : bool
            Whether to turn on LED during capture. Default: False (ambient light)
        exposure_ms : float, optional
            Exposure time in milliseconds. If None, uses current camera setting.

        Returns
        -------
        np.ndarray
            2D image array
        """
        # Set exposure if specified
        if exposure_ms is not None:
            await self.set_bottom_camera_exposure(exposure_ms)

        # Set camera LED mode before capture (controls automatic LED in device)
        await self.set_camera_led_mode(use_led)

        # Capture image (LED is controlled by device based on use_led setting)
        result = await self._submit_plan_and_wait(
            'capture_bottom_image_plan',
            kwargs={'bottom_camera': 'bottom_camera'}
        )

        if result.get('success'):
            docs = result.get('documents', {})
            events = docs.get('events', [])
            if events:
                data = events[0].get('data', {})
                for key in ['bottom_camera', 'bottom_camera_image', 'Bottom PCO']:
                    if key in data:
                        val = data[key]
                        if self._is_file_ref(val):
                            arr, _ = self._resolve_file_ref(val)
                            return arr
                        return np.array(val)

        return np.zeros((100, 100), dtype=np.uint16)

    # =========================================================================
    # Napari Embryo Editing (client-side)
    # =========================================================================

    async def _edit_embryos_in_napari(
        self,
        image: np.ndarray,
        embryos: List[Dict],
        stage_position: Tuple[float, float],
        pixel_size_um: float = DEFAULT_PIXEL_SIZE_UM,
        objective_mag: float = DEFAULT_OBJECTIVE_MAG
    ) -> List[Dict]:
        """
        Open napari for interactive embryo editing.

        This runs on the client side (main thread) to avoid Qt threading issues.
        """
        import napari

        um_per_pixel = pixel_size_um / objective_mag
        image_center_x = image.shape[1] / 2
        image_center_y = image.shape[0] / 2

        # Build points array from detected embryos
        original_points = []
        for emb in embryos:
            px = emb.get('center_x', emb.get('pixel_x', 0))
            py = emb.get('center_y', emb.get('pixel_y', 0))
            original_points.append([py, px])  # napari [row, col]

        # Create napari viewer
        viewer = napari.Viewer(title="Edit Detected Embryos - Close when done")
        viewer.add_image(image, name='Bottom Camera', colormap='gray')

        # Add editable points layer
        points_layer = viewer.add_points(
            original_points if original_points else None,
            name='Embryos (editable)',
            face_color='lime',
            border_color='white',
            size=35,
            symbol='cross'
        )
        points_layer.mode = 'select'

        original_points_set = set(tuple(p) for p in original_points)

        logger.info("[INTERACTIVE] Edit embryos in napari:")
        logger.info("  - Press '2' or click 'Add points' to add new embryos")
        logger.info("  - Press '3' or click 'Select points' to select existing")
        logger.info("  - Press Delete/Backspace to remove selected points")
        logger.info("  - Drag points to move them")
        logger.info("  - Close window when done")

        # Run napari (blocking - waits for user to close window)
        napari.run()

        # Get final points
        final_points = points_layer.data

        # Rebuild embryo list from edited points
        final_embryos = []
        next_id = len(embryos) + 1

        for i, point in enumerate(final_points):
            py, px = point[0], point[1]

            # Check if this point matches an original
            point_tuple = tuple(point)
            original_idx = None
            for j, orig_point in enumerate(original_points):
                if tuple(orig_point) == point_tuple:
                    original_idx = j
                    break

            if original_idx is not None and original_idx < len(embryos):
                # Existing embryo, keep original data
                emb = dict(embryos[original_idx])
            else:
                # New embryo - calculate stage coordinates using centralized function
                stage_x_um, stage_y_um = pixel_to_stage_position(
                    pixel_x=px,
                    pixel_y=py,
                    image_center_x=image_center_x,
                    image_center_y=image_center_y,
                    stage_x=stage_position[0],
                    stage_y=stage_position[1],
                    um_per_pixel=um_per_pixel
                )

                emb = {
                    'embryo_id': f'embryo_{next_id}',
                    'center_x': float(px),
                    'center_y': float(py),
                    'stage_x_um': float(stage_x_um),
                    'stage_y_um': float(stage_y_um),
                    'confidence': 1.0,
                    'source': 'manual_edit'
                }
                next_id += 1

            # Always update pixel coordinates from napari
            emb['center_x'] = float(px)
            emb['center_y'] = float(py)

            # Recalculate stage coordinates using centralized function
            stage_x_um, stage_y_um = pixel_to_stage_position(
                pixel_x=px,
                pixel_y=py,
                image_center_x=image_center_x,
                image_center_y=image_center_y,
                stage_x=stage_position[0],
                stage_y=stage_position[1],
                um_per_pixel=um_per_pixel
            )
            emb['stage_x_um'] = float(stage_x_um)
            emb['stage_y_um'] = float(stage_y_um)

            final_embryos.append(emb)

        # Summary
        final_points_set = set(tuple(p) for p in final_points)
        added = len(final_points_set - original_points_set)
        removed = len(original_points_set - final_points_set)

        logger.info("Edit complete: Original: %d, Final: %d, Added: %d, Removed: %d",
                     len(embryos), len(final_embryos), added, removed)

        return final_embryos

    # =========================================================================
    # SAM Embryo Detection (HTTP API)
    # =========================================================================

    async def detect_embryos(
        self,
        pixel_size_um: float = DEFAULT_PIXEL_SIZE_UM,
        objective_mag: float = DEFAULT_OBJECTIVE_MAG,
        use_claude_review: bool = True,
        min_confidence: float = 0.7,
        exposure_ms: float = None,
        brightness_percentile: float = 99.0,
        min_area: int = 5000,
        max_area: int = 150000,
        open_editor: bool = False,
    ) -> Dict:
        """
        Capture image and detect embryos using brightness detection + SAM.

        This method calls the device layer's /api/detect_embryos endpoint which:
        1. Captures image from bottom camera
        2. Reads stage position
        3. Runs brightness + SAM detection in-process
        4. Returns embryo positions

        Parameters
        ----------
        pixel_size_um : float
            Camera pixel size in micrometers
        objective_mag : float
            Objective magnification
        use_claude_review : bool
            Whether to use Claude Vision for verification
        min_confidence : float
            Minimum confidence threshold
        exposure_ms : float, optional
            Camera exposure time in milliseconds. Higher values improve contrast.
            If None, uses current camera setting.
        brightness_percentile : float
            Percentile threshold for brightness-based detection.
            99.0 = fewer, confident detections. 98.0 = more. Default: 99.0
        min_area : int
            Minimum embryo area in pixels. Default: 5000
        max_area : int
            Maximum embryo area in pixels. Default: 150000
        open_editor : bool
            If True, opens napari after detection for interactive editing.
            User can add/delete/move embryos. Default: False

        Returns
        -------
        dict
            Detection results with 'embryos' list
        """
        # Check SAM availability
        if not self.has_sam:
            return {'error': 'SAM detection not available on server'}

        self._ensure_connected()

        try:
            logger.info("Calling /api/detect_embryos (server-side capture + SAM)...")
            logger.info("Parameters: brightness_percentile=%s, area=%s-%s", brightness_percentile, min_area, max_area)

            # Build request payload
            payload = {
                'pixel_size_um': pixel_size_um,
                'objective_mag': objective_mag,
                'use_claude_review': use_claude_review,
                'min_confidence': min_confidence,
                'brightness_percentile': brightness_percentile,
                'min_area': min_area,
                'max_area': max_area,
            }
            if exposure_ms is not None:
                payload['exposure_ms'] = exposure_ms

            # Call HTTP endpoint (timeout 5 min for SAM + Claude)
            async with self._session.post(
                f"{self.http_url}/api/detect_embryos",
                json=payload,
                timeout=300
            ) as resp:
                result = await resp.json()

            if not result.get('success'):
                return result

            embryos = result.get('embryos', [])
            stage_pos = tuple(result.get('stage_position', [0.0, 0.0]))

            # Load image if path provided (for napari editor)
            image = None
            image_path = result.get('image_path')
            if image_path and open_editor:
                try:
                    import tifffile
                    image = tifffile.imread(image_path)
                except Exception:
                    pass

            # If no image but editor requested, capture one for display
            if image is None and open_editor:
                logger.info("Capturing image for napari editor...")
                image = await self.capture_bottom_image(exposure_ms=exposure_ms)

            # Open napari editor if requested (runs on client side for main thread)
            if open_editor and image is not None:
                logger.info("Opening napari editor for review/editing...")
                embryos = await self._edit_embryos_in_napari(
                    image, embryos, stage_pos, pixel_size_um, objective_mag
                )
                result['embryos'] = embryos

            # Add image to result if available
            if image is not None:
                result['image'] = image

            # Store for later visualization
            self._last_detection = {
                'image': image,
                'embryos': result.get('embryos', []),
                'stage_position': list(stage_pos)
            }

            return result

        except asyncio.TimeoutError:
            return {'success': False, 'error': 'Detection timed out (5 min limit)'}
        except aiohttp.ClientError as e:
            import traceback
            return {
                'error': str(NetworkError(f"Device layer request failed: {e}")),
                'traceback': traceback.format_exc(),
                'success': False
            }
        except Exception as e:
            import traceback
            return {
                'error': str(e),
                'traceback': traceback.format_exc(),
                'success': False
            }

    # Alias for compatibility with existing agent code
    capture_and_detect_embryos = detect_embryos

    async def manual_mark_embryos(
        self,
        pixel_size_um: float = DEFAULT_PIXEL_SIZE_UM,
        objective_mag: float = DEFAULT_OBJECTIVE_MAG,
        exposure_ms: float = None,
        existing_embryos: list = None,
    ) -> Dict:
        """
        Capture image and manually mark embryos via napari.

        Opens a napari window where user can click on embryo centers.
        Existing embryos are shown in green for reference.

        Parameters
        ----------
        pixel_size_um : float
            Camera pixel size in micrometers
        objective_mag : float
            Objective magnification
        exposure_ms : float, optional
            Camera exposure time in milliseconds for better contrast.
            If None, uses current camera setting.
        existing_embryos : list, optional
            List of existing embryo dicts with stage positions.
            Will be converted to pixel positions and displayed.

        Returns
        -------
        dict
            Detection results with 'embryos' list
        """
        self._ensure_connected()

        try:
            # Capture image via queue server
            if exposure_ms:
                logger.info("Setting exposure to %s ms...", exposure_ms)
            logger.info("Capturing bottom camera image...")
            image = await self.capture_bottom_image(exposure_ms=exposure_ms)
            if image.shape == (100, 100):
                return {'error': 'Failed to capture image'}

            logger.info("Image shape: %s", image.shape)

            # Get stage position
            logger.info("Reading stage position...")
            stage_pos = await self.get_stage_position()
            logger.info("Stage position: %s", stage_pos)

            # Convert existing embryos to pixel positions for display
            display_embryos = None
            if existing_embryos:
                um_per_pixel = get_um_per_pixel(pixel_size_um, objective_mag)
                image_center_x = image.shape[1] / 2
                image_center_y = image.shape[0] / 2

                display_embryos = []
                for emb in existing_embryos:
                    stage_x = emb.get('stage_x', emb.get('x', 0))
                    stage_y = emb.get('stage_y', emb.get('y', 0))

                    # Convert stage to pixel using centralized function
                    pixel_x, pixel_y = stage_to_pixel_position(
                        stage_x=stage_x,
                        stage_y=stage_y,
                        current_stage_x=stage_pos[0],
                        current_stage_y=stage_pos[1],
                        image_center_x=image_center_x,
                        image_center_y=image_center_y,
                        um_per_pixel=um_per_pixel
                    )

                    display_embryos.append({
                        'embryo_id': emb.get('embryo_id', '?'),
                        'pixel_x': pixel_x,
                        'pixel_y': pixel_y,
                    })
                logger.info("Showing %d existing embryos", len(display_embryos))

            # Run manual marking client-side (napari requires main thread)
            logger.info("Opening napari window for manual marking...")
            import napari

            um_per_pixel = get_um_per_pixel(pixel_size_um, objective_mag)
            image_center_x = image.shape[1] / 2
            image_center_y = image.shape[0] / 2

            viewer = napari.Viewer(title="Click on embryo centers (close window when done)")
            viewer.add_image(image, name='Bottom Camera', colormap='gray')

            # Add existing embryos as green points
            if display_embryos:
                existing_points = []
                existing_labels = []
                for emb in display_embryos:
                    px = emb.get('pixel_x')
                    py = emb.get('pixel_y')
                    eid = emb.get('embryo_id', '?')
                    if px is not None and py is not None:
                        existing_points.append([py, px])  # napari uses [row, col] = [y, x]
                        existing_labels.append(eid)

                if existing_points:
                    properties = {'label': existing_labels}
                    viewer.add_points(
                        existing_points,
                        name='Existing Embryos',
                        face_color='green',
                        border_color='white',
                        size=30,
                        symbol='cross',
                        properties=properties,
                        text={'string': '{label}', 'size': 12, 'color': 'green'}
                    )

            # Add points layer for new embryos (user will add points here)
            new_points_layer = viewer.add_points(
                name='New Embryos (click to add)',
                face_color='red',
                border_color='white',
                size=30,
                symbol='disc'
            )
            new_points_layer.mode = 'add'  # Start in add mode

            logger.info("[INTERACTIVE] Click on embryo centers in the napari window.")
            logger.info("Close the window when done marking.")

            # Run napari (blocking)
            napari.run()

            # Get the points that were added
            clicked_points = new_points_layer.data  # Array of [y, x] coordinates

            # Convert clicked points to embryo positions
            embryos = []
            for i, point in enumerate(clicked_points):
                py, px = point[0], point[1]  # napari stores as [y, x]

                # Convert pixel to stage position using centralized function
                embryo_x, embryo_y = pixel_to_stage_position(
                    pixel_x=px,
                    pixel_y=py,
                    image_center_x=image_center_x,
                    image_center_y=image_center_y,
                    stage_x=stage_pos[0],
                    stage_y=stage_pos[1],
                    um_per_pixel=um_per_pixel
                )

                embryos.append({
                    'embryo_id': f'embryo_{i+1}',
                    'pixel_x': float(px),
                    'pixel_y': float(py),
                    'stage_x_um': float(embryo_x),
                    'stage_y_um': float(embryo_y),
                    'confidence': 1.0,
                    'source': 'manual_marking'
                })

            logger.info("Marked %d embryos", len(embryos))

            return {
                'success': True,
                'embryos': embryos,
                'num_embryos': len(embryos),
                'image': image,
                'stage_position': list(stage_pos)
            }

        except Exception as e:
            logger.debug("manual_mark_embryos failed with %s: %s", type(e).__name__, e)
            import traceback
            return {
                'error': str(e),
                'traceback': traceback.format_exc(),
                'success': False
            }

    async def edit_embryos(
        self,
        image: np.ndarray,
        embryos: list,
        stage_position: tuple,
        pixel_size_um: float = DEFAULT_PIXEL_SIZE_UM,
        objective_mag: float = DEFAULT_OBJECTIVE_MAG,
    ) -> Dict:
        """
        Interactive embryo editor via napari.

        Opens a napari viewer where user can add, delete, or move embryo positions.
        Close the window when done to apply changes.

        Parameters
        ----------
        image : np.ndarray
            Image to display
        embryos : list
            List of existing embryo dicts with pixel_x, pixel_y, embryo_id
        stage_position : tuple
            Current (x, y) stage position in micrometers
        pixel_size_um : float
            Camera pixel size in micrometers
        objective_mag : float
            Objective magnification

        Returns
        -------
        dict
            Updated embryo list with added/removed counts
        """
        try:
            logger.info("Opening napari embryo editor...")
            edited_embryos = await self._edit_embryos_in_napari(
                image, embryos, stage_position, pixel_size_um, objective_mag
            )

            return {
                'success': True,
                'embryos': edited_embryos,
                'original_count': len(embryos),
                'final_count': len(edited_embryos),
            }

        except Exception as e:
            logger.debug("edit_embryos failed with %s: %s", type(e).__name__, e)
            import traceback
            return {
                'error': str(e),
                'traceback': traceback.format_exc(),
                'success': False
            }

    async def view_image(self, image: np.ndarray = None, title: str = "Image View",
                         exposure_ms: float = None, save_path: str = None,
                         show: bool = True, embryo_annotations: list = None) -> Dict:
        """
        View an image using napari (client-side).

        Parameters
        ----------
        image : np.ndarray, optional
            Image to view. If None, captures from bottom camera.
        title : str
            Window title
        exposure_ms : float, optional
            Camera exposure time if capturing new image.
        save_path : str, optional
            Path to save the image
        show : bool
            Whether to show in napari window (blocking). Default: True
        embryo_annotations : list, optional
            List of embryo dicts with 'embryo_id', 'pixel_x', 'pixel_y', 'label'
            to overlay on the image

        Returns
        -------
        dict
            Success status and save_path if saved
        """
        try:
            if image is None:
                if exposure_ms:
                    logger.info("Setting exposure to %s ms...", exposure_ms)
                logger.info("Capturing image...")
                image = await self.capture_bottom_image(exposure_ms=exposure_ms)

            result = {'success': True}

            # Save if path provided
            if save_path:
                from pathlib import Path
                from PIL import Image as PILImage

                # Normalize image to 8-bit for saving
                if image.dtype != np.uint8:
                    img_normalized = ((image - image.min()) / (image.max() - image.min()) * 255).astype(np.uint8)
                else:
                    img_normalized = image

                pil_img = PILImage.fromarray(img_normalized)

                # Use appropriate format based on extension
                ext = Path(save_path).suffix.lower()
                if ext in ['.jpg', '.jpeg']:
                    pil_img.save(save_path, 'JPEG', quality=70, optimize=True)
                elif ext == '.png':
                    pil_img.save(save_path, 'PNG', optimize=True)
                else:
                    # Fallback to tifffile for other formats
                    import tifffile
                    tifffile.imwrite(save_path, image)

                result['saved_to'] = save_path
                logger.info("Saved image to: %s", save_path)

            # Show if requested (using napari client-side)
            if show:
                import napari

                logger.info("Opening napari viewer (shape: %s)...", image.shape)

                viewer = napari.Viewer(title=title)
                viewer.add_image(image, name='Image', colormap='gray')

                # Add embryo annotations if provided
                if embryo_annotations:
                    in_view_points = []
                    out_of_view_points = []

                    for emb in embryo_annotations:
                        px = emb.get('pixel_x')
                        py = emb.get('pixel_y')

                        if px is not None and py is not None:
                            in_view = 0 <= px < image.shape[1] and 0 <= py < image.shape[0]
                            if in_view:
                                in_view_points.append([py, px])
                            else:
                                out_of_view_points.append([py, px])

                    # Add in-view embryos (green)
                    if in_view_points:
                        viewer.add_points(
                            in_view_points,
                            name='Embryos (in view)',
                            face_color='lime',
                            border_color='white',
                            size=30,
                            symbol='cross'
                        )

                    # Add out-of-view embryos (orange)
                    if out_of_view_points:
                        viewer.add_points(
                            out_of_view_points,
                            name='Embryos (out of view)',
                            face_color='orange',
                            border_color='white',
                            size=30,
                            symbol='cross'
                        )

                napari.run()  # Blocking

            return result

        except Exception as e:
            logger.debug("view_image failed with %s: %s", type(e).__name__, e)
            return {'error': str(e)}

    async def view_detected_embryos(
        self,
        save_path: Optional[str] = None,
        show: bool = True
    ) -> Dict:
        """
        View the last detected embryos with bounding boxes.

        Parameters
        ----------
        save_path : str, optional
            Path to save the visualization image
        show : bool
            Whether to display in matplotlib window

        Returns
        -------
        dict
            Success status
        """
        if self._last_detection is None:
            return {'error': 'No detection results available. Run detect_embryos first.'}

        try:
            image = self._last_detection['image']
            embryos = self._last_detection['embryos']

            if not embryos:
                return {'error': 'No embryos in last detection'}

            # Use client-side view_embryos
            return await self.view_embryos(
                image=image,
                embryos=embryos,
                title=f"Detected {len(embryos)} Embryos",
                save_path=save_path,
                show=show
            )

        except Exception as e:
            logger.debug("view_detected_embryos failed with %s: %s", type(e).__name__, e)
            import traceback
            return {'error': str(e), 'traceback': traceback.format_exc()}

    async def view_embryos(
        self,
        image: np.ndarray,
        embryos: List[Dict],
        title: str = "Embryos",
        save_path: Optional[str] = None,
        show: bool = True
    ) -> Dict:
        """
        View specified embryos with markers on an image using napari (client-side).

        Parameters
        ----------
        image : np.ndarray
            Image to display
        embryos : list of dict
            Embryo data with pixel_x, pixel_y coordinates
        title : str
            Window title
        save_path : str, optional
            Path to save the visualization image
        show : bool
            Whether to display in napari window

        Returns
        -------
        dict
            Success status
        """
        if image is None:
            return {'error': 'No image provided'}

        if not embryos:
            return {'error': 'No embryos to display'}

        try:
            result = {'success': True, 'num_embryos': len(embryos)}

            # Save if path provided
            if save_path:
                from pathlib import Path
                from PIL import Image as PILImage

                # Normalize image to 8-bit for saving
                if image.dtype != np.uint8:
                    img_normalized = ((image - image.min()) / (image.max() - image.min()) * 255).astype(np.uint8)
                else:
                    img_normalized = image

                pil_img = PILImage.fromarray(img_normalized)

                # Use appropriate format based on extension
                ext = Path(save_path).suffix.lower()
                if ext in ['.jpg', '.jpeg']:
                    pil_img.save(save_path, 'JPEG', quality=70, optimize=True)
                elif ext == '.png':
                    pil_img.save(save_path, 'PNG', optimize=True)
                else:
                    # Fallback to tifffile for other formats
                    import tifffile
                    tifffile.imwrite(save_path, image)

                result['saved_to'] = save_path
                logger.info("Saved image to: %s", save_path)

            # Show if requested (using napari client-side)
            if show:
                import napari

                logger.info("Opening napari viewer with %d embryos...", len(embryos))

                viewer = napari.Viewer(title=title)
                viewer.add_image(image, name='Image', colormap='gray')

                # Collect embryo points and colors
                points = []
                colors = []
                color_palette = [
                    'red', 'blue', 'green', 'yellow', 'magenta',
                    'cyan', 'orange', 'purple', 'pink', 'lime'
                ]

                for i, embryo in enumerate(embryos):
                    px = embryo.get('pixel_x', embryo.get('center_x', 0))
                    py = embryo.get('pixel_y', embryo.get('center_y', 0))
                    points.append([py, px])  # napari uses [row, col] = [y, x]
                    colors.append(color_palette[i % len(color_palette)])

                    embryo_id = embryo.get('embryo_id', embryo.get('id', i))
                    logger.debug("%s at pixel (%.0f, %.0f)", embryo_id, px, py)

                # Add points layer for embryos
                if points:
                    viewer.add_points(
                        points,
                        name=f'Embryos ({len(embryos)})',
                        face_color=colors,
                        border_color='white',
                        size=40,
                        symbol='disc'
                    )

                napari.run()  # Blocking

            return result

        except Exception as e:
            logger.debug("view_embryos failed with %s: %s", type(e).__name__, e)
            import traceback
            return {'error': str(e), 'traceback': traceback.format_exc()}

    # =========================================================================
    # Status
    # =========================================================================

    async def get_status(self) -> Dict:
        """
        Get server status.

        Returns
        -------
        dict
            Status information
        """
        status = {}

        # Device Layer Server status
        if self._session:
            try:
                async with self._session.get(f"{self.http_url}/api/status") as resp:
                    if resp.status == 200:
                        server_status = await resp.json()
                        status['queue_server'] = {
                            'manager_state': server_status.get('manager_state'),
                            're_state': server_status.get('re_state', 'idle'),
                            'devices': server_status.get('devices', []),
                            'plans': server_status.get('plans', []),
                        }
                    else:
                        status['queue_server'] = {'error': f'HTTP {resp.status}'}
            except aiohttp.ClientError as e:
                status['queue_server'] = {'error': str(e)}
            except Exception as e:
                status['queue_server'] = {'error': str(e)}
        else:
            status['queue_server'] = {'connected': False}

        # SAM status (via HTTP, same server)
        if self._session and self._qs_connected:
            try:
                async with self._session.get(f"{self.http_url}/api/sam/status") as resp:
                    if resp.status == 200:
                        sam_status = await resp.json()
                        status['sam_server'] = {
                            'available': sam_status.get('available', False),
                            'loaded': sam_status.get('loaded', False),
                            'device': sam_status.get('device', 'unknown'),
                        }
                    else:
                        status['sam_server'] = {'error': f'HTTP {resp.status}'}
            except aiohttp.ClientError as e:
                status['sam_server'] = {'error': str(e)}
            except Exception as e:
                status['sam_server'] = {'error': str(e)}
        else:
            status['sam_server'] = {'connected': False}

        return status


async def create_queue_server_client(
    http_url: str = f"http://{settings.network.device_host}:{settings.network.device_port}",
) -> Optional[QueueServerClient]:
    """
    Create and connect a device layer client.

    Parameters
    ----------
    http_url : str
        Device Layer HTTP API URL

    Returns
    -------
    QueueServerClient or None
        Connected client, or None if connection failed
    """
    client = QueueServerClient(http_url=http_url)
    if await client.connect():
        return client
    return None
