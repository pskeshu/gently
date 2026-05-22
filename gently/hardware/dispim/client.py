"""
diSPIM Microscope — HTTP client for the diSPIM device layer.

Implements the Microscope protocol via HTTP requests to the unified
device_layer.py server (Bluesky plans + SAM detection).
"""

import asyncio
import logging
import traceback
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
from gently.harness.microscope import Microscope
from gently.settings import settings
from gently.exceptions import DeviceLayerError, NetworkError, AcquisitionError


class DiSPIMMicroscope(Microscope):
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
    >>> results = await client.detect_embryos()
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

    async def connect(self) -> bool:
        """
        Connect to Device Layer Server.

        Returns
        -------
        bool
            True if connection successful
        """
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
        """Check if connected to Device Layer Server.

        This is the cached session-level state, flipped only by
        connect() / disconnect() and by real RPC failures from actual
        work. health_check() does NOT mutate this — see its docstring.
        """
        return self._qs_connected

    async def health_check(self, timeout: float = 2.0) -> bool:
        """Actively ping the device layer. READ-ONLY.

        Returns the live ping result for callers that need a fresh
        answer (e.g. the UI status badge). Does NOT mutate
        ``_qs_connected``: that flag is the truth of "is this client's
        session connected" and must only flip on connect()/disconnect()
        or on real RPC failures from actual work. Letting a status poll
        write to it makes the UI's 15-second poll a destructive
        operation — a transient timeout on the local socket then
        disconnects any in-flight acquisition. See the 60dbbc62 session
        for the failure mode this prevents.

        Parameters
        ----------
        timeout : float
            Max seconds to wait for the ping response. Kept short so a
            dead device layer doesn't stall the UI poll.

        Returns
        -------
        bool
            True if the device layer responded with HTTP 200.
        """
        if not self._session:
            return False

        try:
            async with self._session.get(
                f"{self.http_url}/api/status",
                timeout=aiohttp.ClientTimeout(total=timeout),
            ) as resp:
                return resp.status == 200
        except (aiohttp.ClientError, asyncio.TimeoutError, Exception):
            return False

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
    # Session Configuration (FileStore integration)
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
            (e.g. ``"D:/Gently3/incoming"``).

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
    # HTTP Helpers
    # =========================================================================

    async def _api_get(self, path: str) -> dict:
        """GET request using the shared session."""
        self._ensure_connected()
        async with self._session.get(f"{self.http_url}{path}") as resp:
            return await resp.json()

    async def _api_post(self, path: str, json: dict = None) -> dict:
        """POST request using the shared session."""
        self._ensure_connected()
        async with self._session.post(f"{self.http_url}{path}", json=json) as resp:
            return await resp.json()

    async def _submit_plan_and_wait(
        self,
        plan_name: str,
        kwargs: Dict = None,
        timeout: float = 120.0,
    ) -> Dict:
        """Submit a Bluesky plan to the server and wait for completion.

        Parameters
        ----------
        plan_name : str
            Name of the Bluesky plan to execute
        kwargs : dict, optional
            Arguments to pass to the plan
        timeout : float
            Maximum time to wait (seconds)

        Returns
        -------
        dict
            Plan execution result with 'success', 'documents', etc.
        """
        self._ensure_connected()

        payload = {
            "item": {
                "name": plan_name,
                "kwargs": kwargs or {},
            }
        }

        try:
            async with self._session.post(
                f"{self.http_url}/api/queue/item/add",
                json=payload,
                timeout=aiohttp.ClientTimeout(total=timeout)
            ) as resp:
                if resp.status != 200:
                    error_text = await resp.text()
                    return {
                        'success': False,
                        'error': f"HTTP {resp.status}: {error_text}",
                    }

                result = await resp.json()

                # Resolve file references (zero-copy transfer)
                if isinstance(result, dict):
                    docs = result.get('documents', {})
                    events = docs.get('events', [])
                    for event in events:
                        data = event.get('data', {})
                        for key, val in list(data.items()):
                            if self._is_file_ref(val):
                                arr, path = self._resolve_file_ref(val)
                                data[key] = arr
                                # Store path for downstream use
                                if 'volume_path' not in result:
                                    result['volume_path'] = str(path)

                return result

        except asyncio.TimeoutError:
            return {
                'success': False,
                'error': f"Plan '{plan_name}' timed out after {timeout}s",
            }
        except aiohttp.ClientError as e:
            raise NetworkError(f"Device layer request failed: {e}") from e

    # =========================================================================
    # Stage Control
    # =========================================================================

    async def move_to_position(self, x: float, y: float) -> Dict:
        """
        Move stage to absolute position.

        Parameters
        ----------
        x, y : float
            Target position in micrometers
        """
        logger.info("Moving to (%.1f, %.1f) µm", x, y)

        result = await self._submit_plan_and_wait(
            'move_stage_plan',
            kwargs={'xy_stage': 'xy_stage', 'x': x, 'y': y}
        )

        if result.get('success'):
            return {'success': True, 'x': x, 'y': y}

        return result

    async def get_stage_position(self) -> Tuple[float, float]:
        """
        Get current stage position.

        Returns
        -------
        tuple of (float, float)
            Current (x, y) position in micrometers
        """
        result = await self._submit_plan_and_wait('read_stage_plan', kwargs={'xy_stage': 'xy_stage'})

        if result.get('success'):
            docs = result.get('documents', {})
            events = docs.get('events', [])
            if events:
                data = events[0].get('data', {})
                # Look for stage coordinates
                for key in ['XY:31', 'xy_stage', 'stage']:
                    if key in data:
                        val = data[key]
                        if isinstance(val, (list, tuple)) and len(val) >= 2:
                            return (float(val[0]), float(val[1]))
                        if isinstance(val, dict):
                            return (float(val.get('x', 0)), float(val.get('y', 0)))

        raise DeviceLayerError("Failed to read stage position")

    async def get_piezo_position(self) -> float:
        """
        Get current piezo Z position.

        Returns
        -------
        float
            Current Z position in micrometers
        """
        result = await self._submit_plan_and_wait('read_piezo_plan', kwargs={'piezo': 'piezo'})

        if result.get('success'):
            docs = result.get('documents', {})
            events = docs.get('events', [])
            if events:
                data = events[0].get('data', {})
                for key in ['PiezoStage:P:34', 'piezo', 'z_stage']:
                    if key in data:
                        val = data[key]
                        if isinstance(val, (int, float)):
                            return float(val)
                        if isinstance(val, dict):
                            return float(val.get('z', val.get('position', 0)))

        raise DeviceLayerError("Failed to read piezo position")

    # =========================================================================
    # Calibration
    # =========================================================================

    async def calibrate_piezo_galvo(
        self,
        piezo_positions: Optional[List[float]] = None,
        galvo_positions: Optional[List[float]] = None,
        **kwargs,
    ) -> Dict:
        """
        Run piezo-galvo calibration plan.

        Returns
        -------
        dict
            Calibration results with optimal positions
        """
        plan_kwargs = {'lightsheet_snap': 'lightsheet_snap'}
        if piezo_positions is not None:
            plan_kwargs['piezo_positions'] = piezo_positions
        if galvo_positions is not None:
            plan_kwargs['galvo_positions'] = galvo_positions
        plan_kwargs.update(kwargs)

        result = await self._submit_plan_and_wait(
            'calibrate_piezo_galvo_plan',
            kwargs=plan_kwargs,
            timeout=300.0
        )

        if result.get('success'):
            return {
                'success': True,
                'calibration': result.get('calibration', {}),
            }

        return result

    # =========================================================================
    # Imaging
    # =========================================================================

    def _extract_image(self, result: dict, candidate_keys: List[str], multi_event: bool = False) -> Optional[tuple]:
        """Extract image array from plan result documents.

        Returns (array, path) or None if not found.
        """
        if not result.get('success'):
            return None

        docs = result.get('documents', {})
        events = docs.get('events', [])
        if not events:
            return None

        search_events = events if multi_event else [events[0]]
        for event in search_events:
            data = event.get('data', {})
            for key in candidate_keys:
                if key in data:
                    val = data[key]
                    if self._is_file_ref(val):
                        arr, fpath = self._resolve_file_ref(val)
                        return arr, fpath
                    return np.array(val), None

        return None

    async def capture_lightsheet_image(
        self,
        piezo_position: Optional[float] = None,
        galvo_position: Optional[float] = None,
        exposure_ms: float = 10.0,
        **kwargs,
    ) -> Dict:
        """
        Capture a single lightsheet image at specified position.

        Parameters
        ----------
        piezo_position : float, optional
            Z position in micrometers
        galvo_position : float, optional
            Galvo angle in volts
        exposure_ms : float
            Camera exposure time

        Returns
        -------
        dict
            ``{'image': np.ndarray, 'piezo_position': float, 'galvo_position': float, 'success': bool}``
        """
        result = await self._submit_plan_and_wait(
            'capture_lightsheet_image_plan',
            kwargs={
                'lightsheet_snap': 'lightsheet_snap',
                'scanner': 'scanner',
                'piezo': 'piezo',
                'laser_control': 'laser_control',
                'piezo_position': piezo_position if piezo_position is not None else 50.0,
                'galvo_position': galvo_position if galvo_position is not None else 0.0,
            },
            timeout=30.0
        )

        extracted = self._extract_image(result, ['HamCam1', 'lightsheet_snap', 'camera'])
        if extracted:
            arr, fpath = extracted
            ret = {
                'image': arr,
                'piezo_position': piezo_position or 0.0,
                'galvo_position': galvo_position or 0.0,
                'success': True,
            }
            if fpath:
                ret['image_path'] = fpath
            return ret

        return {'error': result.get('error', 'No image data'), 'success': False}

    async def acquire_volume(
        self,
        num_slices: int = 50,
        exposure_ms: float = 10.0,
        galvo_amplitude: float = 0.5,
        galvo_center: float = 0.0,
        piezo_amplitude: float = 25.0,
        piezo_center: float = 50.0,
        laser_config: str = None,
        laser_power_488_pct: float = None,
        laser_power_561_pct: float = None,
        laser_power_405_pct: float = None,
        laser_power_637_pct: float = None,
        **kwargs,
    ) -> Dict:
        """
        Acquire a 3D volume via synchronized galvo-piezo scan.

        Parameters
        ----------
        num_slices : int
            Number of Z slices
        exposure_ms : float
            Camera exposure per slice
        galvo_amplitude, galvo_center : float
            Galvo scan range in volts
        piezo_amplitude, piezo_center : float
            Piezo Z range in micrometers
        laser_config : str, optional
            Laser channel preset ("488 and 561", "488 only", etc.). None
            uses the device-layer default.
        laser_power_488_pct, laser_power_561_pct, laser_power_405_pct, laser_power_637_pct : float, optional
            Per-line laser power %. Hard-limited at the device layer
            (DiSPIMLightSource.POWER_LIMITS_PCT). None leaves current
            setpoint untouched.

        Returns
        -------
        dict
            ``{'volume': np.ndarray, 'shape': tuple, 'success': bool}``
        """
        plan_kwargs = {
            'volume_scanner': 'volume_scanner',
            'num_slices': num_slices,
            'exposure_ms': exposure_ms,
            'galvo_amplitude': galvo_amplitude,
            'galvo_center': galvo_center,
            'piezo_amplitude': piezo_amplitude,
            'piezo_center': piezo_center,
        }
        # Only forward kwargs the user explicitly set — leaves the
        # acquire_single_volume_plan defaults in place otherwise.
        if laser_config is not None:
            plan_kwargs['laser_config'] = laser_config
        if laser_power_488_pct is not None:
            plan_kwargs['laser_power_488_pct'] = laser_power_488_pct
        if laser_power_561_pct is not None:
            plan_kwargs['laser_power_561_pct'] = laser_power_561_pct
        if laser_power_405_pct is not None:
            plan_kwargs['laser_power_405_pct'] = laser_power_405_pct
        if laser_power_637_pct is not None:
            plan_kwargs['laser_power_637_pct'] = laser_power_637_pct

        result = await self._submit_plan_and_wait(
            'acquire_single_volume_plan',
            kwargs=plan_kwargs,
            timeout=120.0
        )

        extracted = self._extract_image(result, ['volume_scanner', 'camera', 'camera_image'], multi_event=True)
        if extracted:
            arr, fpath = extracted
            ret = {
                'volume': arr,
                'shape': arr.shape,
                'success': True,
            }
            if fpath:
                ret['volume_path'] = str(fpath)
            elif result.get('volume_path'):
                ret['volume_path'] = result['volume_path']
            return ret

        return {'error': result.get('error', 'Acquisition failed'), 'success': False}

    # =========================================================================
    # LED / Camera Controls
    # =========================================================================

    async def set_led(self, state: str = 'Closed') -> Dict:
        """Set LED state ('Open' or 'Closed')."""
        return await self._api_post('/api/led/set', {'state': state})

    async def set_laser_power(self, wavelength: int, pct: float) -> Dict:
        """Set per-line laser power %.

        Hits the device layer's ``POST /api/light_source/power`` directly
        — bypasses the Bluesky queue so this is a fast, no-experiment-trace
        operation suitable for setup pokes. Hard-limited at the device
        layer (``DiSPIMLightSource.POWER_LIMITS_PCT``); out-of-range
        values return a structured error.

        Parameters
        ----------
        wavelength : int
            Laser line (488, 561, 405, 637).
        pct : float
            Setpoint percent (must be within hard limit for ``wavelength``).
        """
        return await self._api_post('/api/light_source/power', {
            'wavelength': int(wavelength),
            'pct': float(pct),
        })

    async def get_laser_power(self, wavelength: int) -> Dict:
        """Read the current per-line laser power %.

        Hits ``GET /api/light_source/power?wavelength={wavelength}`` —
        direct, no queue.
        """
        try:
            async with self._session.get(
                f"{self.http_url}/api/light_source/power",
                params={"wavelength": int(wavelength)},
                timeout=5,
            ) as resp:
                return await resp.json()
        except Exception as e:
            return {'success': False, 'error': str(e)}

    async def get_led_status(self) -> Dict:
        """Get current LED status."""
        return await self._api_get('/api/led/status')

    # ------------------------------------------------------------------
    # Live device-state readout (streamed from the device layer poller)
    # ------------------------------------------------------------------
    async def get_device_state(self, refresh: bool = False) -> Dict:
        """One-shot snapshot of all device positions + properties.

        Parameters
        ----------
        refresh : bool
            If True, force the device layer to re-read MMCore right now.
            Otherwise return the most recent poller snapshot (typically <500 ms old).
        """
        path = '/api/devices/state'
        if refresh:
            path += '?refresh=1'
        return await self._api_get(path)

    async def stream_device_states(self, timeout: Optional[float] = None):
        """Async generator yielding parsed device-state events from the SSE stream.

        Yields each event payload as a dict. Comment-style heartbeats (lines
        starting with `:`) are silently skipped. The generator runs until the
        connection drops or the caller breaks out — auto-reconnect is the
        consumer's responsibility (see DeviceStateMonitor).

        Parameters
        ----------
        timeout : float, optional
            Per-event read timeout in seconds. None = wait indefinitely.

        Example
        -------
        >>> async for evt in microscope.stream_device_states():
        ...     positions = evt.get('positions', {})
        ...     print(positions)
        """
        self._ensure_connected()
        # SSE: no overall timeout, but allow per-read timeout via aiohttp.
        client_timeout = aiohttp.ClientTimeout(
            total=None,
            sock_read=timeout,
            sock_connect=10.0,
        )
        url = f"{self.http_url}/api/devices/stream"
        async with self._session.get(url, timeout=client_timeout) as resp:
            resp.raise_for_status()
            buf = b""
            async for chunk in resp.content.iter_any():
                if not chunk:
                    continue
                buf += chunk
                # SSE events terminate with a blank line (\n\n or \r\n\r\n).
                while b"\n\n" in buf:
                    event_block, buf = buf.split(b"\n\n", 1)
                    data_lines = []
                    for line in event_block.splitlines():
                        if not line or line.startswith(b":"):
                            continue
                        if line.startswith(b"data:"):
                            data_lines.append(line[5:].lstrip())
                        # `event:` lines are ignored — we treat all event
                        # types uniformly on the consumer side.
                    if not data_lines:
                        continue
                    raw = b"\n".join(data_lines).decode("utf-8", errors="replace")
                    try:
                        import json as _json
                        yield _json.loads(raw)
                    except Exception as exc:
                        logger.warning("Malformed SSE payload skipped: %s", exc)

    async def stream_bottom_camera(self, timeout: Optional[float] = None):
        """Async generator yielding JPEG frames from the bottom-camera SSE stream.

        Mirrors :meth:`stream_device_states`. The device layer's streamer task
        is subscriber-gated: it starts on first connect and exits when the
        last subscriber drops, so simply iterating this generator costs the
        camera nothing until it's actually running.

        Each yielded payload is the dict the device layer publishes:
        ``{"t": <unix>, "shape": [h, w], "downsample": int,
           "mime": "image/jpeg", "jpeg_b64": <str>}``.
        """
        self._ensure_connected()
        client_timeout = aiohttp.ClientTimeout(
            total=None,
            sock_read=timeout,
            sock_connect=10.0,
        )
        url = f"{self.http_url}/api/bottom_camera/stream"
        async with self._session.get(url, timeout=client_timeout) as resp:
            resp.raise_for_status()
            buf = b""
            async for chunk in resp.content.iter_any():
                if not chunk:
                    continue
                buf += chunk
                while b"\n\n" in buf:
                    event_block, buf = buf.split(b"\n\n", 1)
                    data_lines = []
                    for line in event_block.splitlines():
                        if not line or line.startswith(b":"):
                            continue
                        if line.startswith(b"data:"):
                            data_lines.append(line[5:].lstrip())
                    if not data_lines:
                        continue
                    raw = b"\n".join(data_lines).decode("utf-8", errors="replace")
                    try:
                        import json as _json
                        yield _json.loads(raw)
                    except Exception as exc:
                        logger.warning("Malformed bottom-camera SSE payload skipped: %s", exc)

    async def set_camera_led_mode(self, use_led: bool = False) -> Dict:
        """Enable/disable automatic LED for bottom camera captures."""
        return await self._api_post('/api/camera/led_mode', {'use_led': use_led})

    async def set_bottom_camera_exposure(self, exposure_ms: float) -> Dict:
        """Set bottom camera exposure time in milliseconds."""
        return await self._api_post('/api/camera/exposure', {'exposure_ms': exposure_ms})

    async def get_bottom_camera_exposure(self) -> Dict:
        """Get current bottom camera exposure time."""
        return await self._api_get('/api/camera/exposure')

    async def capture_bottom_image(self, use_led: bool = False, exposure_ms: float = None) -> dict:
        """
        Capture image from bottom camera.

        Parameters
        ----------
        use_led : bool
            Whether to turn on LED during capture.
        exposure_ms : float, optional
            Exposure time in milliseconds. If None, uses current setting.

        Returns
        -------
        dict
            ``{'image': np.ndarray, 'image_path': Path | None}``
        """
        if exposure_ms is not None:
            await self.set_bottom_camera_exposure(exposure_ms)

        await self.set_camera_led_mode(use_led)

        result = await self._submit_plan_and_wait(
            'capture_bottom_image_plan',
            kwargs={'bottom_camera': 'bottom_camera'}
        )

        extracted = self._extract_image(result, ['bottom_camera', 'bottom_camera_image', 'Bottom PCO'])
        if extracted:
            arr, fpath = extracted
            return {'image': arr, 'image_path': fpath}

        return {'image': np.zeros((100, 100), dtype=np.uint16), 'image_path': None}

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
    ) -> Dict:
        """
        Capture image and detect embryos using brightness detection + SAM.

        Returns raw SAM detections plus the bottom-camera image and stage
        position. Interactive editing is the caller's responsibility — the
        web map view (mark_embryos_web) is the standard editor; napari has
        been retired.

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
            Camera exposure time in milliseconds.
        brightness_percentile : float
            Percentile threshold for brightness-based detection.
        min_area, max_area : int
            Embryo area bounds in pixels.

        Returns
        -------
        dict
            ``{'success': bool, 'embryos': [...], 'stage_position': (x, y),
            'image': np.ndarray, ...}``
        """
        if not self.has_sam:
            return {'error': 'SAM detection not available on server'}

        self._ensure_connected()

        try:
            logger.info("Calling /api/detect_embryos (server-side capture + SAM)...")

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

            async with self._session.post(
                f"{self.http_url}/api/detect_embryos",
                json=payload,
                timeout=300
            ) as resp:
                result = await resp.json()

            if not result.get('success'):
                return result

            # Ensure the caller has the image to feed into the map view.
            if result.get('image') is None:
                image = await self._get_detection_image(result, exposure_ms)
                if image is not None:
                    result['image'] = image

            return result

        except asyncio.TimeoutError:
            return {'success': False, 'error': 'Detection timed out (5 min limit)'}
        except aiohttp.ClientError as e:
            return {'error': str(NetworkError(f"Device layer request failed: {e}")),
                    'traceback': traceback.format_exc(), 'success': False}
        except Exception as e:
            return {'error': str(e), 'traceback': traceback.format_exc(), 'success': False}

    async def _get_detection_image(self, detection_result: dict, exposure_ms: float = None) -> Optional[np.ndarray]:
        """Load or capture an image for the detection editor."""
        image_path = detection_result.get('image_path')
        if image_path:
            try:
                import tifffile
                return tifffile.imread(image_path)
            except Exception:
                pass
        # Fallback: capture a fresh image
        snap = await self.capture_bottom_image(exposure_ms=exposure_ms)
        image = snap['image']
        if snap.get('image_path'):
            try:
                snap['image_path'].unlink(missing_ok=True)
            except OSError:
                pass
        return image

    async def view_image(self, image: np.ndarray = None, title: str = "Image View",
                         exposure_ms: float = None, save_path: str = None,
                         show: bool = True, embryo_annotations: list = None) -> Dict:
        """Save a bottom-camera image to disk (replaces the napari display).

        ``show`` and ``title`` are kept for backwards compatibility with
        existing tool call sites but only the disk save still happens.
        Embryo annotations are drawn into the saved PNG if provided.
        """
        try:
            if image is None:
                snap = await self.capture_bottom_image(exposure_ms=exposure_ms)
                image = snap['image']
                if snap.get('image_path'):
                    try:
                        snap['image_path'].unlink(missing_ok=True)
                    except OSError:
                        pass

            if image is None:
                return {'success': False, 'error': 'No image to save'}

            result = {'success': True, 'shape': list(image.shape)}
            if save_path:
                # Reuse the existing PNG writer; draws annotations if any.
                from pathlib import Path as _Path
                _Path(save_path).parent.mkdir(parents=True, exist_ok=True)
                annots = []
                for a in (embryo_annotations or []):
                    px = a.get('pixel_x')
                    py = a.get('pixel_y')
                    if px is None or py is None:
                        continue
                    annots.append({
                        'embryo_number': a.get('label') or a.get('embryo_id') or '?',
                        'pixel_position': (px, py),
                    })
                if annots:
                    from gently.ui.web.embryo_marker import _save_marked_image
                    _save_marked_image(image, annots, _Path(save_path))
                else:
                    from PIL import Image as _PILImage
                    arr = image
                    if arr.dtype != np.uint8:
                        lo, hi = arr.min(), arr.max()
                        arr = ((arr - lo) / max(hi - lo, 1) * 255).astype(np.uint8)
                    _PILImage.fromarray(arr).save(save_path)
                result['saved_to'] = save_path
            return result
        except Exception as e:
            return {'error': str(e), 'traceback': traceback.format_exc(), 'success': False}

    async def view_embryos(
        self,
        image: np.ndarray,
        embryos: List[Dict],
        title: str = "Embryos",
        save_path: Optional[str] = None,
        show: bool = True,
    ) -> Dict:
        """Save an annotated PNG of embryos on an image (replaces napari).

        Markers in ``embryos`` may use ``center_x``/``center_y`` or
        ``pixel_x``/``pixel_y``. ``show`` is ignored — display is now
        the map view's job.
        """
        try:
            if image is None or not embryos:
                return {'success': False, 'error': 'No image or embryos to display'}

            annots = []
            for emb in embryos:
                px = emb.get('center_x', emb.get('pixel_x', 0))
                py = emb.get('center_y', emb.get('pixel_y', 0))
                annots.append({
                    'embryo_number': emb.get('embryo_id', '?'),
                    'pixel_position': (px, py),
                })

            result = {'success': True, 'num_embryos': len(embryos)}
            if save_path:
                from pathlib import Path as _Path
                _Path(save_path).parent.mkdir(parents=True, exist_ok=True)
                from gently.ui.web.embryo_marker import _save_marked_image
                _save_marked_image(image, annots, _Path(save_path))
                result['saved_to'] = save_path
            return result
        except Exception as e:
            return {'error': str(e), 'traceback': traceback.format_exc(), 'success': False}

    async def capture_for_marking(
        self,
        exposure_ms: float = None,
    ) -> Dict:
        """
        Capture a bottom-camera image for manual marking in the map view.

        Replaces the deprecated ``manual_mark_embryos`` (napari) — this just
        returns the image + stage position; the caller (an agent tool) is
        expected to feed it into ``mark_embryos_web``.

        Returns
        -------
        dict
            ``{'success': bool, 'image': np.ndarray, 'stage_position': (x, y),
            'image_shape': [h, w]}``
        """
        self._ensure_connected()

        try:
            snap = await self.capture_bottom_image(use_led=True, exposure_ms=exposure_ms)
            image = snap['image']

            if image is None or (image.shape == (100, 100) and image.max() == 0):
                return {'success': False, 'error': 'Failed to capture image'}

            if snap.get('image_path'):
                try:
                    snap['image_path'].unlink(missing_ok=True)
                except OSError:
                    pass

            stage_pos = await self.get_stage_position()

            return {
                'success': True,
                'image': image,
                'stage_position': list(stage_pos),
                'image_shape': list(image.shape),
            }

        except Exception as e:
            return {'error': str(e), 'traceback': traceback.format_exc(), 'success': False}

    # =========================================================================
    # Status
    # =========================================================================

    async def _get_server_status(self) -> Dict:
        """Query device layer and SAM server status via HTTP."""
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
            except (aiohttp.ClientError, Exception) as e:
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
            except (aiohttp.ClientError, Exception) as e:
                status['sam_server'] = {'error': str(e)}
        else:
            status['sam_server'] = {'connected': False}

        return status


    # =========================================================================
    # Microscope plan implementations
    # These map plan names to the existing methods above, enabling
    # microscope.execute("acquire", **params) alongside the direct methods.
    # =========================================================================

    async def _plan_move(self, x: float, y: float, **kw) -> dict:
        return await self.move_to_position(x, y)

    async def _plan_get_position(self, **kw) -> dict:
        x, y = await self.get_stage_position()
        try:
            z = await self.get_piezo_position()
        except Exception:
            z = None
        return {"success": True, "x": x, "y": y, "z": z}

    async def _plan_acquire(self, **params) -> dict:
        return await self.acquire_volume(**params)

    async def _plan_snap(self, **params) -> dict:
        return await self.capture_lightsheet_image(**params)

    async def _plan_calibrate(self, **params) -> dict:
        return await self.calibrate_piezo_galvo(**params)

    async def _plan_detect(self, **params) -> dict:
        return await self.detect_embryos(**params)

    async def _plan_detect_image(self, **params) -> dict:
        return await self.capture_bottom_image(**params)

    async def _plan_set_illumination(self, state: str = "Closed", **kw) -> dict:
        return await self.set_led(state)

    async def _plan_get_illumination(self, **kw) -> dict:
        return await self.get_led_status()

    async def _plan_status(self, **kw) -> dict:
        return await self._get_server_status()


# Backward-compat alias
QueueServerClient = DiSPIMMicroscope


async def create_queue_server_client(
    http_url: str = f"http://{settings.network.device_host}:{settings.network.device_port}",
) -> Optional[DiSPIMMicroscope]:
    """Create and connect a diSPIM microscope client."""
    client = DiSPIMMicroscope(http_url=http_url)
    if await client.connect():
        return client
    return None
