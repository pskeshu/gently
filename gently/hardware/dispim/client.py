"""
diSPIM Microscope — HTTP client for the diSPIM device layer.

Implements the Microscope protocol via HTTP requests to the unified
device_layer.py server (Bluesky plans + SAM detection).
"""

import asyncio
import logging
import traceback
from typing import Any

import aiohttp
import numpy as np

from gently.core.coordinates import (
    DEFAULT_OBJECTIVE_MAG,
    DEFAULT_PIXEL_SIZE_UM,
)
from gently.exceptions import DeviceLayerError, NetworkError
from gently.harness.microscope import Microscope
from gently.settings import settings

logger = logging.getLogger(__name__)


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

        self._session: aiohttp.ClientSession | None = None  # aiohttp session
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
            assert self._session is not None
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
                assert self._session is not None
                async with self._session.get(f"{self.http_url}/api/sam/status") as resp:
                    if resp.status == 200:
                        sam_status = await resp.json()
                        self._sam_available = sam_status.get("available", False)
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
            assert self._session is not None
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
            raise ConnectionError("Not connected to Microscope Server. Call connect() first.")

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
        assert self._session is not None
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
        from pathlib import Path

        import tifffile

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
        assert self._session is not None
        async with self._session.get(f"{self.http_url}{path}") as resp:
            return await resp.json()

    async def _api_post(self, path: str, json: dict | None = None) -> dict:
        """POST request using the shared session."""
        self._ensure_connected()
        assert self._session is not None
        async with self._session.post(f"{self.http_url}{path}", json=json) as resp:
            return await resp.json()

    async def _submit_plan_and_wait(
        self,
        plan_name: str,
        kwargs: dict | None = None,
        timeout: float = 120.0,
    ) -> dict:
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
            assert self._session is not None
            async with self._session.post(
                f"{self.http_url}/api/queue/item/add",
                json=payload,
                timeout=aiohttp.ClientTimeout(total=timeout),
            ) as resp:
                if resp.status != 200:
                    error_text = await resp.text()
                    return {
                        "success": False,
                        "error": f"HTTP {resp.status}: {error_text}",
                    }

                result = await resp.json()

                # Resolve file references (zero-copy transfer)
                if isinstance(result, dict):
                    docs = result.get("documents", {})
                    events = docs.get("events", [])
                    for event in events:
                        data = event.get("data", {})
                        for key, val in list(data.items()):
                            if self._is_file_ref(val):
                                arr, path = self._resolve_file_ref(val)
                                data[key] = arr
                                # Store path for downstream use
                                if "volume_path" not in result:
                                    result["volume_path"] = str(path)

                return result

        except asyncio.TimeoutError:
            return {
                "success": False,
                "error": f"Plan '{plan_name}' timed out after {timeout}s",
            }
        except aiohttp.ClientError as e:
            raise NetworkError(f"Device layer request failed: {e}") from e

    # =========================================================================
    # Stage Control
    # =========================================================================

    async def move_to_position(self, x: float, y: float) -> dict:
        """
        Move stage to absolute position.

        Parameters
        ----------
        x, y : float
            Target position in micrometers
        """
        logger.info("Moving to (%.1f, %.1f) µm", x, y)

        result = await self._submit_plan_and_wait(
            "move_stage_plan", kwargs={"xy_stage": "xy_stage", "x": x, "y": y}
        )

        if result.get("success"):
            return {"success": True, "x": x, "y": y}

        return result

    async def get_stage_position(self) -> tuple[float, float]:
        """
        Get current stage position.

        Returns
        -------
        tuple of (float, float)
            Current (x, y) position in micrometers
        """
        result = await self._submit_plan_and_wait(
            "read_stage_plan", kwargs={"xy_stage": "xy_stage"}
        )

        if result.get("success"):
            docs = result.get("documents", {})
            events = docs.get("events", [])
            if events:
                data = events[0].get("data", {})
                # Look for stage coordinates. Keys must match what
                # read_stage_plan (bp.count on the xy_stage device) actually
                # emits — the device-layer's own handle_detect_embryos uses
                # "XYStage:XY:31"/"xy_stage_position", so include those here too
                # (the bare "XY:31" was stale and never matched).
                for key in [
                    "xy_stage",
                    "XYStage:XY:31",
                    "xy_stage_position",
                    "XY:31",
                    "stage",
                ]:
                    if key in data:
                        val = data[key]
                        if isinstance(val, (list, tuple)) and len(val) >= 2:
                            return (float(val[0]), float(val[1]))
                        if isinstance(val, dict):
                            return (float(val.get("x", 0)), float(val.get("y", 0)))

        raise DeviceLayerError("Failed to read stage position")

    async def get_piezo_position(self) -> float:
        """
        Get current piezo Z position.

        Returns
        -------
        float
            Current Z position in micrometers
        """
        result = await self._submit_plan_and_wait("read_piezo_plan", kwargs={"piezo": "piezo"})

        if result.get("success"):
            docs = result.get("documents", {})
            events = docs.get("events", [])
            if events:
                data = events[0].get("data", {})
                for key in ["PiezoStage:P:34", "piezo", "z_stage"]:
                    if key in data:
                        val = data[key]
                        if isinstance(val, (int, float)):
                            return float(val)
                        if isinstance(val, dict):
                            return float(val.get("z", val.get("position", 0)) or 0)

        raise DeviceLayerError("Failed to read piezo position")

    # =========================================================================
    # Calibration
    # =========================================================================

    async def calibrate_piezo_galvo(
        self,
        piezo_positions: list[float] | None = None,
        galvo_positions: list[float] | None = None,
        **kwargs,
    ) -> dict:
        """
        Run piezo-galvo calibration plan.

        Returns
        -------
        dict
            Calibration results with optimal positions
        """
        plan_kwargs: dict[str, Any] = {"lightsheet_snap": "lightsheet_snap"}
        if piezo_positions is not None:
            plan_kwargs["piezo_positions"] = piezo_positions
        if galvo_positions is not None:
            plan_kwargs["galvo_positions"] = galvo_positions
        plan_kwargs.update(kwargs)

        result = await self._submit_plan_and_wait(
            "calibrate_piezo_galvo_plan", kwargs=plan_kwargs, timeout=300.0
        )

        if result.get("success"):
            return {
                "success": True,
                "calibration": result.get("calibration", {}),
            }

        return result

    # =========================================================================
    # Imaging
    # =========================================================================

    def _extract_image(
        self, result: dict, candidate_keys: list[str], multi_event: bool = False
    ) -> tuple | None:
        """Extract image array from plan result documents.

        Returns (array, path) or None if not found.
        """
        if not result.get("success"):
            return None

        docs = result.get("documents", {})
        events = docs.get("events", [])
        if not events:
            return None

        search_events = events if multi_event else [events[0]]
        for event in search_events:
            data = event.get("data", {})
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
        piezo_position: float | None = None,
        galvo_position: float | None = None,
        exposure_ms: float = 10.0,
        **kwargs,
    ) -> dict:
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
            ``{'image': np.ndarray, 'piezo_position': float,
            'galvo_position': float, 'success': bool}``
        """
        result = await self._submit_plan_and_wait(
            "capture_lightsheet_image_plan",
            kwargs={
                "lightsheet_snap": "lightsheet_snap",
                "scanner": "scanner",
                "piezo": "piezo",
                "laser_control": "laser_control",
                "piezo_position": piezo_position if piezo_position is not None else 50.0,
                "galvo_position": galvo_position if galvo_position is not None else 0.0,
            },
            timeout=30.0,
        )

        extracted = self._extract_image(result, ["HamCam1", "lightsheet_snap", "camera"])
        if extracted:
            arr, fpath = extracted
            ret = {
                "image": arr,
                "piezo_position": piezo_position or 0.0,
                "galvo_position": galvo_position or 0.0,
                "success": True,
            }
            if fpath:
                ret["image_path"] = fpath
            return ret

        return {"error": result.get("error", "No image data"), "success": False}

    async def acquire_volume(
        self,
        num_slices: int = 50,
        exposure_ms: float = 10.0,
        galvo_amplitude: float = 0.5,
        galvo_center: float = 0.0,
        piezo_amplitude: float = 25.0,
        piezo_center: float = 50.0,
        laser_config: str | None = None,
        laser_power_488_pct: float | None = None,
        laser_power_561_pct: float | None = None,
        laser_power_405_pct: float | None = None,
        laser_power_637_pct: float | None = None,
        **kwargs,
    ) -> dict:
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
        laser_power_488_pct, laser_power_561_pct, laser_power_405_pct,
        laser_power_637_pct : float, optional
            Per-line laser power %. Hard-limited at the device layer
            (DiSPIMLightSource.POWER_LIMITS_PCT). None leaves current
            setpoint untouched.

        Returns
        -------
        dict
            ``{'volume': np.ndarray, 'shape': tuple, 'success': bool}``
        """
        plan_kwargs = {
            "volume_scanner": "volume_scanner",
            "num_slices": num_slices,
            "exposure_ms": exposure_ms,
            "galvo_amplitude": galvo_amplitude,
            "galvo_center": galvo_center,
            "piezo_amplitude": piezo_amplitude,
            "piezo_center": piezo_center,
        }
        # Only forward kwargs the user explicitly set — leaves the
        # acquire_single_volume_plan defaults in place otherwise.
        if laser_config is not None:
            plan_kwargs["laser_config"] = laser_config
        if laser_power_488_pct is not None:
            plan_kwargs["laser_power_488_pct"] = laser_power_488_pct
        if laser_power_561_pct is not None:
            plan_kwargs["laser_power_561_pct"] = laser_power_561_pct
        if laser_power_405_pct is not None:
            plan_kwargs["laser_power_405_pct"] = laser_power_405_pct
        if laser_power_637_pct is not None:
            plan_kwargs["laser_power_637_pct"] = laser_power_637_pct

        result = await self._submit_plan_and_wait(
            "acquire_single_volume_plan", kwargs=plan_kwargs, timeout=120.0
        )

        extracted = self._extract_image(
            result, ["volume_scanner", "camera", "camera_image"], multi_event=True
        )
        if extracted:
            arr, fpath = extracted
            ret = {
                "volume": arr,
                "shape": arr.shape,
                "success": True,
            }
            if fpath:
                ret["volume_path"] = str(fpath)
            elif result.get("volume_path"):
                ret["volume_path"] = result["volume_path"]
            return ret

        return {"error": result.get("error", "Acquisition failed"), "success": False}

    async def acquire_burst(
        self,
        frames: int = 60,
        mode: str = "1hz",
        num_slices: int = 1,
        exposure_ms: float = 5.0,
        galvo_amplitude: float = 0.5,
        galvo_center: float = 0.0,
        piezo_amplitude: float = 25.0,
        piezo_center: float = 50.0,
        laser_config: str | None = None,
        laser_power_488_pct: float | None = None,
        laser_power_561_pct: float | None = None,
        laser_power_405_pct: float | None = None,
        laser_power_637_pct: float | None = None,
        timeout: float | None = None,
    ) -> dict:
        """
        Acquire ``frames`` volumes back-to-back as a single device-layer plan.

        Submits ``burst_plan``, which holds MMCore (and ``pause_state_updates``)
        for the entire burst — no inter-frame race for the position pollers.
        Each frame becomes one Bluesky event with its own file-referenced
        volume.

        Parameters mirror :meth:`acquire_volume`. ``mode`` is ``"1hz"`` or
        ``"asap"``. ``timeout`` defaults to ``frames * 3.0`` seconds (enough
        headroom for 1 Hz cadence plus device overhead).

        Returns
        -------
        dict
            ``{'success': bool, 'frames': [...], 'frames_captured': int,
               'duration_s': float, 'sustained_hz': float}``
            where each entry in ``frames`` is ``{'volume': np.ndarray,
            'volume_path': str|None, 'shape': tuple}``.
        """
        plan_kwargs = {
            "volume_scanner": "volume_scanner",
            "frames": frames,
            "mode": mode,
            "num_slices": num_slices,
            "exposure_ms": exposure_ms,
            "galvo_amplitude": galvo_amplitude,
            "galvo_center": galvo_center,
            "piezo_amplitude": piezo_amplitude,
            "piezo_center": piezo_center,
        }
        if laser_config is not None:
            plan_kwargs["laser_config"] = laser_config
        if laser_power_488_pct is not None:
            plan_kwargs["laser_power_488_pct"] = laser_power_488_pct
        if laser_power_561_pct is not None:
            plan_kwargs["laser_power_561_pct"] = laser_power_561_pct
        if laser_power_405_pct is not None:
            plan_kwargs["laser_power_405_pct"] = laser_power_405_pct
        if laser_power_637_pct is not None:
            plan_kwargs["laser_power_637_pct"] = laser_power_637_pct

        if timeout is None:
            # 3 s/frame headroom (1 s pacing + ~1.5 s plan overhead) with a 60 s floor.
            timeout = max(60.0, frames * 3.0)

        result = await self._submit_plan_and_wait(
            "burst_plan",
            kwargs=plan_kwargs,
            timeout=timeout,
        )

        if not result.get("success"):
            return {"success": False, "error": result.get("error", "Burst failed")}

        # _submit_plan_and_wait already swapped file_refs for ndarrays in-place.
        # Walk every event and pull (volume, path) per frame.
        frames_out: list[dict] = []
        docs = result.get("documents", {}) or {}
        events = docs.get("events", []) or []
        candidates = ("volume_scanner", "camera", "camera_image")
        for ev in events:
            data = ev.get("data", {}) or {}
            for key in candidates:
                if key in data:
                    val = data[key]
                    entry: dict[str, Any] = {}
                    # Per-frame epoch time from the Bluesky event doc — lets the
                    # orchestrator stamp each saved frame with its real acquisition
                    # time instead of having to interpolate from the burst's
                    # aggregate timing.
                    ev_time = ev.get("time")
                    if ev_time is not None:
                        entry["acquired_at_epoch"] = float(ev_time)
                    if isinstance(val, np.ndarray):
                        entry["volume"] = val
                        entry["shape"] = val.shape
                    elif self._is_file_ref(val):
                        arr, path = self._resolve_file_ref(val)
                        entry["volume"] = arr
                        entry["shape"] = arr.shape
                        entry["volume_path"] = str(path)
                    else:
                        entry["volume"] = np.array(val)
                        entry["shape"] = entry["volume"].shape
                    if "__resolved_paths__" in data:
                        # _submit_plan_and_wait doesn't currently populate this
                        # for events, but support it if a future change does.
                        rp = data["__resolved_paths__"]
                        if isinstance(rp, dict) and key in rp:
                            entry["volume_path"] = str(rp[key])
                    frames_out.append(entry)
                    break

        # Pull aggregate timing from the stop document's exit_status / return.
        # burst_plan returns a dict but Bluesky doesn't ship it in the stop
        # doc; reconstruct from event timestamps as a fallback.
        duration_s = 0.0
        sustained_hz = 0.0
        if events:
            first_t = events[0].get("time")
            last_t = events[-1].get("time")
            if first_t is not None and last_t is not None and last_t > first_t:
                duration_s = float(last_t - first_t)
                if duration_s > 0:
                    sustained_hz = len(frames_out) / duration_s

        return {
            "success": True,
            "frames": frames_out,
            "frames_captured": len(frames_out),
            "frames_requested": frames,
            "duration_s": duration_s,
            "sustained_hz": sustained_hz,
            "mode": mode,
        }

    # =========================================================================
    # LED / Camera Controls
    # =========================================================================

    async def set_led(self, state: str = "Closed") -> dict:
        """Set LED state ('Open' or 'Closed')."""
        return await self._api_post("/api/led/set", {"state": state})

    async def set_laser_power(self, wavelength: int, pct: float) -> dict:
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
        return await self._api_post(
            "/api/light_source/power",
            {
                "wavelength": int(wavelength),
                "pct": float(pct),
            },
        )

    async def get_laser_power(self, wavelength: int) -> dict:
        """Read the current per-line laser power %.

        Hits ``GET /api/light_source/power?wavelength={wavelength}`` —
        direct, no queue.
        """
        try:
            assert self._session is not None
            async with self._session.get(
                f"{self.http_url}/api/light_source/power",
                params={"wavelength": int(wavelength)},
                timeout=5,
            ) as resp:
                return await resp.json()
        except Exception as e:
            return {"success": False, "error": str(e)}

    async def set_laser_config(self, config_name: str) -> dict:
        """Apply a Laser config-group preset (e.g. "ALL OFF").

        Hits ``POST /api/laser/config`` — direct, no Bluesky queue.
        Use ``"ALL OFF"`` to gate every laser line off via the PLogic
        OutputChannel; other presets from the MM config group are also
        accepted (e.g. ``"488 only"``, ``"561 only"``).

        Parameters
        ----------
        config_name : str
            Exact preset name from the Laser config group.
        """
        return await self._api_post("/api/laser/config", {"config": config_name})

    async def get_laser_configs(self) -> dict:
        """List available Laser config-group presets.

        Hits ``GET /api/laser/configs`` — returns ``{"configs": [...]}``
        with the preset names from the MM Laser config group.
        """
        return await self._api_get("/api/laser/configs")

    async def get_cameras(self) -> dict:
        """List available SPIM camera roles.

        Hits ``GET /api/cameras`` — returns ``{"cameras": ["A"]}`` on
        single-camera rigs or ``{"cameras": ["A", "B"]}`` when HamCam2 is
        registered as camera_b in the device layer.
        """
        return await self._api_get("/api/cameras")

    async def get_led_status(self) -> dict:
        """Get current LED status."""
        return await self._api_get("/api/led/status")

    async def set_room_light(self, state: str = "off") -> dict:
        """Switch the diSPIM room light on/off via the SwitchBot Bot.

        Hits ``POST /api/room_light/set`` directly (no Bluesky queue, no
        experiment trace) — a setup accessory poke. ``state`` is
        'on' | 'off' | 'press'. Blocks at the device layer until the BLE
        command lands (~1-2 s).
        """
        return await self._api_post("/api/room_light/set", {"state": state})

    async def get_room_light_status(self) -> dict:
        """Read the room light's cached on/off state (no BLE round-trip)."""
        return await self._api_get("/api/room_light/status")

    async def set_temperature(self, target_c: float) -> dict:
        """Command the thermal-controller setpoint (Celsius). Non-blocking — the
        controller ramps; poll get_temperature() for the lock state."""
        return await self._api_post("/api/temperature/set", {"target_c": target_c})

    async def get_temperature(self) -> dict:
        """Get current temperature, setpoint, and lock state."""
        return await self._api_get("/api/temperature/status")

    async def get_temperature_config(self) -> dict:
        """Get the thermalizer connection config (password redacted) + live state."""
        return await self._api_get("/api/temperature/config")

    async def set_temperature_config(self, cfg: dict) -> dict:
        """Reconfigure the thermalizer (serial/mqtt/mock) — live hot-swap where
        possible, else persisted for the next device-layer restart."""
        return await self._api_post("/api/temperature/config", cfg)

    async def test_temperature_config(self, cfg: dict) -> dict:
        """Probe a candidate thermalizer config without committing it."""
        return await self._api_post("/api/temperature/config/test", cfg)

    # ------------------------------------------------------------------
    # Live device-state readout (streamed from the device layer poller)
    # ------------------------------------------------------------------
    async def get_device_state(self, refresh: bool = False) -> dict:
        """One-shot snapshot of all device positions + properties.

        Parameters
        ----------
        refresh : bool
            If True, force the device layer to re-read MMCore right now.
            Otherwise return the most recent poller snapshot (typically <500 ms old).
        """
        path = "/api/devices/state"
        if refresh:
            path += "?refresh=1"
        return await self._api_get(path)

    async def stream_device_states(self, timeout: float | None = None):
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
        assert self._session is not None
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

    async def stream_bottom_camera(self, timeout: float | None = None):
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
        assert self._session is not None
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

    async def stream_lightsheet(self, timeout: float | None = None):
        """Async generator yielding JPEG frames from the lightsheet live SSE stream.

        Mirrors :meth:`stream_bottom_camera`; subscriber-gated on the device layer.
        """
        self._ensure_connected()
        client_timeout = aiohttp.ClientTimeout(
            total=None,
            sock_read=timeout,
            sock_connect=10.0,
        )
        url = f"{self.http_url}/api/lightsheet/stream"
        assert self._session is not None
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
                        logger.warning("Malformed lightsheet SSE payload skipped: %s", exc)

    async def set_lightsheet_live_params(
        self, galvo=None, piezo=None, exposure=None, side=None
    ) -> dict:
        """POST live galvo/piezo/exposure/side to the device-layer lightsheet streamer."""
        body: dict[str, float | str] = {}
        if galvo is not None:
            body["galvo"] = float(galvo)
        if piezo is not None:
            body["piezo"] = float(piezo)
        if exposure is not None:
            body["exposure"] = float(exposure)
        if side is not None:
            body["side"] = str(side)
        return await self._api_post("/api/lightsheet/live/params", body)

    async def set_camera_led_mode(self, use_led: bool = False) -> dict:
        """Enable/disable automatic LED for bottom camera captures."""
        return await self._api_post("/api/camera/led_mode", {"use_led": use_led})

    async def set_bottom_camera_exposure(self, exposure_ms: float) -> dict:
        """Set bottom camera exposure time in milliseconds."""
        return await self._api_post("/api/camera/exposure", {"exposure_ms": exposure_ms})

    async def get_bottom_camera_exposure(self) -> dict:
        """Get current bottom camera exposure time."""
        return await self._api_get("/api/camera/exposure")

    # ------------------------------------------------------------------
    # Fenced focus axes — bottom-camera focus Z and SPIM-head F-drive.
    # Read + relative nudge only (no autofocus). Out-of-range nudges come
    # back as a non-success dict (the device layer returns 400).
    # ------------------------------------------------------------------

    async def get_bottom_z(self) -> dict:
        """Current bottom-camera focus Z position + limits."""
        return await self._api_get("/api/stage/bottom_z")

    async def nudge_bottom_z(self, delta: float) -> dict:
        """Fenced relative move of the bottom-camera focus Z by ``delta`` µm."""
        return await self._api_post("/api/stage/bottom_z/nudge", {"delta": float(delta)})

    async def get_fdrive(self) -> dict:
        """Current SPIM-head F-drive position + limits + distance to floor."""
        return await self._api_get("/api/spim/fdrive")

    async def nudge_fdrive(self, delta: float) -> dict:
        """Fenced relative move of the SPIM-head F-drive by ``delta`` µm."""
        return await self._api_post("/api/spim/fdrive/nudge", {"delta": float(delta)})

    async def capture_bottom_image(
        self, use_led: bool = False, exposure_ms: float | None = None
    ) -> dict:
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
            "capture_bottom_image_plan", kwargs={"bottom_camera": "bottom_camera"}
        )

        extracted = self._extract_image(
            result, ["bottom_camera", "bottom_camera_image", "Bottom PCO"]
        )
        if extracted:
            arr, fpath = extracted
            return {"image": arr, "image_path": fpath}

        return {"image": np.zeros((100, 100), dtype=np.uint16), "image_path": None}

    # =========================================================================
    # SAM Embryo Detection (HTTP API)
    # =========================================================================

    async def detect_embryos(
        self,
        pixel_size_um: float = DEFAULT_PIXEL_SIZE_UM,
        objective_mag: float = DEFAULT_OBJECTIVE_MAG,
        use_claude_review: bool = True,
        min_confidence: float = 0.7,
        exposure_ms: float | None = None,
        brightness_percentile: float = 99.0,
        min_area: int = 5000,
        max_area: int = 150000,
    ) -> dict:
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
            return {"error": "SAM detection not available on server"}

        self._ensure_connected()

        try:
            logger.info("Calling /api/detect_embryos (server-side capture + SAM)...")

            payload = {
                "pixel_size_um": pixel_size_um,
                "objective_mag": objective_mag,
                "use_claude_review": use_claude_review,
                "min_confidence": min_confidence,
                "brightness_percentile": brightness_percentile,
                "min_area": min_area,
                "max_area": max_area,
            }
            if exposure_ms is not None:
                payload["exposure_ms"] = exposure_ms

            assert self._session is not None
            async with self._session.post(
                f"{self.http_url}/api/detect_embryos", json=payload, timeout=300
            ) as resp:
                result = await resp.json()

            if not result.get("success"):
                return result

            # Ensure the caller has the image to feed into the map view.
            if result.get("image") is None:
                image = await self._get_detection_image(result, exposure_ms)
                if image is not None:
                    result["image"] = image

            return result

        except asyncio.TimeoutError:
            return {"success": False, "error": "Detection timed out (5 min limit)"}
        except aiohttp.ClientError as e:
            return {
                "error": str(NetworkError(f"Device layer request failed: {e}")),
                "traceback": traceback.format_exc(),
                "success": False,
            }
        except Exception as e:
            return {
                "error": str(e),
                "traceback": traceback.format_exc(),
                "success": False,
            }

    async def _get_detection_image(
        self, detection_result: dict, exposure_ms: float | None = None
    ) -> np.ndarray | None:
        """Load or capture an image for the detection editor."""
        image_path = detection_result.get("image_path")
        if image_path:
            try:
                import tifffile

                return tifffile.imread(image_path)
            except Exception:
                pass
        # Fallback: capture a fresh image
        snap = await self.capture_bottom_image(exposure_ms=exposure_ms)
        image = snap["image"]
        if snap.get("image_path"):
            try:
                snap["image_path"].unlink(missing_ok=True)
            except OSError:
                pass
        return image

    async def view_image(
        self,
        image: np.ndarray | None = None,
        title: str = "Image View",
        exposure_ms: float | None = None,
        save_path: str | None = None,
        show: bool = True,
        embryo_annotations: list | None = None,
    ) -> dict:
        """Save a bottom-camera image to disk (replaces the napari display).

        ``show`` and ``title`` are kept for backwards compatibility with
        existing tool call sites but only the disk save still happens.
        Embryo annotations are drawn into the saved PNG if provided.
        """
        try:
            if image is None:
                snap = await self.capture_bottom_image(exposure_ms=exposure_ms)
                image = snap["image"]
                if snap.get("image_path"):
                    try:
                        snap["image_path"].unlink(missing_ok=True)
                    except OSError:
                        pass

            if image is None:
                return {"success": False, "error": "No image to save"}

            result = {"success": True, "shape": list(image.shape)}
            if save_path:
                # Reuse the existing PNG writer; draws annotations if any.
                from pathlib import Path as _Path

                _Path(save_path).parent.mkdir(parents=True, exist_ok=True)
                annots = []
                for a in embryo_annotations or []:
                    px = a.get("pixel_x")
                    py = a.get("pixel_y")
                    if px is None or py is None:
                        continue
                    annots.append(
                        {
                            "embryo_number": a.get("label") or a.get("embryo_id") or "?",
                            "pixel_position": (px, py),
                        }
                    )
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
                result["saved_to"] = save_path
            return result
        except Exception as e:
            return {
                "error": str(e),
                "traceback": traceback.format_exc(),
                "success": False,
            }

    async def view_embryos(
        self,
        image: np.ndarray,
        embryos: list[dict],
        title: str = "Embryos",
        save_path: str | None = None,
        show: bool = True,
    ) -> dict:
        """Save an annotated PNG of embryos on an image (replaces napari).

        Markers in ``embryos`` may use ``center_x``/``center_y`` or
        ``pixel_x``/``pixel_y``. ``show`` is ignored — display is now
        the map view's job.
        """
        try:
            if image is None or not embryos:
                return {"success": False, "error": "No image or embryos to display"}

            annots = []
            for emb in embryos:
                px = emb.get("center_x", emb.get("pixel_x", 0))
                py = emb.get("center_y", emb.get("pixel_y", 0))
                annots.append(
                    {
                        "embryo_number": emb.get("embryo_id", "?"),
                        "pixel_position": (px, py),
                    }
                )

            result: dict[str, Any] = {"success": True, "num_embryos": len(embryos)}
            if save_path:
                from pathlib import Path as _Path

                _Path(save_path).parent.mkdir(parents=True, exist_ok=True)
                from gently.ui.web.embryo_marker import _save_marked_image

                _save_marked_image(image, annots, _Path(save_path))
                result["saved_to"] = save_path
            return result
        except Exception as e:
            return {
                "error": str(e),
                "traceback": traceback.format_exc(),
                "success": False,
            }

    async def capture_for_marking(
        self,
        exposure_ms: float | None = None,
    ) -> dict:
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
            # No LED ever — the bottom camera images under room light only.
            snap = await self.capture_bottom_image(use_led=False, exposure_ms=exposure_ms)
            image = snap["image"]

            if image is None or (image.shape == (100, 100) and image.max() == 0):
                return {"success": False, "error": "Failed to capture image"}

            if snap.get("image_path"):
                try:
                    snap["image_path"].unlink(missing_ok=True)
                except OSError:
                    pass

            stage_pos = await self.get_stage_position()

            return {
                "success": True,
                "image": image,
                "stage_position": list(stage_pos),
                "image_shape": list(image.shape),
            }

        except Exception as e:
            return {
                "error": str(e),
                "traceback": traceback.format_exc(),
                "success": False,
            }

    # =========================================================================
    # Status
    # =========================================================================

    async def _get_server_status(self) -> dict:
        """Query device layer and SAM server status via HTTP."""
        status = {}

        # Device Layer Server status
        if self._session:
            try:
                assert self._session is not None
                async with self._session.get(f"{self.http_url}/api/status") as resp:
                    if resp.status == 200:
                        server_status = await resp.json()
                        status["queue_server"] = {
                            "manager_state": server_status.get("manager_state"),
                            "re_state": server_status.get("re_state", "idle"),
                            "devices": server_status.get("devices", []),
                            "plans": server_status.get("plans", []),
                        }
                    else:
                        status["queue_server"] = {"error": f"HTTP {resp.status}"}
            except (aiohttp.ClientError, Exception) as e:
                status["queue_server"] = {"error": str(e)}
        else:
            status["queue_server"] = {"connected": False}

        # SAM status (via HTTP, same server)
        if self._session and self._qs_connected:
            try:
                assert self._session is not None
                async with self._session.get(f"{self.http_url}/api/sam/status") as resp:
                    if resp.status == 200:
                        sam_status = await resp.json()
                        status["sam_server"] = {
                            "available": sam_status.get("available", False),
                            "loaded": sam_status.get("loaded", False),
                            "device": sam_status.get("device", "unknown"),
                        }
                    else:
                        status["sam_server"] = {"error": f"HTTP {resp.status}"}
            except (aiohttp.ClientError, Exception) as e:
                status["sam_server"] = {"error": str(e)}
        else:
            status["sam_server"] = {"connected": False}

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
) -> DiSPIMMicroscope | None:
    """Create and connect a diSPIM microscope client."""
    client = DiSPIMMicroscope(http_url=http_url)
    if await client.connect():
        return client
    return None
