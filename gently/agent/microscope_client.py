"""
Microscope Client for RPC communication with the microscope server

This module provides an async-compatible client for communicating with
the microscope server via rpyc. All hardware operations happen server-side,
and this client provides a clean interface for the copilot.
"""

import asyncio
from typing import Any, Dict, List, Optional, Tuple
import numpy as np


class MicroscopeClient:
    """
    RPC client for microscope server

    Provides async wrappers around rpyc calls to the microscope server.
    All heavy operations (RunEngine, devices, SAM) run on the server.

    Example
    -------
    >>> client = MicroscopeClient(host='localhost', port=18861)
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

    def __init__(self, host: str = 'localhost', port: int = 18861):
        """
        Parameters
        ----------
        host : str
            Server hostname
        port : int
            Server port
        """
        self.host = host
        self.port = port
        self._conn = None
        self._server = None

    async def connect(self) -> bool:
        """
        Connect to the microscope server

        Returns
        -------
        bool
            True if connection successful
        """
        try:
            import rpyc

            # rpyc connection is blocking, so run in thread
            def _connect():
                config = {
                    'allow_all_attrs': True,
                    'allow_pickle': True,
                    'sync_request_timeout': 300,  # 5 minute timeout for long operations
                }
                conn = rpyc.connect(self.host, self.port, config=config)
                return conn

            self._conn = await asyncio.to_thread(_connect)
            self._server = self._conn.root
            return True

        except Exception:
            return False

    def disconnect(self):
        """Disconnect from server"""
        if self._conn:
            self._conn.close()
            self._conn = None
            self._server = None

    @property
    def is_connected(self) -> bool:
        """Check if connected to server"""
        return self._conn is not None and not self._conn.closed

    def _ensure_connected(self):
        """Raise error if not connected"""
        if not self.is_connected:
            raise ConnectionError("Not connected to microscope server. Call connect() first.")

    # === Stage Operations ===

    async def move_to_position(self, x: float, y: float) -> Dict:
        """
        Move XY stage to position

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
        self._ensure_connected()
        result = await asyncio.to_thread(
            self._server.move_to_position, x, y
        )
        return dict(result) if result else {}

    async def get_stage_position(self) -> Tuple[float, float]:
        """
        Get current stage position

        Returns
        -------
        tuple
            (x, y) position in micrometers
        """
        self._ensure_connected()
        result = await asyncio.to_thread(self._server.get_stage_position)
        return tuple(result)

    # === Calibration Operations ===

    async def calibrate_piezo_galvo(self, piezo_positions: List[float] = None) -> Dict:
        """
        Run piezo-galvo calibration

        Parameters
        ----------
        piezo_positions : list of float, optional
            Piezo positions to use for calibration

        Returns
        -------
        dict
            Calibration results with slope, offset, rmse
        """
        self._ensure_connected()
        if piezo_positions is None:
            piezo_positions = [40.0, 60.0]

        result = await asyncio.to_thread(
            self._server.calibrate_piezo_galvo, piezo_positions
        )
        return dict(result) if result else {}

    # === Acquisition Operations ===

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
        Acquire a single volume

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
        self._ensure_connected()
        result = await asyncio.to_thread(
            self._server.acquire_volume,
            num_slices, exposure_ms,
            galvo_amplitude, galvo_center,
            piezo_amplitude, piezo_center
        )

        # Convert rpyc netref to local numpy array
        if result and 'volume' in result:
            result = dict(result)
            result['volume'] = np.array(result['volume'])

        return result if result else {}

    async def capture_bottom_image(self) -> np.ndarray:
        """
        Capture image from bottom camera

        Returns
        -------
        np.ndarray
            2D image array
        """
        self._ensure_connected()
        result = await asyncio.to_thread(self._server.capture_bottom_image)
        return np.array(result)

    # === Embryo Detection ===

    async def detect_embryos(
        self,
        pixel_size_um: float = 6.5,
        objective_mag: float = 4.0,
        use_claude_review: bool = True,
        min_confidence: float = 0.7,
    ) -> Dict:
        """
        Detect embryos using SAM + Claude Vision

        All processing happens server-side.

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

        Returns
        -------
        dict
            Detection results with 'embryos' list
        """
        self._ensure_connected()

        try:
            result = await asyncio.to_thread(
                self._server.detect_embryos,
                pixel_size_um, objective_mag,
                use_claude_review, min_confidence
            )

            # Convert rpyc netref to local dict
            if result:
                result = dict(result)
                if 'embryos' in result:
                    result['embryos'] = [dict(e) for e in result['embryos']]
                if 'image' in result and result['image'] is not None:
                    result['image'] = np.array(result['image'])

            return result if result else {}

        except Exception as e:
            import traceback
            return {
                'error': f"RPC call failed: {str(e)}",
                'traceback': traceback.format_exc()
            }

    # === Acquisition Control ===

    async def pause_acquisition(self) -> Dict:
        """Pause running acquisition"""
        self._ensure_connected()
        result = await asyncio.to_thread(self._server.pause_acquisition)
        return dict(result) if result else {}

    async def resume_acquisition(self) -> Dict:
        """Resume paused acquisition"""
        self._ensure_connected()
        result = await asyncio.to_thread(self._server.resume_acquisition)
        return dict(result) if result else {}

    async def abort_acquisition(self) -> Dict:
        """Abort running acquisition"""
        self._ensure_connected()
        result = await asyncio.to_thread(self._server.abort_acquisition)
        return dict(result) if result else {}

    # === Status ===

    async def get_status(self) -> Dict:
        """
        Get server status

        Returns
        -------
        dict
            Status information including device states
        """
        self._ensure_connected()
        result = await asyncio.to_thread(self._server.get_status)
        return dict(result) if result else {}


async def create_microscope_client(host: str = 'localhost', port: int = 18861) -> Optional[MicroscopeClient]:
    """
    Create and connect a microscope client

    Parameters
    ----------
    host : str
        Server hostname
    port : int
        Server port

    Returns
    -------
    MicroscopeClient or None
        Connected client, or None if connection failed
    """
    client = MicroscopeClient(host=host, port=port)
    if await client.connect():
        return client
    return None
