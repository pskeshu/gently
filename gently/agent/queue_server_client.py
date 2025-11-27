"""
Queue Server Client for MicroscopyCopilot

Provides async interface to:
1. Simple Microscope Server (HTTP API) for hardware control via plans
2. SAM Detection Server (rpyc) for embryo detection

This client connects to our simple_server.py which runs RunEngine
in the main thread, avoiding Windows threading issues.
"""

import asyncio
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import aiohttp


class QueueServerClient:
    """
    Client for Simple Microscope Server + SAM Server.

    Provides async methods for:
    - Submitting Bluesky plans to the server
    - Waiting for plan completion
    - Retrieving results
    - Running SAM embryo detection via separate server

    Example
    -------
    >>> client = QueueServerClient(http_url="http://127.0.0.1:60610", sam_port=18862)
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
        http_url: str = "http://127.0.0.1:60610",
        sam_host: str = "localhost",
        sam_port: int = 18862,
        databroker_catalog: str = "dispim_production",
    ):
        """
        Parameters
        ----------
        http_url : str
            Simple Microscope Server HTTP API URL
        sam_host : str
            SAM server hostname
        sam_port : int
            SAM server port
        databroker_catalog : str
            Databroker catalog name (for future use)
        """
        self.http_url = http_url
        self.sam_host = sam_host
        self.sam_port = sam_port
        self.databroker_catalog = databroker_catalog

        self._session = None  # aiohttp session
        self._sam_conn = None  # rpyc connection
        self._db = None  # databroker catalog
        self._qs_connected = False  # Track actual queue server connection

        # Store last detection results for visualization
        self._last_detection = None  # {image, embryos, stage_position}

    async def connect(self) -> bool:
        """
        Connect to Simple Server and SAM Server.

        Returns
        -------
        bool
            True if all connections successful
        """
        import aiohttp

        self._qs_connected = False  # Track actual queue server connection

        # Create aiohttp session
        self._session = aiohttp.ClientSession()

        # Connect to Simple Microscope Server (HTTP API)
        try:
            async with self._session.get(f"{self.http_url}/api/status") as resp:
                if resp.status == 200:
                    await resp.json()  # Validate response
                    self._qs_connected = True
                else:
                    raise Exception(f"HTTP {resp.status}")
        except Exception:
            self._qs_connected = False

        # Connect to SAM Server (rpyc)
        try:
            import rpyc

            config = {
                'allow_all_attrs': True,
                'allow_pickle': True,
                'sync_request_timeout': 300,
            }
            self._sam_conn = rpyc.connect(
                self.sam_host, self.sam_port, config=config
            )
        except Exception:
            self._sam_conn = None
            # Don't fail completely - SAM is optional

        # Connect to Databroker catalog (v1 style using sqlite)
        try:
            from databroker import Broker
            from pathlib import Path
            import os
            import yaml
            import warnings

            # Load from v1 config file directly (bypasses intake which uses msgpack)
            config_path = Path(os.path.expanduser("~")) / ".config" / "databroker" / f"{self.databroker_catalog}.yml"
            if config_path.exists():
                config = yaml.safe_load(config_path.read_text())
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")  # Suppress v0 fallback warning
                    self._db = Broker.from_config(config)
            else:
                self._db = None
        except Exception:
            self._db = None

        return self._qs_connected

    async def disconnect(self):
        """Disconnect from servers"""
        if self._sam_conn:
            self._sam_conn.close()
            self._sam_conn = None
        if self._session:
            await self._session.close()
            self._session = None

    @property
    def is_connected(self) -> bool:
        """Check if connected to Microscope Server"""
        return self._qs_connected

    @property
    def has_sam(self) -> bool:
        """Check if SAM server is available"""
        return self._sam_conn is not None

    @property
    def has_databroker(self) -> bool:
        """Check if Databroker catalog is available"""
        return self._db is not None

    def _ensure_connected(self):
        """Raise error if not connected"""
        if not self.is_connected:
            raise ConnectionError(
                "Not connected to Microscope Server. Call connect() first."
            )

    # =========================================================================
    # Server Operations
    # =========================================================================

    async def _submit_plan_and_wait(
        self,
        plan_name: str,
        kwargs: Dict = None,
        timeout: float = 300
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
            # Try to extract position from documents
            docs = result.get('documents', {})
            events = docs.get('events', [])
            if events:
                data = events[0].get('data', {})
                # Debug: print available keys
                print(f"    Stage data keys: {list(data.keys())}")

                # Try to find x and y values
                x, y = 0.0, 0.0

                # Look for separate x and y components
                for key in data.keys():
                    if 'x' in key.lower() and 'y' not in key.lower():
                        x = float(data[key]) if not isinstance(data[key], list) else float(data[key][0])
                    elif 'y' in key.lower() and 'x' not in key.lower():
                        y = float(data[key]) if not isinstance(data[key], list) else float(data[key][0])

                # If found, return
                if x != 0.0 or y != 0.0:
                    return (x, y)

                # Try combined position key
                for key in ['xy_stage', 'XYStage:XY:31', 'xy_stage_position']:
                    if key in data:
                        val = data[key]
                        if isinstance(val, (list, tuple)) and len(val) >= 2:
                            return (float(val[0]), float(val[1]))
                        elif isinstance(val, (int, float)):
                            return (float(val), 0.0)

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
            timeout=300
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
            timeout=60
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
                        return {
                            'image': np.array(image_data),
                            'piezo_position': piezo_position,
                            'galvo_position': galvo_position,
                            'run_uid': run_uid,
                            'success': True
                        }

            # Fallback: try databroker if response didn't have image data
            if run_uid and self._db:
                try:
                    run = self._db[run_uid]
                    data = run.primary.read()
                    if 'lightsheet_snap' in data:
                        image = data['lightsheet_snap'].values[0]
                        return {
                            'image': np.array(image),
                            'piezo_position': piezo_position,
                            'galvo_position': galvo_position,
                            'run_uid': run_uid,
                            'success': True
                        }
                except Exception as e:
                    print(f"  Warning: Could not read image from databroker: {e}")

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
                for event in events:
                    data = event.get('data', {})
                    for key in ['volume_scanner', 'camera', 'camera_image']:
                        if key in data:
                            images.append(data[key])
                            break
                if images:
                    volume = np.array(images)
                    return {
                        'volume': volume,
                        'shape': volume.shape,
                        'success': True
                    }

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
                        return np.array(data[key])

        return np.zeros((100, 100), dtype=np.uint16)

    # =========================================================================
    # SAM Embryo Detection
    # =========================================================================

    async def detect_embryos(
        self,
        pixel_size_um: float = 6.5,
        objective_mag: float = 4.0,
        use_claude_review: bool = True,
        min_confidence: float = 0.7,
        exposure_ms: float = None,
        brightness_percentile: float = 99.0,
        min_area: int = 5000,
        max_area: int = 150000,
    ) -> Dict:
        """
        Capture image and detect embryos using brightness detection + SAM.

        This method:
        1. Captures image via Queue Server (Bluesky plan)
        2. Reads stage position via Queue Server
        3. Uses brightness detection to find candidates
        4. Uses SAM with bounding box prompts for precise segmentation

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

        Returns
        -------
        dict
            Detection results with 'embryos' list
        """
        # Check SAM server connection
        if not self.has_sam:
            return {'error': 'SAM Server not connected'}

        try:
            # Capture image via queue server
            if exposure_ms:
                print(f"  Setting exposure to {exposure_ms} ms for better contrast...")
            print("  Capturing bottom camera image...")
            image = await self.capture_bottom_image(exposure_ms=exposure_ms)
            if image.shape == (100, 100):
                return {'error': 'Failed to capture image'}

            print(f"  Image shape: {image.shape}")

            # Get stage position
            print("  Reading stage position...")
            stage_pos = await self.get_stage_position()
            print(f"  Stage position: {stage_pos}")

            # Run brightness + SAM detection via rpyc
            print("  Running brightness + SAM detection...")
            print(f"  Parameters: brightness_percentile={brightness_percentile}, area={min_area}-{max_area}")
            sam_result = await asyncio.to_thread(
                self._sam_conn.root.detect_embryos,
                image,
                stage_pos,
                pixel_size_um=pixel_size_um,
                objective_mag=objective_mag,
                use_claude_review=use_claude_review,
                min_confidence=min_confidence,
                save_visualizations=True,
                output_dir="./detection_results",
                brightness_percentile=brightness_percentile,
                min_area=min_area,
                max_area=max_area
            )

            # Convert rpyc netref to dict
            result = dict(sam_result)
            if 'embryos' in result:
                result['embryos'] = [dict(e) for e in result['embryos']]

            result['image'] = image
            result['stage_position'] = list(stage_pos)

            # Store for later visualization
            self._last_detection = {
                'image': image,
                'embryos': result.get('embryos', []),
                'stage_position': list(stage_pos)
            }

            return result

        except Exception as e:
            import traceback
            return {
                'error': str(e),
                'traceback': traceback.format_exc(),
                'success': False
            }

    # Alias for compatibility with existing copilot code
    capture_and_detect_embryos = detect_embryos

    async def manual_mark_embryos(
        self,
        pixel_size_um: float = 6.5,
        objective_mag: float = 4.0,
        exposure_ms: float = None,
        existing_embryos: list = None,
    ) -> Dict:
        """
        Capture image and manually mark embryos via matplotlib.

        Opens a matplotlib window where user can click on embryo centers.
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
        if not self.has_sam:
            return {'error': 'SAM Server not connected (needed for manual marking)'}

        try:
            # Capture image via queue server
            if exposure_ms:
                print(f"  Setting exposure to {exposure_ms} ms...")
            print("  Capturing bottom camera image...")
            image = await self.capture_bottom_image(exposure_ms=exposure_ms)
            if image.shape == (100, 100):
                return {'error': 'Failed to capture image'}

            print(f"  Image shape: {image.shape}")

            # Get stage position
            print("  Reading stage position...")
            stage_pos = await self.get_stage_position()
            print(f"  Stage position: {stage_pos}")

            # Convert existing embryos to pixel positions for display
            display_embryos = None
            if existing_embryos:
                um_per_pixel = pixel_size_um / objective_mag
                image_center_x = image.shape[1] / 2
                image_center_y = image.shape[0] / 2

                display_embryos = []
                for emb in existing_embryos:
                    stage_x = emb.get('stage_x', emb.get('x', 0))
                    stage_y = emb.get('stage_y', emb.get('y', 0))

                    # Convert stage to pixel
                    dx_stage = stage_x - stage_pos[0]
                    dy_stage = stage_y - stage_pos[1]
                    pixel_x = image_center_x + dx_stage / um_per_pixel
                    pixel_y = image_center_y + dy_stage / um_per_pixel

                    display_embryos.append({
                        'embryo_id': emb.get('embryo_id', '?'),
                        'pixel_x': pixel_x,
                        'pixel_y': pixel_y,
                    })
                print(f"  Showing {len(display_embryos)} existing embryos")

            # Run manual marking via rpyc (opens matplotlib window)
            print("  Opening manual marking window...")
            result = await asyncio.to_thread(
                self._sam_conn.root.manual_mark_embryos,
                image,
                stage_pos,
                pixel_size_um=pixel_size_um,
                objective_mag=objective_mag,
                existing_embryos=display_embryos
            )

            # Convert rpyc netref to dict
            result = dict(result)
            if 'embryos' in result:
                result['embryos'] = [dict(e) for e in result['embryos']]

            result['image'] = image
            result['stage_position'] = list(stage_pos)

            return result

        except Exception as e:
            import traceback
            return {
                'error': str(e),
                'traceback': traceback.format_exc(),
                'success': False
            }

    async def view_image(self, image: np.ndarray = None, title: str = "Image View",
                         exposure_ms: float = None, save_path: str = None,
                         show: bool = True) -> Dict:
        """
        View an image via the SAM server's matplotlib window.

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
            Whether to show in matplotlib window (blocking). Default: True

        Returns
        -------
        dict
            Success status and save_path if saved
        """
        if not self.has_sam:
            return {'error': 'SAM Server not connected'}

        try:
            if image is None:
                if exposure_ms:
                    print(f"  Setting exposure to {exposure_ms} ms...")
                print("  Capturing image...")
                image = await self.capture_bottom_image(exposure_ms=exposure_ms)

            if show:
                print(f"  Opening image viewer (shape: {image.shape})...")
            elif save_path:
                print(f"  Saving image (shape: {image.shape})...")

            result = await asyncio.to_thread(
                self._sam_conn.root.view_image,
                image,
                title=title,
                save_path=save_path,
                show=show
            )

            return dict(result)

        except Exception as e:
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
        if not self.has_sam:
            return {'error': 'SAM Server not connected'}

        if self._last_detection is None:
            return {'error': 'No detection results available. Run detect_embryos first.'}

        try:
            image = self._last_detection['image']
            embryos = self._last_detection['embryos']

            if not embryos:
                return {'error': 'No embryos in last detection'}

            print(f"  Showing {len(embryos)} detected embryos with bounding boxes...")

            result = await asyncio.to_thread(
                self._sam_conn.root.view_embryos,
                image,
                embryos,
                title=f"Detected {len(embryos)} Embryos",
                save_path=save_path,
                show=show
            )

            return dict(result)

        except Exception as e:
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
        View specified embryos with bounding boxes on an image.

        Parameters
        ----------
        image : np.ndarray
            Image to display
        embryos : list of dict
            Embryo data with bounding boxes
        title : str
            Window title
        save_path : str, optional
            Path to save the visualization image
        show : bool
            Whether to display in matplotlib window

        Returns
        -------
        dict
            Success status
        """
        if not self.has_sam:
            return {'error': 'SAM Server not connected'}

        if image is None:
            return {'error': 'No image provided'}

        if not embryos:
            return {'error': 'No embryos to display'}

        try:
            print(f"  Showing {len(embryos)} embryos with bounding boxes...")

            result = await asyncio.to_thread(
                self._sam_conn.root.view_embryos,
                image,
                embryos,
                title=title,
                save_path=save_path,
                show=show
            )

            return dict(result)

        except Exception as e:
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

        # Microscope Server status
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
            except Exception as e:
                status['queue_server'] = {'error': str(e)}
        else:
            status['queue_server'] = {'connected': False}

        # SAM Server status
        if self._sam_conn:
            try:
                sam_status = await asyncio.to_thread(
                    self._sam_conn.root.get_status
                )
                status['sam_server'] = dict(sam_status)
            except Exception as e:
                status['sam_server'] = {'error': str(e)}
        else:
            status['sam_server'] = {'connected': False}

        return status


async def create_queue_server_client(
    http_url: str = "http://127.0.0.1:60610",
    sam_port: int = 18862,
) -> Optional[QueueServerClient]:
    """
    Create and connect a microscope server client.

    Parameters
    ----------
    http_url : str
        Microscope Server HTTP API URL
    sam_port : int
        SAM server port

    Returns
    -------
    QueueServerClient or None
        Connected client, or None if connection failed
    """
    client = QueueServerClient(http_url=http_url, sam_port=sam_port)
    if await client.connect():
        return client
    return None
