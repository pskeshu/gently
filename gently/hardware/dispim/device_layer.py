"""
Gently Device Layer - Unified Hardware Server

Consolidates hardware control into a single process:
- Direct MMCore initialization (no RPyC hop to external Micro-Manager)
- Ophyd device abstraction
- Bluesky RunEngine for plan execution
- SAM embryo detection via HTTP endpoints

This replaces the previous 3-process architecture:
- Process 1: Micro-Manager RPyC (port 18861) - ELIMINATED
- Process 2: simple_server.py (port 60610)   - REPLACED by this
- Process 3: sam_server.py (port 18862)      - REPLACED by this

Usage:
    python start_device_layer.py
    python start_device_layer.py --port 60610 --sam-device cuda
"""

import asyncio
import logging
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)
from dataclasses import dataclass, field

import numpy as np

# Add project root to path
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from aiohttp import web
import yaml

from gently.core.service import Service
from gently.exceptions import HardwareError, AcquisitionError
from gently.log_config import configure_logging
from gently.settings import settings

# Bluesky imports
from bluesky import RunEngine
# BestEffortCallback removed — unused


@dataclass
class PlanRequest:
    """A request to run a plan"""
    plan_name: str
    kwargs: Dict[str, Any]
    future: asyncio.Future = field(default_factory=lambda: asyncio.get_event_loop().create_future())


class DeviceLayerServer(Service):
    """
    Consolidated device layer: MMCore + Ophyd + RunEngine + SAM.

    Runs RunEngine in the main thread via an async task queue.
    Provides HTTP API for plan submission and SAM detection.
    """

    def __init__(
        self,
        config_path: str = "config/config.yml",
        sam_device: str = "cuda",
        host: str = settings.network.device_host,
        port: int = settings.network.device_port,
    ):
        super().__init__(name="device-layer", service_type="hardware", host=host, port=port)
        self.config_path = config_path
        self.config = None
        self.core = None
        self.RE = None
        self.devices = {}
        self.plans = {}

        # SAM configuration
        self._sam_device = sam_device
        self._sam_detector = None  # Lazy loaded
        self._sam_checkpoint = "sam_vit_b_01ec64.pth"
        self._sam_model_type = "vit_b"

        # Task queue for plan execution
        self._plan_queue: asyncio.Queue[PlanRequest] = asyncio.Queue()
        self._running = False

        # Results storage (simple in-memory)
        self._run_history = []
        self._last_documents = {}

        # Plan execution timing log
        self._plan_execution_log = []

        # Volume staging directory - set via POST /session/configure
        # When set, large numpy arrays are written as TIFF files instead of
        # being serialized to JSON lists (which can turn a 400MB uint16 volume
        # into ~2GB of JSON text).
        self._volume_dir: Optional[str] = None

        # Server lifecycle objects (populated in on_start)
        self._app = None
        self._runner = None
        self._executor_task = None

    async def initialize(self):
        """Initialize hardware and RunEngine"""
        logger.info("=" * 60)
        logger.info("GENTLY DEVICE LAYER")
        logger.info("=" * 60)

        # [1/5] Load config
        logger.info("[1/5] Loading configuration...")
        with open(self.config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        logger.info("Config loaded from %s", self.config_path)

        # [2/5] Direct MMCore initialization (replaces RPyC hop)
        logger.info("[2/5] Initializing Micro-Manager Core (direct)...")
        import pymmcore

        self.core = pymmcore.CMMCore()
        self.core.enableStderrLog(True)

        # Add MM directory to PATH for device adapters
        mm_directory = self.config.get('mmdirectory', 'C:/Program Files/Micro-Manager-1.4')
        os.environ["PATH"] += os.pathsep + mm_directory
        self.core.setDeviceAdapterSearchPaths([mm_directory])

        # Load system configuration
        mm_config = self.config.get('mmconfig', 'MMConfig.cfg')
        mm_config_path = os.path.join(mm_directory, mm_config)
        if not os.path.exists(mm_config_path):
            # Try config.yml directory
            mm_config_path = os.path.join(os.path.dirname(self.config_path), mm_config)

        logger.info("Loading: %s", mm_config_path)
        self.core.loadSystemConfiguration(mm_config_path)
        logger.info("MMCore initialized (direct, in-process)")
        logger.info("Loaded devices: %s", self.core.getLoadedDevices())

        # [3/5] Create Ophyd devices
        logger.info("[3/5] Creating Ophyd devices...")
        from .device_factory import create_devices_from_mmcore
        # Suppress rich console output to avoid Unicode issues on Windows
        import io
        old_stdout = sys.stdout
        sys.stdout = io.StringIO()
        try:
            self.devices = create_devices_from_mmcore(self.core)
        finally:
            sys.stdout = old_stdout
        logger.info("Created %d devices", len(self.devices))
        for name in self.devices:
            logger.debug("  - %s", name)

        # [4/5] Initialize RunEngine
        logger.info("[4/5] Initializing RunEngine...")
        self.RE = RunEngine({})

        # Note: Databroker SQLite backend can't store image arrays directly.
        # Data is returned in HTTP response instead.
        self._db = None

        # Simple document collector with numpy serialization
        def serialize_value(v):
            """Convert numpy arrays to JSON-safe format.

            When ``self._volume_dir`` is set and the array exceeds 1 MB,
            the array is written as a TIFF file in the staging directory
            and a lightweight *file reference* dict is returned instead of
            the full data.  This avoids turning a 400 MB uint16 volume
            into ~2 GB of JSON text.
            """
            if isinstance(v, np.ndarray):
                # Large array + staging dir configured -> file ref
                if self._volume_dir and v.nbytes > 1_000_000:
                    import uuid
                    try:
                        import tifffile
                    except ImportError:
                        # tifffile not installed on server - fall back to list
                        return v.tolist()
                    uid = uuid.uuid4().hex[:12]
                    tiff_path = Path(self._volume_dir) / f"{uid}.tif"
                    tiff_path.parent.mkdir(parents=True, exist_ok=True)
                    tifffile.imwrite(str(tiff_path), v)
                    return {
                        "__file_ref__": True,
                        "path": str(tiff_path),
                        "shape": list(v.shape),
                        "dtype": str(v.dtype),
                    }
                return v.tolist()
            elif isinstance(v, (np.integer, np.floating)):
                return v.item()
            elif isinstance(v, dict):
                return {k: serialize_value(val) for k, val in v.items()}
            elif isinstance(v, (list, tuple)):
                return [serialize_value(item) for item in v]
            return v

        def collect_docs(name, doc):
            # Serialize the document to handle numpy arrays
            serialized_doc = serialize_value(dict(doc))

            if name == 'start':
                self._last_documents = {'start': serialized_doc, 'descriptors': [], 'events': [], 'stop': None}
            elif name == 'descriptor':
                self._last_documents['descriptors'].append(serialized_doc)
            elif name == 'event':
                self._last_documents['events'].append(serialized_doc)
            elif name == 'stop':
                self._last_documents['stop'] = serialized_doc
                self._run_history.append(self._last_documents.copy())

        self.RE.subscribe(collect_docs)
        logger.info("RunEngine ready")

        # [5/5] Load plans
        logger.info("[5/5] Loading plans...")
        self._load_plans()

        logger.info("=" * 60)
        logger.info("Device layer initialized successfully")
        logger.info("=" * 60)

    def _load_plans(self):
        """Load available plans"""
        try:
            from .plans.acquisition import (
                move_stage_plan,
                read_stage_plan,
                capture_bottom_image_plan,
                capture_lightsheet_image_plan,
                move_piezo_plan,
                move_scanner_plan,
                set_led_plan,
                set_laser_plan,
            )
            self.plans['move_stage_plan'] = move_stage_plan
            self.plans['read_stage_plan'] = read_stage_plan
            self.plans['capture_bottom_image_plan'] = capture_bottom_image_plan
            self.plans['capture_lightsheet_image_plan'] = capture_lightsheet_image_plan
            self.plans['move_piezo_plan'] = move_piezo_plan
            self.plans['move_scanner_plan'] = move_scanner_plan
            self.plans['set_led_plan'] = set_led_plan
            self.plans['set_laser_plan'] = set_laser_plan
            logger.info("Loaded %d plans", len(self.plans))
        except ImportError as e:
            logger.warning("Could not load some plans: %s", e)

        # Also load main plans if available
        try:
            from .plans.acquisition import (
                calibrate_piezo_galvo_plan,
                acquire_single_volume_plan,
            )
            self.plans['calibrate_piezo_galvo_plan'] = calibrate_piezo_galvo_plan
            self.plans['acquire_single_volume_plan'] = acquire_single_volume_plan
            logger.info("Loaded main acquisition plans")
        except ImportError:
            logger.info("Main acquisition plans not available")

    def _resolve_device_args(self, kwargs: Dict[str, Any]) -> Dict[str, Any]:
        """Replace device name strings with actual device objects"""
        resolved = {}
        for key, value in kwargs.items():
            if isinstance(value, str) and value in self.devices:
                resolved[key] = self.devices[value]
            else:
                resolved[key] = value
        return resolved

    # =========================================================================
    # SAM Detection (Lazy Loading)
    # =========================================================================

    def _get_sam_detector(self):
        """Lazy-load SAM detector on first detection request.

        This defers the ~5-10 second model load until actually needed,
        keeping server startup fast.
        """
        if self._sam_detector is None:
            logger.info("Loading SAM detector (%s on %s)...", self._sam_model_type, self._sam_device)
            from .sam_detection import SAMEmbryoDetector

            self._sam_detector = SAMEmbryoDetector(
                sam_checkpoint=self._sam_checkpoint,
                sam_model_type=self._sam_model_type,
                device=self._sam_device
            )
            logger.info("SAM detector ready")

        return self._sam_detector

    # =========================================================================
    # Plan Execution
    # =========================================================================

    async def _plan_executor(self):
        """Background task that executes plans from the queue"""
        from datetime import datetime

        logger.info("Plan executor started - waiting for plans...")
        self._running = True

        while self._running:
            try:
                # Wait for a plan request
                request = await asyncio.wait_for(
                    self._plan_queue.get(),
                    timeout=1.0
                )
            except asyncio.TimeoutError:
                continue

            # Log execution start with timestamp
            start_time = datetime.now()
            execution_record = {
                'plan_name': request.plan_name,
                'kwargs': {k: str(v) for k, v in request.kwargs.items()},  # Stringify for JSON
                'start_time': start_time.isoformat(),
                'start_time_formatted': start_time.strftime('%H:%M:%S.%f')[:-3],
            }

            logger.info(">>> [%s] Executing: %s", start_time.strftime('%H:%M:%S'), request.plan_name)

            try:
                # Get the plan function
                if request.plan_name not in self.plans:
                    raise ValueError(f"Unknown plan: {request.plan_name}")

                plan_func = self.plans[request.plan_name]

                # Resolve device arguments
                resolved_kwargs = self._resolve_device_args(request.kwargs)

                # Create the plan generator
                plan = plan_func(**resolved_kwargs)

                # Run the plan (this happens in the main thread!)
                result = self.RE(plan)

                # Get the run UID
                uid = result[0] if result else None

                # Log completion
                end_time = datetime.now()
                duration_ms = (end_time - start_time).total_seconds() * 1000

                execution_record.update({
                    'end_time': end_time.isoformat(),
                    'end_time_formatted': end_time.strftime('%H:%M:%S.%f')[:-3],
                    'duration_ms': duration_ms,
                    'success': True,
                    'uid': uid,
                })

                logger.info("<<< [%s] Complete: %s (%.0fms)", end_time.strftime('%H:%M:%S'), request.plan_name, duration_ms)

                # Complete the future with result
                request.future.set_result({
                    'success': True,
                    'uid': uid,
                    'documents': self._last_documents.copy()
                })

            except Exception as e:
                import traceback
                end_time = datetime.now()
                duration_ms = (end_time - start_time).total_seconds() * 1000

                execution_record.update({
                    'end_time': end_time.isoformat(),
                    'end_time_formatted': end_time.strftime('%H:%M:%S.%f')[:-3],
                    'duration_ms': duration_ms,
                    'success': False,
                    'error': str(e),
                })

                logger.error("<<< [%s] Failed: %s - %s", end_time.strftime('%H:%M:%S'), request.plan_name, e)
                request.future.set_exception(e)

            # Store execution record
            self._plan_execution_log.append(execution_record)
            # Keep last 1000 entries
            if len(self._plan_execution_log) > 1000:
                self._plan_execution_log = self._plan_execution_log[-1000:]

    async def submit_plan(self, plan_name: str, kwargs: Dict = None) -> Dict:
        """Submit a plan and wait for completion"""
        kwargs = kwargs or {}

        # Create request with a future
        loop = asyncio.get_event_loop()
        request = PlanRequest(
            plan_name=plan_name,
            kwargs=kwargs,
            future=loop.create_future()
        )

        # Add to queue
        await self._plan_queue.put(request)

        # Wait for completion
        result = await request.future
        return result

    # =========================================================================
    # HTTP API Handlers - Core Operations
    # =========================================================================

    async def handle_status(self, request):
        """GET /api/status"""
        status = {
            'manager_state': 'idle' if self._running else 'stopped',
            're_state': 'idle',
            'devices': list(self.devices.keys()),
            'plans': list(self.plans.keys()),
            'queue_size': self._plan_queue.qsize(),
            'sam_loaded': self._sam_detector is not None,
        }
        return web.json_response(status)

    async def handle_submit_plan(self, request):
        """POST /api/queue/item/add"""
        try:
            data = await request.json()
            plan_name = data.get('item', {}).get('name')
            kwargs = data.get('item', {}).get('kwargs', {})

            if not plan_name:
                return web.json_response(
                    {'success': False, 'error': 'No plan name provided'},
                    status=400
                )

            result = await self.submit_plan(plan_name, kwargs)
            return web.json_response(result)

        except Exception as e:
            import traceback
            return web.json_response({
                'success': False,
                'error': str(e),
                'traceback': traceback.format_exc()
            }, status=500)

    async def handle_get_history(self, request):
        """GET /api/history"""
        return web.json_response({
            'success': True,
            'items': self._run_history[-10:]  # Last 10 runs
        })

    async def handle_get_devices(self, request):
        """GET /api/devices"""
        return web.json_response({
            'success': True,
            'devices': list(self.devices.keys())
        })

    async def handle_get_plans(self, request):
        """GET /api/plans"""
        return web.json_response({
            'success': True,
            'plans': list(self.plans.keys())
        })

    async def handle_get_led_status(self, request):
        """GET /api/led/status - Get current LED state and available configs"""
        try:
            led = self.devices.get('led')
            if led is None:
                return web.json_response({
                    'success': False,
                    'error': 'LED device not found'
                })

            # Read current state
            current_state = led.read()
            led_value = current_state.get(led.name, {}).get('value', 'unknown')

            # Get available configs
            available_configs = led._available_configs

            return web.json_response({
                'success': True,
                'current_state': led_value,
                'available_configs': available_configs,
                'group_name': led.group_name
            })
        except Exception as e:
            import traceback
            return web.json_response({
                'success': False,
                'error': str(e),
                'traceback': traceback.format_exc()
            }, status=500)

    async def handle_set_led(self, request):
        """POST /api/led/set - Set LED state directly (bypass plan queue)"""
        try:
            data = await request.json()
            state = data.get('state', 'Closed')

            led = self.devices.get('led')
            if led is None:
                return web.json_response({
                    'success': False,
                    'error': 'LED device not found'
                })

            # Set LED state directly
            status = led.set(state)
            # Wait for completion
            import time
            timeout = 5.0
            start = time.time()
            while not status.done and (time.time() - start) < timeout:
                await asyncio.sleep(0.1)

            if status.done and status.success:
                return web.json_response({
                    'success': True,
                    'state': state,
                    'message': f'LED set to {state}'
                })
            else:
                return web.json_response({
                    'success': False,
                    'error': f'Failed to set LED to {state}'
                })
        except Exception as e:
            import traceback
            return web.json_response({
                'success': False,
                'error': str(e),
                'traceback': traceback.format_exc()
            }, status=500)

    async def handle_set_camera_led_mode(self, request):
        """POST /api/camera/led_mode - Enable/disable automatic LED for bottom camera"""
        try:
            data = await request.json()
            use_led = data.get('use_led', False)

            bottom_camera = self.devices.get('bottom_camera')
            if bottom_camera is None:
                return web.json_response({
                    'success': False,
                    'error': 'Bottom camera device not found'
                })

            # Set the use_led attribute
            bottom_camera.use_led = use_led

            return web.json_response({
                'success': True,
                'use_led': use_led,
                'message': f'Bottom camera LED mode: {"ON" if use_led else "OFF"}'
            })
        except Exception as e:
            import traceback
            return web.json_response({
                'success': False,
                'error': str(e),
                'traceback': traceback.format_exc()
            }, status=500)

    async def handle_set_camera_exposure(self, request):
        """POST /api/camera/exposure - Set bottom camera exposure time"""
        try:
            data = await request.json()
            exposure_ms = data.get('exposure_ms', 50.0)

            bottom_camera = self.devices.get('bottom_camera')
            if bottom_camera is None:
                return web.json_response({
                    'success': False,
                    'error': 'Bottom camera device not found'
                })

            # Set exposure using the device's configure_exposure method
            bottom_camera.configure_exposure(exposure_ms)

            return web.json_response({
                'success': True,
                'exposure_ms': exposure_ms,
                'message': f'Bottom camera exposure set to {exposure_ms} ms'
            })
        except Exception as e:
            import traceback
            return web.json_response({
                'success': False,
                'error': str(e),
                'traceback': traceback.format_exc()
            }, status=500)

    async def handle_get_camera_exposure(self, request):
        """GET /api/camera/exposure - Get bottom camera exposure time"""
        try:
            bottom_camera = self.devices.get('bottom_camera')
            if bottom_camera is None:
                return web.json_response({
                    'success': False,
                    'error': 'Bottom camera device not found'
                })

            exposure_ms = bottom_camera.exposure_time

            return web.json_response({
                'success': True,
                'exposure_ms': exposure_ms
            })
        except Exception as e:
            import traceback
            return web.json_response({
                'success': False,
                'error': str(e),
                'traceback': traceback.format_exc()
            }, status=500)

    async def handle_get_plan_log(self, request):
        """GET /api/plan_log - Get recent plan execution log with timing"""
        try:
            limit = int(request.query.get('limit', 100))
            return web.json_response({
                'success': True,
                'entries': self._plan_execution_log[-limit:],
                'total_count': len(self._plan_execution_log),
            })
        except Exception as e:
            import traceback
            return web.json_response({
                'success': False,
                'error': str(e),
                'traceback': traceback.format_exc()
            }, status=500)

    async def handle_session_configure(self, request):
        """POST /session/configure - set staging directory for file-ref protocol.

        Body: {"volume_dir": "D:/Gently2/incoming"}
        """
        try:
            data = await request.json()
            volume_dir = data.get("volume_dir")
            if volume_dir:
                # Ensure the directory exists
                Path(volume_dir).mkdir(parents=True, exist_ok=True)
                self._volume_dir = volume_dir
                logger.info("Session configured: volume_dir = %s", volume_dir)
                return web.json_response({
                    "success": True,
                    "volume_dir": volume_dir,
                })
            else:
                # Clear staging
                self._volume_dir = None
                return web.json_response({
                    "success": True,
                    "volume_dir": None,
                    "message": "Volume staging disabled",
                })
        except Exception as e:
            import traceback
            return web.json_response({
                "success": False,
                "error": str(e),
                "traceback": traceback.format_exc(),
            }, status=500)

    # =========================================================================
    # HTTP API Handlers - SAM Detection
    # =========================================================================

    async def handle_sam_status(self, request):
        """GET /api/sam/status - Check SAM model availability"""
        return web.json_response({
            'success': True,
            'available': True,  # SAM is always available (lazy loaded)
            'loaded': self._sam_detector is not None,
            'device': self._sam_device,
            'model_type': self._sam_model_type,
        })

    async def handle_detect_embryos(self, request):
        """POST /api/detect_embryos - Capture image and detect embryos.

        This combines image capture and SAM detection in one HTTP round-trip,
        avoiding the need to serialize images across process boundaries.

        Request body:
        {
            "pixel_size_um": 6.5,
            "objective_mag": 10.0,
            "use_claude_review": true,
            "min_confidence": 0.7,
            "exposure_ms": 50.0,  // optional
            "brightness_percentile": 99.0,
            "min_area": 5000,
            "max_area": 150000
        }

        Returns:
        {
            "success": true,
            "embryos": [...],
            "initial_detections": 5,
            "final_detections": 4,
            "stage_position": [x, y],
            "image_path": "path/to/captured/image.tif"  // if volume_dir configured
        }
        """
        try:
            data = await request.json()

            # Extract parameters with defaults
            from gently.core.coordinates import DEFAULT_PIXEL_SIZE_UM, DEFAULT_OBJECTIVE_MAG

            pixel_size_um = data.get('pixel_size_um', DEFAULT_PIXEL_SIZE_UM)
            objective_mag = data.get('objective_mag', DEFAULT_OBJECTIVE_MAG)
            use_claude_review = data.get('use_claude_review', True)
            min_confidence = data.get('min_confidence', 0.7)
            exposure_ms = data.get('exposure_ms')
            brightness_percentile = data.get('brightness_percentile', 99.0)
            min_area = data.get('min_area', 5000)
            max_area = data.get('max_area', 150000)

            # Set exposure if specified
            if exposure_ms is not None:
                bottom_camera = self.devices.get('bottom_camera')
                if bottom_camera:
                    bottom_camera.configure_exposure(exposure_ms)

            # Capture image via plan
            logger.info("[detect_embryos] Capturing bottom camera image...")
            capture_result = await self.submit_plan(
                'capture_bottom_image_plan',
                kwargs={'bottom_camera': 'bottom_camera'}
            )

            if not capture_result.get('success'):
                return web.json_response({
                    'success': False,
                    'error': f"Image capture failed: {capture_result.get('error', 'Unknown')}"
                }, status=500)

            # Extract image from result
            docs = capture_result.get('documents', {})
            events = docs.get('events', [])
            image = None
            if events:
                event_data = events[0].get('data', {})
                for key in ['bottom_camera', 'bottom_camera_image', 'Bottom PCO']:
                    if key in event_data:
                        val = event_data[key]
                        # Handle file ref
                        if isinstance(val, dict) and val.get('__file_ref__'):
                            import tifffile
                            image = tifffile.imread(val['path'])
                        else:
                            image = np.array(val)
                        break

            if image is None:
                return web.json_response({
                    'success': False,
                    'error': 'No image data in capture result'
                }, status=500)

            logger.info("[detect_embryos] Image shape: %s", image.shape)

            # Read stage position
            logger.info("[detect_embryos] Reading stage position...")
            stage_result = await self.submit_plan(
                'read_stage_plan',
                kwargs={'xy_stage': 'xy_stage'}
            )

            stage_x, stage_y = 0.0, 0.0
            if stage_result.get('success'):
                stage_docs = stage_result.get('documents', {})
                stage_events = stage_docs.get('events', [])
                if stage_events:
                    stage_data = stage_events[0].get('data', {})
                    # DiSPIMXYStage.read() returns {device_name: [x, y]}
                    for key in ['xy_stage', 'XYStage:XY:31', 'xy_stage_position']:
                        if key in stage_data:
                            val = stage_data[key]
                            if isinstance(val, (list, tuple)) and len(val) >= 2:
                                stage_x, stage_y = float(val[0]), float(val[1])
                                break

            stage_position = (stage_x, stage_y)
            logger.info("[detect_embryos] Stage position: %s", stage_position)

            # Run SAM detection in thread to avoid blocking event loop
            logger.info("[detect_embryos] Running SAM detection...")
            detector = self._get_sam_detector()

            sam_result = await asyncio.to_thread(
                self._run_sam_detection,
                detector,
                image,
                stage_position,
                pixel_size_um,
                objective_mag,
                use_claude_review,
                brightness_percentile,
                min_area,
                max_area
            )

            # Save image if volume_dir configured
            image_path = None
            if self._volume_dir:
                import uuid
                try:
                    import tifffile
                    uid = uuid.uuid4().hex[:12]
                    image_path = str(Path(self._volume_dir) / f"detection_{uid}.tif")
                    tifffile.imwrite(image_path, image)
                except ImportError:
                    pass

            # Build response
            response = {
                'success': sam_result.get('success', False),
                'embryos': sam_result.get('embryos', []),
                'initial_detections': sam_result.get('initial_detections', 0),
                'final_detections': sam_result.get('final_detections', 0),
                'stage_position': list(stage_position),
                'verification': sam_result.get('verification', {}),
            }

            if image_path:
                response['image_path'] = image_path

            if 'error' in sam_result:
                response['error'] = sam_result['error']

            return web.json_response(response)

        except Exception as e:
            import traceback
            return web.json_response({
                'success': False,
                'error': str(e),
                'traceback': traceback.format_exc()
            }, status=500)

    def _run_sam_detection(
        self,
        detector,
        image: np.ndarray,
        stage_position: tuple,
        pixel_size_um: float,
        objective_mag: float,
        use_claude_review: bool,
        brightness_percentile: float,
        min_area: int,
        max_area: int
    ) -> dict:
        """Run SAM detection synchronously (called from thread).

        The SAMEmbryoDetector.detect_embryos is async, so we need to
        run it in a new event loop.
        """
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        try:
            result = loop.run_until_complete(
                detector.detect_embryos(
                    image=image,
                    stage_position=stage_position,
                    pixel_size_um=pixel_size_um,
                    objective_mag=objective_mag,
                    use_claude_review=use_claude_review,
                    save_visualizations=True,
                    output_dir=Path("./detection_results"),
                    brightness_percentile=brightness_percentile,
                    min_area=min_area,
                    max_area=max_area
                )
            )

            # Ensure results are serializable (convert numpy types)
            embryos = result.get('embryos', [])
            for embryo in embryos:
                for key, value in list(embryo.items()):
                    if isinstance(value, np.floating):
                        embryo[key] = float(value)
                    elif isinstance(value, np.integer):
                        embryo[key] = int(value)
                    elif isinstance(value, np.ndarray):
                        # Remove mask from response (not JSON serializable)
                        del embryo[key]

            result['success'] = True
            return result

        except Exception as e:
            import traceback
            return {
                'success': False,
                'error': str(e),
                'traceback': traceback.format_exc(),
                'embryos': [],
                'initial_detections': 0,
                'final_detections': 0,
            }
        finally:
            loop.close()

    # =========================================================================
    # Microscope API — generic plan-based interface
    # =========================================================================

    # Plan schemas — each plan is described in Anthropic tool-call format
    # so the agent can use them directly as tool definitions.
    # "bluesky_plan" and "extractor" are internal (not sent to client).
    PLAN_SCHEMAS = {
        "move": {
            "description": "Move XY stage to absolute position in micrometers.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "x": {"type": "number", "description": "X position in µm"},
                    "y": {"type": "number", "description": "Y position in µm"},
                },
                "required": ["x", "y"],
            },
            "bluesky_plan": "move_stage_plan",
            "extractor": "_extract_move",
        },
        "get_position": {
            "description": "Read current XY stage position.",
            "input_schema": {"type": "object", "properties": {}},
            "bluesky_plan": "read_stage_plan",
            "extractor": "_extract_position",
        },
        "acquire": {
            "description": "Acquire a 3D volume via synchronized galvo-piezo scan.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "num_slices": {"type": "integer", "description": "Number of Z slices", "default": 50},
                    "exposure_ms": {"type": "number", "description": "Camera exposure per slice in ms", "default": 10.0},
                    "galvo_amplitude": {"type": "number", "description": "Galvo scan range in volts", "default": 0.5},
                    "galvo_center": {"type": "number", "description": "Galvo center position in volts", "default": 0.0},
                    "piezo_amplitude": {"type": "number", "description": "Piezo Z range in µm", "default": 25.0},
                    "piezo_center": {"type": "number", "description": "Piezo center position in µm", "default": 50.0},
                },
            },
            "bluesky_plan": "acquire_single_volume_plan",
            "extractor": "_extract_volume",
        },
        "snap": {
            "description": "Capture a single lightsheet image at specified Z and galvo position.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "piezo_position": {"type": "number", "description": "Z position in µm"},
                    "galvo_position": {"type": "number", "description": "Galvo angle in volts"},
                    "exposure_ms": {"type": "number", "description": "Camera exposure in ms", "default": 10.0},
                },
            },
            "bluesky_plan": "capture_lightsheet_image_plan",
            "extractor": "_extract_image",
        },
        "detect_image": {
            "description": "Capture image from the bottom detection camera.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "use_led": {"type": "boolean", "description": "Turn on LED during capture", "default": False},
                    "exposure_ms": {"type": "number", "description": "Camera exposure in ms"},
                },
            },
            "bluesky_plan": "capture_bottom_image_plan",
            "extractor": "_extract_image",
        },
        "calibrate": {
            "description": "Run piezo-galvo calibration to find optimal focus parameters for an embryo.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "piezo_positions": {"type": "array", "items": {"type": "number"}, "description": "Piezo positions to sweep (µm)"},
                    "galvo_positions": {"type": "array", "items": {"type": "number"}, "description": "Galvo positions to sweep (volts)"},
                },
            },
            "bluesky_plan": "calibrate_piezo_galvo_plan",
            "extractor": "_extract_calibration",
        },
        "set_illumination": {
            "description": "Set LED illumination state for the detection camera.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "state": {"type": "string", "enum": ["Open", "Closed"], "description": "LED state"},
                },
                "required": ["state"],
            },
            "bluesky_plan": "set_led_plan",
            "extractor": "_extract_success",
        },
        "get_illumination": {
            "description": "Get current LED illumination status.",
            "input_schema": {"type": "object", "properties": {}},
            "bluesky_plan": None,
            "extractor": None,
        },
        "detect": {
            "description": "Detect embryos/samples in the current field of view using SAM segmentation.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "pixel_size_um": {"type": "number", "description": "Camera pixel size in µm", "default": 6.5},
                    "objective_mag": {"type": "number", "description": "Objective magnification", "default": 10.0},
                    "min_confidence": {"type": "number", "description": "Minimum detection confidence", "default": 0.7},
                    "exposure_ms": {"type": "number", "description": "Camera exposure in ms"},
                    "brightness_percentile": {"type": "number", "description": "Brightness threshold percentile", "default": 99.0},
                    "min_area": {"type": "integer", "description": "Minimum embryo area in pixels", "default": 5000},
                    "max_area": {"type": "integer", "description": "Maximum embryo area in pixels", "default": 150000},
                },
            },
            "bluesky_plan": None,
            "extractor": None,
        },
        "status": {
            "description": "Get device layer status including loaded devices and queue state.",
            "input_schema": {"type": "object", "properties": {}},
            "bluesky_plan": None,
            "extractor": None,
        },
    }

    def _extract_from_events(self, documents: dict, candidate_keys: list) -> Any:
        """Pull data from Bluesky event documents by candidate key names."""
        events = documents.get('events', [])
        for event in events:
            data = event.get('data', {})
            for key in candidate_keys:
                if key in data:
                    return data[key]
        return None

    def _extract_move(self, documents: dict, params: dict) -> dict:
        return {"success": True, "x": params.get("x"), "y": params.get("y")}

    def _extract_position(self, documents: dict, params: dict) -> dict:
        events = documents.get('events', [])
        if events:
            data = events[0].get('data', {})
            for key in ['XY:31', 'xy_stage', 'stage']:
                if key in data:
                    val = data[key]
                    if isinstance(val, (list, tuple)) and len(val) >= 2:
                        return {"success": True, "x": float(val[0]), "y": float(val[1])}
                    if isinstance(val, dict):
                        return {"success": True, "x": float(val.get('x', 0)), "y": float(val.get('y', 0))}
        return {"success": False, "error": "Could not read position"}

    def _extract_volume(self, documents: dict, params: dict) -> dict:
        val = self._extract_from_events(documents, ['volume_scanner', 'camera', 'camera_image'])
        if val is not None:
            result = {"success": True}
            if isinstance(val, dict) and val.get('__file_ref__'):
                result['volume'] = val  # file ref — client resolves
                result['shape'] = val.get('shape')
            else:
                result['volume'] = val
                if hasattr(val, 'shape'):
                    result['shape'] = list(val.shape)
            return result
        return {"success": False, "error": "No volume data in result"}

    def _extract_image(self, documents: dict, params: dict) -> dict:
        val = self._extract_from_events(
            documents, ['HamCam1', 'lightsheet_snap', 'camera', 'bottom_camera', 'bottom_camera_image', 'Bottom PCO']
        )
        if val is not None:
            result = {"success": True}
            if isinstance(val, dict) and val.get('__file_ref__'):
                result['image'] = val
                result['shape'] = val.get('shape')
            else:
                result['image'] = val
                if hasattr(val, 'shape'):
                    result['shape'] = list(val.shape)
            return result
        return {"success": False, "error": "No image data in result"}

    def _extract_calibration(self, documents: dict, params: dict) -> dict:
        # Calibration results come back in the plan result
        return {"success": True, "calibration": {}}

    def _extract_success(self, documents: dict, params: dict) -> dict:
        return {"success": True}

    async def handle_microscope_info(self, request):
        """GET /api/microscope — handshake: plans as Anthropic tool schemas."""
        from .description import HARDWARE_DESCRIPTION
        from . import HARDWARE_NAME, HARDWARE_DISPLAY_NAME

        # Build plan list, filtering to actually-available plans
        available_plans = []
        for plan_name, schema in self.PLAN_SCHEMAS.items():
            bluesky_name = schema.get("bluesky_plan")
            if bluesky_name is None or bluesky_name in self.plans:
                # Return client-facing fields (Anthropic tool format)
                available_plans.append({
                    "name": plan_name,
                    "description": schema["description"],
                    "input_schema": schema["input_schema"],
                })

        return web.json_response({
            "name": HARDWARE_NAME,
            "display_name": HARDWARE_DISPLAY_NAME,
            "description": HARDWARE_DESCRIPTION,
            "plans": available_plans,
        })

    async def handle_microscope_execute(self, request):
        """POST /api/microscope/execute — execute a named plan.

        Request: {"plan": "acquire", "params": {"num_slices": 50}}
        Response: {"success": true, "volume": <file_ref>, "shape": [50, 512, 1024]}
        """
        try:
            data = await request.json()
            plan_name = data.get("plan")
            params = data.get("params", {})

            if not plan_name:
                return web.json_response(
                    {"success": False, "error": "No plan name provided"}, status=400
                )

            schema = self.PLAN_SCHEMAS.get(plan_name)
            if schema is None:
                return web.json_response(
                    {"success": False, "error": f"Unknown plan: {plan_name}"},
                    status=400,
                )

            bluesky_name = schema.get("bluesky_plan")
            extractor_name = schema.get("extractor")

            # Directly handled plans (no Bluesky plan)
            if bluesky_name is None:
                if plan_name == "detect":
                    return await self.handle_detect_embryos(request)
                elif plan_name == "get_illumination":
                    return await self.handle_get_led_status(request)
                elif plan_name == "status":
                    return await self.handle_status(request)
                return web.json_response({"success": False, "error": f"Plan '{plan_name}' not implemented"}, status=500)

            if bluesky_name not in self.plans:
                return web.json_response(
                    {"success": False, "error": f"Hardware plan '{bluesky_name}' not loaded"},
                    status=500,
                )

            # Execute the Bluesky plan
            result = await self.submit_plan(bluesky_name, params)

            if not result.get("success"):
                return web.json_response(result)

            # Extract clean result from Bluesky documents
            extractor = getattr(self, extractor_name)
            clean_result = extractor(result.get("documents", {}), params)
            return web.json_response(clean_result)

        except Exception as e:
            import traceback
            return web.json_response(
                {"success": False, "error": str(e), "traceback": traceback.format_exc()},
                status=500,
            )

    # =========================================================================
    # Server Lifecycle
    # =========================================================================

    async def on_start(self):
        """Initialize hardware and start the HTTP server."""
        await self.initialize()

        # Create web app
        self._app = web.Application()

        # Core endpoints (carried forward from simple_server.py)
        self._app.router.add_get('/api/status', self.handle_status)
        self._app.router.add_post('/api/queue/item/add', self.handle_submit_plan)
        self._app.router.add_get('/api/history', self.handle_get_history)
        self._app.router.add_get('/api/devices', self.handle_get_devices)
        self._app.router.add_get('/api/plans', self.handle_get_plans)
        self._app.router.add_get('/api/led/status', self.handle_get_led_status)
        self._app.router.add_post('/api/led/set', self.handle_set_led)
        self._app.router.add_post('/api/camera/led_mode', self.handle_set_camera_led_mode)
        self._app.router.add_post('/api/camera/exposure', self.handle_set_camera_exposure)
        self._app.router.add_get('/api/camera/exposure', self.handle_get_camera_exposure)
        self._app.router.add_get('/api/plan_log', self.handle_get_plan_log)
        self._app.router.add_post('/session/configure', self.handle_session_configure)

        # SAM endpoints (new - replaces RPyC sam_server.py)
        self._app.router.add_get('/api/sam/status', self.handle_sam_status)
        self._app.router.add_post('/api/detect_embryos', self.handle_detect_embryos)

        # Microscope API (generic plan-based interface)
        self._app.router.add_get('/api/microscope', self.handle_microscope_info)
        self._app.router.add_post('/api/microscope/execute', self.handle_microscope_execute)

        # Start plan executor
        self._executor_task = asyncio.create_task(self._plan_executor())

        # Start web server
        self._runner = web.AppRunner(self._app)
        await self._runner.setup()
        site = web.TCPSite(self._runner, self.host, self.port)

        logger.info("=" * 60)
        logger.info("HTTP API available at http://%s:%d", self.host, self.port)
        logger.info("=" * 60)
        logger.info("Endpoints: GET /api/status, GET /api/devices, GET /api/plans, POST /api/queue/item/add, ...")

        await site.start()

    async def on_stop(self):
        """Shut down the HTTP server and plan executor."""
        logger.info("Shutting down...")
        self._running = False
        if self._executor_task:
            self._executor_task.cancel()
            try:
                await self._executor_task
            except asyncio.CancelledError:
                pass
        if self._runner:
            await self._runner.cleanup()
        logger.info("Device layer stopped.")

    async def health_check(self) -> Dict:
        """Return health status with device count, queue size, SAM status."""
        base = await super().health_check()
        base['device_count'] = len(self.devices)
        base['queue_size'] = self._plan_queue.qsize()
        base['sam_loaded'] = self._sam_detector is not None
        return base

    async def run(self, host: str = None, port: int = None):
        """Start the server and run until interrupted."""
        if host is not None:
            self.host = host
        if port is not None:
            self.port = port

        await self.start()

        # Keep running with proper shutdown handling
        self._shutdown_event = asyncio.Event()
        logger.info("Press Ctrl+C to stop")

        try:
            await self._shutdown_event.wait()
        except asyncio.CancelledError:
            pass
        finally:
            await self.stop()


async def main(port: int = settings.network.device_port, sam_device: str = "cuda"):
    server = DeviceLayerServer(sam_device=sam_device, port=port)
    await server.run()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Gently Device Layer Server")
    parser.add_argument("--port", type=int, default=settings.network.device_port, help="HTTP port")
    parser.add_argument("--sam-device", default="cuda", choices=["cuda", "cpu"],
                        help="Device for SAM model (default: cuda)")

    args = parser.parse_args()

    from pathlib import Path
    from datetime import datetime
    log_dir = Path(settings.storage.base_path) / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = str(log_dir / f"device_layer_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
    configure_logging(level="INFO", log_file=log_file)

    logger.info("Starting Gently Device Layer...")
    logger.info("Logging to %s", log_file)
    logger.info("This server provides unified hardware control + SAM detection.")

    try:
        asyncio.run(main(port=args.port, sam_device=args.sam_device))
    except KeyboardInterrupt:
        logger.info("Device layer stopped.")
