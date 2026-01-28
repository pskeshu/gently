"""
Simple Microscope Server for Gently DiSPIM

A lightweight asyncio-based server that runs RunEngine in the main thread,
solving the Windows threading issue with signal handlers.

This replaces the full bluesky-queueserver approach which has Windows
compatibility issues with multiprocessing.

Usage:
    python backend/simple_server.py

The server provides:
- HTTP API on port 60610 for plan submission
- SAM detection on port 18862 (via rpyc)
"""

import asyncio
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, Optional
from dataclasses import dataclass, field
from queue import Queue
import threading

import numpy as np

# Add project root to path
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from aiohttp import web
import yaml

# Bluesky imports
from bluesky import RunEngine
from bluesky.callbacks.best_effort import BestEffortCallback
from event_model import RunRouter


@dataclass
class PlanRequest:
    """A request to run a plan"""
    plan_name: str
    kwargs: Dict[str, Any]
    future: asyncio.Future = field(default_factory=lambda: asyncio.get_event_loop().create_future())


class SimpleMicroscopeServer:
    """
    Simple HTTP server for microscope control.

    Runs RunEngine in the main thread via an async task queue.
    """

    def __init__(self, config_path: str = "config.yml"):
        self.config_path = config_path
        self.config = None
        self.core = None
        self.RE = None
        self.devices = {}
        self.plans = {}

        # Task queue for plan execution
        self._plan_queue: asyncio.Queue[PlanRequest] = asyncio.Queue()
        self._running = False

        # Results storage (simple in-memory)
        self._run_history = []
        self._last_documents = {}

        # Plan execution timing log
        self._plan_execution_log = []

        # Volume staging directory — set via POST /session/configure
        # When set, large numpy arrays are written as TIFF files instead of
        # being serialized to JSON lists (which can turn a 400MB uint16 volume
        # into ~2GB of JSON text).
        self._volume_dir: Optional[str] = None

    async def initialize(self):
        """Initialize hardware and RunEngine"""
        print("=" * 60)
        print("SIMPLE MICROSCOPE SERVER")
        print("=" * 60)

        # Load config
        print("\n[1/4] Loading configuration...")
        with open(self.config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        print(f"  Config loaded from {self.config_path}")

        # Connect to existing Micro-Manager instance via rpyc
        print("\n[2/4] Connecting to Micro-Manager...")
        from client import get_mmc

        self.core = get_mmc()
        print(f"  Connected to MMCore via rpyc")

        # Create devices
        print("\n[3/4] Creating Ophyd devices...")
        from gently.agent.device_factory import create_devices_from_mmcore
        # Suppress rich console output to avoid Unicode issues on Windows
        import io
        import sys
        old_stdout = sys.stdout
        sys.stdout = io.StringIO()
        try:
            self.devices = create_devices_from_mmcore(self.core)
        finally:
            sys.stdout = old_stdout
        print(f"  Created {len(self.devices)} devices")
        for name in self.devices:
            print(f"    - {name}")

        # Initialize RunEngine
        print("\n[4/4] Initializing RunEngine...")
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
                # Large array + staging dir configured → file ref
                if self._volume_dir and v.nbytes > 1_000_000:
                    import uuid
                    try:
                        import tifffile
                    except ImportError:
                        # tifffile not installed on server — fall back to list
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
        print("  RunEngine ready")

        # Import plans
        print("\n[5/5] Loading plans...")
        self._load_plans()

        print("\n" + "=" * 60)
        print("Server initialized successfully")
        print("=" * 60)

    def _load_plans(self):
        """Load available plans"""
        try:
            from gently.plans_qserver import (
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
            print(f"  Loaded {len(self.plans)} plans")
        except ImportError as e:
            print(f"  Warning: Could not load some plans: {e}")

        # Also load main plans if available
        try:
            from gently.plans import (
                calibrate_piezo_galvo_plan,
                acquire_single_volume_plan,
            )
            self.plans['calibrate_piezo_galvo_plan'] = calibrate_piezo_galvo_plan
            self.plans['acquire_single_volume_plan'] = acquire_single_volume_plan
            print("  Loaded main acquisition plans")
        except ImportError:
            print("  Main acquisition plans not available")

    def _resolve_device_args(self, kwargs: Dict[str, Any]) -> Dict[str, Any]:
        """Replace device name strings with actual device objects"""
        resolved = {}
        for key, value in kwargs.items():
            if isinstance(value, str) and value in self.devices:
                resolved[key] = self.devices[value]
            else:
                resolved[key] = value
        return resolved

    async def _plan_executor(self):
        """Background task that executes plans from the queue"""
        from datetime import datetime

        print("\nPlan executor started - waiting for plans...")
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

            print(f"\n>>> [{start_time.strftime('%H:%M:%S')}] Executing: {request.plan_name}")

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

                print(f"<<< [{end_time.strftime('%H:%M:%S')}] Complete: {request.plan_name} ({duration_ms:.0f}ms)")

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

                print(f"<<< [{end_time.strftime('%H:%M:%S')}] Failed: {request.plan_name} - {e}")
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
    # HTTP API Handlers
    # =========================================================================

    async def handle_status(self, request):
        """GET /api/status"""
        status = {
            'manager_state': 'idle' if self._running else 'stopped',
            're_state': 'idle',
            'devices': list(self.devices.keys()),
            'plans': list(self.plans.keys()),
            'queue_size': self._plan_queue.qsize(),
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
        """POST /session/configure — set staging directory for file-ref protocol.

        Body: {"volume_dir": "D:/Gently2/incoming"}
        """
        try:
            data = await request.json()
            volume_dir = data.get("volume_dir")
            if volume_dir:
                # Ensure the directory exists
                Path(volume_dir).mkdir(parents=True, exist_ok=True)
                self._volume_dir = volume_dir
                print(f"  Session configured: volume_dir = {volume_dir}")
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

    async def run(self, host: str = '127.0.0.1', port: int = 60610):
        """Start the server"""
        await self.initialize()

        # Create web app
        app = web.Application()
        app.router.add_get('/api/status', self.handle_status)
        app.router.add_post('/api/queue/item/add', self.handle_submit_plan)
        app.router.add_get('/api/history', self.handle_get_history)
        app.router.add_get('/api/devices', self.handle_get_devices)
        app.router.add_get('/api/plans', self.handle_get_plans)
        app.router.add_get('/api/led/status', self.handle_get_led_status)
        app.router.add_post('/api/led/set', self.handle_set_led)
        app.router.add_post('/api/camera/led_mode', self.handle_set_camera_led_mode)
        app.router.add_post('/api/camera/exposure', self.handle_set_camera_exposure)
        app.router.add_get('/api/camera/exposure', self.handle_get_camera_exposure)
        app.router.add_get('/api/plan_log', self.handle_get_plan_log)
        app.router.add_post('/session/configure', self.handle_session_configure)

        # Start plan executor
        executor_task = asyncio.create_task(self._plan_executor())

        # Start web server
        runner = web.AppRunner(app)
        await runner.setup()
        site = web.TCPSite(runner, host, port)

        print(f"\n{'=' * 60}")
        print(f"HTTP API available at http://{host}:{port}")
        print(f"{'=' * 60}")
        print("\nEndpoints:")
        print(f"  GET  /api/status     - Server status")
        print(f"  GET  /api/devices    - List devices")
        print(f"  GET  /api/plans      - List plans")
        print(f"  POST /api/queue/item/add - Submit plan")
        print(f"  GET  /api/history    - Run history")
        print(f"  GET  /api/led/status - LED status and configs")
        print(f"  POST /api/led/set    - Set LED state directly")
        print(f"  GET  /api/plan_log   - Plan execution timing log")
        print(f"  POST /session/configure - Set volume staging directory")
        print(f"\nPress Ctrl+C to stop")
        print("=" * 60 + "\n")

        await site.start()

        # Keep running
        try:
            while True:
                await asyncio.sleep(3600)
        except asyncio.CancelledError:
            pass
        finally:
            self._running = False
            executor_task.cancel()
            await runner.cleanup()


async def main():
    server = SimpleMicroscopeServer()
    await server.run()


if __name__ == "__main__":
    print("\nStarting Simple Microscope Server...")
    print("This server runs RunEngine in the main thread.\n")

    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n\nServer stopped.")
