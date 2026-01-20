"""
Microscope Server for gently DiSPIM

This server runs in the main thread and handles all hardware operations:
- Micro-Manager core (MMCore)
- Bluesky RunEngine
- Ophyd devices
- SAM embryo detection

The copilot connects to this server via rpyc for hardware control.

Usage:
    python start_server.py
"""

import os
import yaml
import pymmcore
import rpyc
from rpyc.utils.server import ThreadedServer
import numpy as np
from pathlib import Path


def initialize_mmcore(mm_dir: str, config_file: str) -> pymmcore.CMMCore:
    """Initialize MMCore using gently's approach"""
    core = pymmcore.CMMCore()
    core.enableStderrLog(True)

    # Setup MM environment
    os.environ["PATH"] += os.pathsep.join(["", mm_dir])
    core.setDeviceAdapterSearchPaths([mm_dir])

    # Load configuration
    if not os.path.exists(config_file):
        raise FileNotFoundError(f"Configuration file not found: {config_file}")

    core.loadSystemConfiguration(config_file)
    return core


def create_service_class(core: pymmcore.CMMCore):
    """
    Create a MicroscopeService class with shared state.

    This is the correct rpyc pattern for sharing hardware state across clients.
    Each client gets its own service instance, but they all share the same
    core, RunEngine, and devices through class attributes.
    """
    # Initialize shared state once
    print("\nInitializing Bluesky RunEngine...")

    from bluesky import RunEngine
    from databroker import Broker

    shared_RE = RunEngine({})
    shared_db = Broker.named('temp')
    shared_RE.subscribe(shared_db.insert)
    print(f"  RunEngine ready with Databroker catalog: {shared_db.name}")

    # Create Ophyd devices
    print("\nCreating Ophyd devices...")
    try:
        from gently.agent.device_factory import create_devices_from_mmcore
        shared_devices = create_devices_from_mmcore(core)
        print(f"  Created {len(shared_devices)} devices")
    except Exception as e:
        print(f"  Warning: Could not create devices: {e}")
        shared_devices = {}

    # Create service class with shared state as class attributes
    class SharedMicroscopeService(rpyc.Service):
        """Service class with shared hardware state"""

        # Class-level shared state (same for all client connections)
        _shared_core = core
        _shared_RE = shared_RE
        _shared_db = shared_db
        _shared_devices = shared_devices
        _shared_sam_detector = None

        def on_connect(self, conn):
            print("Client connected")

        def on_disconnect(self, conn):
            print("Client disconnected")

        # === Stage Operations ===

        def exposed_move_to_position(self, x: float, y: float) -> dict:
            """Move XY stage to position"""
            import bluesky.plan_stubs as bps

            xy_stage = self._shared_devices.get('xy_stage')
            if not xy_stage:
                return {'error': 'XY stage not available'}

            def plan():
                yield from bps.mv(xy_stage, [x, y])

            try:
                self._shared_RE(plan())
                return {'x': x, 'y': y, 'success': True}
            except Exception as e:
                return {'error': str(e)}

        def exposed_get_stage_position(self) -> tuple:
            """Get current stage position"""
            import bluesky.plan_stubs as bps

            xy_stage = self._shared_devices.get('xy_stage')
            if not xy_stage:
                return (0.0, 0.0)

            def plan():
                yield from bps.trigger_and_read([xy_stage])

            try:
                uids = self._shared_RE(plan())
                if uids:
                    run = self._shared_db[uids[0]]
                    data = run.primary.read()
                    pos = data[xy_stage.name].values[0]
                    return tuple(pos)
            except Exception as e:
                print(f"Error reading stage position: {e}")

            return (0.0, 0.0)

        # === Calibration Operations ===

        def exposed_calibrate_piezo_galvo(self, piezo_positions: list) -> dict:
            """Run piezo-galvo calibration"""
            lightsheet_snap = self._shared_devices.get('lightsheet_snap')
            if not lightsheet_snap:
                return {'error': 'Lightsheet snap device not available'}

            try:
                from gently.plans import calibrate_piezo_galvo_plan

                results = self._shared_RE(calibrate_piezo_galvo_plan(
                    lightsheet_snap=lightsheet_snap,
                    piezo_positions=piezo_positions,
                ))

                if results and 'calibration' in results:
                    return {
                        'calibration': dict(results['calibration']),
                        'success': True
                    }
                return {'error': 'Calibration did not return results'}

            except Exception as e:
                import traceback
                return {'error': str(e), 'traceback': traceback.format_exc()}

        # === Acquisition Operations ===

        def exposed_acquire_volume(
            self,
            num_slices: int,
            exposure_ms: float,
            galvo_amplitude: float,
            galvo_center: float,
            piezo_amplitude: float,
            piezo_center: float,
        ) -> dict:
            """Acquire a single volume"""
            volume_scanner = self._shared_devices.get('volume_scanner')
            if not volume_scanner:
                return {'error': 'Volume scanner not available'}

            try:
                from gently.plans import acquire_single_volume_plan

                results = self._shared_RE(acquire_single_volume_plan(
                    volume_scanner=volume_scanner,
                    num_slices=num_slices,
                    exposure_ms=exposure_ms,
                    galvo_amplitude=galvo_amplitude,
                    galvo_center=galvo_center,
                    piezo_amplitude=piezo_amplitude,
                    piezo_center=piezo_center,
                ))

                if results and 'volume' in results:
                    volume = results['volume']
                    return {
                        'volume': volume,
                        'shape': volume.shape,
                        'dtype': str(volume.dtype),
                        'success': True
                    }
                return {'error': 'Acquisition did not return volume'}

            except Exception as e:
                import traceback
                return {'error': str(e), 'traceback': traceback.format_exc()}

        def exposed_capture_bottom_image(self) -> np.ndarray:
            """Capture image from bottom camera"""
            import bluesky.plan_stubs as bps

            bottom_camera = self._shared_devices.get('bottom_camera')
            if not bottom_camera:
                return np.zeros((100, 100), dtype=np.uint16)

            def plan():
                yield from bps.trigger_and_read([bottom_camera])

            try:
                uids = self._shared_RE(plan())
                if uids:
                    run = self._shared_db[uids[0]]
                    data = run.primary.read()
                    image = data[bottom_camera.name].values[0]
                    return np.array(image)
            except Exception as e:
                print(f"Error capturing image: {e}")

            return np.zeros((100, 100), dtype=np.uint16)

        # === Embryo Detection ===

        def exposed_detect_embryos(
            self,
            pixel_size_um: float,
            objective_mag: float,
            use_claude_review: bool,
            min_confidence: float,
        ) -> dict:
            """Detect embryos using SAM + Claude Vision"""
            import bluesky.plan_stubs as bps

            bottom_camera = self._shared_devices.get('bottom_camera')
            xy_stage = self._shared_devices.get('xy_stage')

            if not bottom_camera:
                return {'error': 'Bottom camera not available'}
            if not xy_stage:
                return {'error': 'XY stage not available'}

            try:
                # Initialize SAM detector if needed (shared across all clients)
                if SharedMicroscopeService._shared_sam_detector is None:
                    print("  Initializing SAM detector...")
                    from gently.agent.sam_detection import SAMEmbryoDetector
                    SharedMicroscopeService._shared_sam_detector = SAMEmbryoDetector(
                        sam_checkpoint="sam_vit_b_01ec64.pth",
                        sam_model_type="vit_b"
                    )

                # Read stage position
                def read_stage_plan():
                    yield from bps.trigger_and_read([xy_stage])

                print("  Reading stage position...")
                uids = self._shared_RE(read_stage_plan())
                run = self._shared_db[uids[0]]
                stage_data = run.primary.read()
                stage_pos = stage_data[xy_stage.name].values[0]
                print(f"  Stage position: ({stage_pos[0]:.1f}, {stage_pos[1]:.1f}) µm")

                # Capture bottom camera image
                def capture_plan():
                    yield from bps.trigger_and_read([bottom_camera])

                print("  Capturing bottom camera image...")
                uids = self._shared_RE(capture_plan())
                run = self._shared_db[uids[0]]
                image_data = run.primary.read()
                image = image_data[bottom_camera.name].values[0]
                print(f"  Captured image: {image.shape}")

                # Run SAM detection
                import asyncio

                # Create event loop for async SAM detection
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)

                try:
                    results = loop.run_until_complete(
                        SharedMicroscopeService._shared_sam_detector.detect_embryos(
                            image=image,
                            stage_position=tuple(stage_pos),
                            pixel_size_um=pixel_size_um,
                            objective_mag=objective_mag,
                            use_claude_review=use_claude_review,
                            save_visualizations=True,
                            output_dir=Path("./detection_results")
                        )
                    )
                finally:
                    loop.close()

                # Return results
                return {
                    'embryos': results.get('embryos', []),
                    'initial_detections': results.get('initial_detections', 0),
                    'final_detections': results.get('final_detections', 0),
                    'verification': results.get('verification', {}),
                    'image': image,
                    'stage_position': list(stage_pos),
                    'success': True
                }

            except Exception as e:
                import traceback
                return {'error': str(e), 'traceback': traceback.format_exc()}

        # === Acquisition Control ===

        def exposed_pause_acquisition(self) -> dict:
            """Pause running acquisition"""
            try:
                self._shared_RE.request_pause()
                return {'success': True, 'state': 'paused'}
            except Exception as e:
                return {'error': str(e)}

        def exposed_resume_acquisition(self) -> dict:
            """Resume paused acquisition"""
            try:
                self._shared_RE.resume()
                return {'success': True, 'state': 'running'}
            except Exception as e:
                return {'error': str(e)}

        def exposed_abort_acquisition(self) -> dict:
            """Abort running acquisition"""
            try:
                self._shared_RE.abort()
                return {'success': True, 'state': 'idle'}
            except Exception as e:
                return {'error': str(e)}

        # === Status ===

        def exposed_get_status(self) -> dict:
            """Get server status"""
            return {
                'run_engine_state': self._shared_RE.state if self._shared_RE else 'not_initialized',
                'devices': list(self._shared_devices.keys()),
                'sam_loaded': SharedMicroscopeService._shared_sam_detector is not None,
            }

        # === Legacy MMCore Access ===

        def exposed_get_core(self):
            """Expose the MMCore instance (for backward compatibility)"""
            return self._shared_core

    return SharedMicroscopeService


def start_server(mm_dir: str, config_file: str, port: int = 18861, hostname: str = "localhost"):
    """Start the microscope server"""
    print("=" * 60)
    print("MICROSCOPE SERVER")
    print("=" * 60)

    print(f"\nInitializing MMCore with {config_file}")
    core = initialize_mmcore(mm_dir, config_file)

    print(f"\nStarting server on {hostname}:{port}")

    # Create service class with shared hardware state
    ServiceClass = create_service_class(core)

    # Configure rpyc to allow all attributes and pickling
    from rpyc.core import DEFAULT_CONFIG
    config = DEFAULT_CONFIG.copy()
    config['allow_all_attrs'] = True
    config['allow_pickle'] = True
    config['sync_request_timeout'] = 300  # 5 minute timeout

    server = ThreadedServer(
        ServiceClass,  # Pass the CLASS, not an instance
        hostname=hostname,
        port=port,
        protocol_config=config
    )

    print("\n" + "=" * 60)
    print(f"Server ready at {hostname}:{port}")
    print("Press Ctrl+C to stop")
    print("=" * 60 + "\n")

    try:
        server.start()
    except KeyboardInterrupt:
        print("\n\nShutting down...")
    finally:
        server.close()
        core.reset()
        print("Server stopped.")


def load_config(config_path="config.yml"):
    """Load configuration from YAML file"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    mm_dir = config['mmdirectory']
    config_file = config['mmconfig']

    print(f"Loaded config from {config_path}")
    print(f"MM Directory: {mm_dir}")
    print(f"MM Config: {config_file}")

    return mm_dir, config_file


if __name__ == "__main__":
    # Load config from file
    mm_dir, config_file = load_config()
    start_server(mm_dir, config_file)
