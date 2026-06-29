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
import contextlib
import json
import logging
import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

# Add project root to path
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import yaml  # noqa: E402
from aiohttp import web  # noqa: E402

# Bluesky imports
from bluesky import RunEngine  # noqa: E402

from gently.core.service import Service  # noqa: E402
from gently.hardware import console_ui as cui  # noqa: E402
from gently.log_config import configure_logging  # noqa: E402
from gently.settings import settings  # noqa: E402

logger = logging.getLogger(__name__)
# BestEffortCallback removed — unused


@dataclass
class PlanRequest:
    """A request to run a plan"""

    plan_name: str
    kwargs: dict[str, Any]
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
        self.system = None  # DiSPIMSystem facade — only place this process touches MMCore directly
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
        self._volume_dir: str | None = None

        # Server lifecycle objects (populated in on_start)
        self._app = None
        self._runner = None
        self._executor_task = None

        # ------------------------------------------------------------------
        # Device-state streaming (ASIdiSPIM StagePositionUpdater pattern)
        # ------------------------------------------------------------------
        # Positions polled at 2 Hz, full system state cache every 15 s.
        # Properties (exposure, gain, laser/shutter state, etc.) almost never
        # change unless something calls a setter, so a slow refresh is fine.
        # Positions change continuously during stage moves and must stay live.
        # Both pollers contend for pymmcore's internal `g_core_lock`, so a
        # short property interval would starve position reads —
        # updateSystemStateCache() takes ~1.5 s on this hardware (Tiger
        # controller over serial).
        # Plans that own MMCore for performance-critical sections (volume
        # acquisition, calibration sweeps, focus sweeps) bracket themselves
        # with `pause_state_updates()` so the poller goes quiet while they
        # run. Plain stage moves / LED changes / snaps don't pause — the
        # adapter's per-device mutex handles the contention fine, and we
        # want the readout to stay live.
        self._state_pos_interval_sec = 0.2  # 5 Hz target for XY (hard floor ~4 Hz on ASI)
        self._state_slow_pos_interval_sec = 1.0  # 1 Hz piezo + galvo
        self._state_prop_interval_sec = 15.0  # ~0.07 Hz full-state cadence
        self._state_pause_counter = 0
        self._state_updating_now = False
        self._state_latest: dict[str, Any] = {
            "positions": {},
            "properties": {},
            "t": 0.0,
            "paused": False,
        }
        self._state_subscribers: list[asyncio.Queue] = []
        self._state_pos_task: asyncio.Task | None = None
        self._state_slow_pos_task: asyncio.Task | None = None
        self._state_prop_task: asyncio.Task | None = None

        # MMCore push-callback support. Adapters that fire OnPropertyChanged /
        # OnXYStagePositionChanged etc. let us skip polling for those events.
        # Whether the ASI Tiger adapter fires on joystick moves is unknown —
        # the bridge logs every callback and broadcasts to a dedicated SSE
        # stream so it can be tested empirically.
        self._mm_callback_bridge = None  # MMEventCallback subclass
        self._mm_callback_loop: asyncio.AbstractEventLoop | None = None
        self._callback_subscribers: list[asyncio.Queue] = []
        # Debounce timer for state-stream broadcasts triggered by callbacks.
        # A flurry of OnPropertyChanged events (e.g. during config-group load)
        # gets coalesced into a single broadcast ~50 ms later.
        self._callback_broadcast_handle: asyncio.Handle | None = None
        self._callback_broadcast_debounce_sec: float = 0.05

        # Bottom-camera live stream (Phase-1 thumbnail). Off by default; the
        # streamer task spins up only while ≥1 SSE subscriber is connected,
        # so the camera is never grabbed when nobody is watching.
        # Tuned for low latency: small thumbnail + cheap auto-contrast keeps
        # the encode path under ~5 ms per frame on the encoding thread.
        self._cam_subscribers: list[asyncio.Queue] = []
        self._cam_task: asyncio.Task | None = None
        self._cam_interval_sec: float = 0.25  # 4 Hz
        self._cam_target_max_dim: int = 360  # ~360px thumbnail
        self._cam_jpeg_quality: int = 55

        # Lightsheet (SPIM) live stream — continuous sequence acquisition.
        self._ls_subscribers: list[asyncio.Queue] = []
        self._ls_task: asyncio.Task | None = None
        self._ls_interval_sec: float = 0.0  # peek as fast as exposure allows
        self._ls_target_max_dim: int = 512
        self._ls_jpeg_quality: int = 70
        self._ls_params: dict = {
            "galvo": 0.0,
            "piezo": 50.0,
            "exposure": 20.0,
            "side": "A",
            # "snap"  → per-frame snapImage()+getImage() — hard memory ceiling, safe default.
            # "continuous" → startContinuousSequenceAcquisition (opt-in, higher FPS).
            "mode": "snap",
        }
        self._ls_seq_started: bool = False
        self._ls_applied: dict = {}  # last-applied galvo/piezo/exposure
        self._ls_parked: dict = {}  # last-parked galvo/piezo setPosition values
        self._ls_spim_idle: bool = False  # whether SPIM state machine was set Idle this session

        # Plans that hold MMCore for long performance-critical work.
        # Anything in this set runs with state polling paused.
        self._heavy_plans = frozenset(
            {
                "acquire_single_volume_plan",
                "burst_plan",
                "timelapse_volume_plan",
                "focus_sweep_plan",
                "calibrate_piezo_galvo_plan",
                "multi_embryo_calibration_workflow",
                # Slow F-drive traverse (25000 -> ~50 um) + fine focus scan + the
                # XY view-registration loop all hold MMCore for a while.
                "spim_head_focus_descent_plan",
                "register_views_xy_plan",
                "spim_head_focus_and_align_plan",
            }
        )

    async def initialize(self):
        """Initialize hardware and RunEngine"""
        logger.info("=" * 60)
        logger.info("GENTLY DEVICE LAYER")
        logger.info("=" * 60)

        cui.out()
        cui.note("Starting device layer...", "bold")

        # [1/5] Load config
        cui.step(1, 5, "Loading configuration")
        logger.info("[1/5] Loading configuration...")
        with open(self.config_path) as f:
            self.config = yaml.safe_load(f)
        logger.info("Config loaded from %s", self.config_path)
        cui.step_done(str(self.config_path))

        # [2/5] MMCore initialization, routed through the DiSPIMSystem facade
        # so this process never touches `core.*` directly outside the
        # devices/ package.
        cui.step(2, 5, "Initializing Micro-Manager core")
        logger.info("[2/5] Initializing Micro-Manager Core (direct)...")
        from .devices.system import DiSPIMSystem

        self.system = DiSPIMSystem()
        self.system.enable_stderr_log(True)

        # Add MM directory to PATH for device adapters
        mm_directory = self.config.get("mmdirectory", "C:/Program Files/Micro-Manager-1.4")
        os.environ["PATH"] += os.pathsep + mm_directory
        self.system.set_device_adapter_search_paths([mm_directory])

        # Load system configuration
        mm_config = self.config.get("mmconfig", "MMConfig.cfg")
        mm_config_path = os.path.join(mm_directory, mm_config)
        if not os.path.exists(mm_config_path):
            # Try config.yml directory
            mm_config_path = os.path.join(os.path.dirname(self.config_path), mm_config)

        logger.info("Loading: %s", mm_config_path)
        self.system.load_system_configuration(mm_config_path)
        logger.info("MMCore initialized (direct, in-process)")
        logger.info("Loaded devices: %s", self.system.get_loaded_devices())
        cui.step_done(Path(mm_config_path).name)

        # Register MMCore event callback so we get push notifications for
        # property changes, stage moves, exposure changes, etc. — anything the
        # adapter chooses to emit. Lives alongside the polling loops; we don't
        # drop polling until we know which events the adapters reliably fire.
        self._register_mmcore_callbacks()

        # [3/5] Create Ophyd devices
        cui.step(3, 5, "Creating devices")
        logger.info("[3/5] Creating Ophyd devices...")
        # Suppress rich console output to avoid Unicode issues on Windows
        import io

        from .device_factory import create_devices_from_mmcore

        old_stdout = sys.stdout
        sys.stdout = io.StringIO()
        try:
            self.devices = create_devices_from_mmcore(self.system.core)
        finally:
            sys.stdout = old_stdout
        logger.info("Created %d devices", len(self.devices))
        for name in self.devices:
            logger.debug("  - %s", name)

        # Optional BLE accessory: SwitchBot Bot. It's a Bluetooth device, not a
        # Micro-Manager adapter, so it's created here (independently of MMCore)
        # and added to the same registry. Plans address it by name, e.g.
        # bps.mv(switchbot, 'on'). Config-gated: no `switchbot:` section => no-op.
        sb_cfg = self.config.get("switchbot")
        if sb_cfg:
            try:
                from gently.hardware.switchbot import SwitchBot

                sb_name = sb_cfg.get("name", "switchbot")
                self.devices[sb_name] = SwitchBot(
                    address=sb_cfg["address"],
                    name=sb_name,
                    timeout=sb_cfg.get("timeout", 20.0),
                )
                logger.info("Created SwitchBot '%s' at %s", sb_name, sb_cfg["address"])
            except Exception as exc:
                logger.warning("Could not create SwitchBot: %s", exc)

        # Optional temperature controller (ACUITYnano). Like the SwitchBot it's
        # not an MMCore adapter — created here from config and added to the same
        # registry. Plans block on it via bps.mv(temperature, 20.0) until the
        # controller reports SYSTEM LOCKED. Config-gated: no `temperature:` => no-op.
        temp_cfg = self.config.get("temperature")
        if temp_cfg:
            try:
                from gently.hardware.temperature import create_temperature_controller

                tc = create_temperature_controller(temp_cfg)
                self.devices[tc.name] = tc
                logger.info(
                    "Created temperature controller '%s' (backend=%s)",
                    tc.name,
                    temp_cfg.get("backend", "serial"),
                )
            except Exception as exc:
                logger.warning("Could not create temperature controller: %s", exc)

        cui.step_done(f"{len(self.devices)} devices")

        # Push XY safety bounds down to the ASI Tiger firmware so the joystick
        # can't drive past Layer-1 software limits. The XY_STAGE_*_UM constants
        # in devices/stage.py are the single source of truth — both the
        # software set() check and the firmware fence read from them.
        # Refuses cleanly (no firmware writes) if the stage is currently
        # outside the requested envelope — operator must drive into bounds
        # first. We do NOT SaveCardSettings so a code-side limit change always
        # wins on next device-layer restart (config-as-code).
        xy_stage = self.devices.get("xy_stage")
        if xy_stage is not None:
            from .devices.stage import (
                XY_STAGE_X_MAX_UM,
                XY_STAGE_X_MIN_UM,
                XY_STAGE_Y_MAX_UM,
                XY_STAGE_Y_MIN_UM,
            )

            try:
                xy_stage.set_firmware_limits(
                    x_min_mm=XY_STAGE_X_MIN_UM / 1000.0,
                    x_max_mm=XY_STAGE_X_MAX_UM / 1000.0,
                    y_min_mm=XY_STAGE_Y_MIN_UM / 1000.0,
                    y_max_mm=XY_STAGE_Y_MAX_UM / 1000.0,
                )
                logger.info(
                    "ASI Tiger firmware soft limits applied: X=[%.2f, %.2f] µm, Y=[%.2f, %.2f] µm",
                    XY_STAGE_X_MIN_UM,
                    XY_STAGE_X_MAX_UM,
                    XY_STAGE_Y_MIN_UM,
                    XY_STAGE_Y_MAX_UM,
                )
            except ValueError as exc:
                # Current position is outside the envelope — refuse to start
                # in an inconsistent state where firmware limits and live
                # position disagree.
                logger.error("Firmware limits refused: %s", exc)
                raise
            except Exception as exc:
                logger.error("Could not apply ASI firmware soft limits: %s", exc)
                raise

            # Tiger persists JoystickEnabled in non-volatile card settings —
            # if a prior session ever called SaveCardSettings with the
            # joystick off, every subsequent boot inherits that state and the
            # physical controller is dead. Force it on at boot so the
            # operator's joystick always works regardless of card history.
            try:
                xy_stage.enable_joystick(True)
            except Exception as exc:
                # Not fatal — the agent can still drive the stage. Log loudly
                # so the operator knows the joystick is unavailable.
                logger.error("Could not enable XY joystick: %s", exc)

        # [4/5] Initialize RunEngine
        cui.step(4, 5, "Initializing RunEngine")
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

            if name == "start":
                self._last_documents = {
                    "start": serialized_doc,
                    "descriptors": [],
                    "events": [],
                    "stop": None,
                }
            elif name == "descriptor":
                self._last_documents["descriptors"].append(serialized_doc)
            elif name == "event":
                self._last_documents["events"].append(serialized_doc)
            elif name == "stop":
                self._last_documents["stop"] = serialized_doc
                self._run_history.append(self._last_documents.copy())

        self.RE.subscribe(collect_docs)
        logger.info("RunEngine ready")
        cui.step_done("ready")

        # [5/5] Load plans
        cui.step(5, 5, "Loading plans")
        logger.info("[5/5] Loading plans...")
        self._load_plans()
        cui.step_done(f"{len(self.plans)} plans")

        logger.info("=" * 60)
        logger.info("Device layer initialized successfully")
        logger.info("=" * 60)

    def _load_plans(self):
        """Load available plans"""
        try:
            from .plans.acquisition import (
                capture_bottom_image_plan,
                capture_lightsheet_image_plan,
                get_light_source_power_plan,
                move_piezo_plan,
                move_scanner_plan,
                move_stage_plan,
                read_stage_plan,
                set_laser_plan,
                set_led_plan,
                set_light_source_power_plan,
            )

            self.plans["move_stage_plan"] = move_stage_plan
            self.plans["read_stage_plan"] = read_stage_plan
            self.plans["capture_bottom_image_plan"] = capture_bottom_image_plan
            self.plans["capture_lightsheet_image_plan"] = capture_lightsheet_image_plan
            self.plans["move_piezo_plan"] = move_piezo_plan
            self.plans["move_scanner_plan"] = move_scanner_plan
            self.plans["set_led_plan"] = set_led_plan
            self.plans["set_laser_plan"] = set_laser_plan
            self.plans["set_light_source_power_plan"] = set_light_source_power_plan
            self.plans["get_light_source_power_plan"] = get_light_source_power_plan
            logger.info("Loaded %d plans", len(self.plans))
        except ImportError as e:
            logger.warning("Could not load some plans: %s", e)

        # Also load main plans if available
        try:
            from .plans.acquisition import (
                acquire_single_volume_plan,
                burst_plan,
                calibrate_piezo_galvo_plan,
            )

            self.plans["calibrate_piezo_galvo_plan"] = calibrate_piezo_galvo_plan
            self.plans["acquire_single_volume_plan"] = acquire_single_volume_plan
            self.plans["burst_plan"] = burst_plan
            logger.info("Loaded main acquisition plans")
        except ImportError:
            logger.info("Main acquisition plans not available")

        # SPIM head focus + dual-view registration plans
        try:
            from .plans.acquisition import (
                register_views_xy_plan,
                spim_head_focus_and_align_plan,
                spim_head_focus_descent_plan,
            )

            self.plans["spim_head_focus_descent_plan"] = spim_head_focus_descent_plan
            self.plans["register_views_xy_plan"] = register_views_xy_plan
            self.plans["spim_head_focus_and_align_plan"] = spim_head_focus_and_align_plan
            logger.info("Loaded SPIM head focus plans")
        except ImportError:
            logger.info("SPIM head focus plans not available")

    def _resolve_device_args(self, kwargs: dict[str, Any]) -> dict[str, Any]:
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
            logger.info(
                "Loading SAM detector (%s on %s)...",
                self._sam_model_type,
                self._sam_device,
            )
            from .sam_detection import SAMEmbryoDetector

            self._sam_detector = SAMEmbryoDetector(
                sam_checkpoint=self._sam_checkpoint,
                sam_model_type=self._sam_model_type,
                device=self._sam_device,
            )
            logger.info("SAM detector ready")

        return self._sam_detector

    # =========================================================================
    # Device State Streaming
    # =========================================================================

    @contextlib.asynccontextmanager
    async def pause_state_updates(self):
        """Bracket performance-critical MMCore work so the state poller backs off.

        Reference-counted: nested callers stack, polling only resumes when the
        outermost context exits. Matches ASIdiSPIM's StagePositionUpdater pattern.
        """
        self._state_pause_counter += 1
        self._state_latest["paused"] = True
        try:
            yield
        finally:
            self._state_pause_counter = max(0, self._state_pause_counter - 1)
            if self._state_pause_counter == 0:
                self._state_latest["paused"] = False

    def _read_xy_position(self) -> dict[str, Any]:
        """Read just the XY stage via the ophyd device's ``read()``.

        Two ASI serial round-trips (~250 ms) — the cost is in the underlying
        ASI XYStage adapter (`W X` + `W Y`), not in the ophyd layer. We go
        through ``xy_stage.read()`` so the only place that touches MMCore is
        inside the ophyd device class.
        """
        out: dict[str, Any] = {}
        xy = self.devices.get("xy_stage")
        if xy is not None:
            try:
                data = xy.read()
                value = data[xy.name]["value"]
                out[xy.name] = {
                    "X": float(value[0]),
                    "Y": float(value[1]),
                    "kind": "xy_stage",
                }
            except Exception as exc:
                logger.debug("XY position read failed: %s", exc)
        return out

    def _read_slow_positions(self) -> dict[str, Any]:
        """Read piezo + galvo via their ophyd ``read()`` methods.

        These rarely change on their own — piezo by Z-knob or commands, galvo
        only programmatically — so a 1 Hz cadence is plenty.
        """
        out: dict[str, Any] = {}

        piezo = self.devices.get("piezo")
        if piezo is not None:
            try:
                data = piezo.read()
                out[piezo.name] = {
                    "Position": float(data[piezo.name]["value"]),
                    "kind": "piezo",
                }
            except Exception as exc:
                logger.debug("Piezo position read failed: %s", exc)

        scanner = self.devices.get("scanner")
        if scanner is not None:
            try:
                data = scanner.read()
                value = data[scanner.name]["value"]
                out[scanner.name] = {
                    "A": float(value[0]),
                    "B": float(value[1]),
                    "kind": "galvo",
                }
            except Exception as exc:
                logger.debug("Galvo position read failed: %s", exc)

        return out

    def _read_full_state(self) -> dict[str, dict[str, str]]:
        """Snapshot every property of every loaded MM device via the system state cache.

        `update_system_state_cache()` rereads from hardware, then
        `get_system_state_cache()` returns the Configuration without further
        hardware traffic. One bulk call covers every device — no need to
        enumerate diSPIM devices by hand. All MMCore traffic routes through
        the DiSPIMSystem facade.
        """
        try:
            self.system.update_system_state_cache()
            cfg = self.system.get_system_state_cache()
        except Exception as exc:
            logger.debug("System state cache read failed: %s", exc)
            return {}

        by_device: dict[str, dict[str, str]] = {}
        try:
            size = cfg.size()
        except Exception:
            return {}

        for i in range(size):
            try:
                s = cfg.getSetting(i)
                dev = s.getDeviceLabel()
                prop = s.getPropertyName()
                val = s.getPropertyValue()
            except Exception:
                continue
            by_device.setdefault(dev, {})[prop] = val

        # Tag with device type so the UI can group/format sensibly.
        for dev in list(by_device):
            try:
                by_device[dev]["__type__"] = self.system.get_device_type(dev)
            except Exception:
                pass

        return by_device

    async def _position_poller(self):
        """Fast loop: only reads positions and broadcasts.

        Runs at `_state_pos_interval_sec` (default 2 Hz) and stays responsive
        even when the property poller is busy with a slow MMCore property
        snapshot.

        Honours `_state_pause_counter`: while > 0 it emits heartbeats every
        ~2 s so the browser can show a "paused" indicator.
        """
        logger.info(
            "Position poller started (target=%.1f Hz)",
            1.0 / self._state_pos_interval_sec,
        )
        last_heartbeat = 0.0

        while not getattr(self, "_shutdown_event", asyncio.Event()).is_set():
            tick_start = time.monotonic()
            try:
                if self._state_pause_counter > 0:
                    now = time.time()
                    if now - last_heartbeat > 2.0:
                        await self._broadcast_state(
                            {
                                **self._state_latest,
                                "t": now,
                                "paused": True,
                                "heartbeat": True,
                            }
                        )
                        last_heartbeat = now
                    await asyncio.sleep(self._state_pos_interval_sec)
                    continue

                read_start = time.monotonic()
                xy_positions = await asyncio.to_thread(self._read_xy_position)
                read_elapsed = time.monotonic() - read_start
                # XY is two serial round-trips on ASI Tiger (~250 ms). Warn if
                # we exceed 400 ms — that means something else is holding the
                # MMCore lock.
                if read_elapsed > 0.4:
                    logger.warning(
                        "XY position read slow: %.2fs (target<%.2fs)",
                        read_elapsed,
                        self._state_pos_interval_sec,
                    )

                now = time.time()
                # Merge XY into existing positions dict — preserves piezo/galvo
                # entries published by the slow-positions poller.
                merged_positions = {
                    **self._state_latest.get("positions", {}),
                    **xy_positions,
                }
                self._state_latest = {
                    **self._state_latest,
                    "positions": merged_positions,
                    "t": now,
                    "paused": False,
                }
                await self._broadcast_state(self._state_latest)
                last_heartbeat = now

            except asyncio.CancelledError:
                logger.info("Position poller cancelled")
                raise
            except Exception:
                logger.exception("Position poller iteration failed")

            elapsed = time.monotonic() - tick_start
            await asyncio.sleep(max(0.0, self._state_pos_interval_sec - elapsed))

    async def _slow_positions_poller(self):
        """Piezo + galvo polling. Joystick doesn't move these, so 1 Hz is plenty.

        Splitting these off the XY fast path was the key to getting XY up to
        ~4 Hz: each ASI serial round-trip is ~125 ms, and XY alone takes two.
        """
        logger.info(
            "Slow-positions poller started (target=%.2f Hz)",
            1.0 / self._state_slow_pos_interval_sec,
        )

        while not getattr(self, "_shutdown_event", asyncio.Event()).is_set():
            tick_start = time.monotonic()
            try:
                if self._state_pause_counter > 0:
                    await asyncio.sleep(self._state_slow_pos_interval_sec)
                    continue

                read_start = time.monotonic()
                slow_positions = await asyncio.to_thread(self._read_slow_positions)
                read_elapsed = time.monotonic() - read_start
                if read_elapsed > 0.6:
                    logger.warning(
                        "Slow-positions read slow: %.2fs",
                        read_elapsed,
                    )

                # Merge — don't clobber XY entries the fast poller maintains.
                merged_positions = {
                    **self._state_latest.get("positions", {}),
                    **slow_positions,
                }
                self._state_latest = {
                    **self._state_latest,
                    "positions": merged_positions,
                    "t": time.time(),
                }
                await self._broadcast_state(self._state_latest)

            except asyncio.CancelledError:
                logger.info("Slow-positions poller cancelled")
                raise
            except Exception:
                logger.exception("Slow-positions poller iteration failed")

            elapsed = time.monotonic() - tick_start
            await asyncio.sleep(max(0.0, self._state_slow_pos_interval_sec - elapsed))

    async def _property_poller(self):
        """Slow loop: refreshes the full MMCore property cache.

        Runs independently of the position poller so a multi-second
        `updateSystemStateCache()` call cannot stall position updates. Updates
        `_state_latest['properties']` in place and broadcasts.
        """
        logger.info(
            "Property poller started (target=%.2f Hz)",
            1.0 / self._state_prop_interval_sec,
        )

        while not getattr(self, "_shutdown_event", asyncio.Event()).is_set():
            tick_start = time.monotonic()
            try:
                if self._state_pause_counter > 0:
                    await asyncio.sleep(self._state_prop_interval_sec)
                    continue

                read_start = time.monotonic()
                properties = await asyncio.to_thread(self._read_full_state)
                read_elapsed = time.monotonic() - read_start
                if read_elapsed > 1.0:
                    logger.warning(
                        "Property read slow: %.2fs",
                        read_elapsed,
                    )

                self._state_latest = {
                    **self._state_latest,
                    "properties": properties,
                    "t": time.time(),
                }
                await self._broadcast_state(self._state_latest)

            except asyncio.CancelledError:
                logger.info("Property poller cancelled")
                raise
            except Exception:
                logger.exception("Property poller iteration failed")

            elapsed = time.monotonic() - tick_start
            await asyncio.sleep(max(0.0, self._state_prop_interval_sec - elapsed))

    async def _broadcast_state(self, payload: dict[str, Any]):
        """Push a state payload to every SSE subscriber. Drop slow clients."""
        if not self._state_subscribers:
            return
        dead: list[asyncio.Queue] = []
        for q in self._state_subscribers:
            try:
                q.put_nowait(payload)
            except asyncio.QueueFull:
                dead.append(q)
        for q in dead:
            try:
                self._state_subscribers.remove(q)
            except ValueError:
                pass
            logger.warning("Dropped slow device-state subscriber")

    # =========================================================================
    # Bottom-camera live stream (Phase 1: low-rate thumbnail)
    # =========================================================================

    def _capture_bottom_frame_sync(self) -> np.ndarray | None:
        """Grab a single frame via the ophyd device's synchronous ``snap()``.

        Blocking — call via ``asyncio.to_thread``. All MMCore traffic happens
        inside ``DiSPIMCamera.snap()``; the streamer holds no direct core
        handle.
        """
        cam = self.devices.get("bottom_camera")
        if cam is None:
            return None
        try:
            return cam.snap()
        except Exception as exc:
            logger.debug("Bottom-camera grab failed: %s", exc)
            return None

    def _encode_frame_for_stream(
        self, img: np.ndarray, max_dim: int | None = None, quality: int | None = None
    ) -> dict[str, Any] | None:
        """Downsample + auto-contrast + JPEG-encode a uint16 frame for SSE.

        Optimised for streaming throughput:
          * stride-slice downsample to ~360 px max dim (cheap, no interp)
          * auto-contrast computed on a 4096-pixel random sample, not the
            full image — np.partition on the subsample is O(n) and avoids
            sorting ~120K pixels every frame
          * JPEG quality 55 (visually fine at thumbnail size)

        ``max_dim`` and ``quality`` default to the ``_cam_*`` instance values,
        allowing callers (e.g. the lightsheet streamer) to pass different
        settings without duplicating the encoder.
        """
        if img is None or img.size == 0:
            return None
        try:
            import base64

            import cv2  # opencv ships with the agent env (SAM uses it)
        except ImportError as exc:
            logger.warning("Cannot encode frame — OpenCV unavailable: %s", exc)
            return None

        target_max_dim = max_dim if max_dim is not None else self._cam_target_max_dim
        jpeg_quality = quality if quality is not None else self._cam_jpeg_quality

        h, w = img.shape[:2]
        # Stride slicing — no interpolation, just take every Nth pixel.
        factor = max(1, max(h, w) // target_max_dim)
        small = img[::factor, ::factor]

        # Auto-contrast off a small random sample. Robust to hot pixels
        # without paying for a full-image percentile.
        if small.dtype != np.uint8:
            flat = small.ravel()
            sample_size = min(4096, flat.size)
            # Strided sample — deterministic, no PRNG cost.
            step = max(1, flat.size // sample_size)
            sample = flat[::step]
            # np.partition gets the [1%, 99%] order stats in O(n).
            k_lo = max(0, int(sample.size * 0.01))
            k_hi = min(sample.size - 1, int(sample.size * 0.99))
            part = np.partition(sample, [k_lo, k_hi])
            lo = float(part[k_lo])
            hi = float(part[k_hi])
            if hi <= lo:
                hi = lo + 1.0
            scale = 255.0 / (hi - lo)
            small = np.clip((small.astype(np.float32) - lo) * scale, 0, 255).astype(np.uint8)

        ok, jpeg = cv2.imencode(".jpg", small, [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality])
        if not ok:
            return None
        b64 = base64.b64encode(jpeg.tobytes()).decode("ascii")
        return {
            "t": time.time(),
            "shape": [int(small.shape[0]), int(small.shape[1])],
            "downsample": factor,
            "mime": "image/jpeg",
            "jpeg_b64": b64,
        }

    async def _bottom_camera_streamer(self):
        """Continuous grab/encode/broadcast loop. Runs while a subscriber lives.

        Yields the MMCore lock between grabs and obeys ``pause_state_updates()``,
        so heavy plans aren't interfered with. When the last subscriber drops,
        the loop returns and the task is allowed to exit; it gets started up
        again on the next subscription.
        """
        logger.info("Bottom-camera streamer started")
        try:
            while self._cam_subscribers:
                if self._state_pause_counter > 0:
                    # A heavy plan owns MMCore; back off until it releases.
                    await asyncio.sleep(self._cam_interval_sec)
                    continue
                tick = time.monotonic()
                img = await asyncio.to_thread(self._capture_bottom_frame_sync)
                payload = self._encode_frame_for_stream(img) if img is not None else None
                if payload is not None:
                    await self._broadcast_camera(payload)
                # Pace the loop — sleep whatever's left of the interval.
                elapsed = time.monotonic() - tick
                await asyncio.sleep(max(0.0, self._cam_interval_sec - elapsed))
        except asyncio.CancelledError:
            logger.info("Bottom-camera streamer cancelled")
            raise
        except Exception:
            logger.exception("Bottom-camera streamer crashed")
        finally:
            logger.info("Bottom-camera streamer exiting")

    async def _broadcast_camera(self, payload: dict[str, Any]):
        if not self._cam_subscribers:
            return
        dead: list[asyncio.Queue] = []
        for q in self._cam_subscribers:
            try:
                q.put_nowait(payload)
            except asyncio.QueueFull:
                # Slow client — drop oldest then push, so steady-state clients
                # see fresh frames instead of getting kicked off.
                try:
                    _ = q.get_nowait()
                    q.put_nowait(payload)
                except Exception:
                    dead.append(q)
        for q in dead:
            try:
                self._cam_subscribers.remove(q)
            except ValueError:
                pass

    # =========================================================================
    # Lightsheet (SPIM) Live Streamer — continuous sequence acquisition
    # =========================================================================

    def _park_lightsheet_sync(self) -> None:
        """Park scanner galvo + imaging piezo at the current live params (static sheet).

        Side 'A' drives devices["scanner"] / devices["piezo"] (Scanner:AB:33 /
        PiezoStage:P:34).  Side 'B' drives devices["scanner_b"] / devices["piezo_b"]
        (Scanner:CD:33 / PiezoStage:Q:35) when registered.  Falls back to side A
        with a warning on single-side rigs so they are never broken.

        Guards:
        - set_spim_state("Idle") fires once per stream session, not per frame
          (4 serial round-trips on first call → 0 on every subsequent frame).
        - setPosition fires only when the commanded value differs from the
          last-applied value (no-op on steady-state frames).
        """
        p = self._ls_params
        side = p.get("side", "A")

        # Resolve scanner for the requested side (with fallback to A).
        if side == "B":
            if "scanner_b" in self.devices:
                scanner = self.devices["scanner_b"]
            else:
                logger.warning(
                    "Side B requested but scanner_b not registered; falling back to side A for scanner"  # noqa: E501
                )
                scanner = self.devices.get("scanner")
        else:
            scanner = self.devices.get("scanner")

        # Resolve piezo for the requested side (with fallback to A).
        if side == "B":
            if "piezo_b" in self.devices:
                piezo = self.devices["piezo_b"]
            else:
                logger.warning(
                    "Side B requested but piezo_b not registered; falling back to side A for piezo"
                )
                piezo = self.devices.get("piezo")
        else:
            piezo = self.devices.get("piezo")

        # SPIM Idle state machine: drive both devices Idle once per session.
        if not self._ls_spim_idle:
            if scanner is not None:
                try:
                    scanner.set_spim_state("Idle")
                except Exception:
                    pass
            if piezo is not None:
                try:
                    piezo.set_spim_state("Idle")
                except Exception:
                    pass
            self._ls_spim_idle = True
        # setPosition only when the value changed from last-applied.
        if scanner is not None:
            want = float(p["galvo"])
            if self._ls_parked.get("galvo") != want:
                scanner.sa_offset_y.setPosition(want)
                self._ls_parked["galvo"] = want
        if piezo is not None:
            want = float(p["piezo"])
            if self._ls_parked.get("piezo") != want:
                piezo.setPosition(want)
                self._ls_parked["piezo"] = want

    def _ensure_lightsheet_sequence_sync(self) -> None:
        """Configure (or reconfigure on exposure/side change) the SPIM camera.

        Two modes gated by ``_ls_params["mode"]``:

        "snap" (default, safe) — per-frame snapImage()+getImage().  No circular
            buffer at all; memory is bounded to one frame.  Sets camera device
            and exposure only when a reconfiguration is needed; never calls
            startContinuousSequenceAcquisition.

        "continuous" (opt-in) — startContinuousSequenceAcquisition with a hard
            footprint cap (256 MB) so the buffer can't balloon on a production
            box.  The grab loop drains with clearCircularBuffer() after each peek.

        Side 'A' uses devices["camera"] (HamCam1); side 'B' uses devices["camera_b"]
        (HamCam2) when registered.  If side B is requested but camera_b is absent,
        falls back to side A and logs a warning — single-camera rigs are unaffected.
        """
        core = self.system.core
        p = self._ls_params
        side = p.get("side", "A")
        mode = p.get("mode", "snap")

        # Resolve the camera for the requested side
        if side == "B" and "camera_b" in self.devices:
            cam = self.devices["camera_b"]
        else:
            if side == "B":
                logger.warning(
                    "Side B requested but camera_b not registered; falling back to side A"
                )
            cam = self.devices.get("camera")

        if cam is None:
            raise RuntimeError("No lightsheet camera configured")

        need_reconfigure = (
            not self._ls_seq_started
            or self._ls_applied.get("exposure") != p["exposure"]
            or self._ls_applied.get("side") != side
        )

        if mode == "snap":
            # Snap mode: configure camera/exposure on change; no continuous sequence.
            if need_reconfigure:
                # Stop any lingering continuous sequence before snap takes over.
                if core.isSequenceRunning():
                    core.stopSequenceAcquisition()
                    try:
                        core.clearCircularBuffer()
                    except Exception:
                        pass
                if core.getCameraDevice() != cam.name:
                    core.setCameraDevice(cam.name)
                core.setExposure(cam.name, float(p["exposure"]))
                self._ls_seq_started = True  # "camera configured" flag; no sequence running
                self._ls_applied["exposure"] = p["exposure"]
                self._ls_applied["side"] = side
        else:
            # Continuous mode: cap buffer footprint then start sequence.
            if need_reconfigure:
                if core.isSequenceRunning():
                    core.stopSequenceAcquisition()
                if core.getCameraDevice() != cam.name:
                    core.setCameraDevice(cam.name)
                core.setExposure(cam.name, float(p["exposure"]))
                # Hard cap: prevent buffer balloon that crashed the production box.
                try:
                    core.setCircularBufferMemoryFootprint(256)
                except Exception:
                    logger.debug(
                        "setCircularBufferMemoryFootprint not available on this core",
                        exc_info=True,
                    )
                core.startContinuousSequenceAcquisition(self._ls_interval_sec * 1000.0)
                self._ls_seq_started = True
                self._ls_applied["exposure"] = p["exposure"]
                self._ls_applied["side"] = side

    def _grab_lightsheet_frame_sync(self):
        """Park → configure camera → grab one frame.

        Snap mode (default): snapImage()+getImage() — no circular buffer,
        hard memory ceiling, safe for long live-view sessions.

        Continuous mode (opt-in): peeks via getLastImage() then drains the
        buffer with clearCircularBuffer() to prevent it sitting full.
        """
        try:
            self._park_lightsheet_sync()
            self._ensure_lightsheet_sequence_sync()
            try:
                from gently.hardware.dispim.devices.acquisition import _safe_obtain
            except (ImportError, AttributeError):
                _safe_obtain = None
            core = self.system.core
            mode = self._ls_params.get("mode", "snap")
            if mode == "snap":
                core.snapImage()
                img = core.getImage()
            else:
                img = core.getLastImage()
                try:
                    core.clearCircularBuffer()
                except Exception:
                    pass
            if _safe_obtain is not None:
                try:
                    img = _safe_obtain(img)
                except Exception:
                    pass
            return np.asarray(img)
        except Exception as exc:
            logger.debug("Lightsheet grab failed: %s", exc)
            return None

    def _stop_lightsheet_sequence_sync(self) -> None:
        try:
            if self.system.core.isSequenceRunning():
                self.system.core.stopSequenceAcquisition()
        except Exception:
            logger.debug("stop lightsheet sequence failed", exc_info=True)
        # Drain the buffer on every stop path (snap mode: no-op; continuous: release memory).
        try:
            self.system.core.clearCircularBuffer()
        except Exception:
            pass
        self._ls_seq_started = False
        self._ls_applied = {}
        # Reset park guard so the next stream session re-idles the state
        # machine and re-parks the axes (hardware may have moved).
        self._ls_spim_idle = False
        self._ls_parked = {}

    async def _lightsheet_streamer(self):
        logger.info("Lightsheet streamer started")
        try:
            while self._ls_subscribers:
                if self._state_pause_counter > 0:
                    # Heavy plan owns MMCore: release the sequence and back off.
                    if self._ls_seq_started:
                        await asyncio.to_thread(self._stop_lightsheet_sequence_sync)
                    await asyncio.sleep(0.1)
                    continue
                tick = time.monotonic()
                img = await asyncio.to_thread(self._grab_lightsheet_frame_sync)
                payload = (
                    self._encode_frame_for_stream(
                        img, self._ls_target_max_dim, self._ls_jpeg_quality
                    )
                    if img is not None
                    else None
                )
                if payload is not None:
                    await self._broadcast_lightsheet(payload)
                elapsed = time.monotonic() - tick
                # Pace to at least the exposure; peek-rate caps near the camera rate.
                floor = max(self._ls_interval_sec, self._ls_params["exposure"] / 1000.0)
                await asyncio.sleep(max(0.0, floor - elapsed))
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.exception("Lightsheet streamer crashed")
        finally:
            await asyncio.to_thread(self._stop_lightsheet_sequence_sync)
            logger.info("Lightsheet streamer exiting")

    async def _broadcast_lightsheet(self, payload: dict[str, Any]):
        if not self._ls_subscribers:
            return
        dead: list[asyncio.Queue] = []
        for q in self._ls_subscribers:
            try:
                q.put_nowait(payload)
            except asyncio.QueueFull:
                try:
                    _ = q.get_nowait()
                    q.put_nowait(payload)
                except Exception:
                    dead.append(q)
        for q in dead:
            try:
                self._ls_subscribers.remove(q)
            except ValueError:
                pass

    async def handle_lightsheet_stream(self, request):
        """GET /api/lightsheet/stream — SSE of base64-JPEG frames from SPIM camera.

        The streamer task spins up on first connect and exits when the last
        subscriber leaves. Uses continuous sequence acquisition for best FPS.
        """
        response = web.StreamResponse(
            status=200,
            reason="OK",
            headers={
                "Content-Type": "text/event-stream",
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            },
        )
        await response.prepare(request)

        queue: asyncio.Queue = asyncio.Queue(maxsize=4)
        self._ls_subscribers.append(queue)
        if len(self._ls_subscribers) == 1 and (self._ls_task is None or self._ls_task.done()):
            self._ls_task = asyncio.create_task(
                self._lightsheet_streamer(), name="lightsheet-streamer"
            )
        try:
            await response.write(b": connected\n\n")
            while True:
                try:
                    payload = await asyncio.wait_for(queue.get(), timeout=10.0)
                except asyncio.TimeoutError:
                    await response.write(b": keepalive\n\n")
                    continue
                if payload is None:
                    break
                await response.write(f"data: {json.dumps(payload)}\n\n".encode())
        except (asyncio.CancelledError, ConnectionResetError, ConnectionAbortedError):
            pass
        except Exception:
            logger.exception("Lightsheet SSE writer failed")
        finally:
            try:
                self._ls_subscribers.remove(queue)
            except ValueError:
                pass
        return response

    async def handle_lightsheet_params(self, request):
        """POST /api/lightsheet/live/params — update live galvo/piezo/exposure/side.

        Body: {"galvo": float, "piezo": float, "exposure": float, "side": "A"|"B"} (all optional).
        Galvo/piezo apply on the next grab; exposure or side change triggers sequence restart.
        """
        body = await request.json()
        for k in ("galvo", "piezo", "exposure"):
            if k in body and body[k] is not None:
                self._ls_params[k] = float(body[k])
        if "side" in body and body["side"] in ("A", "B"):
            self._ls_params["side"] = body["side"]
        return web.json_response({"params": self._ls_params})

    # =========================================================================
    # MMCore Push Callbacks
    # =========================================================================

    def _register_mmcore_callbacks(self):
        """Wire a pymmcore.MMEventCallback to log + broadcast hardware events.

        Some adapters (ASI Tiger included, possibly) emit callbacks for hardware
        changes — joystick moves, property updates, exposure changes — without
        anyone calling MMCore. When they do, we can react with zero polling
        latency. When they don't, the existing pollers cover us.

        Callbacks fire on the MMCore worker thread, so every handler marshals
        the payload onto the asyncio loop via call_soon_threadsafe before
        touching shared state.
        """
        import pymmcore

        self._mm_callback_loop = asyncio.get_running_loop()
        outer = self

        class _Bridge(pymmcore.MMEventCallback):
            def _emit(self, kind: str, **payload):
                payload = {"t": time.time(), "kind": kind, **payload}
                logger.info(
                    "MMCore callback: %s %s",
                    kind,
                    {k: v for k, v in payload.items() if k != "t"},
                )
                loop = outer._mm_callback_loop
                if loop is None or loop.is_closed():
                    return
                loop.call_soon_threadsafe(outer._enqueue_callback, payload)

            def onPropertyChanged(self, dev, prop, value):
                self._emit("property_changed", device=dev, property=prop, value=value)

            def onPropertiesChanged(self):
                self._emit("properties_changed")

            def onConfigGroupChanged(self, group, new_config):
                self._emit("config_group_changed", group=group, config=new_config)

            def onChannelGroupChanged(self, new_channel_group):
                self._emit("channel_group_changed", group=new_channel_group)

            def onExposureChanged(self, camera, exposure):
                self._emit("exposure_changed", camera=camera, exposure_ms=exposure)

            def onSLMExposureChanged(self, slm, exposure):
                self._emit("slm_exposure_changed", slm=slm, exposure_ms=exposure)

            def onStagePositionChanged(self, dev, pos):
                self._emit("stage_position_changed", device=dev, position=pos)

            def onXYStagePositionChanged(self, dev, x, y):
                self._emit("xy_stage_position_changed", device=dev, x=x, y=y)

            def onPixelSizeChanged(self, new_pixel_size_um):
                self._emit("pixel_size_changed", um=new_pixel_size_um)

            def onPixelSizeAffineChanged(self, v0, v1, v2, v3, v4, v5):
                self._emit("pixel_size_affine_changed", affine=[v0, v1, v2, v3, v4, v5])

            def onSystemConfigurationLoaded(self):
                self._emit("system_configuration_loaded")

        self._mm_callback_bridge = _Bridge()
        # MMCore takes ownership; keep a Python ref so it doesn't get GC'd.
        self.system.register_callback(self._mm_callback_bridge)
        logger.info("MMCore callback bridge registered")

    def _enqueue_callback(self, payload: dict[str, Any]):
        """Runs on the asyncio loop (via call_soon_threadsafe).

        Two jobs: forward to /api/devices/callbacks/stream subscribers (for
        diagnostics), and mirror the change into `_state_latest` so the main
        device-state stream picks it up within ~50 ms — no waiting for the
        next property poll cycle.
        """
        # 1. Forward to the diagnostic callback stream.
        if self._callback_subscribers:
            dead: list[asyncio.Queue] = []
            for q in self._callback_subscribers:
                try:
                    q.put_nowait(payload)
                except asyncio.QueueFull:
                    dead.append(q)
            for q in dead:
                try:
                    self._callback_subscribers.remove(q)
                except ValueError:
                    pass
                logger.warning("Dropped slow callback subscriber")

        # 2. Mirror into _state_latest where applicable.
        if self._apply_callback_to_state(payload):
            self._schedule_callback_broadcast()

    def _apply_callback_to_state(self, payload: dict[str, Any]) -> bool:
        """Translate a callback payload into a `_state_latest` mutation.

        Returns True iff something visible changed (the caller will then
        schedule a debounced broadcast). Property changes overwrite the cached
        string value; position-changed callbacks (rare on ASI) refresh the
        positions block.
        """
        kind = payload.get("kind")

        if kind == "property_changed":
            dev = payload.get("device")
            prop = payload.get("property")
            value = payload.get("value")
            if not dev or not prop:
                return False
            props = self._state_latest.setdefault("properties", {})
            bundle = props.setdefault(dev, {})
            if bundle.get(prop) == value:
                return False
            bundle[prop] = value
            self._state_latest["t"] = payload.get("t", time.time())
            return True

        if kind == "exposure_changed":
            dev = payload.get("camera")
            new_ms = payload.get("exposure_ms")
            if not dev or new_ms is None:
                return False
            props = self._state_latest.setdefault("properties", {})
            bundle = props.setdefault(dev, {})
            bundle["Exposure"] = str(new_ms)
            self._state_latest["t"] = payload.get("t", time.time())
            return True

        if kind == "xy_stage_position_changed":
            dev = payload.get("device")
            x = payload.get("x")
            y = payload.get("y")
            if not dev or x is None or y is None:
                return False
            positions = self._state_latest.setdefault("positions", {})
            positions[dev] = {"X": float(x), "Y": float(y), "kind": "xy_stage"}
            self._state_latest["t"] = payload.get("t", time.time())
            return True

        if kind == "stage_position_changed":
            dev = payload.get("device")
            pos = payload.get("position")
            if not dev or pos is None:
                return False
            positions = self._state_latest.setdefault("positions", {})
            entry = positions.setdefault(dev, {})
            entry["Position"] = float(pos)
            entry.setdefault("kind", "piezo")
            self._state_latest["t"] = payload.get("t", time.time())
            return True

        # Other kinds (config_group_changed, pixel_size_changed, etc.) are
        # informational — they go to the diagnostic stream but don't change
        # any device-state field we currently expose.
        return False

    def _schedule_callback_broadcast(self):
        """Debounce: a burst of callbacks coalesces into one broadcast."""
        if self._callback_broadcast_handle is not None:
            return
        loop = self._mm_callback_loop
        if loop is None or loop.is_closed():
            return
        self._callback_broadcast_handle = loop.call_later(
            self._callback_broadcast_debounce_sec,
            self._fire_callback_broadcast,
        )

    def _fire_callback_broadcast(self):
        self._callback_broadcast_handle = None
        # Snapshot so we don't race with poller mutations during the await.
        snapshot = {
            **self._state_latest,
            "positions": dict(self._state_latest.get("positions", {})),
            "properties": dict(self._state_latest.get("properties", {})),
        }
        asyncio.create_task(self._broadcast_state(snapshot))

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
                request = await asyncio.wait_for(self._plan_queue.get(), timeout=1.0)
            except asyncio.TimeoutError:
                continue

            # Log execution start with timestamp
            start_time = datetime.now()
            execution_record = {
                "plan_name": request.plan_name,
                "kwargs": {k: str(v) for k, v in request.kwargs.items()},  # Stringify for JSON
                "start_time": start_time.isoformat(),
                "start_time_formatted": start_time.strftime("%H:%M:%S.%f")[:-3],
            }

            logger.info(
                ">>> [%s] Executing: %s",
                start_time.strftime("%H:%M:%S"),
                request.plan_name,
            )

            # Reset documents before each plan so stale results from
            # a previous plan (e.g. volume file refs from acquire) don't
            # leak into the response of a simple plan (e.g. move).
            self._last_documents = {}

            try:
                # Get the plan function
                if request.plan_name not in self.plans:
                    raise ValueError(f"Unknown plan: {request.plan_name}")

                plan_func = self.plans[request.plan_name]

                # Resolve device arguments
                resolved_kwargs = self._resolve_device_args(request.kwargs)

                # Create the plan generator
                plan = plan_func(**resolved_kwargs)

                # For heavy plans (sequence acquisitions, calibration/focus
                # sweeps) pause the state poller so it doesn't compete with
                # MMCore for camera streaming bandwidth. Plain moves, snaps,
                # LED changes don't pause — those run with live polling.
                if request.plan_name in self._heavy_plans:
                    async with self.pause_state_updates():
                        result = self.RE(plan)
                else:
                    # Run the plan (this happens in the main thread!)
                    result = self.RE(plan)

                # Get the run UID
                uid = result[0] if result else None

                # Log completion
                end_time = datetime.now()
                duration_ms = (end_time - start_time).total_seconds() * 1000

                execution_record.update(
                    {
                        "end_time": end_time.isoformat(),
                        "end_time_formatted": end_time.strftime("%H:%M:%S.%f")[:-3],
                        "duration_ms": duration_ms,
                        "success": True,
                        "uid": uid,
                    }
                )

                logger.info(
                    "<<< [%s] Complete: %s (%.0fms)",
                    end_time.strftime("%H:%M:%S"),
                    request.plan_name,
                    duration_ms,
                )

                # Complete the future with result
                request.future.set_result(
                    {
                        "success": True,
                        "uid": uid,
                        "documents": self._last_documents.copy(),
                    }
                )

            except Exception as e:
                end_time = datetime.now()
                duration_ms = (end_time - start_time).total_seconds() * 1000

                execution_record.update(
                    {
                        "end_time": end_time.isoformat(),
                        "end_time_formatted": end_time.strftime("%H:%M:%S.%f")[:-3],
                        "duration_ms": duration_ms,
                        "success": False,
                        "error": str(e),
                    }
                )

                logger.error(
                    "<<< [%s] Failed: %s - %s",
                    end_time.strftime("%H:%M:%S"),
                    request.plan_name,
                    e,
                )
                request.future.set_exception(e)

            # Store execution record
            self._plan_execution_log.append(execution_record)
            # Keep last 1000 entries
            if len(self._plan_execution_log) > 1000:
                self._plan_execution_log = self._plan_execution_log[-1000:]

    async def submit_plan(self, plan_name: str, kwargs: dict | None = None) -> dict:
        """Submit a plan and wait for completion"""
        kwargs = kwargs or {}

        # Create request with a future
        loop = asyncio.get_event_loop()
        request = PlanRequest(plan_name=plan_name, kwargs=kwargs, future=loop.create_future())

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
            "manager_state": "idle" if self._running else "stopped",
            "re_state": "idle",
            "devices": list(self.devices.keys()),
            "plans": list(self.plans.keys()),
            "queue_size": self._plan_queue.qsize(),
            "sam_loaded": self._sam_detector is not None,
        }
        return web.json_response(status)

    async def handle_submit_plan(self, request):
        """POST /api/queue/item/add"""
        try:
            data = await request.json()
            plan_name = data.get("item", {}).get("name")
            kwargs = data.get("item", {}).get("kwargs", {})

            if not plan_name:
                return web.json_response(
                    {"success": False, "error": "No plan name provided"}, status=400
                )

            result = await self.submit_plan(plan_name, kwargs)
            return web.json_response(result)

        except Exception as e:
            import traceback

            return web.json_response(
                {
                    "success": False,
                    "error": str(e),
                    "traceback": traceback.format_exc(),
                },
                status=500,
            )

    async def handle_get_history(self, request):
        """GET /api/history"""
        return web.json_response(
            {
                "success": True,
                "items": self._run_history[-10:],  # Last 10 runs
            }
        )

    async def handle_get_devices(self, request):
        """GET /api/devices"""
        return web.json_response({"success": True, "devices": list(self.devices.keys())})

    async def handle_get_plans(self, request):
        """GET /api/plans"""
        return web.json_response({"success": True, "plans": list(self.plans.keys())})

    async def handle_get_led_status(self, request):
        """GET /api/led/status - Get current LED state and available configs"""
        try:
            led = self.devices.get("led")
            if led is None:
                return web.json_response({"success": False, "error": "LED device not found"})

            # Read current state
            current_state = led.read()
            led_value = current_state.get(led.name, {}).get("value", "unknown")

            # Get available configs
            available_configs = led._available_configs

            return web.json_response(
                {
                    "success": True,
                    "current_state": led_value,
                    "available_configs": available_configs,
                    "group_name": led.group_name,
                }
            )
        except Exception as e:
            import traceback

            return web.json_response(
                {
                    "success": False,
                    "error": str(e),
                    "traceback": traceback.format_exc(),
                },
                status=500,
            )

    async def handle_set_led(self, request):
        """POST /api/led/set - Set LED state directly (bypass plan queue)"""
        try:
            data = await request.json()
            state = data.get("state", "Closed")

            led = self.devices.get("led")
            if led is None:
                return web.json_response({"success": False, "error": "LED device not found"})

            # Set LED state directly
            status = led.set(state)
            # Wait for completion
            import time

            timeout = 5.0
            start = time.time()
            while not status.done and (time.time() - start) < timeout:
                await asyncio.sleep(0.1)

            if status.done and status.success:
                return web.json_response(
                    {"success": True, "state": state, "message": f"LED set to {state}"}
                )
            else:
                return web.json_response(
                    {"success": False, "error": f"Failed to set LED to {state}"}
                )
        except Exception as e:
            import traceback

            return web.json_response(
                {
                    "success": False,
                    "error": str(e),
                    "traceback": traceback.format_exc(),
                },
                status=500,
            )

    def _room_light_device(self):
        """Resolve the room-light SwitchBot from the device registry.

        Prefers the conventional 'room_light' key (config.yml name), but
        falls back to scanning for any SwitchBot instance so a differently
        named bot still works. Returns None when no bot is configured.
        """
        bot = self.devices.get("room_light")
        if bot is not None:
            return bot
        try:
            from gently.hardware.switchbot import SwitchBot
        except Exception:
            return None
        for dev in self.devices.values():
            if isinstance(dev, SwitchBot):
                return dev
        return None

    async def _set_room_light(self, state: str) -> bool:
        """Drive the room-light SwitchBot to ``state`` ('on'/'off'/'press').

        Blocks until the BLE command lands (or times out). Returns True on
        success, False if no bot is configured or the command failed. Shared
        by the HTTP handler and internal callers (e.g. detect_embryos, which
        images under room light rather than the camera LED).
        """
        bot = self._room_light_device()
        if bot is None:
            return False
        import time

        status = bot.set(state)
        timeout = float(getattr(bot, "timeout", 20.0)) + 5
        start = time.time()
        while not status.done and (time.time() - start) < timeout:
            await asyncio.sleep(0.1)
        return bool(status.done and status.success)

    async def handle_get_room_light_status(self, request):
        """GET /api/room_light/status - cached on/off state of the room light.

        Reads the SwitchBot's last-commanded state (no BLE round-trip, so it's
        cheap to poll). 'unknown' until the first on/off command lands.
        """
        try:
            bot = self._room_light_device()
            if bot is None:
                return web.json_response(
                    {
                        "success": False,
                        "available": False,
                        "error": "room_light device not configured",
                    }
                )
            state = bot.read().get(bot.name, {}).get("value", "unknown")
            return web.json_response({"success": True, "available": True, "state": state})
        except Exception as e:
            import traceback

            return web.json_response(
                {
                    "success": False,
                    "available": False,
                    "error": str(e),
                    "traceback": traceback.format_exc(),
                },
                status=500,
            )

    async def handle_set_room_light(self, request):
        """POST /api/room_light/set - drive the room-light SwitchBot.

        Body: {"state": "on" | "off" | "press"}. Blocks until the BLE command
        lands (the bot's servo move is ~0.5-1 s plus connect latency).
        """
        try:
            data = await request.json()
            state = str(data.get("state", "")).lower()
            if state not in ("on", "off", "press"):
                return web.json_response(
                    {
                        "success": False,
                        "error": f"state {state!r} must be on, off, or press",
                    },
                    status=400,
                )
            bot = self._room_light_device()
            if bot is None:
                return web.json_response(
                    {"success": False, "error": "room_light device not configured"},
                    status=503,
                )

            status = bot.set(state)
            import time

            timeout = float(getattr(bot, "timeout", 20.0)) + 5
            start = time.time()
            while not status.done and (time.time() - start) < timeout:
                await asyncio.sleep(0.1)

            if status.done and status.success:
                new_state = bot.read().get(bot.name, {}).get("value", state)
                return web.json_response({"success": True, "state": new_state})
            return web.json_response(
                {"success": False, "error": f"failed to set room light to {state}"},
                status=502,
            )
        except Exception as e:
            import traceback

            return web.json_response(
                {
                    "success": False,
                    "error": str(e),
                    "traceback": traceback.format_exc(),
                },
                status=500,
            )

    async def handle_get_temperature_status(self, request):
        """GET /api/temperature/status - current temperature, setpoint, lock state."""
        try:
            temp = self.devices.get("temperature")
            if temp is None:
                return web.json_response(
                    {"success": False, "error": "temperature device not found"}
                )
            r = temp.read()
            return web.json_response(
                {
                    "success": True,
                    "temperature_c": r.get(temp.name, {}).get("value"),
                    "setpoint_c": r.get(f"{temp.name}_setpoint", {}).get("value"),
                    "state": r.get(f"{temp.name}_state", {}).get("value"),
                    "peltier_c": r.get(f"{temp.name}_peltier", {}).get("value"),
                }
            )
        except Exception as e:
            import traceback

            return web.json_response(
                {
                    "success": False,
                    "error": str(e),
                    "traceback": traceback.format_exc(),
                },
                status=500,
            )

    async def handle_set_temperature(self, request):
        """POST /api/temperature/set - command setpoint. Body: {target_c, wait?}.

        Non-blocking by default (controller ramps; poll status). wait=true blocks
        until SYSTEM LOCKED or the device's stabilize timeout.
        """
        try:
            data = await request.json()
            target = float(data.get("target_c"))
            wait = bool(data.get("wait", False))
            if not (0.0 <= target <= 99.9):
                return web.json_response(
                    {"success": False, "error": f"target {target} outside [0.0, 99.9]"}
                )
            temp = self.devices.get("temperature")
            if temp is None:
                return web.json_response(
                    {"success": False, "error": "temperature device not found"}
                )

            if not wait:
                temp.enable(True)
                temp.setpoint(target)
                r = temp.read()
                return web.json_response(
                    {
                        "success": True,
                        "target_c": target,
                        "waited": False,
                        "message": f"commanded {target} C (ramping)",
                        "temperature_c": r.get(temp.name, {}).get("value"),
                        "state": r.get(f"{temp.name}_state", {}).get("value"),
                    }
                )

            import time

            status = temp.set(target)
            timeout = float(getattr(temp, "stabilize_timeout", 600.0)) + 10
            start = time.time()
            while not status.done and (time.time() - start) < timeout:
                await asyncio.sleep(0.5)
            r = temp.read()
            if status.done and status.success:
                return web.json_response(
                    {
                        "success": True,
                        "target_c": target,
                        "waited": True,
                        "message": f"locked at {target} C",
                        "temperature_c": r.get(temp.name, {}).get("value"),
                        "state": r.get(f"{temp.name}_state", {}).get("value"),
                    }
                )
            return web.json_response(
                {
                    "success": False,
                    "target_c": target,
                    "error": f"did not stabilize at {target} C within {timeout:.0f}s",
                }
            )
        except Exception as e:
            import traceback

            return web.json_response(
                {
                    "success": False,
                    "error": str(e),
                    "traceback": traceback.format_exc(),
                },
                status=500,
            )

    async def handle_set_camera_led_mode(self, request):
        """POST /api/camera/led_mode - Enable/disable automatic LED for bottom camera"""
        try:
            data = await request.json()
            use_led = data.get("use_led", False)

            bottom_camera = self.devices.get("bottom_camera")
            if bottom_camera is None:
                return web.json_response(
                    {"success": False, "error": "Bottom camera device not found"}
                )

            # Set the use_led attribute
            bottom_camera.use_led = use_led

            return web.json_response(
                {
                    "success": True,
                    "use_led": use_led,
                    "message": f"Bottom camera LED mode: {'ON' if use_led else 'OFF'}",
                }
            )
        except Exception as e:
            import traceback

            return web.json_response(
                {
                    "success": False,
                    "error": str(e),
                    "traceback": traceback.format_exc(),
                },
                status=500,
            )

    async def handle_set_camera_exposure(self, request):
        """POST /api/camera/exposure - Set bottom camera exposure time"""
        try:
            data = await request.json()
            exposure_ms = data.get("exposure_ms", 50.0)

            bottom_camera = self.devices.get("bottom_camera")
            if bottom_camera is None:
                return web.json_response(
                    {"success": False, "error": "Bottom camera device not found"}
                )

            # Set exposure using the device's configure_exposure method
            bottom_camera.configure_exposure(exposure_ms)

            return web.json_response(
                {
                    "success": True,
                    "exposure_ms": exposure_ms,
                    "message": f"Bottom camera exposure set to {exposure_ms} ms",
                }
            )
        except Exception as e:
            import traceback

            return web.json_response(
                {
                    "success": False,
                    "error": str(e),
                    "traceback": traceback.format_exc(),
                },
                status=500,
            )

    async def handle_set_light_source_power(self, request):
        """POST /api/light_source/power — set per-line laser power %.

        Direct (non-queue) path mirroring the existing camera-exposure
        endpoint. Hard-limited by ``DiSPIMLightSource.POWER_LIMITS_PCT``
        — out-of-range values come back as a 400 with the error message
        from the device-layer setter.
        """
        try:
            data = await request.json()
            wavelength = int(data.get("wavelength", 488))
            pct = float(data.get("pct"))
            light_source = self.devices.get("light_source") or self.devices.get("laser_control")
            if light_source is None:
                return web.json_response(
                    {
                        "success": False,
                        "error": "Light source device not found",
                    },
                    status=503,
                )
            try:
                light_source.set_power_pct(wavelength, pct)
            except (ValueError, KeyError) as e:
                return web.json_response(
                    {
                        "success": False,
                        "error": str(e),
                    },
                    status=400,
                )
            readback = light_source.get_power_pct(wavelength)
            return web.json_response(
                {
                    "success": True,
                    "wavelength": wavelength,
                    "pct": pct,
                    "readback_pct": readback,
                }
            )
        except Exception as e:
            import traceback

            return web.json_response(
                {
                    "success": False,
                    "error": str(e),
                    "traceback": traceback.format_exc(),
                },
                status=500,
            )

    async def handle_get_light_source_power(self, request):
        """GET /api/light_source/power?wavelength=488 — read laser power %."""
        try:
            wavelength = int(request.query.get("wavelength", 488))
            light_source = self.devices.get("light_source") or self.devices.get("laser_control")
            if light_source is None:
                return web.json_response(
                    {
                        "success": False,
                        "error": "Light source device not found",
                    },
                    status=503,
                )
            try:
                pct = light_source.get_power_pct(wavelength)
            except KeyError as e:
                return web.json_response(
                    {
                        "success": False,
                        "error": str(e),
                    },
                    status=400,
                )
            return web.json_response(
                {
                    "success": True,
                    "wavelength": wavelength,
                    "pct": float(pct),
                }
            )
        except Exception as e:
            import traceback

            return web.json_response(
                {
                    "success": False,
                    "error": str(e),
                    "traceback": traceback.format_exc(),
                },
                status=500,
            )

    async def handle_set_laser_config(self, request):
        """POST /api/laser/config — apply a Laser config-group preset.

        Body: {"config": <preset_name>}
        Calls light_source.set(config_name) which maps to
        core.setConfig(group_name, config_name) + waitForConfig.
        """
        try:
            data = await request.json()
            config_name = data.get("config")
            if not config_name:
                return web.json_response(
                    {"success": False, "error": "missing 'config' field"},
                    status=400,
                )
            light_source = self.devices.get("light_source") or self.devices.get("laser_control")
            if light_source is None:
                return web.json_response(
                    {"success": False, "error": "Light source device not found"},
                    status=503,
                )
            try:
                light_source.set(config_name)
            except Exception as e:
                return web.json_response(
                    {"success": False, "error": str(e)},
                    status=400,
                )
            return web.json_response({"success": True, "config": config_name})
        except Exception as e:
            import traceback

            return web.json_response(
                {
                    "success": False,
                    "error": str(e),
                    "traceback": traceback.format_exc(),
                },
                status=500,
            )

    async def handle_get_laser_configs(self, request):
        """GET /api/laser/configs — list available Laser config-group presets."""
        try:
            light_source = self.devices.get("light_source") or self.devices.get("laser_control")
            if light_source is None:
                return web.json_response(
                    {"success": False, "error": "Light source device not found"},
                    status=503,
                )
            configs = light_source._get_available_configs()
            return web.json_response({"configs": configs})
        except Exception as e:
            import traceback

            return web.json_response(
                {
                    "success": False,
                    "error": str(e),
                    "traceback": traceback.format_exc(),
                },
                status=500,
            )

    async def handle_get_cameras(self, request):
        """GET /api/cameras — list available SPIM camera roles.

        Returns ``{"cameras": ["A"]}`` on single-camera rigs, or
        ``{"cameras": ["A", "B"]}`` when camera_b (HamCam2) is registered.
        """
        cameras = ["A"]
        if "camera_b" in self.devices:
            cameras.append("B")
        return web.json_response({"cameras": cameras})

    async def handle_get_camera_exposure(self, request):
        """GET /api/camera/exposure - Get bottom camera exposure time"""
        try:
            bottom_camera = self.devices.get("bottom_camera")
            if bottom_camera is None:
                return web.json_response(
                    {"success": False, "error": "Bottom camera device not found"}
                )

            exposure_ms = bottom_camera.exposure_time

            return web.json_response({"success": True, "exposure_ms": exposure_ms})
        except Exception as e:
            import traceback

            return web.json_response(
                {
                    "success": False,
                    "error": str(e),
                    "traceback": traceback.format_exc(),
                },
                status=500,
            )

    async def handle_get_plan_log(self, request):
        """GET /api/plan_log - Get recent plan execution log with timing"""
        try:
            limit = int(request.query.get("limit", 100))
            return web.json_response(
                {
                    "success": True,
                    "entries": self._plan_execution_log[-limit:],
                    "total_count": len(self._plan_execution_log),
                }
            )
        except Exception as e:
            import traceback

            return web.json_response(
                {
                    "success": False,
                    "error": str(e),
                    "traceback": traceback.format_exc(),
                },
                status=500,
            )

    async def handle_session_configure(self, request):
        """POST /session/configure - set staging directory for file-ref protocol.

        Body: {"volume_dir": "D:/Gently3/incoming"}
        """
        try:
            data = await request.json()
            volume_dir = data.get("volume_dir")
            if volume_dir:
                # Ensure the directory exists
                Path(volume_dir).mkdir(parents=True, exist_ok=True)
                self._volume_dir = volume_dir
                logger.info("Session configured: volume_dir = %s", volume_dir)
                return web.json_response(
                    {
                        "success": True,
                        "volume_dir": volume_dir,
                    }
                )
            else:
                # Clear staging
                self._volume_dir = None
                return web.json_response(
                    {
                        "success": True,
                        "volume_dir": None,
                        "message": "Volume staging disabled",
                    }
                )
        except Exception as e:
            import traceback

            return web.json_response(
                {
                    "success": False,
                    "error": str(e),
                    "traceback": traceback.format_exc(),
                },
                status=500,
            )

    # =========================================================================
    # HTTP API Handlers - SAM Detection
    # =========================================================================

    async def handle_sam_status(self, request):
        """GET /api/sam/status - Check SAM model availability"""
        return web.json_response(
            {
                "success": True,
                "available": True,  # SAM is always available (lazy loaded)
                "loaded": self._sam_detector is not None,
                "device": self._sam_device,
                "model_type": self._sam_model_type,
            }
        )

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
            from gently.core.coordinates import (
                DEFAULT_OBJECTIVE_MAG,
                DEFAULT_PIXEL_SIZE_UM,
            )

            pixel_size_um = data.get("pixel_size_um", DEFAULT_PIXEL_SIZE_UM)
            objective_mag = data.get("objective_mag", DEFAULT_OBJECTIVE_MAG)
            use_claude_review = data.get("use_claude_review", True)
            data.get("min_confidence", 0.7)
            exposure_ms = data.get("exposure_ms")
            brightness_percentile = data.get("brightness_percentile", 99.0)
            min_area = data.get("min_area", 5000)
            max_area = data.get("max_area", 150000)

            # Set exposure if specified
            if exposure_ms is not None:
                bottom_camera = self.devices.get("bottom_camera")
                if bottom_camera:
                    bottom_camera.configure_exposure(exposure_ms)

            # Detection always images under ROOM LIGHT, never the camera LED.
            # use_led is a persistent device flag that other flows (e.g. manual
            # marking via capture_for_marking) leave switched on, so force it
            # off here and turn the room light on so there's illumination.
            bottom_camera = self.devices.get("bottom_camera")
            if bottom_camera is not None:
                bottom_camera.use_led = False
            if await self._set_room_light("on"):
                logger.info("[detect_embryos] Room light ON, camera LED disabled")
            else:
                logger.warning(
                    "[detect_embryos] Could not turn room light on "
                    "(no SwitchBot configured?) — capturing under ambient light"
                )

            # Capture image via plan
            logger.info("[detect_embryos] Capturing bottom camera image...")
            capture_result = await self.submit_plan(
                "capture_bottom_image_plan", kwargs={"bottom_camera": "bottom_camera"}
            )

            if not capture_result.get("success"):
                return web.json_response(
                    {
                        "success": False,
                        "error": f"Image capture failed: {capture_result.get('error', 'Unknown')}",
                    },
                    status=500,
                )

            # Extract image from result
            docs = capture_result.get("documents", {})
            events = docs.get("events", [])
            image = None
            if events:
                event_data = events[0].get("data", {})
                for key in ["bottom_camera", "bottom_camera_image", "Bottom PCO"]:
                    if key in event_data:
                        val = event_data[key]
                        # Handle file ref
                        if isinstance(val, dict) and val.get("__file_ref__"):
                            import tifffile

                            image = tifffile.imread(val["path"])
                        else:
                            image = np.array(val)
                        break

            if image is None:
                return web.json_response(
                    {"success": False, "error": "No image data in capture result"},
                    status=500,
                )

            logger.info("[detect_embryos] Image shape: %s", image.shape)

            # Read stage position
            logger.info("[detect_embryos] Reading stage position...")
            stage_result = await self.submit_plan(
                "read_stage_plan", kwargs={"xy_stage": "xy_stage"}
            )

            stage_x, stage_y = 0.0, 0.0
            if stage_result.get("success"):
                stage_docs = stage_result.get("documents", {})
                stage_events = stage_docs.get("events", [])
                if stage_events:
                    stage_data = stage_events[0].get("data", {})
                    # DiSPIMXYStage.read() returns {device_name: [x, y]}
                    for key in ["xy_stage", "XYStage:XY:31", "xy_stage_position"]:
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
                max_area,
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
                "success": sam_result.get("success", False),
                "embryos": sam_result.get("embryos", []),
                "initial_detections": sam_result.get("initial_detections", 0),
                "final_detections": sam_result.get("final_detections", 0),
                "stage_position": list(stage_position),
                "verification": sam_result.get("verification", {}),
            }

            if image_path:
                response["image_path"] = image_path

            if "error" in sam_result:
                response["error"] = sam_result["error"]

            return web.json_response(response)

        except Exception as e:
            import traceback

            return web.json_response(
                {
                    "success": False,
                    "error": str(e),
                    "traceback": traceback.format_exc(),
                },
                status=500,
            )

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
        max_area: int,
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
                    max_area=max_area,
                )
            )

            # Ensure results are serializable (convert numpy types)
            embryos = result.get("embryos", [])
            for embryo in embryos:
                for key, value in list(embryo.items()):
                    if isinstance(value, np.floating):
                        embryo[key] = float(value)
                    elif isinstance(value, np.integer):
                        embryo[key] = int(value)
                    elif isinstance(value, np.ndarray):
                        # Remove mask from response (not JSON serializable)
                        del embryo[key]

            result["success"] = True
            return result

        except Exception as e:
            import traceback

            return {
                "success": False,
                "error": str(e),
                "traceback": traceback.format_exc(),
                "embryos": [],
                "initial_detections": 0,
                "final_detections": 0,
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
                    "num_slices": {
                        "type": "integer",
                        "description": "Number of Z slices",
                        "default": 50,
                    },
                    "exposure_ms": {
                        "type": "number",
                        "description": "Camera exposure per slice in ms",
                        "default": 10.0,
                    },
                    "galvo_amplitude": {
                        "type": "number",
                        "description": "Galvo scan range in volts",
                        "default": 0.5,
                    },
                    "galvo_center": {
                        "type": "number",
                        "description": "Galvo center position in volts",
                        "default": 0.0,
                    },
                    "piezo_amplitude": {
                        "type": "number",
                        "description": "Piezo Z range in µm",
                        "default": 25.0,
                    },
                    "piezo_center": {
                        "type": "number",
                        "description": "Piezo center position in µm",
                        "default": 50.0,
                    },
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
                    "piezo_position": {
                        "type": "number",
                        "description": "Z position in µm",
                    },
                    "galvo_position": {
                        "type": "number",
                        "description": "Galvo angle in volts",
                    },
                    "exposure_ms": {
                        "type": "number",
                        "description": "Camera exposure in ms",
                        "default": 10.0,
                    },
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
                    "use_led": {
                        "type": "boolean",
                        "description": "Turn on LED during capture",
                        "default": False,
                    },
                    "exposure_ms": {
                        "type": "number",
                        "description": "Camera exposure in ms",
                    },
                },
            },
            "bluesky_plan": "capture_bottom_image_plan",
            "extractor": "_extract_image",
        },
        "calibrate": {
            "description": (
                "Run piezo-galvo calibration to find optimal focus parameters for an embryo."
            ),
            "input_schema": {
                "type": "object",
                "properties": {
                    "piezo_positions": {
                        "type": "array",
                        "items": {"type": "number"},
                        "description": "Piezo positions to sweep (µm)",
                    },
                    "galvo_positions": {
                        "type": "array",
                        "items": {"type": "number"},
                        "description": "Galvo positions to sweep (volts)",
                    },
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
                    "state": {
                        "type": "string",
                        "enum": ["Open", "Closed"],
                        "description": "LED state",
                    },
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
            "description": (
                "Detect embryos/samples in the current field of view using SAM segmentation."
            ),
            "input_schema": {
                "type": "object",
                "properties": {
                    "pixel_size_um": {
                        "type": "number",
                        "description": "Camera pixel size in µm",
                        "default": 6.5,
                    },
                    "objective_mag": {
                        "type": "number",
                        "description": "Objective magnification",
                        "default": 10.0,
                    },
                    "min_confidence": {
                        "type": "number",
                        "description": "Minimum detection confidence",
                        "default": 0.7,
                    },
                    "exposure_ms": {
                        "type": "number",
                        "description": "Camera exposure in ms",
                    },
                    "brightness_percentile": {
                        "type": "number",
                        "description": "Brightness threshold percentile",
                        "default": 99.0,
                    },
                    "min_area": {
                        "type": "integer",
                        "description": "Minimum embryo area in pixels",
                        "default": 5000,
                    },
                    "max_area": {
                        "type": "integer",
                        "description": "Maximum embryo area in pixels",
                        "default": 150000,
                    },
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
        events = documents.get("events", [])
        for event in events:
            data = event.get("data", {})
            for key in candidate_keys:
                if key in data:
                    return data[key]
        return None

    def _extract_move(self, documents: dict, params: dict) -> dict:
        return {"success": True, "x": params.get("x"), "y": params.get("y")}

    def _extract_position(self, documents: dict, params: dict) -> dict:
        events = documents.get("events", [])
        if events:
            data = events[0].get("data", {})
            for key in ["XY:31", "xy_stage", "stage"]:
                if key in data:
                    val = data[key]
                    if isinstance(val, (list, tuple)) and len(val) >= 2:
                        return {"success": True, "x": float(val[0]), "y": float(val[1])}
                    if isinstance(val, dict):
                        return {
                            "success": True,
                            "x": float(val.get("x", 0)),
                            "y": float(val.get("y", 0)),
                        }
        return {"success": False, "error": "Could not read position"}

    def _extract_volume(self, documents: dict, params: dict) -> dict:
        val = self._extract_from_events(documents, ["volume_scanner", "camera", "camera_image"])
        if val is not None:
            result = {"success": True}
            if isinstance(val, dict) and val.get("__file_ref__"):
                result["volume"] = val  # file ref — client resolves
                result["shape"] = val.get("shape")
            else:
                result["volume"] = val
                if hasattr(val, "shape"):
                    result["shape"] = list(val.shape)
            return result
        return {"success": False, "error": "No volume data in result"}

    def _extract_image(self, documents: dict, params: dict) -> dict:
        val = self._extract_from_events(
            documents,
            [
                "HamCam1",
                "lightsheet_snap",
                "camera",
                "bottom_camera",
                "bottom_camera_image",
                "Bottom PCO",
            ],
        )
        if val is not None:
            result = {"success": True}
            if isinstance(val, dict) and val.get("__file_ref__"):
                result["image"] = val
                result["shape"] = val.get("shape")
            else:
                result["image"] = val
                if hasattr(val, "shape"):
                    result["shape"] = list(val.shape)
            return result
        return {"success": False, "error": "No image data in result"}

    def _extract_calibration(self, documents: dict, params: dict) -> dict:
        # Calibration results come back in the plan result
        return {"success": True, "calibration": {}}

    def _extract_success(self, documents: dict, params: dict) -> dict:
        return {"success": True}

    async def handle_microscope_info(self, request):
        """GET /api/microscope — handshake: plans as Anthropic tool schemas."""
        from . import HARDWARE_DISPLAY_NAME, HARDWARE_NAME
        from .description import HARDWARE_DESCRIPTION

        # Build plan list, filtering to actually-available plans
        available_plans = []
        for plan_name, schema in self.PLAN_SCHEMAS.items():
            bluesky_name = schema.get("bluesky_plan")
            if bluesky_name is None or bluesky_name in self.plans:
                # Return client-facing fields (Anthropic tool format)
                available_plans.append(
                    {
                        "name": plan_name,
                        "description": schema["description"],
                        "input_schema": schema["input_schema"],
                    }
                )

        return web.json_response(
            {
                "name": HARDWARE_NAME,
                "display_name": HARDWARE_DISPLAY_NAME,
                "description": HARDWARE_DESCRIPTION,
                "plans": available_plans,
            }
        )

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
                return web.json_response(
                    {"success": False, "error": f"Plan '{plan_name}' not implemented"},
                    status=500,
                )

            if bluesky_name not in self.plans:
                return web.json_response(
                    {
                        "success": False,
                        "error": f"Hardware plan '{bluesky_name}' not loaded",
                    },
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
                {
                    "success": False,
                    "error": str(e),
                    "traceback": traceback.format_exc(),
                },
                status=500,
            )

    # =========================================================================
    # Device State HTTP Handlers
    # =========================================================================

    async def handle_devices_state(self, request):
        """GET /api/devices/state - One-shot snapshot of device positions + properties.

        Cheap: returns the most recent poller snapshot without forcing a fresh
        MMCore read. If the caller needs guaranteed-fresh data they can pass
        ?refresh=1 and we'll do an on-demand read on the worker thread.
        """
        try:
            refresh = request.query.get("refresh", "0") in ("1", "true", "yes")
            if refresh and self._state_pause_counter == 0 and not self._state_updating_now:
                self._state_updating_now = True
                try:
                    xy = await asyncio.to_thread(self._read_xy_position)
                    slow = await asyncio.to_thread(self._read_slow_positions)
                    properties = await asyncio.to_thread(self._read_full_state)
                    self._state_latest = {
                        "positions": {**xy, **slow},
                        "properties": properties,
                        "t": time.time(),
                        "paused": False,
                    }
                finally:
                    self._state_updating_now = False
            return web.json_response(self._state_latest)
        except Exception as exc:
            import traceback

            return web.json_response(
                {"error": str(exc), "traceback": traceback.format_exc()},
                status=500,
            )

    async def handle_callbacks_stream(self, request):
        """GET /api/devices/callbacks/stream - SSE of raw MMCore push events.

        Diagnostic-grade stream. Each event the MMCore adapter emits via
        OnPropertyChanged/OnXYStagePositionChanged/etc. becomes one SSE frame.
        Used to discover which events the adapters actually fire (joystick
        moves, automatic refocus, anything we don't trigger from the host).
        """
        response = web.StreamResponse(
            status=200,
            reason="OK",
            headers={
                "Content-Type": "text/event-stream",
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            },
        )
        await response.prepare(request)

        queue: asyncio.Queue = asyncio.Queue(maxsize=256)
        self._callback_subscribers.append(queue)
        peer = request.remote
        logger.info(
            "Callback subscriber connected from %s (total=%d)",
            peer,
            len(self._callback_subscribers),
        )

        try:
            await response.write(b"event: ready\ndata: {}\n\n")
            while True:
                try:
                    payload = await asyncio.wait_for(queue.get(), timeout=10.0)
                except asyncio.TimeoutError:
                    await response.write(b": keepalive\n\n")
                    continue
                if payload is None:
                    break  # shutdown sentinel
                await response.write(f"data: {json.dumps(payload)}\n\n".encode())
        except (asyncio.CancelledError, ConnectionResetError, ConnectionAbortedError):
            pass
        except Exception:
            logger.exception("Callback SSE writer failed")
        finally:
            try:
                self._callback_subscribers.remove(queue)
            except ValueError:
                pass
            logger.info(
                "Callback subscriber disconnected from %s (total=%d)",
                peer,
                len(self._callback_subscribers),
            )

        return response

    async def handle_devices_stream(self, request):
        """GET /api/devices/stream - Server-Sent Events stream of device state.

        On connect, sends the current snapshot as `event: snapshot`. Subsequent
        ticks come as default `data:` events. Heartbeats arrive every ~2 s even
        when the poller is paused, so clients can distinguish "no change" from
        "stream is dead".
        """
        response = web.StreamResponse(
            status=200,
            reason="OK",
            headers={
                "Content-Type": "text/event-stream",
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            },
        )
        await response.prepare(request)

        # Bounded queue — slow clients get dropped rather than back-pressuring the poller.
        queue: asyncio.Queue = asyncio.Queue(maxsize=32)
        self._state_subscribers.append(queue)
        peer = request.remote
        logger.info(
            "Device-state subscriber connected from %s (total=%d)",
            peer,
            len(self._state_subscribers),
        )

        try:
            # Send the most recent snapshot immediately so the UI doesn't
            # have to wait for the next tick.
            snapshot = json.dumps(self._state_latest)
            await response.write(f"event: snapshot\ndata: {snapshot}\n\n".encode())

            while True:
                try:
                    payload = await asyncio.wait_for(queue.get(), timeout=10.0)
                except asyncio.TimeoutError:
                    # Comment heartbeat — keeps proxies/sockets warm without
                    # delivering a parseable event to the client.
                    await response.write(b": keepalive\n\n")
                    continue
                # Shutdown sentinel: on_stop() pushes None into every queue so
                # all SSE handlers exit promptly instead of waiting on aiohttp's
                # shutdown timeout.
                if payload is None:
                    break
                await response.write(f"data: {json.dumps(payload)}\n\n".encode())
        except (asyncio.CancelledError, ConnectionResetError, ConnectionAbortedError):
            pass
        except Exception:
            logger.exception("Device-state SSE writer failed")
        finally:
            try:
                self._state_subscribers.remove(queue)
            except ValueError:
                pass
            logger.info(
                "Device-state subscriber disconnected from %s (total=%d)",
                peer,
                len(self._state_subscribers),
            )

        return response

    async def handle_bottom_camera_stream(self, request):
        """GET /api/bottom_camera/stream — SSE of base64-JPEG frames.

        The streamer task spins up on first connect and exits when the last
        subscriber leaves, so MMCore is left alone whenever nobody's watching.
        """
        response = web.StreamResponse(
            status=200,
            reason="OK",
            headers={
                "Content-Type": "text/event-stream",
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            },
        )
        await response.prepare(request)

        queue: asyncio.Queue = asyncio.Queue(maxsize=4)
        self._cam_subscribers.append(queue)
        # If we just brought the subscriber count to 1, start the streamer.
        if len(self._cam_subscribers) == 1 and (self._cam_task is None or self._cam_task.done()):
            self._cam_task = asyncio.create_task(
                self._bottom_camera_streamer(), name="bottom-camera-streamer"
            )
        peer = request.remote
        logger.info(
            "Bottom-camera subscriber connected from %s (total=%d)",
            peer,
            len(self._cam_subscribers),
        )

        try:
            # Initial comment so the client knows the connection is alive
            # even before the first frame arrives (camera may take ~exposure ms).
            await response.write(b": connected\n\n")
            while True:
                try:
                    payload = await asyncio.wait_for(queue.get(), timeout=10.0)
                except asyncio.TimeoutError:
                    await response.write(b": keepalive\n\n")
                    continue
                if payload is None:
                    break  # shutdown sentinel
                await response.write(f"data: {json.dumps(payload)}\n\n".encode())
        except (asyncio.CancelledError, ConnectionResetError, ConnectionAbortedError):
            pass
        except Exception:
            logger.exception("Bottom-camera SSE writer failed")
        finally:
            try:
                self._cam_subscribers.remove(queue)
            except ValueError:
                pass
            logger.info(
                "Bottom-camera subscriber disconnected from %s (total=%d)",
                peer,
                len(self._cam_subscribers),
            )

        return response

    # =========================================================================
    # Server Lifecycle
    # =========================================================================

    async def on_start(self):
        """Initialize hardware and start the HTTP server."""
        await self.initialize()

        # Create web app
        self._app = web.Application()

        # Core endpoints (carried forward from simple_server.py)
        self._app.router.add_get("/api/status", self.handle_status)
        self._app.router.add_post("/api/queue/item/add", self.handle_submit_plan)
        self._app.router.add_get("/api/history", self.handle_get_history)
        self._app.router.add_get("/api/devices", self.handle_get_devices)
        self._app.router.add_get("/api/plans", self.handle_get_plans)
        self._app.router.add_get("/api/led/status", self.handle_get_led_status)
        self._app.router.add_post("/api/led/set", self.handle_set_led)
        self._app.router.add_get("/api/temperature/status", self.handle_get_temperature_status)
        self._app.router.add_post("/api/temperature/set", self.handle_set_temperature)
        self._app.router.add_get("/api/room_light/status", self.handle_get_room_light_status)
        self._app.router.add_post("/api/room_light/set", self.handle_set_room_light)
        self._app.router.add_post("/api/camera/led_mode", self.handle_set_camera_led_mode)
        self._app.router.add_post("/api/camera/exposure", self.handle_set_camera_exposure)
        self._app.router.add_get("/api/camera/exposure", self.handle_get_camera_exposure)
        self._app.router.add_post("/api/light_source/power", self.handle_set_light_source_power)
        self._app.router.add_get("/api/light_source/power", self.handle_get_light_source_power)
        self._app.router.add_post("/api/laser/config", self.handle_set_laser_config)
        self._app.router.add_get("/api/laser/configs", self.handle_get_laser_configs)
        self._app.router.add_get("/api/plan_log", self.handle_get_plan_log)
        self._app.router.add_post("/session/configure", self.handle_session_configure)

        # SAM endpoints (new - replaces RPyC sam_server.py)
        self._app.router.add_get("/api/sam/status", self.handle_sam_status)
        self._app.router.add_post("/api/detect_embryos", self.handle_detect_embryos)

        # Microscope API (generic plan-based interface)
        self._app.router.add_get("/api/microscope", self.handle_microscope_info)
        self._app.router.add_post("/api/microscope/execute", self.handle_microscope_execute)

        # Device state streaming (positions + properties)
        self._app.router.add_get("/api/devices/state", self.handle_devices_state)
        self._app.router.add_get("/api/devices/stream", self.handle_devices_stream)
        self._app.router.add_get("/api/devices/callbacks/stream", self.handle_callbacks_stream)

        # Bottom-camera live stream (subscriber-gated, off when nobody listens)
        self._app.router.add_get("/api/bottom_camera/stream", self.handle_bottom_camera_stream)

        # Lightsheet (SPIM) live stream — continuous sequence acquisition
        self._app.router.add_get("/api/lightsheet/stream", self.handle_lightsheet_stream)
        self._app.router.add_post("/api/lightsheet/live/params", self.handle_lightsheet_params)
        self._app.router.add_get("/api/cameras", self.handle_get_cameras)

        # Start plan executor
        self._executor_task = asyncio.create_task(self._plan_executor())

        # Start device-state pollers. Three independent loops so slow MMCore
        # calls (property cache, piezo/galvo serial) can never stall the XY
        # fast path:
        #   - XY position (fast)
        #   - piezo + galvo (slow, 1 Hz)
        #   - full property cache (slowest, ~0.07 Hz)
        self._state_pos_task = asyncio.create_task(self._position_poller())
        self._state_slow_pos_task = asyncio.create_task(self._slow_positions_poller())
        self._state_prop_task = asyncio.create_task(self._property_poller())

        # Start web server
        # shutdown_timeout limits how long cleanup() waits for active responses
        # to finish. The default is 60s — way too long for an SSE handler that
        # got the wake-up sentinel and didn't exit promptly.
        self._runner = web.AppRunner(self._app, shutdown_timeout=2.0)
        await self._runner.setup()
        site = web.TCPSite(self._runner, self.host, self.port)

        logger.info("=" * 60)
        logger.info("HTTP API available at http://%s:%d", self.host, self.port)
        logger.info("=" * 60)
        logger.info(
            "Endpoints: GET /api/status, GET /api/devices, GET /api/plans,"
            " POST /api/queue/item/add, ..."
        )

        await site.start()
        self._print_ready_panel()

    def _categorize_devices(self):
        """Group device names into human-readable buckets for the console panel.

        First-match-wins so 'room_light' lands in Accessory (not Light) and
        'volume_scanner' in Motion. Accessory entries carry live state.
        """
        buckets = {
            "Motion": [],
            "Imaging": [],
            "Light": [],
            "Accessory": [],
            "Other": [],
        }
        for name in sorted(self.devices):
            low = name.lower()
            if low in ("room_light", "temperature"):
                label = name
                try:
                    dev = self.devices[name]
                    val = dev.read().get(dev.name, {}).get("value")
                    if val is not None:
                        label = f"{name} ({val})"
                except Exception:
                    pass
                buckets["Accessory"].append(label)
            elif "cam" in low or "snap" in low:
                buckets["Imaging"].append(name)
            elif any(k in low for k in ("stage", "piezo", "galvo", "scanner")):
                buckets["Motion"].append(name)
            elif any(k in low for k in ("laser", "led", "light", "illum")):
                buckets["Light"].append(name)
            else:
                buckets["Other"].append(name)
        return list(buckets.items())

    def _print_ready_panel(self):
        """Curated, always-visible status summary at the terminal.

        Separate from the file log: the operator (often a biologist) gets the
        URL the agent connects to, a grouped device inventory and accessory
        states at a glance — instead of a silent console after the banner.
        """

        def _fmt(names, limit=6):
            if len(names) <= limit:
                return " · ".join(names)
            return " · ".join(names[:limit]) + cui.c(f"  +{len(names) - limit} more", "grey")

        host = self.host or "0.0.0.0"
        url_host = "localhost" if host in ("0.0.0.0", "::", "") else host

        cui.out()
        cui.header(f"GENTLY{cui.MIDDOT}DEVICE LAYER", badge="READY", badge_style="green")
        cui.row("URL", cui.c(f"http://{url_host}:{self.port}", "bold"))
        cui.row("Hardware", str((self.config or {}).get("hardware", "dispim")))
        cui.row("Devices", f"{len(self.devices)} loaded")
        for label, names in self._categorize_devices():
            if names:
                cui.sub(label, _fmt(names))
        cui.row("Detection", f"SAM on {self._sam_device} (loads on first use)")
        cui.row("Plans", f"{len(self.plans)} available")
        cui.rule(heavy=False)
        cui.note("Waiting for the agent to connect.  Press Ctrl+C to stop.")
        cui.rule(heavy=True)
        cui.out()

    async def on_stop(self):
        """Shut down the HTTP server and plan executor."""
        logger.info("Shutting down...")
        cui.out()
        cui.note("Shutting down device layer...", "yellow")
        self._running = False

        # Cancel any pending coalesced-broadcast timer.
        if self._callback_broadcast_handle is not None:
            self._callback_broadcast_handle.cancel()
            self._callback_broadcast_handle = None

        # 1. Wake up every SSE subscriber so they exit immediately instead of
        #    sitting in `wait_for(queue.get(), timeout=10)` and forcing aiohttp
        #    to wait out its shutdown timeout. Done first so handlers drain
        #    before we cancel pollers (which would block on in-flight to_thread).
        for queues in (
            self._state_subscribers,
            self._callback_subscribers,
            self._cam_subscribers,
        ):
            for q in list(queues):
                try:
                    q.put_nowait(None)
                except asyncio.QueueFull:
                    # Replace the head with the sentinel so the handler sees it.
                    try:
                        q.get_nowait()
                        q.put_nowait(None)
                    except Exception:
                        pass

        # 2. Cancel pollers. The in-flight to_thread call cannot be aborted;
        #    cancellation takes effect when the thread returns (~position read
        #    <0.3s, property read <3s, bottom-camera snapImage = exposure +
        #    transfer). If MMCore is hung or a long exposure is set, we'd
        #    otherwise wait forever — so each task gets a 3 s ceiling. A
        #    timed-out thread leaks until interpreter shutdown reaps it.
        for task_attr in (
            "_state_pos_task",
            "_state_slow_pos_task",
            "_state_prop_task",
            "_cam_task",
        ):
            task = getattr(self, task_attr, None)
            if task is not None:
                task.cancel()
                try:
                    await asyncio.wait_for(task, timeout=3.0)
                except (asyncio.CancelledError, asyncio.TimeoutError):
                    if not task.done():
                        logger.warning(
                            "%s did not exit within shutdown timeout; "
                            "leaking thread, continuing shutdown",
                            task_attr,
                        )
                setattr(self, task_attr, None)
        if self._executor_task:
            self._executor_task.cancel()
            try:
                await self._executor_task
            except asyncio.CancelledError:
                pass
        if self._runner:
            await self._runner.cleanup()
        logger.info("Device layer stopped.")
        cui.note("Device layer stopped.", "grey")

    async def health_check(self) -> dict:
        """Return health status with device count, queue size, SAM status."""
        base = await super().health_check()
        base["device_count"] = len(self.devices)
        base["queue_size"] = self._plan_queue.qsize()
        base["sam_loaded"] = self._sam_detector is not None
        return base

    async def run(self, host: str | None = None, port: int | None = None):
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
    parser.add_argument(
        "--sam-device",
        default="cuda",
        choices=["cuda", "cpu"],
        help="Device for SAM model (default: cuda)",
    )

    args = parser.parse_args()

    from datetime import datetime
    from pathlib import Path

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
