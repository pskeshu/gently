#!/usr/bin/env python3
"""
Volume Scanning FPS Benchmark
==============================

Benchmarks volume acquisition throughput by comparing raw MMCore vs ophyd
DiSPIMVolumeScanner approaches, sweeping across num_slices and exposure_ms
combinations.

Usage:
    python diagnostics/benchmark_volume_fps.py --help
    python diagnostics/benchmark_volume_fps.py --slices 25 --exposures 5.0 --repeats 3 --warmup 1
    python diagnostics/benchmark_volume_fps.py --raw-only --slices 25 50 --exposures 5.0 10.0
    python diagnostics/benchmark_volume_fps.py --save --slices 50 --exposures 5.0 --repeats 5
"""

import os
import statistics
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import numpy as np
import pymmcore
import yaml

# Add dispim-control to path for ophyd device imports
DISPIM_CONTROL_DIR = (
    Path(__file__).resolve().parent.parent / "UsersdispimDocumentsGitHubdispim-control"
)
if str(DISPIM_CONTROL_DIR) not in sys.path:
    sys.path.insert(0, str(DISPIM_CONTROL_DIR))


# ---------------------------------------------------------------------------
# Constants -- device names match MMConfig_tracking_screening.cfg
# ---------------------------------------------------------------------------
CAMERA_NAME = "HamCam1"
GALVO_DEVICE = "Scanner:AB:33"
PIEZO_DEVICE = "PiezoStage:P:34"

# Default scan parameters (same as run_multi_embryo_volumes.py)
DEFAULT_GALVO_AMPLITUDE = 0.5  # degrees
DEFAULT_GALVO_CENTER = 0.0  # degrees
DEFAULT_PIEZO_AMPLITUDE = 25.0  # um
DEFAULT_PIEZO_CENTER = 50.0  # um
DEFAULT_LASER_CONFIG = "488 and 561"
DEFAULT_CAMERA_ROI = (128, 896, 2048, 512)  # (x, y, width, height)

# SPIM timing defaults
DEFAULT_TIMING = {
    "scan_delay_ms": 6.75,
    "scan_duration_ms": 5.5,
    "laser_delay_ms": 8.0,
    "laser_duration_ms": 5.0,
    "camera_delay_ms": 8.0,
    "camera_duration_ms": 1.0,
}

# ---------------------------------------------------------------------------
# Test series parameters -- edit these to change what the benchmark runs
# ---------------------------------------------------------------------------
SLICES_SERIES = [25, 50, 100, 200]
EXPOSURE_MS = 5.0
NUM_REPEATS = 10
NUM_WARMUP = 2

# Simulated embryo calibration profiles for round-robin reconfig test.
# Each entry represents a different embryo with distinct galvo/piezo settings.
EMBRYO_PROFILES = [
    {
        "galvo_amplitude": 0.50,
        "galvo_center": 0.00,
        "piezo_amplitude": 25.0,
        "piezo_center": 50.0,
    },
    {
        "galvo_amplitude": 0.45,
        "galvo_center": 0.12,
        "piezo_amplitude": 22.5,
        "piezo_center": 55.0,
    },
    {
        "galvo_amplitude": 0.55,
        "galvo_center": -0.08,
        "piezo_amplitude": 27.5,
        "piezo_center": 45.0,
    },
    {
        "galvo_amplitude": 0.48,
        "galvo_center": 0.05,
        "piezo_amplitude": 24.0,
        "piezo_center": 52.0,
    },
]


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------
@dataclass
class BenchmarkResult:
    approach: str
    num_slices: int
    exposure_ms: float
    timings: list[float] = field(default_factory=list)
    image_counts: list[int] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)

    @property
    def mean(self) -> float:
        return statistics.mean(self.timings) if self.timings else float("nan")

    @property
    def std(self) -> float:
        return statistics.stdev(self.timings) if len(self.timings) > 1 else 0.0

    @property
    def min_t(self) -> float:
        return min(self.timings) if self.timings else float("nan")

    @property
    def max_t(self) -> float:
        return max(self.timings) if self.timings else float("nan")

    @property
    def vol_per_sec(self) -> float:
        return 1.0 / self.mean if self.mean > 0 else 0.0

    @property
    def total_images(self) -> int:
        return sum(self.image_counts)


# ---------------------------------------------------------------------------
# Config loading
# ---------------------------------------------------------------------------
def load_config(path: str) -> tuple[str, str]:
    """Read config.yml and return (mm_dir, config_file)."""
    cfg_path = Path(path)
    if not cfg_path.exists():
        raise FileNotFoundError(f"Config file not found: {cfg_path}")

    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)

    mm_dir = cfg["mmdirectory"]

    # Look for the .cfg next to config.yml first, then in the MM directory
    config_file = str(cfg_path.parent / cfg["mmconfig"])
    if not os.path.exists(config_file):
        config_file = os.path.join(mm_dir, cfg["mmconfig"])

    return mm_dir, config_file


def initialize_mmcore(mm_dir: str, config_file: str) -> pymmcore.CMMCore:
    """Create and configure CMMCore (no RPyC)."""
    core = pymmcore.CMMCore()
    core.enableStderrLog(True)

    os.environ["PATH"] += os.pathsep.join(["", mm_dir])
    core.setDeviceAdapterSearchPaths([mm_dir])

    if not os.path.exists(config_file):
        raise FileNotFoundError(f"MM configuration file not found: {config_file}")

    print(f"Loading MM config: {config_file}")
    core.loadSystemConfiguration(config_file)
    print(f"MMCore initialized ({core.getVersionInfo()})")
    return core


# ---------------------------------------------------------------------------
# Raw MMCore approach
# ---------------------------------------------------------------------------
def configure_hardware_raw(core: pymmcore.CMMCore, num_slices: int, exposure_ms: float):
    """
    Configure all hardware for raw acquisition.

    Mirrors run_multi_embryo_volumes.py:configure_hardware_for_volume() but
    uses fixed scan parameters instead of per-embryo calibration data.
    """
    # Stop any existing sequence
    if core.isSequenceRunning():
        core.stopSequenceAcquisition()
        time.sleep(0.5)
    core.clearCircularBuffer()

    # Reset SPIM state
    try:
        core.setProperty(GALVO_DEVICE, "SPIMState", "Idle")
        time.sleep(0.2)
    except Exception:
        pass

    # System startup
    core.setConfig("System", "Startup")
    core.waitForConfig("System", "Startup")

    # Lasers on (stay on across burst)
    core.setConfig("Laser", DEFAULT_LASER_CONFIG)
    core.waitForConfig("Laser", DEFAULT_LASER_CONFIG)

    # Camera: hardware trigger
    core.setCameraDevice(CAMERA_NAME)
    roi_x, roi_y, roi_w, roi_h = DEFAULT_CAMERA_ROI
    core.setROI(CAMERA_NAME, roi_x, roi_y, roi_w, roi_h)
    core.setProperty(CAMERA_NAME, "TRIGGER SOURCE", "EXTERNAL")
    core.setProperty(CAMERA_NAME, "SENSOR MODE", "PROGRESSIVE")
    core.setProperty(CAMERA_NAME, "TRIGGER ACTIVE", "EDGE")
    core.setExposure(CAMERA_NAME, exposure_ms)

    # Scanner (galvo) setup
    core.setProperty(GALVO_DEVICE, "SPIMState", "Idle")
    time.sleep(0.2)
    core.setProperty(GALVO_DEVICE, "LaserOutputMode", "shutter + side")
    core.setProperty(GALVO_DEVICE, "BeamEnabled", "No")

    # X-axis (light sheet width)
    core.setProperty(GALVO_DEVICE, "SingleAxisXAmplitude(deg)", 8.0)
    core.setProperty(GALVO_DEVICE, "SingleAxisXOffset(deg)", 0.0005)
    core.setProperty(GALVO_DEVICE, "SingleAxisXPattern", "1 - Triangle")

    # Y-axis (synchronized with piezo)
    core.setProperty(GALVO_DEVICE, "SingleAxisYAmplitude(deg)", float(DEFAULT_GALVO_AMPLITUDE))
    core.setProperty(GALVO_DEVICE, "SingleAxisYOffset(deg)", float(DEFAULT_GALVO_CENTER))
    core.setProperty(GALVO_DEVICE, "SingleAxisYPattern", "1 - Triangle")

    # SPIM timing
    core.setProperty(GALVO_DEVICE, "SPIMDelayBeforeScan(ms)", DEFAULT_TIMING["scan_delay_ms"])
    core.setProperty(GALVO_DEVICE, "SPIMNumScansPerSlice", 1)
    core.setProperty(GALVO_DEVICE, "SPIMScanDuration(ms)", DEFAULT_TIMING["scan_duration_ms"])
    core.setProperty(GALVO_DEVICE, "SPIMDelayBeforeLaser(ms)", DEFAULT_TIMING["laser_delay_ms"])
    core.setProperty(GALVO_DEVICE, "SPIMLaserDuration(ms)", DEFAULT_TIMING["laser_duration_ms"])
    core.setProperty(GALVO_DEVICE, "SPIMDelayBeforeCamera(ms)", DEFAULT_TIMING["camera_delay_ms"])
    core.setProperty(GALVO_DEVICE, "SPIMCameraDuration(ms)", DEFAULT_TIMING["camera_duration_ms"])

    core.setProperty(GALVO_DEVICE, "SPIMNumSlices", num_slices)
    core.setProperty(GALVO_DEVICE, "SPIMNumSlicesPerPiezo", 1)
    core.setProperty(GALVO_DEVICE, "SPIMNumSides", 1)
    core.setProperty(GALVO_DEVICE, "SPIMFirstSide", "A")

    # Piezo
    core.setFocusDevice(PIEZO_DEVICE)
    core.setProperty(PIEZO_DEVICE, "SingleAxisAmplitude(um)", float(DEFAULT_PIEZO_AMPLITUDE))
    core.setProperty(PIEZO_DEVICE, "SingleAxisOffset(um)", float(DEFAULT_PIEZO_CENTER))
    core.setProperty(PIEZO_DEVICE, "SingleAxisPattern", "1 - Triangle")
    core.setProperty(PIEZO_DEVICE, "SPIMNumSlices", num_slices)
    core.setProperty(PIEZO_DEVICE, "SPIMState", "Armed")

    time.sleep(0.3)


def acquire_volume_raw(
    core: pymmcore.CMMCore,
    num_slices: int,
    save_dir: Path | None = None,
) -> tuple[int, float]:
    """
    Trigger SPIM and collect images from the circular buffer.

    Returns (num_images_collected, elapsed_seconds).
    """
    core.clearCircularBuffer()

    # Ensure buffer capacity
    buf_cap = core.getBufferTotalCapacity()
    if buf_cap < num_slices:
        core.setCircularBufferMemoryFootprint(512)
        time.sleep(0.1)

    core.prepareSequenceAcquisition(CAMERA_NAME)
    time.sleep(0.1)
    core.startSequenceAcquisition(CAMERA_NAME, num_slices, 0, True)
    time.sleep(0.1)

    # -- timed section --
    t0 = time.perf_counter()
    core.setProperty(GALVO_DEVICE, "SPIMState", "Running")

    images: list | None = [] if save_dir else None
    count = 0
    timeout_s = max(num_slices * 0.05 * 2, 10.0)  # generous timeout
    t_start = time.time()

    while core.getRemainingImageCount() > 0 or core.isSequenceRunning():
        if core.getRemainingImageCount() > 0:
            img = core.popNextImage()
            count += 1
            if images is not None:
                images.append(img)
        if time.time() - t_start > timeout_s:
            break
        time.sleep(0.001)

    elapsed = time.perf_counter() - t0
    # -- end timed section --

    if core.isSequenceRunning():
        core.stopSequenceAcquisition()

    # Reset SPIM and re-arm piezo for next volume
    core.setProperty(GALVO_DEVICE, "SPIMState", "Idle")
    time.sleep(0.2)
    core.setProperty(PIEZO_DEVICE, "SPIMState", "Armed")
    time.sleep(0.3)

    # Optionally save
    if images is not None and save_dir is not None:
        _save_volume(np.array(images), save_dir, "raw", num_slices)

    return count, elapsed


def cleanup_raw(core: pymmcore.CMMCore):
    """Stop sequence, set SPIM idle, turn lasers off."""
    try:
        if core.isSequenceRunning():
            core.stopSequenceAcquisition()
    except Exception:
        pass
    try:
        core.setProperty(GALVO_DEVICE, "SPIMState", "Idle")
    except Exception:
        pass
    try:
        core.setProperty(PIEZO_DEVICE, "SPIMState", "Idle")
    except Exception:
        pass
    try:
        core.setConfig("Laser", "ALL OFF")
        core.waitForConfig("Laser", "ALL OFF")
    except Exception:
        pass
    try:
        core.setProperty(CAMERA_NAME, "TRIGGER SOURCE", "INTERNAL")
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Ophyd approach
# ---------------------------------------------------------------------------
def create_ophyd_devices(core: pymmcore.CMMCore):
    """
    Instantiate ophyd device objects directly (bypasses device_factory.py
    which imports a theme module that may not be available).
    """
    from dispim_control.devices import (
        DiSPIMCamera,
        DiSPIMLaserControl,
        DiSPIMPiezo,
        DiSPIMScanner,
        DiSPIMVolumeScanner,
    )

    scanner = DiSPIMScanner(name=GALVO_DEVICE, core=core)
    camera = DiSPIMCamera(device_name=CAMERA_NAME, core=core)
    piezo = DiSPIMPiezo(name=PIEZO_DEVICE, core=core)
    laser_control = DiSPIMLaserControl(core=core, name="laser_control", group_name="Laser")

    volume_scanner = DiSPIMVolumeScanner(
        scanner=scanner,
        camera=camera,
        piezo=piezo,
        laser_control=laser_control,
        core=core,
        name="volume_scanner",
    )
    return volume_scanner


def acquire_volume_ophyd(
    volume_scanner,
    num_slices: int,
    exposure_ms: float,
    save_dir: Path | None = None,
) -> tuple[int, float]:
    """
    Configure + trigger via ophyd VolumeScanner.

    Returns (num_images_collected, elapsed_seconds).
    Includes laser on/off overhead (ophyd manages lasers per volume).
    """
    volume_scanner.configure(
        num_slices=num_slices,
        exposure_ms=exposure_ms,
        galvo_amplitude=DEFAULT_GALVO_AMPLITUDE,
        galvo_center=DEFAULT_GALVO_CENTER,
        piezo_amplitude=DEFAULT_PIEZO_AMPLITUDE,
        piezo_center=DEFAULT_PIEZO_CENTER,
        laser_config=DEFAULT_LASER_CONFIG,
    )

    t0 = time.perf_counter()
    status = volume_scanner.trigger()
    status.wait(timeout=120)
    elapsed = time.perf_counter() - t0

    # Count images from the stored volume
    vol = volume_scanner._last_volume
    count = vol.shape[0] if vol is not None else 0

    if save_dir is not None and vol is not None:
        _save_volume(vol, save_dir, "ophyd", num_slices)

    return count, elapsed


# ---------------------------------------------------------------------------
# Ophyd burst approach -- configure once, skip per-volume reset
# ---------------------------------------------------------------------------
def configure_ophyd_burst(
    volume_scanner, num_slices: int, exposure_ms: float, core: pymmcore.CMMCore
):
    """
    Configure ophyd devices once for a burst of volumes.

    Uses the ophyd device configure methods but then turns lasers on and
    leaves everything armed -- no per-volume teardown/setup.
    """
    # Use the ophyd configure path (camera, scanner, piezo)
    volume_scanner.configure(
        num_slices=num_slices,
        exposure_ms=exposure_ms,
        galvo_amplitude=DEFAULT_GALVO_AMPLITUDE,
        galvo_center=DEFAULT_GALVO_CENTER,
        piezo_amplitude=DEFAULT_PIEZO_AMPLITUDE,
        piezo_center=DEFAULT_PIEZO_CENTER,
        laser_config=DEFAULT_LASER_CONFIG,
    )

    # Turn lasers on (they stay on for the whole burst)
    core.setConfig("Laser", DEFAULT_LASER_CONFIG)
    core.waitForConfig("Laser", DEFAULT_LASER_CONFIG)
    time.sleep(0.1)


def acquire_volume_ophyd_burst(
    volume_scanner, num_slices: int, core: pymmcore.CMMCore
) -> tuple[int, float]:
    """
    Acquire one volume using ophyd devices but without per-volume reset.

    Skips the laser on/off and camera/scanner reconfiguration that
    DiSPIMVolumeScanner.trigger() does every volume. Only re-arms the piezo.
    """
    camera_name = volume_scanner.camera.name
    scanner = volume_scanner.scanner
    piezo = volume_scanner.piezo

    core.clearCircularBuffer()

    buf_cap = core.getBufferTotalCapacity()
    if buf_cap < num_slices:
        core.setCircularBufferMemoryFootprint(512)
        time.sleep(0.1)

    core.prepareSequenceAcquisition(camera_name)
    time.sleep(0.1)
    core.startSequenceAcquisition(camera_name, num_slices, 0, True)
    time.sleep(0.1)

    # -- timed section --
    t0 = time.perf_counter()
    scanner.set_spim_state("Running")

    count = 0
    timeout_s = max(num_slices * 0.05 * 2, 10.0)
    t_start = time.time()

    while core.getRemainingImageCount() > 0 or core.isSequenceRunning():
        if core.getRemainingImageCount() > 0:
            core.popNextImage()  # discard data
            count += 1
        if time.time() - t_start > timeout_s:
            break
        time.sleep(0.001)

    elapsed = time.perf_counter() - t0
    # -- end timed section --

    if core.isSequenceRunning():
        core.stopSequenceAcquisition()

    # Minimal inter-volume reset (same as raw): SPIM idle + re-arm piezo
    scanner.set_spim_state("Idle")
    piezo.set_spim_state("Armed")

    return count, elapsed


def cleanup_ophyd_burst(volume_scanner, core: pymmcore.CMMCore):
    """Clean up after ophyd burst: reset camera trigger, lasers off."""
    try:
        volume_scanner.camera.set_trigger_mode("INTERNAL")
    except Exception:
        pass
    try:
        volume_scanner.scanner.set_spim_state("Idle")
    except Exception:
        pass
    try:
        volume_scanner.piezo.set_spim_state("Idle")
    except Exception:
        pass
    try:
        core.setConfig("Laser", "ALL OFF")
        core.waitForConfig("Laser", "ALL OFF")
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Burst reconfig approach -- reconfigure galvo/piezo per volume (round-robin)
# ---------------------------------------------------------------------------
def configure_burst_reconfig(
    volume_scanner, num_slices: int, exposure_ms: float, core: pymmcore.CMMCore
):
    """
    One-time setup for burst_reconfig: camera, scanner X-axis & timing, lasers.

    Uses the first embryo profile for initial config, then each acquisition
    will update only the galvo Y / piezo parameters.
    """
    profile = EMBRYO_PROFILES[0]
    volume_scanner.configure(
        num_slices=num_slices,
        exposure_ms=exposure_ms,
        galvo_amplitude=profile["galvo_amplitude"],
        galvo_center=profile["galvo_center"],
        piezo_amplitude=profile["piezo_amplitude"],
        piezo_center=profile["piezo_center"],
        laser_config=DEFAULT_LASER_CONFIG,
    )

    core.setConfig("Laser", DEFAULT_LASER_CONFIG)
    core.waitForConfig("Laser", DEFAULT_LASER_CONFIG)
    time.sleep(0.1)


def acquire_volume_burst_reconfig(
    volume_scanner,
    num_slices: int,
    profile_idx: int,
    core: pymmcore.CMMCore,
) -> tuple[int, float]:
    """
    Acquire one volume after reconfiguring galvo/piezo for a specific embryo.

    Simulates round-robin multi-embryo acquisition: between volumes we only
    update the galvo Y amplitude/center and piezo amplitude/center (the
    per-embryo calibration values). Camera, laser, scanner X-axis, and SPIM
    timing stay unchanged.

    The timer includes the galvo/piezo reconfiguration.
    """
    profile = EMBRYO_PROFILES[profile_idx % len(EMBRYO_PROFILES)]
    camera_name = volume_scanner.camera.name
    scanner = volume_scanner.scanner

    core.clearCircularBuffer()

    buf_cap = core.getBufferTotalCapacity()
    if buf_cap < num_slices:
        core.setCircularBufferMemoryFootprint(512)
        time.sleep(0.1)

    # -- timed section (includes galvo/piezo reconfig) --
    t0 = time.perf_counter()

    # Reconfigure galvo Y-axis for this embryo
    core.setProperty(GALVO_DEVICE, "SingleAxisYAmplitude(deg)", float(profile["galvo_amplitude"]))
    core.setProperty(GALVO_DEVICE, "SingleAxisYOffset(deg)", float(profile["galvo_center"]))

    # Reconfigure piezo for this embryo
    core.setProperty(PIEZO_DEVICE, "SingleAxisAmplitude(um)", float(profile["piezo_amplitude"]))
    core.setProperty(PIEZO_DEVICE, "SingleAxisOffset(um)", float(profile["piezo_center"]))
    core.setProperty(PIEZO_DEVICE, "SPIMState", "Armed")
    time.sleep(0.3)

    core.prepareSequenceAcquisition(camera_name)
    time.sleep(0.1)
    core.startSequenceAcquisition(camera_name, num_slices, 0, True)
    time.sleep(0.1)

    scanner.set_spim_state("Running")

    count = 0
    timeout_s = max(num_slices * 0.05 * 2, 10.0)
    t_start = time.time()

    while core.getRemainingImageCount() > 0 or core.isSequenceRunning():
        if core.getRemainingImageCount() > 0:
            core.popNextImage()
            count += 1
        if time.time() - t_start > timeout_s:
            break
        time.sleep(0.001)

    elapsed = time.perf_counter() - t0
    # -- end timed section --

    if core.isSequenceRunning():
        core.stopSequenceAcquisition()

    scanner.set_spim_state("Idle")

    return count, elapsed


# ---------------------------------------------------------------------------
# Burst reconfig with waitForDevice -- replaces time.sleep() with MMCore API
# ---------------------------------------------------------------------------
def acquire_volume_burst_reconfig_wfd(
    volume_scanner,
    num_slices: int,
    profile_idx: int,
    core: pymmcore.CMMCore,
) -> tuple[int, float]:
    """
    Same as burst_reconfig but uses core.waitForDevice() instead of
    time.sleep() to wait for hardware readiness.

    This tests whether the ASI Tiger controller properly reports device
    busy status, allowing us to replace conservative fixed sleeps with
    responsive device-ready polling.
    """
    profile = EMBRYO_PROFILES[profile_idx % len(EMBRYO_PROFILES)]
    camera_name = volume_scanner.camera.name

    core.clearCircularBuffer()

    buf_cap = core.getBufferTotalCapacity()
    if buf_cap < num_slices:
        core.setCircularBufferMemoryFootprint(512)
        core.waitForDevice(camera_name)

    # -- timed section (includes galvo/piezo reconfig) --
    t0 = time.perf_counter()

    # Reconfigure galvo Y-axis for this embryo
    core.setProperty(GALVO_DEVICE, "SingleAxisYAmplitude(deg)", float(profile["galvo_amplitude"]))
    core.setProperty(GALVO_DEVICE, "SingleAxisYOffset(deg)", float(profile["galvo_center"]))
    core.waitForDevice(GALVO_DEVICE)

    # Reconfigure piezo for this embryo
    core.setProperty(PIEZO_DEVICE, "SingleAxisAmplitude(um)", float(profile["piezo_amplitude"]))
    core.setProperty(PIEZO_DEVICE, "SingleAxisOffset(um)", float(profile["piezo_center"]))
    core.setProperty(PIEZO_DEVICE, "SPIMState", "Armed")
    core.waitForDevice(PIEZO_DEVICE)

    core.prepareSequenceAcquisition(camera_name)
    core.waitForDevice(camera_name)
    core.startSequenceAcquisition(camera_name, num_slices, 0, True)
    core.waitForDevice(camera_name)

    # Trigger SPIM -- no sleep needed, the state machine starts immediately
    core.setProperty(GALVO_DEVICE, "SPIMState", "Running")

    count = 0
    timeout_s = max(num_slices * 0.05 * 2, 10.0)
    t_start = time.time()

    while core.getRemainingImageCount() > 0 or core.isSequenceRunning():
        if core.getRemainingImageCount() > 0:
            core.popNextImage()
            count += 1
        if time.time() - t_start > timeout_s:
            break
        time.sleep(0.001)

    elapsed = time.perf_counter() - t0
    # -- end timed section --

    if core.isSequenceRunning():
        core.stopSequenceAcquisition()

    # Reset scanner -- waitForDevice instead of sleep(0.2)
    core.setProperty(GALVO_DEVICE, "SPIMState", "Idle")
    core.waitForDevice(GALVO_DEVICE)

    return count, elapsed


# ---------------------------------------------------------------------------
# Saving helper
# ---------------------------------------------------------------------------
def _save_volume(volume: np.ndarray, save_dir: Path, approach: str, num_slices: int):
    """Save a volume as a TIFF file."""
    import tifffile

    save_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    fname = save_dir / f"{approach}_s{num_slices}_{ts}.tif"
    tifffile.imwrite(str(fname), volume)


# ---------------------------------------------------------------------------
# Benchmark sweep
# ---------------------------------------------------------------------------
def run_benchmark_sweep(
    core: pymmcore.CMMCore,
    slices_list: list[int],
    exposures_list: list[float],
    num_repeats: int,
    num_warmup: int,
    run_raw: bool,
    run_ophyd: bool,
    save_dir: Path | None = None,
) -> list[BenchmarkResult]:
    """Run the full parameter sweep and return results."""
    results: list[BenchmarkResult] = []

    volume_scanner = None
    if run_ophyd:
        print("\nCreating ophyd devices...")
        volume_scanner = create_ophyd_devices(core)
        print("Ophyd devices ready.")

    total_configs = len(slices_list) * len(exposures_list)
    config_idx = 0

    for num_slices in slices_list:
        for exposure_ms in exposures_list:
            config_idx += 1
            print(f"\n{'=' * 60}")
            print(
                f"Config {config_idx}/{total_configs}: "
                f"slices={num_slices}, exposure={exposure_ms}ms"
            )
            print(f"{'=' * 60}")

            # --- Raw MMCore ---
            if run_raw:
                res_raw = BenchmarkResult("raw", num_slices, exposure_ms)
                print("\n[Raw MMCore] Configuring hardware...")
                try:
                    configure_hardware_raw(core, num_slices, exposure_ms)
                except Exception as e:
                    print(f"  ERROR configuring raw: {e}")
                    res_raw.errors.append(f"configure: {e}")
                    results.append(res_raw)
                    continue

                # Warm-up
                for w in range(num_warmup):
                    print(f"  Warm-up {w + 1}/{num_warmup}...", end=" ", flush=True)
                    try:
                        cnt, dur = acquire_volume_raw(core, num_slices)
                        print(f"{cnt} imgs, {dur:.3f}s")
                    except Exception as e:
                        print(f"ERROR: {e}")

                # Timed repeats
                for r in range(num_repeats):
                    print(f"  Repeat {r + 1}/{num_repeats}...", end=" ", flush=True)
                    try:
                        cnt, dur = acquire_volume_raw(core, num_slices, save_dir=save_dir)
                        res_raw.timings.append(dur)
                        res_raw.image_counts.append(cnt)
                        print(f"{cnt} imgs, {dur:.3f}s ({1.0 / dur:.1f} vol/s)")
                    except Exception as e:
                        print(f"ERROR: {e}")
                        res_raw.errors.append(f"repeat {r + 1}: {e}")

                # Cleanup after raw batch (lasers off)
                cleanup_raw(core)
                results.append(res_raw)
                _print_single_result(res_raw)

            # --- Ophyd ---
            if run_ophyd and volume_scanner is not None:
                res_ophyd = BenchmarkResult("ophyd", num_slices, exposure_ms)

                # Warm-up
                for w in range(num_warmup):
                    print(
                        f"  [Ophyd] Warm-up {w + 1}/{num_warmup}...",
                        end=" ",
                        flush=True,
                    )
                    try:
                        cnt, dur = acquire_volume_ophyd(volume_scanner, num_slices, exposure_ms)
                        print(f"{cnt} imgs, {dur:.3f}s")
                    except Exception as e:
                        print(f"ERROR: {e}")

                # Timed repeats
                for r in range(num_repeats):
                    print(
                        f"  [Ophyd] Repeat {r + 1}/{num_repeats}...",
                        end=" ",
                        flush=True,
                    )
                    try:
                        cnt, dur = acquire_volume_ophyd(
                            volume_scanner, num_slices, exposure_ms, save_dir=save_dir
                        )
                        res_ophyd.timings.append(dur)
                        res_ophyd.image_counts.append(cnt)
                        print(f"{cnt} imgs, {dur:.3f}s ({1.0 / dur:.1f} vol/s)")
                    except Exception as e:
                        print(f"ERROR: {e}")
                        res_ophyd.errors.append(f"repeat {r + 1}: {e}")

                results.append(res_ophyd)
                _print_single_result(res_ophyd)

            # --- Ophyd burst (configure once, no per-volume reset) ---
            if run_ophyd and volume_scanner is not None:
                res_burst = BenchmarkResult("ophyd_burst", num_slices, exposure_ms)
                print("\n[Ophyd Burst] Configuring once...")
                try:
                    configure_ophyd_burst(volume_scanner, num_slices, exposure_ms, core)
                except Exception as e:
                    print(f"  ERROR configuring ophyd burst: {e}")
                    res_burst.errors.append(f"configure: {e}")
                    results.append(res_burst)
                    continue

                # Warm-up
                for w in range(num_warmup):
                    print(f"  Warm-up {w + 1}/{num_warmup}...", end=" ", flush=True)
                    try:
                        cnt, dur = acquire_volume_ophyd_burst(volume_scanner, num_slices, core)
                        print(f"{cnt} imgs, {dur:.3f}s")
                    except Exception as e:
                        print(f"ERROR: {e}")

                # Timed repeats
                for r in range(num_repeats):
                    print(f"  Repeat {r + 1}/{num_repeats}...", end=" ", flush=True)
                    try:
                        cnt, dur = acquire_volume_ophyd_burst(volume_scanner, num_slices, core)
                        res_burst.timings.append(dur)
                        res_burst.image_counts.append(cnt)
                        print(f"{cnt} imgs, {dur:.3f}s ({1.0 / dur:.1f} vol/s)")
                    except Exception as e:
                        print(f"ERROR: {e}")
                        res_burst.errors.append(f"repeat {r + 1}: {e}")

                cleanup_ophyd_burst(volume_scanner, core)
                results.append(res_burst)
                _print_single_result(res_burst)

            # --- Burst reconfig (round-robin galvo/piezo per volume) ---
            if run_ophyd and volume_scanner is not None:
                res_reconfig = BenchmarkResult("burst_reconfig", num_slices, exposure_ms)
                print(
                    f"\n[Burst Reconfig] Configuring once, "
                    f"cycling {len(EMBRYO_PROFILES)} embryo profiles..."
                )
                try:
                    configure_burst_reconfig(volume_scanner, num_slices, exposure_ms, core)
                except Exception as e:
                    print(f"  ERROR configuring burst reconfig: {e}")
                    res_reconfig.errors.append(f"configure: {e}")
                    results.append(res_reconfig)
                    continue

                # Warm-up (cycle through profiles)
                for w in range(num_warmup):
                    print(
                        f"  Warm-up {w + 1}/{num_warmup} (profile {w % len(EMBRYO_PROFILES)})...",
                        end=" ",
                        flush=True,
                    )
                    try:
                        cnt, dur = acquire_volume_burst_reconfig(
                            volume_scanner, num_slices, w, core
                        )
                        print(f"{cnt} imgs, {dur:.3f}s")
                    except Exception as e:
                        print(f"ERROR: {e}")

                # Timed repeats (cycle through profiles)
                for r in range(num_repeats):
                    pidx = r % len(EMBRYO_PROFILES)
                    print(
                        f"  Repeat {r + 1}/{num_repeats} (profile {pidx})...",
                        end=" ",
                        flush=True,
                    )
                    try:
                        cnt, dur = acquire_volume_burst_reconfig(
                            volume_scanner, num_slices, r, core
                        )
                        res_reconfig.timings.append(dur)
                        res_reconfig.image_counts.append(cnt)
                        print(f"{cnt} imgs, {dur:.3f}s ({1.0 / dur:.1f} vol/s)")
                    except Exception as e:
                        print(f"ERROR: {e}")
                        res_reconfig.errors.append(f"repeat {r + 1}: {e}")

                cleanup_ophyd_burst(volume_scanner, core)
                results.append(res_reconfig)
                _print_single_result(res_reconfig)

            # --- Burst reconfig with waitForDevice (no time.sleep) ---
            if run_ophyd and volume_scanner is not None:
                res_wfd = BenchmarkResult("reconfig_wfd", num_slices, exposure_ms)
                print(
                    "\n[Reconfig WFD] Configuring once, "
                    "using waitForDevice() instead of time.sleep()..."
                )
                try:
                    configure_burst_reconfig(volume_scanner, num_slices, exposure_ms, core)
                except Exception as e:
                    print(f"  ERROR configuring reconfig_wfd: {e}")
                    res_wfd.errors.append(f"configure: {e}")
                    results.append(res_wfd)
                    continue

                # Warm-up (cycle through profiles)
                for w in range(num_warmup):
                    print(
                        f"  Warm-up {w + 1}/{num_warmup} (profile {w % len(EMBRYO_PROFILES)})...",
                        end=" ",
                        flush=True,
                    )
                    try:
                        cnt, dur = acquire_volume_burst_reconfig_wfd(
                            volume_scanner, num_slices, w, core
                        )
                        print(f"{cnt} imgs, {dur:.3f}s")
                    except Exception as e:
                        print(f"ERROR: {e}")

                # Timed repeats (cycle through profiles)
                for r in range(num_repeats):
                    pidx = r % len(EMBRYO_PROFILES)
                    print(
                        f"  Repeat {r + 1}/{num_repeats} (profile {pidx})...",
                        end=" ",
                        flush=True,
                    )
                    try:
                        cnt, dur = acquire_volume_burst_reconfig_wfd(
                            volume_scanner, num_slices, r, core
                        )
                        res_wfd.timings.append(dur)
                        res_wfd.image_counts.append(cnt)
                        print(f"{cnt} imgs, {dur:.3f}s ({1.0 / dur:.1f} vol/s)")
                    except Exception as e:
                        print(f"ERROR: {e}")
                        res_wfd.errors.append(f"repeat {r + 1}: {e}")

                cleanup_ophyd_burst(volume_scanner, core)
                results.append(res_wfd)
                _print_single_result(res_wfd)

    return results


# ---------------------------------------------------------------------------
# Output formatting
# ---------------------------------------------------------------------------
def _print_single_result(res: BenchmarkResult):
    """Print a single intermediate result."""
    if not res.timings:
        print(f"  -> {res.approach}: NO SUCCESSFUL RUNS ({len(res.errors)} errors)")
        return
    print(
        f"  -> {res.approach}: {res.vol_per_sec:.2f} vol/s, "
        f"mean={res.mean:.3f}s, std={res.std:.3f}s"
    )


def print_results_table(results: list[BenchmarkResult]):
    """Print formatted ASCII results table."""
    if not results:
        print("No results to display.")
        return

    header = (
        f"{'Slices':>6} | {'Exp(ms)':>7} | {'Approach':>14} | "
        f"{'Vol/s':>7} | {'Mean(s)':>7} | {'Std(s)':>7} | "
        f"{'Min(s)':>7} | {'Max(s)':>7} | {'Images':>6}"
    )
    sep = "-" * len(header)

    print(f"\n{sep}")
    print("RESULTS")
    print(sep)
    print(header)
    print(sep)

    for r in results:
        if r.timings:
            print(
                f"{r.num_slices:>6} | {r.exposure_ms:>7.1f} | {r.approach:>14} | "
                f"{r.vol_per_sec:>7.2f} | {r.mean:>7.3f} | {r.std:>7.3f} | "
                f"{r.min_t:>7.3f} | {r.max_t:>7.3f} | {r.total_images:>6}"
            )
        else:
            print(
                f"{r.num_slices:>6} | {r.exposure_ms:>7.1f} | {r.approach:>14} | "
                f"{'FAIL':>7} | {'---':>7} | {'---':>7} | "
                f"{'---':>7} | {'---':>7} | {'---':>6}"
            )

    print(sep)


def print_summary(results: list[BenchmarkResult]):
    """Print overhead analysis comparing ophyd and ophyd_burst vs raw."""
    from collections import defaultdict

    groups: dict = defaultdict(dict)
    for r in results:
        groups[(r.num_slices, r.exposure_ms)][r.approach] = r

    # Need at least raw + one ophyd variant
    has_data = [
        (k, v)
        for k, v in groups.items()
        if "raw" in v
        and ("ophyd" in v or "ophyd_burst" in v or "burst_reconfig" in v or "reconfig_wfd" in v)
    ]
    if not has_data:
        return

    print()
    header = (
        f"{'Slices':>6} | {'Exp(ms)':>7} | {'Approach':>14} | {'vs Raw(ms)':>10} | {'vs Raw(%)':>9}"
    )
    sep = "-" * len(header)

    print(sep)
    print("OVERHEAD ANALYSIS (vs raw MMCore)")
    print(sep)
    print(header)
    print(sep)

    for (ns, exp), approaches in sorted(has_data):
        raw_mean = approaches["raw"].mean
        if raw_mean <= 0:
            continue
        for label in ["ophyd", "ophyd_burst", "burst_reconfig", "reconfig_wfd"]:
            if label not in approaches or not approaches[label].timings:
                continue
            other_mean = approaches[label].mean
            overhead_ms = (other_mean - raw_mean) * 1000.0
            overhead_pct = ((other_mean - raw_mean) / raw_mean) * 100.0
            print(
                f"{ns:>6} | {exp:>7.1f} | {label:>14} | "
                f"{overhead_ms:>+10.1f} | {overhead_pct:>+8.1f}%"
            )

    print(sep)


def save_results_csv(results: list[BenchmarkResult], path: Path, run_params: dict):
    """Write benchmark results to a CSV file with full metadata."""
    import csv
    import json

    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)

        # --- Metadata header (comment rows) ---
        writer.writerow(["# Volume Scanning FPS Benchmark"])
        writer.writerow(["# datetime", datetime.now().isoformat()])
        writer.writerow(["# slices_series", json.dumps(run_params["slices"])])
        writer.writerow(["# exposure_ms", run_params["exposure_ms"]])
        writer.writerow(["# repeats", run_params["repeats"]])
        writer.writerow(["# warmup", run_params["warmup"]])
        writer.writerow(["# approaches", run_params["approaches"]])
        writer.writerow(["# embryo_profiles", run_params.get("embryo_profiles", "")])
        writer.writerow(["# config", run_params.get("config", "")])
        writer.writerow(["# galvo_amplitude_deg", DEFAULT_GALVO_AMPLITUDE])
        writer.writerow(["# galvo_center_deg", DEFAULT_GALVO_CENTER])
        writer.writerow(["# piezo_amplitude_um", DEFAULT_PIEZO_AMPLITUDE])
        writer.writerow(["# piezo_center_um", DEFAULT_PIEZO_CENTER])
        writer.writerow(["# laser_config", DEFAULT_LASER_CONFIG])
        writer.writerow(["# camera_roi", json.dumps(list(DEFAULT_CAMERA_ROI))])
        writer.writerow(["# spim_timing", json.dumps(DEFAULT_TIMING)])
        writer.writerow([])

        # --- Summary table ---
        writer.writerow(
            [
                "slices",
                "exposure_ms",
                "approach",
                "vol_per_sec",
                "mean_s",
                "std_s",
                "min_s",
                "max_s",
                "total_images",
                "num_repeats",
                "errors",
            ]
        )
        for r in results:
            writer.writerow(
                [
                    r.num_slices,
                    r.exposure_ms,
                    r.approach,
                    f"{r.vol_per_sec:.4f}" if r.timings else "",
                    f"{r.mean:.6f}" if r.timings else "",
                    f"{r.std:.6f}" if r.timings else "",
                    f"{r.min_t:.6f}" if r.timings else "",
                    f"{r.max_t:.6f}" if r.timings else "",
                    r.total_images,
                    len(r.timings),
                    "; ".join(r.errors) if r.errors else "",
                ]
            )

        # --- Per-volume raw timings ---
        writer.writerow([])
        writer.writerow(["# Per-volume timings (seconds)"])
        writer.writerow(
            [
                "slices",
                "exposure_ms",
                "approach",
                "repeat",
                "elapsed_s",
                "image_count",
            ]
        )
        for r in results:
            for i, (t, cnt) in enumerate(zip(r.timings, r.image_counts, strict=False)):
                writer.writerow(
                    [
                        r.num_slices,
                        r.exposure_ms,
                        r.approach,
                        i + 1,
                        f"{t:.6f}",
                        cnt,
                    ]
                )

    print(f"\nResults saved to: {path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    config_path = str(DISPIM_CONTROL_DIR / "config.yml")

    print("Volume Scanning FPS Benchmark")
    print(f"  Slices series: {SLICES_SERIES}")
    print(f"  Exposure:      {EXPOSURE_MS} ms")
    print(f"  Repeats:       {NUM_REPEATS} (+ {NUM_WARMUP} warmup)")
    print("  Approaches:    raw / ophyd / ophyd_burst / burst_reconfig / reconfig_wfd")
    print(
        f"  Embryo profiles: {len(EMBRYO_PROFILES)} (for burst_reconfig & reconfig_wfd round-robin)"
    )

    # Load config and initialize
    mm_dir, config_file = load_config(config_path)
    core = initialize_mmcore(mm_dir, config_file)

    try:
        results = run_benchmark_sweep(
            core=core,
            slices_list=SLICES_SERIES,
            exposures_list=[EXPOSURE_MS],
            num_repeats=NUM_REPEATS,
            num_warmup=NUM_WARMUP,
            run_raw=True,
            run_ophyd=True,
            save_dir=None,
        )

        print_results_table(results)
        print_summary(results)

        # Save CSV
        csv_path = (
            Path("results") / f"benchmark_volume_fps_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        )
        run_params = {
            "slices": SLICES_SERIES,
            "exposure_ms": EXPOSURE_MS,
            "repeats": NUM_REPEATS,
            "warmup": NUM_WARMUP,
            "approaches": "raw / ophyd / ophyd_burst / burst_reconfig / reconfig_wfd",
            "embryo_profiles": len(EMBRYO_PROFILES),
            "config": config_path,
        }
        save_results_csv(results, csv_path, run_params)

    finally:
        # Global cleanup -- always turn off lasers and idle SPIM
        print("\nCleaning up hardware...")
        cleanup_raw(core)
        print("Done. Lasers OFF, SPIM Idle.")


if __name__ == "__main__":
    main()
