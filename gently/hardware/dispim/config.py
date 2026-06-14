"""
Configuration Module for diSPIM Microscope
==========================================

This module provides configuration classes, hardware profiles, and utility functions
for diSPIM microscope control. It centralizes parameters that were previously scattered
across device classes and standalone scripts.

Architecture:
    - Hardware profiles define microscope-specific timing characteristics
    - Configuration dataclasses define parameter sets for different modes
    - Utility functions calculate derived parameters (e.g., SPIM timing)
    - Calibration dataclasses provide file I/O for persistent state

Usage:
    >>> from gently.config import HardwareProfile, CameraConfig, CameraMode
    >>> profile = HardwareProfile()  # Default Flash4 camera
    >>> camera_config = CameraConfig(mode=CameraMode.EXTERNAL_PROGRESSIVE, exposure_ms=10.0)
"""

import json
import math
from dataclasses import asdict, dataclass
from enum import Enum
from pathlib import Path

# ============================================================================
# ENUMERATIONS
# ============================================================================


class CameraMode(Enum):
    """
    Camera trigger and sensor mode configurations.

    Different acquisition workflows require different camera configurations:
    - INTERNAL_AREA: For calibration, live view, manual acquisitions
    - EXTERNAL_PROGRESSIVE: For hardware-triggered SPIM volumes
    """

    INTERNAL_AREA = "internal_area"  # Manual trigger, full sensor
    EXTERNAL_PROGRESSIVE = "external_progressive"  # Hardware trigger, progressive readout


class ScannerPattern(Enum):
    """Galvo scanner waveform patterns."""

    TRIANGLE = "1 - Triangle"
    SAWTOOTH = "2 - Sawtooth"
    RAMP = "3 - Ramp"


class ScannerMode(Enum):
    """Galvo scanner axis control modes."""

    DISABLED = "1 - Disabled"
    INTERNAL = "2 - Internal"
    ENABLED_SYNCED = "3 - Enabled with axes synced"


# ============================================================================
# HARDWARE PROFILES
# ============================================================================


@dataclass
class HardwareProfile:
    """
    Microscope hardware timing characteristics.

    These parameters describe the physical timing constraints of the microscope
    hardware. They are used to calculate SPIM timing sequences.

    Attributes:
        camera_reset_ms: Camera reset time (sensor-specific)
        camera_readout_ms: Camera readout time (depends on ROI size)
        scan_laser_buffer_ms: Safety buffer around laser pulse
        scan_filter_freq_khz: Scanner filter frequency for smoothing
        has_plogic: Whether PLogic card is present (affects timing)

    Examples:
        Default Flash4 camera profile:
        >>> profile = HardwareProfile()

        Custom profile for faster camera:
        >>> profile = HardwareProfile(camera_reset_ms=2.0, camera_readout_ms=8.0)
    """

    camera_reset_ms: float = 3.0  # Hamamatsu Flash4 reset time
    camera_readout_ms: float = 10.0  # Typical for 2048x512 ROI
    scan_laser_buffer_ms: float = 0.25  # Buffer before/after laser pulse
    scan_filter_freq_khz: float = 0.2  # Scanner filter frequency
    has_plogic: bool = False  # PLogic card present


# ============================================================================
# CAMERA CONFIGURATION
# ============================================================================


@dataclass
class CameraConfig:
    """
    Camera acquisition configuration.

    Defines camera settings for different acquisition modes. The mode determines
    trigger source and sensor readout pattern.

    Attributes:
        mode: Trigger and sensor mode (INTERNAL_AREA or EXTERNAL_PROGRESSIVE)
        exposure_ms: Exposure time in milliseconds
        roi: Region of interest as (x, y, width, height)

    Examples:
        Calibration mode (manual triggering):
        >>> config = CameraConfig(
        ...     mode=CameraMode.INTERNAL_AREA,
        ...     exposure_ms=50.0
        ... )

        Hardware SPIM mode:
        >>> config = CameraConfig(
        ...     mode=CameraMode.EXTERNAL_PROGRESSIVE,
        ...     exposure_ms=10.0
        ... )
    """

    mode: CameraMode
    exposure_ms: float
    roi: tuple[int, int, int, int] = (128, 896, 2048, 512)  # Default diSPIM ROI

    def to_mm_properties(self) -> dict:
        """
        Convert to Micro-Manager property dict.

        Returns:
            Dictionary of property name -> value for CMMCore.setProperty()

        Example:
            >>> config = CameraConfig(CameraMode.EXTERNAL_PROGRESSIVE, 10.0)
            >>> props = config.to_mm_properties()
            >>> for name, value in props.items():
            ...     core.setProperty(camera_device, name, value)
        """
        if self.mode == CameraMode.INTERNAL_AREA:
            return {"TRIGGER SOURCE": "INTERNAL", "SENSOR MODE": "AREA"}
        elif self.mode == CameraMode.EXTERNAL_PROGRESSIVE:
            return {
                "TRIGGER SOURCE": "EXTERNAL",
                "SENSOR MODE": "PROGRESSIVE",
                "TRIGGER ACTIVE": "EDGE",
            }
        else:
            raise ValueError(f"Unknown camera mode: {self.mode}")


# ============================================================================
# SCANNER/GALVO CONFIGURATION
# ============================================================================


@dataclass
class GalvoAxisConfig:
    """
    Configuration for a single galvo mirror axis (X or Y).

    The galvo scanner has two axes:
    - X axis: Generates light sheet width (continuous scanning)
    - Y axis: Steps through sample depth (slice-by-slice)

    Attributes:
        amplitude_deg: Scan amplitude in degrees
        offset_deg: Center position offset in degrees
        pattern: Waveform pattern (Triangle, Sawtooth, etc.)
        mode: Control mode (Disabled, Internal, Enabled with sync)

    Examples:
        X-axis for light sheet generation (8° scanning):
        >>> x_config = GalvoAxisConfig(
        ...     amplitude_deg=8.0,
        ...     offset_deg=0.0005,
        ...     pattern=ScannerPattern.TRIANGLE,
        ...     mode=ScannerMode.ENABLED_SYNCED
        ... )

        Y-axis for slice stepping (amplitude set by SPIM):
        >>> y_config = GalvoAxisConfig(
        ...     amplitude_deg=0.0001,  # Minimal, controlled by SPIM
        ...     offset_deg=0.0,
        ...     pattern=ScannerPattern.TRIANGLE,
        ...     mode=ScannerMode.ENABLED_SYNCED
        ... )
    """

    amplitude_deg: float
    offset_deg: float
    pattern: ScannerPattern = ScannerPattern.TRIANGLE
    mode: ScannerMode = ScannerMode.ENABLED_SYNCED

    def to_mm_properties(self, axis: str) -> dict:
        """
        Convert to Micro-Manager property dict for specified axis.

        Args:
            axis: "X" or "Y"

        Returns:
            Dictionary of property name -> value

        Example:
            >>> x_config = GalvoAxisConfig(amplitude_deg=8.0, offset_deg=0.0)
            >>> props = x_config.to_mm_properties("X")
            >>> # props = {"SingleAxisXAmplitude(deg)": 8.0, ...}
        """
        return {
            f"SingleAxis{axis}Amplitude(deg)": float(self.amplitude_deg),
            f"SingleAxis{axis}Offset(deg)": float(self.offset_deg),
            f"SingleAxis{axis}Pattern": self.pattern.value,
            f"SingleAxis{axis}Mode": self.mode.value,
        }


@dataclass
class ScannerConfig:
    """
    Complete scanner/galvo configuration for both axes.

    Attributes:
        x_axis: X-axis configuration (light sheet width)
        y_axis: Y-axis configuration (slice stepping)
        beam_enabled: Whether galvo beam is enabled

    Examples:
        Configuration for hardware-triggered SPIM:
        >>> config = ScannerConfig(
        ...     x_axis=GalvoAxisConfig(amplitude_deg=8.0, offset_deg=0.0005),
        ...     y_axis=GalvoAxisConfig(amplitude_deg=0.0001, offset_deg=0.0),
        ...     beam_enabled=True
        ... )
    """

    x_axis: GalvoAxisConfig
    y_axis: GalvoAxisConfig
    beam_enabled: bool = True

    def to_mm_properties(self) -> dict:
        """
        Convert to Micro-Manager property dict for both axes.

        Returns:
            Dictionary of all property name -> value pairs

        Example:
            >>> config = ScannerConfig(x_axis=..., y_axis=...)
            >>> props = config.to_mm_properties()
            >>> for name, value in props.items():
            ...     core.setProperty(scanner_device, name, value)
        """
        props = {}
        props.update(self.x_axis.to_mm_properties("X"))
        props.update(self.y_axis.to_mm_properties("Y"))
        props["BeamEnabled"] = "Yes" if self.beam_enabled else "No"
        return props


# ============================================================================
# SPIM TIMING CALCULATION
# ============================================================================


def calculate_spim_timing(
    camera_exposure_ms: float, hardware_profile: HardwareProfile | None = None
) -> dict:
    """
    Calculate SPIM hardware timing parameters for synchronized acquisition.

    This function computes the timing sequence for the Tiger controller SPIM state
    machine. It coordinates camera exposure, laser pulsing, and scanner movement to
    ensure proper synchronization.

    The timing sequence for each slice is:
    1. Scanner delay (filter settling)
    2. Scanner moves to position
    3. Laser delay
    4. Laser pulse (during camera exposure)
    5. Camera exposure completes
    6. Camera readout + reset

    Args:
        camera_exposure_ms: Camera exposure time
        hardware_profile: Hardware timing characteristics (uses default if None)

    Returns:
        Dictionary with timing parameters:
            - scanDelay: Scanner filter settling time (ms)
            - scanPeriod: Total scanner cycle time (ms)
            - laserDelay: Delay before laser pulse (ms)
            - laserDuration: Laser pulse duration (ms)
            - cameraDelay: Delay before camera trigger (ms)
            - cameraDuration: Camera trigger duration (ms, must be > 0!)
            - cameraExposure: Camera exposure time (ms)
            - sliceDuration: Total time per slice (ms)
            - frameRate: Acquisition rate (fps)

    Notes:
        - All timing values are rounded to 0.25ms (Tiger controller resolution)
        - cameraDuration must be > 0 or SPIM state machine fails!
        - Scanner filter settling depends on filter frequency and PLogic presence

    Examples:
        >>> timing = calculate_spim_timing(camera_exposure_ms=10.0)
        >>> print(f"Frame rate: {timing['frameRate']:.1f} fps")
        >>> print(f"Slice duration: {timing['sliceDuration']:.2f} ms")

        >>> # Use custom hardware profile
        >>> profile = HardwareProfile(camera_readout_ms=8.0)
        >>> timing = calculate_spim_timing(10.0, profile)
    """
    if hardware_profile is None:
        hardware_profile = HardwareProfile()

    # Tiger controller timing resolution (0.25 ms steps)
    def round_quarter_ms(val: float) -> float:
        return round(val * 4) / 4.0

    def ceil_quarter_ms(val: float) -> float:
        return math.ceil(val * 4) / 4.0

    # Camera timing constraints
    camera_readout_max = ceil_quarter_ms(hardware_profile.camera_readout_ms)
    camera_reset_max = ceil_quarter_ms(hardware_profile.camera_reset_ms)
    global_exposure_delay_max = camera_readout_max + camera_reset_max

    # Laser timing (duration matches exposure)
    laser_duration = round_quarter_ms(camera_exposure_ms)

    # Scanner timing (laser + buffers)
    scan_duration = laser_duration + 2 * hardware_profile.scan_laser_buffer_ms

    # Scanner filter settling delay
    scan_delay_filter = 0.39 / hardware_profile.scan_filter_freq_khz
    if hardware_profile.has_plogic:
        scan_delay_filter -= 0.25  # PLogic compensation

    scan_delay = ceil_quarter_ms(scan_delay_filter)
    scan_period = round_quarter_ms(scan_duration)

    # Laser and camera delays
    laser_delay = hardware_profile.scan_laser_buffer_ms
    camera_delay = laser_delay

    # CRITICAL: cameraDuration must be > 0 or SPIM fails!
    # Set to 1.0 ms (minimum safe value)
    camera_duration = 1.0

    # Camera exposure (actual exposure time)
    camera_exposure = laser_duration

    # Total slice timing
    slice_duration = scan_delay + scan_period + global_exposure_delay_max
    frame_rate = 1000.0 / slice_duration if slice_duration > 0 else 0.0

    return {
        "scanDelay": scan_delay,
        "scanPeriod": scan_period,
        "laserDelay": laser_delay,
        "laserDuration": laser_duration,
        "cameraDelay": camera_delay,
        "cameraDuration": camera_duration,  # Must be > 0!
        "cameraExposure": camera_exposure,
        "sliceDuration": slice_duration,
        "frameRate": frame_rate,
    }


# ============================================================================
# PIEZO-GALVO CALIBRATION
# ============================================================================


@dataclass
class PiezoGalvoCalibration:
    """
    Piezo-galvo synchronization calibration data.

    This calibration establishes the relationship between galvo mirror angle
    (which positions the light sheet) and piezo objective position (which
    positions the sample). For optimal focus throughout a volume, these must
    move together according to: piezo_um = slope * galvo_deg + offset

    Attributes:
        slope_um_per_deg: Piezo movement per degree of galvo (typically ~100 µm/°)
        offset_um: Piezo position when galvo is at 0°
        galvo_top_deg: Galvo angle at top edge of sample (including tolerance)
        galvo_bottom_deg: Galvo angle at bottom edge of sample (including tolerance)
        piezo_top_um: Corresponding piezo position at top
        piezo_bottom_um: Corresponding piezo position at bottom
        timestamp: When calibration was performed
        sample_type: Type of sample used for calibration
        device_piezo: Micro-Manager piezo device name
        device_galvo: Micro-Manager galvo device name
        edge_top_deg: Detected top edge before tolerance (optional)
        edge_bottom_deg: Detected bottom edge before tolerance (optional)
        calib_inset_fraction: Fraction inset from edges for calibration (optional)
        calib_strategy: Calibration strategy used (optional)

    Examples:
        Create from calibration measurements:
        >>> calib = PiezoGalvoCalibration(
        ...     slope_um_per_deg=99.5,
        ...     offset_um=0.0,
        ...     galvo_top_deg=-0.30,
        ...     galvo_bottom_deg=0.30,
        ...     piezo_top_um=-29.85,
        ...     piezo_bottom_um=29.85,
        ...     timestamp=datetime.now().isoformat(),
        ...     sample_type="embryo"
        ... )

        Load from file:
        >>> calib = PiezoGalvoCalibration.from_file("calibration.json")

        Save to file:
        >>> calib.to_file("calibration.json")

        Calculate piezo position for given galvo angle:
        >>> piezo_pos = calib.galvo_to_piezo(0.15)  # galvo at +0.15°
    """

    slope_um_per_deg: float
    offset_um: float
    galvo_top_deg: float
    galvo_bottom_deg: float
    piezo_top_um: float
    piezo_bottom_um: float
    timestamp: str
    sample_type: str = "embryo"
    device_piezo: str = "PiezoStage:P:34"
    device_galvo: str = "Scanner:AB:33"

    # Optional metadata
    edge_top_deg: float | None = None
    edge_bottom_deg: float | None = None
    calib_inset_fraction: float | None = None
    calib_strategy: str | None = None

    def galvo_to_piezo(self, galvo_deg: float) -> float:
        """
        Calculate piezo position for given galvo angle.

        Args:
            galvo_deg: Galvo mirror angle in degrees

        Returns:
            Piezo position in micrometers

        Example:
            >>> calib = PiezoGalvoCalibration(slope_um_per_deg=100.0, offset_um=0.0, ...)
            >>> piezo_pos = calib.galvo_to_piezo(0.25)  # Returns 25.0 µm
        """
        return self.slope_um_per_deg * galvo_deg + self.offset_um

    def piezo_to_galvo(self, piezo_um: float) -> float:
        """
        Calculate galvo angle for given piezo position.

        Args:
            piezo_um: Piezo position in micrometers

        Returns:
            Galvo mirror angle in degrees

        Example:
            >>> calib = PiezoGalvoCalibration(slope_um_per_deg=100.0, offset_um=0.0, ...)
            >>> galvo_angle = calib.piezo_to_galvo(25.0)  # Returns 0.25°
        """
        return (piezo_um - self.offset_um) / self.slope_um_per_deg

    def get_scan_range(self, tolerance_multiplier: float = 1.0) -> tuple[float, float]:
        """
        Get galvo scan range with optional tolerance multiplier.

        Args:
            tolerance_multiplier: Multiply the tolerance margin (1.0 = use as-is)

        Returns:
            Tuple of (galvo_start_deg, galvo_end_deg)

        Example:
            >>> calib = PiezoGalvoCalibration(...)
            >>> start, end = calib.get_scan_range(tolerance_multiplier=1.5)
        """
        # If we have edge data, adjust tolerance
        if self.edge_top_deg is not None and self.edge_bottom_deg is not None:
            # Estimate original tolerance
            original_tolerance = self.galvo_top_deg - self.edge_top_deg  # Should be negative

            # Apply multiplier
            new_tolerance = abs(original_tolerance) * tolerance_multiplier

            # Calculate new range
            galvo_start = self.edge_top_deg - new_tolerance
            galvo_end = self.edge_bottom_deg + new_tolerance

            return (galvo_start, galvo_end)
        else:
            # No edge data, use stored values as-is
            return (self.galvo_top_deg, self.galvo_bottom_deg)

    @classmethod
    def from_file(cls, path: Path) -> "PiezoGalvoCalibration":
        """
        Load calibration from JSON file.

        Args:
            path: Path to JSON calibration file

        Returns:
            PiezoGalvoCalibration instance

        Raises:
            FileNotFoundError: If calibration file doesn't exist
            ValueError: If calibration file is invalid

        Example:
            >>> calib = PiezoGalvoCalibration.from_file("piezo_galvo_calibration_embryo.json")
        """
        path = Path(path)

        if not path.exists():
            raise FileNotFoundError(f"Calibration file not found: {path}")

        with open(path) as f:
            data = json.load(f)

        return cls(**data)

    def to_file(self, path: Path):
        """
        Save calibration to JSON file.

        Args:
            path: Path to save JSON calibration file

        Example:
            >>> calib = PiezoGalvoCalibration(...)
            >>> calib.to_file("piezo_galvo_calibration_embryo.json")
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        with open(path, "w") as f:
            json.dump(asdict(self), f, indent=2)

    def __str__(self) -> str:
        """String representation showing key parameters."""
        return (
            f"PiezoGalvoCalibration("
            f"slope={self.slope_um_per_deg:.2f} µm/°, "
            f"offset={self.offset_um:.2f} µm, "
            f"range={self.galvo_top_deg:+.3f}° to {self.galvo_bottom_deg:+.3f}°, "
            f"sample={self.sample_type})"
        )


# ============================================================================
# PRESET CONFIGURATIONS
# ============================================================================


def get_calibration_camera_config(exposure_ms: float = 50.0) -> CameraConfig:
    """
    Get camera configuration for calibration mode.

    Args:
        exposure_ms: Camera exposure time

    Returns:
        CameraConfig for INTERNAL trigger, AREA sensor mode
    """
    return CameraConfig(mode=CameraMode.INTERNAL_AREA, exposure_ms=exposure_ms)


def get_hardware_spim_camera_config(exposure_ms: float = 10.0) -> CameraConfig:
    """
    Get camera configuration for hardware-triggered SPIM mode.

    Args:
        exposure_ms: Camera exposure time

    Returns:
        CameraConfig for EXTERNAL trigger, PROGRESSIVE sensor mode
    """
    return CameraConfig(mode=CameraMode.EXTERNAL_PROGRESSIVE, exposure_ms=exposure_ms)


def get_standard_scanner_config(y_amplitude_deg: float = 0.04) -> ScannerConfig:
    """
    Get standard scanner configuration for SPIM acquisition.

    Args:
        y_amplitude_deg: Y-axis amplitude (slice stepping range)

    Returns:
        ScannerConfig with standard X-axis (light sheet) and Y-axis (slicing) settings
    """
    return ScannerConfig(
        x_axis=GalvoAxisConfig(
            amplitude_deg=8.0,
            offset_deg=0.0005,
            pattern=ScannerPattern.TRIANGLE,
            mode=ScannerMode.ENABLED_SYNCED,
        ),
        y_axis=GalvoAxisConfig(
            amplitude_deg=y_amplitude_deg,
            offset_deg=0.0,
            pattern=ScannerPattern.TRIANGLE,
            mode=ScannerMode.ENABLED_SYNCED,
        ),
        beam_enabled=True,
    )
