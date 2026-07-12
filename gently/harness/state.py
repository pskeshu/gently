"""
State management for embryos and experiments.

Embryo-related classes across the codebase — single source of truth:

- ``EmbryoState`` (this file): in-memory canonical state during one session.
  Includes identity, position, calibration, acquisition params, status,
  exposure tracking, history, and timelapse runtime fields (stop_condition,
  is_complete, etc.). The agent's ``experiment.embryos`` dict and the
  orchestrator's ``_embryo_states`` dict both hold references to the SAME
  ``EmbryoState`` instances — no duplication.

- ``gently.core.store_types.EmbryoInfo`` (TypedDict): on-disk YAML schema
  for ``embryo.yaml``. Derived from ``EmbryoState`` (subset of fields).

- ``gently.harness.memory.model.EmbryoUnderstanding``: the agent's
  synthesized belief about an embryo, persisted across sessions. Distinct
  from state — this is memory/learning, not raw status.

- ``gently.dataset.embryo_dataset.DatasetEmbryoEntry`` (dataclass):
  catalog entry for an embryo in a benchmark dataset (offline). Distinct
  domain from ``EmbryoState``.

Historical: ``EmbryoAcquisitionState`` was removed in Phase 1.5 — its
fields are now on ``EmbryoState`` directly.
"""

import logging
import re
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

import numpy as np

# Re-export CalibrationPrior from its hardware-specific home for backward compat.
# CalibrationPrior is diSPIM-specific (piezo-galvo linear fit). Other hardware
# modules will define their own calibration models.
from gently.hardware.dispim.calibration import CalibrationPrior

logger = logging.getLogger(__name__)


@dataclass
class FocusDataPoint:
    """
    Single focus measurement - accumulated over time to build a focus map.

    Each time focus is measured (via fine_focus, calibration, or manual adjustment),
    a FocusDataPoint is recorded. Over a timelapse, these accumulate to reveal:
    - The Z vs secondary-axis relationship for each sample
    - Focus drift over time (sample settling, temperature changes)
    - Quality trends (degrading R² might indicate sample issues)

    This enables sample-aware microscopy where each sample has its own
    learned focus profile that improves over time.

    The axes are named generically:
    - z: primary focus axis (µm) — piezo for diSPIM, Z-motor for 2P/confocal
    - secondary_axis: optional second axis — galvo for diSPIM, unused (0.0) for single-axis systems
    """

    z: float  # Primary focus position (µm)
    secondary_axis: float  # Secondary axis position (galvo deg for diSPIM, 0.0 otherwise)
    score: float  # Focus quality score (algorithm-dependent)
    r_squared: float  # Gaussian fit quality (0-1), higher = more reliable
    timestamp: datetime  # When this measurement was made
    method: str  # 'calibration', 'fine_focus', 'manual'
    algorithm: str = "fft_bandpass"  # Focus algorithm used

    # Backward-compatible properties for code that uses the old field names
    @property
    def piezo(self) -> float:
        return self.z

    @property
    def galvo(self) -> float:
        return self.secondary_axis

    def to_dict(self) -> dict:
        """Serialize for JSON storage"""
        return {
            "z": self.z,
            "secondary_axis": self.secondary_axis,
            "score": self.score,
            "r_squared": self.r_squared,
            "timestamp": self.timestamp.isoformat(),
            "method": self.method,
            "algorithm": self.algorithm,
            # Backward-compatible keys for existing serialized data
            "galvo": self.secondary_axis,
            "piezo": self.z,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "FocusDataPoint":
        """Deserialize from JSON. Handles both old (galvo/piezo) and new (z/secondary_axis) keys."""
        return cls(
            z=data.get("z", data.get("piezo", 0.0)),
            secondary_axis=data.get("secondary_axis", data.get("galvo", 0.0)),
            score=data["score"],
            r_squared=data["r_squared"],
            timestamp=datetime.fromisoformat(data["timestamp"]),
            method=data["method"],
            algorithm=data.get("algorithm", "fft_bandpass"),
        )


@dataclass
class ImageRecord:
    """Record of a single acquired image/volume"""

    embryo_id: str
    timepoint: int
    timestamp: datetime
    volume_path: str  # Path to full TIFF volume on disk
    max_projection_b64: str  # Base64-encoded JPEG for Claude Vision
    size_kb: float
    # UID-based data references (new data layer)
    volume_uid: str | None = None  # UID for volume in DataStore
    projection_uid: str | None = None  # UID for max projection in DataStore


@dataclass
class EmbryoState:
    """Complete state of one C. elegans embryo"""

    # Identity
    id: str  # "embryo_1"
    uid: str | None = None  # Global unique identifier for cross-session tracking
    nickname: str | None = None  # Agent-assigned: "the fast one"
    user_label: str | None = None  # User-provided: "control_1"
    # Role key into gently.harness.roles.REGISTRY. Drives cadence policy,
    # detector selection, photodose budget, UI presentation. Default "test"
    # is the safe choice — accidental Calibration→Test only over-protects;
    # accidental Test→Calibration would burn extra dose on the precious sample.
    role: str = "test"
    # Free-form biological sample descriptor (orthogonal to role). Examples:
    # "pan-nuclear GFP", "H2B-mCherry", "wild-type". None = unspecified.
    strain: str | None = None

    # Position — two-stage: coarse (bottom-camera detection or manual map
    # placement, always present once an embryo exists) and fine (populated
    # later by SPIM-objective alignment). Resolved value is exposed by the
    # `stage_position` property so downstream motion/perception can stay
    # agnostic about which stage we're in.
    position_coarse: dict[str, float] = field(default_factory=dict)  # {'x': ..., 'y': ...}
    position_fine: dict[str, float] = field(default_factory=dict)  # empty until SPIM head alignment
    calibration: dict = field(default_factory=dict)  # Galvo/piezo parameters
    detection_confidence: float = 0.0  # SAM/detection confidence score (0-1)

    # Acquisition Parameters (current)
    interval_seconds: float | None = None  # Per-embryo interval; None = use timelapse default
    num_slices: int = 50
    exposure_ms: float = 10.0
    priority: str = "normal"  # high/normal/low
    acquisition_mode: str = "volume"  # "volume" or "snap"
    # Per-embryo 488 laser power %. None = use device-layer default (no
    # change at acquire time). Float values are hard-limited at the device
    # layer by DiSPIMLightSource.POWER_LIMITS_PCT[488] (default 2-6%).
    laser_power_488_pct: float | None = None

    # Status
    last_imaged: datetime | None = None
    timepoints_acquired: int = 0
    should_skip: bool = False
    skip_reason: str | None = None

    # Timelapse runtime state (consolidated from former EmbryoAcquisitionState).
    # Populated/used by TimelapseOrchestrator while this embryo is part of an
    # active timelapse. stop_condition is typed Any to avoid inverting the
    # harness→app dependency direction; the orchestrator stores its
    # StopCondition object here directly. YAML persistence for these fields
    # is a Phase 9 concern.
    stop_condition: Any = None
    is_complete: bool = False
    completion_reason: str | None = None
    error_count: int = 0
    last_error: str | None = None
    detection_triggered_at: int | None = None
    detection_type: str | None = None
    no_object_since_timepoint: int | None = None
    # Count of consecutive "no_object" detections. Reset to 0 whenever
    # the embryo is detected again. When this crosses the role's
    # ``no_object_consecutive_terminal`` threshold, the orchestrator
    # treats the embryo as gone (likely hatched / drifted off) and
    # marks it complete.
    consecutive_no_object: int = 0

    # Async cadence (Phase 4). Each embryo has its own next_due_at;
    # _run_loop is a priority queue keyed on next_due_at, not a synchronized
    # round timer.
    #   cadence_phase: "normal" | "fast" | "burst" | "paused"
    #     - normal: image at the role's default interval (e.g. 5 min)
    #     - fast:   accelerated cadence (e.g. 1 min) after signal onset
    #     - burst:  this embryo is currently in a burst sequence (Phase 7)
    #     - paused: skip in the due loop (over-budget, manually paused, or
    #               idle during another embryo's burst)
    cadence_phase: str = "normal"
    next_due_at: datetime | None = None

    # Light exposure tracking (for phototoxicity monitoring)
    exposure_count: int = 0  # Number of imaging events (snaps + volumes)
    total_exposure_ms: float = 0.0  # Cumulative laser-on time in milliseconds

    # Analysis Results (cached)
    hatching_status: dict = field(default_factory=dict)
    # {hatched: bool, confidence: str, timepoint: int}

    morphology_history: list[dict] = field(default_factory=list)
    # [{timepoint, size, shape, activity_score}]

    fluorescence_history: list[dict] = field(default_factory=list)
    # [{timepoint, mean_intensity, photobleaching_estimate}]

    custom_classifications: dict = field(default_factory=dict)
    # User-defined: {"first_cleavage": {detected: bool, timepoint: 42}}

    # Verification round tracking (for consecutive confirmation)
    pending_verification: bool = False  # True when detection fired, awaiting verification
    consecutive_detection_count: int = 0  # Must reach 5 consecutive verified detections to stop
    last_detection_round: int | None = None  # Round when detection was last verified

    # Detection results from detector system
    detection_results: dict[str, list[dict]] = field(default_factory=dict)
    # detector_name -> list of detection results
    # e.g., {"comma_stage": [{"timepoint": 120, "detected": False, "confidence": "HIGH"}, ...]}

    # CV Subagent analysis results (populated from CV_RESULT_READY events)
    cv_analyses: dict[str, list[dict]] = field(default_factory=dict)
    # result_type -> list of results by timepoint
    # e.g., {"nuclei_count": [{"timepoint": 5, "num_nuclei": 66, ...}]}

    # Quick-access fields for latest CV results (for /embryos display)
    latest_nuclei_count: int | None = None
    latest_developmental_stage: str | None = None
    latest_elongation_ratio: float | None = None

    # Images (recent for context)
    recent_images: list[ImageRecord] = field(default_factory=list)
    # Keep last 10 for temporal context in Claude Vision calls

    # Focus history - accumulated piezo-galvo measurements over time
    focus_history: list[FocusDataPoint] = field(default_factory=list)
    # Each focus operation adds a datapoint, building a focus map for this embryo

    def add_focus_datapoint(
        self,
        z: float | None = None,
        secondary_axis: float = 0.0,
        score: float = 0.0,
        r_squared: float = 0.0,
        method: str = "manual",
        algorithm: str = "fft_bandpass",
        # Backward-compatible kwargs
        galvo: float | None = None,
        piezo: float | None = None,
    ):
        """
        Record a focus measurement for this embryo.

        Called by fine_focus, calibrate_embryo, or manual focus operations.
        Over time, this builds a focus map showing the Z vs secondary-axis
        relationship and any drift that occurs during long timelapses.

        Parameters
        ----------
        z : float
            Primary focus axis position (µm). For diSPIM: piezo position.
        secondary_axis : float
            Secondary axis position. For diSPIM: galvo position. Default 0.0.
        score : float
            Focus quality score
        r_squared : float
            Fit quality (0-1)
        method : str
            How focus was determined: 'fine_focus', 'calibration', 'manual'
        algorithm : str
            Focus algorithm used: 'fft_bandpass', 'gradient', etc.
        galvo : float, optional
            Backward-compatible alias for secondary_axis
        piezo : float, optional
            Backward-compatible alias for z
        """
        # Support old kwarg names
        if z is None:
            z = piezo if piezo is not None else 0.0
        if galvo is not None:
            secondary_axis = galvo

        self.focus_history.append(
            FocusDataPoint(
                z=z,
                secondary_axis=secondary_axis,
                score=score,
                r_squared=r_squared,
                timestamp=datetime.now(),
                method=method,
                algorithm=algorithm,
            )
        )

    def get_focus_at_secondary(
        self,
        secondary_position: float,
        max_age_hours: float | None = None,
        min_r_squared: float = 0.5,
    ) -> float | None:
        """
        Get the best Z position for a given secondary axis position.

        Uses accumulated focus data with optional time weighting.
        If multiple measurements exist at similar secondary positions,
        uses the most recent high-quality one.

        For diSPIM: secondary=galvo, returns optimal piezo position.
        For single-axis systems: call with secondary_position=0.0.

        Parameters
        ----------
        secondary_position : float
            Target secondary axis position
        max_age_hours : float, optional
            Only consider measurements within this time window
        min_r_squared : float
            Minimum fit quality to consider

        Returns
        -------
        float or None
            Optimal Z position, or None if no suitable data
        """
        if not self.focus_history:
            return None

        now = datetime.now()
        candidates = []

        for fp in self.focus_history:
            # Filter by quality
            if fp.r_squared < min_r_squared:
                continue

            # Filter by age
            if max_age_hours is not None:
                age_hours = (now - fp.timestamp).total_seconds() / 3600
                if age_hours > max_age_hours:
                    continue

            # Weight by secondary axis proximity and recency
            axis_distance = abs(fp.secondary_axis - secondary_position)
            age_hours = (now - fp.timestamp).total_seconds() / 3600

            candidates.append(
                {
                    "z": fp.z,
                    "axis_distance": axis_distance,
                    "age_hours": age_hours,
                    "r_squared": fp.r_squared,
                }
            )

        if not candidates:
            return None

        # If we have exact matches, use the most recent
        exact_matches = [c for c in candidates if c["axis_distance"] < 0.01]
        if exact_matches:
            # Sort by recency, return most recent
            exact_matches.sort(key=lambda x: x["age_hours"])
            return exact_matches[0]["z"]

        # Otherwise, interpolate from nearby measurements
        # Sort by axis distance
        candidates.sort(key=lambda x: x["axis_distance"])
        return candidates[0]["z"]  # Return closest match

    # Backward-compatible alias
    def get_focus_at_galvo(self, galvo_position: float, **kwargs) -> float | None:
        """Backward-compatible alias for get_focus_at_secondary."""
        return self.get_focus_at_secondary(galvo_position, **kwargs)

    def get_z_axis_fit(
        self, max_age_hours: float | None = None, min_r_squared: float = 0.5
    ) -> tuple[float, float] | None:
        """
        Fit a linear relationship between Z and secondary axis from accumulated data.

        Returns slope and intercept: z = slope * secondary_axis + intercept

        For diSPIM: this is the piezo-galvo relationship.
        For single-axis systems: returns None (no secondary axis to fit against).

        Parameters
        ----------
        max_age_hours : float, optional
            Only use data within this time window
        min_r_squared : float
            Minimum quality threshold

        Returns
        -------
        Tuple[float, float] or None
            (slope, intercept) or None if insufficient data
        """
        if len(self.focus_history) < 2:
            return None

        now = datetime.now()
        secondary = []
        zs = []

        for fp in self.focus_history:
            if fp.r_squared < min_r_squared:
                continue
            if max_age_hours is not None:
                age_hours = (now - fp.timestamp).total_seconds() / 3600
                if age_hours > max_age_hours:
                    continue
            secondary.append(fp.secondary_axis)
            zs.append(fp.z)

        if len(secondary) < 2:
            return None

        # Linear fit: z = slope * secondary_axis + intercept
        secondary_arr = np.array(secondary)
        zs_arr = np.array(zs)

        # Use polyfit for linear regression
        try:
            coeffs = np.polyfit(secondary_arr, zs_arr, 1)
            return (float(coeffs[0]), float(coeffs[1]))  # slope, intercept
        except Exception:
            return None

    # Backward-compatible alias
    def get_piezo_galvo_fit(self, **kwargs) -> tuple[float, float] | None:
        """Backward-compatible alias for get_z_axis_fit."""
        return self.get_z_axis_fit(**kwargs)

    def get_focus_drift_rate(
        self,
        secondary_position: float = 0.0,
        galvo_position: float | None = None,
        min_measurements: int = 3,
    ) -> float | None:
        """
        Calculate how fast focus is drifting (µm/hour) at a given secondary axis position.

        Parameters
        ----------
        secondary_position : float
            Secondary axis position to analyze drift at
        galvo_position : float, optional
            Backward-compatible alias for secondary_position
        min_measurements : int
            Minimum datapoints needed for drift calculation

        Returns
        -------
        float or None
            Drift rate in µm/hour, or None if insufficient data
        """
        if galvo_position is not None:
            secondary_position = galvo_position
        # Get measurements at similar secondary axis position
        relevant = [
            fp
            for fp in self.focus_history
            if abs(fp.secondary_axis - secondary_position) < 0.1 and fp.r_squared >= 0.5
        ]

        if len(relevant) < min_measurements:
            return None

        # Sort by time
        relevant.sort(key=lambda x: x.timestamp)

        # Calculate drift rate using linear regression on time vs Z
        times_hours = []
        z_positions = []
        t0 = relevant[0].timestamp

        for fp in relevant:
            hours = (fp.timestamp - t0).total_seconds() / 3600
            times_hours.append(hours)
            z_positions.append(fp.z)

        if times_hours[-1] - times_hours[0] < 0.1:  # Less than 6 minutes span
            return None

        try:
            coeffs = np.polyfit(times_hours, z_positions, 1)
            return float(coeffs[0])  # slope = µm/hour
        except Exception:
            return None

    def needs_refocus(
        self,
        max_age_minutes: float = 60,
        secondary_position: float = 0.0,
        galvo_position: float | None = None,
    ) -> bool:
        """
        Determine if this embryo needs focus re-measurement.

        Parameters
        ----------
        max_age_minutes : float
            Focus data older than this is considered stale
        secondary_position : float
            Secondary axis position to check
        galvo_position : float, optional
            Backward-compatible alias for secondary_position

        Returns
        -------
        bool
            True if focus should be re-measured
        """
        if galvo_position is not None:
            secondary_position = galvo_position

        if not self.focus_history:
            return True

        now = datetime.now()

        # Find most recent high-quality measurement at this secondary position
        for fp in reversed(self.focus_history):
            if abs(fp.secondary_axis - secondary_position) < 0.1 and fp.r_squared >= 0.5:
                age_minutes = (now - fp.timestamp).total_seconds() / 60
                return age_minutes > max_age_minutes

        return True  # No suitable measurement found

    def get_focus_summary(self) -> str:
        """Get human-readable summary of focus history"""
        if not self.focus_history:
            return "No focus measurements recorded"

        n_points = len(self.focus_history)
        first = self.focus_history[0]
        last = self.focus_history[-1]

        # Time span
        span_hours = (last.timestamp - first.timestamp).total_seconds() / 3600

        # Drift at secondary_axis=0
        drift = self.get_focus_drift_rate(secondary_position=0.0)

        lines = [
            f"Focus history: {n_points} measurements over {span_hours:.1f} hours",
            f"Latest: z={last.z:.2f}µm @ secondary={last.secondary_axis:.2f}"
            f" (R²={last.r_squared:.3f})",
        ]

        if drift is not None:
            lines.append(f"Drift rate: {drift:.3f} µm/hour")

        fit = self.get_z_axis_fit()
        if fit:
            slope, intercept = fit
            lines.append(f"Z-axis fit: z = {slope:.2f}*secondary + {intercept:.2f}")

        return "\n".join(lines)

    def add_detection_result(self, detector_name: str, result: dict):
        """
        Add detection result from detector system

        Parameters
        ----------
        detector_name : str
            Name of detector
        result : dict
            Detection result (from DetectionResult.to_dict())
        """
        if detector_name not in self.detection_results:
            self.detection_results[detector_name] = []

        self.detection_results[detector_name].append(result)

    def get_latest_detection(self, detector_name: str) -> dict | None:
        """Get most recent detection result for a detector"""
        if detector_name not in self.detection_results:
            return None

        results = self.detection_results[detector_name]
        return results[-1] if results else None

    def was_detected(self, detector_name: str, require_verified: bool = False) -> bool:
        """
        Check if detector has ever fired (detected=True) for this embryo.

        Parameters
        ----------
        detector_name : str
            Name of detector to check
        require_verified : bool
            If True, only return True if the detection was also verified.
            This is important for critical actions like stopping experiments.

        Returns
        -------
        bool
            True if detected (and verified if require_verified=True)
        """
        if detector_name not in self.detection_results:
            return False

        for result in self.detection_results[detector_name]:
            if result.get("detected", False):
                if require_verified:
                    if result.get("verified", False):
                        return True
                else:
                    return True
        return False

    def mark_detection_verified(self, detector_name: str, timepoint: int | None = None) -> bool:
        """
        Mark a detection result as verified by the challenger system.

        Parameters
        ----------
        detector_name : str
            Name of detector
        timepoint : int, optional
            Specific timepoint to mark. If None, marks the most recent detection.

        Returns
        -------
        bool
            True if a detection was found and marked verified
        """
        if detector_name not in self.detection_results:
            return False

        results = self.detection_results[detector_name]
        if not results:
            return False

        # Find the result to mark
        if timepoint is not None:
            # Find by timepoint
            for result in results:
                if result.get("timepoint") == timepoint and result.get("detected", False):
                    result["verified"] = True
                    return True
        else:
            # Mark the most recent detected result
            for result in reversed(results):
                if result.get("detected", False):
                    result["verified"] = True
                    return True

        return False

    def add_cv_result(self, result_type: str, result: dict):
        """
        Add CV analysis result from CV subagent.

        Parameters
        ----------
        result_type : str
            Type of result: "nuclei_count", "stage_classification", "elongation", etc.
        result : dict
            Analysis result data
        """
        if result_type not in self.cv_analyses:
            self.cv_analyses[result_type] = []

        # Add timestamp if not present
        if "timestamp" not in result:
            result["timestamp"] = datetime.now().isoformat()

        self.cv_analyses[result_type].append(result)

        # Update quick-access fields
        if result_type == "nuclei_count" and "num_nuclei" in result:
            self.latest_nuclei_count = result["num_nuclei"]
        elif result_type == "stage_classification" and "stage" in result:
            self.latest_developmental_stage = result["stage"]
        elif result_type == "elongation" and "elongation_ratio" in result:
            self.latest_elongation_ratio = result["elongation_ratio"]

    def get_cv_result(self, result_type: str, timepoint: int | None = None) -> dict | None:
        """
        Get CV analysis result, optionally filtered by timepoint.

        Parameters
        ----------
        result_type : str
            Type of result to retrieve
        timepoint : int, optional
            Specific timepoint, or latest if None

        Returns
        -------
        dict or None
            Result data, or None if not found
        """
        if result_type not in self.cv_analyses:
            return None

        results = self.cv_analyses[result_type]
        if not results:
            return None

        if timepoint is not None:
            # Filter by timepoint
            matching = [r for r in results if r.get("timepoint") == timepoint]
            return matching[-1] if matching else None

        # Return most recent
        return results[-1]

    def get_cv_summary(self) -> dict:
        """
        Get summary of CV analysis results for display.

        Returns
        -------
        dict
            Summary with latest values and counts
        """
        return {
            "nuclei_count": self.latest_nuclei_count,
            "developmental_stage": self.latest_developmental_stage,
            "elongation_ratio": self.latest_elongation_ratio,
            "analyses_count": {
                result_type: len(results) for result_type, results in self.cv_analyses.items()
            },
        }

    def update_from_analysis(self, analysis_result: dict):
        """Update state with new analysis"""
        if "hatching" in analysis_result:
            self.hatching_status = analysis_result["hatching"]

        if "morphology" in analysis_result:
            self.morphology_history.append(
                {"timepoint": self.timepoints_acquired, **analysis_result["morphology"]}
            )

        if "fluorescence" in analysis_result:
            self.fluorescence_history.append(
                {
                    "timepoint": self.timepoints_acquired,
                    **analysis_result["fluorescence"],
                }
            )

    def to_summary(self) -> str:
        """Format for Claude system prompt"""
        status_parts = []

        # Identity (with role — the agent must see this to behave correctly)
        name = self.nickname or self.user_label or self.id
        role_tag = (self.role or "unassigned").upper()
        status_parts.append(f"{name} ({self.id}) [role={role_tag}]")

        # Cadence phase (async timelapse)
        phase = getattr(self, "cadence_phase", "normal")
        if phase and phase != "normal":
            status_parts.append(f"phase={phase}")

        # Timing
        if self.last_imaged:
            mins_ago = (datetime.now() - self.last_imaged).seconds // 60
            status_parts.append(f"last imaged {mins_ago}min ago")
        else:
            status_parts.append("not yet imaged")

        # Status
        if self.hatching_status.get("hatched"):
            status_parts.append(f"hatched at t{self.hatching_status['timepoint']:04d}")
        elif self.should_skip:
            status_parts.append(f"skipped ({self.skip_reason})")
        else:
            status_parts.append("active")

        # Current params
        status_parts.append(f"interval={self.interval_seconds}s")
        status_parts.append(f"slices={self.num_slices}")
        if self.laser_power_488_pct is not None:
            status_parts.append(f"488={self.laser_power_488_pct}%")
        status_parts.append(f"priority={self.priority}")

        return " | ".join(status_parts)

    def record_exposure(
        self,
        exposure_ms: float,
        num_frames: int = 1,
        timestamp: datetime | None = None,
    ):
        """
        Record light exposure for phototoxicity tracking.

        Parameters
        ----------
        exposure_ms : float
            Exposure time per frame in milliseconds
        num_frames : int
            Number of frames captured (1 for snap, num_slices for volume)
        timestamp : datetime, optional
            When the exposure occurred. Defaults to now.
        """
        self.exposure_count += 1
        self.total_exposure_ms += exposure_ms * num_frames
        self.last_imaged = timestamp or datetime.now()

    def get_exposure_summary(self) -> str:
        """Get human-readable exposure summary"""
        if self.exposure_count == 0:
            return "No light exposure recorded"

        total_sec = self.total_exposure_ms / 1000
        if total_sec < 1:
            time_str = f"{self.total_exposure_ms:.0f}ms"
        elif total_sec < 60:
            time_str = f"{total_sec:.1f}s"
        else:
            time_str = f"{total_sec / 60:.1f}min"

        return f"{self.exposure_count} exposures, {time_str} total"

    @property
    def stage_position(self) -> dict[str, float]:
        """Resolved XY position — fine if SPIM-aligned, else coarse.

        Coarse comes from the bottom-camera detection / manual map placement.
        Fine comes from the SPIM-objective alignment workflow (not built yet).
        Callers that just want "where is this embryo" read this; callers that
        care about calibration state read position_coarse / position_fine
        directly.
        """
        return self.position_fine if self.position_fine else self.position_coarse

    @stage_position.setter
    def stage_position(self, value: dict[str, float]) -> None:
        """Back-compat setter — writes to coarse.

        Legacy callers that assigned `embryo.stage_position = {...}` were
        writing a bottom-camera / manual position; that's the coarse stage.
        New code should set position_coarse or position_fine explicitly.
        """
        self.position_coarse = value or {}

    @property
    def has_fine_position(self) -> bool:
        """True once SPIM-objective alignment has refined the coarse position."""
        return bool(self.position_fine)

    def to_dict(self) -> dict:
        """Serialize for API responses"""
        return {
            "id": self.id,
            "uid": self.uid,
            "nickname": self.nickname,
            "user_label": self.user_label,
            "role": self.role,
            "stage_position": self.stage_position,
            "position_coarse": self.position_coarse,
            "position_fine": self.position_fine,
            "has_fine_position": self.has_fine_position,
            "calibration": self.calibration,
            "detection_confidence": self.detection_confidence,
            "interval_seconds": self.interval_seconds,
            "num_slices": self.num_slices,
            "exposure_ms": self.exposure_ms,
            "priority": self.priority,
            "acquisition_mode": self.acquisition_mode,
            "laser_power_488_pct": self.laser_power_488_pct,
            "last_imaged": self.last_imaged.isoformat() if self.last_imaged else None,
            "timepoints_acquired": self.timepoints_acquired,
            "should_skip": self.should_skip,
            "skip_reason": self.skip_reason,
            "exposure_count": self.exposure_count,
            "total_exposure_ms": self.total_exposure_ms,
            "hatching_status": self.hatching_status,
            "pending_verification": self.pending_verification,
            "consecutive_detection_count": self.consecutive_detection_count,
            "last_detection_round": self.last_detection_round,
            "recent_analyses": {
                "morphology": self.morphology_history[-5:] if self.morphology_history else [],
                "fluorescence": self.fluorescence_history[-5:] if self.fluorescence_history else [],
                "custom": self.custom_classifications,
            },
            "focus_history": [fp.to_dict() for fp in self.focus_history],
        }


class ExperimentState:
    """Global experiment state"""

    def __init__(self):
        self.embryos: dict[str, EmbryoState] = {}
        self.start_time: datetime | None = None
        self.acquisition_status: str = "idle"  # idle/running/paused/completed
        self.current_plan_name: str | None = None
        self.plan_history: list[dict] = []
        self.metadata: dict = {}

        # Active plan item — set during plan context resolution at startup.
        # When set, the agent's system prompt includes the full ImagingSpec
        # so it knows what it's here to do without being told.
        self.active_plan_item_id: str | None = None

        # Session-level calibration prior for cross-embryo learning
        # Updated after each successful calibration, used to initialize subsequent embryos
        self.calibration_prior: CalibrationPrior = CalibrationPrior()

        # Observer hook — agent wires this at startup to publish EMBRYOS_UPDATE
        # over the event bus. Kept as a plain callback so this module stays
        # bus-agnostic.
        self.on_embryos_changed: Callable[[], None] | None = None

    def notify_embryos_changed(self) -> None:
        """Fire the on_embryos_changed observer if one is wired.

        Call this after any mutation the agent can't intercept through
        add_embryo / remove_embryo (e.g. a direct write to
        embryo.position_coarse). UI hooks must not raise — failures here are
        swallowed so state mutations stay durable.
        """
        cb = self.on_embryos_changed
        if cb is None:
            return
        try:
            cb()
        except Exception:
            logger.exception("ExperimentState.on_embryos_changed callback failed")

    def add_embryo(
        self,
        embryo_id: str,
        position: dict | None = None,
        calibration: dict | None = None,
        user_label: str | None = None,
        confidence: float = 0.0,
        uid: str | None = None,
        role: str = "test",
        position_fine: dict | None = None,
    ):
        """Register new embryo.

        ``role`` must be a key in :data:`gently.harness.roles.REGISTRY`
        (e.g. ``"test"``, ``"calibration"``, ``"unassigned"``). Unknown roles
        raise KeyError.

        `position` is the coarse XY (bottom-camera detection or manual map
        placement). `position_fine` is reserved for the future SPIM-objective
        alignment workflow and defaults to empty.

        Emits an ``EMBRYO_DETECTED`` event so listeners (e.g. the viz
        server's TimelapseStateTracker, which feeds the device map)
        learn about marked embryos immediately — not just after the
        first acquisition.
        """
        from gently.harness.roles import get_role

        get_role(role)  # raises KeyError if unknown

        # Auto-start experiment when first embryo is added
        if self.start_time is None:
            self.start_time = datetime.now()

        pos = position or {}
        self.embryos[embryo_id] = EmbryoState(
            id=embryo_id,
            uid=uid,
            position_coarse=position or {},
            position_fine=position_fine or {},
            calibration=calibration or {},
            user_label=user_label,
            detection_confidence=confidence,
            role=role,
        )
        self.notify_embryos_changed()

        # Fire the registration event. Late-bound import keeps this module
        # decoupled from the event bus until first use.
        try:
            from gently.core import EventType, get_event_bus

            get_event_bus().publish(
                event_type=EventType.EMBRYO_DETECTED,
                data={
                    "embryo_id": embryo_id,
                    "uid": uid,
                    "x": pos.get("x"),
                    "y": pos.get("y"),
                    "role": role,
                    "user_label": user_label,
                    "confidence": confidence,
                },
                source="experiment.add_embryo",
            )
        except Exception:
            # Don't let event-bus issues block embryo registration.
            pass

    def remove_embryo(self, embryo_id: str) -> bool:
        """Remove embryo from experiment (e.g., false detection)"""
        if embryo_id in self.embryos:
            del self.embryos[embryo_id]
            self.notify_embryos_changed()
            return True
        return False

    def assign_nickname(self, embryo_id: str, nickname: str):
        """Agent assigns intuitive name"""
        if embryo_id in self.embryos:
            self.embryos[embryo_id].nickname = nickname
            self.notify_embryos_changed()

    def get_embryo_by_any_name(self, name: str) -> EmbryoState | None:
        """Get embryo by ID, nickname, or user label"""
        # Direct ID match
        if name in self.embryos:
            return self.embryos[name]

        # Search by nickname or label
        for embryo in self.embryos.values():
            if embryo.nickname == name or embryo.user_label == name:
                return embryo

        # Try extracting number from name like "embryo 3" -> "embryo_3"
        match = re.search(r"(\d+)", name)
        if match:
            num = int(match.group(1))
            # Try simple format first (embryo_3)
            potential_id = f"embryo_{num}"
            if potential_id in self.embryos:
                return self.embryos[potential_id]
            # Also try padded format for backwards compatibility (embryo_003)
            potential_id_padded = f"embryo_{num:03d}"
            if potential_id_padded in self.embryos:
                return self.embryos[potential_id_padded]

        return None

    def get_summary(self) -> str:
        """Full experiment summary for Claude"""
        if not self.start_time:
            return "No active experiment"

        duration = datetime.now() - self.start_time
        hours = duration.seconds // 3600
        minutes = (duration.seconds % 3600) // 60

        lines = [
            f"Experiment Status: {self.acquisition_status}",
            f"Duration: {hours}h {minutes}m",
            f"Embryos: {len(self.embryos)}",
            "",
            "Per-embryo status:",
        ]

        for embryo in sorted(self.embryos.values(), key=lambda e: e.id):
            lines.append(f"  {embryo.to_summary()}")

        if self.current_plan_name:
            lines.append(f"\nCurrent plan: {self.current_plan_name}")

        if self.active_plan_item_id:
            lines.append(f"Active plan item: {self.active_plan_item_id}")

        return "\n".join(lines)

    def to_dict(self) -> dict:
        """Serialize for API responses"""
        return {
            "start_time": self.start_time.isoformat() if self.start_time else None,
            "acquisition_status": self.acquisition_status,
            "current_plan_name": self.current_plan_name,
            "active_plan_item_id": self.active_plan_item_id,
            "embryo_count": len(self.embryos),
            "embryos": {eid: e.to_dict() for eid, e in self.embryos.items()},
            "metadata": self.metadata,
            "calibration_prior": self.calibration_prior.to_dict(),
        }
