"""
State management for embryos and experiments
"""

import re
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from pathlib import Path
import numpy as np


@dataclass
class FocusDataPoint:
    """
    Single focus measurement - accumulated over time to build a focus map.

    Each time focus is measured (via fine_focus, calibration, or manual adjustment),
    a FocusDataPoint is recorded. Over a timelapse, these accumulate to reveal:
    - The piezo-galvo relationship for each embryo
    - Focus drift over time (sample settling, temperature changes)
    - Quality trends (degrading R² might indicate sample issues)

    This enables sample-aware microscopy where each embryo has its own
    learned focus profile that improves over time.
    """
    galvo: float           # Galvo position (V or degrees)
    piezo: float           # Optimal piezo position (µm)
    score: float           # Focus quality score (algorithm-dependent)
    r_squared: float       # Gaussian fit quality (0-1), higher = more reliable
    timestamp: datetime    # When this measurement was made
    method: str            # 'calibration', 'fine_focus', 'manual'
    algorithm: str = 'fft_bandpass'  # Focus algorithm used

    def to_dict(self) -> Dict:
        """Serialize for JSON storage"""
        return {
            'galvo': self.galvo,
            'piezo': self.piezo,
            'score': self.score,
            'r_squared': self.r_squared,
            'timestamp': self.timestamp.isoformat(),
            'method': self.method,
            'algorithm': self.algorithm,
        }

    @classmethod
    def from_dict(cls, data: Dict) -> 'FocusDataPoint':
        """Deserialize from JSON"""
        return cls(
            galvo=data['galvo'],
            piezo=data['piezo'],
            score=data['score'],
            r_squared=data['r_squared'],
            timestamp=datetime.fromisoformat(data['timestamp']),
            method=data['method'],
            algorithm=data.get('algorithm', 'fft_bandpass'),
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
    volume_uid: Optional[str] = None  # UID for volume in DataStore
    projection_uid: Optional[str] = None  # UID for max projection in DataStore


@dataclass
class EmbryoState:
    """Complete state of one C. elegans embryo"""

    # Identity
    id: str  # "embryo_1"
    nickname: Optional[str] = None  # Agent-assigned: "the fast one"
    user_label: Optional[str] = None  # User-provided: "control_1"

    # Position
    stage_position: Dict[str, float] = field(default_factory=dict)  # {'x': 1234.5, 'y': 5678.9}
    calibration: Dict = field(default_factory=dict)  # Galvo/piezo parameters
    detection_confidence: float = 0.0  # SAM/detection confidence score (0-1)

    # Acquisition Parameters (current)
    interval_seconds: float = 120
    num_slices: int = 50
    exposure_ms: float = 10.0
    priority: str = "normal"  # high/normal/low

    # Status
    last_imaged: Optional[datetime] = None
    timepoints_acquired: int = 0
    should_skip: bool = False
    skip_reason: Optional[str] = None

    # Light exposure tracking (for phototoxicity monitoring)
    exposure_count: int = 0  # Number of imaging events (snaps + volumes)
    total_exposure_ms: float = 0.0  # Cumulative laser-on time in milliseconds

    # Analysis Results (cached)
    hatching_status: Dict = field(default_factory=dict)
    # {hatched: bool, confidence: str, timepoint: int}

    morphology_history: List[Dict] = field(default_factory=list)
    # [{timepoint, size, shape, activity_score}]

    fluorescence_history: List[Dict] = field(default_factory=list)
    # [{timepoint, mean_intensity, photobleaching_estimate}]

    custom_classifications: Dict = field(default_factory=dict)
    # User-defined: {"first_cleavage": {detected: bool, timepoint: 42}}

    # Detection results from detector system
    detection_results: Dict[str, List[Dict]] = field(default_factory=dict)
    # detector_name -> list of detection results
    # e.g., {"comma_stage": [{"timepoint": 120, "detected": False, "confidence": "HIGH"}, ...]}

    # Images (recent for context)
    recent_images: List[ImageRecord] = field(default_factory=list)
    # Keep last 10 for temporal context in Claude Vision calls

    # Focus history - accumulated piezo-galvo measurements over time
    focus_history: List[FocusDataPoint] = field(default_factory=list)
    # Each focus operation adds a datapoint, building a focus map for this embryo

    def add_focus_datapoint(self, galvo: float, piezo: float, score: float,
                            r_squared: float, method: str, algorithm: str = 'fft_bandpass'):
        """
        Record a focus measurement for this embryo.

        Called by fine_focus, calibrate_embryo, or manual focus operations.
        Over time, this builds a focus map showing the piezo-galvo relationship
        and any drift that occurs during long timelapses.

        Parameters
        ----------
        galvo : float
            Galvo position where focus was measured
        piezo : float
            Optimal piezo position found
        score : float
            Focus quality score
        r_squared : float
            Fit quality (0-1)
        method : str
            How focus was determined: 'fine_focus', 'calibration', 'manual'
        algorithm : str
            Focus algorithm used: 'fft_bandpass', 'gradient', etc.
        """
        self.focus_history.append(FocusDataPoint(
            galvo=galvo,
            piezo=piezo,
            score=score,
            r_squared=r_squared,
            timestamp=datetime.now(),
            method=method,
            algorithm=algorithm,
        ))

    def get_focus_at_galvo(self, galvo_position: float,
                           max_age_hours: Optional[float] = None,
                           min_r_squared: float = 0.5) -> Optional[float]:
        """
        Get the best piezo position for a given galvo position.

        Uses accumulated focus data with optional time weighting.
        If multiple measurements exist at similar galvo positions,
        uses the most recent high-quality one.

        Parameters
        ----------
        galvo_position : float
            Target galvo position
        max_age_hours : float, optional
            Only consider measurements within this time window
        min_r_squared : float
            Minimum fit quality to consider

        Returns
        -------
        float or None
            Optimal piezo position, or None if no suitable data
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

            # Weight by galvo proximity and recency
            galvo_distance = abs(fp.galvo - galvo_position)
            age_hours = (now - fp.timestamp).total_seconds() / 3600

            candidates.append({
                'piezo': fp.piezo,
                'galvo_distance': galvo_distance,
                'age_hours': age_hours,
                'r_squared': fp.r_squared,
            })

        if not candidates:
            return None

        # If we have exact galvo matches, use the most recent
        exact_matches = [c for c in candidates if c['galvo_distance'] < 0.01]
        if exact_matches:
            # Sort by recency, return most recent
            exact_matches.sort(key=lambda x: x['age_hours'])
            return exact_matches[0]['piezo']

        # Otherwise, interpolate from nearby measurements
        # Sort by galvo distance
        candidates.sort(key=lambda x: x['galvo_distance'])
        return candidates[0]['piezo']  # Return closest galvo match

    def get_piezo_galvo_fit(self, max_age_hours: Optional[float] = None,
                            min_r_squared: float = 0.5) -> Optional[Tuple[float, float]]:
        """
        Fit a linear relationship between piezo and galvo from accumulated data.

        Returns slope and intercept: piezo = slope * galvo + intercept

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
        galvos = []
        piezos = []

        for fp in self.focus_history:
            if fp.r_squared < min_r_squared:
                continue
            if max_age_hours is not None:
                age_hours = (now - fp.timestamp).total_seconds() / 3600
                if age_hours > max_age_hours:
                    continue
            galvos.append(fp.galvo)
            piezos.append(fp.piezo)

        if len(galvos) < 2:
            return None

        # Linear fit: piezo = slope * galvo + intercept
        galvos = np.array(galvos)
        piezos = np.array(piezos)

        # Use polyfit for linear regression
        try:
            coeffs = np.polyfit(galvos, piezos, 1)
            return (float(coeffs[0]), float(coeffs[1]))  # slope, intercept
        except Exception:
            return None

    def get_focus_drift_rate(self, galvo_position: float = 0.0,
                             min_measurements: int = 3) -> Optional[float]:
        """
        Calculate how fast focus is drifting (µm/hour) at a given galvo position.

        Parameters
        ----------
        galvo_position : float
            Galvo position to analyze drift at
        min_measurements : int
            Minimum datapoints needed for drift calculation

        Returns
        -------
        float or None
            Drift rate in µm/hour, or None if insufficient data
        """
        # Get measurements at similar galvo position
        relevant = [fp for fp in self.focus_history
                    if abs(fp.galvo - galvo_position) < 0.1 and fp.r_squared >= 0.5]

        if len(relevant) < min_measurements:
            return None

        # Sort by time
        relevant.sort(key=lambda x: x.timestamp)

        # Calculate drift rate using linear regression on time vs piezo
        times_hours = []
        piezos = []
        t0 = relevant[0].timestamp

        for fp in relevant:
            hours = (fp.timestamp - t0).total_seconds() / 3600
            times_hours.append(hours)
            piezos.append(fp.piezo)

        if times_hours[-1] - times_hours[0] < 0.1:  # Less than 6 minutes span
            return None

        try:
            coeffs = np.polyfit(times_hours, piezos, 1)
            return float(coeffs[0])  # slope = µm/hour
        except Exception:
            return None

    def needs_refocus(self, max_age_minutes: float = 60,
                      galvo_position: float = 0.0) -> bool:
        """
        Determine if this embryo needs focus re-measurement.

        Parameters
        ----------
        max_age_minutes : float
            Focus data older than this is considered stale
        galvo_position : float
            Galvo position to check

        Returns
        -------
        bool
            True if focus should be re-measured
        """
        if not self.focus_history:
            return True

        now = datetime.now()

        # Find most recent high-quality measurement at this galvo
        for fp in reversed(self.focus_history):
            if abs(fp.galvo - galvo_position) < 0.1 and fp.r_squared >= 0.5:
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

        # Drift at galvo=0
        drift = self.get_focus_drift_rate(galvo_position=0.0)

        lines = [
            f"Focus history: {n_points} measurements over {span_hours:.1f} hours",
            f"Latest: piezo={last.piezo:.2f}µm @ galvo={last.galvo:.2f} (R²={last.r_squared:.3f})",
        ]

        if drift is not None:
            lines.append(f"Drift rate: {drift:.3f} µm/hour")

        fit = self.get_piezo_galvo_fit()
        if fit:
            slope, intercept = fit
            lines.append(f"Piezo-galvo fit: piezo = {slope:.2f}*galvo + {intercept:.2f}")

        return "\n".join(lines)

    def add_detection_result(self, detector_name: str, result: Dict):
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

    def get_latest_detection(self, detector_name: str) -> Optional[Dict]:
        """Get most recent detection result for a detector"""
        if detector_name not in self.detection_results:
            return None

        results = self.detection_results[detector_name]
        return results[-1] if results else None

    def was_detected(self, detector_name: str) -> bool:
        """Check if detector has ever fired (detected=True) for this embryo"""
        if detector_name not in self.detection_results:
            return False

        return any(r.get('detected', False) for r in self.detection_results[detector_name])

    def update_from_analysis(self, analysis_result: Dict):
        """Update state with new analysis"""
        if 'hatching' in analysis_result:
            self.hatching_status = analysis_result['hatching']

        if 'morphology' in analysis_result:
            self.morphology_history.append({
                'timepoint': self.timepoints_acquired,
                **analysis_result['morphology']
            })

        if 'fluorescence' in analysis_result:
            self.fluorescence_history.append({
                'timepoint': self.timepoints_acquired,
                **analysis_result['fluorescence']
            })

    def to_summary(self) -> str:
        """Format for Claude system prompt"""
        status_parts = []

        # Identity
        name = self.nickname or self.user_label or self.id
        status_parts.append(f"{name} ({self.id})")

        # Timing
        if self.last_imaged:
            mins_ago = (datetime.now() - self.last_imaged).seconds // 60
            status_parts.append(f"last imaged {mins_ago}min ago")
        else:
            status_parts.append("not yet imaged")

        # Status
        if self.hatching_status.get('hatched'):
            status_parts.append(f"hatched at t{self.hatching_status['timepoint']:04d}")
        elif self.should_skip:
            status_parts.append(f"skipped ({self.skip_reason})")
        else:
            status_parts.append("active")

        # Current params
        status_parts.append(f"interval={self.interval_seconds}s")
        status_parts.append(f"slices={self.num_slices}")
        status_parts.append(f"priority={self.priority}")

        return " | ".join(status_parts)

    def record_exposure(self, exposure_ms: float, num_frames: int = 1):
        """
        Record light exposure for phototoxicity tracking.

        Parameters
        ----------
        exposure_ms : float
            Exposure time per frame in milliseconds
        num_frames : int
            Number of frames captured (1 for snap, num_slices for volume)
        """
        self.exposure_count += 1
        self.total_exposure_ms += exposure_ms * num_frames

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

    def to_dict(self) -> Dict:
        """Serialize for API responses"""
        return {
            'id': self.id,
            'nickname': self.nickname,
            'user_label': self.user_label,
            'stage_position': self.stage_position,
            'calibration': self.calibration,
            'detection_confidence': self.detection_confidence,
            'interval_seconds': self.interval_seconds,
            'num_slices': self.num_slices,
            'exposure_ms': self.exposure_ms,
            'priority': self.priority,
            'last_imaged': self.last_imaged.isoformat() if self.last_imaged else None,
            'timepoints_acquired': self.timepoints_acquired,
            'should_skip': self.should_skip,
            'skip_reason': self.skip_reason,
            'exposure_count': self.exposure_count,
            'total_exposure_ms': self.total_exposure_ms,
            'hatching_status': self.hatching_status,
            'recent_analyses': {
                'morphology': self.morphology_history[-5:] if self.morphology_history else [],
                'fluorescence': self.fluorescence_history[-5:] if self.fluorescence_history else [],
                'custom': self.custom_classifications,
            },
            'focus_history': [fp.to_dict() for fp in self.focus_history],
        }


class ExperimentState:
    """Global experiment state"""

    def __init__(self):
        self.embryos: Dict[str, EmbryoState] = {}
        self.start_time: Optional[datetime] = None
        self.acquisition_status: str = "idle"  # idle/running/paused/completed
        self.current_plan_name: Optional[str] = None
        self.plan_history: List[Dict] = []
        self.metadata: Dict = {}

    def add_embryo(self, embryo_id: str, position: Dict = None,
                   calibration: Dict = None, user_label: Optional[str] = None,
                   confidence: float = 0.0):
        """Register new embryo"""
        # Auto-start experiment when first embryo is added
        if self.start_time is None:
            self.start_time = datetime.now()

        self.embryos[embryo_id] = EmbryoState(
            id=embryo_id,
            stage_position=position or {},
            calibration=calibration or {},
            user_label=user_label,
            detection_confidence=confidence
        )

    def remove_embryo(self, embryo_id: str) -> bool:
        """Remove embryo from experiment (e.g., false detection)"""
        if embryo_id in self.embryos:
            del self.embryos[embryo_id]
            return True
        return False

    def assign_nickname(self, embryo_id: str, nickname: str):
        """Agent assigns intuitive name"""
        if embryo_id in self.embryos:
            self.embryos[embryo_id].nickname = nickname

    def get_embryo_by_any_name(self, name: str) -> Optional[EmbryoState]:
        """Get embryo by ID, nickname, or user label"""
        # Direct ID match
        if name in self.embryos:
            return self.embryos[name]

        # Search by nickname or label
        for embryo in self.embryos.values():
            if embryo.nickname == name or embryo.user_label == name:
                return embryo

        # Try extracting number from name like "embryo 3" -> "embryo_3"
        match = re.search(r'(\d+)', name)
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
            "Per-embryo status:"
        ]

        for embryo in sorted(self.embryos.values(), key=lambda e: e.id):
            lines.append(f"  {embryo.to_summary()}")

        if self.current_plan_name:
            lines.append(f"\nCurrent plan: {self.current_plan_name}")

        return "\n".join(lines)

    def to_dict(self) -> Dict:
        """Serialize for API responses"""
        return {
            'start_time': self.start_time.isoformat() if self.start_time else None,
            'acquisition_status': self.acquisition_status,
            'current_plan_name': self.current_plan_name,
            'embryo_count': len(self.embryos),
            'embryos': {eid: e.to_dict() for eid, e in self.embryos.items()},
            'metadata': self.metadata
        }
