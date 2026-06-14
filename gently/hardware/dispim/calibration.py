"""
diSPIM-specific calibration prior.

The CalibrationPrior tracks the piezo-galvo linear relationship across
embryos within a session, enabling fast calibration for subsequent embryos
once the slope is established from the first.

This is diSPIM-specific because the piezo-galvo coupling is unique to the
dual-view light-sheet architecture. Other hardware (e.g., 2P, confocal)
would have different calibration models.
"""

from dataclasses import dataclass
from datetime import datetime


@dataclass
class CalibrationPrior:
    """
    Session-level calibration prior learned from previously calibrated embryos.

    Enables informed initialization for subsequent embryos, reducing calibration
    time by using cross-embryo learning. The prior is updated after each
    successful calibration using an exponential moving average.
    """

    # Linear relationship: piezo = slope * galvo + offset
    slope_um_per_deg: float = 100.0  # Default heuristic
    offset_um: float = 0.0

    # Confidence metrics
    r_squared_mean: float = 0.0  # Average R-squared from contributing calibrations
    num_calibrations: int = 0  # Number of embryos contributing to prior

    # Observed ranges (for adaptive sweep window sizing)
    slope_min: float = 90.0
    slope_max: float = 110.0
    offset_min: float = -20.0
    offset_max: float = 20.0

    # Edge detection statistics
    typical_extent_deg: float = 0.3  # Average embryo Z extent in degrees
    extent_std_deg: float = 0.1  # Variation in extent

    # Timestamp for staleness checking
    last_updated: datetime | None = None

    # Fast calibration: lock slope after first embryo bootstrap
    session_slope_locked: bool = False
    bootstrap_embryo_id: str | None = None  # Which embryo established the slope

    def lock_session_slope(self, slope: float, r_squared: float, embryo_id: str):
        """
        Lock the session slope after first embryo bootstrap calibration.

        Once locked, subsequent embryos will use this slope and only
        calibrate their individual offset.

        Parameters
        ----------
        slope : float
            Calibrated slope from bootstrap embryo (µm/deg)
        r_squared : float
            Fit quality from bootstrap calibration
        embryo_id : str
            ID of the embryo used for bootstrap
        """
        self.slope_um_per_deg = slope
        self.r_squared_mean = r_squared
        self.session_slope_locked = True
        self.bootstrap_embryo_id = embryo_id
        self.num_calibrations = 1
        self.last_updated = datetime.now()

    def is_ready_for_fast_calibration(self) -> bool:
        """Check if session slope is locked and ready for fast per-embryo calibration."""
        return self.session_slope_locked and self.r_squared_mean >= 0.75

    def update_from_calibration(
        self,
        slope: float,
        offset: float,
        r_squared: float,
        extent_deg: float,
        alpha: float = 0.3,
    ):
        """
        Update prior with new calibration result using exponential moving average.

        Parameters
        ----------
        slope : float
            Calibrated slope (µm/deg)
        offset : float
            Calibrated offset (µm)
        r_squared : float
            Average R-squared from top/bottom calibration
        extent_deg : float
            Detected embryo Z extent in degrees
        alpha : float
            Weighting for new data (higher = faster adaptation)
        """
        if self.num_calibrations == 0:
            # First calibration - use directly
            self.slope_um_per_deg = slope
            self.offset_um = offset
            self.r_squared_mean = r_squared
            self.typical_extent_deg = extent_deg
        else:
            # Exponential moving average
            self.slope_um_per_deg = alpha * slope + (1 - alpha) * self.slope_um_per_deg
            self.offset_um = alpha * offset + (1 - alpha) * self.offset_um
            self.r_squared_mean = alpha * r_squared + (1 - alpha) * self.r_squared_mean
            self.typical_extent_deg = alpha * extent_deg + (1 - alpha) * self.typical_extent_deg

        # Update ranges (expand if needed)
        self.slope_min = min(self.slope_min, slope - 5)
        self.slope_max = max(self.slope_max, slope + 5)
        self.offset_min = min(self.offset_min, offset - 5)
        self.offset_max = max(self.offset_max, offset + 5)

        self.num_calibrations += 1
        self.last_updated = datetime.now()

    def to_dict(self) -> dict:
        """Serialize for JSON storage"""
        return {
            "slope_um_per_deg": self.slope_um_per_deg,
            "offset_um": self.offset_um,
            "r_squared_mean": self.r_squared_mean,
            "num_calibrations": self.num_calibrations,
            "slope_min": self.slope_min,
            "slope_max": self.slope_max,
            "offset_min": self.offset_min,
            "offset_max": self.offset_max,
            "typical_extent_deg": self.typical_extent_deg,
            "extent_std_deg": self.extent_std_deg,
            "last_updated": self.last_updated.isoformat() if self.last_updated else None,
            "session_slope_locked": self.session_slope_locked,
            "bootstrap_embryo_id": self.bootstrap_embryo_id,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "CalibrationPrior":
        """Deserialize from JSON"""
        prior = cls(
            slope_um_per_deg=data.get("slope_um_per_deg", 100.0),
            offset_um=data.get("offset_um", 0.0),
            r_squared_mean=data.get("r_squared_mean", 0.0),
            num_calibrations=data.get("num_calibrations", 0),
            slope_min=data.get("slope_min", 90.0),
            slope_max=data.get("slope_max", 110.0),
            offset_min=data.get("offset_min", -20.0),
            offset_max=data.get("offset_max", 20.0),
            typical_extent_deg=data.get("typical_extent_deg", 0.3),
            extent_std_deg=data.get("extent_std_deg", 0.1),
            session_slope_locked=data.get("session_slope_locked", False),
            bootstrap_embryo_id=data.get("bootstrap_embryo_id"),
        )
        if data.get("last_updated"):
            prior.last_updated = datetime.fromisoformat(data["last_updated"])
        return prior
