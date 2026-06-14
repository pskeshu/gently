"""Domain-specific exception hierarchy for the Gently system.

Base:
    GentlyError
    ├── HardwareError          — Physical device communication failures
    │   ├── DeviceNotFoundError
    │   ├── DeviceTimeoutError
    │   ├── StageMovementError
    │   └── AcquisitionError
    ├── CalibrationError       — Piezo-galvo calibration failures
    │   ├── FocusFitError      — Gaussian/parabolic fit failed
    │   ├── EdgeDetectionError — Embryo edge not found
    │   └── CalibrationQualityError — R² below threshold
    ├── PerceptionError        — VLM stage classification failures
    │   ├── StageClassificationError
    │   └── VerificationError
    ├── StorageError           — Data persistence failures
    │   ├── SessionNotFoundError
    │   └── VolumeNotFoundError
    ├── NetworkError           — Inter-service communication failures
    │   ├── DeviceLayerError   — Device layer HTTP API errors
    │   ├── MeshPeerError      — Mesh peer communication errors
    │   └── ServiceUnavailableError
    └── AgentError             — Agent/conversation failures
        ├── ToolExecutionError
        └── PlanSynthesisError
"""


class GentlyError(Exception):
    """Base exception for all Gently errors."""

    pass


class HardwareError(GentlyError):
    """Physical device communication failure."""

    pass


class DeviceNotFoundError(HardwareError):
    """A requested device was not found in the hardware configuration."""

    pass


class DeviceTimeoutError(HardwareError):
    """Device operation timed out."""

    pass


class StageMovementError(HardwareError):
    """Stage movement failed or was out of range."""

    pass


class AcquisitionError(HardwareError):
    """Image or volume acquisition failed."""

    pass


class CalibrationError(GentlyError):
    """Piezo-galvo calibration failure."""

    pass


class FocusFitError(CalibrationError):
    """Focus curve fitting failed (bad R², insufficient data, etc.)."""

    pass


class EdgeDetectionError(CalibrationError):
    """Embryo edge detection failed (VLM couldn't find boundary)."""

    pass


class CalibrationQualityError(CalibrationError):
    """Calibration quality below acceptable threshold."""

    pass


class PerceptionError(GentlyError):
    """VLM perception/classification failure."""

    pass


class StageClassificationError(PerceptionError):
    """Could not classify developmental stage."""

    pass


class VerificationError(PerceptionError):
    """Verification subagent disagreed or failed."""

    pass


class StorageError(GentlyError):
    """Data persistence failure."""

    pass


class SessionNotFoundError(StorageError):
    """Requested session does not exist."""

    pass


class VolumeNotFoundError(StorageError):
    """Requested volume does not exist."""

    pass


class NetworkError(GentlyError):
    """Inter-service communication failure."""

    pass


class DeviceLayerError(NetworkError):
    """Device layer HTTP API returned an error."""

    pass


class MeshPeerError(NetworkError):
    """Mesh peer communication failed."""

    pass


class ServiceUnavailableError(NetworkError):
    """Required service is not running."""

    pass


class AgentError(GentlyError):
    """Agent/conversation failure."""

    pass


class ToolExecutionError(AgentError):
    """A tool call failed during execution."""

    pass


class PlanSynthesisError(AgentError):
    """Plan synthesis (natural language → Bluesky plan) failed."""

    pass
