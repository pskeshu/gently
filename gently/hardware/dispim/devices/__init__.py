"""
Ophyd device classes for diSPIM microscope control.

Split by domain for maintainability. Import from here for the public API:
    from gently.devices import DiSPIMCamera, DiSPIMPiezo, ...
"""

from .stage import DiSPIMZstage, DiSPIMXYStage
from .camera import DiSPIMCamera, DiSPIMDualCamera, DiSPIMBottomCamera
from .piezo import DiSPIMFDrive, DiSPIMPiezo
from .scanner import DiSPIMScanner
from .optical import DiSPIMLED, DiSPIMLightSource, DiSPIMLaserControl
from .acquisition import DiSPIMVolumeScanner, DiSPIMLightSheetSnap

__all__ = [
    "DiSPIMZstage", "DiSPIMXYStage",
    "DiSPIMCamera", "DiSPIMDualCamera", "DiSPIMBottomCamera",
    "DiSPIMFDrive", "DiSPIMPiezo",
    "DiSPIMScanner",
    "DiSPIMLED", "DiSPIMLightSource",
    # Backwards-compatible alias for DiSPIMLightSource:
    "DiSPIMLaserControl",
    "DiSPIMVolumeScanner", "DiSPIMLightSheetSnap",
]
