"""
Ophyd device classes for diSPIM microscope control.

Split by domain for maintainability. Import from here for the public API:
    from gently.devices import DiSPIMCamera, DiSPIMPiezo, ...
"""

from .acquisition import DiSPIMLightSheetSnap, DiSPIMVolumeScanner
from .camera import DiSPIMBottomCamera, DiSPIMCamera, DiSPIMDualCamera
from .optical import DiSPIMLaserControl, DiSPIMLED, DiSPIMLightSource
from .piezo import DiSPIMFDrive, DiSPIMPiezo
from .scanner import DiSPIMScanner
from .stage import DiSPIMXYStage, DiSPIMZstage

__all__ = [
    "DiSPIMZstage",
    "DiSPIMXYStage",
    "DiSPIMCamera",
    "DiSPIMDualCamera",
    "DiSPIMBottomCamera",
    "DiSPIMFDrive",
    "DiSPIMPiezo",
    "DiSPIMScanner",
    "DiSPIMLED",
    "DiSPIMLightSource",
    # Backwards-compatible alias for DiSPIMLightSource:
    "DiSPIMLaserControl",
    "DiSPIMVolumeScanner",
    "DiSPIMLightSheetSnap",
]
