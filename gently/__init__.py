"""
gently - A Python library for DiSPIM microscope control

Provides clean, Ophyd-style device interfaces for DiSPIM systems
based on Micro-Manager configuration groups.
"""

from .devices import (
    DiSPIMCore,
    DeviceKeys,
    LaserControl,
    LEDControl, 
    SystemControl,
    PiezoControl,
    CameraControl
)

__version__ = "0.1.0"
__all__ = [
    "DiSPIMCore",
    "DeviceKeys",
    "LaserControl",
    "LEDControl",
    "SystemControl", 
    "PiezoControl",
    "CameraControl"
]