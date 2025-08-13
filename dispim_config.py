"""
DiSPIM Device Configuration Module

Python implementation of DiSPIM device control based on the Java ASIdiSPIM plugin.
Provides headless control using pymmcore-plus for DiSPIM microscope systems.
"""

import os
import logging
from typing import Dict, Optional, Tuple, List
from dataclasses import dataclass
from enum import Enum
import numpy as np

try:
    from pymmcore_plus import CMMCorePlus
except ImportError:
    raise ImportError("pymmcore-plus is required. Install with: pip install pymmcore-plus")


class DiSPIMMode(Enum):
    """DiSPIM acquisition modes"""
    SINGLE_VIEW = "single_view"
    DUAL_VIEW = "dual_view"
    STAGE_SCAN = "stage_scan"
    SLICE_SCAN = "slice_scan"


@dataclass
class DiSPIMConfig:
    """DiSPIM configuration parameters"""
    # Device names
    piezo_imaging: str = "PiezoImaging"
    piezo_illumination: str = "PiezoIllumination" 
    galvo_device: str = "GalvoDevice"
    camera_device: str = "Camera"
    xy_stage: str = "XYStage"
    
    # Calibration parameters
    piezo_center: float = 50.0  # μm
    galvo_center: float = 0.0   # degrees
    calibration_slope: float = 1.0
    calibration_offset: float = 0.0
    
    # Acquisition settings
    slice_step_size: float = 0.5  # μm
    exposure_time: float = 10.0   # ms
    num_slices: int = 100
    
    # Safety limits
    piezo_min: float = 0.0
    piezo_max: float = 100.0
    galvo_min: float = -10.0
    galvo_max: float = 10.0


class DiSPIMCore:
    """Core DiSPIM control class using pymmcore-plus"""
    
    def __init__(self, config_file: Optional[str] = None):
        self.core = CMMCorePlus.instance()
        self.config = DiSPIMConfig()
        self.logger = logging.getLogger(__name__)
        self._is_initialized = False
        self._devices_loaded = False
        
        if config_file:
            self.load_system_configuration(config_file)
    
    def load_system_configuration(self, config_file: str) -> None:
        """Load Micro-Manager system configuration file"""
        if not os.path.exists(config_file):
            raise FileNotFoundError(f"Configuration file not found: {config_file}")
        
        try:
            self.core.loadSystemConfiguration(config_file)
            self._devices_loaded = True
            self.logger.info(f"Loaded system configuration: {config_file}")
        except Exception as e:
            raise RuntimeError(f"Failed to load configuration: {e}")
    
    def initialize_devices(self) -> None:
        """Initialize DiSPIM devices and verify connectivity"""
        if not self._devices_loaded:
            raise RuntimeError("System configuration must be loaded first")
        
        # Get available devices
        devices = self.core.getLoadedDevices()
        self.logger.info(f"Available devices: {list(devices)}")
        
        # Verify required devices are present
        required_devices = [
            self.config.piezo_imaging,
            self.config.piezo_illumination,
            self.config.galvo_device,
            self.config.camera_device
        ]
        
        for device in required_devices:
            if device not in devices:
                self.logger.warning(f"Required device not found: {device}")
        
        # Initialize devices
        try:
            self.core.initializeAllDevices()
            self._is_initialized = True
            self.logger.info("All devices initialized successfully")
        except Exception as e:
            raise RuntimeError(f"Device initialization failed: {e}")
    
    def set_piezo_position(self, device: str, position: float) -> None:
        """Set piezo device position with safety checks"""
        if not self._is_initialized:
            raise RuntimeError("Devices must be initialized first")
        
        # Safety bounds checking
        if not (self.config.piezo_min <= position <= self.config.piezo_max):
            raise ValueError(f"Position {position} outside safe range "
                           f"[{self.config.piezo_min}, {self.config.piezo_max}]")
        
        try:
            self.core.setPosition(device, position)
            self.core.waitForDevice(device)
            self.logger.debug(f"Set {device} to position {position} μm")
        except Exception as e:
            raise RuntimeError(f"Failed to set {device} position: {e}")
    
    def set_galvo_position(self, position: float) -> None:
        """Set galvo/micromirror position with safety checks"""
        if not self._is_initialized:
            raise RuntimeError("Devices must be initialized first")
        
        # Safety bounds checking
        if not (self.config.galvo_min <= position <= self.config.galvo_max):
            raise ValueError(f"Position {position} outside safe range "
                           f"[{self.config.galvo_min}, {self.config.galvo_max}]")
        
        try:
            self.core.setPosition(self.config.galvo_device, position)
            self.core.waitForDevice(self.config.galvo_device)
            self.logger.debug(f"Set galvo to position {position} degrees")
        except Exception as e:
            raise RuntimeError(f"Failed to set galvo position: {e}")
    
    def get_device_position(self, device: str) -> float:
        """Get current device position"""
        if not self._is_initialized:
            raise RuntimeError("Devices must be initialized first")
        
        try:
            return self.core.getPosition(device)
        except Exception as e:
            raise RuntimeError(f"Failed to get {device} position: {e}")
    
    def center_piezo_and_galvo(self) -> None:
        """Move piezo and galvo devices to center positions"""
        self.logger.info("Centering piezo and galvo devices")
        self.set_piezo_position(self.config.piezo_imaging, self.config.piezo_center)
        self.set_piezo_position(self.config.piezo_illumination, self.config.piezo_center)
        self.set_galvo_position(self.config.galvo_center)
    
    def compute_galvo_from_piezo(self, piezo_position: float) -> float:
        """Compute galvo position from piezo position using calibration"""
        return (self.config.calibration_slope * piezo_position + 
                self.config.calibration_offset)
    
    def synchronized_move(self, piezo_pos: float) -> None:
        """Move piezo and galvo in synchronization using calibration"""
        galvo_pos = self.compute_galvo_from_piezo(piezo_pos)
        
        # Move devices simultaneously
        self.set_piezo_position(self.config.piezo_imaging, piezo_pos)
        self.set_galvo_position(galvo_pos)
    
    def snap_image(self) -> np.ndarray:
        """Capture a single image"""
        if not self._is_initialized:
            raise RuntimeError("Devices must be initialized first")
        
        try:
            self.core.snapImage()
            return self.core.getImage()
        except Exception as e:
            raise RuntimeError(f"Failed to snap image: {e}")
    
    def set_exposure(self, exposure_ms: float) -> None:
        """Set camera exposure time"""
        try:
            self.core.setExposure(exposure_ms)
            self.logger.debug(f"Set exposure to {exposure_ms} ms")
        except Exception as e:
            raise RuntimeError(f"Failed to set exposure: {e}")
    
    def get_device_info(self) -> Dict[str, Dict]:
        """Get information about all loaded devices"""
        if not self._devices_loaded:
            return {}
        
        devices = self.core.getLoadedDevices()
        info = {}
        
        for device in devices:
            try:
                device_type = self.core.getDeviceType(device)
                info[device] = {
                    'type': str(device_type),
                    'library': self.core.getDeviceLibrary(device),
                    'description': self.core.getDeviceDescription(device)
                }
            except Exception as e:
                info[device] = {'error': str(e)}
        
        return info
    
    def shutdown(self) -> None:
        """Safely shutdown the core and devices"""
        if self._is_initialized:
            try:
                self.center_piezo_and_galvo()
                self.core.reset()
                self.logger.info("DiSPIM core shutdown complete")
            except Exception as e:
                self.logger.error(f"Error during shutdown: {e}")


if __name__ == "__main__":
    # Example usage
    logging.basicConfig(level=logging.INFO)
    
    # Initialize DiSPIM
    dispim = DiSPIMCore()
    
    # Load configuration (replace with actual path)
    # dispim.load_system_configuration("/path/to/your/dispim_config.cfg")
    # dispim.initialize_devices()
    
    print("DiSPIM Core initialized successfully")
    print("Device info:", dispim.get_device_info())