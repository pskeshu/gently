"""
DiSPIM Device Control Module

Ophyd-style device classes for DiSPIM microscope control based on ASI DiSPIM plugin architecture.
Provides signal-based interfaces with timestamps and metadata.
"""

import os
import time
import logging
from typing import Dict, Optional, List, Any, Union
from abc import ABC, abstractmethod
import numpy as np
import pymmcore


class DeviceKeys:
    """Device key mappings based on ASI DiSPIM plugin architecture"""
    # Cameras
    CAMERA_A = "HamCam1"              # Side A camera  
    CAMERA_B = "HamCam2"              # Side B camera
    CAMERA_BOTTOM = "Bottom PCO"       # Bottom camera
    MULTI_CAMERA = "Multi Camera"      # Multi-camera device
    
    # Piezo stages
    PIEZO_A = "PiezoStage:P:34"       # Side A imaging piezo
    PIEZO_B = "PiezoStage:Q:35"       # Side B imaging piezo
    
    # Galvo/Scanner devices  
    GALVO_A = "Scanner:AB:33"         # Side A scanner
    GALVO_B = "Scanner:CD:33"         # Side B scanner
    
    # Stage devices
    XY_STAGE = "XYStage:XY:31"        # XY positioning stage
    LOWER_Z = "ZStage:Z:32"           # Lower Z drive
    UPPER_Z = "ZStage:V:37"           # Upper Z drive
    
    # Control devices
    PLOGIC = "PLogic:E:36"            # Programmable logic for shutters/lasers
    LED = "LED:X:31"                  # LED illumination
    TIGER_COMM = "TigerCommHub"       # ASI Tiger controller hub
    
    # Laser control
    COHERENT_REMOTE = "Coherent-Scientific Remote"  # Laser controller


class MMSignal:
    """Micro-Manager signal that provides Ophyd-like interface with timestamps"""
    
    def __init__(self, name: str, device_name: str = None, property_name: str = None, 
                 units: str = "", dtype: str = "number", shape: List = None):
        self.name = name
        self.device_name = device_name
        self.property_name = property_name
        self.units = units
        self.dtype = dtype
        self.shape = shape or []
        self._last_value = None
        self._last_timestamp = None
    
    def read(self) -> Dict[str, Dict[str, Any]]:
        """Read signal value with timestamp"""
        # Subclasses should override this to get actual values
        return {
            self.name: {
                'value': self._last_value,
                'timestamp': self._last_timestamp or time.time()
            }
        }
    
    def describe(self) -> Dict[str, Dict[str, Any]]:
        """Describe signal metadata"""
        desc = {
            self.name: {
                'dtype': self.dtype,
                'shape': self.shape,
                'source': f'MM:{self.device_name}' if self.device_name else 'MM:core'
            }
        }
        if self.units:
            desc[self.name]['units'] = self.units
        return desc
    
    def update_value(self, value: Any):
        """Update cached value with timestamp"""
        self._last_value = value
        self._last_timestamp = time.time()


class MMPositionSignal(MMSignal):
    """Signal for stage position values"""
    
    def __init__(self, name: str, device_name: str, core: 'DiSPIMCore'):
        super().__init__(name, device_name, units="um", dtype="number")
        self.core = core
    
    def read(self) -> Dict[str, Dict[str, Any]]:
        """Read current position from MM"""
        try:
            value = self.core.get_position(self.device_name)
            self.update_value(value)
        except Exception as e:
            logging.getLogger(__name__).error(f"Error reading {self.name}: {e}")
        
        return super().read()


class MMPropertySignal(MMSignal):
    """Signal for reading actual device properties"""
    
    def __init__(self, name: str, device_name: str, property_name: str, core: 'DiSPIMCore', **kwargs):
        super().__init__(name, device_name, property_name, **kwargs)
        self.core = core
    
    def read(self) -> Dict[str, Dict[str, Any]]:
        """Read current property value from MM device"""
        try:
            value = self.core.mmc.getProperty(self.device_name, self.property_name)
            self.update_value(value)
        except Exception as e:
            logging.getLogger(__name__).error(f"Error reading {self.name}: {e}")
            # Keep last known value if read fails
        
        return super().read()


class DiSPIMCore:
    """Core DiSPIM control class - wrapper around pymmcore with device management"""
    
    def __init__(self, mm_dir: str, config_path: str, enable_logging: bool = True):
        """
        Initialize DiSPIM core with Micro-Manager setup
        
        Args:
            mm_dir: Path to Micro-Manager installation directory
            config_path: Path to Micro-Manager configuration file
            enable_logging: Enable MM stderr logging
        """
        self.mm_dir = mm_dir
        self.config_path = config_path
        self.logger = logging.getLogger(__name__)
        
        # Initialize pymmcore
        self.mmc = pymmcore.CMMCore()
        if enable_logging:
            self.mmc.enableStderrLog(True)
        
        # Setup MM environment and load configuration
        self._setup_micromanager()
        self._load_configuration()
        
        # Device mapping based on loaded devices
        self.devices = self._discover_devices()
        
        self.logger.info(f"DiSPIM Core initialized with {len(self.devices)} devices")
    
    def _setup_micromanager(self):
        """Setup Micro-Manager paths and environment"""
        # Add MM directory to PATH (needed on Windows)
        os.environ["PATH"] += os.pathsep.join(["", self.mm_dir])
        
        # Set device adapter search paths
        self.mmc.setDeviceAdapterSearchPaths([self.mm_dir])
        
        self.logger.info(f"MM directory set to: {self.mm_dir}")
    
    def _load_configuration(self):
        """Load Micro-Manager system configuration"""
        if not os.path.exists(self.config_path):
            raise FileNotFoundError(f"Configuration file not found: {self.config_path}")
        
        try:
            self.mmc.loadSystemConfiguration(self.config_path)
            self.logger.info(f"Loaded configuration: {self.config_path}")
        except Exception as e:
            raise RuntimeError(f"Failed to load MM configuration: {e}")
    
    def _discover_devices(self) -> Dict[str, str]:
        """Discover and map loaded MM devices"""
        loaded_devices = list(self.mmc.getLoadedDevices())
        device_map = {}
        
        # Map known device keys to actual loaded devices
        for key, device_name in vars(DeviceKeys).items():
            if not key.startswith('_') and device_name in loaded_devices:
                device_map[key] = device_name
        
        self.logger.info(f"Discovered devices: {list(device_map.keys())}")
        return device_map
    
    def get_device(self, device_key: str) -> Optional[str]:
        """Get actual MM device name for a device key"""
        return self.devices.get(device_key)
    
    def get_available_config_groups(self) -> List[str]:
        """Get list of available configuration groups"""
        return list(self.mmc.getAvailableConfigGroups())
    
    def get_available_configs(self, group_name: str) -> List[str]:
        """Get available configurations for a group"""
        return list(self.mmc.getAvailableConfigs(group_name))
    
    def set_config(self, group_name: str, config_name: str):
        """Set configuration for a group"""
        try:
            self.mmc.setConfig(group_name, config_name)
            self.logger.debug(f"Set config {group_name}:{config_name}")
        except Exception as e:
            self.logger.error(f"Failed to set config {group_name}:{config_name}: {e}")
            raise
    
    def snap_image(self) -> np.ndarray:
        """Capture a single image from current camera"""
        try:
            self.mmc.snapImage()
            return self.mmc.getImage()
        except Exception as e:
            self.logger.error(f"Failed to snap image: {e}")
            raise
    
    def get_position(self, device_name: str) -> float:
        """Get position of a stage device"""
        try:
            return self.mmc.getPosition(device_name)
        except Exception as e:
            self.logger.error(f"Failed to get position for {device_name}: {e}")
            raise
    
    def set_position(self, device_name: str, position: float):
        """Set position of a stage device"""
        try:
            self.mmc.setPosition(device_name, position)
            self.mmc.waitForDevice(device_name)
            self.logger.debug(f"Set {device_name} to position {position}")
        except Exception as e:
            self.logger.error(f"Failed to set {device_name} position: {e}")
            raise


class LaserControl:
    """Control class for laser devices with Ophyd-like interface"""
    
    def __init__(self, core: DiSPIMCore):
        self.core = core
        self.group_name = "Laser"
        self.logger = logging.getLogger(__name__)
        
        # Get PLogic device that controls lasers (from MM config)
        self.plogic_device = core.get_device("PLOGIC")
        
        # Create signal for reading actual laser output channel state
        self.signals = {}
        if self.plogic_device:
            self.signals['laser_output'] = MMPropertySignal(
                "laser_output", self.plogic_device, "OutputChannel", core, dtype="string")
        
        # Also read individual laser states from coherent controller if available
        self.coherent_device = core.get_device("COHERENT_REMOTE")
        if self.coherent_device:
            # Add signals for individual laser states
            laser_names = ["405-100C", "488-100C", "637-140C", "OBIS LS 561-100"]
            for laser in laser_names:
                signal_name = f"laser_{laser.split('-')[0].split()[0].lower()}_state"
                property_name = f"Laser {laser} - State"
                try:
                    self.signals[signal_name] = MMPropertySignal(
                        signal_name, self.coherent_device, property_name, core, dtype="string")
                except:
                    # Skip if property doesn't exist
                    pass
        
        if not self.signals:
            self.logger.warning("No laser control devices found")
        
        # Verify laser config group exists for control actions
        if self.group_name not in self.core.get_available_config_groups():
            self.logger.warning(f"Laser config group not found")
    
    def read(self) -> Dict[str, Dict[str, Any]]:
        """Read current laser states from actual devices"""
        result = {}
        for signal in self.signals.values():
            try:
                result.update(signal.read())
            except Exception as e:
                self.logger.debug(f"Could not read laser signal {signal.name}: {e}")
        return result
    
    def describe(self) -> Dict[str, Dict[str, Any]]:
        """Describe laser control signals"""
        result = {}
        for signal in self.signals.values():
            result.update(signal.describe())
        return result
    
    def set_405_only(self):
        """Set laser to 405nm only"""
        self.core.set_config(self.group_name, "405 only ")
    
    def set_488_only(self):
        """Set laser to 488nm only"""
        self.core.set_config(self.group_name, "488 only")
    
    def set_561_only(self):
        """Set laser to 561nm only"""  
        self.core.set_config(self.group_name, "561 only")
    
    def set_637_only(self):
        """Set laser to 637nm only"""
        self.core.set_config(self.group_name, "637 only")
    
    def set_488_and_561(self):
        """Set laser to 488nm and 561nm"""
        self.core.set_config(self.group_name, "488 and 561")
    
    def all_on(self):
        """Turn on all lasers"""
        self.core.set_config(self.group_name, "ALL ON")
    
    def all_off(self):
        """Turn off all lasers"""
        self.core.set_config(self.group_name, "ALL OFF")
    
    def get_available_configs(self) -> List[str]:
        """Get available laser configurations"""
        return self.core.get_available_configs(self.group_name)


class LEDControl:
    """Control class for LED device with Ophyd-like interface"""
    
    def __init__(self, core: DiSPIMCore):
        self.core = core
        self.group_name = "LED"
        self.logger = logging.getLogger(__name__)
        
        # Get actual LED device name
        self.led_device = core.get_device("LED")
        
        # Create signal for reading actual LED state
        if self.led_device:
            self.led_state = MMPropertySignal("led_state", self.led_device, "State", core, dtype="string")
        else:
            self.led_state = None
            self.logger.warning("LED device not found")
        
        # Verify LED config group exists for control actions
        if self.group_name not in self.core.get_available_config_groups():
            self.logger.warning(f"LED config group not found")
    
    def read(self) -> Dict[str, Dict[str, Any]]:
        """Read current LED state from actual device"""
        if self.led_state:
            return self.led_state.read()
        else:
            return {}
    
    def describe(self) -> Dict[str, Dict[str, Any]]:
        """Describe LED control signals"""
        if self.led_state:
            return self.led_state.describe()
        else:
            return {}
    
    def open(self):
        """Open LED shutter using config group"""
        self.core.set_config(self.group_name, "Open")
    
    def close(self):
        """Close LED shutter using config group"""
        self.core.set_config(self.group_name, "Closed")
    
    def get_available_configs(self) -> List[str]:
        """Get available LED configurations"""
        return self.core.get_available_configs(self.group_name)


class SystemControl:
    """Control class for system configuration - action-only interface"""
    
    def __init__(self, core: DiSPIMCore):
        self.core = core
        self.group_name = "System"
        self.logger = logging.getLogger(__name__)
        
        # System configs are action-only, but we can track key device states
        self.signals = {}
        
        # Read piezo motor states (from system config)
        piezo_a = core.get_device("PIEZO_A")
        piezo_b = core.get_device("PIEZO_B")
        
        if piezo_a:
            self.signals['piezo_a_motor'] = MMPropertySignal(
                "piezo_a_motor", piezo_a, "MotorOnOff", core, dtype="string")
        if piezo_b:
            self.signals['piezo_b_motor'] = MMPropertySignal(
                "piezo_b_motor", piezo_b, "MotorOnOff", core, dtype="string")
        
        # Verify system config group exists
        if self.group_name not in self.core.get_available_config_groups():
            self.logger.warning(f"System config group not found")
    
    def read(self) -> Dict[str, Dict[str, Any]]:
        """Read system device states"""
        result = {}
        for signal in self.signals.values():
            try:
                result.update(signal.read())
            except Exception as e:
                self.logger.debug(f"Could not read system signal {signal.name}: {e}")
        return result
    
    def describe(self) -> Dict[str, Dict[str, Any]]:
        """Describe system control signals"""
        result = {}
        for signal in self.signals.values():
            result.update(signal.describe())
        return result
    
    def startup(self):
        """Apply startup configuration"""
        try:
            self.core.set_config(self.group_name, "Startup")
            self.logger.info("Startup configuration applied")
        except Exception as e:
            self.logger.error(f"Could not apply startup configuration: {e}")
            raise
    
    def shutdown(self):
        """Apply shutdown configuration"""
        try:
            self.core.set_config(self.group_name, "Shutdown")
            self.logger.info("Shutdown configuration applied")
        except Exception as e:
            self.logger.error(f"Could not apply shutdown configuration: {e}")
            raise
    
    def get_available_configs(self) -> List[str]:
        """Get available system configurations"""
        return self.core.get_available_configs(self.group_name)


class PiezoControl:
    """Control class for piezo stage devices with Ophyd-like interface"""
    
    def __init__(self, core: DiSPIMCore):
        self.core = core
        self.logger = logging.getLogger(__name__)
        
        # Get piezo device names
        self.piezo_a_device = core.get_device("PIEZO_A")
        self.piezo_b_device = core.get_device("PIEZO_B")
        
        # Create position signals
        self.signals = {}
        if self.piezo_a_device:
            self.signals['piezo_a_position'] = MMPositionSignal(
                "piezo_a_position", self.piezo_a_device, core)
        if self.piezo_b_device:
            self.signals['piezo_b_position'] = MMPositionSignal(
                "piezo_b_position", self.piezo_b_device, core)
        
        if not self.signals:
            self.logger.warning("No piezo devices found")
    
    def read(self) -> Dict[str, Dict[str, Any]]:
        """Read all piezo positions with timestamps"""
        result = {}
        for signal in self.signals.values():
            result.update(signal.read())
        return result
    
    def describe(self) -> Dict[str, Dict[str, Any]]:
        """Describe piezo control signals"""
        result = {}
        for signal in self.signals.values():
            result.update(signal.describe())
        return result
    
    def set_position_a(self, position: float):
        """Set position of piezo A (P:34)"""
        if self.piezo_a_device:
            self.core.set_position(self.piezo_a_device, position)
            self.signals['piezo_a_position'].update_value(position)
        else:
            raise RuntimeError("Piezo A device not available")
    
    def set_position_b(self, position: float):
        """Set position of piezo B (Q:35)"""
        if self.piezo_b_device:
            self.core.set_position(self.piezo_b_device, position)
            self.signals['piezo_b_position'].update_value(position)
        else:
            raise RuntimeError("Piezo B device not available")
    
    def get_position_a(self) -> float:
        """Get position of piezo A"""
        if 'piezo_a_position' in self.signals:
            data = self.signals['piezo_a_position'].read()
            return data['piezo_a_position']['value']
        else:
            raise RuntimeError("Piezo A device not available")
    
    def get_position_b(self) -> float:
        """Get position of piezo B"""
        if 'piezo_b_position' in self.signals:
            data = self.signals['piezo_b_position'].read()
            return data['piezo_b_position']['value']
        else:
            raise RuntimeError("Piezo B device not available")


class CameraControl:
    """Control class for camera devices with Ophyd-like interface"""
    
    def __init__(self, core: DiSPIMCore):
        self.core = core
        self.logger = logging.getLogger(__name__)
        
        # Get camera device names
        self.camera_a = core.get_device("CAMERA_A")
        self.camera_b = core.get_device("CAMERA_B") 
        self.camera_bottom = core.get_device("CAMERA_BOTTOM")
        
        # Signals for image data and metadata
        self.image_signal = MMSignal("image", dtype="array", shape=[])
        self.exposure_signal = MMSignal("exposure", units="ms", dtype="number")
        
        if not (self.camera_a or self.camera_b or self.camera_bottom):
            self.logger.warning("No camera devices found")
    
    def read(self) -> Dict[str, Dict[str, Any]]:
        """Read image data and camera settings with timestamps"""
        result = {}
        result.update(self.image_signal.read())
        result.update(self.exposure_signal.read())
        return result
    
    def describe(self) -> Dict[str, Dict[str, Any]]:
        """Describe camera signals"""
        result = {}
        result.update(self.image_signal.describe())
        result.update(self.exposure_signal.describe())
        return result
    
    def set_camera(self, camera_name: str):
        """Set active camera device"""
        try:
            self.core.mmc.setCameraDevice(camera_name)
            self.logger.info(f"Set active camera to: {camera_name}")
        except Exception as e:
            self.logger.error(f"Failed to set camera {camera_name}: {e}")
            raise
    
    def snap_image(self, camera_name: Optional[str] = None) -> np.ndarray:
        """Snap image with specified camera and update signals"""
        if camera_name:
            self.set_camera(camera_name)
        
        image = self.core.snap_image()
        
        # Update signals
        self.image_signal.update_value(image)
        
        # Try to get exposure time
        try:
            exposure = self.core.mmc.getExposure()
            self.exposure_signal.update_value(exposure)
        except:
            pass
        
        return image
    
    def get_image_info(self, image: np.ndarray) -> Dict[str, Any]:
        """Get information about captured image"""
        return {
            "shape": image.shape,
            "dtype": str(image.dtype),
            "min": np.min(image),
            "max": np.max(image),
            "mean": np.mean(image)
        }