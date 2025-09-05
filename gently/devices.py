"""
Gently DiSPIM Devices
====================

Ophyd device classes for DiSPIM microscope control with proper Bluesky integration.
Creates device-agnostic interfaces that work with standard Bluesky plan stubs.

Based on ASI DiSPIM plugin architecture but structured as proper Ophyd devices
for use with device-agnostic plans like:
    - bps.mv(piezo, position)
    - bps.trigger_and_read([camera])
    - focus_sweep(positioner, positions, detector)
"""

import time
import logging
from collections import OrderedDict
from typing import Dict, Tuple
import numpy as np


from ophyd import Device, DeviceStatus
from ophyd.status import AndStatus

import pymmcore



class DiSPIMPiezo(Device):
    """
    DiSPIM piezo positioner - works with bps.mv(piezo, position)
    
    Device-agnostic: any plan that moves a positioner will work with this device
    """
    
    def __init__(self, device_name: str, core: pymmcore.CMMCore, 
                 limits: Tuple[float, float] = (-50.0, 150.0), **kwargs):
        self.device_name = device_name
        self.core = core
        self._limits = limits
        self.tolerance = 0.1  # µm
        
        super().__init__(**kwargs)
    
    @property
    def limits(self):
        return self._limits
        
    def move(self, position, **kwargs):
        """Move piezo to position - called by bps.mv()"""
        position = float(position)
        
        # Safety check
        if not (self._limits[0] <= position <= self._limits[1]):
            raise ValueError(f"Position {position} outside limits {self._limits}")
        
        self.log.info(f"Moving {self.device_name} to {position} µm")
        
        # Direct MM core implementation like deepthought
        status = DeviceStatus(obj=self, timeout=10)

        def wait():
            try:
                self.core.setPosition(self.device_name, position)
                self.core.waitForDevice(self.device_name)
            except Exception as exc:
                status.set_exception(exc)
            else:
                status.set_finished()

        import threading
        threading.Thread(target=wait).start()

        return status
    
    def read(self):
        """Read current piezo position - required for Bluesky"""
        try:
            value = self.core.getPosition(self.device_name)
        except Exception as e:
            self.log.warning(f"Failed to read position from {self.device_name}: {e}")
            value = 0.0
                
        data = OrderedDict()
        data[self.device_name] = {
            'value': float(value),
            'timestamp': time.time()
        }
        return data
    
    def describe(self):
        """Describe piezo device - required for Bluesky"""
        data = OrderedDict()
        data[self.device_name] = {
            'source': self.device_name,
            'dtype': 'number',
            'shape': []
        }
        return data


class DiSPIMGalvo(Device):
    """
    DiSPIM galvanometer positioner - works with bps.mv(galvo, angle)
    
    Device-agnostic: any plan that moves a positioner will work with this device
    """
    
    def __init__(self, device_name: str, core: pymmcore.CMMCore,
                 limits: Tuple[float, float] = (-5.0, 5.0), **kwargs):
        self.device_name = device_name
        self.core = core
        self._limits = limits
        self.tolerance = 0.01  # degrees
        
        # Set this as the active galvo device in MM
        try:
            self.core.setGalvoDevice(self.device_name)
        except Exception as e:
            self.log.warning(f"Could not set galvo device {self.device_name}: {e}")
        
        super().__init__(**kwargs)
    
    @property
    def limits(self):
        return self._limits
        
    def move(self, position, **kwargs):
        """Move galvo to position - called by bps.mv()"""
        position = float(position)
        
        # Safety check
        if not (self._limits[0] <= position <= self._limits[1]):
            raise ValueError(f"Position {position} outside limits {self._limits}")
        
        self.log.info(f"Moving {self.device_name} to {position}°")
        
        # Direct MM core implementation using galvo APIs
        status = DeviceStatus(obj=self, timeout=10)

        def wait():
            try:
                self.core.setGalvoPosition(position)
                self.core.waitForDevice(self.device_name)
            except Exception as exc:
                status.set_exception(exc)
            else:
                status.set_finished()

        import threading
        threading.Thread(target=wait).start()

        return status
    
    def read(self):
        """Read current galvo position - required for Bluesky"""
        try:
            # Use proper MM galvo API
            value = self.core.getGalvoPosition()
        except Exception as e:
            self.log.warning(f"Failed to read galvo position from {self.device_name}: {e}")
            value = 0.0
                
        data = OrderedDict()
        data[self.device_name] = {
            'value': float(value),
            'timestamp': time.time()
        }
        return data
    
    def describe(self):
        """Describe galvo device - required for Bluesky"""
        data = OrderedDict()
        data[self.device_name] = {
            'source': self.device_name,
            'dtype': 'number',
            'shape': []
        }
        return data


class DiSPIMXYStage(Device):
    """
    DiSPIM XY stage - works with bps.mv(xy_stage, [x, y])
    
    Device-agnostic: any plan that moves XY positions will work with this device
    Based on deepthought XYStage implementation
    """
    
    def __init__(self, xy_device_name: str, core: pymmcore.CMMCore, **kwargs):
        self.xy_device_name = xy_device_name
        self.core = core
        
        super().__init__(**kwargs)
    
    def move(self, position):
        """Move XY stage to position [x, y] - called by bps.mv(xy_stage, [x, y])"""
        try:
            x, y = position  # Unpack [x, y] coordinates
            self.log.info(f"Moving XY stage to ({x}, {y})")
            
            # Set XY position using MM core
            self.core.setXYPosition(x, y)
            self.core.waitForDevice(self.xy_device_name)
            
            status = DeviceStatus(self)
            status.set_finished()
            return status
            
        except Exception as e:
            self.log.error(f"Failed to move XY stage: {e}")
            status = DeviceStatus(self)
            status.set_exception(e)
            return status
    
    def read(self):
        """Read current XY stage positions - required for Bluesky"""
        try:
            xy_pos = np.array(self.core.getXYPosition())
        except Exception as e:
            self.log.warning(f"Failed to read XY positions: {e}")
            xy_pos = np.array([0.0, 0.0])
        
        data = OrderedDict()
        data[self.xy_device_name] = {
            'value': xy_pos,
            'timestamp': time.time()
        }
        return data
    
    def describe(self):
        """Describe XY stage device - required for Bluesky"""
        data = OrderedDict()
        data[self.xy_device_name] = {
            'source': self.xy_device_name,
            'dtype': 'array',
            'shape': [2]
        }
        return data


class DiSPIMCamera(Device):
    """
    DiSPIM camera detector - works with bps.trigger_and_read([camera])
    
    Device-agnostic: any plan that acquires from a detector will work with this device
    """
    
    def __init__(self, device_name: str, core: pymmcore.CMMCore, **kwargs):
        super().__init__(**kwargs)
        self.device_name = device_name
        self.core = core
        self._acquiring = False
        self._last_image = None
        self._last_image_time = None
        
    def trigger(self):
        """Trigger image acquisition - called by bps.trigger()"""
        self.log.debug(f"Triggering {self.device_name}")
        
        def acquire_image():
            try:
                # Set camera and snap
                self.core.setCameraDevice(self.device_name)
                self.core.snapImage()
                self._last_image = self.core.getImage()
                self._last_image_time = time.time()
                self._acquiring = False
                return True
            except Exception as e:
                self.log.error(f"Image acquisition failed: {e}")
                self._acquiring = False
                return False
        
        self._acquiring = True
        
        # Run acquisition
        success = acquire_image()
        
        status = DeviceStatus(self)
        if success:
            status.set_finished()
        else:
            status.set_exception(RuntimeError("Image acquisition failed"))
        
        return status
    
    def read(self):
        """Read acquired image data - called by bps.read()"""
        if self._last_image is not None:
            data = OrderedDict()
            data[self.device_name] = {
                'value': self._last_image,
                'timestamp': self._last_image_time or time.time()
            }
            return data
        else:
            return OrderedDict()
    
    def describe(self):
        """Describe detector data format"""
        data = OrderedDict()
        data[self.device_name] = {
            'source': self.device_name,
            'dtype': 'array',
            'shape': getattr(self._last_image, 'shape', [])
        }
        return data
    
    @property
    def exposure_time(self):
        """Get current exposure time"""
        try:
            return self.core.getExposure() / 1000.0  # Convert ms to s
        except:
            return 0.01  # Default 10ms
    
    @exposure_time.setter 
    def exposure_time(self, value_s):
        """Set exposure time in seconds"""
        try:
            self.core.setExposure(value_s * 1000.0)  # Convert s to ms
        except Exception as e:
            self.log.error(f"Failed to set exposure: {e}")


class DiSPIMLaserControl(Device):
    """
    DiSPIM laser control - works with bps.mv(laser, 'config_name')
    
    Device-agnostic: any plan that sets configurations will work with this device
    """
    
    def __init__(self, core: pymmcore.CMMCore, **kwargs):
        self.core = core
        self.group_name = "Laser"
        
        super().__init__(**kwargs)
        
        # Cache available configs
        self._available_configs = self._get_available_configs()
    
    def _get_available_configs(self):
        """Get available laser configurations"""
        try:
            return list(self.core.getAvailableConfigs(self.group_name))
        except:
            return []
    
    def set_config(self, config_name: str):
        """Set laser configuration"""
        if config_name not in self._available_configs:
            raise ValueError(f"Config '{config_name}' not available. "
                           f"Available: {self._available_configs}")
        
        try:
            self.core.setConfig(self.group_name, config_name)
            self.log.info(f"Set laser config to: {config_name}")
        except Exception as e:
            self.log.error(f"Failed to set laser config: {e}")
            raise
    
    def read(self):
        """Read current laser configuration - required for Bluesky"""
        try:
            current_config = self.core.getCurrentConfig(self.group_name)
        except:
            current_config = 'unknown'
            
        data = OrderedDict()
        data[self.group_name] = {
            'value': current_config,
            'timestamp': time.time()
        }
        return data
    
    def describe(self):
        """Describe laser control device - required for Bluesky"""
        data = OrderedDict()
        data[self.group_name] = {
            'source': self.group_name,
            'dtype': 'string',
            'shape': []
        }
        return data


class DiSPIMLightSheet(Device):
    """
    Single-sided DiSPIM light sheet device
    
    Composite device containing all components needed for one side of DiSPIM imaging.
    Works with device-agnostic plans through individual component access.
    """
    
    def __init__(self, side: str, core: pymmcore.CMMCore, 
                 device_mapping: Dict[str, str], **kwargs):
        """
        Initialize single-sided light sheet
        
        Parameters
        ----------
        side : str
            'A' or 'B' for the two DiSPIM sides
        core : pymmcore.CMMCore
            MM core instance
        device_mapping : Dict[str, str]
            Mapping of logical names to actual MM device names
            e.g. {'piezo_a': 'PiezoStage:P:34', 'galvo_a': 'Scanner:AB:33', ...}
        """
        self.side = side
        self.core = core
        self.device_mapping = device_mapping
        
        # Map device names based on side
        piezo_key = f'piezo_{side.lower()}'
        galvo_key = f'galvo_{side.lower()}'
        camera_key = f'camera_{side.lower()}'
        
        super().__init__(**kwargs)
        
        # Create components with actual device names
        self.piezo = DiSPIMPiezo(
            device_mapping[piezo_key], core, name=f'piezo_{side}'
        )
        self.galvo = DiSPIMGalvo(
            device_mapping[galvo_key], core, name=f'galvo_{side}'
        )
        self.camera = DiSPIMCamera(
            device_mapping[camera_key], core, name=f'camera_{side}'
        )
        
        # Calibration parameters (will be set by calibration procedures)
        self.calibration_slope = 1.0
        self.calibration_offset = 0.0
        self.calibration_valid = False
    
    def synchronized_move(self, piezo_position: float):
        """
        Move piezo and galvo in synchronization based on calibration
        
        This is a DiSPIM-specific operation that can be used in plans:
        yield from light_sheet.synchronized_move(z_pos)
        """
        # Calculate galvo position from calibration
        galvo_position = (piezo_position - self.calibration_offset) / self.calibration_slope
        
        # Move both devices
        return AndStatus(
            self.piezo.move(piezo_position),
            self.galvo.move(galvo_position)
        )
    
    def stage(self):
        """Prepare device for use - saves current positions"""
        self._staged_piezo_pos = self.piezo.user_readback.get()
        self._staged_galvo_pos = self.galvo.user_readback.get()
        return super().stage()
    
    def unstage(self):
        """Return device to staged positions"""
        if hasattr(self, '_staged_piezo_pos') and hasattr(self, '_staged_galvo_pos'):
            # Return to staged positions
            return AndStatus(
                self.piezo.move(self._staged_piezo_pos),
                self.galvo.move(self._staged_galvo_pos)
            )
        return super().unstage()
    
    def read(self):
        """Read all light sheet device positions - required for Bluesky"""
        result = {}
        result.update(self.piezo.read())
        result.update(self.galvo.read())
        result.update(self.camera.read())
        return result
    
    def describe(self):
        """Describe light sheet device - required for Bluesky"""
        result = {}
        result.update(self.piezo.describe())
        result.update(self.galvo.describe())
        result.update(self.camera.describe())
        return result


class DiSPIMSystem(Device):
    """
    Complete dual-sided DiSPIM system
    
    Top-level device containing both light sheet sides and bottom camera.
    Foundation for complete DiSPIM workflows including embryo detection and acquisition.
    """
    
    def __init__(self, core: pymmcore.CMMCore, **kwargs):
        """
        Initialize complete DiSPIM system
        
        Parameters
        ----------
        core : pymmcore.CMMCore
            MM core instance
        """
        self.core = core
        
        # Device mapping based on ASI DiSPIM standard names
        device_mapping = {
            'piezo_a': 'PiezoStage:P:34',
            'galvo_a': 'Scanner:AB:33', 
            'camera_a': 'HamCam1',
            'piezo_b': 'PiezoStage:Q:35',
            'galvo_b': 'Scanner:CD:33',
            'camera_b': 'HamCam2'
        }
        
        super().__init__(**kwargs)
        
        # Create components
        self.side_a = DiSPIMLightSheet('A', core, device_mapping, name='side_a')
        self.side_b = DiSPIMLightSheet('B', core, device_mapping, name='side_b')
        self.xy_stage = DiSPIMXYStage('XYStage:XY:31', core, name='xy_stage')
        self.bottom_camera = DiSPIMCamera('Bottom PCO', core, name='bottom_camera')
        self.laser = DiSPIMLaserControl(core, name='laser')
    
    def center_all_devices(self):
        """Center all positioning devices - useful for initialization"""
        # Default center positions (can be overridden)
        piezo_center = 75.0  # µm
        galvo_center = 0.0   # degrees
        xy_center = 0.0      # µm
        
        return AndStatus(
            self.side_a.piezo.move(piezo_center),
            self.side_a.galvo.move(galvo_center),
            self.side_b.piezo.move(piezo_center), 
            self.side_b.galvo.move(galvo_center),
            self.xy_stage.move_xy(xy_center, xy_center)
        )
    
    def read(self):
        """Read all DiSPIM system devices - required for Bluesky"""
        result = {}
        result.update(self.side_a.read())
        result.update(self.side_b.read())
        result.update(self.xy_stage.read())
        result.update(self.bottom_camera.read())
        result.update(self.laser.read())
        return result
    
    def describe(self):
        """Describe complete DiSPIM system - required for Bluesky"""
        result = {}
        result.update(self.side_a.describe())
        result.update(self.side_b.describe())
        result.update(self.xy_stage.describe())
        result.update(self.bottom_camera.describe())
        result.update(self.laser.describe())
        return result


def create_dispim_system(mm_dir: str, config_file: str) -> DiSPIMSystem:
    """
    Factory function to create a configured DiSPIM system
    
    Parameters
    ----------
    mm_dir : str
        Path to Micro-Manager installation
    config_file : str
        Path to MM configuration file
        
    Returns
    -------
    DiSPIMSystem
        Fully configured DiSPIM system ready for Bluesky plans
    """
    import os
    
    # Initialize MM core
    core = pymmcore.CMMCore()
    core.enableStderrLog(True)
    
    # Setup MM environment
    os.environ["PATH"] += os.pathsep.join(["", mm_dir])
    core.setDeviceAdapterSearchPaths([mm_dir])
    
    # Load configuration
    if not os.path.exists(config_file):
        raise FileNotFoundError(f"Configuration file not found: {config_file}")
    
    core.loadSystemConfiguration(config_file)
    
    # Create system
    system = DiSPIMSystem(core, name='dispim_system')
    
    logging.getLogger(__name__).info("DiSPIM system created successfully")
    return system


if __name__ == "__main__":
    # Example usage - would normally use actual MM paths
    logging.basicConfig(level=logging.INFO)
   