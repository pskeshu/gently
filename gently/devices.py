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
from typing import Dict, Optional, List, Any, Union, Tuple
import numpy as np


from ophyd import Device, Component as Cpt, Signal, DeviceStatus
from ophyd import EpicsMotor, DetectorBase
from ophyd.signal import SignalRO, EpicsSignal
from ophyd.status import SubscriptionStatus, AndStatus

import pymmcore


class MMCoreSignal(Signal):
    """Signal that interfaces directly with pymmcore for DiSPIM devices"""
    
    def __init__(self, device_name: str, property_name: str = "Position", 
                 core: pymmcore.CMMCore = None, **kwargs):
        super().__init__(**kwargs)
        self.device_name = device_name
        self.property_name = property_name
        self.core = core
        self._last_value = 0.0
    
    def get(self):
        """Get current value from MM core"""
        if not self.core:
            return self._last_value
            
        try:
            if self.property_name == "Position":
                value = float(self.core.getPosition(self.device_name))
            else:
                value = self.core.getProperty(self.device_name, self.property_name)
                # Try to convert to float if possible
                try:
                    value = float(value)
                except (ValueError, TypeError):
                    pass
            
            self._last_value = value
            return value
        except Exception as e:
            self.log.warning(f"Failed to read {self.device_name}.{self.property_name}: {e}")
            return self._last_value
    
    def put(self, value, **kwargs):
        """Set value in MM core"""
        if not self.core:
            status = DeviceStatus(self)
            status.set_finished()
            return status
            
        try:
            if self.property_name == "Position":
                self.core.setPosition(self.device_name, float(value))
                self.core.waitForDevice(self.device_name)
            else:
                self.core.setProperty(self.device_name, self.property_name, str(value))
            
            self._last_value = value
            status = DeviceStatus(self)
            status.set_finished()
            return status
            
        except Exception as e:
            self.log.error(f"Failed to set {self.device_name}.{self.property_name}: {e}")
            status = DeviceStatus(self)
            status.set_exception(e)
            return status


class DiSPIMPiezo(Device):
    """
    DiSPIM piezo positioner - works with bps.mv(piezo, position)
    
    Device-agnostic: any plan that moves a positioner will work with this device
    """
    
    def __init__(self, device_name: str, core: pymmcore.CMMCore, 
                 limits: Tuple[float, float] = (-50.0, 150.0), **kwargs):
        self.device_name = device_name
        self.core = core
        self._limits = limits  # Use private attribute to avoid conflict with Ophyd
        self.tolerance = 0.1  # µm
        
        # Create signals directly as attributes
        self.position = MMCoreSignal(device_name, "Position", core, name='position')
        self.user_readback = MMCoreSignal(device_name, "Position", core, name='user_readback')
        
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
        
        # Start move
        status = self.position.put(position, **kwargs)
        
        def check_done():
            current = self.user_readback.get()
            return abs(current - position) < self.tolerance
        
        # Create subscription status that waits for move completion
        move_status = SubscriptionStatus(self.user_readback, check_done, timeout=10.0)
        
        return move_status
    
    def read(self):
        """Read current piezo position - required for Bluesky"""
        return {
            f'{self.name}_user_readback': {
                'value': self.user_readback.get(),
                'timestamp': time.time()
            }
        }
    
    def describe(self):
        """Describe piezo device - required for Bluesky"""
        return {
            f'{self.name}_user_readback': {
                'source': f'DiSPIM Piezo {self.device_name}',
                'dtype': 'number',
                'shape': [],
                'units': 'um'
            }
        }


class DiSPIMGalvo(Device):
    """
    DiSPIM galvanometer positioner - works with bps.mv(galvo, angle)
    
    Device-agnostic: any plan that moves a positioner will work with this device
    """
    
    def __init__(self, device_name: str, core: pymmcore.CMMCore,
                 limits: Tuple[float, float] = (-5.0, 5.0), **kwargs):
        self.device_name = device_name
        self.core = core
        self._limits = limits  # Use private attribute to avoid conflict with Ophyd
        self.tolerance = 0.01  # degrees
        
        # Create signals directly as attributes
        self.position = MMCoreSignal(device_name, "Position", core, name='position')
        self.user_readback = MMCoreSignal(device_name, "Position", core, name='user_readback')
        
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
        
        # Start move
        status = self.position.put(position, **kwargs)
        
        def check_done():
            current = self.user_readback.get()
            return abs(current - position) < self.tolerance
        
        move_status = SubscriptionStatus(self.user_readback, check_done, timeout=5.0)
        
        return move_status
    
    def read(self):
        """Read current galvo position - required for Bluesky"""
        return {
            f'{self.name}_user_readback': {
                'value': self.user_readback.get(),
                'timestamp': time.time()
            }
        }
    
    def describe(self):
        """Describe galvo device - required for Bluesky"""
        return {
            f'{self.name}_user_readback': {
                'source': f'DiSPIM Galvo {self.device_name}',
                'dtype': 'number',
                'shape': [],
                'units': 'deg'
            }
        }


class DiSPIMXYStage(Device):
    """
    DiSPIM XY stage - works with bps.mv(xy_stage.x, x, xy_stage.y, y)
    
    Device-agnostic: any plan that moves XY positions will work with this device
    """
    
    def __init__(self, xy_device_name: str, core: pymmcore.CMMCore, **kwargs):
        self.xy_device_name = xy_device_name
        self.core = core
        
        super().__init__(**kwargs)
        
        # Create X and Y components
        self.x = DiSPIMPiezo(xy_device_name + "-X", core, name='x')
        self.y = DiSPIMPiezo(xy_device_name + "-Y", core, name='y')
    
    def move_xy(self, x: float, y: float):
        """Convenience method for moving both axes"""
        return AndStatus(self.x.move(x), self.y.move(y))
    
    def read(self):
        """Read current XY stage positions - required for Bluesky"""
        result = {}
        result.update(self.x.read())
        result.update(self.y.read())
        return result
    
    def describe(self):
        """Describe XY stage device - required for Bluesky"""
        result = {}
        result.update(self.x.describe())
        result.update(self.y.describe())
        return result


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
        if not self.core:
            status = DeviceStatus(self)
            status.set_finished()
            return status
        
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
            return {
                f'{self.name}_image': {
                    'value': self._last_image,
                    'timestamp': self._last_image_time or time.time()
                },
                f'{self.name}_stats': {
                    'value': {
                        'shape': self._last_image.shape,
                        'dtype': str(self._last_image.dtype),
                        'mean': float(np.mean(self._last_image)),
                        'max': int(np.max(self._last_image)),
                        'min': int(np.min(self._last_image))
                    },
                    'timestamp': self._last_image_time or time.time()
                }
            }
        else:
            return {}
    
    def describe(self):
        """Describe detector data format"""
        return {
            f'{self.name}_image': {
                'source': f'DiSPIM Camera {self.device_name}',
                'dtype': 'array',
                'shape': getattr(self._last_image, 'shape', []),
                'external': 'FILESTORE:TIFF'
            },
            f'{self.name}_stats': {
                'source': f'DiSPIM Camera {self.device_name} Statistics',
                'dtype': 'object',
                'shape': []
            }
        }
    
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
        
        # Configure config signal
        self.config = MMCoreSignal("Laser", "Config", core, name='config')
        
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
            return {
                f'{self.name}_config': {
                    'value': current_config,
                    'timestamp': time.time()
                }
            }
        except:
            return {
                f'{self.name}_config': {
                    'value': 'unknown',
                    'timestamp': time.time()
                }
            }
    
    def describe(self):
        """Describe laser control device - required for Bluesky"""
        return {
            f'{self.name}_config': {
                'source': f'DiSPIM Laser Control {self.group_name}',
                'dtype': 'string',
                'shape': [],
                'choices': self._available_configs
            }
        }


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
   