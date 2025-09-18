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

 # to do: add unit to the data. it is micrometer here
"""

import time
import logging
from collections import OrderedDict
from typing import Dict, Tuple
import numpy as np


from ophyd.status import Status

import pymmcore
import rpyc


class DiSPIMZstage:
    """
    DiSPIM Z Stage positioner - works with bps.mv(z_stage, position)

    Device-agnostic: any plan that moves a positioner will work with this device
    """

    def __init__(self, device_name: str, core: pymmcore.CMMCore,
                 limits: Tuple[float, float] = (50.0, 250.0), **kwargs):
        self.device_name = device_name
        self.core = core
        self._limits = limits
        self.tolerance = 0.1  # µm
        self.name = kwargs.get('name', device_name)
        self.parent = None
    
    @property
    def limits(self):
        return self._limits
        
    def set(self, position, **kwargs):
        """Move Z stage to position - called by bps.mv()"""
        position = float(position)

        # Round to avoid floating point precision issues
        position = round(position, 2)  # Round to 0.01 μm precision

        # Safety check
        if not (self._limits[0] <= position <= self._limits[1]):
            raise ValueError(f"Position {position} outside limits {self._limits}")
        
        # Direct MM core implementation like deepthought
        status = Status(obj=self, timeout=10)

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
        """Read current Z stage position - required for Bluesky"""
        try:
            value = self.core.getPosition(self.device_name)
        except Exception as e:
            print(f"Failed to read position from {self.device_name}: {e}")
            value = 0.0
                
        data = OrderedDict()
        data[self.name] = {
            'value': float(value),
            'timestamp': time.time()
        }
        return data
    
    def describe(self):
        """Describe Z stage device - required for Bluesky"""
        data = OrderedDict()
        data[self.name] = {
            'source': self.device_name,
            'dtype': 'number',
            'shape': [],
        }
        return data
    
    def read_configuration(self):
        """Required for Bluesky"""
        return OrderedDict()
    
    def describe_configuration(self):
        """Required for Bluesky"""
        return OrderedDict()


class DiSPIMXYStage:
    """
    DiSPIM XY stage - works with bps.mv(xy_stage, [x, y])
    
    Device-agnostic: any plan that moves XY positions will work with this device
    Based on deepthought XYStage implementation
    """
    
    def __init__(self, xy_device_name: str, core: pymmcore.CMMCore, **kwargs):
        self.xy_device_name = xy_device_name
        self.core = core
        
        self.name = kwargs.get('name', device_name)
        self.parent = None
    
    def move(self, position):
        """Move XY stage to position [x, y] - called by bps.mv(xy_stage, [x, y])"""
        try:
            x, y = position  # Unpack [x, y] coordinates
            print(f"Moving XY stage to ({x}, {y})")
            
            # Set XY position using MM core
            self.core.setXYPosition(x, y)
            self.core.waitForDevice(self.xy_device_name)
            
            status = Status(self)
            status.set_finished()
            return status
            
        except Exception as e:
            print(f"Failed to move XY stage: {e}")
            status = Status(self)
            status.set_exception(e)
            return status
    
    def read(self):
        """Read current XY stage positions - required for Bluesky"""
        try:
            xy_pos = np.array(self.core.getXYPosition())
        except Exception as e:
            print(f"Failed to read XY positions: {e}")
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
    
    def read_configuration(self):
        """Required for Bluesky"""
        return OrderedDict()
    
    def describe_configuration(self):
        """Required for Bluesky"""
        return OrderedDict()


class DiSPIMCamera:
    """
    DiSPIM camera detector - works with bps.trigger_and_read([camera])
    
    Device-agnostic: any plan that acquires from a detector will work with this device
    """
    
    def __init__(self, device_name: str, core: pymmcore.CMMCore, **kwargs):
        self.name = kwargs.get('name', device_name)
        self.parent = None
        self.device_name = device_name
        self.core = core
        self._acquiring = False
        self._last_image = None
        self._last_image_time = None
        
    def trigger(self):
        """Trigger image acquisition - called by bps.trigger()"""        
        def acquire_image():
            try:
                # Set camera and snap
                self.core.setCameraDevice(self.device_name)
                self.core.snapImage()
                
                # Use rpyc.classic.obtain to transfer numpy array properly
                self._last_image = rpyc.classic.obtain(self.core.getImage())
                self._last_image_time = time.time()
                self._acquiring = False
                return True
            except Exception as e:
                print(f"Image acquisition failed: {e}")
                self._acquiring = False
                return False
        
        self._acquiring = True
        
        # Run acquisition
        success = acquire_image()
        
        status = Status(self)
        if success:
            status.set_finished()
        else:
            status.set_exception(RuntimeError("Image acquisition failed"))
        
        return status
    
    def read(self):
        """Read acquired image data - called by bps.read()"""
        if self._last_image is not None:
            data = OrderedDict()
            data[self.name] = {
                'value': self._last_image,
                'timestamp': self._last_image_time or time.time()
            }
            return data
        else:
            return OrderedDict()
    
    def describe(self):
        """Describe detector data format"""
        data = OrderedDict()
        data[self.name] = {
            'source': self.device_name,
            'dtype': 'array',
            'shape': getattr(self._last_image, 'shape', [])
        }
        return data
    
    def read_configuration(self):
        """Required for Bluesky"""
        return OrderedDict()
    
    def describe_configuration(self):
        """Required for Bluesky"""
        return OrderedDict()
    
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
            print(f"Failed to set exposure: {e}")


class DiSPIMLaserControl:
    """
    DiSPIM laser control - works with bps.mv(laser, 'config_name')
    
    Device-agnostic: any plan that sets configurations will work with this device
    """
    
    def __init__(self, core: pymmcore.CMMCore, **kwargs):
        self.core = core
        self.group_name = "Laser"
        
        self.name = kwargs.get('name', device_name)
        self.parent = None
        
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
            print(f"Set laser config to: {config_name}")
        except Exception as e:
            print(f"Failed to set laser config: {e}")
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
    
    def read_configuration(self):
        """Required for Bluesky"""
        return OrderedDict()
    
    def describe_configuration(self):
        """Required for Bluesky"""
        return OrderedDict()


if __name__ == "__main__":
    # Example usage - would normally use actual MM paths
    logging.basicConfig(level=logging.INFO)
   