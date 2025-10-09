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

Device Configuration (from MMConfig_tracking_screening.cfg):
    - ZStage:V:37 -> DiSPIMFDrive (F-axis module)
    - PiezoStage:P:34, Q:35 -> DiSPIMPiezo (objective focus)
    - Scanner:AB:33, CD:33 -> DiSPIMScanner (galvo mirrors)
    - HamCam1, HamCam2 -> DiSPIMCamera (dual cameras)
    - LED:X:31 -> DiSPIMLED (LED shutter)
    - Laser ConfigGroup -> DiSPIMLaserControl

All devices include units in their data (micrometers for stages/piezos, volts for scanners).

TODO: investigate the coordinate system of xy stage units. Ideally should be in um.
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
            'timestamp': time.time(),
            'units': 'micrometers'
        }
        return data

    def describe(self):
        """Describe Z stage device - required for Bluesky"""
        data = OrderedDict()
        data[self.name] = {
            'source': self.device_name,
            'dtype': 'number',
            'shape': [],
            'units': 'micrometers'
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

    def __init__(self, xy_device_name: str, core: pymmcore.CMMCore,
                 x_limits: Tuple[float, float] = (600.0, 2200.0),
                 y_limits: Tuple[float, float] = (-700.0, 2300.0), **kwargs):
        self.xy_device_name = xy_device_name
        self.core = core
        self._x_limits = x_limits
        self._y_limits = y_limits

        self.name = kwargs.get('name', xy_device_name)
        self.parent = None

    @property
    def x_limits(self):
        return self._x_limits

    @property
    def y_limits(self):
        return self._y_limits
    
    def set(self, position, **kwargs):
        """Move XY stage to position [x, y] - called by bps.mv(xy_stage, [x, y])"""
        try:
            x, y = position  # Unpack [x, y] coordinates
            x = float(x)
            y = float(y)

            # Safety checks
            if not (self._x_limits[0] <= x <= self._x_limits[1]):
                raise ValueError(f"X position {x} outside limits {self._x_limits}")
            if not (self._y_limits[0] <= y <= self._y_limits[1]):
                raise ValueError(f"Y position {y} outside limits {self._y_limits}")

            status = Status(obj=self, timeout=30)

            def wait():
                try:
                    # Set XY position using MM core
                    self.core.setXYPosition(x, y)
                    self.core.waitForDevice(self.xy_device_name)
                except Exception as exc:
                    status.set_exception(exc)
                else:
                    status.set_finished()

            import threading
            threading.Thread(target=wait).start()

            return status

        except (ValueError, TypeError) as e:
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
            'timestamp': time.time(),
            'units': 'micrometers'
        }
        return data

    def describe(self):
        """Describe XY stage device - required for Bluesky"""
        data = OrderedDict()
        data[self.xy_device_name] = {
            'source': self.xy_device_name,
            'dtype': 'array',
            'shape': [2],
            'units': 'micrometers'
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


class DiSPIMDualCamera:
    """
    DiSPIM Dual Camera - synchronized access to both SPIM cameras

    Manages HamCam1 and HamCam2 for dual-view SPIM imaging
    Can trigger both cameras individually or simultaneously
    """

    def __init__(self, camera_a_name: str, camera_b_name: str,
                 core: pymmcore.CMMCore, **kwargs):
        self.camera_a_name = camera_a_name
        self.camera_b_name = camera_b_name
        self.core = core
        self.name = kwargs.get('name', 'dual_spim_camera')
        self.parent = None

        # Individual camera devices
        self.camera_a = DiSPIMCamera(camera_a_name, core, name=f"{self.name}_a")
        self.camera_b = DiSPIMCamera(camera_b_name, core, name=f"{self.name}_b")

    def trigger_both(self):
        """Trigger both cameras simultaneously"""
        status_a = self.camera_a.trigger()
        status_b = self.camera_b.trigger()

        # Create combined status
        combined_status = Status(self)

        def wait_both():
            try:
                status_a.wait(timeout=10)
                status_b.wait(timeout=10)
                combined_status.set_finished()
            except Exception as exc:
                combined_status.set_exception(exc)

        import threading
        threading.Thread(target=wait_both).start()

        return combined_status

    def trigger(self):
        """Default trigger behavior - trigger both cameras"""
        return self.trigger_both()

    def read(self):
        """Read both camera images"""
        data_a = self.camera_a.read()
        data_b = self.camera_b.read()

        # Combine data
        combined = OrderedDict()
        combined.update(data_a)
        combined.update(data_b)
        return combined

    def describe(self):
        """Describe both cameras"""
        desc_a = self.camera_a.describe()
        desc_b = self.camera_b.describe()

        # Combine descriptions
        combined = OrderedDict()
        combined.update(desc_a)
        combined.update(desc_b)
        return combined

    def read_configuration(self):
        """Required for Bluesky"""
        return OrderedDict()

    def describe_configuration(self):
        """Required for Bluesky"""
        return OrderedDict()


class DiSPIMFDrive:
    """
    DiSPIM F-drive (SPIM Head motor) - works with bps.mv(fdrive, position)

    ASI Tiger V:37 axis - controls F-axis module for lowering objectives
    Device-agnostic: any plan that moves a positioner will work with this device
    """

    def __init__(self, device_name: str, core: pymmcore.CMMCore,
                 limits: Tuple[float, float] = (0.0, 25000.0), **kwargs):
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
        """Move F-drive to position - called by bps.mv()"""
        position = float(position)
        position = round(position, 2)  # Round to 0.01 μm precision

        # Safety check
        if not (self._limits[0] <= position <= self._limits[1]):
            raise ValueError(f"Position {position} outside limits {self._limits}")

        status = Status(obj=self, timeout=30)

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
        """Read current F-drive position - required for Bluesky"""
        try:
            value = self.core.getPosition(self.device_name)
        except Exception as e:
            print(f"Failed to read position from {self.device_name}: {e}")
            value = 0.0

        data = OrderedDict()
        data[self.name] = {
            'value': float(value),
            'timestamp': time.time(),
            'units': 'micrometers'
        }
        return data

    def describe(self):
        """Describe F-drive device - required for Bluesky"""
        data = OrderedDict()
        data[self.name] = {
            'source': self.device_name,
            'dtype': 'number',
            'shape': [],
            'units': 'micrometers'
        }
        return data

    def read_configuration(self):
        """Required for Bluesky"""
        return OrderedDict()

    def describe_configuration(self):
        """Required for Bluesky"""
        return OrderedDict()


class DiSPIMPiezo:
    """
    DiSPIM Piezo stage - works with bps.mv(piezo, position)

    ASI Tiger PiezoStage (P:34 or Q:35) - objective focus control
    Device-agnostic: any plan that moves a positioner will work with this device
    """

    def __init__(self, device_name: str, core: pymmcore.CMMCore,
                 limits: Tuple[float, float] = (0.0, 200.0), **kwargs):
        self.device_name = device_name
        self.core = core
        self._limits = limits
        self.tolerance = 0.01  # µm
        self.name = kwargs.get('name', device_name)
        self.parent = None

    @property
    def limits(self):
        return self._limits

    def set(self, position, **kwargs):
        """Move piezo to position - called by bps.mv()"""
        position = float(position)
        position = round(position, 3)  # Round to 0.001 μm precision

        # Safety check
        if not (self._limits[0] <= position <= self._limits[1]):
            raise ValueError(f"Position {position} outside limits {self._limits}")

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
        """Read current piezo position - required for Bluesky"""
        try:
            value = self.core.getPosition(self.device_name)
        except Exception as e:
            print(f"Failed to read position from {self.device_name}: {e}")
            value = 0.0

        data = OrderedDict()
        data[self.name] = {
            'value': float(value),
            'timestamp': time.time(),
            'units': 'micrometers'
        }
        return data

    def describe(self):
        """Describe piezo device - required for Bluesky"""
        data = OrderedDict()
        data[self.name] = {
            'source': self.device_name,
            'dtype': 'number',
            'shape': [],
            'units': 'micrometers'
        }
        return data

    def read_configuration(self):
        """Required for Bluesky"""
        return OrderedDict()

    def describe_configuration(self):
        """Required for Bluesky"""
        return OrderedDict()


class DiSPIMScanner:
    """
    DiSPIM Scanner/Galvo control - works with bps.mv(scanner, [a_pos, b_pos])

    ASI Tiger Scanner (AB:33 or CD:33) - controls galvo mirrors for light sheet
    Device-agnostic: any plan that moves a 2D positioner will work with this device
    """

    def __init__(self, device_name: str, core: pymmcore.CMMCore,
                 limits: Tuple[float, float] = (-5.0, 5.0), **kwargs):
        self.device_name = device_name
        self.core = core
        self._limits = limits
        self.name = kwargs.get('name', device_name)
        self.parent = None

    @property
    def limits(self):
        return self._limits

    def set(self, position, **kwargs):
        """Move scanner to position [a, b] - called by bps.mv()"""
        try:
            a_pos, b_pos = position
            a_pos = float(a_pos)
            b_pos = float(b_pos)

            # Safety checks
            if not (self._limits[0] <= a_pos <= self._limits[1]):
                raise ValueError(f"A position {a_pos} outside limits {self._limits}")
            if not (self._limits[0] <= b_pos <= self._limits[1]):
                raise ValueError(f"B position {b_pos} outside limits {self._limits}")

            status = Status(obj=self, timeout=5)

            def wait():
                try:
                    # Scanner uses XY position interface for AB axes
                    self.core.setXYPosition(self.device_name, a_pos, b_pos)
                    self.core.waitForDevice(self.device_name)
                except Exception as exc:
                    status.set_exception(exc)
                else:
                    status.set_finished()

            import threading
            threading.Thread(target=wait).start()

            return status

        except (ValueError, TypeError) as e:
            status = Status(self)
            status.set_exception(e)
            return status

    def read(self):
        """Read current scanner positions - required for Bluesky"""
        try:
            # getXYPosition returns tuple (x, y) which maps to (a, b)
            ab_pos = np.array(self.core.getXYPosition(self.device_name))
        except Exception as e:
            print(f"Failed to read scanner positions from {self.device_name}: {e}")
            ab_pos = np.array([0.0, 0.0])

        data = OrderedDict()
        data[self.name] = {
            'value': ab_pos,
            'timestamp': time.time(),
            'units': 'volts'
        }
        return data

    def describe(self):
        """Describe scanner device - required for Bluesky"""
        data = OrderedDict()
        data[self.name] = {
            'source': self.device_name,
            'dtype': 'array',
            'shape': [2],
            'units': 'volts'
        }
        return data

    def read_configuration(self):
        """Required for Bluesky"""
        return OrderedDict()

    def describe_configuration(self):
        """Required for Bluesky"""
        return OrderedDict()


class DiSPIMLED:
    """
    DiSPIM LED control - works with bps.mv(led, state)

    ASI Tiger LED (LED:X:31) - LED shutter control
    Device-agnostic: any plan that sets device state will work
    """

    def __init__(self, device_name: str, core: pymmcore.CMMCore, **kwargs):
        self.device_name = device_name
        self.core = core
        self.name = kwargs.get('name', device_name)
        self.parent = None

    def set(self, state: str, **kwargs):
        """Set LED state - called by bps.mv(led, 'Open') or bps.mv(led, 'Closed')"""
        if state not in ['Open', 'Closed']:
            raise ValueError(f"State must be 'Open' or 'Closed', got '{state}'")

        status = Status(obj=self, timeout=5)

        def wait():
            try:
                self.core.setProperty(self.device_name, 'State', state)
                self.core.waitForDevice(self.device_name)
            except Exception as exc:
                status.set_exception(exc)
            else:
                status.set_finished()

        import threading
        threading.Thread(target=wait).start()

        return status

    def read(self):
        """Read current LED state - required for Bluesky"""
        try:
            state = self.core.getProperty(self.device_name, 'State')
        except Exception as e:
            print(f"Failed to read LED state from {self.device_name}: {e}")
            state = 'unknown'

        data = OrderedDict()
        data[self.name] = {
            'value': state,
            'timestamp': time.time()
        }
        return data

    def describe(self):
        """Describe LED device - required for Bluesky"""
        data = OrderedDict()
        data[self.name] = {
            'source': self.device_name,
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


class DiSPIMLaserControl:
    """
    DiSPIM laser control - works with bps.mv(laser, 'config_name')

    Device-agnostic: any plan that sets configurations will work with this device
    """

    def __init__(self, core: pymmcore.CMMCore, group_name: str = "Laser", **kwargs):
        self.core = core
        self.group_name = group_name

        self.name = kwargs.get('name', group_name)
        self.parent = None

        # Cache available configs
        self._available_configs = self._get_available_configs()

    def _get_available_configs(self):
        """Get available laser configurations"""
        try:
            return list(self.core.getAvailableConfigs(self.group_name))
        except:
            return []

    def set(self, config_name: str, **kwargs):
        """Set laser configuration - called by bps.mv(laser, 'config_name')"""
        if config_name not in self._available_configs:
            raise ValueError(f"Config '{config_name}' not available. "
                           f"Available: {self._available_configs}")

        status = Status(obj=self, timeout=5)

        def wait():
            try:
                self.core.setConfig(self.group_name, config_name)
                self.core.waitForConfig(self.group_name, config_name)
            except Exception as exc:
                status.set_exception(exc)
            else:
                status.set_finished()

        import threading
        threading.Thread(target=wait).start()

        return status

    def read(self):
        """Read current laser configuration - required for Bluesky"""
        try:
            current_config = self.core.getCurrentConfig(self.group_name)
        except:
            current_config = 'unknown'

        data = OrderedDict()
        data[self.name] = {
            'value': current_config,
            'timestamp': time.time()
        }
        return data

    def describe(self):
        """Describe laser control device - required for Bluesky"""
        data = OrderedDict()
        data[self.name] = {
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
   