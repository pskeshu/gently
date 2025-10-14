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
                 limits: Tuple[float, float] = (800, 25000.0), **kwargs):
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


class DiSPIMLightSheetSnap:
    """
    Single light sheet image trigger - works with bps.trigger()

    Simplest SPIM device: creates light sheet (X-axis scanning) at a single
    Y position and triggers one image acquisition. No Z-scanning.

    This uses SPIM mode with num_slices=1 for hardware-synchronized
    light sheet generation and camera triggering. Acquires image using
    camera sequence mode and popNextTaggedImage().

    Usage:
        # Configure light sheet
        ls_snap.configure(sheet_width_deg=2.0, y_position_deg=0.0)

        # Trigger single light sheet image
        yield from bps.trigger(ls_snap)

        # Read image
        data = yield from bps.read(ls_snap)
        image = data['lightsheet_snap']['value']
    """

    def __init__(self, scanner_device_name: str, camera_device_name: str,
                 core: pymmcore.CMMCore, **kwargs):
        self.scanner_name = scanner_device_name
        self.camera_name = camera_device_name
        self.core = core
        self.name = kwargs.get('name', 'lightsheet_snap')
        self.parent = None

        self._configured = False
        self._last_image = None
        self._last_image_time = None

    def configure(self,
                  sheet_width_deg: float = 2.0,     # Light sheet width
                  sheet_offset_deg: float = 0.0,     # Light sheet center (X)
                  y_position_deg: float = 0.0,       # Y-axis position (Z-plane)
                  scan_duration_ms: float = 10.0,
                  camera_delay_ms: float = 0.5,
                  camera_duration_ms: float = 9.0):
        """
        Configure light sheet for single image acquisition

        Parameters
        ----------
        sheet_width_deg : float
            Light sheet width in degrees (X-axis scan range)
        sheet_offset_deg : float
            Light sheet center position in degrees (X-axis)
        y_position_deg : float
            Y-axis position in degrees (selects Z-plane)
        scan_duration_ms : float
            Scan duration in milliseconds
        camera_delay_ms : float
            Delay before camera trigger in milliseconds
        camera_duration_ms : float
            Camera exposure duration in milliseconds
        """
        try:
            # Stop any running sequence acquisition first
            if self.core.isSequenceRunning():
                self.core.stopSequenceAcquisition()
                time.sleep(0.1)  # Brief pause for cleanup

            # Configure camera for external hardware triggering
            self.core.setCameraDevice(self.camera_name)
            self.core.setProperty(self.camera_name, "TRIGGER SOURCE", "EXTERNAL")

            # Light sheet generation (X-axis continuous scanning)
            self.core.setProperty(self.scanner_name, "SingleAxisXAmplitude(deg)", sheet_width_deg)
            self.core.setProperty(self.scanner_name, "SingleAxisXOffset(deg)", sheet_offset_deg)
            self.core.setProperty(self.scanner_name, "SingleAxisXPattern", "1 - Triangle")  # Triangle wave for scanning
            self.core.setProperty(self.scanner_name, "SingleAxisXMode", "3 - Enabled with axes synced")  # Sync with SPIM

            # Y-axis position (single Z-plane, no scanning)
            self.core.setProperty(self.scanner_name, "SingleAxisYAmplitude(deg)", 0.0)  # No Y-scan
            self.core.setProperty(self.scanner_name, "SingleAxisYOffset(deg)", y_position_deg)
            self.core.setProperty(self.scanner_name, "SingleAxisYMode", "0 - Disabled")

            # SPIM parameters for single slice
            self.core.setProperty(self.scanner_name, "SPIMNumSlices", 1)
            self.core.setProperty(self.scanner_name, "SPIMNumSides", 1)  # Single side acquisition
            self.core.setProperty(self.scanner_name, "SPIMScanDuration(ms)", scan_duration_ms)
            self.core.setProperty(self.scanner_name, "SPIMDelayBeforeCamera(ms)", camera_delay_ms)
            self.core.setProperty(self.scanner_name, "SPIMCameraDuration(ms)", camera_duration_ms)

            self._configured = True

            print(f"Light sheet configured: width={sheet_width_deg}°, "
                  f"Y-position={y_position_deg}°, camera trigger=EXTERNAL")

        except Exception as e:
            raise RuntimeError(f"Failed to configure light sheet: {e}")

    def trigger(self):
        """
        Trigger single light sheet image acquisition - called by bps.trigger()

        This starts camera sequence acquisition, triggers SPIM, and collects
        the image from the circular buffer using popNextTaggedImage().

        Returns
        -------
        Status
            Ophyd Status object that completes when image is acquired
        """
        if not self._configured:
            raise RuntimeError("Must call configure() before trigger()")

        status = Status(obj=self, timeout=10)

        def run_acquisition():
            try:
                # Start camera sequence acquisition (1 image)
                self.core.startSequenceAcquisition(self.camera_name, 1, 0, True)

                # Trigger SPIM to run (directly to Running, not Armed first)
                # This matches ASI diSPIM plugin behavior for SLICE_SCAN_ONLY mode
                self.core.setProperty(self.scanner_name, "SPIMState", "Running")

                # Wait for first image to arrive in circular buffer
                # Match ASI plugin pattern: wait for image OR sequence to complete
                timeout_ms = max(3000, int(10 * 10.0 + 2 * 0.5))  # Based on scan_duration_ms
                start_time = time.time()

                while self.core.getRemainingImageCount() == 0:
                    elapsed_ms = (time.time() - start_time) * 1000
                    if elapsed_ms >= timeout_ms:
                        msg = "Camera did not send first image within timeout.\n"
                        msg += "Make sure camera trigger cables are connected properly."
                        raise TimeoutError(msg)
                    time.sleep(0.005)

                # Check if sequence stopped prematurely (indicates error)
                if not self.core.isSequenceRunning(self.camera_name) and self.core.getRemainingImageCount() == 0:
                    raise RuntimeError("Camera sequence stopped without sending image")

                # Pop image from circular buffer
                tagged_img = self.core.popNextTaggedImage()

                # Extract pixel data using rpyc to transfer numpy array properly
                self._last_image = rpyc.classic.obtain(tagged_img.pix)
                self._last_image_time = time.time()

                # Wait for SPIM to complete
                timeout_start = time.time()
                while True:
                    state = self.core.getProperty(self.scanner_name, "SPIMState")
                    if state == "Idle":
                        break

                    time.sleep(0.01)

                    if time.time() - timeout_start > 10:
                        raise TimeoutError("SPIM did not complete in 10s")

                status.set_finished()

            except Exception as e:
                status.set_exception(e)
            finally:
                # Make sure to stop sequence if it's still running
                try:
                    if self.core.isSequenceRunning(self.camera_name):
                        self.core.stopSequenceAcquisition(self.camera_name)
                except:
                    pass

        import threading
        threading.Thread(target=run_acquisition).start()
        return status

    def read(self):
        """Read acquired image - required for Bluesky"""
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
        """Describe light sheet image data - required for Bluesky"""
        data = OrderedDict()
        data[self.name] = {
            'source': f'{self.scanner_name}+{self.camera_name}',
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


class DiSPIMVolumeScanner:
    """
    Hardware-triggered SPIM volume acquisition device - works with bps.trigger()

    Encapsulates complete volume acquisition workflow using hardware-triggered
    SPIM mode. Single trigger acquires entire 3D volume (100 slices @ 59fps).

    Based on proven test_volume_acq.py implementation with all critical fixes:
    - PROGRESSIVE sensor mode (required for hardware triggering)
    - Explicit SPIM timing property configuration
    - Proper circular buffer management
    - Galvo Y-axis slice stepping (no piezo needed)

    Usage:
        # Create device
        vol_scanner = DiSPIMVolumeScanner("Scanner:AB:33", "HamCam1", core)

        # Configure acquisition
        vol_scanner.configure(num_slices=100, exposure_ms=5.0, slice_step_um=1.0)

        # Acquire volume in Bluesky plan
        yield from bps.trigger_and_read([vol_scanner])

        # Read volume
        data = yield from bps.read(vol_scanner)
        volume = data['volume_scanner']['value']  # 3D numpy array (Z, Y, X)

    Reference:
        Full documentation in doc/asidispim_camera_triggering.md
        Standalone script: test_volume_acq.py
    """

    def __init__(self, scanner_device_name: str, camera_device_name: str,
                 core: pymmcore.CMMCore, **kwargs):
        self.scanner_name = scanner_device_name
        self.camera_name = camera_device_name
        self.core = core
        self.name = kwargs.get('name', 'volume_scanner')
        self.parent = None

        # Acquisition configuration
        self._configured = False
        self._num_slices = None
        self._exposure_ms = None
        self._slice_step_um = None
        self._timing = None

        # Acquired data
        self._last_volume = None
        self._last_volume_time = None

        # Camera timing parameters (Hamamatsu Flash4 typical values)
        self.camera_reset_ms = 3.0
        self.camera_readout_ms = 10.0
        self.scan_laser_buffer_ms = 0.25
        self.scan_filter_freq_khz = 0.2
        self.has_plogic = True

    def configure(self, num_slices: int = 100, exposure_ms: float = 5.0,
                  slice_step_um: float = 1.0):
        """
        Configure volume acquisition parameters

        Parameters
        ----------
        num_slices : int
            Number of Z slices to acquire
        exposure_ms : float
            Light exposure time in milliseconds (max ~10-12ms for PROGRESSIVE mode)
        slice_step_um : float
            Distance between slices in micrometers
        """
        self._num_slices = num_slices
        self._exposure_ms = exposure_ms
        self._slice_step_um = slice_step_um

        # Calculate SPIM timing parameters
        self._timing = self._calculate_spim_timing(exposure_ms)

        self._configured = True

        print(f"Volume acquisition configured:")
        print(f"  Slices: {num_slices}")
        print(f"  Exposure: {exposure_ms} ms")
        print(f"  Step size: {slice_step_um} µm")
        print(f"  Expected time: {num_slices * self._timing['sliceDuration'] / 1000.0:.1f}s")

    def _calculate_spim_timing(self, camera_exposure_ms: float) -> Dict:
        """
        Calculate SPIM timing parameters following ASI diSPIM plugin logic

        Based on AcquisitionPanel.java:1105-1240 and test_volume_acq.py
        """
        import math

        # Round to 0.25ms (Tiger controller resolution)
        def round_quarter_ms(val):
            return round(val * 4) / 4.0

        def ceil_quarter_ms(val):
            return math.ceil(val * 4) / 4.0

        camera_readout_max = ceil_quarter_ms(self.camera_readout_ms)
        camera_reset_max = ceil_quarter_ms(self.camera_reset_ms)
        global_exposure_delay_max = camera_readout_max + camera_reset_max

        laser_duration = round_quarter_ms(camera_exposure_ms)
        scan_duration = laser_duration + 2 * self.scan_laser_buffer_ms

        # Account for Bessel filter delay and PLogic delay
        scan_delay_filter = 0.39 / self.scan_filter_freq_khz
        if self.has_plogic:
            scan_delay_filter -= 0.25

        timing = {
            'scanDelay': round_quarter_ms(global_exposure_delay_max - self.scan_laser_buffer_ms - scan_delay_filter),
            'scanPeriod': round_quarter_ms(scan_duration),
            'laserDelay': round_quarter_ms(global_exposure_delay_max),
            'laserDuration': laser_duration,
            'cameraDelay': camera_readout_max,
            'cameraDuration': 1.0,  # Short pulse for EDGE mode
            'cameraExposure': camera_exposure_ms + 0.1,
            'sliceDuration': max(scan_duration, laser_duration,
                                camera_readout_max + camera_exposure_ms)
        }

        return timing

    def trigger(self):
        """
        Trigger hardware-triggered volume acquisition - called by bps.trigger()

        Executes complete volume acquisition workflow:
        1. Configure camera (PROGRESSIVE mode, EXTERNAL trigger)
        2. Configure SPIM timing properties
        3. Start camera sequence acquisition
        4. Trigger SPIM state machine
        5. Wait for images
        6. Retrieve volume from buffer

        Returns
        -------
        Status
            Ophyd Status object that completes when volume is acquired
        """
        if not self._configured:
            raise RuntimeError("Must call configure() before trigger()")

        status = Status(obj=self, timeout=60)

        def run_acquisition():
            try:
                # Apply system configuration
                try:
                    self.core.setConfig("System", "Startup")
                    self.core.waitForConfig("System", "Startup")
                except:
                    pass  # Config may not exist

                # Turn on lasers
                try:
                    self.core.setConfig("Laser", "488 and 561")
                    self.core.waitForConfig("Laser", "488 and 561")
                except:
                    pass  # Config may not exist

                # Configure camera for hardware trigger
                self.core.setCameraDevice(self.camera_name)
                self.core.setProperty(self.camera_name, "TRIGGER SOURCE", "EXTERNAL")
                self.core.setProperty(self.camera_name, "SENSOR MODE", "PROGRESSIVE")  # CRITICAL!
                self.core.setProperty(self.camera_name, "TRIGGER ACTIVE", "EDGE")
                self.core.setExposure(self.camera_name, self._exposure_ms)
                time.sleep(0.1)

                # Configure Tiger controller
                self.core.setProperty(self.scanner_name, "SPIMState", "Idle")
                time.sleep(0.2)

                # CRITICAL: Set laser output mode
                self.core.setProperty(self.scanner_name, "LaserOutputMode", "shutter + side")

                # Disable beam scanning (controlled by SPIM)
                self.core.setProperty(self.scanner_name, "BeamEnabled", "No")

                # Configure galvo X-axis (light sheet)
                self.core.setProperty(self.scanner_name, "SingleAxisXAmplitude(deg)", 2.0)
                self.core.setProperty(self.scanner_name, "SingleAxisXOffset(deg)", 0.0)
                self.core.setProperty(self.scanner_name, "SingleAxisXPattern", "1 - Triangle")
                self.core.setProperty(self.scanner_name, "SingleAxisXMode", "3 - Enabled with axes synced")

                # Configure galvo Y-axis (slice stepping)
                # Y amplitude depends on number of slices and step size
                # Typical calibration: 100 deg/mm, 1um step -> 0.0001 deg per slice
                y_amplitude = (self._num_slices - 1) * self._slice_step_um / 1000.0 / 2.0  # Half range
                self.core.setProperty(self.scanner_name, "SingleAxisYAmplitude(deg)", y_amplitude)
                self.core.setProperty(self.scanner_name, "SingleAxisYOffset(deg)", 0.0)
                self.core.setProperty(self.scanner_name, "SingleAxisYPattern", "1 - Triangle")
                self.core.setProperty(self.scanner_name, "SingleAxisYMode", "3 - Enabled with axes synced")

                # Set SPIM state machine parameters
                self.core.setProperty(self.scanner_name, "SPIMNumSlices", self._num_slices)
                self.core.setProperty(self.scanner_name, "SPIMNumSides", 1)
                self.core.setProperty(self.scanner_name, "SPIMAlternateDirectionsEnable", "No")
                self.core.setProperty(self.scanner_name, "SPIMDelayBeforeSide(ms)", 0.0)
                self.core.setProperty(self.scanner_name, "SPIMDelayBeforeRepeat(ms)", 0.0)

                # CRITICAL: Set all SPIM timing properties explicitly
                self.core.setProperty(self.scanner_name, "SPIMDelayBeforeScan(ms)", self._timing['scanDelay'])
                self.core.setProperty(self.scanner_name, "SPIMScanDuration(ms)", self._timing['scanPeriod'])
                self.core.setProperty(self.scanner_name, "SPIMDelayBeforeLaser(ms)", self._timing['laserDelay'])
                self.core.setProperty(self.scanner_name, "SPIMLaserDuration(ms)", self._timing['laserDuration'])
                self.core.setProperty(self.scanner_name, "SPIMDelayBeforeCamera(ms)", self._timing['cameraDelay'])
                self.core.setProperty(self.scanner_name, "SPIMCameraDuration(ms)", self._timing['cameraDuration'])

                # Configure circular buffer
                if self.core.isSequenceRunning():
                    self.core.stopSequenceAcquisition()
                    time.sleep(0.5)

                self.core.clearCircularBuffer()

                buffer_capacity = self.core.getBufferTotalCapacity()
                if buffer_capacity < self._num_slices:
                    self.core.setCircularBufferMemoryFootprint(1200)  # MB
                    time.sleep(0.1)

                # Start camera sequence acquisition
                self.core.prepareSequenceAcquisition(self.camera_name)
                time.sleep(0.1)
                self.core.startSequenceAcquisition(self.camera_name, self._num_slices, 0, True)
                time.sleep(0.1)

                # Verify sequence started
                if not self.core.isSequenceRunning(self.camera_name):
                    raise RuntimeError("Camera sequence failed to start! Check SENSOR MODE = PROGRESSIVE")

                # Trigger SPIM state machine
                self.core.setProperty(self.scanner_name, "SPIMState", "Running")

                # Wait for images
                expected_time = self._num_slices * self._timing['sliceDuration'] / 1000.0
                timeout_sec = expected_time * 2 + 10.0

                start_time = time.time()
                while self.core.getRemainingImageCount() < self._num_slices:
                    elapsed = time.time() - start_time
                    if elapsed > timeout_sec:
                        count = self.core.getRemainingImageCount()
                        raise TimeoutError(f"Timeout: got {count}/{self._num_slices} images")
                    time.sleep(0.01)

                # Retrieve images
                images = []
                for i in range(self._num_slices):
                    img = self.core.popNextImage()
                    # Handle rpyc transfer
                    try:
                        img = rpyc.classic.obtain(img)
                    except (ImportError, AttributeError):
                        pass
                    images.append(img)

                # Create volume array
                self._last_volume = np.array(images)
                self._last_volume_time = time.time()

                elapsed = time.time() - start_time
                fps = self._num_slices / elapsed
                print(f"Volume acquired: {self._last_volume.shape}, {elapsed:.1f}s, {fps:.1f} fps")

                status.set_finished()

            except Exception as e:
                status.set_exception(e)
            finally:
                # Cleanup
                try:
                    if self.core.isSequenceRunning(self.camera_name):
                        self.core.stopSequenceAcquisition(self.camera_name)
                except:
                    pass

                try:
                    self.core.setProperty(self.scanner_name, "SPIMState", "Idle")
                except:
                    pass

                try:
                    self.core.setConfig("Laser", "ALL OFF")
                except:
                    pass

        import threading
        threading.Thread(target=run_acquisition).start()
        return status

    def read(self):
        """Read acquired volume - required for Bluesky"""
        if self._last_volume is not None:
            data = OrderedDict()
            data[self.name] = {
                'value': self._last_volume,
                'timestamp': self._last_volume_time or time.time(),
                'units': 'counts'
            }
            return data
        else:
            return OrderedDict()

    def describe(self):
        """Describe volume data - required for Bluesky"""
        data = OrderedDict()
        data[self.name] = {
            'source': f'{self.scanner_name}+{self.camera_name}',
            'dtype': 'array',
            'shape': getattr(self._last_volume, 'shape', []),
            'units': 'counts'
        }
        return data

    def read_configuration(self):
        """Required for Bluesky"""
        config = OrderedDict()
        if self._configured:
            config[f'{self.name}_num_slices'] = {
                'value': self._num_slices,
                'timestamp': time.time()
            }
            config[f'{self.name}_exposure_ms'] = {
                'value': self._exposure_ms,
                'timestamp': time.time()
            }
            config[f'{self.name}_slice_step_um'] = {
                'value': self._slice_step_um,
                'timestamp': time.time()
            }
        return config

    def describe_configuration(self):
        """Required for Bluesky"""
        desc = OrderedDict()
        if self._configured:
            desc[f'{self.name}_num_slices'] = {
                'source': 'configuration',
                'dtype': 'number',
                'shape': []
            }
            desc[f'{self.name}_exposure_ms'] = {
                'source': 'configuration',
                'dtype': 'number',
                'shape': [],
                'units': 'milliseconds'
            }
            desc[f'{self.name}_slice_step_um'] = {
                'source': 'configuration',
                'dtype': 'number',
                'shape': [],
                'units': 'micrometers'
            }
        return desc


if __name__ == "__main__":
    # Example usage - would normally use actual MM paths
    logging.basicConfig(level=logging.INFO)
   