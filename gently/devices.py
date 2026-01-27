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

    def __init__(self, name: str, core: pymmcore.CMMCore,
                 limits: Tuple[float, float] = (50.0, 250.0)):
        self.name = name
        self.core = core
        self.parent = None  # Required for Bluesky
        self._limits = limits
        self.tolerance = 0.1  # µm
    
    @property
    def limits(self):
        return self._limits
        
    def set(self, position):
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
                self.core.setPosition(self.name, position)
                self.core.waitForDevice(self.name)
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
            value = self.core.getPosition(self.name)
        except Exception as e:
            print(f"Failed to read position from {self.name}: {e}")
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
            'source': self.name,
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

    def __init__(self, name: str, core: pymmcore.CMMCore,
                 x_limits: Tuple[float, float] = (500.0, 2500.0),
                 y_limits: Tuple[float, float] = (-1000.0, 1000.0)):
        self.name = name
        self.core = core
        self.parent = None  # Required for Bluesky
        self._x_limits = x_limits
        self._y_limits = y_limits

    @property
    def x_limits(self):
        return self._x_limits

    @property
    def y_limits(self):
        return self._y_limits
    
    def set(self, position):
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
                    self.core.waitForDevice(self.name)
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
        xy_pos = np.array(self.core.getXYPosition())
       
        data = OrderedDict()
        data[self.name] = {
            'value': xy_pos,
            'timestamp': time.time(),
            'units': 'micrometers'
        }
        return data

    def describe(self):
        """Describe XY stage device - required for Bluesky"""
        data = OrderedDict()
        data[self.name] = {
            'source': self.name,
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

    # Synchronous convenience methods (usable outside RunEngine)
    def get_position(self) -> np.ndarray:
        """
        Get current XY stage position as numpy array.

        Returns
        -------
        np.ndarray
            Current position as [x, y] in micrometers

        Notes
        -----
        This is a synchronous convenience method that can be used outside
        the RunEngine for interactive use, setup, and debugging. For use
        within plans, prefer yield from bps.rd(xy_stage).
        """
        return self.read()[self.name]['value']

    def get_x(self) -> float:
        """
        Get current X stage position.

        Returns
        -------
        float
            X position in micrometers
        """
        return self.get_position()[0]

    def get_y(self) -> float:
        """
        Get current Y stage position.

        Returns
        -------
        float
            Y position in micrometers
        """
        return self.get_position()[1]

    # Coordinate conversion utilities for embryo centering
    @staticmethod
    def pixel_to_stage_offset(pixel_offset_x: float,
                               pixel_offset_y: float,
                               pixel_size_um: float) -> Tuple[float, float]:
        """
        Convert pixel offsets to stage movement in micrometers.

        IMPORTANT: X-axis is INVERTED - stage +X moves features LEFT in camera view.
        This is a hardware characteristic of the diSPIM coordinate system.

        Parameters
        ----------
        pixel_offset_x : float
            Horizontal pixel displacement (positive = right in image)
        pixel_offset_y : float
            Vertical pixel displacement (positive = down in image)
        pixel_size_um : float
            Effective pixel size in micrometers (physical pixel size / magnification)

        Returns
        -------
        Tuple[float, float]
            Stage movement required (dx_um, dy_um)

        Notes
        -----
        This method delegates to gently.coordinates for the actual calculation.
        """
        from .coordinates import pixel_displacement_to_stage_movement
        return pixel_displacement_to_stage_movement(pixel_offset_x, pixel_offset_y, pixel_size_um)


class DiSPIMCamera:
    """
    DiSPIM camera detector - works with bps.trigger_and_read([camera])
    
    Device-agnostic: any plan that acquires from a detector will work with this device
    """
    
    def __init__(self, device_name: str, core: pymmcore.CMMCore):
        self.name = device_name
        self.core = core
        self.parent = None  # Required for Bluesky
        self._last_image = None
        self._last_image_time = None
        
    def trigger(self):
        """Trigger image acquisition - called by bps.trigger()"""        

        def wait():
            try:
                # Set camera and snap
                self.core.setCameraDevice(self.name)
                self.core.snapImage()
                
                # Use rpyc.classic.obtain to transfer numpy array properly
                self._last_image = rpyc.classic.obtain(self.core.getImage())
                self._last_image_time = time.time()

            except Exception as exc:
                status.set_exception(exc)

            else:
                status.set_finished()


        status = Status(obj=self, timeout=30)

        import threading
        threading.Thread(target=wait).start()

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
            'source': self.name,
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

    # Hardware configuration methods
    def configure_exposure(self, exposure_ms: float):
        """
        Configure camera exposure time.

        This is a synchronous convenience method that encapsulates the common
        pattern of setting the camera device and configuring exposure.

        Parameters
        ----------
        exposure_ms : float
            Camera exposure time in milliseconds

        Notes
        -----
        This method can be used outside plans for setup and configuration.
        It automatically selects this camera as the active device and allows
        hardware time to settle after configuration.
        """
        self.core.setCameraDevice(self.name)
        self.core.setExposure(self.name, exposure_ms)
        self.core.waitForDevice(self.name)

    def set_roi(self, x: int, y: int, width: int, height: int):
        """Set camera region of interest."""
        self.core.setROI(self.name, x, y, width, height)

    def set_trigger_mode(self, mode: str):
        """
        Set trigger mode.

        Parameters
        ----------
        mode : str
            'INTERNAL' for software triggering or 'EXTERNAL' for hardware triggering
        """
        self.core.setProperty(self.name, "TRIGGER SOURCE", mode)

    def set_sensor_mode(self, mode: str):
        """
        Set sensor mode.

        Parameters
        ----------
        mode : str
            'AREA' for full frame readout or 'PROGRESSIVE' for rolling shutter (required for SPIM)
        """
        self.core.setProperty(self.name, "SENSOR MODE", mode)

    def set_trigger_active(self, mode: str):
        """
        Set trigger active mode.

        Parameters
        ----------
        mode : str
            'EDGE' for edge-triggered or 'LEVEL' for level-triggered
        """
        self.core.setProperty(self.name, "TRIGGER ACTIVE", mode)

    def configure_for_calibration(self, exposure_ms: float,
                                   roi: Tuple[int, int, int, int] = (128, 896, 2048, 512)):
        """
        Configure camera for calibration imaging (single light sheet snapshots).

        Uses INTERNAL trigger and AREA sensor mode for simple snapshot acquisition.

        Parameters
        ----------
        exposure_ms : float
            Camera exposure time in milliseconds
        roi : Tuple[int, int, int, int], optional
            Region of interest as (x, y, width, height). Default is diSPIM light sheet ROI.
        """
        self.core.setCameraDevice(self.name)
        self.set_roi(*roi)
        self.set_trigger_mode("INTERNAL")
        self.set_sensor_mode("AREA")
        self.core.setExposure(self.name, exposure_ms)
        self.core.waitForDevice(self.name)

    def configure_for_volume_acquisition(self, exposure_ms: float,
                                          roi: Tuple[int, int, int, int] = (128, 896, 2048, 512)):
        """
        Configure camera for hardware-triggered volume acquisition.

        Uses EXTERNAL trigger and PROGRESSIVE sensor mode for synchronized SPIM scanning.
        PROGRESSIVE mode is CRITICAL for proper synchronization with piezo/galvo.

        Parameters
        ----------
        exposure_ms : float
            Camera exposure time in milliseconds
        roi : Tuple[int, int, int, int], optional
            Region of interest as (x, y, width, height). Default is diSPIM light sheet ROI.
            CRITICAL: ROI must be set before hardware triggering!
        """
        self.core.setCameraDevice(self.name)
        self.set_roi(*roi)  # CRITICAL: ROI must be set for hardware triggering
        self.set_trigger_mode("EXTERNAL")
        self.set_sensor_mode("PROGRESSIVE")  # CRITICAL for SPIM!
        self.set_trigger_active("EDGE")
        self.core.setExposure(self.name, exposure_ms)
        self.core.waitForDevice(self.name)

    @property
    def exposure_time(self):
        """Get current exposure time"""
        try:
            return self.core.getExposure()
        except:
            return 0.01  # Default 10ms
    
    @exposure_time.setter 
    def exposure_time(self, value_ms):
        """Set exposure time in milliseconds"""
        try:
            self.core.setExposure(value_ms)
        except Exception as e:
            print(f"Failed to set exposure: {e}")


class DiSPIMDualCamera:
    """
    DiSPIM Dual Camera - synchronized access to both SPIM views

    Manages a single camera (HamCam1) that captures side-by-side stitched images
    from both SPIM views. The image is split in the middle to provide View A and View B.
    """

    def __init__(self, camera_name: str, core: pymmcore.CMMCore, name: str = "dual_camera"):
        self.name = name
        self.camera_name = camera_name
        self.core = core
        self.parent = None  # Required for Bluesky

        # Single camera device that captures stitched image
        self.camera = DiSPIMCamera(camera_name, core)

    def trigger(self):
        """Trigger camera to capture stitched image"""
        return self.camera.trigger()

    def read(self):
        """Read stitched image and split into View A and View B"""
        # Get the stitched image from single camera
        camera_data = self.camera.read()

        if self.camera.name in camera_data:
            stitched_image = camera_data[self.camera.name]['value']
            timestamp = camera_data[self.camera.name]['timestamp']

            # Split image in the middle (width dimension)
            height, width = stitched_image.shape[:2]
            mid_width = width // 2

            image_a = stitched_image[:, :mid_width]  # Left half
            image_b = stitched_image[:, mid_width:]  # Right half

            # Return as separate data entries
            data = OrderedDict()
            data[f'{self.name}_image_a'] = {
                'value': image_a,
                'timestamp': timestamp
            }
            data[f'{self.name}_image_b'] = {
                'value': image_b,
                'timestamp': timestamp
            }
            return data
        else:
            return OrderedDict()

    def describe(self):
        """Describe both image outputs (View A and View B)"""
        # Get camera description to determine image properties
        camera_desc = self.camera.describe()

        data = OrderedDict()

        # Describe image_a and image_b outputs
        # Shape will be half width of original stitched image
        if self.camera.name in camera_desc:
            original_shape = camera_desc[self.camera.name].get('shape', [])
            if len(original_shape) >= 2:
                # Split width dimension in half
                split_shape = [original_shape[0], original_shape[1] // 2]
                if len(original_shape) > 2:
                    split_shape.extend(original_shape[2:])
            else:
                split_shape = original_shape

            data[f'{self.name}_image_a'] = {
                'source': f'{self.name}_image_a',
                'dtype': 'array',
                'shape': split_shape
            }
            data[f'{self.name}_image_b'] = {
                'source': f'{self.name}_image_b',
                'dtype': 'array',
                'shape': split_shape
            }

        return data

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

    def __init__(self, name: str, core: pymmcore.CMMCore,
                 limits: Tuple[float, float] = (20.0, 25000.0)):
        self.name = name
        self.core = core
        self.parent = None  # Required for Bluesky
        self._limits = limits
        self.tolerance = 0.1  # µm

    @property
    def limits(self):
        return self._limits

    def set(self, position):
        """Move F-drive to position - called by bps.mv()"""
        position = float(position)
        position = round(position, 2)  # Round to 0.01 μm precision

        # Safety check
        if not (self._limits[0] <= position <= self._limits[1]):
            raise ValueError(f"Position {position} outside limits {self._limits}")

        status = Status(obj=self, timeout=30)

        def wait():
            try:
                self.core.setPosition(self.name, position)
                self.core.waitForDevice(self.name)
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
            value = self.core.getPosition(self.name)
        except Exception as e:
            print(f"Failed to read position from {self.name}: {e}")

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
            'source': self.name,
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

    def __init__(self, name: str, core: pymmcore.CMMCore,
                 limits: Tuple[float, float] = (-200, 200.0)):
        self.name = name
        self.core = core
        self.parent = None  # Required for Bluesky
        self._limits = limits
        self.tolerance = 0.01  # µm

    @property
    def limits(self):
        return self._limits

    def set(self, position):
        """Move piezo to position - called by bps.mv()"""
        position = float(position)
        position = round(position, 3)  # Round to 0.001 μm precision

        # Safety check
        if not (self._limits[0] <= position <= self._limits[1]):
            raise ValueError(f"Position {position} outside limits {self._limits}")

        status = Status(obj=self, timeout=10)

        def wait():
            try:
                self.core.setPosition(self.name, position)
                self.core.waitForDevice(self.name)
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
            value = self.core.getPosition(self.name)
        except Exception as e:
            print(f"Failed to read position from {self.name}: {e}")
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
            'source': self.name,
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

    # Hardware configuration methods for SPIM
    def set_as_focus_device(self):
        """Set this piezo as the Micro-Manager focus device."""
        self.core.setFocusDevice(self.name)

    def configure_amplitude_offset(self,
                                    amplitude_um: float,
                                    offset_um: float,
                                    pattern: str = "1 - Triangle"):
        """
        Configure piezo amplitude and offset for scanning.

        Parameters
        ----------
        amplitude_um : float
            Scanning amplitude in micrometers
        offset_um : float
            Center offset in micrometers
        pattern : str
            Waveform pattern (default: "1 - Triangle")
        """
        self.core.setProperty(self.name, "SingleAxisAmplitude(um)", float(amplitude_um))
        self.core.setProperty(self.name, "SingleAxisOffset(um)", float(offset_um))
        self.core.setProperty(self.name, "SingleAxisPattern", pattern)

    def set_spim_state(self, state: str):
        """
        Set SPIM state for piezo.

        Parameters
        ----------
        state : str
            'Idle' to stop or 'Armed' to prepare for hardware triggering
        """
        self.core.setProperty(self.name, "SPIMState", state)
        if state == "Armed":
            self.core.waitForDevice(self.name)

    def configure_for_spim(self, num_slices: int):
        """
        Configure piezo for SPIM acquisition.

        Parameters
        ----------
        num_slices : int
            Number of Z slices for the volume
        """
        self.core.setProperty(self.name, "SPIMNumSlices", num_slices)

    def configure_for_volume_acquisition(self,
                                          amplitude_um: float,
                                          offset_um: float,
                                          num_slices: int):
        """
        Configure piezo for hardware-triggered volume acquisition.

        Combines all necessary setup steps: sets as focus device, configures
        amplitude/offset, sets SPIM parameters, and arms the device.

        Parameters
        ----------
        amplitude_um : float
            Piezo scanning amplitude in micrometers
        offset_um : float
            Piezo center offset in micrometers
        num_slices : int
            Number of Z slices in volume
        """
        self.set_as_focus_device()
        self.configure_amplitude_offset(amplitude_um, offset_um)
        self.configure_for_spim(num_slices)
        self.set_spim_state("Armed")


class _ScannerAxisOffset:
    """
    Movable component for a single scanner axis offset.
    Compatible with bps.mv() for use in Bluesky plans.
    """

    def __init__(self, scanner, axis: str, property_name: str):
        self.scanner = scanner
        self.axis = axis
        self.property_name = property_name
        self.name = f"{scanner.name}_{axis}_offset"
        self.parent = scanner

    def set(self, value):
        """Move axis offset to specified position - called by bps.mv()"""
        status = Status(obj=self, timeout=5)

        def wait():
            try:
                self.scanner.core.setProperty(
                    self.scanner.name, self.property_name, float(value)
                )
                self.scanner.core.waitForDevice(self.scanner.name)
            except Exception as exc:
                status.set_exception(exc)
            else:
                status.set_finished()

        import threading
        threading.Thread(target=wait).start()
        return status

    def read(self):
        """Read current offset value"""
        try:
            value = float(self.scanner.core.getProperty(
                self.scanner.name, self.property_name
            ))
        except Exception:
            value = 0.0

        return OrderedDict({
            self.name: {
                'value': value,
                'timestamp': time.time(),
                'units': 'degrees'
            }
        })

    def describe(self):
        """Describe component"""
        return OrderedDict({
            self.name: {
                'source': self.name,
                'dtype': 'number',
                'shape': [],
                'units': 'degrees'
            }
        })


class DiSPIMScanner:
    """
    DiSPIM Scanner/Galvo control - works with bps.mv(scanner, [a_pos, b_pos])

    ASI Tiger Scanner (AB:33 or CD:33) - controls galvo mirrors for light sheet
    Device-agnostic: any plan that moves a 2D positioner will work with this device

    Individual axis offsets can be moved with:
        bps.mv(scanner.sa_offset_x, x_value)  # X-axis offset
        bps.mv(scanner.sa_offset_y, y_value)  # Y-axis offset (galvo position)
    """

    def __init__(self, name: str, core: pymmcore.CMMCore,
                 limits: Tuple[float, float] = (-5.0, 5.0)):
        self.name = name
        self.core = core
        self.parent = None  # Required for Bluesky
        self._limits = limits

        # Create movable axis offset components for use with bps.mv()
        self.sa_offset_x = _ScannerAxisOffset(self, 'x', 'SingleAxisXOffset(deg)')
        self.sa_offset_y = _ScannerAxisOffset(self, 'y', 'SingleAxisYOffset(deg)')

    @property
    def limits(self):
        return self._limits

    def set(self, position):
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
                    # Scanner uses galvo position interface for AB axes
                    self.core.setGalvoPosition(self.name, a_pos, b_pos)
                    self.core.waitForDevice(self.name)
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
            # getGalvoPosition returns tuple (a, b) voltages for galvo device
            ab_pos = np.array(self.core.getGalvoPosition(self.name))
        except Exception as e:
            print(f"Failed to read scanner positions from {self.name}: {e}")
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
            'source': self.name,
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

    # Hardware configuration methods for SPIM scanning
    def enable_beam(self, enabled: bool = True):
        """
        Enable or disable the laser beam.

        Parameters
        ----------
        enabled : bool
            True to enable beam, False to disable
        """
        self.core.setProperty(self.name, "BeamEnabled", "Yes" if enabled else "No")

    def set_laser_output_mode(self, mode: str):
        """
        Set laser output mode.

        Parameters
        ----------
        mode : str
            Laser output mode, e.g., "shutter + side" for side A imaging
        """
        self.core.setProperty(self.name, "LaserOutputMode", mode)

    def set_spim_state(self, state: str):
        """
        Set SPIM state machine state.

        Parameters
        ----------
        state : str
            'Idle' to stop, 'Armed' to prepare, 'Running' to trigger acquisition
        """
        self.core.setProperty(self.name, "SPIMState", state)
        if state == "Idle":
            self.core.waitForDevice(self.name)

    def configure_x_axis(self, amplitude_deg: float, offset_deg: float,
                         pattern: str = "1 - Triangle",
                         mode: str = "3 - Enabled with axes synced"):
        """
        Configure galvo X-axis (light sheet width scanning).

        Parameters
        ----------
        amplitude_deg : float
            Scanning amplitude in degrees (typically 8.0 for full light sheet)
        offset_deg : float
            Center offset in degrees
        pattern : str
            Waveform pattern (default: "1 - Triangle")
        mode : str
            Scan mode (default: "3 - Enabled with axes synced")
        """
        self.core.setProperty(self.name, "SingleAxisXAmplitude(deg)", amplitude_deg)
        self.core.setProperty(self.name, "SingleAxisXOffset(deg)", offset_deg)
        self.core.setProperty(self.name, "SingleAxisXPattern", pattern)
        self.core.setProperty(self.name, "SingleAxisXMode", mode)

    def configure_y_axis(self, amplitude_deg: float, offset_deg: float,
                         pattern: str = "1 - Triangle",
                         mode: str = "3 - Enabled with axes synced"):
        """
        Configure galvo Y-axis (light sheet Z-plane positioning).

        Parameters
        ----------
        amplitude_deg : float
            Scanning amplitude in degrees (synchronized with piezo for volume scanning)
        offset_deg : float
            Center offset in degrees (positions the light sheet vertically)
        pattern : str
            Waveform pattern (default: "1 - Triangle")
        mode : str
            Scan mode (default: "3 - Enabled with axes synced")
        """
        self.core.setProperty(self.name, "SingleAxisYAmplitude(deg)", amplitude_deg)
        self.core.setProperty(self.name, "SingleAxisYOffset(deg)", offset_deg)
        self.core.setProperty(self.name, "SingleAxisYPattern", pattern)
        self.core.setProperty(self.name, "SingleAxisYMode", mode)

    def set_y_offset(self, angle_deg: float):
        """
        Set Y-axis offset for light sheet positioning.

        Used during calibration to move the light sheet to different Z planes.

        Parameters
        ----------
        angle_deg : float
            Y-axis offset angle in degrees
        """
        self.core.setProperty(self.name, "SingleAxisYOffset(deg)", float(angle_deg))
        self.core.waitForDevice(self.name)

    def configure_spim_timing(self,
                              scan_delay_ms: float = 6.75,
                              num_scans_per_slice: int = 1,
                              scan_duration_ms: float = 5.5,
                              laser_delay_ms: float = 8.0,
                              laser_duration_ms: float = 5.0,
                              camera_delay_ms: float = 8.0,
                              camera_duration_ms: float = 1.0):
        """
        Configure SPIM timing parameters for hardware-triggered acquisition.

        Parameters
        ----------
        scan_delay_ms : float
            Delay before starting galvo scan (default: 6.75ms)
        num_scans_per_slice : int
            Number of galvo scans per Z slice (default: 1)
        scan_duration_ms : float
            Duration of galvo scan (default: 5.5ms)
        laser_delay_ms : float
            Delay before laser pulse (default: 8.0ms)
        laser_duration_ms : float
            Duration of laser pulse (default: 5.0ms)
        camera_delay_ms : float
            Delay before camera trigger (default: 8.0ms)
        camera_duration_ms : float
            Duration of camera exposure (default: 1.0ms)
        """
        self.core.setProperty(self.name, "SPIMDelayBeforeScan(ms)", scan_delay_ms)
        self.core.setProperty(self.name, "SPIMNumScansPerSlice", num_scans_per_slice)
        self.core.setProperty(self.name, "SPIMScanDuration(ms)", scan_duration_ms)
        self.core.setProperty(self.name, "SPIMDelayBeforeLaser(ms)", laser_delay_ms)
        self.core.setProperty(self.name, "SPIMLaserDuration(ms)", laser_duration_ms)
        self.core.setProperty(self.name, "SPIMDelayBeforeCamera(ms)", camera_delay_ms)
        self.core.setProperty(self.name, "SPIMCameraDuration(ms)", camera_duration_ms)

    def configure_spim_parameters(self,
                                   num_slices: int,
                                   slices_per_piezo: int = 1,
                                   num_sides: int = 1,
                                   first_side: str = "A"):
        """
        Configure SPIM acquisition parameters.

        Parameters
        ----------
        num_slices : int
            Total number of Z slices in volume
        slices_per_piezo : int
            Number of slices per piezo step (default: 1)
        num_sides : int
            Number of SPIM sides (1 or 2, default: 1)
        first_side : str
            First side to image ('A' or 'B', default: 'A')
        """
        self.core.setProperty(self.name, "SPIMNumSlices", num_slices)
        self.core.setProperty(self.name, "SPIMNumSlicesPerPiezo", slices_per_piezo)
        self.core.setProperty(self.name, "SPIMNumSides", num_sides)
        self.core.setProperty(self.name, "SPIMFirstSide", first_side)

    def configure_for_calibration(self):
        """
        Configure scanner for calibration (continuous light sheet for focus sweeps).

        Sets up:
        - Enabled beam
        - Laser output mode for triggering lasers
        - X-axis scanning (8° amplitude for full light sheet width)
        - Y-axis with minimal amplitude (will adjust offset for positioning)
        """
        self.enable_beam(True)
        self.set_laser_output_mode("shutter + side")  # Enable laser triggering
        self.configure_x_axis(amplitude_deg=8.0, offset_deg=0.0005)
        self.configure_y_axis(amplitude_deg=0.0001, offset_deg=0.0)
        self.core.waitForDevice(self.name)

    def configure_for_volume_acquisition(self,
                                          galvo_amplitude: float,
                                          galvo_center: float,
                                          num_slices: int,
                                          timing_params: Dict = None):
        """
        Configure scanner for hardware-triggered volume acquisition.

        Sets up the complete SPIM state machine for synchronized piezo/galvo/camera scanning.

        Parameters
        ----------
        galvo_amplitude : float
            Galvo Y-axis amplitude in degrees (matched to piezo amplitude)
        galvo_center : float
            Galvo Y-axis center offset in degrees
        num_slices : int
            Number of Z slices
        timing_params : Dict, optional
            Custom timing parameters (uses defaults if None)
        """
        # Reset state machine
        self.set_spim_state("Idle")
        self.set_laser_output_mode("shutter + side")
        self.enable_beam(False)

        # Configure scanning axes
        self.configure_x_axis(amplitude_deg=8.0, offset_deg=0.0005)
        self.configure_y_axis(amplitude_deg=galvo_amplitude, offset_deg=galvo_center)

        # Configure timing (use defaults if not provided)
        if timing_params is None:
            self.configure_spim_timing()
        else:
            self.configure_spim_timing(**timing_params)

        # Configure acquisition parameters
        self.configure_spim_parameters(num_slices=num_slices)

class DiSPIMLED:
    """
    DiSPIM LED control - works with bps.mv(led, state)

    ASI Tiger LED (LED:X:31) - LED shutter control via ConfigGroup
    Device-agnostic: any plan that sets device state will work
    """

    def __init__(self, core: pymmcore.CMMCore, name: str = "LED", group_name: str = None):
        self.core = core
        self.name = name
        self.group_name = group_name or name
        self.parent = None  # Required for Bluesky bps.mv()

        # Cache available configs (should be 'Open' and 'Closed')
        self._available_configs = self._get_available_configs()

    def _get_available_configs(self):
        """Get available LED configurations"""
        try:
            return list(self.core.getAvailableConfigs(self.group_name))
        except:
            return []

    def set(self, state: str):
        """Set LED state - called by bps.mv(led, 'Open') or bps.mv(led, 'Closed')"""
        if state not in self._available_configs:
            raise ValueError(f"State '{state}' not available. "
                           f"Available: {self._available_configs}")

        status = Status(obj=self, timeout=5)

        def wait():
            try:
                self.core.setConfig(self.group_name, state)
                self.core.waitForConfig(self.group_name, state)
            except Exception as exc:
                status.set_exception(exc)
            else:
                status.set_finished()

        import threading
        threading.Thread(target=wait).start()

        return status

    def read(self):
        """Read current LED configuration - required for Bluesky"""
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
        """Describe LED device - required for Bluesky"""
        data = OrderedDict()
        data[self.name] = {
            'source': self.name,
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

    def __init__(self, core: pymmcore.CMMCore, name: str = "Laser", group_name: str = None):
        self.core = core
        self.name = name
        self.group_name = group_name or name
        self.parent = None  # Required for Bluesky bps.mv()

        # Cache available configs
        self._available_configs = self._get_available_configs()

    def _get_available_configs(self):
        """Get available laser configurations"""
        try:
            return list(self.core.getAvailableConfigs(self.group_name))
        except:
            return []

    def set(self, config_name: str):
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


class DiSPIMVolumeScanner:
    """
    Compound device for hardware-triggered SPIM volume acquisition.

    Orchestrates camera, scanner, piezo, and lasers for synchronized 3D volume capture.
    Handles all the complexity of circular buffer management, rpyc transfer, state
    machine coordination, and automatic laser enable/disable.

    Laser Management:
    - Automatically enables configured lasers before acquisition
    - Automatically disables lasers after acquisition (prevents photobleaching!)
    - Always disables lasers on error (critical for sample health)
    - Laser timing during scan controlled by scanner's LaserOutputMode property

    This device encapsulates the entire hardware-triggered acquisition workflow,
    allowing plans to simply call trigger_and_read([volume_scanner]).

    Parameters
    ----------
    scanner : DiSPIMScanner
        Scanner device for galvo control
    camera : DiSPIMCamera
        Camera device for image acquisition
    piezo : DiSPIMPiezo
        Piezo device for Z-axis scanning
    laser_control : DiSPIMLaserControl
        Laser control device for managing laser configs.
        Required to ensure explicit laser management (no accidental photobleaching!)
    core : pymmcore.CMMCore
        Micro-Manager core instance
    name : str
        Device name (default: "volume_scanner")
    """

    def __init__(self,
                 scanner: DiSPIMScanner,
                 camera: DiSPIMCamera,
                 piezo: DiSPIMPiezo,
                 laser_control: 'DiSPIMLaserControl',
                 core: pymmcore.CMMCore,
                 name: str = "volume_scanner"):
        """
        Initialize volume scanner with all required devices.

        Note: laser_control is required because proper SPIM operation always
        requires explicit laser management to avoid photobleaching and ensure
        reproducible illumination.
        """
        self.name = name
        self.parent = None  # Required for Bluesky
        self.scanner = scanner
        self.camera = camera
        self.piezo = piezo
        self.laser_control = laser_control
        self.core = core

        self._last_volume = None
        self._last_volume_time = None
        self._configured = False

        # Configuration cache
        self._num_slices = None
        self._exposure_ms = None
        self._laser_config = None

    def configure(self,
                  num_slices: int,
                  exposure_ms: float,
                  galvo_amplitude: float,
                  galvo_center: float,
                  piezo_amplitude: float,
                  piezo_center: float,
                  laser_config: str = "488 and 561",
                  timing_params: Dict = None):
        """
        Configure all devices for hardware-triggered volume acquisition.

        Parameters
        ----------
        num_slices : int
            Number of Z slices in the volume
        exposure_ms : float
            Camera exposure time in milliseconds
        galvo_amplitude : float
            Galvo Y-axis amplitude in degrees (synchronized with piezo)
        galvo_center : float
            Galvo Y-axis center offset in degrees
        piezo_amplitude : float
            Piezo scanning amplitude in micrometers
        piezo_center : float
            Piezo center offset in micrometers
        laser_config : str
            Laser configuration name (default: "488 and 561").
            Common options: "488 and 561", "488 only", "561 only"
        timing_params : Dict, optional
            Custom SPIM timing parameters (uses defaults if None)
        """
        # Configure camera for hardware triggering
        self.camera.configure_for_volume_acquisition(exposure_ms)

        # Configure scanner for volume acquisition
        self.scanner.configure_for_volume_acquisition(
            galvo_amplitude=galvo_amplitude,
            galvo_center=galvo_center,
            num_slices=num_slices,
            timing_params=timing_params
        )

        # Configure piezo for volume acquisition
        self.piezo.configure_for_volume_acquisition(
            amplitude_um=piezo_amplitude,
            offset_um=piezo_center,
            num_slices=num_slices
        )

        self._num_slices = num_slices
        self._exposure_ms = exposure_ms
        self._laser_config = laser_config
        self._configured = True

    def trigger(self):
        """
        Start hardware-triggered volume acquisition.

        Returns
        -------
        Status
            Ophyd status object that finishes when volume is acquired
        """
        if not self._configured:
            raise RuntimeError("Device not configured. Call configure() first.")

        status = Status(obj=self, timeout=120)

        def wait():
            try:
                # Enable lasers
                self.core.setConfig(self.laser_control.group_name, self._laser_config)
                self.core.waitForConfig(self.laser_control.group_name, self._laser_config)

                # Prepare circular buffer
                self.core.clearCircularBuffer()
                buffer_capacity = self.core.getBufferTotalCapacity()
                if buffer_capacity < self._num_slices:
                    self.core.setCircularBufferMemoryFootprint(512)
                    self.core.waitForDevice(self.camera.name)

                # Start sequence acquisition
                self.core.prepareSequenceAcquisition(self.camera.name)
                self.core.waitForDevice(self.camera.name)
                self.core.startSequenceAcquisition(self.camera.name, self._num_slices, 0, True)
                self.core.waitForDevice(self.camera.name)

                # Trigger SPIM state machine
                self.scanner.set_spim_state("Running")

                # Collect images
                images = []
                timeout_s = 15  # Fixed 15 second timeout for volume acquisition
                start_time = time.time()

                while self.core.getRemainingImageCount() > 0 or self.core.isSequenceRunning():
                    if self.core.getRemainingImageCount() > 0:
                        img = self.core.popNextImage()

                        # Handle rpyc transfer
                        try:
                            img = rpyc.classic.obtain(img)
                        except (ImportError, AttributeError):
                            pass

                        images.append(img)

                    if time.time() - start_time > timeout_s:
                        raise TimeoutError(f"Volume acquisition timeout after {timeout_s:.1f}s")

                    time.sleep(0.01)

                # Stop sequence
                if self.core.isSequenceRunning():
                    self.core.stopSequenceAcquisition()

                # Reset hardware states
                self.camera.set_trigger_mode("INTERNAL")
                self.scanner.set_spim_state("Idle")
                self.piezo.set_spim_state("Idle")

                # Disable lasers (important for sample health!)
                self.core.setConfig(self.laser_control.group_name, "ALL OFF")
                self.core.waitForConfig(self.laser_control.group_name, "ALL OFF")

                # Store volume
                self._last_volume = np.array(images)
                self._last_volume_time = time.time()

            except Exception as exc:
                # Cleanup on error - always turn off lasers!
                try:
                    self.core.stopSequenceAcquisition()
                    self.camera.set_trigger_mode("INTERNAL")
                    self.scanner.set_spim_state("Idle")
                    self.piezo.set_spim_state("Idle")
                    # Critical: disable lasers even on error
                    self.core.setConfig(self.laser_control.group_name, "ALL OFF")
                    self.core.waitForConfig(self.laser_control.group_name, "ALL OFF")
                except:
                    pass
                status.set_exception(exc)
            else:
                status.set_finished()

        import threading
        threading.Thread(target=wait).start()

        return status

    def read(self):
        """Read acquired volume data."""
        if self._last_volume is not None:
            data = OrderedDict()
            data[self.name] = {
                'value': self._last_volume,
                'timestamp': self._last_volume_time or time.time()
            }
            return data
        else:
            return OrderedDict()

    def describe(self):
        """Describe volume data format."""
        data = OrderedDict()
        data[self.name] = {
            'source': self.name,
            'dtype': 'array',
            'shape': getattr(self._last_volume, 'shape', []),
            'units': 'counts'
        }
        return data

    def read_configuration(self):
        """Required for Bluesky."""
        return OrderedDict()

    def describe_configuration(self):
        """Required for Bluesky."""
        return OrderedDict()


class DiSPIMBottomCamera(DiSPIMCamera):
    """
    Specialized camera device for bottom-view embryo detection with transmitted light.

    Extends DiSPIMCamera with:
    - Automatic LED management (on before capture, off after)
    - Pixel calibration for stage coordinate conversion
    - Embryo-specific convenience methods

    LED Management:
    - Automatically turns LED on before capture
    - Automatically turns LED off after capture (prevents heating!)
    - Always turns LED off on error (critical for sample health)

    Used for finding and centering embryos in the sample chamber.
    """

    def __init__(self,
                 device_name: str,
                 core: pymmcore.CMMCore,
                 led_control: 'DiSPIMLED',
                 pixel_size_um: float = 6.5,
                 magnification: float = 10.0):
        """
        Initialize bottom camera with LED control and calibrated pixel size.

        Parameters
        ----------
        device_name : str
            Name of the camera device in Micro-Manager
        core : pymmcore.CMMCore
            Micro-Manager core instance
        led_control : DiSPIMLED
            LED device for transmitted light illumination.
            Required to ensure explicit LED management (no accidental sample heating!)
        pixel_size_um : float, optional
            Physical pixel size in micrometers (default: 6.5 for PCO camera)
        magnification : float, optional
            Objective magnification (default: 10.0 for 10x objective)
        """
        super().__init__(device_name, core)
        self.led_control = led_control
        self.pixel_size_um = pixel_size_um
        self.magnification = magnification
        self.effective_pixel_size = pixel_size_um / magnification
        self.use_led = True  # Set to False to disable automatic LED control

    def pixel_to_um(self, pixels: float) -> float:
        """
        Convert pixels to micrometers.

        Parameters
        ----------
        pixels : float
            Number of pixels

        Returns
        -------
        float
            Distance in micrometers
        """
        return pixels * self.effective_pixel_size

    def trigger(self):
        """
        Trigger image acquisition with optional LED management.

        Overrides parent trigger() to add LED control (if use_led=True):
        1. Turn LED on (if use_led=True)
        2. Capture image
        3. Turn LED off (if use_led=True, always even on error)

        Set self.use_led = False to capture without LED (ambient light only).

        Returns
        -------
        Status
            Ophyd status object that finishes when image is acquired
        """
        status = Status(obj=self, timeout=30)

        def wait():
            try:
                # Turn LED on for transmitted light imaging (if enabled)
                if self.use_led:
                    self.led_control.set("Open").wait(timeout=5)
                    time.sleep(0.1)  # Allow LED to stabilize

                # Capture image
                self.core.setCameraDevice(self.name)
                self.core.snapImage()
                self._last_image = rpyc.classic.obtain(self.core.getImage())
                self._last_image_time = time.time()

                # Turn LED off (if enabled - important to prevent sample heating!)
                if self.use_led:
                    self.led_control.set("Closed").wait(timeout=5)

            except Exception as exc:
                # Critical: always turn off LED even on error (if enabled)
                if self.use_led:
                    try:
                        self.led_control.set("Closed").wait(timeout=5)
                    except:
                        pass
                status.set_exception(exc)
            else:
                status.set_finished()

        import threading
        threading.Thread(target=wait).start()

        return status

    def capture_for_marking(self, exposure_ms: float):
        """
        Capture image configured for embryo marking.

        Convenience method that sets exposure and captures using internal trigger.
        LED is automatically managed by trigger().

        Parameters
        ----------
        exposure_ms : float
            Camera exposure time in milliseconds

        Returns
        -------
        Status
            Ophyd status object for the capture
        """
        self.core.setCameraDevice(self.name)
        self.core.setExposure(self.name, exposure_ms)
        self.core.waitForDevice(self.name)
        return self.trigger()  # LED automatically handled


class DiSPIMLightSheetSnap:
    """
    Compound device for single light sheet image acquisition during calibration.

    Combines scanner and camera for synchronized single-image snapshots.
    Used during focus sweeps and piezo-galvo calibration.
    """

    def __init__(self,
                 scanner: DiSPIMScanner,
                 camera: DiSPIMCamera,
                 name: str = "lightsheet_snap"):
        self.name = name
        self.parent = None  # Required for Bluesky
        self.scanner = scanner
        self.camera = camera

        self._last_image = None
        self._last_image_time = None

    def configure(self,
                  sheet_width_deg: float = 8.0,
                  y_position_deg: float = 0.0,
                  exposure_ms: float = 50.0):
        """
        Configure light sheet parameters for single snapshot.

        Parameters
        ----------
        sheet_width_deg : float
            Light sheet width (X-axis amplitude) in degrees (default: 8.0)
        y_position_deg : float
            Light sheet Y-position offset in degrees (default: 0.0)
        exposure_ms : float
            Camera exposure time in milliseconds (default: 50.0)
        """
        # Configure scanner for continuous light sheet
        self.scanner.configure_for_calibration()
        self.scanner.set_y_offset(y_position_deg)

        # Configure camera for single snapshot
        self.camera.configure_for_calibration(exposure_ms)

    def set_y_position(self, angle_deg: float):
        """
        Adjust light sheet Y-position for focus sweeps.

        Parameters
        ----------
        angle_deg : float
            Y-axis offset angle in degrees
        """
        self.scanner.set_y_offset(angle_deg)

    def trigger(self):
        """
        Capture single light sheet image.

        Returns
        -------
        Status
            Ophyd status object for the capture
        """
        return self.camera.trigger()

    def read(self):
        """Read captured image."""
        return self.camera.read()

    def describe(self):
        """Describe image data format."""
        return self.camera.describe()

    def read_configuration(self):
        """Required for Bluesky."""
        return OrderedDict()

    def describe_configuration(self):
        """Required for Bluesky."""
        return OrderedDict()


if __name__ == "__main__":
    # Example usage - would normally use actual MM paths
    logging.basicConfig(level=logging.INFO)
   