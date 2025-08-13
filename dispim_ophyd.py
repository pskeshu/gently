"""
DiSPIM Ophyd Device Classes

Ophyd device abstraction layer for DiSPIM microscope components.
Provides standardized device interfaces for use with Bluesky plans.
"""

import time
import logging
from typing import Dict, Any, Optional, List
from dataclasses import dataclass
import numpy as np

try:
    from ophyd import Device, Component as Cpt, Signal, DeviceStatus
    from ophyd import PositionerBase, DetectorBase
    from ophyd.signal import SignalRO
    from ophyd.status import SubscriptionStatus
except ImportError:
    raise ImportError("ophyd is required. Install with: pip install ophyd")

from dispim_config import DiSPIMCore, DiSPIMConfig
from dispim_calibration import DiSPIMCalibration


class PyMMCoreSignal(Signal):
    """Signal that interfaces with pymmcore device properties"""
    
    def __init__(self, device_name: str, property_name: str = None, 
                 core: DiSPIMCore = None, **kwargs):
        super().__init__(**kwargs)
        self.device_name = device_name
        self.property_name = property_name or "Position"
        self.core = core
        self._readback_value = 0.0
    
    def get(self):
        """Get current value from device"""
        if self.core and hasattr(self.core.core, 'getProperty'):
            try:
                return float(self.core.core.getProperty(self.device_name, self.property_name))
            except:
                pass
        return self._readback_value
    
    def put(self, value, **kwargs):
        """Set device value"""
        if self.core and hasattr(self.core.core, 'setProperty'):
            try:
                self.core.core.setProperty(self.device_name, self.property_name, str(value))
                self._readback_value = float(value)
                return DeviceStatus(self)
            except Exception as e:
                self.log.error(f"Failed to set {self.device_name}.{self.property_name}: {e}")
        
        self._readback_value = float(value)
        status = DeviceStatus(self)
        status.set_finished()
        return status


class DiSPIMPiezo(PositionerBase):
    """Ophyd device for DiSPIM piezo actuators"""
    
    position = Cpt(PyMMCoreSignal, '', kind='hinted')
    user_readback = Cpt(PyMMCoreSignal, '', kind='hinted')
    
    def __init__(self, device_name: str, core: DiSPIMCore, **kwargs):
        self.device_name = device_name
        self.core = core
        
        # Configure signals
        kwargs['position'] = PyMMCoreSignal(device_name=device_name, core=core)
        kwargs['user_readback'] = PyMMCoreSignal(device_name=device_name, core=core)
        
        super().__init__(**kwargs)
        
        # Safety limits from config
        self.limits = (core.config.piezo_min, core.config.piezo_max)
        
    def move(self, position, **kwargs):
        """Move piezo to position"""
        self.log.info(f"Moving {self.device_name} to {position}")
        
        # Safety check
        if not (self.limits[0] <= position <= self.limits[1]):
            raise ValueError(f"Position {position} outside limits {self.limits}")
        
        try:
            self.core.set_piezo_position(self.device_name, position)
            
            # Create status object
            def check_done():
                current = self.core.get_device_position(self.device_name)
                return abs(current - position) < 0.1  # 0.1 μm tolerance
            
            status = SubscriptionStatus(self.user_readback, check_done, timeout=10.0)
            return status
            
        except Exception as e:
            self.log.error(f"Failed to move {self.device_name}: {e}")
            raise


class DiSPIMGalvo(PositionerBase):
    """Ophyd device for DiSPIM galvanometer/micromirror"""
    
    position = Cpt(PyMMCoreSignal, '', kind='hinted')
    user_readback = Cpt(PyMMCoreSignal, '', kind='hinted')
    
    def __init__(self, core: DiSPIMCore, **kwargs):
        self.core = core
        device_name = core.config.galvo_device
        
        # Configure signals
        kwargs['position'] = PyMMCoreSignal(device_name=device_name, core=core)
        kwargs['user_readback'] = PyMMCoreSignal(device_name=device_name, core=core)
        
        super().__init__(**kwargs)
        
        # Safety limits from config
        self.limits = (core.config.galvo_min, core.config.galvo_max)
        
    def move(self, position, **kwargs):
        """Move galvo to position"""
        self.log.info(f"Moving galvo to {position}")
        
        # Safety check
        if not (self.limits[0] <= position <= self.limits[1]):
            raise ValueError(f"Position {position} outside limits {self.limits}")
        
        try:
            self.core.set_galvo_position(position)
            
            # Create status object
            def check_done():
                current = self.core.get_device_position(self.core.config.galvo_device)
                return abs(current - position) < 0.01  # 0.01° tolerance
            
            status = SubscriptionStatus(self.user_readback, check_done, timeout=5.0)
            return status
            
        except Exception as e:
            self.log.error(f"Failed to move galvo: {e}")
            raise


class DiSPIMCamera(DetectorBase):
    """Ophyd device for DiSPIM camera"""
    
    def __init__(self, core: DiSPIMCore, **kwargs):
        self.core = core
        super().__init__(**kwargs)
        
        self._acquiring = False
        self._last_image = None
        
    @property
    def exposure_time(self):
        """Current exposure time in seconds"""
        try:
            return self.core.core.getExposure() / 1000.0  # Convert ms to s
        except:
            return self.core.config.exposure_time / 1000.0
    
    @exposure_time.setter
    def exposure_time(self, value):
        """Set exposure time in seconds"""
        exposure_ms = value * 1000.0
        self.core.set_exposure(exposure_ms)
    
    def trigger(self):
        """Trigger single image acquisition"""
        self.log.debug("Triggering camera")
        
        def acquisition_complete():
            # Start acquisition
            try:
                self._last_image = self.core.snap_image()
                self._acquiring = False
                return True
            except Exception as e:
                self.log.error(f"Camera acquisition failed: {e}")
                self._acquiring = False
                return False
        
        self._acquiring = True
        status = SubscriptionStatus(lambda: not self._acquiring, 
                                  timeout=self.exposure_time + 5.0)
        
        # Run acquisition in background
        import threading
        thread = threading.Thread(target=acquisition_complete)
        thread.daemon = True
        thread.start()
        
        return status
    
    def read(self):
        """Read the most recent image data"""
        if self._last_image is not None:
            return {
                f'{self.name}_image': {
                    'value': self._last_image,
                    'timestamp': time.time()
                }
            }
        else:
            return {}
    
    def describe(self):
        """Describe the detector data"""
        return {
            f'{self.name}_image': {
                'source': f'DiSPIM Camera {self.core.config.camera_device}',
                'shape': getattr(self._last_image, 'shape', []),
                'dtype': 'array',
                'external': 'FILESTORE:TIFF'  # For large image storage
            }
        }


class DiSPIMLightSheet(Device):
    """Composite device for synchronized light sheet control"""
    
    piezo_imaging = Cpt(DiSPIMPiezo, '')
    piezo_illumination = Cpt(DiSPIMPiezo, '')
    galvo = Cpt(DiSPIMGalvo)
    camera = Cpt(DiSPIMCamera)
    
    def __init__(self, core: DiSPIMCore, calibration: DiSPIMCalibration = None, **kwargs):
        self.core = core
        self.calibration = calibration
        
        # Initialize component devices
        kwargs['piezo_imaging'] = DiSPIMPiezo(
            device_name=core.config.piezo_imaging, 
            core=core, 
            name='piezo_imaging'
        )
        kwargs['piezo_illumination'] = DiSPIMPiezo(
            device_name=core.config.piezo_illumination,
            core=core,
            name='piezo_illumination'
        )
        kwargs['galvo'] = DiSPIMGalvo(core=core, name='galvo')
        kwargs['camera'] = DiSPIMCamera(core=core, name='camera')
        
        super().__init__(**kwargs)
    
    def synchronized_move(self, piezo_position):
        """Move piezo and galvo in synchronization"""
        self.log.info(f"Synchronized move to piezo position {piezo_position}")
        
        # Calculate galvo position using calibration
        galvo_position = self.core.compute_galvo_from_piezo(piezo_position)
        
        # Move both devices
        piezo_status = self.piezo_imaging.move(piezo_position)
        galvo_status = self.galvo.move(galvo_position)
        
        # Return combined status
        from ophyd.status import AndStatus
        return AndStatus(piezo_status, galvo_status)
    
    def center_devices(self):
        """Center all positioning devices"""
        self.log.info("Centering light sheet devices")
        
        statuses = []
        statuses.append(self.piezo_imaging.move(self.core.config.piezo_center))
        statuses.append(self.piezo_illumination.move(self.core.config.piezo_center))
        statuses.append(self.galvo.move(self.core.config.galvo_center))
        
        from ophyd.status import AndStatus
        return AndStatus(*statuses)
    
    def run_autofocus(self, device='imaging'):
        """Run autofocus on specified device"""
        if not self.calibration:
            raise RuntimeError("Calibration object required for autofocus")
        
        device_map = {
            'imaging': self.core.config.piezo_imaging,
            'illumination': self.core.config.piezo_illumination,
            'galvo': self.core.config.galvo_device
        }
        
        device_name = device_map.get(device)
        if not device_name:
            raise ValueError(f"Unknown device for autofocus: {device}")
        
        optimal_position = self.calibration.run_autofocus(device_name)
        self.log.info(f"Autofocus on {device} complete: {optimal_position:.2f}")
        
        return optimal_position
    
    def calibrate_two_point(self, point1, point2, auto_focus=True):
        """Perform two-point calibration"""
        if not self.calibration:
            raise RuntimeError("Calibration object required for two-point calibration")
        
        result = self.calibration.two_point_calibration(point1, point2, auto_focus)
        
        if result.is_valid:
            self.log.info(f"Calibration successful: slope={result.slope:.4f}, "
                         f"offset={result.offset:.2f}")
        else:
            self.log.error("Calibration failed validation")
        
        return result
    
    def stage_scan_positions(self, start, stop, num_slices):
        """Generate positions for stage scanning"""
        return np.linspace(start, stop, num_slices)
    
    def read(self):
        """Read all device positions and camera data"""
        result = {}
        
        # Read positions
        result.update(self.piezo_imaging.read())
        result.update(self.piezo_illumination.read())
        result.update(self.galvo.read())
        
        # Read camera if available
        result.update(self.camera.read())
        
        return result
    
    def describe(self):
        """Describe all device data"""
        result = {}
        result.update(self.piezo_imaging.describe())
        result.update(self.piezo_illumination.describe())
        result.update(self.galvo.describe())
        result.update(self.camera.describe())
        
        return result


def create_dispim_devices(config_file: str = None) -> DiSPIMLightSheet:
    """Factory function to create configured DiSPIM device ensemble"""
    
    # Initialize core
    core = DiSPIMCore(config_file)
    
    # Initialize calibration
    calibration = DiSPIMCalibration(core)
    
    # Create composite device
    light_sheet = DiSPIMLightSheet(
        core=core,
        calibration=calibration,
        name='dispim_light_sheet'
    )
    
    return light_sheet


if __name__ == "__main__":
    # Example usage
    logging.basicConfig(level=logging.INFO)
    
    # This would normally be called with actual config file
    # dispim = create_dispim_devices("/path/to/dispim_config.cfg")
    
    print("DiSPIM Ophyd devices created successfully")
    print("Available devices: DiSPIMPiezo, DiSPIMGalvo, DiSPIMCamera, DiSPIMLightSheet")