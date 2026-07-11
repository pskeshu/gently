"""
DiSPIM camera detector devices (single, dual, and bottom camera).
"""

import logging
import time
from collections import OrderedDict
from typing import TYPE_CHECKING, Any

import numpy as np
import pymmcore
from ophyd.status import Status

from gently.exceptions import AcquisitionError, HardwareError

from ._common import _safe_obtain

if TYPE_CHECKING:
    from .optical import DiSPIMLED

logger = logging.getLogger(__name__)


class DiSPIMCamera:
    """
    DiSPIM camera detector - works with bps.trigger_and_read([camera])

    Device-agnostic: any plan that acquires from a detector will work with this device
    """

    def __init__(self, device_name: str, core: pymmcore.CMMCore):
        self.name = device_name
        self.core = core
        self.parent = None  # Required for Bluesky
        self._last_image: Any = None
        self._last_image_time: float | None = None

    def _ensure_active(self) -> None:
        """Make this the default MMCore camera if it isn't already.

        Single source of truth for the set-camera step. Calling
        ``setCameraDevice`` unconditionally fires MMCore's
        ``onConfigGroupChanged`` callback and logs ``"Default camera set
        to ..."`` every time — at streaming cadence (≥4 Hz) that's noise.
        Querying ``getCameraDevice`` first is a cheap getter (no driver
        round-trip, no callbacks) and lets us skip the set entirely when
        the camera is already active.

        Applies uniformly to every DiSPIMCamera subclass — the bottom
        camera, HamCam, anything we add later — so the optimisation is
        not specific to the live streamer.
        """
        if self.core.getCameraDevice() != self.name:
            self.core.setCameraDevice(self.name)

    def trigger(self):
        """Trigger image acquisition - called by bps.trigger()"""

        def wait():
            try:
                # Set camera and snap
                self._ensure_active()
                self.core.snapImage()

                # Use _safe_obtain to transfer numpy array properly
                self._last_image = _safe_obtain(self.core.getImage())
                self._last_image_time = time.time()

            except (RuntimeError, AcquisitionError) as exc:
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
                "value": self._last_image,
                "timestamp": self._last_image_time or time.time(),
            }
            return data
        else:
            return OrderedDict()

    def describe(self):
        """Describe detector data format"""
        data = OrderedDict()
        data[self.name] = {
            "source": self.name,
            "dtype": "array",
            "shape": getattr(self._last_image, "shape", []),
        }
        return data

    def read_configuration(self):
        """Required for Bluesky"""
        return OrderedDict()

    def describe_configuration(self):
        """Required for Bluesky"""
        return OrderedDict()

    def snap(self) -> np.ndarray:
        """Synchronous single-frame capture.

        Unlike :meth:`trigger`, which spawns a thread and returns an ophyd
        Status object (right for Bluesky plan integration), this is a
        straight blocking call. Use it from continuous capture loops where
        the per-frame Status/thread overhead isn't worth paying — most
        notably the bottom-camera live-stream task in the device layer.

        Returns the captured frame as a numpy array. Also updates the
        device's ``_last_image`` / ``_last_image_time`` cache so that
        subsequent ``read()`` calls see the same value.

        Skips ``setCameraDevice`` when this camera is already the active
        one (via :meth:`_ensure_active`) — avoids spurious MMCore
        ``onConfigGroupChanged`` callbacks and log noise at streaming
        cadence.
        """
        self._ensure_active()
        self.core.snapImage()
        img = _safe_obtain(self.core.getImage())
        self._last_image = img
        self._last_image_time = time.time()
        return np.asarray(img)

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
        self._ensure_active()
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

    def configure_for_calibration(
        self, exposure_ms: float, roi: tuple[int, int, int, int] = (128, 896, 2048, 512)
    ):
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
        self._ensure_active()
        self.set_roi(*roi)
        self.set_trigger_mode("INTERNAL")
        self.set_sensor_mode("AREA")
        self.core.setExposure(self.name, exposure_ms)
        self.core.waitForDevice(self.name)

    def configure_for_volume_acquisition(
        self, exposure_ms: float, roi: tuple[int, int, int, int] = (128, 896, 2048, 512)
    ):
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
        self._ensure_active()
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
        except Exception:
            return 0.01  # Default 10ms

    @exposure_time.setter
    def exposure_time(self, value_ms):
        """Set exposure time in milliseconds"""
        try:
            self.core.setExposure(value_ms)
        except (RuntimeError, HardwareError) as e:
            logger.error("Failed to set exposure: %s", e)


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
            stitched_image = camera_data[self.camera.name]["value"]
            timestamp = camera_data[self.camera.name]["timestamp"]

            # Split image in the middle (width dimension)
            height, width = stitched_image.shape[:2]
            mid_width = width // 2

            image_a = stitched_image[:, :mid_width]  # Left half
            image_b = stitched_image[:, mid_width:]  # Right half

            # Return as separate data entries
            data = OrderedDict()
            data[f"{self.name}_image_a"] = {"value": image_a, "timestamp": timestamp}
            data[f"{self.name}_image_b"] = {"value": image_b, "timestamp": timestamp}
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
            original_shape = camera_desc[self.camera.name].get("shape", [])
            if len(original_shape) >= 2:
                # Split width dimension in half
                split_shape = [original_shape[0], original_shape[1] // 2]
                if len(original_shape) > 2:
                    split_shape.extend(original_shape[2:])
            else:
                split_shape = original_shape

            data[f"{self.name}_image_a"] = {
                "source": f"{self.name}_image_a",
                "dtype": "array",
                "shape": split_shape,
            }
            data[f"{self.name}_image_b"] = {
                "source": f"{self.name}_image_b",
                "dtype": "array",
                "shape": split_shape,
            }

        return data

    def read_configuration(self):
        """Required for Bluesky"""
        return OrderedDict()

    def describe_configuration(self):
        """Required for Bluesky"""
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

    def __init__(
        self,
        device_name: str,
        core: pymmcore.CMMCore,
        led_control: "DiSPIMLED",  # noqa: F821
        pixel_size_um: float = 6.5,
        magnification: float = 10.0,
    ):
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
        # Retained for API compatibility but ignored: the bottom camera never
        # drives the LED (see trigger()). Imaging uses room light only.
        self.use_led = False

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
        Trigger image acquisition.

        The bottom camera NEVER drives the LED — imaging is done under room
        light only. The ``use_led`` flag is retained for API compatibility but
        is intentionally ignored here so that no caller (manual marking,
        detection, live preview, …) can ever flash the LED.

        Returns
        -------
        Status
            Ophyd status object that finishes when image is acquired
        """
        status = Status(obj=self, timeout=30)

        def wait():
            try:
                # LED is never used — capture under ambient/room light only.
                self._ensure_active()
                self.core.snapImage()
                self._last_image = _safe_obtain(self.core.getImage())
                self._last_image_time = time.time()

            except Exception as exc:
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
        self._ensure_active()
        self.core.setExposure(self.name, exposure_ms)
        self.core.waitForDevice(self.name)
        return self.trigger()  # LED automatically handled
