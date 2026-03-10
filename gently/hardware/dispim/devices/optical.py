"""
DiSPIM optical control devices (LED and laser).
"""

import time
import logging
from collections import OrderedDict

from ophyd.status import Status
import pymmcore

logger = logging.getLogger(__name__)


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
