"""
diSPIM hardware module.

Exports hardware knowledge for the dual-view inverted Selective Plane
Illumination Microscopy (diSPIM) system.
"""

from .description import HARDWARE_DESCRIPTION

HARDWARE_NAME = "dispim"
HARDWARE_DISPLAY_NAME = "diSPIM"
CAPABILITIES = {
    "xy_stage",
    "z_control",
    "volume",
    "snap",
    "dual_view",
    "detection",
    "fluorescence",
    "transmitted",
}


def create_device_layer(config: dict):
    """Create and return a DeviceLayerServer for diSPIM hardware.

    Parameters
    ----------
    config : dict
        Configuration dict with keys: 'config_path', 'sam_device'

    Returns
    -------
    DeviceLayerServer
        The server instance (call .run(port=N) to start)
    """
    from .device_layer import DeviceLayerServer
    return DeviceLayerServer(
        config_path=config.get('config_path', 'config/config.yml'),
        sam_device=config.get('sam_device', 'cuda'),
    )


def create_client(http_url: str):
    """Create an HTTP client for communicating with the diSPIM device layer.

    Parameters
    ----------
    http_url : str
        URL of the device layer server (e.g., "http://127.0.0.1:60610")

    Returns
    -------
    QueueServerClient
        The client instance (call .connect() before use)
    """
    from gently.app.queue_server_client import QueueServerClient
    return QueueServerClient(http_url=http_url)
