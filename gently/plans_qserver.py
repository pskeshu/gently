"""
Queue Server Utility Plans for Gently DiSPIM

Simple plans that wrap common device operations for use with Bluesky Queue Server.
These plans allow basic operations (move, read, capture) to be submitted through
the queue server API.

These plans are loaded by backend/queue_server_startup.py and become available
in the queue server namespace.
"""

import bluesky.plan_stubs as bps
import bluesky.plans as bp
from bluesky.preprocessors import run_wrapper
from typing import Generator, Any


def move_stage_plan(xy_stage, x: float, y: float) -> Generator[Any, Any, dict]:
    """
    Move XY stage to specified position.

    Parameters
    ----------
    xy_stage : ophyd.Device
        The XY stage device
    x : float
        Target X position in micrometers
    y : float
        Target Y position in micrometers

    Returns
    -------
    dict
        Result with new position

    Example (via queue server API)
    ------------------------------
    >>> from bluesky_queueserver_api import BPlan
    >>> plan = BPlan("move_stage_plan", xy_stage="xy_stage", x=1000.0, y=500.0)
    >>> rm.item_add(plan)
    """
    yield from bps.mv(xy_stage, [x, y])
    return {'x': x, 'y': y, 'success': True}


def read_stage_plan(xy_stage) -> Generator[Any, Any, None]:
    """
    Read current XY stage position.

    The position is stored in the run data and can be retrieved from Databroker.

    Parameters
    ----------
    xy_stage : ophyd.Device
        The XY stage device

    Example (via queue server API)
    ------------------------------
    >>> plan = BPlan("read_stage_plan", xy_stage="xy_stage")
    >>> rm.item_add(plan)
    >>> # After completion, retrieve from databroker:
    >>> run = db[-1]
    >>> pos = run.primary.read()['xy_stage'].values[0]
    """
    # Use bp.count which handles run wrapper automatically
    yield from bp.count([xy_stage], num=1)


def capture_bottom_image_plan(bottom_camera, led=None) -> Generator[Any, Any, None]:
    """
    Capture a single image from the bottom camera.

    Optionally controls LED lighting during capture.

    Parameters
    ----------
    bottom_camera : ophyd.Device
        The bottom camera device
    led : ophyd.Device, optional
        LED device for illumination

    Example (via queue server API)
    ------------------------------
    >>> plan = BPlan("capture_bottom_image_plan", bottom_camera="bottom_camera", led="led")
    >>> rm.item_add(plan)
    >>> # After completion, retrieve image from databroker:
    >>> run = db[-1]
    >>> image = run.primary.read()['bottom_camera'].values[0]
    """
    # Turn on LED if provided (uses 'Open' for on, 'Closed' for off - MM ConfigGroup presets)
    if led is not None:
        try:
            yield from bps.mv(led, 'Open')
        except Exception:
            pass  # LED control is optional

    # Capture image using bp.count which handles run wrapper
    yield from bp.count([bottom_camera], num=1)

    # Turn off LED
    if led is not None:
        try:
            yield from bps.mv(led, 'Closed')
        except Exception:
            pass


def capture_lightsheet_image_plan(
    lightsheet_snap,
    scanner,
    piezo,
    piezo_position: float = 50.0,
    galvo_position: float = 0.0
) -> Generator[Any, Any, None]:
    """
    Capture a single lightsheet image at specified piezo/galvo positions.

    Parameters
    ----------
    lightsheet_snap : ophyd.Device
        The lightsheet snap device (camera + scanner)
    scanner : ophyd.Device
        Scanner device for galvo control
    piezo : ophyd.Device
        Piezo stage device
    piezo_position : float
        Piezo position in micrometers
    galvo_position : float
        Galvo position in degrees

    Example (via queue server API)
    ------------------------------
    >>> plan = BPlan("capture_lightsheet_image_plan",
    ...              lightsheet_snap="lightsheet_snap",
    ...              scanner="scanner", piezo="piezo",
    ...              piezo_position=50.0, galvo_position=0.0)
    """
    # Move to positions
    yield from bps.mv(piezo, piezo_position)
    yield from bps.mv(scanner.sa_offset_y, galvo_position)

    # Capture using bp.count which handles run wrapper
    yield from bp.count([lightsheet_snap], num=1)


def move_piezo_plan(piezo, position: float) -> Generator[Any, Any, dict]:
    """
    Move piezo to specified position.

    Parameters
    ----------
    piezo : ophyd.Device
        The piezo device
    position : float
        Target position in micrometers

    Returns
    -------
    dict
        Result with new position
    """
    yield from bps.mv(piezo, position)
    return {'position': position, 'success': True}


def move_scanner_plan(scanner, offset_y: float) -> Generator[Any, Any, dict]:
    """
    Move scanner galvo to specified offset.

    Parameters
    ----------
    scanner : ophyd.Device
        The scanner device
    offset_y : float
        Y offset (galvo position) in degrees

    Returns
    -------
    dict
        Result with new position
    """
    yield from bps.mv(scanner.sa_offset_y, offset_y)
    return {'offset_y': offset_y, 'success': True}


def set_laser_plan(laser_control, state: str = 'ON') -> Generator[Any, Any, dict]:
    """
    Set laser state.

    Parameters
    ----------
    laser_control : ophyd.Device
        The laser control device
    state : str
        'ON' or 'OFF'

    Returns
    -------
    dict
        Result with new state
    """
    yield from bps.mv(laser_control.state, state)
    return {'state': state, 'success': True}


def set_led_plan(led, state: str = 'Open') -> Generator[Any, Any, dict]:
    """
    Set LED state.

    Parameters
    ----------
    led : ophyd.Device
        The LED device
    state : str
        'Open' (on) or 'Closed' (off) - Micro-Manager ConfigGroup presets

    Returns
    -------
    dict
        Result with new state
    """
    yield from bps.mv(led, state)
    return {'state': state, 'success': True}
