"""
Hardware Control Tools

Tools for controlling microscope hardware including stage movement,
LED control, calibration, and image acquisition.
"""

from typing import Dict, List
import json

from ..tool_registry import tool, ToolCategory, ToolExample
from ..tool_helpers import require_copilot, get_embryo_or_error


@tool(
    name="move_to_embryo",
    description="""Move the XY stage to a specific embryo's stored position. The embryo must have been detected and have a valid stage_position.
Use when user says "go to embryo X", "move to embryo X", or before imaging a specific embryo.
This only moves XY - piezo/galvo are controlled separately during acquisition. Movement takes ~0.5 seconds.""",
    category=ToolCategory.MOVEMENT,
    requires_microscope=True,
    examples=[
        ToolExample("Go to embryo 1", {"embryo_id": "embryo_1"}),
        ToolExample("Move to embryo 3", {"embryo_id": "embryo_3"}),
    ],
)
async def move_to_embryo(embryo_id: str, context: Dict) -> str:
    """Move stage to embryo position"""
    copilot = context.get('copilot')
    client = context.get('client')

    if not copilot:
        return "Error: No copilot context"

    embryo, err = get_embryo_or_error(copilot, embryo_id)
    if err:
        return err

    if not embryo.stage_position:
        return f"Embryo '{embryo_id}' has no stored position. Run calibration first."

    try:
        x = embryo.stage_position.get('x', 0)
        y = embryo.stage_position.get('y', 0)
        await client.move_to_position(x, y)

        return f"Moved to {embryo_id}\nPosition: ({x:.2f}, {y:.2f}) um"

    except Exception as e:
        import traceback
        return f"Error moving to embryo: {str(e)}\n{traceback.format_exc()}"


@tool(
    name="get_stage_position",
    description="""Get the current XY stage position in micrometers. Returns the real-time position from the hardware.
Use when user asks "where is the stage?", "current position?", or when you need to know the microscope's current location.
This reads from hardware - different from embryo stored positions which are in the experiment data.""",
    category=ToolCategory.HARDWARE,
    requires_microscope=True,
    examples=[
        ToolExample("Where is the stage?", {}),
        ToolExample("Current XY position?", {}),
    ],
)
async def get_stage_position(context: Dict) -> str:
    """Get current stage position"""
    client = context.get('client')

    if not client:
        return "Error: No microscope client connected"

    try:
        pos = await client.get_stage_position()
        return f"Current stage position: X={pos[0]:.1f} µm, Y={pos[1]:.1f} µm"

    except Exception as e:
        return f"Error reading stage position: {str(e)}"


@tool(
    name="set_led",
    description="Set the LED illumination state",
    category=ToolCategory.HARDWARE,
    requires_microscope=True,
)
async def set_led(state: str, context: Dict) -> str:
    """Set LED state"""
    client = context.get('client')

    try:
        result = await client.set_led(state)
        if result.get('success'):
            return f"LED set to '{state}'"
        else:
            return f"Error setting LED: {result.get('error', 'Unknown error')}"
    except Exception as e:
        return f"Error setting LED: {str(e)}"


@tool(
    name="get_led_status",
    description="Get the current LED illumination status",
    category=ToolCategory.HARDWARE,
    requires_microscope=True,
)
async def get_led_status(context: Dict) -> str:
    """Get LED status"""
    client = context.get('client')

    try:
        result = await client.get_led_status()
        if result.get('success'):
            current = result.get('current_state', 'unknown')
            available = result.get('available_configs', [])
            group = result.get('group_name', 'unknown')

            return (f"LED Status:\n"
                    f"  Current state: {current}\n"
                    f"  ConfigGroup: {group}\n"
                    f"  Available configs: {available}")
        else:
            return f"Error getting LED status: {result.get('error', 'Unknown error')}"
    except Exception as e:
        return f"Error getting LED status: {str(e)}"


@tool(
    name="calibrate_embryo",
    description="""Run piezo-galvo calibration for a specific embryo to find optimal imaging parameters. Automatically moves to embryo position first.
Use after detection to prepare an embryo for volume acquisition. Calibration finds piezo_center and galvo_center values.
Required before acquire_volume - stores calibration in embryo state. Takes ~30 seconds per embryo.""",
    category=ToolCategory.HARDWARE,
    requires_microscope=True,
    examples=[
        ToolExample("Calibrate embryo 1", {"embryo_id": "embryo_1"}),
    ],
)
async def calibrate_embryo(
    embryo_id: str,
    piezo_positions: List[float] = None,
    context: Dict = None
) -> str:
    """Calibrate embryo piezo-galvo - moves to embryo first, then calibrates"""
    copilot = context.get('copilot')
    client = context.get('client')

    if not copilot:
        return "Error: No copilot context"

    embryo, err = get_embryo_or_error(copilot, embryo_id)
    if err:
        return err

    positions = piezo_positions or [40.0, 60.0]

    try:
        # First move to embryo position
        pos = embryo.stage_position
        if pos and pos.get('x') is not None and pos.get('y') is not None:
            print(f"  Moving to {embryo.id} at ({pos['x']:.1f}, {pos['y']:.1f})...")
            await client.move_to_position(pos['x'], pos['y'])
        else:
            print(f"  Warning: No position stored for {embryo.id}, calibrating at current position")

        # Run calibration at current position
        print(f"  Running piezo-galvo calibration...")
        result = await client.calibrate_piezo_galvo(piezo_positions=positions)

        if result.get('success'):
            embryo.calibration = result.get('calibration', {})
            copilot._mark_significant_action("calibration")
            return f"Calibrated {embryo.id}\nCalibration: {json.dumps(result.get('calibration', {}), indent=2)}"
        else:
            return f"Calibration failed: {result.get('error', 'Unknown error')}"

    except Exception as e:
        return f"Error calibrating embryo: {str(e)}"


@tool(
    name="acquire_volume",
    description="""Acquire a single 3D lightsheet volume for a specific embryo. Moves to embryo position and uses its calibration data.
Use when user wants a full 3D stack of an embryo (e.g., "acquire volume of embryo 1", "take a 3D image").
Embryo must be calibrated first. Default 50 slices at 10ms exposure takes ~2.5 seconds. Turns laser on during acquisition.""",
    category=ToolCategory.HARDWARE,
    requires_microscope=True,
    examples=[
        ToolExample("Acquire volume of embryo 1", {"embryo_id": "embryo_1"}),
        ToolExample("Take a 3D image of embryo 2 with 80 slices", {"embryo_id": "embryo_2", "num_slices": 80}),
    ],
)
async def acquire_volume(
    embryo_id: str,
    num_slices: int = 50,
    exposure_ms: float = 10.0,
    context: Dict = None
) -> str:
    """Acquire single volume - moves to embryo first, uses calibration"""
    copilot = context.get('copilot')
    client = context.get('client')

    if not copilot:
        return "Error: No copilot context"

    embryo, err = get_embryo_or_error(copilot, embryo_id)
    if err:
        return err

    try:
        # Move to embryo position first
        pos = embryo.stage_position
        if pos and pos.get('x') is not None and pos.get('y') is not None:
            print(f"  Moving to {embryo.id} at ({pos['x']:.1f}, {pos['y']:.1f})...")
            await client.move_to_position(pos['x'], pos['y'])
        else:
            print(f"  Warning: No position stored for {embryo.id}, acquiring at current position")

        # Get calibration parameters (use defaults if not calibrated)
        cal = embryo.calibration or {}
        galvo_amplitude = cal.get('galvo_amplitude', 0.5)
        galvo_center = cal.get('galvo_center', 0.0)
        piezo_amplitude = cal.get('piezo_amplitude', 25.0)
        piezo_center = cal.get('piezo_center', 50.0)

        if not embryo.calibration:
            print(f"  Warning: {embryo.id} not calibrated, using default parameters")

        print(f"  Acquiring {num_slices}-slice volume...")
        result = await client.acquire_volume(
            num_slices=num_slices,
            exposure_ms=exposure_ms,
            galvo_amplitude=galvo_amplitude,
            galvo_center=galvo_center,
            piezo_amplitude=piezo_amplitude,
            piezo_center=piezo_center
        )

        if result.get('success'):
            # Update embryo state
            embryo.timepoints_acquired += 1
            from datetime import datetime
            embryo.last_imaged = datetime.now()
            return f"Acquired volume for {embryo.id}\nShape: {result.get('shape', 'unknown')}"
        else:
            return f"Acquisition failed: {result.get('error', 'Unknown error')}"

    except Exception as e:
        return f"Error acquiring volume: {str(e)}"


@tool(
    name="view_image",
    description="""Capture and display the current bottom camera widefield image. Shows what's visible at the current stage position.
Use when user says "show me the view", "take a picture", "what does it look like?", or to check sample positioning.
This is the widefield/brightfield camera looking up at the sample - good for seeing embryo outlines and overall positioning.
Image is automatically saved to camera_captures/ folder.""",
    category=ToolCategory.HARDWARE,
    requires_microscope=True,
    examples=[
        ToolExample("Show me the current view", {}),
        ToolExample("What does the sample look like?", {}),
    ],
)
async def view_image(
    title: str = "Bottom Camera Image",
    exposure_ms: float = None,
    show: bool = True,
    context: Dict = None
) -> str:
    """Capture and display bottom camera image"""
    client = context.get('client')

    try:
        print(f"  Capturing bottom camera image...")
        image = await client.capture_bottom_image(exposure_ms=exposure_ms)

        if image is None or image.shape == (100, 100):
            return "Failed to capture image from bottom camera"

        if show:
            from datetime import datetime
            from pathlib import Path
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            save_path = f"camera_captures/bottom_camera_{timestamp}.png"
            Path("camera_captures").mkdir(exist_ok=True)

            print(f"  Displaying image...")
            view_result = await client.view_image(
                image=image,
                title=title,
                save_path=save_path,
                show=True
            )
            return f"Captured bottom camera image ({image.shape[0]}x{image.shape[1]})\nSaved to: {save_path}"
        else:
            return f"Captured bottom camera image ({image.shape[0]}x{image.shape[1]})"

    except Exception as e:
        return f"Error capturing image: {str(e)}"


@tool(
    name="capture_lightsheet",
    description="""Capture a single 2D lightsheet fluorescence image at specified piezo/galvo position. Uses 50ms exposure by default.
Use when user says "take a lightsheet image", "lightsheet snap", or wants to see fluorescence at a specific Z position.
This is a COMPLETE action - do NOT follow up with acquire_volume unless user explicitly asks for a 3D volume.
If piezo_position is not specified, uses the CURRENT piezo position (preserves focus after fine_focus).
The galvo_position (default 0) controls light sheet Y offset.""",
    category=ToolCategory.HARDWARE,
    requires_microscope=True,
    examples=[
        ToolExample("Take a lightsheet image", {}),
        ToolExample("Lightsheet snap at piezo 50", {"piezo_position": 50.0}),
    ],
)
async def capture_lightsheet(
    piezo_position: float = None,
    galvo_position: float = 0.0,
    show: bool = True,
    context: Dict = None
) -> str:
    """Capture and optionally display a single lightsheet image"""
    client = context.get('client')

    try:
        # If no piezo position specified, use current position
        if piezo_position is None:
            piezo_position = await client.get_piezo_position()
            print(f"  Using current piezo position: {piezo_position:.1f}um")

        print(f"  Capturing lightsheet at piezo={piezo_position}um, galvo={galvo_position}V...")
        result = await client.capture_lightsheet_image(
            piezo_position=piezo_position,
            galvo_position=galvo_position
        )

        if result.get('success'):
            image = result.get('image')
            run_uid = result.get('run_uid', 'unknown')

            if image is not None and show:
                # Display the image
                from datetime import datetime
                from pathlib import Path
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                save_path = f"lightsheet_captures/lightsheet_{timestamp}.png"
                Path("lightsheet_captures").mkdir(exist_ok=True)

                print(f"  Displaying lightsheet image...")
                view_result = await client.view_image(
                    image=image,
                    title=f"Lightsheet: piezo={piezo_position}um, galvo={galvo_position}V",
                    save_path=save_path,
                    show=True
                )
                return f"Captured lightsheet image at piezo={piezo_position}um, galvo={galvo_position}V\nSaved to: {save_path}\nRun UID: {run_uid}"
            elif image is None:
                return f"Lightsheet captured at piezo={piezo_position}um, galvo={galvo_position}V (image not displayed - databroker retrieval issue)\nRun UID: {run_uid}"
            else:
                return f"Captured lightsheet at piezo={piezo_position}um, galvo={galvo_position}V\nRun UID: {run_uid}"
        else:
            return f"Failed: {result.get('error', 'Unknown error')}"

    except Exception as e:
        return f"Error capturing lightsheet: {str(e)}"
