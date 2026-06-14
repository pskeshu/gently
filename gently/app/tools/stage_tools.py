"""
Stage Movement Tools

Tools for controlling microscope XY stage movement.
"""

from gently.harness.tools.helpers import ctx_get, get_embryo_or_error
from gently.harness.tools.registry import ToolCategory, ToolExample, tool


@tool(
    name="move_to_embryo",
    description="""Move the XY stage to a specific embryo's stored position. The embryo must
have been detected and have a valid stage_position.
Use when user says "go to embryo X", "move to embryo X", or before imaging a specific embryo.
This only moves XY - piezo/galvo are controlled separately during acquisition. Movement
takes ~0.5 seconds.""",
    category=ToolCategory.MOVEMENT,
    requires_microscope=True,
    examples=[
        ToolExample("Go to embryo 1", {"embryo_id": "embryo_1"}),
        ToolExample("Move to embryo 3", {"embryo_id": "embryo_3"}),
    ],
)
async def move_to_embryo(embryo_id: str, context: dict) -> str:
    """Move stage to embryo position"""
    agent = ctx_get(context, "agent")
    client = ctx_get(context, "client")

    if not agent:
        return "Error: No agent context"

    embryo, err = get_embryo_or_error(agent, embryo_id)
    if err:
        return err

    if not embryo.stage_position:
        return f"Embryo '{embryo_id}' has no stored position. Run calibration first."

    try:
        x = embryo.stage_position.get("x", 0)
        y = embryo.stage_position.get("y", 0)
        await client.move_to_position(x, y)

        return f"Moved to {embryo_id}\nPosition: ({x:.2f}, {y:.2f}) um"

    except Exception as e:
        import traceback

        return f"Error moving to embryo: {str(e)}\n{traceback.format_exc()}"


@tool(
    name="get_stage_position",
    description="""Get the current XY stage position in micrometers. Returns the real-time
position from the hardware.
Use when user asks "where is the stage?", "current position?", or when you need to know
the microscope's current location. This reads from hardware - different from embryo stored
positions which are in the experiment data.""",
    category=ToolCategory.HARDWARE,
    requires_microscope=True,
    examples=[
        ToolExample("Where is the stage?", {}),
        ToolExample("Current XY position?", {}),
    ],
)
async def get_stage_position(context: dict) -> str:
    """Get current stage position"""
    client = ctx_get(context, "client")

    if not client:
        return "Error: No microscope client connected"

    try:
        pos = await client.get_stage_position()
        return f"Current stage position: X={pos[0]:.1f} \u00b5m, Y={pos[1]:.1f} \u00b5m"

    except Exception as e:
        return f"Error reading stage position: {str(e)}"


@tool(
    name="move_stage",
    description="""Move the XY stage to specific coordinates in micrometers.
Use when user wants to move to arbitrary coordinates (e.g., "move to x=1000, y=500",
"move stage to 1200, -600").
For moving to a specific embryo, use move_to_embryo instead.""",
    category=ToolCategory.HARDWARE,
    requires_microscope=True,
    examples=[
        ToolExample("Move to x=1000, y=500", {"x": 1000, "y": 500}),
        ToolExample("Move stage to coordinates 1200, -600", {"x": 1200, "y": -600}),
    ],
)
async def move_stage(x: float, y: float, context: dict | None = None) -> str:
    """Move stage to arbitrary XY coordinates"""
    client = ctx_get(context, "client")

    if not client:
        return "Error: No microscope client connected"

    try:
        await client.move_to_position(x=x, y=y)
        pos = await client.get_stage_position()
        return f"Moved to X={pos[0]:.1f} \u00b5m, Y={pos[1]:.1f} \u00b5m"

    except Exception as e:
        return f"Error moving stage: {str(e)}"
