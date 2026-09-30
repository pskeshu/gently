"""
LED Control Tools

Tools for controlling microscope LED illumination.
"""

from gently.harness.tools.helpers import ctx_get
from gently.harness.tools.registry import ToolCategory, ToolExample, tool


@tool(
    name="set_led",
    description="Set the LED illumination state",
    category=ToolCategory.HARDWARE,
    requires_microscope=True,
)
async def set_led(state: str, context: dict) -> str:
    """Set LED state"""
    client = ctx_get(context, "client")

    try:
        result = await client.set_led(state)
        if result.get("success"):
            return f"LED set to '{state}'"
        else:
            return f"Error setting LED: {result.get('error', 'Unknown error')}"
    except Exception as e:
        return f"Error setting LED: {str(e)}"


@tool(
    name="set_led_intensity",
    description="""Set the brightness of the transmitted-light LED, in whole percent (1-100).

Does not open or close the LED: set while Closed, the value is held and used the next
time the LED opens. To turn the LED off use set_led with 'Closed', not a low intensity.""",
    category=ToolCategory.HARDWARE,
    requires_microscope=True,
    examples=[
        ToolExample("Dim the LED to 20 percent", {"pct": 20}),
        ToolExample("LED at full brightness", {"pct": 100}),
    ],
)
async def set_led_intensity(pct: int, context: dict) -> str:
    """Set LED brightness and report what the device holds afterwards."""
    client = ctx_get(context, "client")

    try:
        result = await client.set_led_intensity(pct)
        if result.get("success"):
            readback = result.get("readback_pct")
            if readback is not None:
                return f"LED intensity set to {pct}% (readback: {readback}%)"
            return f"LED intensity set to {pct}% (readback unavailable)"
        else:
            return f"Error setting LED intensity: {result.get('error', 'Unknown error')}"
    except Exception as e:
        return f"Error setting LED intensity: {str(e)}"


@tool(
    name="get_led_status",
    description="Get the current LED illumination status",
    category=ToolCategory.HARDWARE,
    requires_microscope=True,
)
async def get_led_status(context: dict) -> str:
    """Get LED status"""
    client = ctx_get(context, "client")

    try:
        result = await client.get_led_status()
        if result.get("success"):
            current = result.get("current_state", "unknown")
            available = result.get("available_configs", [])
            group = result.get("group_name", "unknown")
            intensity = result.get("intensity_pct")
            brightness = "unknown" if intensity is None else f"{intensity}%"

            return (
                f"LED Status:\n"
                f"  Current state: {current}\n"
                f"  Intensity: {brightness}\n"
                f"  ConfigGroup: {group}\n"
                f"  Available configs: {available}"
            )
        else:
            return f"Error getting LED status: {result.get('error', 'Unknown error')}"
    except Exception as e:
        return f"Error getting LED status: {str(e)}"
