"""
LED Control Tools

Tools for controlling microscope LED illumination.
"""

from gently.harness.tools.helpers import ctx_get
from gently.harness.tools.registry import ToolCategory, tool


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

            return (
                f"LED Status:\n"
                f"  Current state: {current}\n"
                f"  ConfigGroup: {group}\n"
                f"  Available configs: {available}"
            )
        else:
            return f"Error getting LED status: {result.get('error', 'Unknown error')}"
    except Exception as e:
        return f"Error getting LED status: {str(e)}"
