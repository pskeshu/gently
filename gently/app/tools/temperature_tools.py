"""
Temperature Control Tools

Agent tools for the ACUITYnano thermal controller. Temperature drives C. elegans
development rate, so these let the agent hold or shift the sample temperature as
part of closed-loop experiments.
"""

from gently.harness.tools.helpers import ctx_get
from gently.harness.tools.registry import ToolCategory, ToolExample, tool


@tool(
    name="set_temperature",
    description=(
        "Set the sample temperature setpoint in Celsius (0.0-99.9). The thermal "
        "controller ramps toward the target and this returns immediately — poll "
        "get_temperature until the state reads '[ SYSTEM LOCKED ]' before imaging. "
        "Temperature controls C. elegans development rate (~15 C slow, 20 C standard, "
        "25 C fast)."
    ),
    category=ToolCategory.HARDWARE,
    requires_microscope=True,
    examples=[
        ToolExample("Hold the sample at 20 degrees", {"target_c": 20.0}),
        ToolExample("Warm the embryos to 25 C to speed development", {"target_c": 25.0}),
    ],
)
async def set_temperature(target_c: float, context: dict) -> str:
    """Command the thermal controller to a target temperature.

    Parameters
    ----------
    target_c : float
        Target temperature in degrees Celsius (0.0-99.9).
    """
    client = ctx_get(context, "client")
    try:
        result = await client.set_temperature(target_c)
        if result.get("success"):
            return (
                f"Commanded {target_c} C. Currently {result.get('temperature_c')} C, "
                f"state {result.get('state')!r}. Ramping — call get_temperature to confirm lock."
            )
        return f"Error setting temperature: {result.get('error', 'unknown error')}"
    except Exception as e:
        return f"Error setting temperature: {e}"


@tool(
    name="get_temperature",
    description=(
        "Read the current sample temperature, target setpoint, and lock state from the "
        "thermal controller. Use to confirm the sample has stabilized at the setpoint "
        "('[ SYSTEM LOCKED ]') before acquiring."
    ),
    category=ToolCategory.HARDWARE,
    requires_microscope=True,
    examples=[
        ToolExample("What's the current temperature?"),
        ToolExample("Has the sample reached temperature yet?"),
    ],
)
async def get_temperature(context: dict) -> str:
    """Read current temperature, setpoint, and lock state."""
    client = ctx_get(context, "client")
    try:
        r = await client.get_temperature()
        if r.get("success"):
            msg = (
                f"Temperature {r.get('temperature_c')} C "
                f"(setpoint {r.get('setpoint_c')} C, state {r.get('state')!r}"
            )
            if r.get("peltier_c") is not None:
                msg += f", peltier {r.get('peltier_c')} C"
            return msg + ")"
        return f"Error reading temperature: {r.get('error', 'unknown error')}"
    except Exception as e:
        return f"Error reading temperature: {e}"
