"""
Light Source Tools

Direct control of laser power per wavelength, independent of any embryo.
Useful for pre-experiment setup ("show me the field at 4%"), calibration
sweeps, or ad-hoc inspection.

The hard 2-6% safety limit on 488 (and per-wavelength limits in
``DiSPIMLightSource.POWER_LIMITS_PCT``) is enforced inside the device
layer — out-of-range values are rejected regardless of caller.

For experiment-scoped power changes that should ride with the embryo
(e.g. sticky-downward ramp during a timelapse), prefer
``modify_parameters(embryo_id, {"laser_power_488_pct": ...}, ...)``.
"""

from gently.harness.tools.helpers import ctx_get, require_agent
from gently.harness.tools.registry import ToolCategory, ToolExample, tool


@tool(
    name="set_laser_power",
    description="""Set per-line laser power % directly (not tied to any embryo). Submits a
Bluesky plan via the queue server so the change is traceable.

Hard-limited at the device layer (DiSPIMLightSource.POWER_LIMITS_PCT). 488 is constrained
to 2-6% by default. Out-of-range values are rejected at the device layer (ValueError); the
tool returns the error.

After setting, the tool reads back the actual setpoint and includes it in the response so
the agent can verify.

Use for: pre-experiment setup, ad-hoc inspection, calibration. For experiment-scoped
per-embryo changes during a timelapse, use modify_parameters with laser_power_488_pct
instead.""",
    category=ToolCategory.HARDWARE,
    requires_microscope=True,
    examples=[
        ToolExample("Set 488 to 4 percent", {"wavelength": 488, "pct": 4.0}),
        ToolExample("Drop 488 to 2.5 percent", {"wavelength": 488, "pct": 2.5}),
    ],
)
async def set_laser_power(
    wavelength: int,
    pct: float,
    context: dict | None = None,
) -> str:
    """Set laser power and read back the actual setpoint."""
    agent, err = require_agent(context)
    if err:
        return err
    client = ctx_get(context, "client")
    if not client:
        return "Error: Microscope not connected."

    try:
        result = await client.set_laser_power(wavelength, pct)
    except Exception as e:
        return f"set_laser_power({wavelength}nm, {pct}%) failed: {e}"

    if not result.get("success"):
        err_msg = result.get("error") or "unknown error"
        return f"set_laser_power({wavelength}nm, {pct}%) failed: {err_msg}"

    # Read back for verification
    try:
        rb = await client.get_laser_power(wavelength)
        actual = rb.get("pct") if rb.get("success") else None
    except Exception:
        actual = None

    if actual is not None:
        return f"{wavelength}nm power set to {pct}% (readback: {actual:.4f}%)"
    return f"{wavelength}nm power set to {pct}% (readback unavailable)"


@tool(
    name="get_laser_power",
    description="""Read the current per-line laser power % from the device. Useful to verify
state before/after a change, or to spot-check the current illumination during a long run.""",
    category=ToolCategory.HARDWARE,
    requires_microscope=True,
    examples=[
        ToolExample("What is the 488 laser power right now?", {"wavelength": 488}),
        ToolExample("Read 561 laser power", {"wavelength": 561}),
    ],
)
async def get_laser_power(
    wavelength: int,
    context: dict | None = None,
) -> str:
    """Read current laser power %."""
    agent, err = require_agent(context)
    if err:
        return err
    client = ctx_get(context, "client")
    if not client:
        return "Error: Microscope not connected."

    try:
        result = await client.get_laser_power(wavelength)
    except Exception as e:
        return f"get_laser_power({wavelength}nm) failed: {e}"

    if not result.get("success"):
        return f"get_laser_power({wavelength}nm) failed: {result.get('error', 'unknown error')}"

    return f"{wavelength}nm power is currently {result.get('pct'):.4f}%"
