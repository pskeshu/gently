"""
Temperature Protocol Tools

Agent tool for the scripted temp-change burst protocol. The thermal ramp takes
several minutes so the driver runs as a background asyncio task — the agent turn
returns immediately with a "started" confirmation.
"""

import asyncio

from gently.harness.tools.helpers import (
    require_agent,
    require_microscope,
    require_timelapse_orchestrator,
)
from gently.harness.tools.registry import ToolCategory, ToolExample, tool


@tool(
    name="run_temp_change_burst_protocol",
    description=(
        "Start a scripted temperature-change burst protocol in the background. "
        "Acquires brightfield burst(s) before the setpoint change, then polls the "
        "thermal controller — imaging continuously during the ramp — and acquires "
        "burst(s) after lock is reached. "
        "The thermal ramp typically takes several minutes; this tool returns "
        "immediately and the protocol runs as a background task. "
        "Use when you want time-resolved brightfield data spanning a temperature "
        "transition (e.g. cold-shock → recovery, 15 C → 25 C developmental-rate shift)."
    ),
    category=ToolCategory.EXPERIMENT,
    requires_microscope=True,
    examples=[
        ToolExample(
            "Capture a temperature shift from 15 C to 25 C for embryo_1",
            {"embryo_id": "embryo_1", "target_setpoint_c": 25.0},
        ),
        ToolExample(
            "Cold-shock embryo_2 to 15 C with 2 bursts before and after",
            {
                "embryo_id": "embryo_2",
                "target_setpoint_c": 15.0,
                "bursts_before": 2,
                "bursts_after": 2,
            },
        ),
    ],
)
async def run_temp_change_burst_protocol_tool(
    embryo_id: str,
    target_setpoint_c: float,
    frames: int = 60,
    bursts_before: int = 1,
    bursts_after: int = 1,
    tactic_id: str | None = None,
    context: dict | None = None,
) -> str:
    """Launch the temp-change burst driver as a background asyncio task.

    Parameters
    ----------
    embryo_id : str
        ID of the embryo to image during the temperature transition.
    target_setpoint_c : float
        Target temperature setpoint in degrees Celsius.
    frames : int
        Number of frames per burst acquisition (default 60).
    bursts_before : int
        Number of brightfield bursts to capture before the setpoint change (default 1).
    bursts_after : int
        Number of brightfield bursts to capture after the controller locks (default 1).
    """
    agent, err = require_agent(context)
    if err:
        return err

    orchestrator, err = require_timelapse_orchestrator(agent)
    if err:
        return err

    client, err = require_microscope(context)
    if err:
        return err

    from gently.app.orchestration.temperature_protocol import (
        run_temp_change_burst_protocol as _driver,
    )
    from gently.app.orchestration.timelapse_models import TimelapseStatus

    if getattr(orchestrator, "_status", None) == TimelapseStatus.RUNNING:
        return (
            "Refusing to start temp-change protocol: a timelapse is currently running "
            "(would contend for the RunEngine). Stop the timelapse first."
        )

    # Flip the matching plan tactic to active before launching the background driver
    # (guarded no-op when tactic_id or context store are absent).
    if tactic_id:
        cs = getattr(agent, "context_store", None)
        session = getattr(agent, "session_id", None)
        if cs and session:
            cs.transition_tactic(session, tactic_id, "active")

    asyncio.create_task(
        _driver(
            orchestrator,
            embryo_id,
            target_setpoint_c,
            frames=frames,
            bursts_before=bursts_before,
            bursts_after=bursts_after,
            tactic_id=tactic_id,
        )
    )

    return (
        f"Temp-change burst protocol started for {embryo_id} → {target_setpoint_c} C "
        f"({bursts_before} burst(s) before, {bursts_after} burst(s) after lock). "
        f"Running in background — use get_temperature to monitor ramp progress."
    )
