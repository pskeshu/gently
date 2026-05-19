"""
Experiment and Embryo Management Tools

Tools for managing experiments and tracking embryo states.
"""

from typing import Dict, List, Optional
from datetime import datetime
import json

from gently.harness.tools.registry import tool, ToolCategory, ToolExample
from gently.harness.tools.helpers import require_agent, get_embryo_or_error


@tool(
    name="get_current_time",
    description="""Get the current date and time. Use this when you need to know what time it is now,
for example when the user says "image until 4pm" or "run for the next 2 hours" and you need to calculate durations.""",
    category=ToolCategory.UTILITY,
    examples=[
        ToolExample("What time is it?", {}),
        ToolExample("Image from now to 4pm", {}),
    ],
)
def get_current_time(context: Dict) -> str:
    """Get current date and time"""
    now = datetime.now()
    return f"Current time: {now.strftime('%Y-%m-%d %H:%M:%S')} ({now.strftime('%I:%M %p')})"


@tool(
    name="get_experiment_summary",
    description="""Get a comprehensive summary of the current experiment including all embryos, their XY stage positions, calibration status, and imaging history.
Use this tool when the user asks about embryo locations, experiment status, how many embryos exist, or wants an overview.
This is the primary tool for answering questions like "where are the embryos?" or "what's the current status?"
Returns all embryo IDs with their coordinates - no parameters needed.""",
    category=ToolCategory.EXPERIMENT,
    examples=[
        ToolExample("Where are all the embryos?", {}),
        ToolExample("What's the experiment status?", {}),
        ToolExample("How many embryos do we have?", {}),
    ],
)
def get_experiment_summary(context: Dict) -> str:
    """Get full experiment summary"""
    agent, err = require_agent(context)
    if err:
        return err
    return agent.experiment.get_summary()


@tool(
    name="get_calibration_certificate",
    description="""Return the structured calibration certificate for one embryo.

The certificate is the orchestrator's coarse, sanctioned view of calibration
quality. It contains the verified state (True / False / "pending"), the R²
values at the top and bottom calibration positions, the slope, any concerns
flagged by the rule engine, and the most recent VLM check result (or None if
verification has not yet completed).

USE THIS instead of dredging R² values out of query_embryo_status — the
certificate is the single field that downstream gates (start_timelapse,
acquire_volume) consult.""",
    category=ToolCategory.CALIBRATION,
    examples=[
        ToolExample("Is the calibration for embryo 1 healthy?", {"embryo_id": "embryo_1"}),
        ToolExample("Show the certificate for the bad one", {"embryo_id": "embryo_4"}),
    ],
)
def get_calibration_certificate(embryo_id: str, context: Dict) -> str:
    agent, err = require_agent(context)
    if err:
        return err

    embryo, err = get_embryo_or_error(agent, embryo_id)
    if err:
        return err

    cert = None
    if isinstance(embryo.calibration, dict):
        cert = embryo.calibration.get('certificate')
    if not cert:
        return (
            f"{embryo.id} has no calibration certificate. "
            "Either it has not been calibrated, or its calibration predates "
            "the certificate scheme. Run calibrate_embryo to refresh."
        )
    return json.dumps(cert, indent=2, default=str)


@tool(
    name="is_ready_to_image",
    description="""Single gate the agent must consult before any acquisition.

Returns READY / NOT READY plus a one-line reason per embryo. Calling this is
mandatory before start_adaptive_timelapse or acquire_volume — those tools also
gate on the certificate, but checking here lets you remediate proactively
(recalibrate, await pending VLM verification) instead of being refused.

If embryo_ids is omitted, checks every embryo in the experiment.""",
    category=ToolCategory.CALIBRATION,
    examples=[
        ToolExample("Are we good to start the timelapse?", {}),
        ToolExample("Is embryo 1 ready?", {"embryo_ids": ["embryo_1"]}),
    ],
)
def is_ready_to_image(
    embryo_ids: Optional[List[str]] = None,
    context: Dict = None,
) -> str:
    agent, err = require_agent(context)
    if err:
        return err

    focus = getattr(agent, 'focus', None)
    if focus is None:
        return "Focus controller unavailable — cannot check readiness"

    if not embryo_ids:
        embryo_ids = list(agent.experiment.embryos.keys())
    if not embryo_ids:
        return "No embryos in experiment yet"

    lines: List[str] = []
    any_blocked = False
    any_pending = False
    for eid in embryo_ids:
        ready, reason = focus.is_ready_to_image(eid)
        if ready:
            lines.append(f"READY     {eid}: {reason}")
        else:
            any_blocked = True
            if 'pending' in reason.lower():
                any_pending = True
            lines.append(f"NOT READY {eid}: {reason}")

    if not any_blocked:
        header = "All embryos are ready to image."
    elif any_pending:
        header = (
            "At least one embryo is still being verified. Call "
            "await_calibration_verification to block until pending VLM "
            "checks complete, or proceed with the embryos that are READY."
        )
    else:
        header = (
            "One or more embryos have unresolved calibration concerns. "
            "Inspect the certificate via get_calibration_certificate and "
            "either recalibrate or skip the affected embryos before imaging."
        )
    return header + "\n" + "\n".join(lines)


@tool(
    name="await_calibration_verification",
    description="""Block until pending VLM calibration verifications complete.

Use this when is_ready_to_image reports "verification still in progress" and
you need a definitive answer before proceeding. Returns a per-embryo summary
of the final certificate state. Times out after the specified number of
seconds (default 60) — if the timeout expires the certificates remain in
'pending' and you can call this again.""",
    category=ToolCategory.CALIBRATION,
    examples=[
        ToolExample("Wait until calibration is verified", {}),
        ToolExample("Wait up to 2 minutes for embryo_1", {"embryo_id": "embryo_1", "timeout_seconds": 120}),
    ],
)
async def await_calibration_verification(
    embryo_id: Optional[str] = None,
    timeout_seconds: float = 60.0,
    context: Dict = None,
) -> str:
    agent, err = require_agent(context)
    if err:
        return err

    focus = getattr(agent, 'focus', None)
    if focus is None:
        return "Focus controller unavailable"

    if not focus.has_pending(embryo_id):
        return "No pending verifications."

    results = await focus.await_pending(embryo_id=embryo_id, timeout=timeout_seconds)
    if not results:
        return "Timed out waiting; verifications are still pending."

    lines: List[str] = []
    for eid, cert in results.items():
        if cert is None:
            lines.append(f"{eid}: no certificate (lost?)")
            continue
        verified = cert.get('verified')
        if verified is True:
            lines.append(f"{eid}: VERIFIED")
        elif verified == 'pending':
            lines.append(f"{eid}: still pending (timeout reached)")
        else:
            concerns = cert.get('concerns') or []
            lines.append(f"{eid}: FAILED — " + "; ".join(concerns))
    return "\n".join(lines)


@tool(
    name="query_embryo_status",
    description="""Query detailed status of a specific embryo including position, calibration data, imaging history, and detection results.
Use this when the user asks about a specific embryo by ID or number (e.g., "how is embryo 3?", "check embryo_1").
Returns JSON with stage_position, piezo_center, galvo_center, timepoints_acquired, and detection_results.
The embryo_id can be like "embryo_1", "embryo_3", etc.""",
    category=ToolCategory.EMBRYO,
    examples=[
        ToolExample("What's happening with embryo 1?", {"embryo_id": "embryo_1"}),
        ToolExample("Check on embryo 3", {"embryo_id": "embryo_3"}),
    ],
)
def query_embryo_status(embryo_id: str, context: Dict) -> str:
    """Query embryo status"""
    agent, err = require_agent(context)
    if err:
        return err

    embryo, err = get_embryo_or_error(agent, embryo_id)
    if err:
        return f"{err}. Available: {list(agent.experiment.embryos.keys())}"

    return json.dumps(embryo.to_dict(), indent=2)


@tool(
    name="skip_embryo",
    description="""Mark an embryo to be skipped in future timelapse acquisitions. The embryo remains in the experiment but won't be imaged.
Use when user wants to temporarily stop imaging an embryo (e.g., "skip embryo 2", "stop imaging embryo_3").
Requires a reason to document why the embryo is being skipped. Can be resumed later with resume_embryo.""",
    category=ToolCategory.EMBRYO,
    examples=[
        ToolExample("Skip embryo 2, it's dead", {"embryo_id": "embryo_2", "reason": "embryo dead"}),
    ],
)
def skip_embryo(embryo_id: str, reason: str, context: Dict) -> str:
    """Skip embryo in future acquisitions"""
    agent, err = require_agent(context)
    if err:
        return err

    embryo, err = get_embryo_or_error(agent, embryo_id)
    if err:
        return err

    embryo.should_skip = True
    embryo.skip_reason = reason

    return f"Marked {embryo_id} to skip. Reason: {reason}"


@tool(
    name="remove_embryo",
    description="""Permanently remove an embryo from the experiment. This is irreversible - use for false detections or debris.
Use when user says "remove embryo X", "delete embryo X", or "that's not an embryo".
Unlike skip_embryo, this completely removes the embryo from tracking. Use carefully.""",
    category=ToolCategory.EMBRYO,
    examples=[
        ToolExample("Remove embryo 4, it's a false positive", {"embryo_id": "embryo_4"}),
    ],
)
def remove_embryo(embryo_id: str, context: Dict) -> str:
    """Remove embryo from experiment completely"""
    agent, err = require_agent(context)
    if err:
        return err

    embryo, err = get_embryo_or_error(agent, embryo_id)
    if err:
        return err

    actual_id = embryo.id
    if agent.experiment.remove_embryo(actual_id):
        return f"Removed {actual_id} from experiment"
    else:
        return f"Failed to remove {embryo_id}"


@tool(
    name="resume_embryo",
    description="""Resume imaging a previously skipped embryo. Clears the skip flag so the embryo will be included in future acquisitions.
Use when user wants to start imaging an embryo again after it was skipped (e.g., "resume embryo 2", "start imaging embryo_3 again").""",
    category=ToolCategory.EMBRYO,
    examples=[
        ToolExample("Resume imaging embryo 2", {"embryo_id": "embryo_2"}),
    ],
)
def resume_embryo(embryo_id: str, context: Dict) -> str:
    """Resume skipped embryo"""
    agent, err = require_agent(context)
    if err:
        return err

    embryo, err = get_embryo_or_error(agent, embryo_id)
    if err:
        return err

    embryo.should_skip = False
    embryo.skip_reason = None

    return f"Resumed imaging {embryo_id}"


@tool(
    name="assign_nickname",
    description="""Assign a memorable nickname to an embryo for easier reference in conversation.
Use when you notice distinguishing characteristics or the user wants to name an embryo (e.g., "call embryo 1 speedy", "nickname embryo_2 as the fast one").
Nicknames make conversation more natural - you can then refer to embryos by nickname.""",
    category=ToolCategory.EMBRYO,
    examples=[
        ToolExample("Call embryo 1 speedy", {"embryo_id": "embryo_1", "nickname": "speedy"}),
    ],
)
def assign_nickname(embryo_id: str, nickname: str, context: Dict) -> str:
    """Assign nickname to embryo"""
    agent, err = require_agent(context)
    if err:
        return err

    embryo, err = get_embryo_or_error(agent, embryo_id)
    if err:
        return err

    old_nickname = embryo.nickname
    embryo.nickname = nickname

    if old_nickname:
        return f"Renamed {embryo_id}: '{old_nickname}' -> '{nickname}'"
    else:
        return f"Nicknamed {embryo_id} as '{nickname}'"


@tool(
    name="modify_parameters",
    description="""Modify acquisition parameters for a specific embryo. Can change interval_seconds, num_slices, exposure_ms, priority, or acquisition_mode.
Use when user wants to adjust imaging for one embryo (e.g., "image embryo 2 faster", "use snap mode for embryo_1").
acquisition_mode can be "volume" (full 3D stack, default) or "snap" (single 2D lightsheet image - faster, less light exposure).
Requires a reason to document why parameters are being changed. Changes take effect at the next acquisition.""",
    category=ToolCategory.EMBRYO,
    examples=[
        ToolExample("Image embryo 2 every 30 seconds", {"embryo_id": "embryo_2", "changes": {"interval_seconds": 30}, "reason": "pre-hatching monitoring"}),
        ToolExample("Use snap mode for embryo 1", {"embryo_id": "embryo_1", "changes": {"acquisition_mode": "snap"}, "reason": "reduce light exposure"}),
    ],
)
def modify_parameters(
    embryo_id: str,
    changes: Dict,
    reason: str,
    context: Dict
) -> str:
    """Modify embryo acquisition parameters"""
    agent, err = require_agent(context)
    if err:
        return err

    embryo, err = get_embryo_or_error(agent, embryo_id)
    if err:
        return err

    old_params = {
        'interval_seconds': embryo.interval_seconds,
        'num_slices': embryo.num_slices,
        'exposure_ms': embryo.exposure_ms,
        'priority': embryo.priority,
        'acquisition_mode': embryo.acquisition_mode,
    }

    if 'interval_seconds' in changes:
        embryo.interval_seconds = changes['interval_seconds']
    if 'num_slices' in changes:
        embryo.num_slices = changes['num_slices']
    if 'exposure_ms' in changes:
        embryo.exposure_ms = changes['exposure_ms']
    if 'priority' in changes:
        embryo.priority = changes['priority']
    if 'acquisition_mode' in changes:
        mode = changes['acquisition_mode']
        if mode in ('volume', 'snap'):
            embryo.acquisition_mode = mode
        else:
            return f"Invalid acquisition_mode '{mode}'. Use 'volume' or 'snap'."

    return (f"Modified {embryo_id} parameters:\n"
            f"Reason: {reason}\n\n"
            f"Changes:\n{json.dumps(changes, indent=2)}\n\n"
            f"Previous: {json.dumps(old_params, indent=2)}")
