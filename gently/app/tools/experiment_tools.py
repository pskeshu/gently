"""
Experiment and Embryo Management Tools

Tools for managing experiments and tracking embryo states.
"""

import json
from datetime import datetime

from gently.harness.tools.helpers import get_embryo_or_error, require_agent
from gently.harness.tools.registry import ToolCategory, ToolExample, tool


@tool(
    name="get_current_time",
    description="""Get the current date and time. Use this when you need to know what time it is
now, for example when the user says "image until 4pm" or "run for the next 2 hours" and you need
to calculate durations.""",
    category=ToolCategory.UTILITY,
    examples=[
        ToolExample("What time is it?", {}),
        ToolExample("Image from now to 4pm", {}),
    ],
)
def get_current_time(context: dict) -> str:
    """Get current date and time"""
    now = datetime.now()
    return f"Current time: {now.strftime('%Y-%m-%d %H:%M:%S')} ({now.strftime('%I:%M %p')})"


@tool(
    name="get_experiment_summary",
    description="""Get a comprehensive summary of the current experiment including all embryos,
their XY stage positions, calibration status, and imaging history.
Use this tool when the user asks about embryo locations, experiment status, how many embryos
exist, or wants an overview.
This is the primary tool for answering questions like "where are the embryos?" or
"what's the current status?" Returns all embryo IDs with their coordinates - no parameters
needed.""",
    category=ToolCategory.EXPERIMENT,
    examples=[
        ToolExample("Where are all the embryos?", {}),
        ToolExample("What's the experiment status?", {}),
        ToolExample("How many embryos do we have?", {}),
    ],
)
def get_experiment_summary(context: dict) -> str:
    """Get full experiment summary"""
    agent, err = require_agent(context)
    if err:
        return err
    return agent.experiment.get_summary()


@tool(
    name="query_embryo_status",
    description="""Query detailed status of a specific embryo including position, calibration data,
imaging history, and detection results.
Use this when the user asks about a specific embryo by ID or number (e.g., "how is embryo 3?",
"check embryo_1"). Returns JSON with stage_position, piezo_center, galvo_center,
timepoints_acquired, and detection_results. The embryo_id can be like "embryo_1",
"embryo_3", etc.""",
    category=ToolCategory.EMBRYO,
    examples=[
        ToolExample("What's happening with embryo 1?", {"embryo_id": "embryo_1"}),
        ToolExample("Check on embryo 3", {"embryo_id": "embryo_3"}),
    ],
)
def query_embryo_status(embryo_id: str, context: dict) -> str:
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
    description="""Mark an embryo to be skipped in future timelapse acquisitions. The embryo
remains in the experiment but won't be imaged.
Use when user wants to temporarily stop imaging an embryo (e.g., "skip embryo 2",
"stop imaging embryo_3"). Requires a reason to document why the embryo is being skipped.
Can be resumed later with resume_embryo.""",
    category=ToolCategory.EMBRYO,
    examples=[
        ToolExample(
            "Skip embryo 2, it's dead",
            {"embryo_id": "embryo_2", "reason": "embryo dead"},
        ),
    ],
)
def skip_embryo(embryo_id: str, reason: str, context: dict) -> str:
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
    description="""Permanently remove an embryo from the experiment. This is irreversible
- use for false detections or debris.
Use when user says "remove embryo X", "delete embryo X", or "that's not an embryo".
Unlike skip_embryo, this completely removes the embryo from tracking. Use carefully.""",
    category=ToolCategory.EMBRYO,
    examples=[
        ToolExample("Remove embryo 4, it's a false positive", {"embryo_id": "embryo_4"}),
    ],
)
def remove_embryo(embryo_id: str, context: dict) -> str:
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
    description="""Resume imaging a previously skipped embryo. Clears the skip flag so the embryo
will be included in future acquisitions.
Use when user wants to start imaging an embryo again after it was skipped (e.g.,
"resume embryo 2", "start imaging embryo_3 again").""",
    category=ToolCategory.EMBRYO,
    examples=[
        ToolExample("Resume imaging embryo 2", {"embryo_id": "embryo_2"}),
    ],
)
def resume_embryo(embryo_id: str, context: dict) -> str:
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
Use when you notice distinguishing characteristics or the user wants to name an embryo
(e.g., "call embryo 1 speedy", "nickname embryo_2 as the fast one").
Nicknames make conversation more natural - you can then refer to embryos by nickname.""",
    category=ToolCategory.EMBRYO,
    examples=[
        ToolExample("Call embryo 1 speedy", {"embryo_id": "embryo_1", "nickname": "speedy"}),
    ],
)
def assign_nickname(embryo_id: str, nickname: str, context: dict) -> str:
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
    description="""Modify acquisition parameters for a specific embryo. Supported keys:
interval_seconds, num_slices, exposure_ms, priority, acquisition_mode, laser_power_488_pct.
Use when user wants to adjust imaging for one embryo (e.g., "image embryo 2 faster",
"use snap mode for embryo_1", "drop 488 to 3% for embryo_3").
acquisition_mode can be "volume" (full 3D stack, default) or "snap" (single 2D lightsheet
image - faster, less light exposure).
laser_power_488_pct is hard-limited at the device layer (currently 2-6% — see
DiSPIMLightSource.POWER_LIMITS_PCT). Out-of-range values are rejected at the tool boundary
AND at the device. Requires a reason to document why parameters are being changed.
Changes take effect at the next acquisition.""",
    category=ToolCategory.EMBRYO,
    examples=[
        ToolExample(
            "Image embryo 2 every 30 seconds",
            {
                "embryo_id": "embryo_2",
                "changes": {"interval_seconds": 30},
                "reason": "pre-hatching monitoring",
            },
        ),
        ToolExample(
            "Use snap mode for embryo 1",
            {
                "embryo_id": "embryo_1",
                "changes": {"acquisition_mode": "snap"},
                "reason": "reduce light exposure",
            },
        ),
        ToolExample(
            "Drop 488 power on embryo 3 to 3%",
            {
                "embryo_id": "embryo_3",
                "changes": {"laser_power_488_pct": 3.0},
                "reason": "signal saturating",
            },
        ),
    ],
)
def modify_parameters(embryo_id: str, changes: dict, reason: str, context: dict) -> str:
    """Modify embryo acquisition parameters"""
    agent, err = require_agent(context)
    if err:
        return err

    embryo, err = get_embryo_or_error(agent, embryo_id)
    if err:
        return err

    old_params = {
        "interval_seconds": embryo.interval_seconds,
        "num_slices": embryo.num_slices,
        "exposure_ms": embryo.exposure_ms,
        "priority": embryo.priority,
        "acquisition_mode": embryo.acquisition_mode,
        "laser_power_488_pct": embryo.laser_power_488_pct,
    }

    if "interval_seconds" in changes:
        embryo.interval_seconds = changes["interval_seconds"]
    if "num_slices" in changes:
        embryo.num_slices = changes["num_slices"]
    if "exposure_ms" in changes:
        embryo.exposure_ms = changes["exposure_ms"]
    if "priority" in changes:
        embryo.priority = changes["priority"]
    if "acquisition_mode" in changes:
        mode = changes["acquisition_mode"]
        if mode in ("volume", "snap"):
            embryo.acquisition_mode = mode
        else:
            return f"Invalid acquisition_mode '{mode}'. Use 'volume' or 'snap'."
    if "laser_power_488_pct" in changes:
        # Soft-validate at the tool layer so the agent gets a clean error
        # without round-tripping to the device. Hard limit is enforced
        # at DiSPIMLightSource.set_power_pct regardless.
        from gently.hardware.dispim.devices.optical import DiSPIMLightSource

        pct = changes["laser_power_488_pct"]
        lo, hi = DiSPIMLightSource.POWER_LIMITS_PCT.get(488, (0.0, 100.0))
        if pct is not None and not (lo <= pct <= hi):
            return (
                f"laser_power_488_pct={pct} outside hard safety limit "
                f"[{lo}, {hi}]%. (Limit is baked into the device layer; "
                f"change DiSPIMLightSource.POWER_LIMITS_PCT to retune.)"
            )
        embryo.laser_power_488_pct = pct

    return (
        f"Modified {embryo_id} parameters:\n"
        f"Reason: {reason}\n\n"
        f"Changes:\n{json.dumps(changes, indent=2)}\n\n"
        f"Previous: {json.dumps(old_params, indent=2)}"
    )


@tool(
    name="assign_embryo_roles",
    description="""Assign experimental roles (test / calibration / unassigned) to one or more
embryos. Use when the user has marked embryos in the map view and is classifying which are
biological subjects (test) vs reference/calibration samples (calibration).
Roles drive cadence policy, detector selection, photodose budget, and UI color. Pass a dict
mapping embryo_id -> role name.
Available roles come from gently.harness.roles.REGISTRY: 'test', 'calibration', 'unassigned'.""",
    category=ToolCategory.EMBRYO,
    examples=[
        ToolExample("Mark embryo 1 as calibration", {"roles": {"embryo_1": "calibration"}}),
        ToolExample(
            "Embryos 1-2 are calibration, 3-5 are test",
            {
                "roles": {
                    "embryo_1": "calibration",
                    "embryo_2": "calibration",
                    "embryo_3": "test",
                    "embryo_4": "test",
                    "embryo_5": "test",
                }
            },
        ),
    ],
)
def assign_embryo_roles(roles: dict[str, str], context: dict) -> str:
    """Assign roles to embryos. Validates against gently.harness.roles.REGISTRY."""
    from gently.core import EventType, get_event_bus
    from gently.harness.roles import get_role, list_roles

    agent, err = require_agent(context)
    if err:
        return err

    # Validate all roles before mutating anything (atomic semantics)
    unknown_embryos = [eid for eid in roles if eid not in agent.experiment.embryos]
    if unknown_embryos:
        return (
            f"Unknown embryo(s): {unknown_embryos}. "
            f"Available: {list(agent.experiment.embryos.keys())}"
        )

    invalid_roles = []
    for eid, role_name in roles.items():
        try:
            get_role(role_name)
        except KeyError:
            invalid_roles.append((eid, role_name))
    if invalid_roles:
        return f"Invalid role(s): {invalid_roles}. Available roles: {list_roles()}"

    # Apply
    event_bus = get_event_bus()
    changes = []
    for eid, role_name in roles.items():
        embryo = agent.experiment.embryos[eid]
        old_role = embryo.role
        if old_role == role_name:
            continue
        embryo.role = role_name
        changes.append(f"{eid}: {old_role} -> {role_name}")

        # Persist to embryo.yaml if FileStore is wired up.
        if getattr(agent, "store", None) and getattr(agent, "session_id", None):
            pos = embryo.stage_position or {}
            agent.store.register_embryo(
                agent.session_id,
                eid,
                position_x=pos.get("x"),
                position_y=pos.get("y"),
                calibration=embryo.calibration,
                role=role_name,
            )

        event_bus.publish(
            event_type=EventType.STATUS_CHANGED,
            data={
                "embryo_id": eid,
                "change": "role_assigned",
                "old_role": old_role,
                "new_role": role_name,
            },
            source="assign_embryo_roles",
        )

    if not changes:
        return "No role changes — all embryos already at the requested roles."
    return "Assigned roles:\n" + "\n".join(f"  • {c}" for c in changes)
