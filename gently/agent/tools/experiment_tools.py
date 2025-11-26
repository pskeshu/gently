"""
Experiment and Embryo Management Tools

Tools for managing experiments and tracking embryo states.
"""

from typing import Dict
import json

from ..tool_registry import tool, ToolCategory
from ..tool_helpers import require_copilot, get_embryo_or_error


@tool(
    name="get_experiment_summary",
    description="Get a summary of the current experiment status including all embryos",
    category=ToolCategory.EXPERIMENT,
)
def get_experiment_summary(context: Dict) -> str:
    """Get full experiment summary"""
    copilot, err = require_copilot(context)
    if err:
        return err
    return copilot.experiment.get_summary()


@tool(
    name="query_embryo_status",
    description="Query the status of a specific embryo by ID or name",
    category=ToolCategory.EMBRYO,
)
def query_embryo_status(embryo_id: str, context: Dict) -> str:
    """Query embryo status"""
    copilot, err = require_copilot(context)
    if err:
        return err

    embryo, err = get_embryo_or_error(copilot, embryo_id)
    if err:
        return f"{err}. Available: {list(copilot.experiment.embryos.keys())}"

    return json.dumps(embryo.to_dict(), indent=2)


@tool(
    name="skip_embryo",
    description="Mark an embryo to be skipped in future acquisitions",
    category=ToolCategory.EMBRYO,
)
def skip_embryo(embryo_id: str, reason: str, context: Dict) -> str:
    """Skip embryo in future acquisitions"""
    copilot, err = require_copilot(context)
    if err:
        return err

    embryo, err = get_embryo_or_error(copilot, embryo_id)
    if err:
        return err

    embryo.should_skip = True
    embryo.skip_reason = reason

    return f"Marked {embryo_id} to skip. Reason: {reason}"


@tool(
    name="remove_embryo",
    description="Permanently remove an embryo from the experiment (e.g., false detection)",
    category=ToolCategory.EMBRYO,
)
def remove_embryo(embryo_id: str, context: Dict) -> str:
    """Remove embryo from experiment completely"""
    copilot, err = require_copilot(context)
    if err:
        return err

    embryo, err = get_embryo_or_error(copilot, embryo_id)
    if err:
        return err

    actual_id = embryo.id
    if copilot.experiment.remove_embryo(actual_id):
        return f"Removed {actual_id} from experiment"
    else:
        return f"Failed to remove {embryo_id}"


@tool(
    name="resume_embryo",
    description="Resume imaging a previously skipped embryo",
    category=ToolCategory.EMBRYO,
)
def resume_embryo(embryo_id: str, context: Dict) -> str:
    """Resume skipped embryo"""
    copilot, err = require_copilot(context)
    if err:
        return err

    embryo, err = get_embryo_or_error(copilot, embryo_id)
    if err:
        return err

    embryo.should_skip = False
    embryo.skip_reason = None

    return f"Resumed imaging {embryo_id}"


@tool(
    name="assign_nickname",
    description="Assign a memorable nickname to an embryo",
    category=ToolCategory.EMBRYO,
)
def assign_nickname(embryo_id: str, nickname: str, context: Dict) -> str:
    """Assign nickname to embryo"""
    copilot, err = require_copilot(context)
    if err:
        return err

    embryo, err = get_embryo_or_error(copilot, embryo_id)
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
    description="Modify acquisition parameters for a specific embryo",
    category=ToolCategory.EMBRYO,
)
def modify_parameters(
    embryo_id: str,
    changes: Dict,
    reason: str,
    context: Dict
) -> str:
    """Modify embryo acquisition parameters"""
    copilot, err = require_copilot(context)
    if err:
        return err

    embryo, err = get_embryo_or_error(copilot, embryo_id)
    if err:
        return err

    old_params = {
        'interval_seconds': embryo.interval_seconds,
        'num_slices': embryo.num_slices,
        'exposure_ms': embryo.exposure_ms,
        'priority': embryo.priority
    }

    if 'interval_seconds' in changes:
        embryo.interval_seconds = changes['interval_seconds']
    if 'num_slices' in changes:
        embryo.num_slices = changes['num_slices']
    if 'exposure_ms' in changes:
        embryo.exposure_ms = changes['exposure_ms']
    if 'priority' in changes:
        embryo.priority = changes['priority']

    return (f"Modified {embryo_id} parameters:\n"
            f"Reason: {reason}\n\n"
            f"Changes:\n{json.dumps(changes, indent=2)}\n\n"
            f"Previous: {json.dumps(old_params, indent=2)}")
