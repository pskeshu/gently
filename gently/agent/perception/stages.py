"""
Re-export shim — canonical definitions live in gently.organisms.celegans.stages.

All existing imports (e.g. ``from gently.agent.perception.stages import STAGES``)
continue to work transparently.
"""

from gently.organisms.celegans.stages import (  # noqa: F401
    DevelopmentalStage,
    STAGES,
    STAGE_CRITERIA,
    TRANSITION_ZONES,
    get_transition_zone,
    get_adjacent_stages,
    get_stage_description,
    format_stage_criteria_for_prompt,
    get_all_criteria_for_prompt,
)
