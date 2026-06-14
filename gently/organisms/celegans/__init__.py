"""
C. elegans organism module.

Single source of truth for all C. elegans-specific knowledge:
stage definitions, biology text, detector presets, timing tables, etc.

Exports well-known names that the rest of the system expects from any
organism module.
"""

from pathlib import Path

from .biology import BIOLOGY_KNOWLEDGE
from .detection_defaults import DETECTION_DEFAULTS
from .detector_presets import get_detector_presets
from .perception_prompt import PERCEPTION_SYSTEM_PROMPT
from .stages import (
    STAGE_CRITERIA,
    STAGES,
    TRANSITION_ZONES,
    DevelopmentalStage,
    format_stage_criteria_for_prompt,
    get_adjacent_stages,
    get_all_criteria_for_prompt,
    get_stage_description,
    get_transition_zone,
)

__all__ = [
    "BIOLOGY_KNOWLEDGE",
    "DETECTION_DEFAULTS",
    "get_detector_presets",
    "PERCEPTION_SYSTEM_PROMPT",
    "STAGE_CRITERIA",
    "STAGES",
    "TRANSITION_ZONES",
    "DevelopmentalStage",
    "format_stage_criteria_for_prompt",
    "get_adjacent_stages",
    "get_all_criteria_for_prompt",
    "get_stage_description",
    "get_transition_zone",
    "ORGANISM_NAME",
    "ORGANISM_DISPLAY_NAME",
    "SAMPLE_TERM",
    "SAMPLE_TERM_PLURAL",
    "TERMINAL_STAGES",
    "STOP_CONDITIONS",
    "PRE_TERMINAL_SPEEDUP_STAGE",
    "EXAMPLES_PATH",
]

# --- Organism identity ---
ORGANISM_NAME = "celegans"
ORGANISM_DISPLAY_NAME = "C. elegans"
SAMPLE_TERM = "embryo"
SAMPLE_TERM_PLURAL = "embryos"

# --- Terminal / stop-condition stages ---
TERMINAL_STAGES = {"hatched"}

STOP_CONDITIONS = {
    "hatching": {"hatching", "hatched"},
    "comma": {"comma", "1.5fold", "2fold", "pretzel", "hatching", "hatched"},
}

PRE_TERMINAL_SPEEDUP_STAGE = "pretzel"

# --- Path to reference images ---
EXAMPLES_PATH = Path(__file__).resolve().parent.parent.parent / "examples"
