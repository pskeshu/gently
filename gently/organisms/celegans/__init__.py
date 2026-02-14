"""
C. elegans organism module.

Single source of truth for all C. elegans-specific knowledge:
stage definitions, biology text, detector presets, timing tables, etc.

Exports well-known names that the rest of the system expects from any
organism module.
"""

from pathlib import Path

from .stages import (
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
from .biology import BIOLOGY_KNOWLEDGE
from .detector_presets import get_detector_presets
from .detection_defaults import DETECTION_DEFAULTS
from .perception_prompt import PERCEPTION_SYSTEM_PROMPT

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
