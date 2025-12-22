"""
Unified C. elegans developmental stage definitions.

Single source of truth for all stage-related constants and utilities.
"""

from enum import Enum
from typing import List, Dict, Any


class DevelopmentalStage(str, Enum):
    """
    Unified C. elegans developmental stages.

    Stages are ordered chronologically from fertilization to hatching.
    String values match what the VLM returns and what's used in file paths.

    Simplified stages:
    - "bean" removed (too brief, merged into early)
    - "2fold"/"3fold" merged into "pretzel" (hard to distinguish)

    Special states:
    - "arrested" is not part of normal progression (dead/arrested embryo)
    """
    EARLY = "early"           # Gastrulation through early morphogenesis, oval shape
    COMMA = "comma"           # Clear C-shape, head/tail distinguishable
    FOLD_1_5 = "1.5fold"      # Elongation, ~1.5x eggshell length
    PRETZEL = "pretzel"       # Tight coil, 2-3 body segments (formerly 2fold/3fold)
    HATCHING = "hatching"     # Active emergence, shell breach visible
    HATCHED = "hatched"       # Fully emerged L1 larva
    ARRESTED = "arrested"     # Dead or developmentally arrested embryo (special state)

    @classmethod
    def ordered_list(cls) -> List["DevelopmentalStage"]:
        """Return stages in developmental order."""
        return [
            cls.EARLY, cls.COMMA, cls.FOLD_1_5,
            cls.PRETZEL, cls.HATCHING, cls.HATCHED
        ]

    @classmethod
    def ordered_values(cls) -> List[str]:
        """Return stage string values in developmental order."""
        return [s.value for s in cls.ordered_list()]

    @classmethod
    def get_order(cls, stage: str) -> int:
        """Get the ordinal position of a stage (0-indexed)."""
        return cls.ordered_values().index(stage)

    @classmethod
    def is_terminal(cls, stage: str) -> bool:
        """Check if this is the final stage (hatched)."""
        return stage == cls.HATCHED.value

    @classmethod
    def is_valid(cls, stage: str) -> bool:
        """Check if a string is a valid stage name (including special states)."""
        return stage in cls.all_valid_values()

    @classmethod
    def all_valid_values(cls) -> List[str]:
        """Return all valid stage values including special states like 'arrested'."""
        return cls.ordered_values() + ["arrested"]

    @classmethod
    def is_special_state(cls, stage: str) -> bool:
        """Check if this is a special state (not part of normal progression)."""
        return stage == cls.ARRESTED.value

    @classmethod
    def compare(cls, stage_a: str, stage_b: str) -> int:
        """
        Compare two stages.

        Returns:
            -1 if stage_a is earlier than stage_b
            0 if they are the same
            1 if stage_a is later than stage_b
        """
        order_a = cls.get_order(stage_a)
        order_b = cls.get_order(stage_b)
        if order_a < order_b:
            return -1
        elif order_a > order_b:
            return 1
        return 0


# Convenience: list of stage string values (for backwards compatibility)
STAGES = DevelopmentalStage.ordered_values()


# Stage boundary definitions for VLM guidance
# Each stage has:
#   - features: what to look for (positive indicators)
#   - NOT_if: what rules out this stage (negative indicators)
STAGE_CRITERIA: Dict[str, Dict[str, Any]] = {
    "early": {
        "features": [
            "oval/elliptical shape",
            "grainy texture with many visible nuclei",
            "uniform or slightly asymmetric cellular mass",
            "compact, no clear C-curve yet",
            "may have subtle elongation but no pronounced bend",
        ],
        "NOT_if": [
            "clear C-curve or comma shape",
            "pronounced ventral bend",
            "elongation beyond oval",
        ],
        "typical_duration_min": 90,  # includes former bean stage
    },
    "comma": {
        "features": [
            "clear C-curve or comma shape",
            "pronounced ventral bend",
            "head/tail distinctly different",
            "body axis clearly established",
        ],
        "NOT_if": [
            "no clear bend (that's early)",
            "elongation beyond eggshell (that's 1.5fold)",
            "still oval/round shape",
        ],
        "typical_duration_min": 45,
    },
    "1.5fold": {
        "features": [
            "elongated ~1.5x eggshell length",
            "embryo clearly longer than egg width",
            "body starting to fold back on itself",
            "partial fold visible",
        ],
        "NOT_if": [
            "fits within egg diameter (that's comma)",
            "tight coil with 2-3 segments (that's pretzel)",
        ],
        "typical_duration_min": 40,
    },
    "pretzel": {
        "features": [
            "tightly coiled pretzel-like shape",
            "2-3 body segments visible",
            "maximum compaction within shell",
            "may show occasional twitching",
        ],
        "NOT_if": [
            "only partial fold (that's 1.5fold)",
            "any part outside shell (that's hatching)",
            "shell breach visible",
        ],
        "typical_duration_min": 110,  # combined 2fold + 3fold duration
    },
    "hatching": {
        "features": [
            "eggshell breach/tear VISIBLE",
            "part of embryo OUTSIDE shell",
            "part of embryo STILL INSIDE shell",
            "active pushing/wriggling to escape",
        ],
        "NOT_if": [
            "fully inside shell (that's pretzel)",
            "fully outside shell (that's hatched)",
            "no visible shell breach",
        ],
        "typical_duration_min": 15,  # relatively quick process
    },
    "hatched": {
        "features": [
            "larva FULLY OUTSIDE eggshell",
            "empty or nearly-empty eggshell visible",
            "free-moving L1 larva",
            "elongated worm body, no longer coiled in shell",
        ],
        "NOT_if": [
            "any part still inside shell (that's hatching)",
        ],
        "typical_duration_min": None,  # terminal state
    },
    "arrested": {
        "features": [
            "No visible morphological change over extended period",
            "Same appearance across many consecutive timepoints",
            "May show degradation, fragmentation, or unusual granular texture",
            "No twitching or movement (in later stages where this would be expected)",
            "Possibly collapsed, disintegrating, or abnormal appearance",
            "Development has clearly stalled",
        ],
        "NOT_if": [
            "Clear morphological progression between timepoints",
            "Normal healthy appearance for the declared stage",
            "Movement or twitching visible",
            "Recent stage transition occurred",
        ],
        "typical_duration_min": None,  # special state, not part of normal progression
    },
}


def get_stage_description(stage: str) -> str:
    """Get a brief description for a stage."""
    descriptions = {
        "early": "Gastrulation through early morphogenesis, oval shape, no clear C-curve",
        "comma": "Clear C-shaped curve with established body axis",
        "1.5fold": "Elongation beginning, embryo ~1.5x shell length",
        "pretzel": "Tightly coiled, 2-3 body segments visible",
        "hatching": "Active emergence through shell breach",
        "hatched": "Fully emerged L1 larva",
        "arrested": "Development arrested - dead or stalled embryo",
    }
    return descriptions.get(stage, "Unknown stage")


def format_stage_criteria_for_prompt(stage: str) -> str:
    """Format stage criteria for inclusion in VLM prompt."""
    if stage not in STAGE_CRITERIA:
        return ""

    criteria = STAGE_CRITERIA[stage]
    lines = [f"### {stage.upper()}"]

    lines.append("Features to look for:")
    for feature in criteria["features"]:
        lines.append(f"  - {feature}")

    lines.append("NOT this stage if:")
    for exclusion in criteria["NOT_if"]:
        lines.append(f"  - {exclusion}")

    return "\n".join(lines)


def get_all_criteria_for_prompt() -> str:
    """Get all stage criteria formatted for VLM prompt."""
    sections = []
    for stage in STAGES:
        sections.append(format_stage_criteria_for_prompt(stage))
    return "\n\n".join(sections)
