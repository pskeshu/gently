"""
Unified C. elegans developmental stage definitions.

Single source of truth for all stage-related constants and utilities.
"""

from enum import Enum
from typing import Any


class DevelopmentalStage(str, Enum):
    """
    Unified C. elegans developmental stages.

    Stages are ordered chronologically from fertilization to hatching.
    String values match what the VLM returns and what's used in file paths.

    Full stage progression:
    early -> bean -> comma -> 1.5fold -> 2fold -> pretzel -> hatching -> hatched

    Special states:
    - "arrested" is not part of normal progression (dead/arrested embryo)
    """

    EARLY = "early"  # Gastrulation through early morphogenesis, oval shape
    BEAN = "bean"  # Elongated oval, "bean-shaped", pre-comma curvature
    COMMA = "comma"  # Clear C-shape, head/tail distinguishable
    FOLD_1_5 = "1.5fold"  # Elongation, ~1.5x eggshell length
    FOLD_2 = "2fold"  # Body folded back twice, between 1.5fold and pretzel
    PRETZEL = "pretzel"  # Tight coil, 3+ body segments (formerly 3fold)
    HATCHING = "hatching"  # Active emergence, shell breach visible
    HATCHED = "hatched"  # Fully emerged L1 larva
    ARRESTED = "arrested"  # Dead or developmentally arrested embryo (special state)
    NO_OBJECT = "no_object"  # No embryo visible in field of view (special state)

    @classmethod
    def ordered_list(cls) -> list["DevelopmentalStage"]:
        """Return stages in developmental order."""
        return [
            cls.EARLY,
            cls.BEAN,
            cls.COMMA,
            cls.FOLD_1_5,
            cls.FOLD_2,
            cls.PRETZEL,
            cls.HATCHING,
            cls.HATCHED,
        ]

    @classmethod
    def ordered_values(cls) -> list[str]:
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
    def all_valid_values(cls) -> list[str]:
        """Return all valid stage values including special states like 'arrested'."""
        return cls.ordered_values() + ["arrested", "no_object"]

    @classmethod
    def is_special_state(cls, stage: str) -> bool:
        """Check if this is a special state (not part of normal progression)."""
        return stage in (cls.ARRESTED.value, cls.NO_OBJECT.value)

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
STAGE_CRITERIA: dict[str, dict[str, Any]] = {
    "early": {
        "features": [
            "oval/elliptical shape",
            "grainy texture with many visible nuclei",
            "uniform or slightly asymmetric cellular mass",
            "compact, no clear elongation yet",
            "round to slightly oval, no pronounced axis",
        ],
        "NOT_if": [
            "elongated bean-like shape",
            "clear axis of elongation",
            "any hint of curvature or C-shape",
        ],
        "typical_duration_min": 60,  # until bean stage
    },
    "bean": {
        "features": [
            "elongated oval, 'bean-shaped' appearance",
            "clear axis of elongation established",
            "slightly asymmetric - one end may be narrower",
            "pre-comma curvature - hint of bend but not C-shaped",
            "smooth outline, no tight curvature yet",
        ],
        "NOT_if": [
            "still round/spherical (that's early)",
            "clear C-curve or comma shape (that's comma)",
            "pronounced ventral bend",
        ],
        "typical_duration_min": 30,
    },
    "comma": {
        "features": [
            "clear C-curve or comma shape",
            "pronounced ventral bend",
            "head/tail distinctly different",
            "body axis clearly established",
        ],
        "NOT_if": [
            "no clear bend, just elongated (that's bean)",
            "elongation beyond eggshell (that's 1.5fold)",
            "still oval/round shape (that's early)",
        ],
        "typical_duration_min": 30,
    },
    "1.5fold": {
        "features": [
            "elongated ~1.5x eggshell length",
            "embryo clearly longer than egg width",
            "body starting to fold back on itself",
            "one fold/bend visible, tail beginning to turn back",
        ],
        "NOT_if": [
            "fits within egg diameter (that's comma)",
            "two clear folds visible (that's 2fold)",
            "tight coil with 3 segments (that's pretzel)",
        ],
        "typical_duration_min": 30,
    },
    "2fold": {
        "features": [
            "body folded back on itself twice",
            "two clear bends/folds visible",
            "~2x eggshell length when straightened",
            "more compaction than 1.5fold, less than pretzel",
            "head and tail both curving inward",
        ],
        "NOT_if": [
            "only one fold visible (that's 1.5fold)",
            "tight pretzel coil with 3+ segments (that's pretzel)",
            "body extending beyond shell (that's hatching)",
        ],
        "typical_duration_min": 45,
    },
    "pretzel": {
        "features": [
            "tightly coiled pretzel-like shape",
            "3 or more body segments visible",
            "maximum compaction within shell",
            "may show occasional twitching",
            "often called '3fold' - three folds/bends",
        ],
        "NOT_if": [
            "only two folds visible (that's 2fold)",
            "any part outside shell (that's hatching)",
            "shell breach visible",
        ],
        "typical_duration_min": 60,
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
    "no_object": {
        "features": [
            "Empty field of view - no embryo or eggshell visible",
            "Only background/substrate visible",
            "May contain debris, dust particles, or imaging artifacts",
            "No recognizable biological structure",
            "Uniform or noisy background without distinct objects",
        ],
        "NOT_if": [
            "Any recognizable embryo or eggshell present",
            "Clear biological structure visible, even if unclear stage",
            "Distinct oval or round object that could be an embryo",
        ],
        "typical_duration_min": None,  # special state, not part of normal progression
    },
}


# Transition zones between stages
# Used for detecting transitional states and setting expectations for temporal analysis
TRANSITION_ZONES: dict[str, dict[str, Any]] = {
    "early_to_bean": {
        "from_stage": "early",
        "to_stage": "bean",
        "key_features": [
            "subtle elongation beginning",
            "shape becoming oval rather than round",
            "axis of elongation emerging",
            "one end may start to look slightly different",
        ],
        "duration_typical_min": 10,
        "description": "Embryo transitioning from round to elongated oval shape",
    },
    "bean_to_comma": {
        "from_stage": "bean",
        "to_stage": "comma",
        "key_features": [
            "curvature beginning to form",
            "hint of C-shape emerging",
            "ventral side starting to indent",
            "asymmetry becoming more pronounced",
        ],
        "duration_typical_min": 10,
        "description": "Bean shape developing into characteristic C-curve",
    },
    "comma_to_1.5fold": {
        "from_stage": "comma",
        "to_stage": "1.5fold",
        "key_features": [
            "C-curve deepening",
            "body starting to extend beyond egg width",
            "beginning to turn back on itself",
            "elongation becoming more pronounced",
        ],
        "duration_typical_min": 10,
        "description": "Curve becoming fold, body extending beyond shell diameter",
    },
    "1.5fold_to_2fold": {
        "from_stage": "1.5fold",
        "to_stage": "2fold",
        "key_features": [
            "first fold tightening",
            "second bend beginning to form",
            "tail curving back further",
            "body becoming more compact",
        ],
        "duration_typical_min": 15,
        "description": "Single fold developing into double fold",
    },
    "2fold_to_pretzel": {
        "from_stage": "2fold",
        "to_stage": "pretzel",
        "key_features": [
            "third fold forming",
            "body coiling tighter",
            "maximum compaction approaching",
            "pretzel-like configuration emerging",
        ],
        "duration_typical_min": 15,
        "description": "Double fold tightening into triple-fold pretzel",
    },
    "pretzel_to_hatching": {
        "from_stage": "pretzel",
        "to_stage": "hatching",
        "key_features": [
            "shell boundary becoming irregular",
            "possible weakening of shell visible",
            "increased movement/twitching",
            "pressure against shell apparent",
        ],
        "duration_typical_min": 10,
        "description": "Coiled embryo preparing to breach shell",
    },
}


def get_transition_zone(from_stage: str, to_stage: str) -> dict[str, Any]:
    """Get transition zone info between two stages."""
    key = f"{from_stage}_to_{to_stage}"
    return TRANSITION_ZONES.get(key, {})


def get_adjacent_stages(stage: str) -> tuple:
    """Get the previous and next stages for a given stage."""
    ordered = DevelopmentalStage.ordered_values()
    try:
        idx = ordered.index(stage)
        prev_stage = ordered[idx - 1] if idx > 0 else None
        next_stage = ordered[idx + 1] if idx < len(ordered) - 1 else None
        return prev_stage, next_stage
    except ValueError:
        return None, None


def get_stage_description(stage: str) -> str:
    """Get a brief description for a stage."""
    descriptions = {
        "early": "Gastrulation through early morphogenesis, round to oval shape",
        "bean": "Elongated oval, bean-shaped, pre-comma curvature",
        "comma": "Clear C-shaped curve with established body axis",
        "1.5fold": "Elongation beginning, embryo ~1.5x shell length, one fold",
        "2fold": "Body folded twice, more compact than 1.5fold",
        "pretzel": "Tightly coiled, 3+ body segments visible (3fold)",
        "hatching": "Active emergence through shell breach",
        "hatched": "Fully emerged L1 larva",
        "arrested": "Development arrested - dead or stalled embryo",
        "no_object": "No embryo visible in field of view",
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
