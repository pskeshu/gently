"""
C. elegans detector presets.

VLM prompts for detecting specific developmental events
(hatching, comma stage, pretzel, gastrulation, first division).
"""


def get_detector_presets() -> dict:
    """
    Get predefined detector presets for common C. elegans stages.

    Returns
    -------
    dict
        Preset detector configurations keyed by event name.
    """
    return {
        "hatching": {
            "name": "hatching",
            "description": "Detects when C. elegans embryo hatches from eggshell",
            "prompt": """Analyze this C. elegans embryo image (diSPIM light sheet max
projection) and determine if the embryo has HATCHED.

TRUE HATCHING looks like (must meet at least one):
- Most or all of the worm body is OUTSIDE the eggshell boundary
- Worm is free-floating, elongated, "worm-like" in the field of view (NOT coiled/intertwined)
- Empty field of view where embryo used to be (worm has left entirely)
- Worm partially visible, moving in/out of the frame (not confined to original egg location)
- Clear spatial separation between the worm body and the (now-empty or deflated) eggshell

NOT HATCHING - common false positives to AVOID:
- Worm is still coiled/pretzel-shaped INSIDE an expanded eggshell
- Eggshell appears stretched or larger, but worm remains CONTAINED within
- Worm appears to fill the eggshell completely but NO part extends beyond the boundary
- Vigorous movement WITHIN the shell (even if dramatic) without actual breach
- Late 3-fold stage where worm is tightly packed but still enclosed

CRITICAL: The worm must have PHYSICALLY EXITED through a visible breach point.
Simply appearing "ready to hatch" or having an expanded shell is NOT hatching.

Focus on the CURRENT/LATEST image (the final one shown).

Respond in this exact format:
DETECTED: [YES/NO]
CONFIDENCE: [HIGH/MEDIUM/LOW]
REASONING: [Brief explanation - specifically state if worm is INSIDE or OUTSIDE the shell]""",
            "use_temporal_context": True,
            "temporal_context_size": 10,
            "confidence_threshold": "HIGH",
            "stop_timelapse": True,  # Auto-stop when hatching detected
        },
        "comma": {
            "name": "comma",
            "description": "Detects comma stage (major morphogenesis)",
            "prompt": """Analyze this C. elegans embryo and determine if it has reached the
COMMA STAGE.

Key characteristics of comma stage (~400 minutes, ~6.5 hours):
- Distinct comma or bean shape (ventral curvature)
- Clear anterior-posterior elongation
- Visible head/tail differentiation
- Movement patterns visible
- Still within eggshell

Focus on the CURRENT/LATEST image.

Respond in this exact format:
DETECTED: [YES/NO]
CONFIDENCE: [HIGH/MEDIUM/LOW]
REASONING: [Brief explanation]""",
            "use_temporal_context": True,
            "temporal_context_size": 5,
            "confidence_threshold": "MEDIUM",
        },
        "pretzel": {
            "name": "pretzel",
            "description": "Detects pretzel/3-fold stage (highly elongated)",
            "prompt": """Analyze this C. elegans embryo and determine if it has reached the
PRETZEL/3-FOLD STAGE.

Key characteristics of 3-fold stage (~550 minutes, ~9 hours):
- Highly elongated, approximately 3x the eggshell length
- Tightly folded/coiled within eggshell (pretzel-like)
- Active movement visible
- Clear segmentation and pharynx structure
- Still within eggshell

Focus on the CURRENT/LATEST image.

Respond in this exact format:
DETECTED: [YES/NO]
CONFIDENCE: [HIGH/MEDIUM/LOW]
REASONING: [Brief explanation]""",
            "use_temporal_context": True,
            "temporal_context_size": 5,
            "confidence_threshold": "MEDIUM",
        },
        "gastrulation": {
            "name": "gastrulation",
            "description": "Detects onset of gastrulation",
            "prompt": """Analyze this C. elegans embryo and determine if GASTRULATION has begun.

Key characteristics of gastrulation (~210 minutes, ~3.5 hours):
- Visible internalization of cells (especially E cells - gut precursors)
- Loss of clear spherical shape
- Cell movements visible
- Typically after ~26-28 cell stage

Focus on the CURRENT/LATEST image.

Respond in this exact format:
DETECTED: [YES/NO]
CONFIDENCE: [HIGH/MEDIUM/LOW]
REASONING: [Brief explanation]""",
            "use_temporal_context": True,
            "temporal_context_size": 5,
            "confidence_threshold": "MEDIUM",
        },
        "first_division": {
            "name": "first_division",
            "description": "Detects first cell division (1-cell to 2-cell)",
            "prompt": """Analyze this C. elegans embryo and determine if FIRST CELL DIVISION
has occurred.

Key characteristics:
- Transition from single large cell to two cells
- Unequal division: larger AB cell (anterior) and smaller P1 cell (posterior)
- Clear cell boundary/cleavage plane visible
- Occurs ~40-50 minutes after fertilization

Focus on the CURRENT/LATEST image.

Respond in this exact format:
DETECTED: [YES/NO]
CONFIDENCE: [HIGH/MEDIUM/LOW]
REASONING: [Brief explanation]""",
            "use_temporal_context": True,
            "temporal_context_size": 3,
            "confidence_threshold": "HIGH",
        },
    }
