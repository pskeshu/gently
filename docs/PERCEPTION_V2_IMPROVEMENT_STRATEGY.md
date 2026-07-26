# C. elegans Embryo Perception System v2 Improvement Strategy

## Executive Summary

This document provides a comprehensive analysis of the current perception system's issues and detailed improvement strategies for the v2 refactor. The analysis is based on code review, example image examination, and the user-reported issues.

---

## Table of Contents

1. [Current Architecture Overview](#1-current-architecture-overview)
2. [Identified Issues](#2-identified-issues)
3. [Root Cause Analysis](#3-root-cause-analysis)
4. [Improvement Strategies](#4-improvement-strategies)
5. [Implementation Roadmap](#5-implementation-roadmap)
6. [Code Changes Summary](#6-code-changes-summary)

---

## 1. Current Architecture Overview

### 1.1 Key Components

| File | Purpose |
|------|---------|
| `gently/agent/perception/engine.py` | Core VLM interaction, prompt building, stage classification |
| `gently/agent/perception/session.py` | Per-embryo session state, stage history, confidence tracking |
| `gently/agent/perception/manager.py` | Multi-embryo coordination, session lifecycle |
| `gently/agent/perception/example_store.py` | Reference image loading and encoding |
| `gently/agent/developmental_tracker.py` | Alternative tracker with different stage enum |

### 1.2 Stage Definitions (Current)

**CRITICAL ISSUE: Inconsistent stage definitions across files**

```
engine.py:        ["early", "bean", "comma", "1.5fold", "2fold", "3fold", "hatching", "hatched"]
example_store.py: ["early", "comma", "pretzel", "3fold", "hatching", "hatched"]  # MISSING: bean, 1.5fold, 2fold
stage_annotator:  ["early", "bean", "comma", "1.5fold", "2fold", "3fold", "hatched"]  # MISSING: hatching
dev_tracker.py:   Uses different enum (GASTRULATION, COMMA, ONE_POINT_FIVE_FOLD, etc.)
```

### 1.3 Example Images Structure

```
gently/examples/stages/
    early/      (3 examples)
    bean/       (3 examples)
    comma/      (3 examples)
    1.5fold/    (3 examples)
    2fold/      (3 examples)
    3fold/      (3 examples)
    hatched/    (3 examples)

MISSING: hatching/ folder - NO HATCHING EXAMPLES!
```

---

## 2. Identified Issues

### 2.1 Bean Stage Detection is Late

**Symptoms:**
- Bean stage is detected later than expected
- Bean and comma stages overlap temporally
- Transitions appear delayed

**Evidence from Example Images:**
- Bean example_001.jpg: Shows elongated embryo with curvature - resembles early comma
- Bean example_002.jpg: Even more elongated with pronounced curve
- These "bean" examples may actually be early comma stages

### 2.2 Transition Definitions Need Improvement

**Current Approach (session.py:67-83):**
```python
def at_least_stage(self, stage: str) -> bool:
    """Check if embryo has reached at least the given stage."""
    order = ["early", "bean", "comma", "1.5fold", "2fold", "3fold", "hatching", "hatched"]
    # ... simple linear comparison
```

**Problems:**
- Binary "at_least" check doesn't capture transition states
- No concept of "transitioning between" stages
- No hysteresis to prevent oscillation between adjacent stages

### 2.3 Pretzel/Hatching/Hatched Confusion

**Root Causes:**
1. No "hatching" example images exist
2. Prompt describes hatching as "emergence in progress" but this is vague
3. "Pretzel" appears in example_store.py but not in engine.py stages
4. 3fold/pretzel terminology is inconsistent

### 2.4 Long Hatching Phase Misclassified

**Problem:** Embryos in active hatching (eggshell breach, partial emergence) are classified as "hatched"

**Cause:**
- No reference images for the hatching process
- VLM prompt doesn't clearly distinguish:
  - 3fold (tightly coiled, occasional movement)
  - hatching (active breach, partial emergence)
  - hatched (fully emerged, empty shell)

---

## 3. Root Cause Analysis

### 3.1 Example Image Quality Issues

After visual inspection of the current examples:

| Stage | Assessment | Issue |
|-------|------------|-------|
| early | Good | Clear gastrulation morphology |
| bean | Poor | Examples show comma-like features (elongation, curvature) |
| comma | Variable | Some show extreme constriction (peanut shape) |
| 1.5fold | Adequate | Need more variety |
| 2fold | Adequate | Need more variety |
| 3fold | Good | Clear pretzel morphology |
| hatching | MISSING | No examples at all |
| hatched | Good | Clear emerged larvae |

### 3.2 Prompt Engineering Weaknesses

**Current prompt issues in engine.py:**
1. Stage descriptions are too brief
2. No explicit guidance on boundary cases
3. Confidence calibration is undefined
4. No mention of how to handle transitional states

### 3.3 Architectural Gaps

1. **No temporal smoothing**: Each frame classified independently
2. **No transition detection**: No explicit "entering stage X" vs "clearly in stage X"
3. **No confidence calibration**: "HIGH/MEDIUM/LOW" without quantitative backing
4. **No regression prevention**: System can classify backward in development

---

## 4. Improvement Strategies

### 4.1 Stage Definition Standardization

**Recommendation: Unify all stage definitions**

Create a single source of truth:

```python
# gently/agent/perception/stages.py (NEW FILE)

from enum import Enum
from typing import List, Dict, Tuple


class DevelopmentalStage(str, Enum):
    """Unified C. elegans developmental stages."""

    EARLY = "early"  # Gastrulation, ~100+ cells, oval shape
    BEAN = "bean"  # Early morphogenesis, slight asymmetry
    COMMA = "comma"  # Clear C-shape, head/tail distinguishable
    FOLD_1_5 = "1.5fold"  # Elongation, ~1.5x eggshell
    FOLD_2 = "2fold"  # Further elongation, folding back
    FOLD_3 = "3fold"  # Tight coil, maximum compaction (pretzel)
    HATCHING = "hatching"  # Active emergence, shell breach visible
    HATCHED = "hatched"  # Fully emerged L1 larva

    @classmethod
    def ordered_list(cls) -> List["DevelopmentalStage"]:
        return [
            cls.EARLY,
            cls.BEAN,
            cls.COMMA,
            cls.FOLD_1_5,
            cls.FOLD_2,
            cls.FOLD_3,
            cls.HATCHING,
            cls.HATCHED,
        ]

    @classmethod
    def get_order(cls, stage: "DevelopmentalStage") -> int:
        return cls.ordered_list().index(stage)

    @classmethod
    def is_terminal(cls, stage: "DevelopmentalStage") -> bool:
        return stage == cls.HATCHED


# Stage boundary definitions for VLM guidance
STAGE_BOUNDARIES = {
    DevelopmentalStage.EARLY: {
        "entry_features": ["oval/elliptical shape", "grainy texture", "uniform cellular mass"],
        "exit_features": ["slight elongation", "asymmetry appearing"],
        "NOT_this_stage": ["any curvature", "head/tail distinction"],
    },
    DevelopmentalStage.BEAN: {
        "entry_features": ["slight elongation", "one end narrowing", "kidney shape hint"],
        "exit_features": ["clear C-curve", "pronounced bend"],
        "NOT_this_stage": ["compact oval", "full comma bend"],
    },
    # ... etc
}
```

### 4.2 Example Image Curation Guidelines

**Immediate Actions:**

1. **Create hatching examples (CRITICAL)**
   - Need 5-10 examples showing:
     - Early hatching: Shell breach visible, embryo still mostly inside
     - Mid hatching: Head/anterior emerging
     - Late hatching: Mostly out, tail still inside

2. **Re-curate bean examples**
   - Current bean examples appear to be early comma
   - True bean should show:
     - Subtle asymmetry only
     - No clear C-curve
     - "Kidney bean" shape, not "comma"

3. **Add transition examples**
   - Create a new category: boundary examples
   - "bean_to_comma_transition", "3fold_to_hatching_transition"
   - Use as negative examples ("this is NOT yet comma, still bean")

**Example Curation Protocol:**

```markdown
## Example Image Curation Checklist

### For each stage, examples should:
- [ ] Be clearly IN the stage (not transitional)
- [ ] Come from different embryos (diversity)
- [ ] Show typical morphology (not edge cases)
- [ ] Have good image quality (in focus, good contrast)

### Examples to EXCLUDE:
- Transitional images (save for boundary training)
- Unusual morphologies
- Poor quality images
- Ambiguous cases

### Minimum per stage: 5 examples
### Recommended per stage: 10 examples

### Boundary examples (separate folder):
- 3-5 examples of "just before" transition
- 3-5 examples of "just after" transition
```

### 4.3 VLM Prompt Improvements

**New prompt structure for engine.py:**

```python
STAGE_CLASSIFICATION_PROMPT_V2 = """
You are analyzing a C. elegans embryo image to determine its developmental stage.

## IMAGE LAYOUT
The image shows two views side-by-side:
- LEFT (TOP view): Looking down at embryo (XY max projection)
- RIGHT (SIDE view): Profile view (XZ max projection, rotated)
Both views are from the same timepoint and should be analyzed together.

## DEVELOPMENTAL STAGES (chronological order)

### EARLY (gastrulation stage)
- Shape: Oval/elliptical, compact
- Texture: Grainy, many visible nuclei (~100+ cells)
- Key feature: NO elongation, NO asymmetry
- NOT early if: Any curvature, head/tail distinction, kidney shape

### BEAN
- Shape: Slightly elongated, subtle kidney shape
- Key feature: One end slightly narrower than the other
- Asymmetry: Beginning but NOT pronounced
- NOT bean if: Clear C-curve (that's comma), fully symmetric (that's early)
- Duration: Brief stage, easy to miss

### COMMA
- Shape: Clear C-curve or comma shape
- Key feature: Pronounced ventral bend
- Head/tail: Distinctly different ends
- NOT comma if: Only subtle asymmetry (that's bean), elongation beyond eggshell (that's 1.5fold)

### 1.5FOLD
- Shape: Elongated ~1.5x eggshell length
- Key feature: Embryo clearly longer than egg width
- Body starts folding back on itself
- NOT 1.5fold if: Fits within egg diameter (comma), tight coil (2fold+)

### 2FOLD
- Shape: Elongated ~2x, folding back
- Key feature: Body doubled back significantly
- More compact than 1.5fold within shell
- NOT 2fold if: Only slight fold (1.5fold), tight pretzel (3fold)

### 3FOLD (pretzel)
- Shape: Tightly coiled, 3 body segments visible
- Key feature: Maximum compaction, pretzel-like
- May show occasional twitching
- NOT 3fold if: Active sustained movement (hatching), loose coil (2fold)

### HATCHING (CRITICAL - distinguish from hatched!)
- Key feature: ACTIVE EMERGENCE IN PROGRESS
- Eggshell: Breach/tear VISIBLE
- Embryo: Partially inside, partially outside
- Movement: Active pushing/wriggling to escape
- NOT hatching if: Fully inside (3fold), fully outside (hatched)

### HATCHED
- Key feature: Larva FULLY OUTSIDE eggshell
- Empty or nearly-empty eggshell visible
- Free-swimming/crawling L1 larva
- NOT hatched if: Any part still inside shell (that's hatching)

## CLASSIFICATION RULES

1. Use BOTH views (top and side) - side view is especially useful for fold stages
2. If between stages, classify as the EARLIER stage unless features are definitive
3. HATCHING vs HATCHED: Look for shell breach AND partial emergence (hatching) vs complete emergence (hatched)
4. Confidence should reflect how clearly the stage criteria are met

## RESPONSE FORMAT (JSON)

{
    "stage": "stage_name",
    "confidence": 0.85,
    "primary_features": ["feature1", "feature2"],
    "rules_out": ["excluded_stage1", "excluded_stage2"],
    "reasoning": "Brief explanation"
}
"""
```

### 4.4 Transition Detection Logic

**New TransitionDetector class:**

```python
# gently/agent/perception/transitions.py (NEW FILE)

from dataclasses import dataclass
from typing import List, Optional, Tuple
from .stages import DevelopmentalStage


@dataclass
class TransitionState:
    """Represents the current transition state."""

    current_stage: DevelopmentalStage
    confidence: float
    is_transitioning: bool
    next_stage: Optional[DevelopmentalStage]
    transition_progress: float  # 0.0 to 1.0


class TransitionDetector:
    """
    Detects and tracks stage transitions with temporal smoothing.

    Key features:
    - Hysteresis: Requires N confirmations before stage change
    - No regression: Cannot go backward in development (except hatched->hatching edge case)
    - Transition tracking: Reports "transitioning to X" before full transition
    """

    def __init__(
        self,
        confirmation_threshold: int = 2,
        transition_confidence_threshold: float = 0.7,
        regression_allowed: bool = False,
    ):
        self.confirmation_threshold = confirmation_threshold
        self.transition_confidence_threshold = transition_confidence_threshold
        self.regression_allowed = regression_allowed

        # State tracking
        self.confirmed_stage: Optional[DevelopmentalStage] = None
        self.pending_stage: Optional[DevelopmentalStage] = None
        self.pending_count: int = 0
        self.history: List[Tuple[DevelopmentalStage, float]] = []

    def update(self, detected_stage: DevelopmentalStage, confidence: float) -> TransitionState:
        """
        Update with new detection and return transition state.

        Parameters
        ----------
        detected_stage : DevelopmentalStage
            Stage detected by VLM
        confidence : float
            Detection confidence (0.0 to 1.0)

        Returns
        -------
        TransitionState
            Current transition state with smoothed stage
        """
        self.history.append((detected_stage, confidence))

        # Initialize if first detection
        if self.confirmed_stage is None:
            self.confirmed_stage = detected_stage
            return TransitionState(
                current_stage=detected_stage,
                confidence=confidence,
                is_transitioning=False,
                next_stage=None,
                transition_progress=0.0,
            )

        # Check for regression (going backward)
        if not self.regression_allowed:
            current_order = DevelopmentalStage.get_order(self.confirmed_stage)
            detected_order = DevelopmentalStage.get_order(detected_stage)
            if detected_order < current_order:
                # Reject regression, maintain current stage
                return TransitionState(
                    current_stage=self.confirmed_stage,
                    confidence=confidence,
                    is_transitioning=False,
                    next_stage=None,
                    transition_progress=0.0,
                )

        # Same as confirmed - reset pending
        if detected_stage == self.confirmed_stage:
            self.pending_stage = None
            self.pending_count = 0
            return TransitionState(
                current_stage=self.confirmed_stage,
                confidence=confidence,
                is_transitioning=False,
                next_stage=None,
                transition_progress=0.0,
            )

        # Different stage detected
        if detected_stage == self.pending_stage:
            self.pending_count += 1
        else:
            self.pending_stage = detected_stage
            self.pending_count = 1

        # Check if should confirm transition
        if self.pending_count >= self.confirmation_threshold:
            # Confirm transition
            self.confirmed_stage = self.pending_stage
            self.pending_stage = None
            self.pending_count = 0
            return TransitionState(
                current_stage=self.confirmed_stage,
                confidence=confidence,
                is_transitioning=False,
                next_stage=None,
                transition_progress=1.0,
            )

        # Still transitioning
        progress = self.pending_count / self.confirmation_threshold
        return TransitionState(
            current_stage=self.confirmed_stage,
            confidence=confidence,
            is_transitioning=True,
            next_stage=self.pending_stage,
            transition_progress=progress,
        )

    def get_smoothed_stage(self, window: int = 3) -> DevelopmentalStage:
        """Get mode stage from recent history."""
        if not self.history:
            return DevelopmentalStage.EARLY

        recent = self.history[-window:]
        stages = [s for s, c in recent]
        return max(set(stages), key=stages.count)
```

### 4.5 Confidence Calibration

**New confidence system:**

```python
# gently/agent/perception/confidence.py (NEW FILE)

from dataclasses import dataclass
from typing import Dict
from .stages import DevelopmentalStage


@dataclass
class CalibratedConfidence:
    """Calibrated confidence with interpretable thresholds."""

    raw_score: float  # 0.0 to 1.0 from VLM
    calibrated_score: float  # Adjusted based on stage difficulty
    interpretation: str  # "high", "medium", "low"

    @property
    def is_reliable(self) -> bool:
        return self.calibrated_score >= 0.7


# Stage-specific confidence calibration
# Some stages are harder to classify, adjust thresholds accordingly
STAGE_DIFFICULTY = {
    DevelopmentalStage.EARLY: 0.0,  # Easy - distinct morphology
    DevelopmentalStage.BEAN: 0.3,  # Hard - brief, subtle
    DevelopmentalStage.COMMA: 0.2,  # Medium - can overlap with bean
    DevelopmentalStage.FOLD_1_5: 0.2,  # Medium
    DevelopmentalStage.FOLD_2: 0.2,  # Medium
    DevelopmentalStage.FOLD_3: 0.1,  # Easy - distinct pretzel
    DevelopmentalStage.HATCHING: 0.25,  # Medium-hard - brief window
    DevelopmentalStage.HATCHED: 0.0,  # Easy - distinct
}


def calibrate_confidence(
    raw_confidence: float,
    stage: DevelopmentalStage,
) -> CalibratedConfidence:
    """
    Calibrate confidence based on stage difficulty.

    Harder-to-classify stages get a confidence penalty.
    """
    difficulty = STAGE_DIFFICULTY.get(stage, 0.2)
    calibrated = raw_confidence * (1.0 - difficulty * 0.3)

    if calibrated >= 0.8:
        interpretation = "high"
    elif calibrated >= 0.6:
        interpretation = "medium"
    else:
        interpretation = "low"

    return CalibratedConfidence(
        raw_score=raw_confidence,
        calibrated_score=calibrated,
        interpretation=interpretation,
    )
```

### 4.6 Hatching Detection Improvements

**Specific improvements for hatching/hatched distinction:**

```python
# Add to engine.py or create hatching_detector.py

HATCHING_SPECIFIC_PROMPT = """
Analyze this C. elegans embryo for HATCHING STATUS.

Focus specifically on:
1. EGGSHELL INTEGRITY
   - Intact (no breach) -> NOT hatching
   - Breach visible (tear/hole) -> Could be hatching
   - Shell mostly empty -> Likely hatched

2. EMBRYO POSITION RELATIVE TO SHELL
   - Fully inside shell -> NOT hatching (3fold or earlier)
   - Partially emerged (head out, tail in) -> HATCHING
   - Fully outside shell -> HATCHED

3. MOVEMENT PATTERN
   - Occasional twitching inside shell -> 3fold
   - Active pushing/wriggling at breach -> HATCHING
   - Free movement outside shell -> HATCHED

## KEY DISTINCTION: HATCHING vs HATCHED

HATCHING:
- Shell breach IS visible
- Part of embryo IS outside
- Part of embryo IS STILL INSIDE
- Active emergence behavior

HATCHED:
- Larva COMPLETELY outside
- Empty shell visible (may be nearby or out of frame)
- No part of worm touching shell interior

Return JSON:
{
    "shell_status": "intact" | "breached" | "empty",
    "embryo_position": "inside" | "partial" | "outside",
    "movement": "none" | "twitching" | "emerging" | "free",
    "classification": "not_hatching" | "hatching" | "hatched",
    "confidence": 0.0-1.0,
    "reasoning": "explanation"
}
"""


class HatchingDetector:
    """Specialized detector for hatching stages."""

    def __init__(self, engine: "PerceptionEngine"):
        self.engine = engine
        self.consecutive_hatching = 0
        self.consecutive_hatched = 0

    async def detect(self, image_b64: str) -> dict:
        """
        Detect hatching status with specialized prompt.

        Uses stricter criteria than general stage classification.
        """
        result = await self.engine.classify_with_prompt(
            image_b64=image_b64,
            prompt=HATCHING_SPECIFIC_PROMPT,
        )

        # Track consecutive detections for confirmation
        if result.get("classification") == "hatching":
            self.consecutive_hatching += 1
            self.consecutive_hatched = 0
        elif result.get("classification") == "hatched":
            self.consecutive_hatched += 1
            # Require 2+ consecutive "hatched" to confirm
            # (hatching can look like hatched momentarily)
            if self.consecutive_hatched < 2:
                result["classification"] = "hatching"
                result["note"] = "Awaiting confirmation of hatched status"
        else:
            self.consecutive_hatching = 0
            self.consecutive_hatched = 0

        return result
```

---

## 5. Implementation Roadmap

### Phase 1: Critical Fixes (Week 1)

1. **Create hatching examples**
   - Location: `gently/examples/stages/hatching/`
   - Minimum: 5 examples showing different hatching phases
   - Use stage_annotator.py to curate from existing timelapses

2. **Unify stage definitions**
   - Create `stages.py` as single source of truth
   - Update engine.py, session.py, example_store.py to import from stages.py
   - Add "hatching" to stage_annotator.py

3. **Fix example_store.py stage list**
   - Current: `["early", "comma", "pretzel", "3fold", "hatching", "hatched"]`
   - Should be: `["early", "bean", "comma", "1.5fold", "2fold", "3fold", "hatching", "hatched"]`

### Phase 2: Prompt Improvements (Week 2)

1. **Implement improved prompts**
   - Replace current stage descriptions with detailed V2 prompt
   - Add explicit "NOT this stage" criteria
   - Add hatching-specific prompt

2. **Re-curate bean examples**
   - Review all current bean examples
   - Replace with true bean-stage embryos (pre-comma)
   - May need to collect new examples

### Phase 3: Transition Logic (Week 3)

1. **Implement TransitionDetector**
   - Add hysteresis for stage changes
   - Implement regression prevention
   - Add transition progress tracking

2. **Integrate with PerceptionSession**
   - Replace simple at_least_stage with transition-aware logic
   - Add transition state to session history

### Phase 4: Testing and Validation (Week 4)

1. **Create test dataset**
   - Ground-truth labeled timelapse
   - Calculate accuracy per stage
   - Focus on bean/comma and hatching/hatched boundaries

2. **Tune thresholds**
   - Confirmation threshold for transitions
   - Confidence thresholds per stage
   - Hatching confirmation requirements

---

## 6. Code Changes Summary

### Files to Modify

| File | Changes |
|------|---------|
| `engine.py` | New prompts, JSON confidence output, hatching-specific detection |
| `session.py` | Integrate TransitionDetector, update at_least_stage logic |
| `manager.py` | Pass transition state to events |
| `example_store.py` | Update stage list to match engine.py |
| `stage_annotator.py` | Add "hatching" stage |

### New Files to Create

| File | Purpose |
|------|---------|
| `stages.py` | Unified stage definitions |
| `transitions.py` | TransitionDetector class |
| `confidence.py` | Confidence calibration |
| `hatching_detector.py` | Specialized hatching detection (optional) |

### New Example Folders

```
gently/examples/stages/
    hatching/           # NEW - 5-10 examples
    boundaries/         # NEW - transition examples
        bean_to_comma/
        3fold_to_hatching/
        hatching_to_hatched/
```

---

## Appendix A: Bean vs Comma Visual Criteria

### TRUE BEAN (what examples should show)
- Shape: Slightly asymmetric oval
- One end marginally narrower
- NO clear bend/curve
- Looks like a kidney bean rotated 45 degrees
- Cell mass still relatively uniform

### TRUE COMMA (what examples should show)
- Shape: Clear C-curve
- Pronounced ventral bend
- Head region distinct from tail
- Body axis clearly established
- Beginning of elongation visible

### DISTINGUISHING FEATURE
The key question: "Is there a clear BEND in the body axis?"
- No bend, just asymmetric ends -> BEAN
- Clear bend forming C-shape -> COMMA

---

## Appendix B: Hatching Timeline Breakdown

```
3FOLD (pre-hatching):
[0-5 min before breach]
- Tight coil inside shell
- Occasional twitching
- Shell fully intact

EARLY HATCHING:
[0-5 min after breach]
- Shell breach visible (small hole/tear)
- Embryo still mostly inside
- Active pushing movements

MID HATCHING:
[5-15 min]
- Head/anterior half emerged
- Posterior half still inside
- Active wriggling

LATE HATCHING:
[15-30 min]
- Mostly emerged
- Only tail still inside
- Almost free

HATCHED:
[After emergence]
- Fully outside shell
- Empty shell visible
- Free-moving L1 larva
```

---

## Appendix C: Confidence Interpretation Guide

| Confidence | Meaning | Action |
|------------|---------|--------|
| >= 0.85 | High - Clear stage features | Trust classification |
| 0.70-0.84 | Medium - Most features present | Log, may need review |
| 0.50-0.69 | Low - Ambiguous | Flag for review, use temporal context |
| < 0.50 | Very Low - Unclear | Consider "unknown", check image quality |

---

*Document created: 2024-12-20*
*Author: Claude (VLM Microscopy Consultant)*
