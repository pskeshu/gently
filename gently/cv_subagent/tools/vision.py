"""
Claude Vision Tools for CV Agent

Tools for visual analysis using Claude's vision capabilities.
These tools enable the agent to analyze embryo images with rich context.
"""

import base64
import logging
import os
from typing import Any, Dict, List, Optional

from .registry import cv_tool, ToolCategory, ToolExample
from .preparation import get_prepared_image

logger = logging.getLogger(__name__)


def _resolve_image(image_input: str) -> Optional[str]:
    """Resolve image input to base64 data.

    Accepts either:
    - A prepared_image_uid (e.g., "prepared_vol_abc_123456")
    - Direct base64 data
    """
    if image_input.startswith("prepared_"):
        # Look up from cache
        base64_data = get_prepared_image(image_input)
        if base64_data is None:
            logger.error(f"Prepared image {image_input} not found in cache")
            return None
        return base64_data
    else:
        # Assume it's direct base64
        return image_input


# =============================================================================
# Claude Vision Analysis
# =============================================================================

@cv_tool(
    name="claude_vision_analyze",
    description="""Analyze an image using Claude Vision with rich context.

Use this tool AFTER preparing the image with scale bars and annotations.
Pass the prepared_image_uid from prepare_for_vision.

Provide detailed context in the prompt including:
- Nuclei count from segmentation
- Morphology metrics (elongation, circularity)
- Scale information
- What specifically to look for

The more quantitative context you provide, the more accurate the analysis.""",
    category=ToolCategory.VISION,
    examples=[
        ToolExample("Analyze embryo image", {"image_input": "prepared_vol_abc_123456", "prompt": "This image shows a C. elegans embryo with 24 nuclei. What developmental stage is this?"}),
        ToolExample("Check for anomalies", {"image_input": "prepared_vol_xyz_789012", "prompt": "Check this embryo for any developmental abnormalities. Elongation ratio is 2.1."}),
    ],
)
def claude_vision_analyze(
    image_input: str,
    prompt: str,
    include_developmental_context: bool = True,
) -> Dict[str, Any]:
    """
    Analyze an image using Claude Vision

    Parameters
    ----------
    image_input : str
        Either a prepared_image_uid from prepare_for_vision, or direct base64 data
    prompt : str
        Analysis prompt with context about what to look for
    include_developmental_context : bool
        If True, prepend C. elegans developmental context

    Returns
    -------
    dict
        analysis: Claude's analysis text
        model: Model used
        tokens_used: Approximate tokens used
    """
    # Resolve image input to base64
    image_base64 = _resolve_image(image_input)
    if image_base64 is None:
        return {
            "error": f"Could not resolve image: {image_input}",
            "analysis": None,
        }

    logger.info(f"Running Claude Vision analysis, prompt length={len(prompt)}")

    # Get API key
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        return {
            "error": "ANTHROPIC_API_KEY not set",
            "analysis": None,
        }

    # Build the analysis prompt
    system_prompt = ""
    if include_developmental_context:
        system_prompt = """You are an expert in C. elegans embryo development analysis.

C. elegans Developmental Stages (by nuclei count):
- 1-cell: 1 nucleus (zygote, P0)
- 2-cell: 2 nuclei (AB and P1)
- 4-cell: 4 nuclei (ABa, ABp, EMS, P2)
- 8-cell: 8 nuclei
- ~28-cell: gastrulation begins, cells move inward
- gastrula: ~28-100 nuclei, active cell movements
- comma: ~550 nuclei, elongation begins (embryo forms comma shape)
- 1.5-fold: embryo folded 1.5x its width
- 2-fold: embryo folded 2x its width
- 3-fold: embryo folded 3x its width (maximum elongation)
- pretzel: tightly coiled, twitching movement begins
- hatching: larva emerges from egg shell

Key morphological features:
- Early stages: spherical cells, visible cell boundaries
- Gastrulation: cells move inward, cavity forms
- Elongation: embryo lengthens, muscle development
- Fold stages: measured by length-to-width ratio
- Late stages: pharynx visible, twitching movement

When analyzing, consider both visual features AND any quantitative data provided."""

    try:
        import anthropic
        client = anthropic.Anthropic(api_key=api_key)

        # Build message with image
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": "image/png",
                            "data": image_base64,
                        },
                    },
                    {
                        "type": "text",
                        "text": prompt,
                    },
                ],
            }
        ]

        # Call Claude Vision
        response = client.messages.create(
            model="claude-sonnet-4-20250514",  # Vision-capable model
            max_tokens=2048,
            system=system_prompt if system_prompt else None,
            messages=messages,
        )

        # Extract response
        analysis_text = ""
        for block in response.content:
            if hasattr(block, 'text'):
                analysis_text += block.text

        return {
            "analysis": analysis_text,
            "model": response.model,
            "tokens_used": {
                "input": response.usage.input_tokens,
                "output": response.usage.output_tokens,
            },
        }

    except ImportError:
        return {
            "error": "anthropic package not installed",
            "analysis": None,
        }
    except Exception as e:
        logger.error(f"Vision analysis failed: {e}")
        return {
            "error": str(e),
            "analysis": None,
        }


@cv_tool(
    name="classify_developmental_stage",
    description="""Classify the developmental stage of a C. elegans embryo.

This is a high-level tool that combines vision analysis with quantitative data.
Pass the prepared_image_uid from prepare_for_vision.
Provide nuclei count and morphology metrics for best results.""",
    category=ToolCategory.VISION,
    examples=[
        ToolExample("Classify stage with cell count", {"image_input": "prepared_vol_abc_123456", "nuclei_count": 24, "elongation_ratio": 1.8}),
        ToolExample("Quick classification without metrics", {"image_input": "prepared_vol_xyz_789012"}),
    ],
)
def classify_developmental_stage(
    image_input: str,
    nuclei_count: Optional[int] = None,
    elongation_ratio: Optional[float] = None,
    additional_context: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Classify embryo developmental stage

    Parameters
    ----------
    image_input : str
        Either a prepared_image_uid from prepare_for_vision, or direct base64 data
    nuclei_count : int, optional
        Number of nuclei detected by segmentation
    elongation_ratio : float, optional
        Length/width ratio from morphology analysis
    additional_context : str, optional
        Any additional context (e.g., observed features)

    Returns
    -------
    dict
        stage: Predicted developmental stage
        confidence: Confidence score (0-1)
        reasoning: Explanation of classification
        alternative_stages: Other possible stages with scores
    """
    # Build context-rich prompt
    prompt_parts = [
        "Analyze this C. elegans embryo image and classify its developmental stage.",
        "",
        "Please determine the developmental stage and provide:",
        "1. Primary stage classification",
        "2. Confidence level (high/medium/low)",
        "3. Key visual features supporting your classification",
        "4. Any alternative stages that could apply",
        "",
    ]

    # Add quantitative context
    if nuclei_count is not None:
        prompt_parts.append(f"QUANTITATIVE DATA:")
        prompt_parts.append(f"- Nuclei count from automated segmentation: {nuclei_count}")

        # Add nuclei-based stage hints
        if nuclei_count == 1:
            prompt_parts.append("  (Suggests 1-cell/zygote stage)")
        elif nuclei_count == 2:
            prompt_parts.append("  (Suggests 2-cell stage)")
        elif nuclei_count <= 4:
            prompt_parts.append("  (Suggests 4-cell stage)")
        elif nuclei_count <= 8:
            prompt_parts.append("  (Suggests 8-cell stage)")
        elif nuclei_count <= 28:
            prompt_parts.append("  (Suggests late cleavage, approaching gastrulation)")
        elif nuclei_count <= 100:
            prompt_parts.append("  (Suggests gastrula stage)")
        elif nuclei_count <= 350:
            prompt_parts.append("  (Suggests comma stage)")
        else:
            prompt_parts.append("  (Suggests fold stage or later)")

    if elongation_ratio is not None:
        prompt_parts.append(f"- Elongation ratio (length/width): {elongation_ratio:.2f}")

        # Add elongation-based hints
        if elongation_ratio < 1.5:
            prompt_parts.append("  (Suggests pre-elongation stage)")
        elif elongation_ratio < 2.0:
            prompt_parts.append("  (Suggests comma or early fold)")
        elif elongation_ratio < 3.0:
            prompt_parts.append("  (Suggests 1.5-fold to 2-fold)")
        else:
            prompt_parts.append("  (Suggests 2-fold to 3-fold)")

    if additional_context:
        prompt_parts.append(f"- Additional observations: {additional_context}")

    prompt_parts.append("")
    prompt_parts.append("Based on the image AND the quantitative data above, provide your classification.")
    prompt_parts.append("")
    prompt_parts.append("Format your response as:")
    prompt_parts.append("STAGE: [stage name]")
    prompt_parts.append("CONFIDENCE: [high/medium/low]")
    prompt_parts.append("REASONING: [explanation]")
    prompt_parts.append("ALTERNATIVES: [other possible stages, if any]")

    prompt = "\n".join(prompt_parts)

    # Run vision analysis
    result = claude_vision_analyze(
        image_input=image_input,
        prompt=prompt,
        include_developmental_context=True,
    )

    if result.get("error"):
        return result

    # Parse the response
    analysis = result.get("analysis", "")

    # Extract structured information
    stage = _extract_field(analysis, "STAGE")
    confidence_str = _extract_field(analysis, "CONFIDENCE")
    reasoning = _extract_field(analysis, "REASONING")
    alternatives = _extract_field(analysis, "ALTERNATIVES")

    # Convert confidence to numeric
    confidence_map = {"high": 0.9, "medium": 0.7, "low": 0.5}
    confidence = confidence_map.get(confidence_str.lower(), 0.6) if confidence_str else 0.6

    return {
        "stage": stage or "unknown",
        "confidence": confidence,
        "reasoning": reasoning or analysis,
        "alternative_stages": alternatives,
        "raw_analysis": analysis,
        "quantitative_context": {
            "nuclei_count": nuclei_count,
            "elongation_ratio": elongation_ratio,
        },
    }


@cv_tool(
    name="detect_visual_anomalies",
    description="""Detect visual anomalies or abnormalities in an embryo image.

Look for developmental defects, unusual morphology, or unexpected features.
Pass the prepared_image_uid from prepare_for_vision.""",
    category=ToolCategory.VISION,
    examples=[
        ToolExample("Check for anomalies", {"image_input": "prepared_vol_abc_123456", "expected_stage": "gastrula", "expected_nuclei": 200}),
        ToolExample("General anomaly detection", {"image_input": "prepared_vol_xyz_789012"}),
    ],
)
def detect_visual_anomalies(
    image_input: str,
    expected_stage: Optional[str] = None,
    expected_nuclei: Optional[int] = None,
    comparison_context: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Detect visual anomalies in embryo image

    Parameters
    ----------
    image_base64 : str
        Base64 encoded embryo image
    expected_stage : str, optional
        Expected developmental stage
    expected_nuclei : int, optional
        Expected number of nuclei
    comparison_context : str, optional
        Context about what to compare against

    Returns
    -------
    dict
        anomalies_detected: bool
        anomaly_list: List of detected anomalies
        severity: Overall severity (none/minor/moderate/severe)
        recommendations: Suggested actions
    """
    prompt_parts = [
        "Analyze this C. elegans embryo image for any anomalies or abnormalities.",
        "",
        "Look for:",
        "- Asymmetric cell divisions",
        "- Unusual cell sizes or shapes",
        "- Missing or extra cells compared to expected",
        "- Abnormal positioning of cells",
        "- Signs of cell death or fragmentation",
        "- Developmental arrest indicators",
        "- Morphological defects",
        "",
    ]

    if expected_stage:
        prompt_parts.append(f"Expected stage: {expected_stage}")
        prompt_parts.append(f"Look for features inconsistent with this stage.")
        prompt_parts.append("")

    if expected_nuclei:
        prompt_parts.append(f"Expected nuclei count: {expected_nuclei}")
        prompt_parts.append("Note any discrepancy between expected and observed.")
        prompt_parts.append("")

    if comparison_context:
        prompt_parts.append(f"Additional context: {comparison_context}")
        prompt_parts.append("")

    prompt_parts.append("Provide your analysis in this format:")
    prompt_parts.append("ANOMALIES_DETECTED: [yes/no]")
    prompt_parts.append("ANOMALY_LIST: [comma-separated list, or 'none']")
    prompt_parts.append("SEVERITY: [none/minor/moderate/severe]")
    prompt_parts.append("DESCRIPTION: [detailed description]")
    prompt_parts.append("RECOMMENDATIONS: [suggested actions]")

    prompt = "\n".join(prompt_parts)

    result = claude_vision_analyze(
        image_input=image_input,
        prompt=prompt,
        include_developmental_context=True,
    )

    if result.get("error"):
        return result

    analysis = result.get("analysis", "")

    # Parse response
    anomalies_str = _extract_field(analysis, "ANOMALIES_DETECTED")
    anomaly_list_str = _extract_field(analysis, "ANOMALY_LIST")
    severity = _extract_field(analysis, "SEVERITY")
    description = _extract_field(analysis, "DESCRIPTION")
    recommendations = _extract_field(analysis, "RECOMMENDATIONS")

    anomalies_detected = anomalies_str.lower() == "yes" if anomalies_str else False
    anomaly_list = []
    if anomaly_list_str and anomaly_list_str.lower() != "none":
        anomaly_list = [a.strip() for a in anomaly_list_str.split(",")]

    return {
        "anomalies_detected": anomalies_detected,
        "anomaly_list": anomaly_list,
        "severity": severity or "none",
        "description": description,
        "recommendations": recommendations,
        "raw_analysis": analysis,
    }


@cv_tool(
    name="compare_timepoints",
    description="""Compare embryo images across multiple timepoints.

Useful for tracking developmental progression and detecting changes.
Pass the prepared_image_uid from create_timeline_image.""",
    category=ToolCategory.VISION,
    examples=[
        ToolExample("Track development progression", {"timeline_image_input": "prepared_timeline_123456", "timepoint_labels": ["t=0", "t=1", "t=2"], "focus_aspect": "progression"}),
        ToolExample("Focus on cell divisions", {"timeline_image_input": "prepared_timeline_789012", "nuclei_counts": [4, 6, 8, 12], "focus_aspect": "divisions"}),
    ],
)
def compare_timepoints(
    timeline_image_input: str,
    timepoint_labels: Optional[List[str]] = None,
    nuclei_counts: Optional[List[int]] = None,
    focus_aspect: str = "progression",
) -> Dict[str, Any]:
    """
    Compare embryo development across timepoints

    Parameters
    ----------
    timeline_image_base64 : str
        Base64 encoded timeline montage image
    timepoint_labels : list, optional
        Labels for each timepoint in the image
    nuclei_counts : list, optional
        Nuclei counts at each timepoint
    focus_aspect : str
        What to focus on: "progression", "divisions", "morphology", "anomalies"

    Returns
    -------
    dict
        progression_summary: Overall development summary
        changes_detected: List of changes between timepoints
        division_events: Detected cell divisions
        concerns: Any concerning observations
    """
    prompt_parts = [
        "Analyze this timeline of C. elegans embryo development.",
        "The image shows the same embryo at multiple timepoints arranged in sequence.",
        "",
    ]

    if timepoint_labels:
        prompt_parts.append(f"Timepoint labels: {', '.join(timepoint_labels)}")

    if nuclei_counts:
        prompt_parts.append(f"Nuclei counts at each timepoint: {nuclei_counts}")
        prompt_parts.append("")

        # Calculate expected divisions
        for i in range(1, len(nuclei_counts)):
            if nuclei_counts[i] > nuclei_counts[i-1]:
                diff = nuclei_counts[i] - nuclei_counts[i-1]
                prompt_parts.append(
                    f"  Note: {diff} new nuclei between t{i-1} and t{i} "
                    f"(possible division event)"
                )

    prompt_parts.append("")

    if focus_aspect == "progression":
        prompt_parts.append("Focus on: Overall developmental progression")
        prompt_parts.append("- Is development proceeding normally?")
        prompt_parts.append("- Are stages progressing in expected sequence?")
    elif focus_aspect == "divisions":
        prompt_parts.append("Focus on: Cell division events")
        prompt_parts.append("- Can you identify when divisions occurred?")
        prompt_parts.append("- Are divisions symmetric/asymmetric as expected?")
    elif focus_aspect == "morphology":
        prompt_parts.append("Focus on: Morphological changes")
        prompt_parts.append("- How is the embryo shape changing?")
        prompt_parts.append("- Is elongation progressing normally?")
    elif focus_aspect == "anomalies":
        prompt_parts.append("Focus on: Anomaly detection")
        prompt_parts.append("- Are there any concerning changes?")
        prompt_parts.append("- Does anything deviate from normal development?")

    prompt_parts.append("")
    prompt_parts.append("Provide analysis in this format:")
    prompt_parts.append("PROGRESSION_SUMMARY: [overall development summary]")
    prompt_parts.append("CHANGES_DETECTED: [list of changes between timepoints]")
    prompt_parts.append("DIVISION_EVENTS: [identified divisions, or 'none visible']")
    prompt_parts.append("CONCERNS: [any concerning observations, or 'none']")
    prompt_parts.append("STAGE_PROGRESSION: [stages at each timepoint if identifiable]")

    prompt = "\n".join(prompt_parts)

    result = claude_vision_analyze(
        image_input=timeline_image_input,
        prompt=prompt,
        include_developmental_context=True,
    )

    if result.get("error"):
        return result

    analysis = result.get("analysis", "")

    return {
        "progression_summary": _extract_field(analysis, "PROGRESSION_SUMMARY"),
        "changes_detected": _extract_field(analysis, "CHANGES_DETECTED"),
        "division_events": _extract_field(analysis, "DIVISION_EVENTS"),
        "concerns": _extract_field(analysis, "CONCERNS"),
        "stage_progression": _extract_field(analysis, "STAGE_PROGRESSION"),
        "raw_analysis": analysis,
        "quantitative_context": {
            "nuclei_counts": nuclei_counts,
            "timepoint_labels": timepoint_labels,
        },
    }


# =============================================================================
# Helper Functions
# =============================================================================

def _extract_field(text: str, field_name: str) -> Optional[str]:
    """Extract a field value from formatted analysis text"""
    if not text:
        return None

    # Look for patterns like "FIELD_NAME: value" or "FIELD_NAME:\nvalue"
    import re

    # Try exact match first
    pattern = rf"{field_name}:\s*(.+?)(?=\n[A-Z_]+:|$)"
    match = re.search(pattern, text, re.IGNORECASE | re.DOTALL)

    if match:
        value = match.group(1).strip()
        # Clean up any trailing sections
        value = value.split("\n\n")[0].strip()
        return value

    return None
