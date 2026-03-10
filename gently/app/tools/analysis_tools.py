"""
Analysis and VLM Tools

Tools for analyzing embryo images using Claude Vision.
"""

from typing import Dict, Optional

from gently.harness.tools.registry import tool, ToolCategory
from gently.harness.tools.helpers import require_copilot, get_embryo_or_error


@tool(
    name="analyze_volume",
    description="Analyze an embryo volume using Claude Vision API",
    category=ToolCategory.ANALYSIS,
)
async def analyze_volume(
    embryo_id: str,
    analysis_prompt: str,
    use_recent_context: bool = False,
    timepoint: Optional[int] = None,
    context: Dict = None
) -> str:
    """Analyze embryo volume with Claude Vision"""
    copilot, err = require_copilot(context)
    if err:
        return err

    embryo, err = get_embryo_or_error(copilot, embryo_id)
    if err:
        return err

    try:
        result = await copilot._analyze_with_vision(
            embryo_id=embryo.id,
            prompt=analysis_prompt,
            use_context=use_recent_context,
            timepoint=timepoint
        )
        return result
    except Exception as e:
        return f"Error analyzing volume: {str(e)}"


@tool(
    name="get_detection_summary",
    description="Get summary of all detections across all embryos",
    category=ToolCategory.DETECTION,
)
def get_detection_summary(context: Dict) -> str:
    """Get detection summary"""
    copilot, err = require_copilot(context)
    if err:
        return err

    lines = ["Detection Summary:", ""]

    for embryo_id, embryo in copilot.experiment.embryos.items():
        if embryo.detection_results:
            lines.append(f"* {embryo_id}:")
            for det_name, results in embryo.detection_results.items():
                if results:
                    latest = results[-1]
                    lines.append(f"  - {det_name}: {latest.get('detected', False)} at t={latest.get('timepoint', '?')}")
            lines.append("")

    if len(lines) == 2:
        return "No detections recorded yet."

    return "\n".join(lines)
