"""
Embryo Detection Tools

Tools for detecting and marking embryos in microscope images.
"""

from typing import Dict
from datetime import datetime
from pathlib import Path

from ..tool_registry import tool, ToolCategory, ToolExample
from ..tool_helpers import require_copilot


@tool(
    name="detect_embryos",
    description="""Automatically detect embryos in the current field of view using brightness detection and SAM segmentation.
Use when user says "find embryos", "detect embryos", or at the start of an experiment to locate samples.
Captures a bottom camera image and identifies bright spots as potential embryos. Returns embryo IDs and positions.
Detected embryos are automatically added to the experiment. Use show_detected_embryos to visualize results.""",
    category=ToolCategory.DETECTION,
    requires_microscope=True,
    examples=[
        ToolExample("Find all embryos", {}),
        ToolExample("Detect embryos automatically", {}),
    ],
)
async def detect_embryos(
    auto_calibrate: bool = False,
    min_confidence: float = 0.7,
    use_claude_review: bool = False,
    exposure_ms: float = None,
    brightness_percentile: float = 99.0,
    min_area: int = 5000,
    max_area: int = 150000,
    context: Dict = None
) -> str:
    """Detect embryos automatically"""
    copilot = context.get('copilot')
    client = context.get('client')

    if not copilot:
        return "Error: No copilot context"

    if not client:
        return "Error: Microscope not connected. Cannot detect embryos in offline mode."

    if not client.has_sam:
        return "Error: SAM server not connected. Embryo detection requires the SAM segmentation server."

    try:
        result = await client.detect_embryos(
            min_confidence=min_confidence,
            use_claude_review=use_claude_review,
            exposure_ms=exposure_ms,
            brightness_percentile=brightness_percentile,
            min_area=min_area,
            max_area=max_area
        )

        if result.get('success'):
            embryos = result.get('embryos', [])

            # Add to experiment
            for emb in embryos:
                position = {
                    'x': emb.get('stage_x_um', emb.get('stage_x', 0)),
                    'y': emb.get('stage_y_um', emb.get('stage_y', 0))
                }
                copilot.experiment.add_embryo(
                    embryo_id=emb['embryo_id'],
                    position=position,
                    confidence=emb.get('confidence', 0.0)
                )

            if auto_calibrate and embryos:
                return f"Detected {len(embryos)} embryos. Starting calibration..."
            else:
                return f"Detected {len(embryos)} embryos\nUse show_detected_embryos to visualize."
        else:
            return f"Detection failed: {result.get('error', 'Unknown error')}"

    except Exception as e:
        return f"Error detecting embryos: {str(e)}"


@tool(
    name="manual_mark_embryos",
    description="""Open an interactive window to manually mark embryos by clicking on them. Existing embryos are shown in green for reference.
Use when automatic detection missed embryos, or user wants to add embryos manually (e.g., "let me mark embryos", "I'll click on them").
Opens a matplotlib window - user clicks to mark positions, then closes the window. New embryos get unique IDs automatically.""",
    category=ToolCategory.DETECTION,
    requires_microscope=True,
    examples=[
        ToolExample("Let me mark embryos manually", {}),
        ToolExample("I want to click on embryos", {}),
    ],
)
async def manual_mark_embryos(
    exposure_ms: float = None,
    context: Dict = None
) -> str:
    """Manual embryo marking - shows existing embryos, adds new ones with unique IDs"""
    copilot = context.get('copilot')
    client = context.get('client')

    if not copilot:
        return "Error: No copilot context"

    if not client:
        return "Error: Microscope not connected. Cannot mark embryos in offline mode."

    try:
        # Build list of existing embryos with their stage positions
        existing_embryos = []
        for embryo_id, embryo_state in copilot.experiment.embryos.items():
            pos = embryo_state.stage_position or {}
            existing_embryos.append({
                'embryo_id': embryo_id,
                'stage_x': pos.get('x', 0),
                'stage_y': pos.get('y', 0),
            })

        result = await client.manual_mark_embryos(
            exposure_ms=exposure_ms,
            existing_embryos=existing_embryos if existing_embryos else None
        )

        if result.get('success'):
            embryos = result.get('embryos', [])

            if not embryos:
                return "No embryos marked. Close the window after clicking on embryo centers."

            # Find next available embryo ID
            existing_ids = set(copilot.experiment.embryos.keys())
            max_num = 0
            for eid in existing_ids:
                if eid.startswith('embryo_'):
                    try:
                        num = int(eid.replace('embryo_', ''))
                        max_num = max(max_num, num)
                    except ValueError:
                        pass
            next_num = max_num + 1

            # Assign new unique IDs and add to experiment
            added_ids = []
            for emb in embryos:
                new_id = f'embryo_{next_num}'
                next_num += 1

                position = {
                    'x': emb.get('stage_x_um', emb.get('stage_x', 0)),
                    'y': emb.get('stage_y_um', emb.get('stage_y', 0))
                }

                emb['embryo_id'] = new_id

                copilot.experiment.add_embryo(
                    embryo_id=new_id,
                    position=position,
                    confidence=emb.get('confidence', 1.0)
                )
                added_ids.append(new_id)

            return f"Added {len(added_ids)} embryo(s): {', '.join(added_ids)}"
        else:
            return f"Marking failed: {result.get('error', 'Unknown error')}"

    except Exception as e:
        return f"Error: {str(e)}"


@tool(
    name="show_detected_embryos",
    description="""Capture a fresh image and display all tracked embryos with labeled bounding boxes. Shows embryo IDs at their positions.
Use when user wants to see where embryos are visually (e.g., "show me the embryos", "display embryo positions").
Captures a new bottom camera image and overlays all active (non-skipped) embryo positions. Image is saved to detection_results/.""",
    category=ToolCategory.DETECTION,
    requires_microscope=True,
    examples=[
        ToolExample("Show me the embryos", {}),
        ToolExample("Display embryo positions", {}),
    ],
)
async def show_detected_embryos(
    save_to_file: bool = True,
    context: Dict = None
) -> str:
    """Show detected embryos visualization using experiment.embryos as source of truth"""
    copilot = context.get('copilot')
    client = context.get('client')

    if not copilot:
        return "Error: No copilot context"

    if not client:
        return "Error: Microscope not connected. Cannot show embryos in offline mode."

    if not copilot.experiment.embryos:
        return "No embryos in experiment. Run detect_embryos first."

    try:
        image = await client.capture_bottom_image()
        if image is None or image.shape == (100, 100):
            return "Failed to capture image for visualization."

        current_stage = await client.get_stage_position()

        # Calculate pixel positions from experiment embryo positions
        pixel_size_um = 6.5
        objective_mag = 4.0
        um_per_pixel = pixel_size_um / objective_mag

        image_center_x = image.shape[1] / 2
        image_center_y = image.shape[0] / 2

        embryos = []
        for embryo_id, embryo_state in copilot.experiment.embryos.items():
            pos = embryo_state.stage_position or {}

            stage_x = pos.get('x', current_stage[0])
            stage_y = pos.get('y', current_stage[1])

            dx_stage = stage_x - current_stage[0]
            dy_stage = stage_y - current_stage[1]

            pixel_x = image_center_x + dx_stage / um_per_pixel
            pixel_y = image_center_y + dy_stage / um_per_pixel

            embryos.append({
                'embryo_id': embryo_id,
                'pixel_x': pixel_x,
                'pixel_y': pixel_y,
                'stage_x_um': stage_x,
                'stage_y_um': stage_y,
                'confidence': embryo_state.detection_confidence,
            })

        if not embryos:
            return "No embryos to display."

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_path = f"detection_results/detected_embryos_{timestamp}.png"
        Path("detection_results").mkdir(exist_ok=True)

        view_result = await client.view_embryos(
            image=image,
            embryos=embryos,
            title=f"Embryos ({len(embryos)})",
            save_path=save_path,
            show=True
        )

        if view_result.get('success'):
            embryo_ids = [e.get('embryo_id', '?') for e in embryos]
            return f"Showing {len(embryos)} embryos: {', '.join(embryo_ids)}\nSaved to: {save_path}"
        elif view_result.get('error'):
            return f"Display error: {view_result.get('error')}"
        else:
            return f"Visualization complete. Check {save_path}"

    except Exception as e:
        return f"Error showing detections: {str(e)}"
