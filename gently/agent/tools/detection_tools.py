"""
Embryo Detection Tools

Tools for detecting and marking embryos in microscope images.
"""

import uuid
from typing import Dict
from datetime import datetime
from pathlib import Path

from ..tool_registry import tool, ToolCategory, ToolExample
from ..tool_helpers import require_copilot
from gently.coordinates import (
    stage_to_pixel_position,
    get_um_per_pixel,
    DEFAULT_PIXEL_SIZE_UM,
    DEFAULT_OBJECTIVE_MAG,
)


@tool(
    name="detect_embryos",
    description="""Automatically detect embryos in the current field of view using brightness detection and SAM segmentation.
Use when user says "find embryos", "detect embryos", or at the start of an experiment to locate samples.
Captures a bottom camera image and identifies bright spots as potential embryos.
Opens napari after detection for immediate editing - add, delete, or move embryos as needed.
Close napari when done to confirm the embryo list. Detected embryos are added to the experiment.""",
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
    open_editor: bool = True,
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
            max_area=max_area,
            open_editor=open_editor
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
                    confidence=emb.get('confidence', 0.0),
                    uid=emb.get('uid'),  # Preserve UID from detection
                )

            if auto_calibrate and embryos:
                return f"Detected {len(embryos)} embryos. Starting calibration..."
            else:
                if open_editor:
                    return f"Detection complete: {len(embryos)} embryos confirmed after editing."
                else:
                    return f"Detected {len(embryos)} embryos. Use show_detected_embryos to visualize or edit_embryos to modify."
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
                    confidence=emb.get('confidence', 1.0),
                    uid=str(uuid.uuid4()),  # Generate new UID for manually marked embryo
                )
                added_ids.append(new_id)

            return f"Added {len(added_ids)} embryo(s): {', '.join(added_ids)}"
        else:
            return f"Marking failed: {result.get('error', 'Unknown error')}"

    except Exception as e:
        return f"Error: {str(e)}"


@tool(
    name="edit_embryos",
    description="""Open an interactive napari editor to modify embryo positions.
Allows adding new embryos, removing existing ones, and moving embryos to correct positions.
Use when user wants to adjust detection results (e.g., "edit embryos", "remove embryo_3", "adjust embryo positions", "fix detection").
Opens napari with current embryos displayed - user can add/delete/move points, then close window to apply changes.""",
    category=ToolCategory.DETECTION,
    requires_microscope=True,
    examples=[
        ToolExample("Edit embryos", {}),
        ToolExample("Let me adjust embryo positions", {}),
        ToolExample("Fix the detection", {}),
    ],
)
async def edit_embryos(
    exposure_ms: float = None,
    context: Dict = None
) -> str:
    """Interactive embryo editor - add, remove, or move embryo positions in napari"""
    copilot = context.get('copilot')
    client = context.get('client')

    if not copilot:
        return "Error: No copilot context"

    if not client:
        return "Error: Microscope not connected. Cannot edit embryos in offline mode."

    if not copilot.experiment.embryos:
        return "No embryos to edit. Run detect_embryos or manual_mark_embryos first."

    try:
        # Capture fresh image
        image = await client.capture_bottom_image(exposure_ms=exposure_ms)
        if image is None:
            return "Failed to capture image for editing."

        # Get current stage position
        stage_pos = await client.get_stage_position()

        # Get current image dimensions for pixel coordinate calculation
        image_center_x = image.shape[1] / 2
        image_center_y = image.shape[0] / 2

        # Build list of existing embryos with pixel positions
        existing_embryos = []
        um_per_pixel = get_um_per_pixel()  # Uses centralized defaults from coordinates.py

        for embryo_id, embryo_state in copilot.experiment.embryos.items():
            if embryo_state.should_skip:
                continue  # Don't show skipped embryos

            pos = embryo_state.stage_position or {}
            stage_x = pos.get('x', 0)
            stage_y = pos.get('y', 0)

            # Convert stage position to pixel position relative to current view
            dx_um = stage_x - stage_pos[0]
            dy_um = stage_y - stage_pos[1]
            pixel_x = image_center_x + dx_um / um_per_pixel
            pixel_y = image_center_y + dy_um / um_per_pixel

            existing_embryos.append({
                'embryo_id': embryo_id,
                'pixel_x': pixel_x,
                'pixel_y': pixel_y,
                'stage_x_um': stage_x,
                'stage_y_um': stage_y,
            })

        # Call the edit function on SAM server
        result = await client.edit_embryos(
            image=image,
            embryos=existing_embryos,
            stage_position=stage_pos,
            pixel_size_um=DEFAULT_PIXEL_SIZE_UM,
            objective_mag=DEFAULT_OBJECTIVE_MAG
        )

        if result.get('success'):
            edited_embryos = result.get('embryos', [])
            original_count = result.get('original_count', 0)
            added = result.get('added', 0)
            removed = result.get('removed', 0)

            # Clear existing embryos and rebuild from edit result
            # Keep track of which were removed
            old_ids = set(copilot.experiment.embryos.keys())

            # Find next available embryo ID for new ones
            max_num = 0
            for eid in old_ids:
                if eid.startswith('embryo_'):
                    try:
                        num = int(eid.replace('embryo_', ''))
                        max_num = max(max_num, num)
                    except ValueError:
                        pass
            next_num = max_num + 1

            # Process edited embryos
            new_ids = set()
            for emb in edited_embryos:
                emb_id = emb.get('embryo_id', f'embryo_{next_num}')

                # If it's a new embryo (source=manual_edit), assign new ID
                if emb.get('source') == 'manual_edit' or emb_id not in old_ids:
                    emb_id = f'embryo_{next_num}'
                    next_num += 1

                position = {
                    'x': emb.get('stage_x_um', 0),
                    'y': emb.get('stage_y_um', 0)
                }

                # Update or add embryo
                if emb_id in copilot.experiment.embryos:
                    # Update existing
                    copilot.experiment.embryos[emb_id].stage_position = position
                else:
                    # Add new
                    copilot.experiment.add_embryo(
                        embryo_id=emb_id,
                        position=position,
                        confidence=emb.get('confidence', 1.0),
                        uid=str(uuid.uuid4()),  # Generate new UID for embryo added via editor
                    )

                new_ids.add(emb_id)

            # Remove embryos that were deleted in editor
            removed_ids = old_ids - new_ids
            for rid in removed_ids:
                if rid in copilot.experiment.embryos:
                    del copilot.experiment.embryos[rid]

            # Build summary
            summary = f"Edit complete: {len(new_ids)} embryos"
            if added > 0:
                summary += f", +{added} added"
            if len(removed_ids) > 0:
                summary += f", -{len(removed_ids)} removed ({', '.join(removed_ids)})"

            return summary
        else:
            return f"Edit failed: {result.get('error', 'Unknown error')}"

    except Exception as e:
        import traceback
        traceback.print_exc()
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
        um_per_pixel = get_um_per_pixel()  # Uses centralized defaults from coordinates.py

        image_center_x = image.shape[1] / 2
        image_center_y = image.shape[0] / 2

        embryos = []
        for embryo_id, embryo_state in copilot.experiment.embryos.items():
            pos = embryo_state.stage_position or {}

            stage_x = pos.get('x', current_stage[0])
            stage_y = pos.get('y', current_stage[1])

            # Convert stage to pixel using centralized function
            pixel_x, pixel_y = stage_to_pixel_position(
                stage_x=stage_x,
                stage_y=stage_y,
                current_stage_x=current_stage[0],
                current_stage_y=current_stage[1],
                image_center_x=image_center_x,
                image_center_y=image_center_y,
                um_per_pixel=um_per_pixel
            )

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
        save_path = f"detection_results/detected_embryos_{timestamp}.jpg"
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
