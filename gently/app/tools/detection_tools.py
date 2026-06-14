"""
Embryo Detection Tools

Tools for detecting and marking embryos in microscope images.

All marking / editing UI flows now go through the web map view
(``gently.ui.web.embryo_marker.mark_embryos_web``). napari has been
retired; the map view is the single spatial GUI.
"""

import uuid
from datetime import datetime
from pathlib import Path

import numpy as np

from gently.core.coordinates import (
    DEFAULT_OBJECTIVE_MAG,
    DEFAULT_PIXEL_SIZE_UM,
    get_um_per_pixel,
    pixel_to_stage_position,
    stage_to_pixel_position,
)
from gently.harness.tools.helpers import ctx_get
from gently.harness.tools.registry import ToolCategory, ToolExample, tool


async def _route_to_map_view(
    agent,
    image: np.ndarray,
    initial_markers: list[dict],
    stage_position: tuple[float, float],
    default_role: str = "test",
    pixel_size_um: float = DEFAULT_PIXEL_SIZE_UM,
    timeout: float | None = None,
) -> tuple[list[dict] | None, str | None]:
    """Hand off image + markers to the web map view; await user-edited result.

    Returns ``(marked, None)`` on success or ``(None, error_message)`` if the
    map view isn't available.
    """
    if getattr(agent, "viz_server", None) is None:
        return None, (
            "Map view requires the web visualization server. Start it with start_viz_server first."
        )

    from gently.ui.web.embryo_marker import mark_embryos_web

    marked = await mark_embryos_web(
        viz_server=agent.viz_server,
        image=image,
        initial_stage_position=stage_position,
        pixel_size_um=pixel_size_um,
        initial_markers=initial_markers,
        default_role=default_role,
        timeout=timeout,
    )
    return marked, None


def _next_embryo_number(experiment) -> int:
    """Next available embryo number not currently in use."""
    max_num = 0
    for eid in experiment.embryos.keys():
        if eid.startswith("embryo_"):
            try:
                max_num = max(max_num, int(eid.replace("embryo_", "")))
            except ValueError:
                pass
    return max_num + 1


def _stage_from_pixel(
    pixel_x: float,
    pixel_y: float,
    image_shape: tuple[int, int],
    current_stage: tuple[float, float],
    pixel_size_um: float = DEFAULT_PIXEL_SIZE_UM,
    objective_mag: float = DEFAULT_OBJECTIVE_MAG,
) -> tuple[float, float]:
    """Convert a pixel position in the bottom-cam image to a stage XY."""
    h, w = image_shape[:2]
    um_per_px = get_um_per_pixel(pixel_size_um, objective_mag)
    return pixel_to_stage_position(
        pixel_x,
        pixel_y,
        w / 2,
        h / 2,
        current_stage[0],
        current_stage[1],
        um_per_px,
    )


@tool(
    name="detect_embryos",
    description="""Automatically detect embryos in the current field of view using brightness
detection + SAM segmentation, then hand off to the web map view for editing and role
assignment.

Use when user says "find embryos", "detect embryos", or at the start of an experiment to
locate samples. Captures a bottom camera image, runs SAM detection, and opens the web map
view with SAM markers pre-placed. User adds/removes markers, cycles each marker's role
(Test / Calibration / unassigned), and presses Done. The confirmed embryos are registered
with their roles in the experiment.""",
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
    exposure_ms: float | None = None,
    brightness_percentile: float = 99.0,
    min_area: int = 5000,
    max_area: int = 150000,
    default_role: str = "test",
    context: dict | None = None,
) -> str:
    """Detect embryos via SAM + edit/assign roles in the web map view."""
    agent = ctx_get(context, "agent")
    client = ctx_get(context, "client")

    if not agent:
        return "Error: No agent context"
    if not client:
        return "Error: Microscope not connected. Cannot detect embryos in offline mode."
    if not client.has_sam:
        return (
            "Error: SAM server not connected."
            " Embryo detection requires the SAM segmentation server."
        )

    try:
        result = await client.detect_embryos(
            min_confidence=min_confidence,
            use_claude_review=use_claude_review,
            exposure_ms=exposure_ms,
            brightness_percentile=brightness_percentile,
            min_area=min_area,
            max_area=max_area,
        )

        if not result.get("success"):
            return f"Detection failed: {result.get('error', 'Unknown error')}"

        sam_embryos = result.get("embryos", [])
        image = result.get("image")
        stage_pos = tuple(result.get("stage_position", [0.0, 0.0]))

        if image is None:
            return (
                f"Detection ran but no image was returned for the map view"
                f" (got {len(sam_embryos)} SAM detections)."
            )

        # Hand off SAM detections as editable initial markers in the map view.
        initial_markers = [
            {
                "pixel_x": emb.get("pixel_x"),
                "pixel_y": emb.get("pixel_y"),
                "role": default_role,
                "source": "sam",
                "confidence": emb.get("confidence", 0.0),
                "embryo_id": emb.get("embryo_id"),
            }
            for emb in sam_embryos
            if emb.get("pixel_x") is not None and emb.get("pixel_y") is not None
        ]

        marked, err = await _route_to_map_view(
            agent=agent,
            image=image,
            initial_markers=initial_markers,
            stage_position=stage_pos,
            default_role=default_role,
        )
        if err:
            return err
        marked = marked or []

        # Register confirmed embryos. Use existing IDs from SAM where possible;
        # assign fresh ones for manual additions.
        next_num = _next_embryo_number(agent.experiment)
        added = []
        for m in marked:
            existing_id = m.get("embryo_id")
            if existing_id and existing_id not in agent.experiment.embryos:
                emb_id = existing_id
            elif existing_id and existing_id in agent.experiment.embryos:
                emb_id = existing_id  # update path
            else:
                emb_id = f"embryo_{next_num}"
                next_num += 1

            stage_x, stage_y = _stage_from_pixel(
                m["pixel_x"],
                m["pixel_y"],
                image.shape,
                stage_pos,
            )
            position = {"x": stage_x, "y": stage_y}

            if emb_id in agent.experiment.embryos:
                # Update existing in-place (preserve other fields)
                emb = agent.experiment.embryos[emb_id]
                emb.stage_position = position
                emb.role = m.get("role", default_role)
            else:
                agent.experiment.add_embryo(
                    embryo_id=emb_id,
                    position=position,
                    confidence=m.get("confidence") or 0.0,
                    uid=str(uuid.uuid4()),
                    role=m.get("role", default_role),
                )
            added.append((emb_id, m.get("role", default_role)))

        # OPERATOR_MARKED_EMBRYOS — operator confirmed via the web canvas.
        # This is the intent signal eval/shadow listeners hook for ReactiveCandidate.
        if added:
            bus = getattr(agent, "_event_bus", None)
            if bus is not None:
                from gently.core.event_bus import EventType

                try:
                    bus.publish(
                        event_type=EventType.OPERATOR_MARKED_EMBRYOS,
                        data={
                            "embryo_ids": [eid for eid, _ in added],
                            "count": len(added),
                            "stage_origin": list(stage_pos),
                            "pre_edit_count": len(sam_embryos),
                        },
                        source="detect_embryos:web-editor",
                    )
                except Exception:
                    pass

        role_counts: dict[str, int] = {}
        for _, r in added:
            role_counts[r] = role_counts.get(r, 0) + 1
        role_summary = ", ".join(f"{n} {r}" for r, n in sorted(role_counts.items()))

        if auto_calibrate and added:
            return (
                f"Detected & registered {len(added)} embryos ({role_summary})."
                " Starting calibration..."
            )
        return f"Detection complete: {len(added)} embryo(s) ({role_summary})."

    except Exception as e:
        import traceback

        traceback.print_exc()
        return f"Error detecting embryos: {str(e)}"


@tool(
    name="manual_mark_embryos",
    description="""Capture a bottom-camera image and open the web map view for manual embryo
marking. User clicks to add markers and cycles each marker's role (Test / Calibration /
unassigned).

Use when automatic detection missed embryos, or when the user wants to add embryos manually
(e.g., "let me mark embryos", "I'll click on them"). Newly marked embryos are registered with
the role assigned in the map view; existing embryos remain untouched.""",
    category=ToolCategory.DETECTION,
    requires_microscope=True,
    examples=[
        ToolExample("Let me mark embryos manually", {}),
        ToolExample("I want to click on embryos", {}),
    ],
)
async def manual_mark_embryos(
    exposure_ms: float | None = None,
    default_role: str = "test",
    context: dict | None = None,
) -> str:
    """Manual marking via the web map view."""
    agent = ctx_get(context, "agent")
    client = ctx_get(context, "client")

    if not agent:
        return "Error: No agent context"
    if not client:
        return "Error: Microscope not connected. Cannot mark embryos in offline mode."

    try:
        snap = await client.capture_for_marking(exposure_ms=exposure_ms)
        if not snap.get("success"):
            return f"Failed to capture image for marking: {snap.get('error', 'unknown')}"

        image = snap["image"]
        stage_pos = tuple(snap["stage_position"])

        # Pre-place existing (non-skipped) embryos as reference markers so
        # the user knows what's already detected — they can leave them
        # alone, or remove/relabel as needed.
        initial_markers = []
        um_per_px = get_um_per_pixel()
        h, w = image.shape[:2]
        for embryo_id, emb in agent.experiment.embryos.items():
            if emb.should_skip:
                continue
            pos = emb.stage_position or {}
            sx, sy = pos.get("x", 0), pos.get("y", 0)
            px, py = stage_to_pixel_position(
                stage_x=sx,
                stage_y=sy,
                current_stage_x=stage_pos[0],
                current_stage_y=stage_pos[1],
                image_center_x=w / 2,
                image_center_y=h / 2,
                um_per_pixel=um_per_px,
            )
            initial_markers.append(
                {
                    "pixel_x": px,
                    "pixel_y": py,
                    "role": emb.role,
                    "source": "existing",
                    "embryo_id": embryo_id,
                    "confidence": emb.detection_confidence,
                }
            )

        marked, err = await _route_to_map_view(
            agent=agent,
            image=image,
            initial_markers=initial_markers,
            stage_position=stage_pos,
            default_role=default_role,
        )
        if err:
            return err
        marked = marked or []

        if not marked:
            return "No markers placed."

        # Reconcile: any marker whose embryo_id matches an existing one
        # updates that embryo (position + role); markers without a matching
        # embryo_id are NEW embryos that get fresh IDs.
        existing_ids = set(agent.experiment.embryos.keys())
        next_num = _next_embryo_number(agent.experiment)
        added_ids, updated_ids = [], []
        for m in marked:
            stage_x, stage_y = _stage_from_pixel(
                m["pixel_x"],
                m["pixel_y"],
                image.shape,
                stage_pos,
            )
            pos = {"x": stage_x, "y": stage_y}
            existing_id = m.get("embryo_id")

            if existing_id and existing_id in agent.experiment.embryos:
                emb = agent.experiment.embryos[existing_id]
                emb.stage_position = pos
                emb.role = m.get("role", default_role)
                updated_ids.append(existing_id)
            else:
                new_id = f"embryo_{next_num}"
                next_num += 1
                agent.experiment.add_embryo(
                    embryo_id=new_id,
                    position=pos,
                    confidence=m.get("confidence") or 1.0,
                    uid=str(uuid.uuid4()),
                    role=m.get("role", default_role),
                )
                added_ids.append(new_id)

        # Removals: any existing embryo NOT mentioned in marked is dropped.
        seen_ids = {m.get("embryo_id") for m in marked if m.get("embryo_id")}
        removed_ids = [eid for eid in existing_ids if eid not in seen_ids and eid not in added_ids]
        for rid in removed_ids:
            agent.experiment.embryos.pop(rid, None)

        parts = []
        if added_ids:
            parts.append(f"+{len(added_ids)} added: {', '.join(added_ids)}")
        if updated_ids:
            parts.append(f"{len(updated_ids)} updated")
        if removed_ids:
            parts.append(f"-{len(removed_ids)} removed: {', '.join(removed_ids)}")
        return "Marking complete. " + ("; ".join(parts) if parts else "no changes.")

    except Exception as e:
        import traceback

        traceback.print_exc()
        return f"Error: {str(e)}"


@tool(
    name="edit_embryos",
    description="""Capture a fresh bottom-camera image and open the web map view to edit
current embryos — add, remove, move, or re-label Test/Calibration. Same surface as
manual_mark_embryos; this tool exists to match user intent when they say "edit" rather than
"mark".

Use when user wants to adjust existing detection results (e.g., "edit embryos",
"remove embryo_3", "swap roles", "fix detection").""",
    category=ToolCategory.DETECTION,
    requires_microscope=True,
    examples=[
        ToolExample("Edit embryos", {}),
        ToolExample("Let me adjust embryo positions", {}),
        ToolExample("Fix the detection", {}),
    ],
)
async def edit_embryos(
    exposure_ms: float | None = None,
    default_role: str = "test",
    context: dict | None = None,
) -> str:
    """Edit existing embryos via the web map view."""
    agent = ctx_get(context, "agent")
    if not agent:
        return "Error: No agent context"
    if not agent.experiment.embryos:
        return "No embryos to edit. Run detect_embryos or manual_mark_embryos first."

    # Same flow as manual_mark_embryos: pre-populate with existing markers,
    # let user edit, reconcile. notify_embryos_changed is fired by
    # manual_mark_embryos / add_embryo internally.
    return await manual_mark_embryos(
        exposure_ms=exposure_ms,
        default_role=default_role,
        context=context,
    )


@tool(
    name="show_detected_embryos",
    description="""Capture a fresh image and display all tracked embryos with labeled bounding
boxes. Shows embryo IDs at their positions.
Use when user wants to see where embryos are visually (e.g., "show me the embryos",
"display embryo positions"). Captures a new bottom camera image and overlays all active
(non-skipped) embryo positions. Image is saved to detection_results/.""",
    category=ToolCategory.DETECTION,
    requires_microscope=True,
    examples=[
        ToolExample("Show me the embryos", {}),
        ToolExample("Display embryo positions", {}),
    ],
)
async def show_detected_embryos(save_to_file: bool = True, context: dict | None = None) -> str:
    """Show detected embryos visualization using experiment.embryos as source of truth"""
    agent = ctx_get(context, "agent")
    client = ctx_get(context, "client")

    if not agent:
        return "Error: No agent context"

    if not client:
        return "Error: Microscope not connected. Cannot show embryos in offline mode."

    if not agent.experiment.embryos:
        return "No embryos in experiment. Run detect_embryos first."

    try:
        snap = await client.capture_bottom_image()
        image = snap["image"]
        if image is None or image.shape == (100, 100):
            return "Failed to capture image for visualization."

        current_stage = await client.get_stage_position()

        # Archive the bottom camera image with metadata
        if snap.get("image_path") and agent.store and agent.session_id:
            try:
                from gently.harness.tools.helpers import build_snapshot_metadata

                meta = build_snapshot_metadata(current_stage, image.shape, agent.experiment)
                agent.store.register_snapshot(
                    agent.session_id, "bottom_camera", snap["image_path"], metadata=meta
                )
            except Exception:
                pass

        # Calculate pixel positions from experiment embryo positions
        um_per_pixel = get_um_per_pixel()  # Uses centralized defaults from coordinates.py

        image_center_x = image.shape[1] / 2
        image_center_y = image.shape[0] / 2

        embryos = []
        for embryo_id, embryo_state in agent.experiment.embryos.items():
            pos = embryo_state.stage_position or {}

            stage_x = pos.get("x", current_stage[0])
            stage_y = pos.get("y", current_stage[1])

            # Convert stage to pixel using centralized function
            pixel_x, pixel_y = stage_to_pixel_position(
                stage_x=stage_x,
                stage_y=stage_y,
                current_stage_x=current_stage[0],
                current_stage_y=current_stage[1],
                image_center_x=image_center_x,
                image_center_y=image_center_y,
                um_per_pixel=um_per_pixel,
            )

            embryos.append(
                {
                    "embryo_id": embryo_id,
                    "pixel_x": pixel_x,
                    "pixel_y": pixel_y,
                    "stage_x_um": stage_x,
                    "stage_y_um": stage_y,
                    "confidence": embryo_state.detection_confidence,
                }
            )

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
            show=True,
        )

        if view_result.get("success"):
            embryo_ids = [e.get("embryo_id", "?") for e in embryos]
            return f"Showing {len(embryos)} embryos: {', '.join(embryo_ids)}\nSaved to: {save_path}"
        elif view_result.get("error"):
            return f"Display error: {view_result.get('error')}"
        else:
            return f"Visualization complete. Check {save_path}"

    except Exception as e:
        return f"Error showing detections: {str(e)}"
