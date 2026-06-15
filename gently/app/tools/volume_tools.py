"""
Volume Viewing Tools

Tools for viewing and listing acquired lightsheet volumes.
"""

import logging

from gently.core.coordinates import get_um_per_pixel, stage_to_pixel_position
from gently.harness.tools.helpers import ctx_get, require_agent, require_microscope
from gently.harness.tools.registry import ToolCategory, ToolExample, tool

logger = logging.getLogger(__name__)


@tool(
    name="view_image",
    description="""Capture and display the current bottom camera widefield image. Shows what's
visible at the current stage position.
Use when user says "show me the view", "take a picture", "what does it look like?", or to
check sample positioning. This is the widefield/brightfield camera looking up at the sample
- good for seeing embryo outlines and overall positioning.
Image is automatically saved to camera_captures/ folder.""",
    category=ToolCategory.HARDWARE,
    requires_microscope=True,
    examples=[
        ToolExample("Show me the current view", {}),
        ToolExample("What does the sample look like?", {}),
    ],
)
async def view_image(
    title: str = "Bottom Camera Image",
    exposure_ms: float | None = None,
    show: bool = True,
    show_embryos: bool = True,
    context: dict | None = None,
) -> str:
    """Capture and display bottom camera image with embryo annotations"""
    client, err = require_microscope(context)
    if err:
        return err
    agent = ctx_get(context, "agent")

    try:
        snap = await client.capture_bottom_image(exposure_ms=exposure_ms)
        image = snap["image"]

        if image is None or image.shape == (100, 100):
            return "Failed to capture image from bottom camera"

        # Get current stage position for coordinate conversion
        stage_pos = await client.get_stage_position()

        # Archive the bottom camera image with metadata
        if snap.get("image_path") and agent and agent.store and agent.session_id:
            try:
                from gently.harness.tools.helpers import build_snapshot_metadata

                meta = build_snapshot_metadata(
                    stage_pos, image.shape, agent.experiment if agent else None
                )
                agent.store.register_snapshot(
                    agent.session_id, "bottom_camera", snap["image_path"], metadata=meta
                )
            except Exception:
                pass

        # Prepare embryo annotations if requested
        embryo_annotations = []
        if show_embryos and agent and agent.experiment.embryos:
            um_per_pixel = get_um_per_pixel()  # Uses centralized defaults from coordinates.py
            image_center_x = image.shape[1] / 2
            image_center_y = image.shape[0] / 2

            for embryo_id, embryo in agent.experiment.embryos.items():
                if embryo.stage_position:
                    # Convert stage position to pixel position using centralized function
                    emb_x = embryo.stage_position.get("x", 0)
                    emb_y = embryo.stage_position.get("y", 0)
                    pixel_x, pixel_y = stage_to_pixel_position(
                        stage_x=emb_x,
                        stage_y=emb_y,
                        current_stage_x=stage_pos[0],
                        current_stage_y=stage_pos[1],
                        image_center_x=image_center_x,
                        image_center_y=image_center_y,
                        um_per_pixel=um_per_pixel,
                    )

                    embryo_annotations.append(
                        {
                            "embryo_id": embryo_id,
                            "pixel_x": pixel_x,
                            "pixel_y": pixel_y,
                            "label": embryo.user_label or embryo_id,
                        }
                    )

        if show:
            from datetime import datetime
            from pathlib import Path

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            save_path = f"camera_captures/bottom_camera_{timestamp}.jpg"
            Path("camera_captures").mkdir(exist_ok=True)

            await client.view_image(
                image=image,
                title=title,
                save_path=save_path,
                show=True,
                embryo_annotations=embryo_annotations if embryo_annotations else None,
            )
            num_visible = len(
                [
                    a
                    for a in embryo_annotations
                    if 0 <= a["pixel_x"] < image.shape[1] and 0 <= a["pixel_y"] < image.shape[0]
                ]
            )
            annotation_msg = (
                f"\nShowing {num_visible} embryo(s) in view" if embryo_annotations else ""
            )
            return (
                f"Captured bottom camera image ({image.shape[0]}x{image.shape[1]})"
                f"\nSaved to: {save_path}{annotation_msg}"
            )
        else:
            return f"Captured bottom camera image ({image.shape[0]}x{image.shape[1]})"

    except Exception as e:
        return f"Error capturing image: {str(e)}"


@tool(
    name="view_volume",
    description="""Open an acquired volume in the in-browser 3D viewer.
Opens by embryo ID \u2014 the latest volume, or a specific timepoint. The volume
appears in the web UI's volume viewer (interactive 3D raymarcher + projections)
for everyone watching the session; nothing pops up on the instrument desktop.
Use when the user says "open volume", "view volume", "show the 3D data", or
"look at timepoint N of embryo X".""",
    category=ToolCategory.ANALYSIS,
    requires_microscope=False,
    examples=[
        ToolExample("Open latest volume for embryo 2", {"embryo_id": "embryo_2"}),
        ToolExample("Open specific timepoint", {"embryo_id": "embryo_2", "timepoint": 5}),
    ],
)
async def view_volume(
    embryo_id: str | None = None,
    timepoint: int | None = None,
    file_path: str | None = None,
    context: dict | None = None,
) -> str:
    """Open a volume in the browser-based viewer (no blocking desktop window)."""
    from pathlib import Path

    agent, err = require_agent(context)
    if err:
        return err

    session_id = agent.session_id

    # file_path is legacy. In-browser viewing is addressed by embryo + timepoint,
    # so map a FileStore path (embryos/{embryo_id}/volumes/t{NNNN}.tif) back to
    # those when possible.
    if file_path and not embryo_id:
        p = Path(file_path)
        if not p.exists():
            return f"Error: File not found: {file_path}"
        stem = p.stem  # e.g. "t0005"
        try:
            if stem.startswith("t"):
                timepoint = int(stem[1:])
            # .../embryos/{embryo_id}/volumes/t{NNNN}.tif \u2192 embryo dir is parent of "volumes"
            embryo_id = p.parent.parent.name
        except (ValueError, IndexError):
            pass
        if not embryo_id or timepoint is None:
            return (
                "Volume viewing is now in-browser and addressed by embryo + "
                "timepoint. Please specify embryo_id (and optionally timepoint) "
                "rather than a raw file path."
            )

    if not embryo_id:
        return "Error: Specify embryo_id (and optionally timepoint)."

    # Resolve the timepoint (specific or latest) and confirm the volume exists.
    if timepoint is not None:
        volume_path = agent.store.get_volume_path(session_id, embryo_id, timepoint)
        if not volume_path or not Path(volume_path).exists():
            volumes = agent.store.list_volumes(session_id, embryo_id)
            available = sorted(v["timepoint"] for v in volumes)
            if not available:
                return f"No volumes found for {embryo_id} in session {session_id}"
            return f"Timepoint {timepoint} not found for {embryo_id}. Available: {available}"
    else:
        volumes = agent.store.list_volumes(session_id, embryo_id)
        if not volumes:
            return f"No volumes found for {embryo_id} in session {session_id}"
        timepoint = max(v["timepoint"] for v in volumes)

    # Drive the in-browser viewer \u2014 no blocking Qt/desktop window.
    viz = getattr(agent, "viz_server", None)
    if viz is None:
        return (
            f"Resolved {embryo_id} t{timepoint:04d}, but the web UI isn't running, "
            f"so there's nowhere to display it. Start the web UI and try again."
        )

    try:
        n_clients = await viz.open_volume_in_browser(embryo_id, timepoint)
    except Exception as e:
        logger.exception("open_volume_in_browser failed")
        return f"Error opening volume in the web viewer: {e}"

    url = f"http://localhost:{getattr(viz, 'port', 8080)}/"
    if n_clients <= 0:
        return (
            f"Resolved {embryo_id} t{timepoint:04d}, but no browser is connected. "
            f"Open {url} and select that embryo/timepoint to view it."
        )
    return (
        f"\u2713 Opening {embryo_id} t{timepoint:04d} in the web volume viewer "
        f"({n_clients} view(s) connected) \u2014 {url}"
    )


@tool(
    name="list_volumes",
    description="""List available volumes for an embryo or all embryos.
Shows volume files with timepoints and file sizes. Scans the storage directory for all
volumes (not just recent ones in memory).
Use to see what data is available before viewing.""",
    category=ToolCategory.ANALYSIS,
    requires_microscope=False,
    examples=[
        ToolExample("List volumes for embryo 2", {"embryo_id": "embryo_2"}),
        ToolExample("List all volumes", {}),
    ],
)
async def list_volumes(embryo_id: str | None = None, context: dict | None = None) -> str:
    """List available volumes"""
    agent, err = require_agent(context)
    if err:
        return err

    session_id = agent.session_id
    lines = []

    # Get volumes from FileStore
    all_volumes_list = agent.store.list_volumes(session_id, embryo_id)

    # Group by embryo_id
    all_volumes: dict[str, list[dict]] = {}  # embryo_id -> list of volume records
    for vol in all_volumes_list:
        eid = vol["embryo_id"]
        if eid not in all_volumes:
            all_volumes[eid] = []
        all_volumes[eid].append(vol)

    # Sort by timepoint
    for eid in all_volumes:
        all_volumes[eid].sort(key=lambda x: x["timepoint"])

    if embryo_id:
        # List volumes for specific embryo
        if embryo_id not in all_volumes:
            return f"No volumes found for {embryo_id} in session {session_id}"

        volumes = all_volumes[embryo_id]
        lines.append(f"Volumes for {embryo_id}: {len(volumes)} file(s)")
        lines.append(f"Session: {session_id}")
        lines.append("")

        for vol in volumes:
            tp = vol["timepoint"]
            path = agent.store.get_volume_path(session_id, embryo_id, tp)
            if path and path.exists():
                size_mb = path.stat().st_size / (1024 * 1024)
                lines.append(f"  t{tp:04d}: {path.name} ({size_mb:.1f} MB)")
            else:
                lines.append(f"  t{tp:04d}: (file missing)")

    else:
        # List volumes for all embryos
        if not all_volumes:
            return f"No volumes found in session {session_id}"

        total_files = sum(len(v) for v in all_volumes.values())
        lines.append(
            f"Available volumes: {total_files} file(s) across {len(all_volumes)} embryo(s)"
        )
        lines.append(f"Session: {session_id}")

        for eid in sorted(all_volumes.keys()):
            volumes = all_volumes[eid]
            timepoints = [v["timepoint"] for v in volumes]
            tp_range = (
                f"t{min(timepoints):04d}-t{max(timepoints):04d}"
                if len(timepoints) > 1
                else f"t{timepoints[0]:04d}"
            )

            # Calculate total size
            total_size = 0
            for vol in volumes:
                path = agent.store.get_volume_path(session_id, eid, vol["timepoint"])
                if path and path.exists():
                    total_size += path.stat().st_size / (1024 * 1024)
            lines.append(
                f"\n{eid}: {len(volumes)} volume(s) [{tp_range}] ({total_size:.1f} MB total)"
            )

            # Show last few timepoints
            for vol in volumes[-3:]:
                tp = vol["timepoint"]
                path = agent.store.get_volume_path(session_id, eid, tp)
                if path and path.exists():
                    size_mb = path.stat().st_size / (1024 * 1024)
                    lines.append(f"    t{tp:04d}: {size_mb:.1f} MB")
            if len(volumes) > 3:
                lines.append(f"    ... and {len(volumes) - 3} more")

    return "\n".join(lines)
