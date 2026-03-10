"""
Volume Viewing Tools

Tools for viewing and listing acquired lightsheet volumes.
"""

import logging
from typing import Dict, Optional
from datetime import datetime

logger = logging.getLogger(__name__)

from gently.harness.tools.registry import tool, ToolCategory, ToolExample
from gently.harness.tools.helpers import require_agent, get_embryo_or_error
from gently.core.coordinates import stage_to_pixel_position, get_um_per_pixel


@tool(
    name="view_image",
    description="""Capture and display the current bottom camera widefield image. Shows what's visible at the current stage position.
Use when user says "show me the view", "take a picture", "what does it look like?", or to check sample positioning.
This is the widefield/brightfield camera looking up at the sample - good for seeing embryo outlines and overall positioning.
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
    exposure_ms: float = None,
    show: bool = True,
    show_embryos: bool = True,
    context: Dict = None
) -> str:
    """Capture and display bottom camera image with embryo annotations"""
    client = context.get('client')
    agent = context.get('agent')

    try:
        image = await client.capture_bottom_image(exposure_ms=exposure_ms)

        if image is None or image.shape == (100, 100):
            return "Failed to capture image from bottom camera"

        # Get current stage position for coordinate conversion
        stage_pos = await client.get_stage_position()

        # Prepare embryo annotations if requested
        embryo_annotations = []
        if show_embryos and agent and agent.experiment.embryos:
            um_per_pixel = get_um_per_pixel()  # Uses centralized defaults from coordinates.py
            image_center_x = image.shape[1] / 2
            image_center_y = image.shape[0] / 2

            for embryo_id, embryo in agent.experiment.embryos.items():
                if embryo.stage_position:
                    # Convert stage position to pixel position using centralized function
                    emb_x = embryo.stage_position.get('x', 0)
                    emb_y = embryo.stage_position.get('y', 0)
                    pixel_x, pixel_y = stage_to_pixel_position(
                        stage_x=emb_x,
                        stage_y=emb_y,
                        current_stage_x=stage_pos[0],
                        current_stage_y=stage_pos[1],
                        image_center_x=image_center_x,
                        image_center_y=image_center_y,
                        um_per_pixel=um_per_pixel
                    )

                    embryo_annotations.append({
                        'embryo_id': embryo_id,
                        'pixel_x': pixel_x,
                        'pixel_y': pixel_y,
                        'label': embryo.user_label or embryo_id
                    })

        if show:
            from datetime import datetime
            from pathlib import Path
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            save_path = f"camera_captures/bottom_camera_{timestamp}.jpg"
            Path("camera_captures").mkdir(exist_ok=True)

            view_result = await client.view_image(
                image=image,
                title=title,
                save_path=save_path,
                show=True,
                embryo_annotations=embryo_annotations if embryo_annotations else None
            )
            num_visible = len([a for a in embryo_annotations
                              if 0 <= a['pixel_x'] < image.shape[1] and 0 <= a['pixel_y'] < image.shape[0]])
            annotation_msg = f"\nShowing {num_visible} embryo(s) in view" if embryo_annotations else ""
            return f"Captured bottom camera image ({image.shape[0]}x{image.shape[1]})\nSaved to: {save_path}{annotation_msg}"
        else:
            return f"Captured bottom camera image ({image.shape[0]}x{image.shape[1]})"

    except Exception as e:
        return f"Error capturing image: {str(e)}"


@tool(
    name="view_volume",
    description="""Open a volume in napari for 3D visualization.
Can open a volume by file path OR by embryo ID (opens latest volume or specific timepoint).
Use when user says "open volume", "view volume", "show volume in napari", or "look at the 3D data".""",
    category=ToolCategory.ANALYSIS,
    requires_microscope=False,
    examples=[
        ToolExample("Open latest volume for embryo 2", {"embryo_id": "embryo_2"}),
        ToolExample("Open specific timepoint", {"embryo_id": "embryo_2", "timepoint": 5}),
        ToolExample("Open volume file", {"file_path": "D:/Gently/volumes/embryo_1_t0001.tif"}),
    ],
)
async def view_volume(
    embryo_id: str = None,
    timepoint: int = None,
    file_path: str = None,
    context: Dict = None
) -> str:
    """Open a volume in napari for visualization"""
    import napari
    import tifffile
    import numpy as np
    from pathlib import Path

    agent, err = require_agent(context)
    if err:
        return err

    volume = None
    volume_path = None
    title = "Volume Viewer"

    # Determine which volume to open
    if file_path:
        # Open from file path
        volume_path = Path(file_path)
        if not volume_path.exists():
            return f"Error: File not found: {file_path}"
        title = f"Volume: {volume_path.name}"

    elif embryo_id:
        # Get volume for embryo from GentlyStore
        session_id = agent.session_id

        if timepoint is not None:
            # Try to find specific timepoint via GentlyStore
            volume_path = agent.store.get_volume_path(session_id, embryo_id, timepoint)
            if volume_path and volume_path.exists():
                title = f"{embryo_id} - t{timepoint:04d}"
            else:
                # Check recent_images as fallback
                embryo, err = get_embryo_or_error(agent, embryo_id)
                if err:
                    return err
                if embryo.recent_images:
                    matching = [img for img in embryo.recent_images if img.timepoint == timepoint]
                    if matching:
                        volume_path = Path(matching[0].volume_path)
                        title = f"{embryo_id} - t{timepoint:04d}"

                if not volume_path or not volume_path.exists():
                    # List available timepoints from store
                    volumes = agent.store.list_volumes(session_id, embryo_id)
                    available = sorted([v['timepoint'] for v in volumes])
                    return f"Timepoint {timepoint} not found for {embryo_id}. Available: {available}"
        else:
            # Find latest volume from store
            volumes = agent.store.list_volumes(session_id, embryo_id)
            if not volumes:
                return f"No volumes found for {embryo_id} in session {session_id}"

            # Find highest timepoint
            latest = max(volumes, key=lambda v: v['timepoint'])
            latest_tp = latest['timepoint']
            volume_path = agent.store.get_volume_path(session_id, embryo_id, latest_tp)

            title = f"{embryo_id} - t{latest_tp:04d}"

    else:
        return "Error: Specify either embryo_id or file_path"

    # Load the volume
    try:
        volume = tifffile.imread(str(volume_path))
        logger.info("Loaded volume: %s, dtype=%s", volume.shape, volume.dtype)
    except Exception as e:
        return f"Error loading volume: {e}"

    # Open in napari
    logger.info("Opening napari viewer...")
    viewer = napari.Viewer(title=title)

    # Add volume with appropriate settings
    viewer.add_image(
        volume,
        name='Volume',
        colormap='gray',
        rendering='mip',  # Maximum intensity projection for 3D
    )

    # Add scale bar info
    viewer.scale_bar.visible = True
    viewer.scale_bar.unit = "um"

    napari.run()

    return f"\u2713 Opened volume in napari: {volume_path.name} (shape: {volume.shape})"


@tool(
    name="list_volumes",
    description="""List available volumes for an embryo or all embryos.
Shows volume files with timepoints and file sizes. Scans the storage directory for all volumes (not just recent ones in memory).
Use to see what data is available before viewing.""",
    category=ToolCategory.ANALYSIS,
    requires_microscope=False,
    examples=[
        ToolExample("List volumes for embryo 2", {"embryo_id": "embryo_2"}),
        ToolExample("List all volumes", {}),
    ],
)
async def list_volumes(
    embryo_id: str = None,
    context: Dict = None
) -> str:
    """List available volumes"""
    agent, err = require_agent(context)
    if err:
        return err

    session_id = agent.session_id
    lines = []

    # Get volumes from GentlyStore
    all_volumes_list = agent.store.list_volumes(session_id, embryo_id)

    # Group by embryo_id
    all_volumes = {}  # embryo_id -> list of volume records
    for vol in all_volumes_list:
        eid = vol['embryo_id']
        if eid not in all_volumes:
            all_volumes[eid] = []
        all_volumes[eid].append(vol)

    # Sort by timepoint
    for eid in all_volumes:
        all_volumes[eid].sort(key=lambda x: x['timepoint'])

    if embryo_id:
        # List volumes for specific embryo
        if embryo_id not in all_volumes:
            return f"No volumes found for {embryo_id} in session {session_id}"

        volumes = all_volumes[embryo_id]
        lines.append(f"Volumes for {embryo_id}: {len(volumes)} file(s)")
        lines.append(f"Session: {session_id}")
        lines.append("")

        for vol in volumes:
            tp = vol['timepoint']
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
        lines.append(f"Available volumes: {total_files} file(s) across {len(all_volumes)} embryo(s)")
        lines.append(f"Session: {session_id}")

        for eid in sorted(all_volumes.keys()):
            volumes = all_volumes[eid]
            timepoints = [v['timepoint'] for v in volumes]
            tp_range = f"t{min(timepoints):04d}-t{max(timepoints):04d}" if len(timepoints) > 1 else f"t{timepoints[0]:04d}"

            # Calculate total size
            total_size = 0
            for vol in volumes:
                path = agent.store.get_volume_path(session_id, eid, vol['timepoint'])
                if path and path.exists():
                    total_size += path.stat().st_size / (1024 * 1024)
            lines.append(f"\n{eid}: {len(volumes)} volume(s) [{tp_range}] ({total_size:.1f} MB total)")

            # Show last few timepoints
            for vol in volumes[-3:]:
                tp = vol['timepoint']
                path = agent.store.get_volume_path(session_id, eid, tp)
                if path and path.exists():
                    size_mb = path.stat().st_size / (1024 * 1024)
                    lines.append(f"    t{tp:04d}: {size_mb:.1f} MB")
            if len(volumes) > 3:
                lines.append(f"    ... and {len(volumes) - 3} more")

    return "\n".join(lines)
