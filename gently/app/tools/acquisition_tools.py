"""
Acquisition Tools

Tools for acquiring lightsheet volumes and images from the microscope.
"""

import asyncio
import logging
import time
from typing import Any

import numpy as np

from gently.harness.tools.helpers import ctx_get, get_embryo_or_error
from gently.harness.tools.registry import ToolCategory, ToolExample, tool

logger = logging.getLogger(__name__)


def _publish_scan_geometry(
    agent: Any,
    *,
    embryo_id: str,
    stage_position: dict | None,
    num_slices: int,
    exposure_ms: float,
    galvo_amplitude: float,
    galvo_center: float,
    piezo_amplitude: float,
    piezo_center: float,
) -> None:
    """Emit SCAN_GEOMETRY_UPDATE describing the cuboid being acquired.

    Drives the 3D optical-space view (the addressable volume + the scan cuboid
    and light-sheet mode). Telemetry only — callers guard against exceptions so
    this never interferes with an acquisition. The payload is also stashed on
    the agent for REST bootstrap (``/api/devices/scan_geometry``).
    """
    from gently.core import EventType, get_event_bus

    z_extent_um = 2.0 * piezo_amplitude
    slice_spacing_um = z_extent_um / (num_slices - 1) if num_slices > 1 else 0.0
    sx = stage_position.get("x") if stage_position else None
    sy = stage_position.get("y") if stage_position else None

    payload: dict[str, Any] = {
        "embryo_id": embryo_id,
        "stage_position_um": {"x": sx, "y": sy},
        "scan": {
            "num_slices": num_slices,
            "exposure_ms": exposure_ms,
            "galvo_amplitude_deg": galvo_amplitude,
            "galvo_center_deg": galvo_center,
            "piezo_amplitude_um": piezo_amplitude,
            "piezo_center_um": piezo_center,
        },
        "derived": {
            "z_extent_um": z_extent_um,
            "slice_spacing_um": slice_spacing_um,
            "z_min_um": piezo_center - piezo_amplitude,
            "z_max_um": piezo_center + piezo_amplitude,
        },
        # diSPIM here is scanned-light-sheet only; a future pencil/beam tool
        # would emit "pencil". See the 3D optical-space view notes.
        "mode": "sheet",
        "ts": time.time(),
    }
    agent.last_scan_geometry = payload
    get_event_bus().publish(
        event_type=EventType.SCAN_GEOMETRY_UPDATE,
        data=payload,
        source="acquisition-tools",
    )


@tool(
    name="acquire_volume",
    description="""Acquire a single 3D lightsheet volume for a specific embryo. Moves to embryo
position and uses its calibration data.
Use when user wants a full 3D stack of an embryo (e.g., "acquire volume of embryo 1",
"take a 3D image"). Embryo must be calibrated first. Default 50 slices at 10ms exposure
takes ~2.5 seconds. Turns laser on during acquisition.

The z_buffer_um parameter can override the calibrated Z range to add more empty space
above/below the embryo. This is useful for segmentation without needing to recalibrate.
Set to None to use calibrated range.""",
    category=ToolCategory.HARDWARE,
    requires_microscope=True,
    examples=[
        ToolExample("Acquire volume of embryo 1", {"embryo_id": "embryo_1"}),
        ToolExample(
            "Take a 3D image of embryo 2 with 80 slices",
            {"embryo_id": "embryo_2", "num_slices": 80},
        ),
        ToolExample(
            "Acquire with more Z padding",
            {"embryo_id": "embryo_1", "z_buffer_um": 20.0},
        ),
    ],
)
async def acquire_volume(
    embryo_id: str,
    num_slices: int = 50,
    exposure_ms: float = 10.0,
    z_buffer_um: float | None = None,
    context: dict | None = None,
) -> str:
    """Acquire single volume - moves to embryo first, uses calibration"""
    agent = ctx_get(context, "agent")
    client = ctx_get(context, "client")

    if not agent:
        return "Error: No agent context"

    embryo, err = get_embryo_or_error(agent, embryo_id)
    if err:
        return err

    try:
        # Move to embryo position first
        pos = embryo.stage_position
        if pos and pos.get("x") is not None and pos.get("y") is not None:
            await client.move_to_position(pos["x"], pos["y"])

        # Get calibration parameters (use defaults if not calibrated)
        cal = embryo.calibration or {}
        galvo_amplitude = cal.get("galvo_amplitude", 0.5)
        galvo_center = cal.get("galvo_center", 0.0)
        piezo_amplitude = cal.get("piezo_amplitude", 25.0)
        piezo_center = cal.get("piezo_center", 50.0)

        # Override Z range if z_buffer_um is specified
        z_buffer_applied = None
        if z_buffer_um is not None and cal:
            # Get the original embryo extent from calibration
            calibrated_buffer = cal.get("z_buffer_um", 5.0)  # Old default was 5µm
            slope = cal.get("slope_um_per_deg", 100.0)

            # Calculate additional buffer needed
            additional_buffer_um = z_buffer_um - calibrated_buffer
            if additional_buffer_um > 0:
                # Convert µm to degrees and add to amplitude
                additional_buffer_deg = additional_buffer_um / 100.0
                galvo_amplitude = galvo_amplitude + additional_buffer_deg
                # Piezo amplitude scales with slope
                piezo_amplitude = piezo_amplitude + (additional_buffer_um * abs(slope) / 100.0)
                z_buffer_applied = z_buffer_um

        # Publish the resolved scan geometry for the 3D optical-space view.
        # Telemetry only — must never break the acquisition.
        try:
            _publish_scan_geometry(
                agent,
                embryo_id=embryo_id,
                stage_position=pos,
                num_slices=num_slices,
                exposure_ms=exposure_ms,
                galvo_amplitude=galvo_amplitude,
                galvo_center=galvo_center,
                piezo_amplitude=piezo_amplitude,
                piezo_center=piezo_center,
            )
        except Exception:
            logger.debug("SCAN_GEOMETRY_UPDATE publish failed", exc_info=True)

        result = await client.acquire_volume(
            num_slices=num_slices,
            exposure_ms=exposure_ms,
            galvo_amplitude=galvo_amplitude,
            galvo_center=galvo_center,
            piezo_amplitude=piezo_amplitude,
            piezo_center=piezo_center,
            laser_power_488_pct=embryo.laser_power_488_pct,
        )

        if result.get("success"):
            volume = result.get("volume")
            timepoint = embryo.timepoints_acquired  # Current timepoint (0-indexed)

            # Increment timepoints acquired
            embryo.timepoints_acquired += 1

            # Record light exposure (num_slices frames at exposure_ms each)
            embryo.record_exposure(exposure_ms=exposure_ms, num_frames=num_slices)

            # Store in FileStore
            saved_path = None
            if agent.store and agent.session_id:
                try:
                    from pathlib import Path as _Path

                    pos = embryo.stage_position or {}
                    agent.store.register_embryo(
                        agent.session_id,
                        embryo_id,
                        position_x=pos.get("x"),
                        position_y=pos.get("y"),
                        calibration=embryo.calibration,
                        role=embryo.role,
                    )
                    acq_metadata = {
                        "num_slices": num_slices,
                        "exposure_ms": exposure_ms,
                        "interval_seconds": embryo.interval_seconds,
                        "acquisition_mode": embryo.acquisition_mode,
                        "laser_power_488_pct": embryo.laser_power_488_pct,
                        "role": embryo.role,
                        "calibration": {
                            "galvo_amplitude": galvo_amplitude,
                            "galvo_center": galvo_center,
                            "piezo_amplitude": piezo_amplitude,
                            "piezo_center": piezo_center,
                        },
                    }
                    from gently.app.temperature_sampler import temperature_stamp as _ts

                    _temp_stamp = _ts(
                        getattr(getattr(agent, "temperature_sampler", None), "latest", None)
                    )
                    if _temp_stamp is not None:
                        acq_metadata["temperature"] = _temp_stamp
                    volume_path_ref = result.get("volume_path")
                    if volume_path_ref is not None:
                        saved_path = agent.store.register_volume(
                            agent.session_id,
                            embryo_id,
                            timepoint,
                            incoming_path=_Path(volume_path_ref),
                            metadata=acq_metadata,
                            volume_data=volume,
                        )
                    elif volume is not None:
                        saved_path = agent.store.put_volume(
                            agent.session_id,
                            embryo_id,
                            timepoint,
                            volume,
                            metadata=acq_metadata,
                        )
                except Exception as store_err:
                    logger.warning("FileStore write failed (non-fatal): %s", store_err)

            # Push max projection to viz server
            if agent.viz_server and volume is not None:
                try:
                    # Create max intensity projection (View A only)
                    vol = volume
                    # If 4D (Views, Z, Y, X), select View A (index 0)
                    if vol.ndim == 4:
                        vol = vol[0]  # View A
                    max_proj = np.max(vol, axis=0)
                    # Include session_id in UID to ensure uniqueness across sessions
                    session_prefix = f"{agent.session_id[:8]}_" if agent.session_id else ""
                    agent.push_viz(
                        array=max_proj,
                        uid=f"volume_{session_prefix}{embryo_id}_t{timepoint:04d}",
                        data_type="volume_projection",
                        metadata={
                            "embryo_id": embryo_id,
                            "timepoint": timepoint,
                            "shape": list(volume.shape) if hasattr(volume, "shape") else None,
                            "num_slices": num_slices,
                            "exposure_ms": exposure_ms,
                        },
                    )
                except Exception as viz_err:
                    logger.warning("Failed to push volume to viz: %s", viz_err)

            # Build response
            shape_str = str(result.get("shape", "unknown"))
            z_info = f" (z_buffer: {z_buffer_applied}\u00b5m)" if z_buffer_applied else ""
            if saved_path:
                return (
                    f"Acquired volume for {embryo.id}{z_info}\nShape: {shape_str}"
                    f"\nSaved: {saved_path}"
                )
            else:
                return (
                    f"Acquired volume for {embryo.id}{z_info}\nShape: {shape_str}"
                    "\n(Volume not saved to disk)"
                )
        else:
            return f"Acquisition failed: {result.get('error', 'Unknown error')}"

    except Exception as e:
        return f"Error acquiring volume: {str(e)}"


@tool(
    name="capture_lightsheet",
    description="""Capture a single 2D lightsheet fluorescence image at specified piezo/galvo
position. Uses 50ms exposure by default.
Use when user says "take a lightsheet image", "lightsheet snap", or wants to see fluorescence
at a specific Z position. This is a COMPLETE action - do NOT follow up with acquire_volume
unless user explicitly asks for a 3D volume.

IMPORTANT: Always pass embryo_id when capturing for an embryo. This ensures the image is
captured at the correct focus position from the embryo's focus_history (set by fine_focus).
Without embryo_id, focus may be incorrect.

The piezo position is determined by priority:
1. Explicit piezo_position parameter (if provided)
2. Embryo's focus_history for the given galvo_position (if embryo_id provided and has focus data)
3. Hardware query fallback (unreliable, may return 0)

If embryo has no focus data for the requested galvo_position, consider running fine_focus first.""",
    category=ToolCategory.HARDWARE,
    requires_microscope=True,
    examples=[
        ToolExample("Take a lightsheet image of embryo 1", {"embryo_id": "embryo_1"}),
        ToolExample(
            "Lightsheet snap at specific piezo",
            {"embryo_id": "embryo_1", "piezo_position": 50.0},
        ),
        ToolExample(
            "Capture at different galvo",
            {"embryo_id": "embryo_1", "galvo_position": 0.5},
        ),
    ],
)
async def capture_lightsheet(
    piezo_position: float | None = None,
    galvo_position: float = 0.0,
    embryo_id: str | None = None,
    show: bool = True,
    context: dict | None = None,
) -> str:
    """Capture and optionally display a single lightsheet image"""
    client = ctx_get(context, "client")
    agent = ctx_get(context, "agent")

    try:
        embryo = None
        # If embryo_id specified, get the embryo and move to its position
        if embryo_id and agent:
            embryo = agent.experiment.get_embryo_by_any_name(embryo_id)
            if embryo and embryo.stage_position:
                # Move stage to embryo's position
                await client.move_to_position(
                    x=embryo.stage_position["x"], y=embryo.stage_position["y"]
                )

        # Determine piezo position for best focus
        # Priority: explicit param > embryo focus history > hardware query
        focus_source = None
        if piezo_position is None:
            # Check embryo's focus history first (from fine_focus)
            if embryo and embryo.focus_history:
                # Try interpolation if we have 2+ points
                fit = embryo.get_piezo_galvo_fit()
                if fit is not None:
                    slope, intercept = fit
                    piezo_position = slope * galvo_position + intercept
                    focus_source = "interpolated"
                else:
                    # Single point or exact match
                    piezo_position = embryo.get_focus_at_galvo(galvo_position)
                    if piezo_position is not None:
                        focus_source = "focus_history"

            # Fall back to hardware query (unreliable)
            if piezo_position is None:
                piezo_position = await client.get_piezo_position()
                focus_source = "hardware_query"

        result = await client.capture_lightsheet_image(
            piezo_position=piezo_position, galvo_position=galvo_position
        )

        if result.get("success"):
            image = result.get("image")
            run_uid = result.get("run_uid", "unknown")

            # Update embryo's last_imaged and exposure tracking if specified
            if embryo:
                # Default lightsheet exposure is 50ms
                embryo.record_exposure(exposure_ms=50.0, num_frames=1)

            # Build focus info string
            focus_info = ""
            if focus_source == "interpolated":
                focus_info = " (focus: interpolated from calibration)"
            elif focus_source == "focus_history":
                focus_info = " (focus: from fine_focus)"
            elif focus_source == "hardware_query":
                focus_info = " (focus: hardware query - may be inaccurate)"

            if image is not None and show:
                # Display the image
                from datetime import datetime
                from pathlib import Path

                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                save_path = f"lightsheet_captures/lightsheet_{timestamp}.jpg"
                Path("lightsheet_captures").mkdir(exist_ok=True)

                await client.view_image(
                    image=image,
                    title=f"Lightsheet: piezo={piezo_position:.2f}um, galvo={galvo_position}V",
                    save_path=save_path,
                    show=True,
                )
                return (
                    f"\u2713 Captured lightsheet at piezo={piezo_position:.2f}\u03bcm,"
                    f" galvo={galvo_position}V{focus_info}\nSaved to: {save_path}"
                )
            elif image is None:
                return (
                    f"\u2713 Lightsheet captured at piezo={piezo_position:.2f}\u03bcm,"
                    f" galvo={galvo_position}V{focus_info} (image not displayed)"
                    f"\nRun UID: {run_uid}"
                )
            else:
                return (
                    f"\u2713 Captured lightsheet at piezo={piezo_position:.2f}\u03bcm,"
                    f" galvo={galvo_position}V{focus_info}"
                )
        else:
            return f"Failed: {result.get('error', 'Unknown error')}"

    except Exception as e:
        return f"Error capturing lightsheet: {str(e)}"


@tool(
    name="batch_lightsheet",
    description="""Capture lightsheet images from ALL embryos and show them together in the web UI.
Use when user says "lightsheet all embryos", "capture all embryos", "show me all embryos in
lightsheet".
Moves to each embryo, captures a lightsheet image, saves it, and pushes it to the
web viewer (live image strip) for everyone watching. Much more efficient than
capturing one at a time.""",
    category=ToolCategory.HARDWARE,
    requires_microscope=True,
    examples=[
        ToolExample("Take lightsheet of all embryos", {}),
        ToolExample("Capture all embryos", {}),
    ],
)
async def batch_lightsheet(galvo_position: float = 0.0, context: dict | None = None) -> str:
    """Capture lightsheet images from all embryos and show them in the web UI"""
    agent = ctx_get(context, "agent")
    client = ctx_get(context, "client")

    if not agent or not client:
        return "Error: Agent or microscope not available"

    if not agent.experiment.embryos:
        return "No embryos in experiment. Run detect_embryos first."

    # Collect images from all embryos
    images = []
    embryo_ids = []
    errors = []

    active_embryos = [
        (eid, emb) for eid, emb in agent.experiment.embryos.items() if not emb.should_skip
    ]

    if not active_embryos:
        return "No active embryos to capture. All embryos are marked as skipped."

    logger.info("Capturing lightsheet from %d embryos...", len(active_embryos))

    for embryo_id, embryo in active_embryos:
        try:
            # Move to embryo position
            if embryo.stage_position:
                x = embryo.stage_position.get("x", 0)
                y = embryo.stage_position.get("y", 0)
                logger.info("Moving to %s at (%.1f, %.1f)...", embryo_id, x, y)
                await client.move_to_position(x, y)
                # Wait for stage to settle
                await asyncio.sleep(0.5)

            # Get piezo and galvo positions from calibration or defaults
            piezo_position = 4.0  # default for galvo=0
            embryo_galvo = galvo_position  # use parameter as default

            if embryo.calibration:
                # Get piezo center
                if embryo.calibration.get("piezo_center"):
                    piezo_position = embryo.calibration["piezo_center"]
                elif embryo.calibration.get("focus_position"):
                    piezo_position = embryo.calibration["focus_position"]

                # Get galvo center (critical for light sheet alignment)
                if embryo.calibration.get("galvo_center"):
                    embryo_galvo = embryo.calibration["galvo_center"]

            # Capture lightsheet
            logger.info(
                "Capturing %s at piezo=%.1f um, galvo=%.2f...",
                embryo_id,
                piezo_position,
                embryo_galvo,
            )
            result = await client.capture_lightsheet_image(
                piezo_position=piezo_position, galvo_position=embryo_galvo
            )

            if result.get("success") and result.get("image") is not None:
                images.append(result["image"])
                embryo_ids.append(embryo_id)
                # Track light exposure (default 50ms)
                embryo.record_exposure(exposure_ms=50.0, num_frames=1)
            else:
                errors.append(f"{embryo_id}: {result.get('error', 'no image')}")

        except Exception as e:
            errors.append(f"{embryo_id}: {str(e)}")

    if not images:
        return f"Failed to capture any images. Errors: {'; '.join(errors)}"

    # Save images
    from datetime import datetime
    from pathlib import Path

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = Path("lightsheet_captures") / f"batch_{timestamp}"
    save_dir.mkdir(parents=True, exist_ok=True)

    for _i, (img, eid) in enumerate(zip(images, embryo_ids, strict=False)):
        import tifffile

        save_path = save_dir / f"{eid}.tiff"
        tifffile.imwrite(str(save_path), img)

    logger.info("Saved %d images to %s", len(images), save_dir)

    # Push each captured image to the web UI \u2014 no blocking desktop window.
    # They appear in the live viewer / recent strip for everyone watching.
    pushed = 0
    if agent.viz_server is not None:
        for img, eid in zip(images, embryo_ids, strict=False):
            uid = f"batch_lightsheet_{eid}_{timestamp}"
            agent.push_viz(
                img,
                uid,
                "image",
                {"embryo_id": eid, "source": "batch_lightsheet", "label": eid},
            )
            pushed += 1
        logger.info("Pushed %d batch-lightsheet images to the web UI", pushed)

    # Summary
    summary = f"\u2713 Captured {len(images)} embryos: {', '.join(embryo_ids)}"
    if pushed:
        summary += f"\nShowing {pushed} image(s) in the web UI viewer."
    elif agent.viz_server is None:
        summary += "\n(Web UI not running \u2014 images saved to disk only.)"
    if errors:
        summary += f"\n\u26a0 Errors: {'; '.join(errors)}"
    summary += f"\nSaved to: {save_dir}"

    return summary
