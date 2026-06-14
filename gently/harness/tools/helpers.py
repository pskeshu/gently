"""
Shared helper utilities for tool implementations

This module provides common validation and extraction patterns
used across multiple tools to reduce code duplication.
"""

from datetime import datetime
from typing import Any


def ctx_get(context: dict | None, key: str) -> Any:
    """
    Look up a key in a (possibly missing) tool execution context

    Parameters
    ----------
    context : dict | None
        Tool execution context
    key : str
        Key to look up

    Returns
    -------
    Any
        The value for ``key``, or ``None`` if ``context`` is ``None`` or
        the key is absent.
    """
    if context is None:
        return None
    return context.get(key)


def require_agent(context: dict | None) -> tuple[Any, str | None]:
    """
    Extract agent from context or return error message

    Parameters
    ----------
    context : dict | None
        Tool execution context

    Returns
    -------
    tuple
        (agent, None) if found, (None, error_message) if not
    """
    agent = ctx_get(context, "agent")
    if not agent:
        return None, "Error: No agent context"
    return agent, None


def get_embryo_or_error(agent, embryo_id: str) -> tuple[Any, str | None]:
    """
    Get embryo by any name or return error message

    Parameters
    ----------
    agent : MicroscopyAgent
        Agent instance
    embryo_id : str
        Embryo ID, nickname, or label

    Returns
    -------
    tuple
        (embryo, None) if found, (None, error_message) if not
    """
    embryo = agent.experiment.get_embryo_by_any_name(embryo_id)
    if not embryo:
        return None, f"Embryo '{embryo_id}' not found"
    return embryo, None


def require_microscope(context: dict | None) -> tuple[Any, str | None]:
    """
    Get microscope client from context or return error message

    Parameters
    ----------
    context : dict | None
        Tool execution context

    Returns
    -------
    tuple
        (client, None) if connected, (None, error_message) if not
    """
    client = ctx_get(context, "client")
    if not client:
        return None, "Not connected to microscope. Use connect_microscope first."
    return client, None


def require_interaction_logger(agent) -> tuple[Any, str | None]:
    """
    Get interaction logger or return error message

    Parameters
    ----------
    agent : MicroscopyAgent
        Agent instance

    Returns
    -------
    tuple
        (logger, None) if available, (None, error_message) if not
    """
    if not hasattr(agent, "interaction_logger") or not agent.interaction_logger:
        return None, "Interaction logging not enabled."
    return agent.interaction_logger, None


def require_developmental_tracker(agent) -> tuple[Any, str | None]:
    """
    Get developmental tracker or return error message

    Parameters
    ----------
    agent : MicroscopyAgent
        Agent instance

    Returns
    -------
    tuple
        (tracker, None) if available, (None, error_message) if not
    """
    if not hasattr(agent, "developmental_tracker") or not agent.developmental_tracker:
        return (
            None,
            "No stage classifications recorded yet. Use classify_embryo_stage first.",
        )
    return agent.developmental_tracker, None


def require_timelapse_orchestrator(agent) -> tuple[Any, str | None]:
    """
    Get timelapse orchestrator or return error message

    Parameters
    ----------
    agent : MicroscopyAgent
        Agent instance

    Returns
    -------
    tuple
        (orchestrator, None) if available, (None, error_message) if not
    """
    if not hasattr(agent, "timelapse_orchestrator") or agent.timelapse_orchestrator is None:
        return None, "Timelapse orchestrator not initialized."
    return agent.timelapse_orchestrator, None


def require_databroker(agent) -> tuple[Any, str | None]:
    """
    Get databroker connection or return error message

    Parameters
    ----------
    agent : MicroscopyAgent
        Agent instance

    Returns
    -------
    tuple
        (databroker, None) if available, (None, error_message) if not
    """
    if not hasattr(agent, "databroker") or agent.databroker is None:
        return None, "No databroker connection. Data persistence not available."
    return agent.databroker, None


def get_timestamp_string() -> str:
    """
    Get standard timestamp string for filenames

    Returns
    -------
    str
        Timestamp in YYYYMMDD_HHMMSS format
    """
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def format_duration(seconds: float) -> str:
    """
    Format a duration in seconds to human-readable string

    Parameters
    ----------
    seconds : float
        Duration in seconds

    Returns
    -------
    str
        Human-readable duration (e.g., "2h 30m", "45s")
    """
    if seconds < 60:
        return f"{seconds:.0f}s"
    elif seconds < 3600:
        mins = seconds / 60
        return f"{mins:.1f}m"
    else:
        hours = seconds / 3600
        return f"{hours:.1f}h"


def build_snapshot_metadata(
    stage_position: tuple[float, float],
    image_shape: tuple[int, ...],
    experiment=None,
    pixel_size_um: float = 6.5,
    objective_mag: float = 10.0,
    safety_limits: dict | None = None,
) -> dict:
    """Build metadata dict for a bottom camera snapshot.

    Captures everything needed to reconstruct embryo positions
    from the image later (training, annotation, replay).

    Parameters
    ----------
    stage_position : (x, y)
        XY stage coordinates at time of capture.
    image_shape : tuple
        Shape of the captured image (H, W) or (H, W, C).
    experiment : ExperimentState, optional
        If provided, all embryo positions are included.
    pixel_size_um : float
        Camera sensor pixel size in micrometers.
    objective_mag : float
        Objective magnification on the bottom camera.
    safety_limits : dict, optional
        Stage safety perimeter, e.g.
        ``{"x": (2000, 4000), "y": (-1000, 1000)}``.
        If None, uses diSPIM hardware defaults.

    Returns
    -------
    dict
        Metadata suitable for ``FileStore.register_snapshot()``.
    """
    um_per_pixel = pixel_size_um / objective_mag
    h, w = image_shape[0], image_shape[1]

    if safety_limits is None:
        # Default diSPIM XY stage limits (µm). Must match the default in
        # gently/hardware/dispim/devices/stage.py::DiSPIMXYStage.__init__.
        safety_limits = {"x": (2000.0, 4000.0), "y": (-1000.0, 1000.0)}

    meta: dict[str, Any] = {
        "stage_x": stage_position[0],
        "stage_y": stage_position[1],
        "image_width": w,
        "image_height": h,
        "coordinate_transform": {
            "pixel_size_um": pixel_size_um,
            "objective_mag": objective_mag,
            "um_per_pixel": um_per_pixel,
            "image_center_x": w / 2,
            "image_center_y": h / 2,
        },
        "safety_perimeter": safety_limits,
    }

    if experiment and experiment.embryos:
        embryos: list[dict] = []
        for eid, emb in experiment.embryos.items():
            pos = emb.stage_position or {}
            embryos.append(
                {
                    "embryo_id": eid,
                    "stage_x": pos.get("x"),
                    "stage_y": pos.get("y"),
                    "nickname": getattr(emb, "nickname", None),
                }
            )
        meta["embryos"] = embryos

    return meta
