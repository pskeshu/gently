"""
Event Publishing Utilities for CV Subagent

Centralizes event publishing logic for consistent event emission
across all CV tools and analysis pipelines.
"""

import logging
from typing import Any, Dict, List, Optional

from gently.core.event_bus import EventType, Event, get_event_bus

logger = logging.getLogger(__name__)

# CV Subagent source identifier
CV_SOURCE = "cv-subagent"


def publish_segmentation_completed(
    mask_uid: str,
    source_uid: str,
    num_cells: int,
    embryo_id: Optional[str] = None,
    timepoint: Optional[int] = None,
    model_type: str = "cellpose",
    task_id: Optional[str] = None,
    extra_data: Optional[Dict] = None,
) -> Event:
    """
    Publish SEGMENTATION_COMPLETED event

    Parameters
    ----------
    mask_uid : str
        UID of the generated masks
    source_uid : str
        UID of the source volume
    num_cells : int
        Number of cells/nuclei detected
    embryo_id : str, optional
        Embryo identifier
    timepoint : int, optional
        Timepoint index
    model_type : str
        Segmentation model used
    task_id : str, optional
        Associated task ID
    extra_data : dict, optional
        Additional data to include

    Returns
    -------
    Event
        The published event
    """
    data = {
        "mask_uid": mask_uid,
        "source_uid": source_uid,
        "num_cells": num_cells,
        "model_type": model_type,
    }

    if embryo_id:
        data["embryo_id"] = embryo_id
    if timepoint is not None:
        data["timepoint"] = timepoint
    if task_id:
        data["task_id"] = task_id
    if extra_data:
        data.update(extra_data)

    event_bus = get_event_bus()
    return event_bus.publish(
        event_type=EventType.SEGMENTATION_COMPLETED,
        data=data,
        source=CV_SOURCE,
    )


def publish_stage_detected(
    embryo_id: str,
    stage: str,
    confidence: float,
    timepoint: Optional[int] = None,
    nuclei_count: Optional[int] = None,
    elongation: Optional[float] = None,
    task_id: Optional[str] = None,
    extra_data: Optional[Dict] = None,
) -> Event:
    """
    Publish STAGE_DETECTED event

    Parameters
    ----------
    embryo_id : str
        Embryo identifier
    stage : str
        Detected developmental stage
    confidence : float
        Confidence score (0-1)
    timepoint : int, optional
        Timepoint index
    nuclei_count : int, optional
        Number of nuclei detected
    elongation : float, optional
        Embryo elongation ratio
    task_id : str, optional
        Associated task ID
    extra_data : dict, optional
        Additional data to include

    Returns
    -------
    Event
        The published event
    """
    data = {
        "embryo_id": embryo_id,
        "stage": stage,
        "confidence": confidence,
    }

    if timepoint is not None:
        data["timepoint"] = timepoint
    if nuclei_count is not None:
        data["nuclei_count"] = nuclei_count
    if elongation is not None:
        data["elongation"] = elongation
    if task_id:
        data["task_id"] = task_id
    if extra_data:
        data.update(extra_data)

    event_bus = get_event_bus()
    return event_bus.publish(
        event_type=EventType.STAGE_DETECTED,
        data=data,
        source=CV_SOURCE,
    )


def publish_cell_division_detected(
    embryo_id: str,
    timepoint: int,
    parent_label: int,
    daughter_labels: List[int],
    parent_centroid: Optional[List[float]] = None,
    task_id: Optional[str] = None,
    extra_data: Optional[Dict] = None,
) -> Event:
    """
    Publish CELL_DIVISION_DETECTED event

    Parameters
    ----------
    embryo_id : str
        Embryo identifier
    timepoint : int
        Timepoint when division detected
    parent_label : int
        Label of parent cell
    daughter_labels : list
        Labels of daughter cells
    parent_centroid : list, optional
        Centroid coordinates of parent cell
    task_id : str, optional
        Associated task ID
    extra_data : dict, optional
        Additional data to include

    Returns
    -------
    Event
        The published event
    """
    data = {
        "embryo_id": embryo_id,
        "timepoint": timepoint,
        "parent_label": parent_label,
        "daughter_labels": daughter_labels,
    }

    if parent_centroid:
        data["parent_centroid"] = parent_centroid
    if task_id:
        data["task_id"] = task_id
    if extra_data:
        data.update(extra_data)

    event_bus = get_event_bus()
    return event_bus.publish(
        event_type=EventType.CELL_DIVISION_DETECTED,
        data=data,
        source=CV_SOURCE,
    )


def publish_lineage_updated(
    embryo_id: str,
    num_tracks: int,
    num_divisions: int,
    timepoints_covered: List[int],
    task_id: Optional[str] = None,
    extra_data: Optional[Dict] = None,
) -> Event:
    """
    Publish LINEAGE_UPDATED event

    Parameters
    ----------
    embryo_id : str
        Embryo identifier
    num_tracks : int
        Total number of cell tracks
    num_divisions : int
        Number of division events detected
    timepoints_covered : list
        Timepoints included in tracking
    task_id : str, optional
        Associated task ID
    extra_data : dict, optional
        Additional data to include

    Returns
    -------
    Event
        The published event
    """
    data = {
        "embryo_id": embryo_id,
        "num_tracks": num_tracks,
        "num_divisions": num_divisions,
        "timepoints_covered": timepoints_covered,
    }

    if task_id:
        data["task_id"] = task_id
    if extra_data:
        data.update(extra_data)

    event_bus = get_event_bus()
    return event_bus.publish(
        event_type=EventType.LINEAGE_UPDATED,
        data=data,
        source=CV_SOURCE,
    )


def publish_anomaly_detected(
    embryo_id: str,
    anomaly_type: str,
    description: str,
    severity: str = "warning",  # "info", "warning", "critical"
    timepoint: Optional[int] = None,
    task_id: Optional[str] = None,
    extra_data: Optional[Dict] = None,
) -> Event:
    """
    Publish ANOMALY_DETECTED event

    Parameters
    ----------
    embryo_id : str
        Embryo identifier
    anomaly_type : str
        Type of anomaly (e.g., "delayed_division", "abnormal_morphology")
    description : str
        Human-readable description
    severity : str
        Severity level: "info", "warning", "critical"
    timepoint : int, optional
        Timepoint where anomaly detected
    task_id : str, optional
        Associated task ID
    extra_data : dict, optional
        Additional data to include

    Returns
    -------
    Event
        The published event
    """
    data = {
        "embryo_id": embryo_id,
        "anomaly_type": anomaly_type,
        "description": description,
        "severity": severity,
    }

    if timepoint is not None:
        data["timepoint"] = timepoint
    if task_id:
        data["task_id"] = task_id
    if extra_data:
        data.update(extra_data)

    event_bus = get_event_bus()
    return event_bus.publish(
        event_type=EventType.ANOMALY_DETECTED,
        data=data,
        source=CV_SOURCE,
    )


def publish_cv_task_queued(
    task_id: str,
    intent: str,
    embryo_id: str,
    plan: Optional[List[str]] = None,
) -> Event:
    """Publish CV_TASK_QUEUED event"""
    data = {
        "task_id": task_id,
        "intent": intent,
        "embryo_id": embryo_id,
    }
    if plan:
        data["plan"] = plan

    event_bus = get_event_bus()
    return event_bus.publish(
        event_type=EventType.CV_TASK_QUEUED,
        data=data,
        source=CV_SOURCE,
    )


def publish_cv_task_completed(
    task_id: str,
    result: Dict[str, Any],
    embryo_id: Optional[str] = None,
) -> Event:
    """Publish CV_TASK_COMPLETED event"""
    data = {
        "task_id": task_id,
        "result": result,
    }
    if embryo_id:
        data["embryo_id"] = embryo_id

    event_bus = get_event_bus()
    return event_bus.publish(
        event_type=EventType.CV_TASK_COMPLETED,
        data=data,
        source=CV_SOURCE,
    )


def publish_cv_task_failed(
    task_id: str,
    error: str,
    embryo_id: Optional[str] = None,
) -> Event:
    """Publish CV_TASK_FAILED event"""
    data = {
        "task_id": task_id,
        "error": error,
    }
    if embryo_id:
        data["embryo_id"] = embryo_id

    event_bus = get_event_bus()
    return event_bus.publish(
        event_type=EventType.CV_TASK_FAILED,
        data=data,
        source=CV_SOURCE,
    )


def publish_cv_agent_thinking(
    task_id: str,
    thinking: str,
    iteration: int,
    embryo_id: Optional[str] = None,
) -> Event:
    """
    Publish CV_AGENT_THINKING event for real-time streaming of agent reasoning.

    Allows subscribers (viz server, copilot) to display the agent's
    reasoning process as it happens, providing transparency into
    multi-step analysis decisions.

    Parameters
    ----------
    task_id : str
        Associated task ID
    thinking : str
        The thinking/reasoning text from Claude
    iteration : int
        Current iteration number in the agentic loop
    embryo_id : str, optional
        Embryo identifier

    Returns
    -------
    Event
        The published event
    """
    data = {
        "task_id": task_id,
        "thinking": thinking,
        "iteration": iteration,
    }
    if embryo_id:
        data["embryo_id"] = embryo_id

    event_bus = get_event_bus()
    return event_bus.publish(
        event_type=EventType.CV_AGENT_THINKING,
        data=data,
        source=CV_SOURCE,
    )
