"""
Narrative Generation for the Visualization Server
===================================================

Generates experiment narrative summaries from timelapse tracker state.
"""

from datetime import datetime
from typing import Any, Dict, Optional

from .timelapse_tracker import TimelapseStateTracker


def generate_narrative_summary(
    tracker: TimelapseStateTracker,
    since: Optional[str] = None,
) -> Dict[str, Any]:
    """Generate a narrative summary of the experiment state.

    This generates a local summary based on the timelapse tracker state.
    Can be extended to use Claude Haiku for AI-powered narratives.

    Args:
        tracker: The timelapse state tracker
        since: Optional ISO timestamp to generate differential summary

    Returns:
        Dict with status, headline, summary, and details
    """
    # If no active experiment
    if tracker.status == "IDLE" and not tracker.embryos:
        return {
            "status": "normal",
            "headline": "No Active Experiment",
            "summary": None,
            "details": ["Start a timelapse to see experiment summaries here."],
            "generated_at": datetime.now().isoformat()
        }

    # Count embryos
    embryo_count = len(tracker.embryos)
    active_embryos = [e for e in tracker.embryos.values() if not e.get("is_complete")]
    completed_embryos = [e for e in tracker.embryos.values() if e.get("is_complete")]

    # Check if using perception system (has stage data) or legacy detectors
    is_perception = False
    stage_info = {}  # embryo_id -> current stage
    hatching_embryos = []
    stage_order = ['early', 'bean', 'comma', '1.5fold', '2fold', '3fold', 'pretzel', 'hatching', 'hatched']

    for embryo_id, reasoning_list in tracker.detection_reasoning.items():
        # Check for stage data (perception system)
        stages = [r.get("stage") for r in reasoning_list if r.get("stage")]
        if stages:
            is_perception = True
            stage_info[embryo_id] = stages[-1]  # Latest stage
            # Check for hatching
            if any(r.get("is_hatching") for r in reasoning_list):
                hatching_embryos.append(embryo_id)

    # Count detections (legacy) or stage progression (perception)
    total_detections = 0
    detection_details = []

    if is_perception:
        # Group embryos by stage for perception
        for embryo_id, stage in stage_info.items():
            detection_details.append(f"{embryo_id}: {stage.replace('fold', '-fold').title()}")
    else:
        # Legacy detection counting
        for embryo_id, reasoning_list in tracker.detection_reasoning.items():
            positives = [r for r in reasoning_list if r.get("detected")]
            total_detections += len(positives)
            for d in positives:
                detector = d.get("detector_name", "unknown")
                tp = d.get("timepoint", "?")
                detection_details.append(f"{embryo_id}: {detector.title()} at T{tp}")

    # Build details list
    details = []

    if len(active_embryos) > 0:
        details.append(f"{len(active_embryos)} embryo{'s' if len(active_embryos) != 1 else ''} actively imaging")

    if len(completed_embryos) > 0:
        details.append(f"{len(completed_embryos)} embryo{'s' if len(completed_embryos) != 1 else ''} completed")

    details.append(f"{tracker.total_timepoints} total timepoints acquired")

    if tracker.base_interval:
        interval_str = f"{tracker.base_interval // 60} min" if tracker.base_interval >= 60 else f"{tracker.base_interval}s"
        details.append(f"Imaging interval: {interval_str}")

    if is_perception and stage_info:
        # Show stage distribution for perception
        stage_counts = {}
        for stage in stage_info.values():
            stage_counts[stage] = stage_counts.get(stage, 0) + 1
        # Sort by stage order
        sorted_stages = sorted(stage_counts.items(),
                               key=lambda x: stage_order.index(x[0].lower()) if x[0].lower() in stage_order else 99)
        stage_summary = ", ".join(f"{count} {stage}" for stage, count in sorted_stages)
        details.append(f"Stages: {stage_summary}")
        if hatching_embryos:
            details.append(f"Hatching detected: {', '.join(hatching_embryos)}")
    elif detection_details:
        if len(detection_details) <= 3:
            details.append(f"{total_detections} detection{'s' if total_detections != 1 else ''}: {', '.join(detection_details)}")
        else:
            details.append(f"{total_detections} detection{'s' if total_detections != 1 else ''}: {', '.join(detection_details[:3])}...")

    # Calculate duration
    if tracker.started_at:
        started = datetime.fromisoformat(tracker.started_at) if isinstance(tracker.started_at, str) else tracker.started_at
        duration_sec = (datetime.now() - started).total_seconds()
        hours = int(duration_sec // 3600)
        minutes = int((duration_sec % 3600) // 60)
        duration_str = f"{hours}h {minutes}m" if hours > 0 else f"{minutes}m"
        details.append(f"Running for {duration_str}")

    # Determine status and headline
    if is_perception:
        if hatching_embryos:
            status = "notable"
            headline = f"Hatching Detected in {len(hatching_embryos)} Embryo{'s' if len(hatching_embryos) != 1 else ''}"
        elif stage_info:
            # Find most advanced stage
            max_stage_idx = max(stage_order.index(s.lower()) if s.lower() in stage_order else 0
                                for s in stage_info.values())
            max_stage = stage_order[max_stage_idx].replace('fold', '-fold').title()
            status = "normal"
            headline = f"Most Advanced: {max_stage}"
        else:
            status = "normal"
            headline = "Experiment In Progress"
    elif total_detections > 0:
        status = "notable"
        headline = f"{total_detections} Detection{'s' if total_detections != 1 else ''} Found"
    elif len(completed_embryos) > 0:
        status = "normal"
        headline = f"{len(completed_embryos)}/{embryo_count} Embryos Complete"
    else:
        status = "normal"
        headline = "Experiment In Progress"

    # Build summary text
    summary = None
    if is_perception:
        if hatching_embryos:
            summary = f"Hatching has been detected in {', '.join(hatching_embryos)}. Monitoring continues for all embryos."
        elif stage_info:
            # Summarize stage distribution
            unique_stages = set(stage_info.values())
            if len(unique_stages) == 1:
                summary = f"All {len(stage_info)} embryos are at {list(unique_stages)[0].replace('fold', '-fold').title()} stage."
            else:
                summary = f"Embryos are progressing through developmental stages. {len(stage_info)} embryos tracked."
    elif total_detections > 0:
        latest = detection_details[-1] if detection_details else None
        summary = f"Positive detections have been identified. {latest}. All imaging continues normally."
    elif len(completed_embryos) > 0:
        summary = f"{len(completed_embryos)} embryo{'s have' if len(completed_embryos) != 1 else ' has'} reached their stop condition. {len(active_embryos)} still being imaged."

    return {
        "status": status,
        "headline": headline,
        "summary": summary,
        "details": details,
        "generated_at": datetime.now().isoformat()
    }
