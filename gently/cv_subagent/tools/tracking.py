"""
Cell Tracking Tools for CV Agent

Tools for tracking cells across multiple timepoints and detecting division events.
Essential for lineage tracing in C. elegans embryos.
"""

import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .registry import cv_tool, ToolCategory, ToolExample
from .data_access import get_cached_volume

logger = logging.getLogger(__name__)


@cv_tool(
    name="track_objects",
    description="""Track cells/nuclei across multiple timepoints.

Links objects between consecutive timepoint masks based on centroid proximity.
Detects division events when one object in frame t becomes two in frame t+1.""",
    category=ToolCategory.ANALYSIS,
    examples=[
        ToolExample("Track cells across 5 timepoints", {"mask_uids": ["m1", "m2", "m3", "m4", "m5"]}),
        ToolExample("Track with larger search distance", {"mask_uids": ["m1", "m2", "m3"], "max_distance": 100.0}),
    ],
)
def track_objects(
    mask_uids: List[str],
    max_distance: float = 50.0,
    min_overlap: float = 0.3,
) -> Dict[str, Any]:
    """
    Track objects across timepoints

    Parameters
    ----------
    mask_uids : list
        List of mask UIDs in temporal order
    max_distance : float
        Maximum distance (voxels) for linking objects
    min_overlap : float
        Minimum overlap fraction for linking (0-1)

    Returns
    -------
    dict
        num_tracks: Number of unique tracks
        tracks: List of track dictionaries
        division_events: Detected cell divisions
        cell_counts: Number of cells at each timepoint
    """
    logger.info(f"Tracking objects across {len(mask_uids)} timepoints")

    if len(mask_uids) < 2:
        return {"error": "Need at least 2 timepoints for tracking"}

    # Load all masks and extract centroids
    timepoint_data = []
    for i, uid in enumerate(mask_uids):
        masks = get_cached_volume(uid)
        if masks is None:
            return {"error": f"Masks {uid} not found"}

        centroids = _extract_centroids(masks)
        timepoint_data.append({
            "mask_uid": uid,
            "timepoint": i,
            "num_cells": int(masks.max()),
            "centroids": centroids,
            "masks": masks,
        })

    # Link objects between consecutive timepoints
    all_links = []
    division_events = []
    cell_counts = [td["num_cells"] for td in timepoint_data]

    for t in range(len(timepoint_data) - 1):
        current = timepoint_data[t]
        next_tp = timepoint_data[t + 1]

        links, divisions = _link_timepoints(
            current["centroids"],
            next_tp["centroids"],
            current["masks"],
            next_tp["masks"],
            max_distance=max_distance,
            min_overlap=min_overlap,
        )

        all_links.append({
            "from_timepoint": t,
            "to_timepoint": t + 1,
            "links": links,
        })

        for div in divisions:
            division_events.append({
                "timepoint": t + 1,
                "parent_label": div["parent"],
                "daughter_labels": div["daughters"],
                "parent_centroid": div.get("parent_centroid"),
            })

    # Build tracks from links
    tracks = _build_tracks(all_links, len(timepoint_data))

    logger.info(f"Found {len(tracks)} tracks, {len(division_events)} divisions")

    return {
        "num_tracks": len(tracks),
        "tracks": tracks,
        "division_events": division_events,
        "cell_counts": cell_counts,
        "num_timepoints": len(mask_uids),
        "links": all_links,
    }


@cv_tool(
    name="detect_divisions",
    description="""Detect cell division events between two timepoints.

Analyzes mask changes to identify where one cell became two.""",
    category=ToolCategory.ANALYSIS,
    examples=[
        ToolExample("Check for divisions between t1 and t2", {"masks_uid_t1": "mask_t1", "masks_uid_t2": "mask_t2"}),
    ],
)
def detect_divisions(
    masks_uid_t1: str,
    masks_uid_t2: str,
    min_size_ratio: float = 0.3,
) -> Dict[str, Any]:
    """
    Detect division events between two timepoints

    Parameters
    ----------
    masks_uid_t1 : str
        Masks from earlier timepoint
    masks_uid_t2 : str
        Masks from later timepoint
    min_size_ratio : float
        Minimum daughter/parent size ratio to count as division

    Returns
    -------
    dict
        divisions_detected: Number of division events
        events: List of division event details
        cell_count_change: Change in cell count
    """
    masks_t1 = get_cached_volume(masks_uid_t1)
    masks_t2 = get_cached_volume(masks_uid_t2)

    if masks_t1 is None:
        return {"error": f"Masks {masks_uid_t1} not found"}
    if masks_t2 is None:
        return {"error": f"Masks {masks_uid_t2} not found"}

    n1 = int(masks_t1.max())
    n2 = int(masks_t2.max())
    cell_count_change = n2 - n1

    centroids_t1 = _extract_centroids(masks_t1)
    centroids_t2 = _extract_centroids(masks_t2)

    _, divisions = _link_timepoints(
        centroids_t1, centroids_t2,
        masks_t1, masks_t2,
        max_distance=50.0,
        min_overlap=0.2,
    )

    # Validate divisions by size ratio
    valid_divisions = []
    for div in divisions:
        parent_size = np.sum(masks_t1 == div["parent"])
        daughter_sizes = [np.sum(masks_t2 == d) for d in div["daughters"]]

        # Check if daughters are reasonable size relative to parent
        ratios = [d / parent_size for d in daughter_sizes if parent_size > 0]
        if all(r >= min_size_ratio for r in ratios):
            valid_divisions.append({
                "parent_label": div["parent"],
                "daughter_labels": div["daughters"],
                "parent_size": int(parent_size),
                "daughter_sizes": [int(s) for s in daughter_sizes],
                "size_ratios": [float(r) for r in ratios],
            })

    return {
        "divisions_detected": len(valid_divisions),
        "events": valid_divisions,
        "cell_count_t1": n1,
        "cell_count_t2": n2,
        "cell_count_change": cell_count_change,
        "expected_divisions": max(0, cell_count_change),
    }


@cv_tool(
    name="count_cells_over_time",
    description="Count cells at each timepoint from a list of mask UIDs.",
    category=ToolCategory.ANALYSIS,
    examples=[
        ToolExample("Count cells across 5 timepoints", {"mask_uids": ["m1", "m2", "m3", "m4", "m5"]}),
    ],
)
def count_cells_over_time(mask_uids: List[str]) -> Dict[str, Any]:
    """
    Count cells at each timepoint

    Parameters
    ----------
    mask_uids : list
        List of mask UIDs in temporal order

    Returns
    -------
    dict
        counts: Cell count at each timepoint
        total_increase: Total cell count increase
        division_timepoints: Timepoints where count increased
    """
    counts = []
    for uid in mask_uids:
        masks = get_cached_volume(uid)
        if masks is not None:
            counts.append(int(masks.max()))
        else:
            counts.append(0)

    # Find timepoints where division likely occurred
    division_timepoints = []
    for i in range(1, len(counts)):
        if counts[i] > counts[i-1]:
            division_timepoints.append({
                "timepoint": i,
                "count_before": counts[i-1],
                "count_after": counts[i],
                "increase": counts[i] - counts[i-1],
            })

    return {
        "counts": counts,
        "num_timepoints": len(counts),
        "start_count": counts[0] if counts else 0,
        "end_count": counts[-1] if counts else 0,
        "total_increase": (counts[-1] - counts[0]) if counts else 0,
        "division_timepoints": division_timepoints,
    }


# =============================================================================
# Helper Functions
# =============================================================================

def _extract_centroids(masks: np.ndarray) -> Dict[int, Tuple[float, ...]]:
    """Extract centroids for each labeled object"""
    centroids = {}
    for label in range(1, int(masks.max()) + 1):
        coords = np.argwhere(masks == label)
        if len(coords) > 0:
            centroid = tuple(coords.mean(axis=0))
            centroids[label] = centroid
    return centroids


def _link_timepoints(
    centroids_t1: Dict[int, Tuple],
    centroids_t2: Dict[int, Tuple],
    masks_t1: np.ndarray,
    masks_t2: np.ndarray,
    max_distance: float = 50.0,
    min_overlap: float = 0.3,
) -> Tuple[List[Dict], List[Dict]]:
    """
    Link objects between two timepoints

    Returns (links, divisions)
    """
    links = []
    divisions = []
    used_t2 = set()

    for label1, cent1 in centroids_t1.items():
        # Find potential matches in t2
        candidates = []
        for label2, cent2 in centroids_t2.items():
            if label2 in used_t2:
                continue

            # Calculate distance
            dist = np.sqrt(sum((c1 - c2) ** 2 for c1, c2 in zip(cent1, cent2)))

            if dist <= max_distance:
                # Calculate overlap
                mask1 = masks_t1 == label1
                mask2 = masks_t2 == label2
                overlap = np.sum(mask1 & mask2) / np.sum(mask1) if np.sum(mask1) > 0 else 0

                candidates.append({
                    "label": label2,
                    "distance": dist,
                    "overlap": overlap,
                    "centroid": cent2,
                })

        # Sort by distance
        candidates.sort(key=lambda x: x["distance"])

        if len(candidates) == 1:
            # Simple link
            links.append({
                "from_label": label1,
                "to_label": candidates[0]["label"],
                "distance": candidates[0]["distance"],
            })
            used_t2.add(candidates[0]["label"])

        elif len(candidates) >= 2:
            # Potential division - check if top 2 candidates are close and similar distance
            c0, c1 = candidates[0], candidates[1]

            # If both candidates are within threshold, likely a division
            if c1["distance"] <= max_distance * 1.5:
                divisions.append({
                    "parent": label1,
                    "daughters": [c0["label"], c1["label"]],
                    "parent_centroid": cent1,
                })
                used_t2.add(c0["label"])
                used_t2.add(c1["label"])
            else:
                # Just link to closest
                links.append({
                    "from_label": label1,
                    "to_label": c0["label"],
                    "distance": c0["distance"],
                })
                used_t2.add(c0["label"])

    return links, divisions


def _build_tracks(all_links: List[Dict], num_timepoints: int) -> List[Dict]:
    """Build complete tracks from frame-to-frame links"""
    tracks = []
    track_id = 0

    # Start with all objects in first frame
    if num_timepoints == 0:
        return []

    # Build a graph of links
    forward_links = {}  # (t, label) -> (t+1, label)

    for link_data in all_links:
        t = link_data["from_timepoint"]
        for link in link_data["links"]:
            key = (t, link["from_label"])
            forward_links[key] = (t + 1, link["to_label"])

    # Find track starts (objects in t=0, or objects that appear without parent)
    tracked = set()

    # Start from first timepoint
    for link_data in all_links:
        if link_data["from_timepoint"] == 0:
            for link in link_data["links"]:
                start_label = link["from_label"]
                if (0, start_label) in tracked:
                    continue

                # Follow this track
                track = {
                    "track_id": track_id,
                    "labels": [(0, start_label)],
                }

                current = (0, start_label)
                while current in forward_links:
                    next_pos = forward_links[current]
                    track["labels"].append(next_pos)
                    tracked.add(current)
                    current = next_pos

                tracked.add(current)
                track["start_timepoint"] = 0
                track["end_timepoint"] = track["labels"][-1][0]
                track["duration"] = len(track["labels"])
                tracks.append(track)
                track_id += 1

    return tracks
