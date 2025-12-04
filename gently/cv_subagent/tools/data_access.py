"""
Data Access Tools for CV Agent

Tools for accessing volume data using Gently's central data store.
These tools provide the CV agent access to historical embryo data
through the same data infrastructure used by the rest of the system.

Volume Metadata Schema
======================
When volumes are stored in the data store, they should include:

Required metadata:
- embryo_id : str
    Unique identifier for the embryo (e.g., "embryo_1", "E001")
- timepoint : int
    Timepoint index (0-based, sequential)

Recommended metadata:
- timestamp : str (ISO format)
    When the volume was acquired
- scale_um_per_px : float
    XY pixel size in micrometers (typical: 0.3-0.5)
- z_spacing_um : float
    Z step size in micrometers (typical: 0.5-2.0)
- channel : str
    Imaging channel (e.g., "GFP", "H2B-mCherry")
- view : str
    DiSPIM view ("A" or "B")
- session_id : str
    Experiment session identifier

Example volume storage:
    store.store(
        data=volume_array,
        data_type="volume",
        metadata={
            "embryo_id": "embryo_1",
            "timepoint": 5,
            "scale_um_per_px": 0.406,
            "z_spacing_um": 1.0,
            "channel": "H2B-GFP",
            "session_id": "exp_20241201",
        }
    )
"""

import logging
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .registry import cv_tool, ToolCategory, ToolExample

logger = logging.getLogger(__name__)

# Import Gently's data store
try:
    from gently.core.data_store import get_data_store, DataStore, DataReference
    HAS_DATA_STORE = True
except ImportError:
    HAS_DATA_STORE = False
    logger.warning("Gently data store not available, using fallback")


def _get_store() -> Optional["DataStore"]:
    """Get the global data store"""
    if HAS_DATA_STORE:
        try:
            return get_data_store()
        except Exception as e:
            logger.warning(f"Could not get data store: {e}")
    return None


# =============================================================================
# Embryo Discovery Tools
# =============================================================================

@cv_tool(
    name="list_embryos",
    description="List all embryos available in the data store with their metadata.",
    category=ToolCategory.DATA_ACCESS,
    examples=[
        ToolExample("What embryos are available?", {}),
        ToolExample("List embryos from session exp_001", {"session_id": "exp_001"}),
    ],
)
def list_embryos(
    session_id: Optional[str] = None,
    limit: int = 50,
) -> Dict[str, Any]:
    """
    List all embryos in the data store

    Parameters
    ----------
    session_id : str, optional
        Filter by experiment session
    limit : int
        Maximum embryos to return

    Returns
    -------
    dict
        embryos: List of embryo summaries with timepoint counts
        total: Total embryos found
    """
    logger.info(f"Listing embryos, session={session_id}, limit={limit}")

    store = _get_store()

    if store:
        # Query all volumes
        filters = {}
        if session_id:
            filters["session_id"] = session_id

        refs = store.query(data_type="volume", **filters)

        # Group by embryo_id
        embryo_data = {}
        for ref in refs:
            embryo_id = ref.metadata.get("embryo_id")
            if not embryo_id:
                continue

            if embryo_id not in embryo_data:
                embryo_data[embryo_id] = {
                    "embryo_id": embryo_id,
                    "timepoints": [],
                    "first_timestamp": ref.timestamp,
                    "last_timestamp": ref.timestamp,
                    "session_id": ref.metadata.get("session_id"),
                    "channel": ref.metadata.get("channel"),
                    "scale_um_per_px": ref.metadata.get("scale_um_per_px"),
                }

            tp = ref.metadata.get("timepoint", 0)
            embryo_data[embryo_id]["timepoints"].append(tp)

            if ref.timestamp < embryo_data[embryo_id]["first_timestamp"]:
                embryo_data[embryo_id]["first_timestamp"] = ref.timestamp
            if ref.timestamp > embryo_data[embryo_id]["last_timestamp"]:
                embryo_data[embryo_id]["last_timestamp"] = ref.timestamp

        # Format results
        embryos = []
        for embryo_id, data in embryo_data.items():
            data["timepoints"] = sorted(set(data["timepoints"]))
            data["num_timepoints"] = len(data["timepoints"])
            data["first_timestamp"] = data["first_timestamp"].isoformat()
            data["last_timestamp"] = data["last_timestamp"].isoformat()
            embryos.append(data)

        # Sort by last_timestamp (most recent first)
        embryos.sort(key=lambda x: x["last_timestamp"], reverse=True)

        return {
            "embryos": embryos[:limit],
            "total": len(embryos),
            "returned": min(len(embryos), limit),
        }

    # Fallback: return synthetic embryo list
    return {
        "embryos": [
            {
                "embryo_id": "embryo_1",
                "num_timepoints": 20,
                "timepoints": list(range(20)),
                "session_id": "synthetic",
                "scale_um_per_px": 0.5,
            }
        ],
        "total": 1,
        "returned": 1,
        "source": "synthetic",
    }


@cv_tool(
    name="get_embryo_info",
    description="Get detailed information about a specific embryo.",
    category=ToolCategory.DATA_ACCESS,
    examples=[
        ToolExample("Get info about embryo 1", {"embryo_id": "embryo_1"}),
        ToolExample("Show details for E003", {"embryo_id": "E003"}),
    ],
)
def get_embryo_info(embryo_id: str) -> Dict[str, Any]:
    """
    Get detailed information about an embryo

    Parameters
    ----------
    embryo_id : str
        ID of the embryo

    Returns
    -------
    dict
        Detailed embryo information including all timepoints,
        acquisition parameters, and derived analyses
    """
    logger.info(f"Getting info for embryo {embryo_id}")

    store = _get_store()

    if store:
        # Get all volumes for this embryo
        refs = store.query(data_type="volume", embryo_id=embryo_id)

        if not refs:
            return {"error": f"No data found for embryo {embryo_id}"}

        # Collect metadata
        timepoints = []
        timestamps = []
        shapes = []

        for ref in refs:
            tp = ref.metadata.get("timepoint", 0)
            timepoints.append(tp)
            timestamps.append(ref.timestamp.isoformat())

            # Try to get shape from metadata or by loading
            shape = ref.metadata.get("shape")
            if shape:
                shapes.append(shape)

        # Get related analyses
        analyses = store.query(data_type="analysis", embryo_id=embryo_id)

        # Get segmentation masks
        masks = store.query(data_type="masks", embryo_id=embryo_id)

        return {
            "embryo_id": embryo_id,
            "num_timepoints": len(set(timepoints)),
            "timepoints": sorted(set(timepoints)),
            "time_range": {
                "first": min(timestamps) if timestamps else None,
                "last": max(timestamps) if timestamps else None,
            },
            "acquisition": {
                "scale_um_per_px": refs[0].metadata.get("scale_um_per_px"),
                "z_spacing_um": refs[0].metadata.get("z_spacing_um"),
                "channel": refs[0].metadata.get("channel"),
                "volume_shape": shapes[0] if shapes else None,
            },
            "related_data": {
                "num_analyses": len(analyses),
                "num_masks": len(masks),
            },
            "session_id": refs[0].metadata.get("session_id"),
        }

    # Fallback
    return {
        "embryo_id": embryo_id,
        "num_timepoints": 20,
        "timepoints": list(range(20)),
        "source": "synthetic",
    }


# =============================================================================
# Volume Access Tools
# =============================================================================

@cv_tool(
    name="get_volume",
    description="Load a volume by embryo ID and timepoint. Returns volume_uid for use with other tools.",
    category=ToolCategory.DATA_ACCESS,
    examples=[
        ToolExample("Load the latest volume for embryo 1", {"embryo_id": "embryo_1"}),
        ToolExample("Get timepoint 5 for embryo_2", {"embryo_id": "embryo_2", "timepoint": 5}),
    ],
)
def get_volume(
    embryo_id: str,
    timepoint: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Load volume data for an embryo from the central data store

    Parameters
    ----------
    embryo_id : str
        ID of the embryo
    timepoint : int, optional
        Specific timepoint (None for latest)

    Returns
    -------
    dict
        volume_uid: Unique ID for the loaded volume
        shape: Volume dimensions [Z, Y, X]
        timepoint: The timepoint loaded
        metadata: Additional metadata
    """
    logger.info(f"Loading volume for {embryo_id}, timepoint={timepoint}")

    store = _get_store()

    if store:
        # Query the data store for volumes matching this embryo
        refs = store.query(
            data_type="volume",
            embryo_id=embryo_id,
        )

        if timepoint is not None:
            # Filter by specific timepoint
            refs = [r for r in refs if r.metadata.get("timepoint") == timepoint]
        else:
            # Sort by timepoint descending to get latest
            refs = sorted(refs, key=lambda r: r.metadata.get("timepoint", 0), reverse=True)

        if refs:
            ref = refs[0]
            try:
                volume_data = store.retrieve(ref)

                # Cache for CV tool use
                cache_volume(ref.uid, volume_data, ref.metadata)

                return {
                    "volume_uid": ref.uid,
                    "shape": list(volume_data.shape) if hasattr(volume_data, 'shape') else [],
                    "timepoint": ref.metadata.get("timepoint", timepoint or 0),
                    "dtype": str(volume_data.dtype) if hasattr(volume_data, 'dtype') else "unknown",
                    "metadata": {
                        k: v for k, v in ref.metadata.items()
                        if k != "_data"  # Exclude internal data reference
                    },
                    "source": "data_store",
                }
            except Exception as e:
                logger.warning(f"Could not retrieve volume: {e}")

    # Fall back to synthetic data if no real data found
    logger.info("No volume found in data store, generating synthetic data")
    return _generate_synthetic_volume(embryo_id, timepoint)


@cv_tool(
    name="get_embryo_history",
    description="Get list of available timepoints for an embryo with timestamps.",
    category=ToolCategory.DATA_ACCESS,
    examples=[
        ToolExample("What timepoints are available for embryo 1?", {"embryo_id": "embryo_1"}),
        ToolExample("Get the last 5 timepoints for embryo_2", {"embryo_id": "embryo_2", "last_n": 5}),
    ],
)
def get_embryo_history(
    embryo_id: str,
    last_n: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Get available timepoints for an embryo from the data store

    Parameters
    ----------
    embryo_id : str
        ID of the embryo
    last_n : int, optional
        Return only last N timepoints

    Returns
    -------
    dict
        timepoints: List of available timepoint indices
        timestamps: List of acquisition timestamps
        total: Total number of timepoints
        embryo_id: The embryo ID
    """
    logger.info(f"Getting history for {embryo_id}, last_n={last_n}")

    store = _get_store()
    timepoint_data = []

    if store:
        # Query for all volumes for this embryo
        refs = store.query(
            data_type="volume",
            embryo_id=embryo_id,
        )

        # Collect timepoint info
        for ref in refs:
            tp = ref.metadata.get("timepoint", 0)
            timepoint_data.append({
                "timepoint": tp,
                "timestamp": ref.timestamp.isoformat(),
                "uid": ref.uid,
            })

        # Sort by timepoint
        timepoint_data.sort(key=lambda x: x["timepoint"])

    # Fall back to synthetic if no data
    if not timepoint_data:
        # Simulate having 20 timepoints
        timepoint_data = [
            {"timepoint": i, "timestamp": None, "uid": None}
            for i in range(20)
        ]
        logger.debug("Using synthetic timepoint history")

    # Apply last_n filter
    total = len(timepoint_data)
    if last_n is not None:
        timepoint_data = timepoint_data[-last_n:]

    return {
        "embryo_id": embryo_id,
        "timepoints": [d["timepoint"] for d in timepoint_data],
        "timestamps": [d["timestamp"] for d in timepoint_data],
        "volume_uids": [d["uid"] for d in timepoint_data],
        "total": total,
        "returned": len(timepoint_data),
    }


@cv_tool(
    name="get_latest_volume",
    description="Get the most recent volume for an embryo.",
    category=ToolCategory.DATA_ACCESS,
    examples=[
        ToolExample("Get the latest volume for embryo 1", {"embryo_id": "embryo_1"}),
    ],
)
def get_latest_volume(embryo_id: str) -> Dict[str, Any]:
    """
    Get the latest volume for an embryo

    Parameters
    ----------
    embryo_id : str
        ID of the embryo

    Returns
    -------
    dict
        Same as get_volume but for the latest timepoint
    """
    # Get history to find latest timepoint
    history = get_embryo_history(embryo_id, last_n=1)
    latest_tp = history["timepoints"][-1] if history["timepoints"] else 0

    # Load the latest volume
    return get_volume(embryo_id, timepoint=latest_tp)


@cv_tool(
    name="get_volume_range",
    description="Load multiple consecutive volumes for temporal analysis.",
    category=ToolCategory.DATA_ACCESS,
    examples=[
        ToolExample("Load timepoints 0-4 for embryo 1", {"embryo_id": "embryo_1", "start_timepoint": 0, "end_timepoint": 4}),
        ToolExample("Get volumes 10-15 for tracking", {"embryo_id": "embryo_2", "start_timepoint": 10, "end_timepoint": 15}),
    ],
)
def get_volume_range(
    embryo_id: str,
    start_timepoint: int,
    end_timepoint: int,
) -> Dict[str, Any]:
    """
    Load a range of volumes for temporal analysis

    Parameters
    ----------
    embryo_id : str
        ID of the embryo
    start_timepoint : int
        First timepoint (inclusive)
    end_timepoint : int
        Last timepoint (inclusive)

    Returns
    -------
    dict
        volumes: List of volume info dicts
        loaded: Number of volumes loaded
    """
    logger.info(f"Loading volumes {embryo_id} t{start_timepoint}-{end_timepoint}")

    volumes = []
    for tp in range(start_timepoint, end_timepoint + 1):
        vol_info = get_volume(embryo_id, timepoint=tp)
        if "error" not in vol_info:
            volumes.append(vol_info)
        else:
            logger.warning(f"Could not load t{tp}: {vol_info.get('error')}")

    return {
        "embryo_id": embryo_id,
        "requested_range": [start_timepoint, end_timepoint],
        "volumes": volumes,
        "volume_uids": [v["volume_uid"] for v in volumes],
        "loaded": len(volumes),
        "timepoints_loaded": [v["timepoint"] for v in volumes],
    }


@cv_tool(
    name="query_volumes",
    description="Query volumes from the data store with flexible filters.",
    category=ToolCategory.DATA_ACCESS,
    examples=[
        ToolExample("Find all GFP channel volumes", {"channel": "GFP"}),
        ToolExample("Query volumes from session exp_001 timepoints 5-10", {"session_id": "exp_001", "timepoint_min": 5, "timepoint_max": 10}),
    ],
)
def query_volumes(
    embryo_id: Optional[str] = None,
    session_id: Optional[str] = None,
    channel: Optional[str] = None,
    timepoint_min: Optional[int] = None,
    timepoint_max: Optional[int] = None,
    limit: int = 10,
) -> Dict[str, Any]:
    """
    Query volumes from the data store

    Parameters
    ----------
    embryo_id : str, optional
        Filter by embryo ID
    session_id : str, optional
        Filter by session ID
    channel : str, optional
        Filter by imaging channel
    timepoint_min : int, optional
        Minimum timepoint (inclusive)
    timepoint_max : int, optional
        Maximum timepoint (inclusive)
    limit : int
        Maximum results to return

    Returns
    -------
    dict
        results: List of matching volume references
        total: Total matches found
    """
    store = _get_store()

    if store:
        # Build filters
        filters = {}
        if embryo_id:
            filters["embryo_id"] = embryo_id
        if session_id:
            filters["session_id"] = session_id
        if channel:
            filters["channel"] = channel

        refs = store.query(data_type="volume", **filters)

        # Apply timepoint range filter
        if timepoint_min is not None or timepoint_max is not None:
            filtered_refs = []
            for r in refs:
                tp = r.metadata.get("timepoint", 0)
                if timepoint_min is not None and tp < timepoint_min:
                    continue
                if timepoint_max is not None and tp > timepoint_max:
                    continue
                filtered_refs.append(r)
            refs = filtered_refs

        return {
            "results": [
                {
                    "uid": r.uid,
                    "embryo_id": r.metadata.get("embryo_id"),
                    "timepoint": r.metadata.get("timepoint"),
                    "timestamp": r.timestamp.isoformat(),
                    "session_id": r.metadata.get("session_id"),
                    "channel": r.metadata.get("channel"),
                }
                for r in refs[:limit]
            ],
            "total": len(refs),
            "returned": min(len(refs), limit),
        }

    return {
        "results": [],
        "total": 0,
        "returned": 0,
        "error": "Data store not available",
    }


# =============================================================================
# Analysis Results Storage
# =============================================================================

@cv_tool(
    name="store_analysis_result",
    description="Store an analysis result linked to its source volume.",
    category=ToolCategory.DATA_ACCESS,
    examples=[
        ToolExample("Store segmentation result", {"volume_uid": "vol_abc", "result_type": "segmentation", "result": {"num_cells": 24}}),
        ToolExample("Save classification result", {"volume_uid": "vol_xyz", "result_type": "classification", "result": {"stage": "gastrula", "confidence": 0.92}}),
    ],
)
def store_analysis_result(
    volume_uid: str,
    result_type: str,
    result: Dict[str, Any],
    embryo_id: Optional[str] = None,
    timepoint: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Store an analysis result with provenance

    Parameters
    ----------
    volume_uid : str
        UID of the source volume
    result_type : str
        Type of analysis (e.g., "segmentation", "classification", "tracking")
    result : dict
        Analysis result data
    embryo_id : str, optional
        Embryo ID (will try to get from volume metadata if not provided)
    timepoint : int, optional
        Timepoint (will try to get from volume metadata if not provided)

    Returns
    -------
    dict
        result_uid: UID of stored result
    """
    store = _get_store()

    if store:
        # Get volume metadata for context
        vol_ref = store.get_reference(volume_uid)
        if vol_ref:
            embryo_id = embryo_id or vol_ref.metadata.get("embryo_id")
            timepoint = timepoint or vol_ref.metadata.get("timepoint")

        # Store result with parent link
        ref = store.store(
            data=result,
            data_type="analysis",
            metadata={
                "result_type": result_type,
                "embryo_id": embryo_id,
                "timepoint": timepoint,
                "source_volume_uid": volume_uid,
            },
            parent_uid=volume_uid,
        )

        return {
            "result_uid": ref.uid,
            "result_type": result_type,
            "embryo_id": embryo_id,
            "timepoint": timepoint,
        }

    # Fallback: just return the result
    return {
        "result_uid": None,
        "result_type": result_type,
        "stored": False,
        "error": "Data store not available",
    }


@cv_tool(
    name="get_analysis_results",
    description="Get previous analysis results for an embryo or volume.",
    category=ToolCategory.DATA_ACCESS,
    examples=[
        ToolExample("Get all analyses for embryo 1", {"embryo_id": "embryo_1"}),
        ToolExample("Get segmentation results only", {"embryo_id": "embryo_1", "result_type": "segmentation"}),
    ],
)
def get_analysis_results(
    embryo_id: Optional[str] = None,
    volume_uid: Optional[str] = None,
    result_type: Optional[str] = None,
    limit: int = 10,
) -> Dict[str, Any]:
    """
    Get previous analysis results

    Parameters
    ----------
    embryo_id : str, optional
        Filter by embryo ID
    volume_uid : str, optional
        Filter by source volume
    result_type : str, optional
        Filter by result type (segmentation, classification, etc.)
    limit : int
        Maximum results to return

    Returns
    -------
    dict
        results: List of analysis results
    """
    store = _get_store()

    if store:
        # Build query
        filters = {}
        if embryo_id:
            filters["embryo_id"] = embryo_id
        if result_type:
            filters["result_type"] = result_type

        refs = store.query(data_type="analysis", **filters)

        # Filter by volume_uid if specified
        if volume_uid:
            refs = [r for r in refs if r.parent_uid == volume_uid]

        results = []
        for ref in refs[:limit]:
            try:
                data = store.retrieve(ref)
                results.append({
                    "uid": ref.uid,
                    "result_type": ref.metadata.get("result_type"),
                    "embryo_id": ref.metadata.get("embryo_id"),
                    "timepoint": ref.metadata.get("timepoint"),
                    "timestamp": ref.timestamp.isoformat(),
                    "data": data,
                })
            except Exception as e:
                logger.warning(f"Could not retrieve {ref.uid}: {e}")

        return {
            "results": results,
            "total": len(refs),
            "returned": len(results),
        }

    return {
        "results": [],
        "total": 0,
        "returned": 0,
        "error": "Data store not available",
    }


# =============================================================================
# Volume Cache for CV Tools
# =============================================================================

# In-memory cache for volumes being processed
# This is separate from the data store - just for the current analysis session
_volume_cache: Dict[str, Dict[str, Any]] = {}


def cache_volume(uid: str, data: np.ndarray, metadata: Optional[Dict] = None):
    """
    Cache a volume for use by CV tools

    Parameters
    ----------
    uid : str
        Volume UID
    data : np.ndarray
        Volume data
    metadata : dict, optional
        Associated metadata
    """
    _volume_cache[uid] = {
        "data": data,
        "metadata": metadata or {},
    }


def get_cached_volume(uid: str) -> Optional[np.ndarray]:
    """
    Get volume data from cache

    Parameters
    ----------
    uid : str
        Volume UID

    Returns
    -------
    np.ndarray or None
        Volume data if found
    """
    if uid in _volume_cache:
        return _volume_cache[uid]["data"]

    # Try to get from data store
    store = _get_store()
    if store:
        ref = store.get_reference(uid)
        if ref:
            try:
                data = store.retrieve(ref)
                # Cache it for future access
                cache_volume(uid, data, ref.metadata)
                return data
            except Exception as e:
                logger.warning(f"Could not retrieve {uid}: {e}")

    return None


def get_cached_volume_info(uid: str) -> Optional[Dict[str, Any]]:
    """Get cached volume metadata"""
    if uid in _volume_cache:
        return _volume_cache[uid]
    return None


def clear_cache():
    """Clear the volume cache"""
    _volume_cache.clear()


# =============================================================================
# Synthetic Data Generation (for testing)
# =============================================================================

def _generate_synthetic_volume(
    embryo_id: str,
    timepoint: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Generate synthetic volume data for testing

    Creates a volume with a simulated embryo (ellipsoid with nuclei)
    and caches it for use by other tools.
    """
    import uuid

    # Volume dimensions (typical diSPIM)
    z_size, y_size, x_size = 50, 256, 256

    # Create empty volume
    volume = np.zeros((z_size, y_size, x_size), dtype=np.uint16)

    # Add background noise
    volume += np.random.randint(0, 100, volume.shape, dtype=np.uint16)

    # Create ellipsoid for embryo
    center = (z_size // 2, y_size // 2, x_size // 2)
    radii = (z_size // 3, y_size // 3, x_size // 3)

    z, y, x = np.ogrid[:z_size, :y_size, :x_size]
    ellipsoid = (
        ((z - center[0]) / radii[0]) ** 2 +
        ((y - center[1]) / radii[1]) ** 2 +
        ((x - center[2]) / radii[2]) ** 2
    ) <= 1

    # Add embryo signal
    volume[ellipsoid] += np.random.randint(500, 1500, np.sum(ellipsoid), dtype=np.uint16)

    # Add some "nuclei" (bright spots) - number based on simulated stage
    num_nuclei = np.random.randint(4, 30)
    for _ in range(num_nuclei):
        nz = np.random.randint(center[0] - radii[0]//2, center[0] + radii[0]//2)
        ny = np.random.randint(center[1] - radii[1]//2, center[1] + radii[1]//2)
        nx = np.random.randint(center[2] - radii[2]//2, center[2] + radii[2]//2)

        nucleus_radius = 5
        for dz in range(-nucleus_radius, nucleus_radius+1):
            for dy in range(-nucleus_radius, nucleus_radius+1):
                for dx in range(-nucleus_radius, nucleus_radius+1):
                    if dz**2 + dy**2 + dx**2 <= nucleus_radius**2:
                        pz, py, px = nz + dz, ny + dy, nx + dx
                        if 0 <= pz < z_size and 0 <= py < y_size and 0 <= px < x_size:
                            volume[pz, py, px] = min(65535, volume[pz, py, px] + 2000)

    # Generate UID and cache
    volume_uid = f"synthetic_{uuid.uuid4().hex[:8]}"
    metadata = {
        "embryo_id": embryo_id,
        "timepoint": timepoint or 0,
        "synthetic": True,
        "num_nuclei_approx": num_nuclei,
        "scale_um_per_px": 0.5,
        "z_spacing_um": 1.0,
    }

    cache_volume(volume_uid, volume, metadata)

    return {
        "volume_uid": volume_uid,
        "shape": list(volume.shape),
        "timepoint": timepoint or 0,
        "dtype": str(volume.dtype),
        "metadata": metadata,
        "source": "synthetic",
    }
