#!/usr/bin/env python3
"""
Database Export Utilities for Multi-Embryo Calibration
======================================================

This module provides utilities for exporting Bluesky databroker runs
to JSON database format compatible with the original multi_embryo_database.json structure.

The databroker serves as the primary storage during acquisition, and these
utilities enable exporting to JSON format for compatibility with existing
analysis tools and workflows.
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


def format_timestamp(dt: datetime | None = None) -> str:
    """
    Format datetime as ISO 8601 string for JSON storage.

    Parameters
    ----------
    dt : datetime, optional
        Datetime to format. If None, uses current time.

    Returns
    -------
    str
        ISO 8601 formatted timestamp
    """
    if dt is None:
        dt = datetime.now()
    return dt.isoformat()


def numpy_to_python(obj: Any) -> Any:
    """
    Convert numpy types to Python native types for JSON serialization.

    Parameters
    ----------
    obj : Any
        Object to convert (can be numpy array, scalar, or regular Python type)

    Returns
    -------
    Any
        Python native type
    """
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (np.integer, np.floating)):
        return obj.item()
    elif isinstance(obj, np.bool_):
        return bool(obj)
    elif isinstance(obj, dict):
        return {key: numpy_to_python(value) for key, value in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [numpy_to_python(item) for item in obj]
    else:
        return obj


def format_embryo_calibration_for_json(calibration_data: dict) -> dict:
    """
    Format embryo calibration data from databroker for JSON export.

    Converts numpy types to Python native types and ensures consistent
    structure with the original multi_embryo_database.json format.

    Parameters
    ----------
    calibration_data : dict
        Calibration data from databroker run metadata

    Returns
    -------
    dict
        JSON-compatible calibration dictionary
    """
    # Convert all numpy types to Python natives
    calibration_json = numpy_to_python(calibration_data)

    # Ensure required fields exist
    required_fields = [
        "slope_um_per_deg",
        "offset_um",
        "galvo_top_deg",
        "galvo_bottom_deg",
        "piezo_top_um",
        "piezo_bottom_um",
        "sample_type",
        "timestamp",
        "device_piezo",
        "device_galvo",
    ]

    for field in required_fields:
        if field not in calibration_json:
            # Provide sensible defaults for missing fields
            if field == "sample_type":
                calibration_json[field] = "embryo"
            elif field == "timestamp":
                calibration_json[field] = format_timestamp()
            elif "device" in field:
                calibration_json[field] = "unknown"
            else:
                calibration_json[field] = None

    return calibration_json


def format_embryo_entry_for_json(embryo_data: dict) -> dict:
    """
    Format single embryo entry from databroker for JSON export.

    Parameters
    ----------
    embryo_data : dict
        Embryo data from databroker run metadata

    Returns
    -------
    dict
        JSON-compatible embryo entry
    """
    entry = {
        "embryo_number": int(embryo_data.get("embryo_number", 0)),
        "marking_timestamp": embryo_data.get("marking_timestamp", format_timestamp()),
        "bottom_camera_position_pixel": {
            "x": float(embryo_data.get("pixel_x", 0.0)),
            "y": float(embryo_data.get("pixel_y", 0.0)),
        },
        "initial_stage_position_um": {
            "x": float(embryo_data.get("initial_stage_x", 0.0)),
            "y": float(embryo_data.get("initial_stage_y", 0.0)),
        },
        "stage_position_after_centering_um": {
            "x": float(embryo_data.get("centered_stage_x", 0.0)),
            "y": float(embryo_data.get("centered_stage_y", 0.0)),
        },
    }

    # Add calibration data if present
    if "calibration" in embryo_data:
        entry["calibration"] = format_embryo_calibration_for_json(embryo_data["calibration"])

    return entry


def export_multi_embryo_database(
    databroker_catalog, session_uid: str, output_path: Path, pretty_print: bool = True
) -> Path:
    """
    Export multi-embryo calibration data from databroker to JSON database file.

    Queries databroker for all runs associated with a session UID and exports
    them to the multi_embryo_database.json format.

    Parameters
    ----------
    databroker_catalog
        Databroker catalog instance
    session_uid : str
        Session run UID (top-level multi-embryo run)
    output_path : Path
        Output path for JSON file
    pretty_print : bool, optional
        Pretty-print JSON with indentation (default: True)

    Returns
    -------
    Path
        Path to exported JSON file

    Raises
    ------
    KeyError
        If session UID not found in databroker
    ValueError
        If session data is incomplete or invalid
    """
    # Get session run from databroker
    try:
        session_run = databroker_catalog[session_uid]
    except KeyError:
        raise KeyError(f"Session UID {session_uid} not found in databroker") from None

    # Handle different databroker API versions
    try:
        # v2 API
        session_metadata = session_run.metadata["start"]
    except (AttributeError, KeyError):
        # v1 API
        session_metadata = session_run["start"]

    # Initialize database structure
    database = {
        "created": session_metadata.get("time", format_timestamp()),
        "embryos": {},
        "last_updated": format_timestamp(),
    }

    # Get list of embryo run UIDs from session metadata
    embryo_uids = session_metadata.get("embryo_runs", [])

    if not embryo_uids:
        logger.warning("No embryo runs found in session %s...", session_uid[:8])
        logger.warning("Session may still be running or no embryos were calibrated.")

    # Process each embryo run
    for embryo_uid in embryo_uids:
        try:
            embryo_run = databroker_catalog[embryo_uid]

            # Get embryo metadata
            try:
                embryo_metadata = embryo_run.metadata["start"]
            except (AttributeError, KeyError):
                embryo_metadata = embryo_run["start"]

            embryo_id = embryo_metadata.get(
                "embryo_id", f"embryo_{len(database['embryos']) + 1:03d}"
            )

            # Format embryo entry
            embryo_entry = format_embryo_entry_for_json(embryo_metadata)

            # Add to database
            database["embryos"][embryo_id] = embryo_entry

        except Exception as e:
            logger.warning("Could not export embryo %s...: %s", embryo_uid[:8], e)
            continue

    # Write to JSON file
    output_path = Path(output_path)
    with open(output_path, "w") as f:
        if pretty_print:
            json.dump(database, f, indent=2)
        else:
            json.dump(database, f)

    logger.info(
        "Exported multi-embryo database: File=%s, Embryos=%d, Session=%s...",
        output_path,
        len(database["embryos"]),
        session_uid[:8],
    )

    return output_path


def load_multi_embryo_database(database_path: Path) -> dict:
    """
    Load existing multi-embryo database from JSON file.

    Parameters
    ----------
    database_path : Path
        Path to multi_embryo_database.json

    Returns
    -------
    dict
        Database dictionary
    """
    database_path = Path(database_path)

    if not database_path.exists():
        # Return empty database structure
        return {
            "created": format_timestamp(),
            "embryos": {},
            "last_updated": format_timestamp(),
        }

    with open(database_path) as f:
        database = json.load(f)

    return database


def save_multi_embryo_database(database: dict, database_path: Path):
    """
    Save multi-embryo database to JSON file.

    Parameters
    ----------
    database : dict
        Database dictionary
    database_path : Path
        Output path for JSON file
    """
    database_path = Path(database_path)

    # Update last_updated timestamp
    database["last_updated"] = format_timestamp()

    with open(database_path, "w") as f:
        json.dump(database, f, indent=2)


def add_embryo_to_database(database: dict, embryo_id: str, embryo_data: dict) -> dict:
    """
    Add or update embryo entry in database.

    Parameters
    ----------
    database : dict
        Database dictionary
    embryo_id : str
        Embryo identifier (e.g., "embryo_001")
    embryo_data : dict
        Embryo data dictionary

    Returns
    -------
    dict
        Updated database
    """
    # Format embryo entry
    embryo_entry = format_embryo_entry_for_json(embryo_data)

    # Add to database
    database["embryos"][embryo_id] = embryo_entry
    database["last_updated"] = format_timestamp()

    return database


def get_embryo_calibration(database: dict, embryo_id: str) -> dict | None:
    """
    Get calibration data for specific embryo from database.

    Parameters
    ----------
    database : dict
        Database dictionary
    embryo_id : str
        Embryo identifier

    Returns
    -------
    dict or None
        Calibration dictionary, or None if not found
    """
    embryo_entry = database.get("embryos", {}).get(embryo_id)

    if embryo_entry is None:
        return None

    return embryo_entry.get("calibration")


def list_embryos(database: dict) -> list[str]:
    """
    List all embryo IDs in database.

    Parameters
    ----------
    database : dict
        Database dictionary

    Returns
    -------
    list of str
        Embryo IDs sorted by embryo number
    """
    embryos = database.get("embryos", {})

    # Sort by embryo_number
    sorted_embryos = sorted(embryos.items(), key=lambda x: x[1].get("embryo_number", 0))

    return [embryo_id for embryo_id, _ in sorted_embryos]
