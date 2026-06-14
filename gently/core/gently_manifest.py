"""
Gently manifest — writes the root gently.yaml to a Gently3 storage root.

Called once on first initialization of a new storage root.
"""

from datetime import datetime
from pathlib import Path

import yaml

MANIFEST_VERSION = 3

MANIFEST = {
    "version": MANIFEST_VERSION,
    "created": None,  # filled at runtime
    "description": (
        "Gently microscopy data store. All experimental data is in sessions/. "
        "Agent memory (campaigns, learnings, plans) is in agent/. "
        "Process logs are in logs/. Hardware config is in config/."
    ),
    "structure": {
        "sessions": (
            "One folder per imaging session, named {date}_{time}_{name}_{id}. "
            "Each session contains embryo data (volumes, projections, traces), "
            "agent interaction records, and timelapse state."
        ),
        "agent": (
            "The agent's persistent memory — campaigns, plans, learnings, "
            "observations, and per-embryo understanding. Keyed by uid for "
            "cross-session embryo tracking."
        ),
        "logs": "Process-level logs (not session-scoped). One file per process run.",
        "config": "Instance-level configuration — hardware profile, mesh identity.",
        "incoming": (
            "Transient staging directory for device-to-agent volume transfer. "
            "Files here are moved to sessions/ within seconds."
        ),
        "ml": "ML training runs and data assessments.",
    },
    "conventions": {
        "yaml": "Human-readable metadata and state (hand-editable)",
        "jsonl": "Append-only machine-generated logs and records",
        "json": "Structured data not intended for hand-editing (conversation, traces)",
    },
    "session_folder_pattern": "{YYYYMMDD}_{HHMM}_{slug}_{id8}",
    "embryo_data": {
        "volumes": "t{NNNN}.tif — zlib-compressed TIFF stacks",
        "projections": "t{NNNN}.jpg — max-intensity JPEG projections",
        "traces": "t{NNNN}.json — complete perception record per timepoint",
        "predictions.jsonl": (
            "One-line-per-timepoint summary: {timepoint, stage, confidence, timestamp}"
        ),
        "ground_truth.yaml": "Human annotations: [{stage, start_timepoint, end_timepoint}]",
    },
    "agent_memory": {
        "campaigns": "Research campaigns with plans, templates, version history",
        "learnings": "Individual YAML files — agent-synthesized insights",
        "observations": "Individual YAML files — synthesized experiment notes",
        "active": "Short-lived items: expectations, watchpoints, questions",
        "embryo_understanding": "Per-embryo understanding, keyed by persistent uid",
    },
}


def write_manifest(root: Path):
    """Write gently.yaml to the storage root if it doesn't already exist."""
    manifest_path = root / "gently.yaml"
    if manifest_path.exists():
        return

    data = dict(MANIFEST)
    data["created"] = datetime.now().strftime("%Y-%m-%d")

    with open(manifest_path, "w", encoding="utf-8") as f:
        yaml.dump(
            data,
            f,
            default_flow_style=False,
            sort_keys=False,
            allow_unicode=True,
            width=100,
        )


def read_manifest(root: Path) -> dict:
    """Read the gently.yaml manifest. Returns empty dict if missing."""
    manifest_path = root / "gently.yaml"
    if not manifest_path.exists():
        return {}
    with open(manifest_path, encoding="utf-8") as f:
        return yaml.safe_load(f) or {}
