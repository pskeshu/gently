"""
Ground Truth Manager for Perception Benchmarks.

Loads and queries ground truth labels from transition timepoint format.
"""

import json
from dataclasses import dataclass, field
from pathlib import Path

# Stage progression order
STAGE_ORDER = [
    "early",
    "bean",
    "comma",
    "1.5fold",
    "2fold",
    "pretzel",
    "hatching",
    "hatched",
]


@dataclass
class GroundTruth:
    """
    Ground truth labels for embryo developmental stages.

    Stores stage transitions as {embryo_id: {stage: start_timepoint}}.
    Can query the ground truth stage for any embryo at any timepoint.
    """

    # {embryo_id: {stage: start_timepoint}}
    transitions: dict[str, dict[str, int]] = field(default_factory=dict)

    # Metadata
    session_id: str | None = None
    annotator: str | None = None
    notes: str | None = None

    def get_stage_at(self, embryo_id: str, timepoint: int) -> str | None:
        """
        Get the ground truth stage for a given embryo at a given timepoint.

        Parameters
        ----------
        embryo_id : str
            The embryo identifier (e.g., "embryo_1")
        timepoint : int
            The timepoint to query

        Returns
        -------
        str or None
            The stage at this timepoint, or None if no ground truth available
        """
        if embryo_id not in self.transitions:
            return None

        embryo_transitions = self.transitions[embryo_id]

        # Find the latest stage that started at or before this timepoint
        current_stage = None
        current_start = -1

        for stage, start_tp in embryo_transitions.items():
            if start_tp <= timepoint and start_tp > current_start:
                current_stage = stage
                current_start = start_tp

        return current_stage

    def get_transition_timepoint(self, embryo_id: str, stage: str) -> int | None:
        """Get the timepoint when a stage starts for a given embryo."""
        if embryo_id not in self.transitions:
            return None
        return self.transitions[embryo_id].get(stage)

    def get_stages_for_embryo(self, embryo_id: str) -> list[str]:
        """Get list of stages (in order) for a given embryo."""
        if embryo_id not in self.transitions:
            return []

        embryo_transitions = self.transitions[embryo_id]

        # Sort by start timepoint
        sorted_stages = sorted(embryo_transitions.keys(), key=lambda s: embryo_transitions[s])
        return sorted_stages

    def get_timepoint_range(self, embryo_id: str) -> tuple:
        """
        Get the range of timepoints covered by ground truth for an embryo.

        Returns (min_timepoint, max_timepoint) where max is the last labeled timepoint.
        """
        if embryo_id not in self.transitions:
            return (0, 0)

        starts = list(self.transitions[embryo_id].values())
        if not starts:
            return (0, 0)

        return (min(starts), max(starts))

    @property
    def embryo_ids(self) -> list[str]:
        """Get list of all embryo IDs with ground truth."""
        return list(self.transitions.keys())

    def to_dict(self) -> dict:
        """Serialize to dictionary for JSON storage."""
        return {
            "session_id": self.session_id,
            "annotator": self.annotator,
            "notes": self.notes,
            "transitions": self.transitions,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "GroundTruth":
        """Load from dictionary."""
        return cls(
            transitions=data.get("transitions", {}),
            session_id=data.get("session_id"),
            annotator=data.get("annotator"),
            notes=data.get("notes"),
        )

    @classmethod
    def from_json(cls, path: Path) -> "GroundTruth":
        """Load ground truth from JSON file."""
        with open(path) as f:
            data = json.load(f)
        return cls.from_dict(data)

    def save_json(self, path: Path) -> None:
        """Save ground truth to JSON file."""
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)


def create_ground_truth_from_email_format(
    annotations: dict[str, str],
    session_id: str | None = None,
    annotator: str | None = None,
) -> GroundTruth:
    """
    Create GroundTruth from email-style annotations.

    Parameters
    ----------
    annotations : Dict[str, str]
        Dictionary mapping embryo_id to annotation string.
        E.g., {"embryo_1": "bean 43, comma 49, 1.5-fold 55, 2-fold 70, 3-fold 90"}

    Returns
    -------
    GroundTruth
        Parsed ground truth object
    """
    transitions = {}

    for embryo_id, annotation_str in annotations.items():
        embryo_transitions = {"early": 0}  # All embryos start at early

        # Parse "stage N" pairs
        parts = annotation_str.split(",")
        for part in parts:
            part = part.strip()
            if not part:
                continue

            # Split into stage and timepoint
            tokens = part.split()
            if len(tokens) >= 2:
                stage_name = tokens[0].lower()
                try:
                    timepoint = int(tokens[1])
                except ValueError:
                    continue

                # Normalize stage names
                if stage_name == "1.5-fold":
                    stage_name = "1.5fold"
                elif stage_name == "2-fold":
                    stage_name = "2fold"
                elif stage_name == "3-fold":
                    stage_name = "pretzel"

                embryo_transitions[stage_name] = timepoint

        transitions[embryo_id] = embryo_transitions

    return GroundTruth(
        transitions=transitions,
        session_id=session_id,
        annotator=annotator,
    )
