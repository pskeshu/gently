"""Typed dictionary shapes used by FileStore and the legacy GentlyStore.

Use these for type annotations on store methods and their callers.
"""

from typing import TypedDict


class SessionInfo(TypedDict):
    session_id: str
    name: str | None
    description: str | None
    created_at: str
    last_active: str
    metadata: dict | None


class EmbryoInfo(TypedDict, total=False):
    embryo_id: str
    session_id: str
    embryo_uid: str | None
    nickname: str | None
    # Coarse XY (µm) from bottom-camera detection or manual map placement.
    # Shape: {"x": float, "y": float}. Always present once the embryo exists.
    position_coarse: dict | None
    # Fine XY (µm) from SPIM-objective alignment. None until that workflow
    # refines the coarse position. Shape: {"x": float, "y": float}.
    position_fine: dict | None
    # Legacy flat fields. Still accepted on write and surfaced on read for
    # callers that haven't migrated; new code should use position_coarse.
    position_x: float | None
    position_y: float | None
    calibration: dict | None
    role: str | None  # key into gently.harness.roles.REGISTRY
    strain: str | None  # free-form biological sample descriptor, e.g. "pan-nuclear GFP"
    created_at: str


class VolumeInfo(TypedDict):
    session_id: str
    embryo_id: str
    timepoint: int
    file_path: str
    shape: list[int] | None
    dtype: str | None
    acquired_at: str
    metadata: dict | None


class ProjectionInfo(TypedDict):
    session_id: str
    embryo_id: str
    timepoint: int
    file_path: str
    width: int | None
    height: int | None
    size_kb: float | None
    created_at: str


class PerceptionRunInfo(TypedDict):
    run_id: int
    session_id: str | None
    name: str
    perception_method: str
    model_name: str | None
    trace_type: str
    source: str
    config: dict | None
    status: str
    created_at: str
    completed_at: str | None
    error_message: str | None


class PredictionInfo(TypedDict):
    prediction_id: int
    run_id: int
    session_id: str
    embryo_id: str
    timepoint: int
    predicted_stage: str
    confidence: float | None
    reasoning: str | None
    is_transitional: int
    ground_truth_stage: str | None
    is_correct: int | None
    execution_time_ms: float | None
    trace_file: str | None
    observed_features: dict | None
    created_at: str


class GroundTruthEntry(TypedDict):
    id: int
    session_id: str
    embryo_id: str
    stage: str
    start_timepoint: int
    end_timepoint: int | None
    annotator: str | None
    notes: str | None
    created_at: str


class StoreStats(TypedDict):
    sessions: int
    embryos: int
    volumes: int
    projections: int
    perception_runs: int
    predictions: int
    ground_truth: int
    disk_usage_mb: float
    db_size_mb: float
