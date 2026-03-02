"""Typed dictionary shapes for GentlyStore return values.

Use these for type annotations on store methods and their callers.
"""
from typing import List, Optional, TypedDict


class SessionInfo(TypedDict):
    session_id: str
    name: Optional[str]
    description: Optional[str]
    created_at: str
    last_active: str
    metadata: Optional[dict]


class EmbryoInfo(TypedDict):
    embryo_id: str
    session_id: str
    embryo_uid: Optional[str]
    nickname: Optional[str]
    position_x: Optional[float]
    position_y: Optional[float]
    calibration: Optional[dict]
    created_at: str


class VolumeInfo(TypedDict):
    session_id: str
    embryo_id: str
    timepoint: int
    file_path: str
    shape: Optional[List[int]]
    dtype: Optional[str]
    acquired_at: str
    metadata: Optional[dict]


class ProjectionInfo(TypedDict):
    session_id: str
    embryo_id: str
    timepoint: int
    file_path: str
    width: Optional[int]
    height: Optional[int]
    size_kb: Optional[float]
    created_at: str


class PerceptionRunInfo(TypedDict):
    run_id: int
    session_id: Optional[str]
    name: str
    perception_method: str
    model_name: Optional[str]
    trace_type: str
    source: str
    config: Optional[dict]
    status: str
    created_at: str
    completed_at: Optional[str]
    error_message: Optional[str]


class PredictionInfo(TypedDict):
    prediction_id: int
    run_id: int
    session_id: str
    embryo_id: str
    timepoint: int
    predicted_stage: str
    confidence: Optional[float]
    reasoning: Optional[str]
    is_transitional: int
    ground_truth_stage: Optional[str]
    is_correct: Optional[int]
    execution_time_ms: Optional[float]
    trace_file: Optional[str]
    observed_features: Optional[dict]
    created_at: str


class GroundTruthEntry(TypedDict):
    id: int
    session_id: str
    embryo_id: str
    stage: str
    start_timepoint: int
    end_timepoint: Optional[int]
    annotator: Optional[str]
    notes: Optional[str]
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
