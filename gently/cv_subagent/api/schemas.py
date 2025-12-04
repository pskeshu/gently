"""
Pydantic schemas for CV Subagent API

Defines request and response models for the intent-based CV analysis API.
"""

from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Union

from pydantic import BaseModel, Field


# =============================================================================
# Enums
# =============================================================================

class TaskStatusEnum(str, Enum):
    """Task execution status"""
    QUEUED = "queued"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class TaskPriorityEnum(str, Enum):
    """Task priority levels"""
    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"
    URGENT = "urgent"


# =============================================================================
# Request Models
# =============================================================================

class AnalyzeRequest(BaseModel):
    """
    Request for CV analysis

    The CV subagent receives high-level intent and determines
    which tools to use autonomously.
    """
    intent: str = Field(
        ...,
        description="High-level intent describing what to analyze",
        examples=[
            "classify embryo stage",
            "count cells and track divisions",
            "detect developmental anomalies",
        ],
    )
    embryo_id: str = Field(
        ...,
        description="ID of the embryo to analyze",
        examples=["embryo_1", "embryo_2"],
    )
    timepoints: Optional[List[int]] = Field(
        None,
        description="Specific timepoints to analyze (optional)",
        examples=[[0, 1, 2, 3, 4]],
    )
    volume_uids: Optional[List[str]] = Field(
        None,
        description="Specific volume UIDs to analyze (optional)",
    )
    context: Optional[Dict[str, Any]] = Field(
        default_factory=dict,
        description="Additional context for the agent",
        examples=[{
            "current_stage": "4-cell",
            "experiment_goal": "tracking early divisions",
        }],
    )
    priority: TaskPriorityEnum = Field(
        TaskPriorityEnum.NORMAL,
        description="Task priority",
    )
    callback_event: Optional[str] = Field(
        None,
        description="Event type to publish when complete",
    )


# =============================================================================
# Response Models
# =============================================================================

class AnalyzeResponse(BaseModel):
    """Response from analysis submission"""
    task_id: str = Field(
        ...,
        description="Unique task identifier",
    )
    status: TaskStatusEnum = Field(
        ...,
        description="Current task status",
    )
    plan: List[str] = Field(
        default_factory=list,
        description="Agent's execution plan",
        examples=[[
            "1. Load latest volume for embryo_1",
            "2. Detect embryo ROI",
            "3. Run Cellpose segmentation",
            "4. Classify developmental stage",
        ]],
    )
    estimated_time_seconds: Optional[float] = Field(
        None,
        description="Estimated processing time",
    )


class TaskStatus(BaseModel):
    """Detailed task status"""
    task_id: str
    status: TaskStatusEnum
    created_at: datetime
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    progress: Optional[float] = Field(
        None,
        description="Progress percentage (0-100)",
    )
    current_step: Optional[str] = Field(
        None,
        description="Current processing step",
    )
    result: Optional[Dict[str, Any]] = Field(
        None,
        description="Analysis result (when completed)",
    )
    error: Optional[str] = Field(
        None,
        description="Error message (when failed)",
    )
    plan: List[str] = Field(
        default_factory=list,
        description="Execution plan",
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Task metadata",
    )


class TaskListResponse(BaseModel):
    """Response containing list of tasks"""
    tasks: List[TaskStatus]
    total: int
    limit: int
    offset: int


class TaskQueueStats(BaseModel):
    """Task queue statistics"""
    active_tasks: int
    queued_tasks: int
    completed_tasks: int
    failed_tasks: int
    max_concurrent: int


class ServiceStatus(BaseModel):
    """CV Subagent service status"""
    name: str
    version: str
    state: str
    host: str
    port: int
    gpu_available: bool
    gpu_memory_used_mb: Optional[float] = None
    gpu_memory_total_mb: Optional[float] = None
    models_loaded: List[str] = Field(default_factory=list)
    task_queue: TaskQueueStats
    capabilities: List[str] = Field(default_factory=list)
    uptime_seconds: Optional[float] = None


class HealthResponse(BaseModel):
    """Health check response"""
    name: str
    state: str
    healthy: bool
    uptime_seconds: float
    task_queue: Dict[str, int]
    gpu_available: bool


# =============================================================================
# Analysis Result Models
# =============================================================================

class CellInfo(BaseModel):
    """Information about a detected cell"""
    cell_id: int
    centroid: List[float] = Field(
        ...,
        description="[z, y, x] coordinates",
    )
    volume_voxels: int
    volume_um3: Optional[float] = None
    mean_intensity: Optional[float] = None
    bbox: Optional[List[int]] = Field(
        None,
        description="Bounding box [z1, y1, x1, z2, y2, x2]",
    )


class SegmentationResult(BaseModel):
    """Result from 3D segmentation"""
    num_cells: int
    cells: List[CellInfo]
    mask_uid: Optional[str] = Field(
        None,
        description="UID of saved mask volume",
    )
    processing_time_ms: float


class StageClassification(BaseModel):
    """Developmental stage classification result"""
    predicted_stage: str = Field(
        ...,
        description="Predicted developmental stage",
        examples=["gastrula", "comma", "2-fold"],
    )
    confidence: float = Field(
        ...,
        ge=0,
        le=1,
        description="Confidence score",
    )
    alternative_stages: List[Dict[str, Union[str, float]]] = Field(
        default_factory=list,
        description="Alternative stage predictions with confidence",
    )
    evidence: Dict[str, Any] = Field(
        default_factory=dict,
        description="Evidence used for classification",
        examples=[{
            "nuclei_count": 24,
            "elongation_ratio": 1.8,
            "visual_assessment": "gastrulation visible",
        }],
    )


class DivisionEvent(BaseModel):
    """Cell division event"""
    timepoint: int
    parent_cell_id: int
    child_cell_ids: List[int]
    confidence: float


class TrackingResult(BaseModel):
    """Cell tracking result"""
    num_tracks: int
    division_events: List[DivisionEvent]
    cell_count_progression: List[int] = Field(
        ...,
        description="Cell count at each timepoint",
    )
    tracks: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Individual cell tracks",
    )


class AnalysisResult(BaseModel):
    """Complete analysis result"""
    task_id: str
    intent: str
    embryo_id: str
    completed_at: datetime

    # Results (populated based on analysis type)
    segmentation: Optional[SegmentationResult] = None
    stage_classification: Optional[StageClassification] = None
    tracking: Optional[TrackingResult] = None

    # Summary
    summary: str = Field(
        ...,
        description="Human-readable summary of the analysis",
    )

    # Metadata
    processing_time_ms: float
    tools_used: List[str] = Field(
        default_factory=list,
        description="Tools used during analysis",
    )
