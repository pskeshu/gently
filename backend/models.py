"""
Pydantic models for API request/response validation.
"""

from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from datetime import datetime


# ============================================================================
# Session Models
# ============================================================================

class SessionCreate(BaseModel):
    name: str = Field(..., min_length=1, max_length=200, description="Session name")
    description: Optional[str] = Field(None, description="Optional description")


class SessionResponse(BaseModel):
    id: int
    name: str
    description: Optional[str]
    created_at: str
    status: str
    num_embryos: int
    num_volume_runs: int


# ============================================================================
# Embryo Models
# ============================================================================

class EmbryoMarkRequest(BaseModel):
    session_id: int
    embryo_number: int
    pixel_x: float
    pixel_y: float
    stage_x_initial: float
    stage_y_initial: float


class EmbryoCenterRequest(BaseModel):
    image_height: int
    image_width: int


class EmbryoResponse(BaseModel):
    id: int
    session_id: int
    embryo_number: int
    embryo_id: str
    pixel_position: Optional[Dict[str, float]]
    stage_position_initial: Optional[Dict[str, float]]
    stage_position_centered: Optional[Dict[str, float]]
    calibration_status: str
    created_at: str
    num_images: int
    calibration: Optional[Dict[str, Any]] = None


# ============================================================================
# Image Models
# ============================================================================

class ImageStoreRequest(BaseModel):
    embryo_id: int
    image_type: str
    image_data: str


class ImageResponse(BaseModel):
    id: int
    embryo_id: int
    image_type: str
    timestamp: str
    image_data: Optional[str] = None


# ============================================================================
# Volume Models
# ============================================================================

class VolumeRunCreate(BaseModel):
    session_id: int
    name: Optional[str] = None
    embryo_ids: List[int] = Field(..., description="List of embryo IDs to acquire")
    num_slices: int = Field(50, ge=1, le=500, description="Number of slices per volume")
    num_timepoints: int = Field(1, ge=1, le=1000, description="Number of timepoints")
    interval_minutes: float = Field(0.0, ge=0.0, description="Interval between timepoints (minutes)")


class VolumeRunResponse(BaseModel):
    id: int
    session_id: int
    name: Optional[str]
    num_slices: int
    num_timepoints: int
    interval_minutes: float
    status: str
    started_at: Optional[str]
    completed_at: Optional[str]
    output_dir: Optional[str]
    num_acquisitions: int
    successful_acquisitions: int


class VolumeAcquisitionResponse(BaseModel):
    id: int
    volume_run_id: int
    embryo_id: int
    embryo_number: Optional[int]
    timepoint: int
    success: bool
    filename: Optional[str]
    shape: Optional[List[int]]
    timestamp: str
    error_message: Optional[str]


# ============================================================================
# Hardware Models
# ============================================================================

class HardwareStatusResponse(BaseModel):
    connected: bool
    stage_position: Dict[str, float]
    bottom_camera: str
    spim_camera: str
    timestamp: float
    error: Optional[str] = None


class CaptureImageResponse(BaseModel):
    success: bool
    image: Optional[str] = None  # base64 encoded
    shape: Optional[List[int]] = None
    stage_position: Optional[Dict[str, float]] = None
    timestamp: Optional[float] = None
    error: Optional[str] = None


# ============================================================================
# WebSocket Message Models
# ============================================================================

class WSMessage(BaseModel):
    """Base WebSocket message"""
    type: str
    timestamp: float = Field(default_factory=lambda: datetime.now().timestamp())


class WSEmbryoProgress(WSMessage):
    """Progress update for embryo calibration/acquisition"""
    type: str = "embryo_progress"
    embryo_id: int
    embryo_number: int
    total_embryos: int
    stage: str  # marking, centering, calibrating, moving, configuring, acquiring, saving


class WSSliceProgress(WSMessage):
    """Progress update for slice acquisition"""
    type: str = "slice_progress"
    current_slice: int
    total_slices: int
    percentage: float


class WSTimepointComplete(WSMessage):
    """Timepoint completion notification"""
    type: str = "timepoint_complete"
    timepoint: int
    total_timepoints: int
    next_timepoint_at: Optional[str] = None


class WSCalibrationComplete(WSMessage):
    """Calibration completion notification"""
    type: str = "calibration_complete"
    embryo_id: int
    success: bool
    calibration_data: Optional[Dict[str, Any]] = None
    error: Optional[str] = None


class WSVolumeRunStatus(WSMessage):
    """Volume run status update"""
    type: str = "volume_run_status"
    volume_run_id: int
    status: str  # running, completed, failed, cancelled
    message: str


class WSError(WSMessage):
    """Error notification"""
    type: str = "error"
    error: str
    details: Optional[str] = None
