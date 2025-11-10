"""
FastAPI backend for Multi-Embryo Calibration and Volume Acquisition.

Provides REST API and WebSocket endpoints for the frontend.
"""

import sys
from pathlib import Path

# Add parent directory to Python path to import client.py
backend_dir = Path(__file__).parent
parent_dir = backend_dir.parent
sys.path.insert(0, str(parent_dir))

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from typing import List, Optional
import asyncio
from datetime import datetime
import json

from database import (
    init_database, get_session,
    Session, Embryo, Image, VolumeRun, VolumeAcquisition
)
from models import (
    SessionCreate, SessionResponse, EmbryoMarkRequest, EmbryoResponse,
    EmbryoCenterRequest, CaptureImageResponse, HardwareStatusResponse,
    ImageStoreRequest, VolumeRunCreate, VolumeRunResponse, VolumeAcquisitionResponse,
    WSMessage
)
import hardware_control as hw
from volume_service import run_volume_acquisition

# Initialize FastAPI app
app = FastAPI(
    title="Multi-Embryo Calibration API",
    description="Backend for embryo calibration and volume acquisition workflows",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# WebSocket connection manager
class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)
        print(f"✓ WebSocket client connected (total: {len(self.active_connections)})")

    def disconnect(self, websocket: WebSocket):
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)
            print(f"✗ WebSocket client disconnected (total: {len(self.active_connections)})")

    async def broadcast(self, message: dict):
        """Broadcast message to all connected clients."""
        for connection in self.active_connections:
            try:
                await connection.send_json(message)
            except:
                pass

ws_manager = ConnectionManager()


# ============================================================================
# Background Tasks
# ============================================================================

async def run_volume_acquisition_background(
    volume_run_id: int,
    embryo_data: List[dict],
    num_slices: int,
    num_timepoints: int,
    interval_minutes: float,
    output_dir: Path
):
    """
    Background task to run volume acquisition.

    Updates database and broadcasts WebSocket progress.
    """
    db = get_session()
    try:
        # Update status to running
        volume_run = db.query(VolumeRun).filter_by(id=volume_run_id).first()
        if not volume_run:
            print(f"Error: Volume run {volume_run_id} not found")
            return

        volume_run.status = "running"
        db.commit()

        # Broadcast start
        await ws_manager.broadcast({
            "type": "volume_run_status",
            "volume_run_id": volume_run_id,
            "status": "running",
            "message": "Volume acquisition started"
        })

        # Progress callback
        async def progress_callback(message: dict):
            """Send progress updates via WebSocket"""
            await ws_manager.broadcast(message)

        # Run acquisition
        print(f"\n{'='*70}")
        print(f"STARTING VOLUME ACQUISITION - RUN #{volume_run_id}")
        print(f"{'='*70}")
        print(f"  Embryos: {len(embryo_data)}")
        print(f"  Slices: {num_slices}")
        print(f"  Timepoints: {num_timepoints}")
        print(f"  Output: {output_dir}")

        results = await run_volume_acquisition(
            embryos=embryo_data,
            num_slices=num_slices,
            num_timepoints=num_timepoints,
            interval_minutes=interval_minutes,
            output_dir=output_dir,
            progress_callback=progress_callback
        )

        # Store results in database
        for result in results:
            acquisition = VolumeAcquisition(
                volume_run_id=volume_run_id,
                embryo_id=result['embryo_id'],
                timepoint=result['timepoint'],
                success=result['success'],
                filename=result.get('filename'),
                shape_json=list(result['shape']) if result.get('shape') is not None else None,
                error_message=result.get('error')
            )
            db.add(acquisition)

        # Update volume run status
        volume_run.status = "completed"
        volume_run.completed_at = datetime.now()
        db.commit()

        # Broadcast completion
        await ws_manager.broadcast({
            "type": "volume_run_status",
            "volume_run_id": volume_run_id,
            "status": "completed",
            "message": f"Volume acquisition completed: {len([r for r in results if r['success']])}/{len(results)} successful"
        })

        print(f"\n{'='*70}")
        print(f"VOLUME ACQUISITION COMPLETE - RUN #{volume_run_id}")
        print(f"  Successful: {len([r for r in results if r['success']])}/{len(results)}")
        print(f"{'='*70}\n")

    except Exception as e:
        print(f"\nError in volume acquisition background task: {e}")
        import traceback
        traceback.print_exc()

        # Update database
        try:
            volume_run = db.query(VolumeRun).filter_by(id=volume_run_id).first()
            if volume_run:
                volume_run.status = "failed"
                volume_run.completed_at = datetime.now()
                db.commit()
        except:
            pass

        # Broadcast error
        await ws_manager.broadcast({
            "type": "volume_run_status",
            "volume_run_id": volume_run_id,
            "status": "failed",
            "message": f"Volume acquisition failed: {str(e)}"
        })

    finally:
        db.close()


# ============================================================================
# Startup / Shutdown
# ============================================================================

@app.on_event("startup")
async def startup_event():
    """Initialize database on startup."""
    init_database()
    print("✓ FastAPI backend started")


@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "service": "Multi-Embryo Calibration API",
        "version": "1.0.0",
        "status": "running"
    }


@app.get("/api/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat()
    }


# ============================================================================
# Session Endpoints
# ============================================================================

@app.post("/api/sessions", response_model=SessionResponse)
async def create_session(request: SessionCreate):
    """Create a new calibration session."""
    db = get_session()
    try:
        # Check for duplicate name
        existing = db.query(Session).filter_by(name=request.name).first()
        if existing:
            raise HTTPException(status_code=400, detail=f"Session '{request.name}' already exists")

        session = Session(
            name=request.name,
            description=request.description,
            status="active"
        )
        db.add(session)
        db.commit()
        db.refresh(session)

        return SessionResponse(**session.to_dict())
    finally:
        db.close()


@app.get("/api/sessions", response_model=List[SessionResponse])
async def list_sessions(status: Optional[str] = Query(None)):
    """List all sessions, optionally filtered by status."""
    db = get_session()
    try:
        query = db.query(Session)
        if status:
            query = query.filter_by(status=status)

        sessions = query.order_by(Session.created_at.desc()).all()
        return [SessionResponse(**s.to_dict()) for s in sessions]
    finally:
        db.close()


@app.get("/api/sessions/{session_id}", response_model=SessionResponse)
async def get_session_detail(session_id: int):
    """Get detailed information about a session."""
    db = get_session()
    try:
        session = db.query(Session).filter_by(id=session_id).first()
        if not session:
            raise HTTPException(status_code=404, detail="Session not found")

        return SessionResponse(**session.to_dict())
    finally:
        db.close()


@app.put("/api/sessions/{session_id}/status")
async def update_session_status(session_id: int, status: str):
    """Update session status (active or archived)."""
    db = get_session()
    try:
        session = db.query(Session).filter_by(id=session_id).first()
        if not session:
            raise HTTPException(status_code=404, detail="Session not found")

        session.status = status
        db.commit()

        return {"success": True, "session_id": session_id, "status": status}
    finally:
        db.close()


@app.delete("/api/sessions/{session_id}")
async def delete_session(session_id: int):
    """Delete a session and all associated data."""
    db = get_session()
    try:
        session = db.query(Session).filter_by(id=session_id).first()
        if not session:
            raise HTTPException(status_code=404, detail="Session not found")

        db.delete(session)
        db.commit()

        return {"success": True, "session_id": session_id, "message": "Session deleted"}
    finally:
        db.close()


# ============================================================================
# Hardware Endpoints
# ============================================================================

@app.get("/api/hardware/status", response_model=HardwareStatusResponse)
async def get_hardware_status():
    """Get current hardware status."""
    status = hw.get_hardware_status()
    return HardwareStatusResponse(**status)


@app.post("/api/hardware/capture", response_model=CaptureImageResponse)
async def capture_image():
    """Capture image from bottom camera."""
    # Configure camera
    hw.configure_bottom_camera()

    # Capture image
    result = hw.capture_bottom_camera_image()

    if not result["success"]:
        raise HTTPException(status_code=500, detail=result.get("error", "Image capture failed"))

    # Don't include raw array in response
    response_data = {
        "success": result["success"],
        "image": result["image"],
        "shape": list(result["shape"]),
        "stage_position": result["stage_position"],
        "timestamp": result["timestamp"]
    }

    return CaptureImageResponse(**response_data)


# ============================================================================
# Embryo Endpoints
# ============================================================================

@app.post("/api/embryos/mark", response_model=EmbryoResponse)
async def mark_embryo(request: EmbryoMarkRequest):
    """Mark an embryo at a pixel position."""
    db = get_session()
    try:
        # Check session exists
        session = db.query(Session).filter_by(id=request.session_id).first()
        if not session:
            raise HTTPException(status_code=404, detail="Session not found")

        # Create embryo record
        embryo_id_str = f"embryo_{request.embryo_number:03d}"
        embryo = Embryo(
            session_id=request.session_id,
            embryo_number=request.embryo_number,
            embryo_id=embryo_id_str,
            pixel_x=request.pixel_x,
            pixel_y=request.pixel_y,
            stage_x_initial=request.stage_x_initial,
            stage_y_initial=request.stage_y_initial,
            calibration_status="pending"
        )
        db.add(embryo)
        db.commit()
        db.refresh(embryo)

        return EmbryoResponse(**embryo.to_dict())
    finally:
        db.close()


@app.get("/api/embryos", response_model=List[EmbryoResponse])
async def list_embryos(
    session_id: Optional[int] = Query(None),
    include_calibration: bool = Query(True)
):
    """List embryos, optionally filtered by session."""
    db = get_session()
    try:
        query = db.query(Embryo)
        if session_id is not None:
            query = query.filter_by(session_id=session_id)

        embryos = query.order_by(Embryo.embryo_number).all()
        return [EmbryoResponse(**e.to_dict(include_calibration=include_calibration)) for e in embryos]
    finally:
        db.close()


@app.get("/api/embryos/{embryo_id}", response_model=EmbryoResponse)
async def get_embryo(embryo_id: int):
    """Get detailed embryo information."""
    db = get_session()
    try:
        embryo = db.query(Embryo).filter_by(id=embryo_id).first()
        if not embryo:
            raise HTTPException(status_code=404, detail="Embryo not found")

        return EmbryoResponse(**embryo.to_dict(include_calibration=True))
    finally:
        db.close()


@app.post("/api/embryos/{embryo_id}/center")
async def center_embryo(embryo_id: int, request: EmbryoCenterRequest):
    """Move stage to center the embryo."""
    db = get_session()
    try:
        embryo = db.query(Embryo).filter_by(id=embryo_id).first()
        if not embryo:
            raise HTTPException(status_code=404, detail="Embryo not found")

        # Move stage
        result = hw.move_stage_to_center_embryo(
            embryo.pixel_x,
            embryo.pixel_y,
            embryo.stage_x_initial,
            embryo.stage_y_initial,
            (request.image_height, request.image_width)
        )

        if not result["success"]:
            raise HTTPException(status_code=500, detail=result.get("error", "Stage movement failed"))

        # Update embryo record
        embryo.stage_x_centered = result["actual_position"]["x"]
        embryo.stage_y_centered = result["actual_position"]["y"]
        db.commit()

        # Broadcast progress
        await ws_manager.broadcast({
            "type": "embryo_progress",
            "embryo_id": embryo_id,
            "embryo_number": embryo.embryo_number,
            "stage": "centered",
            "stage_position": result["actual_position"]
        })

        return {
            "success": True,
            "embryo_id": embryo_id,
            "target_position": result["target_position"],
            "actual_position": result["actual_position"],
            "displacement_um": result["displacement_um"]
        }
    finally:
        db.close()


@app.post("/api/embryos/{embryo_id}/calibrate")
async def calibrate_embryo(embryo_id: int):
    """Run full calibration workflow for an embryo."""
    db = get_session()
    try:
        embryo = db.query(Embryo).filter_by(id=embryo_id).first()
        if not embryo:
            raise HTTPException(status_code=404, detail="Embryo not found")

        # Update status
        embryo.calibration_status = "calibrating"
        db.commit()

        # Broadcast progress
        await ws_manager.broadcast({
            "type": "embryo_progress",
            "embryo_id": embryo_id,
            "embryo_number": embryo.embryo_number,
            "stage": "calibrating"
        })

        # Run calibration (this will take a while)
        result = hw.run_calibration_for_embryo(embryo.embryo_id)

        if not result["success"]:
            embryo.calibration_status = "failed"
            db.commit()

            await ws_manager.broadcast({
                "type": "calibration_complete",
                "embryo_id": embryo_id,
                "success": False,
                "error": result.get("error")
            })

            raise HTTPException(status_code=500, detail=result.get("error", "Calibration failed"))

        # Store calibration data
        embryo.calibration_data = result["calibration"]
        embryo.calibration_status = "completed"
        db.commit()

        # Broadcast completion
        await ws_manager.broadcast({
            "type": "calibration_complete",
            "embryo_id": embryo_id,
            "embryo_number": embryo.embryo_number,
            "success": True,
            "calibration_data": result["calibration"]
        })

        return {
            "success": True,
            "embryo_id": embryo_id,
            "calibration": result["calibration"]
        }
    finally:
        db.close()


# ============================================================================
# Image Endpoints
# ============================================================================

@app.post("/api/images")
async def store_image(request: ImageStoreRequest):
    """Store an image for an embryo."""
    db = get_session()
    try:
        embryo = db.query(Embryo).filter_by(id=request.embryo_id).first()
        if not embryo:
            raise HTTPException(status_code=404, detail="Embryo not found")

        image = Image(
            embryo_id=request.embryo_id,
            image_type=request.image_type,
            image_data=request.image_data
        )
        db.add(image)
        db.commit()
        db.refresh(image)

        return {"success": True, "image_id": image.id}
    finally:
        db.close()


@app.get("/api/images/{image_id}")
async def get_image(image_id: int):
    """Get image data."""
    db = get_session()
    try:
        image = db.query(Image).filter_by(id=image_id).first()
        if not image:
            raise HTTPException(status_code=404, detail="Image not found")

        return {
            "id": image.id,
            "embryo_id": image.embryo_id,
            "image_type": image.image_type,
            "image_data": image.image_data,
            "timestamp": image.timestamp.isoformat()
        }
    finally:
        db.close()


# ============================================================================
# Volume Run Endpoints
# ============================================================================

@app.post("/api/volumes/runs", response_model=VolumeRunResponse)
async def create_volume_run(request: VolumeRunCreate):
    """Create a new volume acquisition run and start acquisition in background."""
    db = get_session()
    try:
        # Validate session
        session = db.query(Session).filter_by(id=request.session_id).first()
        if not session:
            raise HTTPException(status_code=404, detail="Session not found")

        # Validate embryos exist and are calibrated
        embryo_list = []
        for embryo_id in request.embryo_ids:
            embryo = db.query(Embryo).filter_by(id=embryo_id).first()
            if not embryo:
                raise HTTPException(status_code=404, detail=f"Embryo {embryo_id} not found")
            if embryo.calibration_status != "completed":
                raise HTTPException(status_code=400, detail=f"Embryo {embryo_id} not calibrated")
            if not embryo.calibration_data:
                raise HTTPException(status_code=400, detail=f"Embryo {embryo_id} missing calibration data")
            embryo_list.append(embryo)

        # Create output directory
        output_dir = Path("multi_embryo_volumes") / datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir.mkdir(parents=True, exist_ok=True)

        # Create volume run
        volume_run = VolumeRun(
            session_id=request.session_id,
            name=request.name or f"Volume Run {datetime.now().strftime('%Y-%m-%d %H:%M')}",
            num_slices=request.num_slices,
            num_timepoints=request.num_timepoints,
            interval_minutes=request.interval_minutes,
            output_dir=str(output_dir),
            status="pending",
            started_at=datetime.now()
        )
        db.add(volume_run)
        db.commit()
        db.refresh(volume_run)

        volume_run_id = volume_run.id

        # Prepare embryo data for acquisition
        embryo_data = []
        for embryo in embryo_list:
            embryo_data.append({
                'db_id': embryo.id,
                'embryo_id': embryo.embryo_id,
                'embryo_number': embryo.embryo_number,
                'calibration': embryo.calibration_data,
                'stage_x_centered': embryo.stage_x_centered,
                'stage_y_centered': embryo.stage_y_centered
            })

        # Start acquisition in background
        asyncio.create_task(
            run_volume_acquisition_background(
                volume_run_id,
                embryo_data,
                request.num_slices,
                request.num_timepoints,
                request.interval_minutes,
                output_dir
            )
        )

        return VolumeRunResponse(**volume_run.to_dict())
    except HTTPException:
        raise
    except Exception as e:
        print(f"Error creating volume run: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Failed to create volume run: {str(e)}")
    finally:
        db.close()


@app.get("/api/volumes/runs", response_model=List[VolumeRunResponse])
async def list_volume_runs(session_id: Optional[int] = Query(None)):
    """List volume runs, optionally filtered by session."""
    db = get_session()
    try:
        query = db.query(VolumeRun)
        if session_id is not None:
            query = query.filter_by(session_id=session_id)

        runs = query.order_by(VolumeRun.started_at.desc()).all()
        return [VolumeRunResponse(**r.to_dict()) for r in runs]
    finally:
        db.close()


@app.get("/api/volumes/runs/{run_id}", response_model=VolumeRunResponse)
async def get_volume_run(run_id: int):
    """Get detailed information about a volume run."""
    db = get_session()
    try:
        run = db.query(VolumeRun).filter_by(id=run_id).first()
        if not run:
            raise HTTPException(status_code=404, detail="Volume run not found")

        return VolumeRunResponse(**run.to_dict())
    finally:
        db.close()


# ============================================================================
# WebSocket Endpoints
# ============================================================================

@app.websocket("/ws/calibration")
async def websocket_calibration(websocket: WebSocket):
    """WebSocket endpoint for calibration progress updates."""
    await ws_manager.connect(websocket)
    try:
        while True:
            # Keep connection alive
            data = await websocket.receive_text()
            # Echo heartbeat
            await websocket.send_json({
                "type": "ack",
                "timestamp": datetime.now().timestamp()
            })
    except WebSocketDisconnect:
        ws_manager.disconnect(websocket)


# ============================================================================
# Run Server
# ============================================================================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
