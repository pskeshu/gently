"""
Database models for multi-embryo calibration and volume acquisition.

Uses SQLAlchemy ORM with SQLite backend.
"""

from sqlalchemy import create_engine, Column, Integer, String, Float, Boolean, Text, DateTime, ForeignKey, JSON
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship
from datetime import datetime
from pathlib import Path

Base = declarative_base()


class Session(Base):
    """A calibration session represents one experimental sample/session."""
    __tablename__ = "sessions"

    id = Column(Integer, primary_key=True)
    name = Column(String(200), nullable=False, unique=True)
    description = Column(Text, nullable=True)
    created_at = Column(DateTime, default=datetime.now)
    status = Column(String(50), default="active")  # active, archived

    # Relationships
    embryos = relationship("Embryo", back_populates="session", cascade="all, delete-orphan")
    volume_runs = relationship("VolumeRun", back_populates="session", cascade="all, delete-orphan")

    def to_dict(self):
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "status": self.status,
            "num_embryos": len(self.embryos) if self.embryos else 0,
            "num_volume_runs": len(self.volume_runs) if self.volume_runs else 0
        }


class Embryo(Base):
    """An embryo within a session, with calibration data."""
    __tablename__ = "embryos"

    id = Column(Integer, primary_key=True)
    session_id = Column(Integer, ForeignKey("sessions.id"), nullable=False)
    embryo_number = Column(Integer, nullable=False)  # 1-based index within session
    embryo_id = Column(String(100), nullable=False)  # e.g., "embryo_001"

    # Marking data (from bottom camera)
    pixel_x = Column(Float, nullable=True)
    pixel_y = Column(Float, nullable=True)

    # Stage positions
    stage_x_initial = Column(Float, nullable=True)
    stage_y_initial = Column(Float, nullable=True)
    stage_x_centered = Column(Float, nullable=True)
    stage_y_centered = Column(Float, nullable=True)

    # Calibration data (JSON: slope, offset, galvo/piezo params)
    calibration_data = Column(JSON, nullable=True)
    calibration_status = Column(String(50), default="pending")  # pending, calibrating, completed, failed

    # Bluesky integration - link to databroker runs
    calibration_run_uid = Column(String(100), nullable=True)  # UID of Bluesky calibration run in databroker
    queue_item_uid = Column(String(100), nullable=True)  # UID of Queue Server item

    created_at = Column(DateTime, default=datetime.now)

    # Relationships
    session = relationship("Session", back_populates="embryos")
    images = relationship("Image", back_populates="embryo", cascade="all, delete-orphan")
    volume_acquisitions = relationship("VolumeAcquisition", back_populates="embryo")

    def to_dict(self, include_calibration=True):
        result = {
            "id": self.id,
            "session_id": self.session_id,
            "embryo_number": self.embryo_number,
            "embryo_id": self.embryo_id,
            "pixel_position": {
                "x": self.pixel_x,
                "y": self.pixel_y
            } if self.pixel_x is not None else None,
            "stage_position_initial": {
                "x": self.stage_x_initial,
                "y": self.stage_y_initial
            } if self.stage_x_initial is not None else None,
            "stage_position_centered": {
                "x": self.stage_x_centered,
                "y": self.stage_y_centered
            } if self.stage_x_centered is not None else None,
            "calibration_status": self.calibration_status,
            "calibration_run_uid": self.calibration_run_uid,
            "queue_item_uid": self.queue_item_uid,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "num_images": len(self.images) if self.images else 0
        }

        if include_calibration and self.calibration_data:
            result["calibration"] = self.calibration_data

        return result


class Image(Base):
    """Image associated with an embryo (initial, centered, verification)."""
    __tablename__ = "images"

    id = Column(Integer, primary_key=True)
    embryo_id = Column(Integer, ForeignKey("embryos.id"), nullable=False)
    image_type = Column(String(50), nullable=False)  # initial, centered, verification, etc.
    image_data = Column(Text, nullable=False)  # base64 encoded PNG
    timestamp = Column(DateTime, default=datetime.now)

    # Relationships
    embryo = relationship("Embryo", back_populates="images")

    def to_dict(self, include_data=False):
        result = {
            "id": self.id,
            "embryo_id": self.embryo_id,
            "image_type": self.image_type,
            "timestamp": self.timestamp.isoformat() if self.timestamp else None
        }
        if include_data:
            result["image_data"] = self.image_data
        return result


class VolumeRun(Base):
    """A volume acquisition run for multiple embryos in a session."""
    __tablename__ = "volume_runs"

    id = Column(Integer, primary_key=True)
    session_id = Column(Integer, ForeignKey("sessions.id"), nullable=False)
    name = Column(String(200), nullable=True)

    # Acquisition parameters
    num_slices = Column(Integer, nullable=False)
    num_timepoints = Column(Integer, default=1)
    interval_minutes = Column(Float, default=0.0)

    # Status
    status = Column(String(50), default="pending")  # pending, running, completed, failed, cancelled
    started_at = Column(DateTime, nullable=True)
    completed_at = Column(DateTime, nullable=True)

    # Output directory
    output_dir = Column(String(500), nullable=True)

    # Bluesky integration - link to databroker runs
    bluesky_run_uid = Column(String(100), nullable=True)  # UID of Bluesky volume acquisition run in databroker
    queue_item_uid = Column(String(100), nullable=True)  # UID of Queue Server item

    # Relationships
    session = relationship("Session", back_populates="volume_runs")
    acquisitions = relationship("VolumeAcquisition", back_populates="volume_run", cascade="all, delete-orphan")

    def to_dict(self):
        return {
            "id": self.id,
            "session_id": self.session_id,
            "name": self.name,
            "num_slices": self.num_slices,
            "num_timepoints": self.num_timepoints,
            "interval_minutes": self.interval_minutes,
            "status": self.status,
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "output_dir": self.output_dir,
            "bluesky_run_uid": self.bluesky_run_uid,
            "queue_item_uid": self.queue_item_uid,
            "num_acquisitions": len(self.acquisitions) if self.acquisitions else 0,
            "successful_acquisitions": sum(1 for a in self.acquisitions if a.success) if self.acquisitions else 0
        }


class VolumeAcquisition(Base):
    """Individual volume acquisition for one embryo at one timepoint."""
    __tablename__ = "volume_acquisitions"

    id = Column(Integer, primary_key=True)
    volume_run_id = Column(Integer, ForeignKey("volume_runs.id"), nullable=False)
    embryo_id = Column(Integer, ForeignKey("embryos.id"), nullable=False)

    timepoint = Column(Integer, nullable=False)
    success = Column(Boolean, default=False)

    # Output
    filename = Column(String(500), nullable=True)  # Path to TIFF file
    shape_json = Column(JSON, nullable=True)  # [slices, height, width]

    timestamp = Column(DateTime, default=datetime.now)
    error_message = Column(Text, nullable=True)

    # Relationships
    volume_run = relationship("VolumeRun", back_populates="acquisitions")
    embryo = relationship("Embryo", back_populates="volume_acquisitions")

    def to_dict(self):
        return {
            "id": self.id,
            "volume_run_id": self.volume_run_id,
            "embryo_id": self.embryo_id,
            "embryo_number": self.embryo.embryo_number if self.embryo else None,
            "timepoint": self.timepoint,
            "success": self.success,
            "filename": self.filename,
            "shape": self.shape_json,
            "timestamp": self.timestamp.isoformat() if self.timestamp else None,
            "error_message": self.error_message
        }


# Database connection and session management
DATABASE_PATH = Path("embryo_calibration.db")

def get_engine():
    """Create SQLAlchemy engine."""
    return create_engine(f"sqlite:///{DATABASE_PATH}", echo=False)

def init_database():
    """Initialize database schema."""
    engine = get_engine()
    Base.metadata.create_all(engine)
    print(f"✓ Database initialized: {DATABASE_PATH}")

def get_session():
    """Get database session."""
    engine = get_engine()
    SessionLocal = sessionmaker(bind=engine)
    return SessionLocal()
