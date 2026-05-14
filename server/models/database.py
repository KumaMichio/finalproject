"""
Database models — SQLAlchemy ORM
SQLite backend, tao bang tu dong khi khoi dong server.
"""

from datetime import datetime

from sqlalchemy import (
    Column, Integer, Text, Real, Boolean, DateTime, ForeignKey, Index,
    create_engine,
)
from sqlalchemy.orm import declarative_base, sessionmaker, relationship

Base = declarative_base()

# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------

class Camera(Base):
    __tablename__ = "cameras"

    id = Column(Text, primary_key=True)                 # "CAM_001"
    name = Column(Text, nullable=False, default="")
    position_x = Column(Real, default=0)
    position_y = Column(Real, default=0)
    position_z = Column(Real, default=0)
    rotation_pitch = Column(Real, default=0)
    rotation_yaw = Column(Real, default=0)
    rotation_roll = Column(Real, default=0)
    resolution_w = Column(Integer, default=960)
    resolution_h = Column(Integer, default=540)
    fov = Column(Integer, default=90)
    fps = Column(Integer, default=10)
    status = Column(Text, default="active")             # active / inactive / error
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    alerts = relationship("Alert", back_populates="camera")
    rois = relationship("ROI", back_populates="camera")


class Alert(Base):
    __tablename__ = "alerts"

    id = Column(Integer, primary_key=True, autoincrement=True)
    type = Column(Text, nullable=False)                 # ROI_WARNING, OVERSPEED, …
    severity = Column(Text, nullable=False, default="warning")  # info / warning / critical
    global_id = Column(Integer, nullable=True)
    camera_id = Column(Text, ForeignKey("cameras.id"), nullable=True)
    roi_name = Column(Text, nullable=True)
    message = Column(Text, nullable=True)
    details = Column(Text, nullable=True)               # JSON string
    snapshot_path = Column(Text, nullable=True)
    clip_path = Column(Text, nullable=True)
    status = Column(Text, default="new")                # new / acknowledged / resolved
    acknowledged_by = Column(Text, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    acknowledged_at = Column(DateTime, nullable=True)

    camera = relationship("Camera", back_populates="alerts")

    __table_args__ = (
        Index("idx_alerts_time", "created_at"),
        Index("idx_alerts_camera", "camera_id"),
        Index("idx_alerts_type", "type"),
        Index("idx_alerts_severity", "severity"),
    )


class TrackedObject(Base):
    __tablename__ = "tracked_objects"

    global_id = Column(Integer, primary_key=True)
    object_class = Column(Text, nullable=False)         # car, person, bus, truck
    first_seen = Column(DateTime, nullable=False, default=datetime.utcnow)
    last_seen = Column(DateTime, nullable=False, default=datetime.utcnow)
    total_cameras = Column(Integer, default=1)
    status = Column(Text, default="active")             # active / lost / archived

    history = relationship("TrackingHistory", back_populates="tracked_object")


class TrackingHistory(Base):
    __tablename__ = "tracking_history"

    id = Column(Integer, primary_key=True, autoincrement=True)
    global_id = Column(Integer, ForeignKey("tracked_objects.global_id"))
    camera_id = Column(Text, ForeignKey("cameras.id"))
    frame_number = Column(Integer, nullable=True)
    box_x1 = Column(Integer)
    box_y1 = Column(Integer)
    box_x2 = Column(Integer)
    box_y2 = Column(Integer)
    center_x = Column(Real)
    center_y = Column(Real)
    confidence = Column(Real, nullable=True)
    timestamp = Column(DateTime, default=datetime.utcnow)

    tracked_object = relationship("TrackedObject", back_populates="history")

    __table_args__ = (
        Index("idx_history_object", "global_id"),
        Index("idx_history_camera", "camera_id"),
        Index("idx_history_time", "timestamp"),
    )


class ROI(Base):
    __tablename__ = "rois"

    id = Column(Integer, primary_key=True, autoincrement=True)
    camera_id = Column(Text, ForeignKey("cameras.id"))
    name = Column(Text, nullable=False)
    polygon = Column(Text, nullable=False)              # JSON: [[x,y], …]
    alert_types = Column(Text, default="entry")         # entry,exit,loiter,speed
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.utcnow)

    camera = relationship("Camera", back_populates="rois")


# ---------------------------------------------------------------------------
# Engine & Session
# ---------------------------------------------------------------------------

DATABASE_URL = "sqlite:///tracking_system.db"

engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)


def init_db():
    """Tao tat ca bang neu chua ton tai."""
    Base.metadata.create_all(bind=engine)


def get_db():
    """FastAPI dependency — yield 1 session per request."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
