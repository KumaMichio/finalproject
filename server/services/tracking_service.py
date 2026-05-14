"""
Tracking service — query tracked objects + history.
"""

from datetime import datetime
from typing import Optional

from sqlalchemy.orm import Session

from models.database import TrackedObject, TrackingHistory


# ---- Tracked Objects ----

def list_tracked_objects(
    db: Session,
    status: Optional[str] = None,
    object_class: Optional[str] = None,
    limit: int = 100,
    offset: int = 0,
) -> list[TrackedObject]:
    q = db.query(TrackedObject)
    if status:
        q = q.filter(TrackedObject.status == status)
    if object_class:
        q = q.filter(TrackedObject.object_class == object_class)
    return q.order_by(TrackedObject.last_seen.desc()).offset(offset).limit(limit).all()


def get_tracked_object(db: Session, global_id: int) -> Optional[TrackedObject]:
    return db.query(TrackedObject).filter(TrackedObject.global_id == global_id).first()


def upsert_tracked_object(db: Session, global_id: int, object_class: str,
                           camera_id: str) -> TrackedObject:
    """Tao moi hoac cap nhat tracked object."""
    obj = get_tracked_object(db, global_id)
    now = datetime.utcnow()
    if obj is None:
        obj = TrackedObject(
            global_id=global_id,
            object_class=object_class,
            first_seen=now,
            last_seen=now,
            total_cameras=1,
            status="active",
        )
        db.add(obj)
    else:
        obj.last_seen = now
        obj.status = "active"
    db.commit()
    db.refresh(obj)
    return obj


# ---- Tracking History ----

def add_history(db: Session, global_id: int, camera_id: str,
                frame_number: int, box: list[int],
                confidence: Optional[float] = None) -> TrackingHistory:
    center_x = (box[0] + box[2]) / 2
    center_y = (box[1] + box[3]) / 2
    entry = TrackingHistory(
        global_id=global_id,
        camera_id=camera_id,
        frame_number=frame_number,
        box_x1=box[0], box_y1=box[1], box_x2=box[2], box_y2=box[3],
        center_x=center_x,
        center_y=center_y,
        confidence=confidence,
        timestamp=datetime.utcnow(),
    )
    db.add(entry)
    db.commit()
    return entry


def get_trajectory(db: Session, global_id: int,
                   camera_id: Optional[str] = None,
                   limit: int = 500) -> list[TrackingHistory]:
    q = db.query(TrackingHistory).filter(TrackingHistory.global_id == global_id)
    if camera_id:
        q = q.filter(TrackingHistory.camera_id == camera_id)
    return q.order_by(TrackingHistory.timestamp.asc()).limit(limit).all()


def count_active_objects(db: Session) -> int:
    return db.query(TrackedObject).filter(TrackedObject.status == "active").count()
