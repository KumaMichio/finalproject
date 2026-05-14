"""
Camera service — CRUD operations + CARLA integration.
"""

from datetime import datetime
from typing import Optional

from sqlalchemy.orm import Session

from models.database import Camera


def list_cameras(db: Session, status: Optional[str] = None) -> list[Camera]:
    q = db.query(Camera)
    if status:
        q = q.filter(Camera.status == status)
    return q.order_by(Camera.id).all()


def get_camera(db: Session, camera_id: str) -> Optional[Camera]:
    return db.query(Camera).filter(Camera.id == camera_id).first()


def create_camera(db: Session, data: dict) -> Camera:
    camera = Camera(**data, created_at=datetime.utcnow(), updated_at=datetime.utcnow())
    db.add(camera)
    db.commit()
    db.refresh(camera)
    return camera


def update_camera(db: Session, camera_id: str, data: dict) -> Optional[Camera]:
    camera = get_camera(db, camera_id)
    if not camera:
        return None
    for key, value in data.items():
        if value is not None:
            setattr(camera, key, value)
    camera.updated_at = datetime.utcnow()
    db.commit()
    db.refresh(camera)
    return camera


def delete_camera(db: Session, camera_id: str) -> bool:
    camera = get_camera(db, camera_id)
    if not camera:
        return False
    db.delete(camera)
    db.commit()
    return True
