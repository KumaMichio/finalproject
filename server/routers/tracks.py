"""
/api/tracks — Tracked objects + trajectory endpoints.
"""

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from models.database import get_db
from models.schemas import TrackedObjectResponse, TrackingHistoryResponse
from services import tracking_service

router = APIRouter()


@router.get("/", response_model=list[TrackedObjectResponse])
def get_tracks(
    status: str | None = None,
    object_class: str | None = None,
    limit: int = 100,
    offset: int = 0,
    db: Session = Depends(get_db),
):
    return tracking_service.list_tracked_objects(
        db, status=status, object_class=object_class, limit=limit, offset=offset
    )


@router.get("/{global_id}", response_model=TrackedObjectResponse)
def get_track(global_id: int, db: Session = Depends(get_db)):
    obj = tracking_service.get_tracked_object(db, global_id)
    if not obj:
        raise HTTPException(404, f"Object {global_id} not found")
    return obj


@router.get("/{global_id}/trajectory", response_model=list[TrackingHistoryResponse])
def get_trajectory(
    global_id: int,
    camera_id: str | None = None,
    limit: int = 500,
    db: Session = Depends(get_db),
):
    obj = tracking_service.get_tracked_object(db, global_id)
    if not obj:
        raise HTTPException(404, f"Object {global_id} not found")
    return tracking_service.get_trajectory(db, global_id, camera_id=camera_id, limit=limit)
