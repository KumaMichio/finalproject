"""
/api/cameras — Camera CRUD endpoints.
"""

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from models.database import get_db
from models.schemas import CameraCreate, CameraUpdate, CameraResponse
from services import camera_service

router = APIRouter()


@router.get("/", response_model=list[CameraResponse])
def get_cameras(status: str | None = None, db: Session = Depends(get_db)):
    return camera_service.list_cameras(db, status=status)


@router.get("/{camera_id}", response_model=CameraResponse)
def get_camera(camera_id: str, db: Session = Depends(get_db)):
    camera = camera_service.get_camera(db, camera_id)
    if not camera:
        raise HTTPException(404, f"Camera {camera_id} not found")
    return camera


@router.post("/", response_model=CameraResponse, status_code=201)
def create_camera(data: CameraCreate, db: Session = Depends(get_db)):
    existing = camera_service.get_camera(db, data.id)
    if existing:
        raise HTTPException(409, f"Camera {data.id} already exists")
    return camera_service.create_camera(db, data.model_dump())


@router.put("/{camera_id}", response_model=CameraResponse)
def update_camera(camera_id: str, data: CameraUpdate, db: Session = Depends(get_db)):
    camera = camera_service.update_camera(
        db, camera_id, data.model_dump(exclude_unset=True)
    )
    if not camera:
        raise HTTPException(404, f"Camera {camera_id} not found")
    return camera


@router.delete("/{camera_id}")
def delete_camera(camera_id: str, db: Session = Depends(get_db)):
    ok = camera_service.delete_camera(db, camera_id)
    if not ok:
        raise HTTPException(404, f"Camera {camera_id} not found")
    return {"detail": f"Camera {camera_id} deleted"}
