"""
/api/rois — Region of Interest CRUD endpoints.
"""

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from models.database import get_db, ROI
from models.schemas import ROICreate, ROIUpdate, ROIResponse

router = APIRouter()


def _hot_reload_rois():
    """Day thay doi ROI vao pipeline dang chay (khong can restart)."""
    try:
        from services.ai_processor import get_ai_processor
        ai = get_ai_processor()
        if ai is not None:
            ai.reload_rois_from_db()
    except Exception:
        # Khong chan API neu pipeline chua chay / loi day nong
        pass


@router.get("/", response_model=list[ROIResponse])
def get_rois(camera_id: str | None = None, db: Session = Depends(get_db)):
    q = db.query(ROI)
    if camera_id:
        q = q.filter(ROI.camera_id == camera_id)
    return q.order_by(ROI.camera_id, ROI.name).all()


@router.get("/{roi_id}", response_model=ROIResponse)
def get_roi(roi_id: int, db: Session = Depends(get_db)):
    roi = db.query(ROI).filter(ROI.id == roi_id).first()
    if not roi:
        raise HTTPException(404, f"ROI {roi_id} not found")
    return roi


@router.post("/", response_model=ROIResponse, status_code=201)
def create_roi(data: ROICreate, db: Session = Depends(get_db)):
    roi = ROI(**data.model_dump())
    db.add(roi)
    db.commit()
    db.refresh(roi)
    _hot_reload_rois()
    return roi


@router.put("/{roi_id}", response_model=ROIResponse)
def update_roi(roi_id: int, data: ROIUpdate, db: Session = Depends(get_db)):
    roi = db.query(ROI).filter(ROI.id == roi_id).first()
    if not roi:
        raise HTTPException(404, f"ROI {roi_id} not found")
    for key, value in data.model_dump(exclude_unset=True).items():
        if value is not None:
            setattr(roi, key, value)
    db.commit()
    db.refresh(roi)
    _hot_reload_rois()
    return roi


@router.delete("/{roi_id}")
def delete_roi(roi_id: int, db: Session = Depends(get_db)):
    roi = db.query(ROI).filter(ROI.id == roi_id).first()
    if not roi:
        raise HTTPException(404, f"ROI {roi_id} not found")
    db.delete(roi)
    db.commit()
    _hot_reload_rois()
    return {"detail": f"ROI {roi_id} deleted"}
