"""
/api/stats — System statistics endpoints.
"""

from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from models.database import get_db
from models.schemas import SystemStats
from services import alert_service, tracking_service
from services.stream_service import frame_buffer

router = APIRouter()


@router.get("/", response_model=SystemStats)
def get_stats(db: Session = Depends(get_db)):
    from app import get_uptime

    active_cameras = len(frame_buffer.get_camera_ids())
    active_tracks = tracking_service.count_active_objects(db)
    total_alerts_today = alert_service.count_alerts_today(db)

    from services.ai_processor import get_ai_processor
    ai = get_ai_processor()
    fps = ai.fps if ai else 0

    return SystemStats(
        fps=round(fps, 1),
        active_cameras=active_cameras,
        active_tracks=active_tracks,
        total_alerts_today=total_alerts_today,
        uptime_seconds=get_uptime(),
    )
