"""
/api/alerts — Alert CRUD + acknowledge endpoints.
"""

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from models.database import get_db
from models.schemas import AlertResponse, AlertAcknowledge
from services import alert_service

router = APIRouter()


@router.get("/", response_model=list[AlertResponse])
def get_alerts(
    camera_id: str | None = None,
    type: str | None = None,
    severity: str | None = None,
    status: str | None = None,
    hours: int | None = None,
    limit: int = 100,
    offset: int = 0,
    db: Session = Depends(get_db),
):
    return alert_service.list_alerts(
        db,
        camera_id=camera_id,
        alert_type=type,
        severity=severity,
        status=status,
        hours=hours,
        limit=limit,
        offset=offset,
    )


@router.get("/{alert_id}", response_model=AlertResponse)
def get_alert(alert_id: int, db: Session = Depends(get_db)):
    alert = alert_service.get_alert(db, alert_id)
    if not alert:
        raise HTTPException(404, f"Alert {alert_id} not found")
    return alert


@router.put("/{alert_id}/acknowledge", response_model=AlertResponse)
def acknowledge_alert(
    alert_id: int,
    body: AlertAcknowledge,
    db: Session = Depends(get_db),
):
    alert = alert_service.acknowledge_alert(
        db, alert_id, acknowledged_by=body.acknowledged_by
    )
    if not alert:
        raise HTTPException(404, f"Alert {alert_id} not found")
    return alert
