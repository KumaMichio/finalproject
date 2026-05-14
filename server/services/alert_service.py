"""
Alert service — CRUD + filtering.
"""

from datetime import datetime, timedelta
from typing import Optional

from sqlalchemy.orm import Session

from models.database import Alert


def list_alerts(
    db: Session,
    camera_id: Optional[str] = None,
    alert_type: Optional[str] = None,
    severity: Optional[str] = None,
    status: Optional[str] = None,
    hours: Optional[int] = None,
    limit: int = 100,
    offset: int = 0,
) -> list[Alert]:
    q = db.query(Alert)

    if camera_id:
        q = q.filter(Alert.camera_id == camera_id)
    if alert_type:
        q = q.filter(Alert.type == alert_type)
    if severity:
        q = q.filter(Alert.severity == severity)
    if status:
        q = q.filter(Alert.status == status)
    if hours:
        since = datetime.utcnow() - timedelta(hours=hours)
        q = q.filter(Alert.created_at >= since)

    return q.order_by(Alert.created_at.desc()).offset(offset).limit(limit).all()


def get_alert(db: Session, alert_id: int) -> Optional[Alert]:
    return db.query(Alert).filter(Alert.id == alert_id).first()


def create_alert(db: Session, data: dict) -> Alert:
    alert = Alert(**data, created_at=datetime.utcnow())
    db.add(alert)
    db.commit()
    db.refresh(alert)
    return alert


def acknowledge_alert(
    db: Session, alert_id: int, acknowledged_by: str = "operator"
) -> Optional[Alert]:
    alert = get_alert(db, alert_id)
    if not alert:
        return None
    alert.status = "acknowledged"
    alert.acknowledged_by = acknowledged_by
    alert.acknowledged_at = datetime.utcnow()
    db.commit()
    db.refresh(alert)
    return alert


def count_alerts_today(db: Session) -> int:
    today_start = datetime.utcnow().replace(hour=0, minute=0, second=0, microsecond=0)
    return db.query(Alert).filter(Alert.created_at >= today_start).count()
