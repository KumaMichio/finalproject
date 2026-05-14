"""
Pydantic schemas — request / response models cho FastAPI.
"""

from __future__ import annotations
from datetime import datetime
from typing import Optional
from pydantic import BaseModel

# ---- Camera ----

class CameraBase(BaseModel):
    id: str
    name: str = ""
    position_x: float = 0
    position_y: float = 0
    position_z: float = 0
    rotation_pitch: float = 0
    rotation_yaw: float = 0
    rotation_roll: float = 0
    resolution_w: int = 960
    resolution_h: int = 540
    fov: int = 90
    fps: int = 10

class CameraCreate(CameraBase):
    pass

class CameraUpdate(BaseModel):
    name: Optional[str] = None
    position_x: Optional[float] = None
    position_y: Optional[float] = None
    position_z: Optional[float] = None
    rotation_pitch: Optional[float] = None
    rotation_yaw: Optional[float] = None
    rotation_roll: Optional[float] = None
    resolution_w: Optional[int] = None
    resolution_h: Optional[int] = None
    fov: Optional[int] = None
    fps: Optional[int] = None

class CameraResponse(CameraBase):
    status: str = "active"
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True

# ---- Alert ----

class AlertCreate(BaseModel):
    type: str
    severity: str = "warning"
    global_id: Optional[int] = None
    camera_id: Optional[str] = None
    roi_name: Optional[str] = None
    message: Optional[str] = None
    details: Optional[str] = None

class AlertResponse(BaseModel):
    id: int
    type: str
    severity: str
    global_id: Optional[int] = None
    camera_id: Optional[str] = None
    roi_name: Optional[str] = None
    message: Optional[str] = None
    details: Optional[str] = None
    snapshot_path: Optional[str] = None
    clip_path: Optional[str] = None
    status: str
    acknowledged_by: Optional[str] = None
    created_at: datetime
    acknowledged_at: Optional[datetime] = None

    class Config:
        from_attributes = True

class AlertAcknowledge(BaseModel):
    acknowledged_by: str = "operator"

# ---- Tracked Object ----

class TrackedObjectResponse(BaseModel):
    global_id: int
    object_class: str
    first_seen: datetime
    last_seen: datetime
    total_cameras: int
    status: str

    class Config:
        from_attributes = True

# ---- Tracking History ----

class TrackingHistoryResponse(BaseModel):
    id: int
    global_id: int
    camera_id: Optional[str] = None
    frame_number: Optional[int] = None
    box_x1: Optional[int] = None
    box_y1: Optional[int] = None
    box_x2: Optional[int] = None
    box_y2: Optional[int] = None
    center_x: Optional[float] = None
    center_y: Optional[float] = None
    confidence: Optional[float] = None
    timestamp: datetime

    class Config:
        from_attributes = True

# ---- ROI ----

class ROICreate(BaseModel):
    camera_id: str
    name: str
    polygon: str          # JSON string: "[[x,y],[x,y],...]"
    alert_types: str = "entry"

class ROIUpdate(BaseModel):
    name: Optional[str] = None
    polygon: Optional[str] = None
    alert_types: Optional[str] = None
    is_active: Optional[bool] = None

class ROIResponse(BaseModel):
    id: int
    camera_id: str
    name: str
    polygon: str
    alert_types: str
    is_active: bool
    created_at: datetime

    class Config:
        from_attributes = True

# ---- Stats ----

class SystemStats(BaseModel):
    fps: float = 0
    active_cameras: int = 0
    active_tracks: int = 0
    total_alerts_today: int = 0
    uptime_seconds: float = 0

# ---- WebSocket messages ----

class WSAlertMessage(BaseModel):
    event: str = "alert"
    data: AlertResponse

class WSTrackMessage(BaseModel):
    event: str = "track_update"
    global_id: int
    camera_id: str
    box: list[int]
    object_class: str
    frame_count: int

class WSStatsMessage(BaseModel):
    event: str = "stats"
    data: SystemStats
