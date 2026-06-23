"""
/api/map — Object marking + long-term route prediction cho map view.

Khac voi /api/tracks (lich su tracking trong DB), router nay tra ve trang
thai REAL-TIME tu AIProcessor (lat/lon, huong re, route du doan) — khong
luu DB, chi ton tai trong RAM trong khi pipeline dang chay.
"""

from fastapi import APIRouter, HTTPException

from services.ai_processor import get_ai_processor

router = APIRouter()


@router.get("/state")
def get_map_state():
    ai = get_ai_processor()
    if ai is None:
        raise HTTPException(503, "AI processor chua duoc khoi dong (--with-ai)")
    return ai.get_map_state()


@router.post("/mark/{global_id}")
def mark_object(global_id: int):
    ai = get_ai_processor()
    if ai is None:
        raise HTTPException(503, "AI processor chua duoc khoi dong (--with-ai)")
    result = ai.mark_object(global_id)
    if not result["ok"]:
        raise HTTPException(400, result["message"])
    return result


@router.post("/unmark")
def unmark_object():
    ai = get_ai_processor()
    if ai is None:
        raise HTTPException(503, "AI processor chua duoc khoi dong (--with-ai)")
    return ai.unmark_object()
