"""
/api/scenarios — Kich hoat kich ban tai nan trong CARLA de demo
IncidentDetector + AlertSystem (hit-and-run, dam nguoi di bo, v.v.)
"""

from fastapi import APIRouter, HTTPException

from services.ai_processor import get_ai_processor, AIProcessor

router = APIRouter()


@router.get("/")
def list_scenarios():
    return {"scenarios": list(AIProcessor.SCENARIOS.keys())}


@router.post("/{name}")
def trigger_scenario(name: str):
    ai = get_ai_processor()
    if ai is None:
        raise HTTPException(503, "AI processor chua duoc khoi dong (--with-ai)")

    result = ai.trigger_scenario(name)
    if not result["ok"]:
        raise HTTPException(400, result["message"])
    return result
