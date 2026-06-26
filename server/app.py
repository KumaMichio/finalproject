"""
FastAPI application — entry point cho backend server.

Chay:
    cd AI_custom/server
    uvicorn app:app --host 0.0.0.0 --port 8000 --reload

Chay voi AI pipeline (ket noi CARLA):
    python app.py --with-ai

Chay voi AI pipeline tren video CCTV thuc (khong can CARLA):
    python app.py --with-ai --source file --video-path data/video1.mp4 --video-path data/video2.mp4

Chay voi map/route prediction (can config co section 'route_prediction',
vd camera_config_pho_hue.yaml). --camera-ids PHAI khop dung 'camera_id' khai
bao trong --config (mac dinh khong truyen se la CAM_001, CAM_002... khong
khop voi calibration -> map/route prediction tat am tham, xem ai_processor.py):
    python app.py --with-ai --source file --video-path "<video Pho Hue>.mp4" \
        --config ../custom_tracking_system/config/camera_config_pho_hue.yaml \
        --camera-ids CAM_PHO_HUE_REAL
"""

import asyncio
import logging
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware

from models.database import init_db
from routers import cameras, tracks, alerts, rois, stats, websocket, stream, scenarios, maproute
from config import CAMERA_CONFIG_PATH, CARLA_HOST, CARLA_PORT

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Lifespan — khoi dong / tat server
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(application: FastAPI):
    # app.state, not a module global: "python app.py" runs this file as
    # __main__, while routers/stats.py does `from app import get_uptime` —
    # that import creates a SECOND, separate module object named "app" with
    # its own fresh globals, so a module-level START_TIME set here would
    # never be visible there (get_uptime() in that copy would see the
    # un-set default and return raw time.time() instead of elapsed seconds).
    # app.state lives on the actual FastAPI instance, shared regardless of
    # which module identity imported it.
    application.state.start_time = time.time()

    # Khoi tao database (tao bang neu chua co)
    init_db()

    # Ghi event loop vao FrameBuffer de put_frame() co the signal thread-safe
    from services.stream_service import frame_buffer
    frame_buffer.set_loop(asyncio.get_running_loop())

    # Khoi dong AI processor neu duoc yeu cau (--with-ai flag)
    _ai = None
    if getattr(application.state, "start_ai", False):
        from services.ai_processor import create_ai_processor
        source = getattr(application.state, "source", "carla")
        config_path = getattr(application.state, "config_path", None) or str(CAMERA_CONFIG_PATH)
        _ai = create_ai_processor(
            config_path,
            carla_host=CARLA_HOST,
            carla_port=CARLA_PORT,
            source=source,
            video_paths=getattr(application.state, "video_paths", None),
            camera_ids=getattr(application.state, "camera_ids", None),
            loop_video=getattr(application.state, "loop_video", True),
        )
        _ai.set_event_loop(asyncio.get_running_loop())
        _ai.start()
        logger.info("AI processor started (source=%s)", source)

    yield

    # Cleanup khi shutdown
    if _ai:
        _ai.stop()


def get_uptime(start_time: float) -> float:
    return time.time() - start_time


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Multi-Camera CCTV Tracking System",
    description="Backend API cho he thong giam sat camera AI real-time",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS — cho phep frontend truy cap
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],        # Production: thay bang domain cu the
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Routers
# ---------------------------------------------------------------------------

app.include_router(cameras.router, prefix="/api/cameras", tags=["Cameras"])
app.include_router(tracks.router, prefix="/api/tracks", tags=["Tracks"])
app.include_router(alerts.router, prefix="/api/alerts", tags=["Alerts"])
app.include_router(rois.router, prefix="/api/rois", tags=["ROIs"])
app.include_router(stats.router, prefix="/api/stats", tags=["Stats"])
app.include_router(websocket.router, prefix="/ws", tags=["WebSocket"])
app.include_router(stream.router, prefix="/stream", tags=["Stream"])
app.include_router(scenarios.router, prefix="/api/scenarios", tags=["Scenarios"])
app.include_router(maproute.router, prefix="/api/map", tags=["Map"])


# ---------------------------------------------------------------------------
# Root
# ---------------------------------------------------------------------------

@app.get("/")
async def root(request: Request):
    from services.ai_processor import get_ai_processor

    ai = get_ai_processor()
    ai_status = ai.get_status() if ai else {"status": "not_started"}

    return {
        "name": "Multi-Camera CCTV Tracking System",
        "version": "1.0.0",
        "status": "running",
        "uptime": get_uptime(request.app.state.start_time),
        "ai_pipeline": ai_status,
        "docs": "/docs",
    }


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse
    import uvicorn

    parser = argparse.ArgumentParser(description="Tracking System API Server")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--with-ai", action="store_true",
                        help="Start AI pipeline (requires CARLA server running, "
                             "unless --source file)")
    parser.add_argument("--source", choices=["carla", "file"], default="carla",
                        help="'carla': connect to CARLA (default, hanh vi cu). "
                             "'file': video CCTV thuc qua VideoSource, khong can CARLA.")
    parser.add_argument("--config", default=None,
                        help="Override camera config YAML (mac dinh: CAMERA_CONFIG_PATH trong "
                             "config.py). Dung cho config co section 'route_prediction' "
                             "(map/route prediction tren CCTV thuc, vd camera_config_pho_hue.yaml)")
    parser.add_argument("--video-path", action="append", default=None,
                        help="Path toi 1 video file (--source file). "
                             "Lap lai de co nhieu camera.")
    parser.add_argument("--camera-ids", default=None,
                        help="Danh sach camera ID phan tach bang dau phay, khop thu tu "
                             "voi --video-path (mac dinh: CAM_001, CAM_002, ...)")
    parser.add_argument("--loop-video", action="store_true", default=True,
                        help="Lap lai video khi het (mac dinh: True, --source file)")
    parser.add_argument("--reload", action="store_true")
    args = parser.parse_args()

    # Truyen flag vao app.state de lifespan doc
    app.state.start_ai = args.with_ai
    app.state.source = args.source
    app.state.config_path = args.config
    app.state.video_paths = args.video_path
    app.state.camera_ids = args.camera_ids.split(",") if args.camera_ids else None
    app.state.loop_video = args.loop_video

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    uvicorn.run(
        "app:app" if args.reload else app,
        host=args.host,
        port=args.port,
        reload=args.reload,
    )
