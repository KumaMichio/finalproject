"""
FastAPI application — entry point cho backend server.

Chay:
    cd AI_custom/server
    uvicorn app:app --host 0.0.0.0 --port 8000 --reload

Chay voi AI pipeline (ket noi CARLA):
    python app.py --with-ai
"""

import asyncio
import logging
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from models.database import init_db
from routers import cameras, tracks, alerts, rois, stats, websocket, stream
from config import CAMERA_CONFIG_PATH, CARLA_HOST, CARLA_PORT

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Lifespan — khoi dong / tat server
# ---------------------------------------------------------------------------

START_TIME: float = 0


@asynccontextmanager
async def lifespan(application: FastAPI):
    global START_TIME
    START_TIME = time.time()

    # Khoi tao database (tao bang neu chua co)
    init_db()

    # Khoi dong AI processor neu duoc yeu cau (--with-ai flag)
    _ai = None
    if getattr(application.state, "start_ai", False):
        from services.ai_processor import create_ai_processor
        _ai = create_ai_processor(
            str(CAMERA_CONFIG_PATH),
            carla_host=CARLA_HOST,
            carla_port=CARLA_PORT,
        )
        _ai.set_event_loop(asyncio.get_running_loop())
        _ai.start()
        logger.info("AI processor started with CARLA")

    yield

    # Cleanup khi shutdown
    if _ai:
        _ai.stop()


def get_uptime() -> float:
    return time.time() - START_TIME


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


# ---------------------------------------------------------------------------
# Root
# ---------------------------------------------------------------------------

@app.get("/")
async def root():
    from services.ai_processor import get_ai_processor

    ai = get_ai_processor()
    ai_status = ai.get_status() if ai else {"status": "not_started"}

    return {
        "name": "Multi-Camera CCTV Tracking System",
        "version": "1.0.0",
        "status": "running",
        "uptime": get_uptime(),
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
                        help="Start AI pipeline (requires CARLA server running)")
    parser.add_argument("--reload", action="store_true")
    args = parser.parse_args()

    # Truyen flag vao app.state de lifespan doc
    app.state.start_ai = args.with_ai

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
