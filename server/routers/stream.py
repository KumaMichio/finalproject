"""
/stream — MJPEG video streaming endpoints.

Su dung:
  Browser: <img src="http://host:8000/stream/CAM_001">
  Hoac fetch tu JavaScript de hien thi live feed.
"""

import asyncio
import logging

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse

from services.stream_service import frame_buffer
from config import STREAM_FPS

logger = logging.getLogger(__name__)
router = APIRouter()


async def _mjpeg_generator(camera_id: str):
    """Yield MJPEG frames cho 1 camera."""
    interval = 1.0 / STREAM_FPS
    while True:
        jpeg_bytes = await frame_buffer.wait_for_frame(camera_id, timeout=2.0)
        if jpeg_bytes is None:
            # Gui 1 frame trong (1x1 pixel) de giu connection song
            jpeg_bytes = (
                b'\xff\xd8\xff\xe0\x00\x10JFIF\x00\x01\x01\x00\x00\x01\x00\x01\x00\x00'
                b'\xff\xdb\x00C\x00\x08\x06\x06\x07\x06\x05\x08\x07\x07\x07\t\t'
                b'\x08\n\x0c\x14\r\x0c\x0b\x0b\x0c\x19\x12\x13\x0f\x14\x1d\x1a'
                b'\x1f\x1e\x1d\x1a\x1c\x1c $.\' ",#\x1c\x1c(7),01444\x1f\'9=82<.342'
                b'\xff\xc0\x00\x0b\x08\x00\x01\x00\x01\x01\x01\x11\x00'
                b'\xff\xc4\x00\x1f\x00\x00\x01\x05\x01\x01\x01\x01\x01\x01\x00\x00'
                b'\x00\x00\x00\x00\x00\x00\x01\x02\x03\x04\x05\x06\x07\x08\t\n\x0b'
                b'\xff\xc4\x00\xb5\x10\x00\x02\x01\x03\x03\x02\x04\x03\x05\x05\x04'
                b'\x04\x00\x00\x01}\x01\x02\x03\x00\x04\x11\x05\x12!1A\x06\x13Qa\x07'
                b'\x22q\x142\x81\x91\xa1\x08#B\xb1\xc1\x15R\xd1\xf0$3br\x82\t\n\x16'
                b'\x17\x18\x19\x1a%&\'()*456789:CDEFGHIJSTUVWXYZcdefghijstuvwxyz'
                b'\x83\x84\x85\x86\x87\x88\x89\x8a\x92\x93\x94\x95\x96\x97\x98\x99'
                b'\x9a\xa2\xa3\xa4\xa5\xa6\xa7\xa8\xa9\xaa\xb2\xb3\xb4\xb5\xb6\xb7'
                b'\xb8\xb9\xba\xc2\xc3\xc4\xc5\xc6\xc7\xc8\xc9\xca\xd2\xd3\xd4\xd5'
                b'\xd6\xd7\xd8\xd9\xda\xe1\xe2\xe3\xe4\xe5\xe6\xe7\xe8\xe9\xea\xf1'
                b'\xf2\xf3\xf4\xf5\xf6\xf7\xf8\xf9\xfa'
                b'\xff\xda\x00\x08\x01\x01\x00\x00?\x00T\xdb\xc1\xa4(\xa0\x02\x80\x0f\xff\xd9'
            )
        yield (
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n\r\n"
            + jpeg_bytes
            + b"\r\n"
        )
        await asyncio.sleep(interval)


@router.get("/{camera_id}")
async def video_stream(camera_id: str):
    """MJPEG stream cho 1 camera.

    Su dung trong browser:
        <img src="http://localhost:8000/stream/CAM_001" />
    """
    return StreamingResponse(
        _mjpeg_generator(camera_id),
        media_type="multipart/x-mixed-replace; boundary=frame",
    )


@router.get("/")
async def list_streams():
    """Danh sach camera dang co frame."""
    camera_ids = frame_buffer.get_camera_ids()
    return {
        "cameras": camera_ids,
        "streams": {cid: f"/stream/{cid}" for cid in camera_ids},
    }
