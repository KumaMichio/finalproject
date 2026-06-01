"""
Stream service — MJPEG video streaming + frame buffer.

AI processor ghi annotated frames vao day.
Clients doc frames qua generator (MJPEG).
"""

import asyncio
import time
from collections import defaultdict
from typing import Optional

import cv2
import numpy as np

from config import JPEG_QUALITY


class FrameBuffer:
    """Thread-safe frame buffer cho nhieu camera."""

    def __init__(self):
        # {camera_id: (frame_bytes_jpeg, timestamp)}
        self._frames: dict[str, tuple[bytes, float]] = {}
        # Asyncio event de thong bao co frame moi
        self._events: dict[str, asyncio.Event] = defaultdict(asyncio.Event)
        # Event loop — set boi FastAPI lifespan de put_frame() co the signal thread-safe
        self._loop: asyncio.AbstractEventLoop | None = None

    def set_loop(self, loop: asyncio.AbstractEventLoop):
        """Goi tu FastAPI lifespan truoc khi AI thread bat dau."""
        self._loop = loop

    def put_frame(self, camera_id: str, frame: np.ndarray):
        """Ghi 1 frame (numpy BGR) vao buffer. Goi tu AI processor thread."""
        _, jpeg = cv2.imencode(
            ".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY]
        )
        self._frames[camera_id] = (jpeg.tobytes(), time.time())
        # Signal cho tat ca consumer dang cho (thread-safe)
        event = self._events.get(camera_id)
        if event:
            if self._loop and self._loop.is_running():
                self._loop.call_soon_threadsafe(event.set)
            else:
                event.set()

    def get_latest(self, camera_id: str) -> Optional[bytes]:
        """Lay frame jpeg moi nhat. Tra ve None neu chua co."""
        entry = self._frames.get(camera_id)
        if entry is None:
            return None
        return entry[0]

    async def wait_for_frame(self, camera_id: str, timeout: float = 2.0) -> Optional[bytes]:
        """Cho frame moi (async). Dung cho MJPEG streaming."""
        event = self._events[camera_id]
        event.clear()
        try:
            await asyncio.wait_for(event.wait(), timeout=timeout)
        except asyncio.TimeoutError:
            pass
        return self.get_latest(camera_id)

    def get_camera_ids(self) -> list[str]:
        return list(self._frames.keys())


# Singleton instance — import tu cac module khac
frame_buffer = FrameBuffer()
