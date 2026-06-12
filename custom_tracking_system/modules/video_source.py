"""
Video Source — abstract layer giua camera va AI pipeline.

Muc dich:
  AI pipeline chi thay frames (numpy array), khong biet nguon la gi.
  Giong het camera that: khong biet truoc doi tuong nao se xuat hien.

Supported sources:
  - CARLAVideoSource:  doc tu CARLA sensor (dung cho mo phong)
  - RTSPVideoSource:   doc tu camera IP that (rtsp://...)
  - FileVideoSource:   doc tu video file (.mp4, .avi)
  - WebcamVideoSource: doc tu webcam (index 0, 1, ...)

Tat ca source deu implement cung interface:
  get_frame() -> {camera_id, frame, timestamp}
"""

import time
import socket
import struct
import logging
from abc import ABC, abstractmethod
from typing import Optional
from collections import deque

import cv2
import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Abstract base
# ---------------------------------------------------------------------------

class VideoSource(ABC):
    """Interface chung cho moi nguon video.
    AI pipeline chi tuong tac qua interface nay.
    """

    @abstractmethod
    def get_frame(self) -> Optional[dict]:
        """Lay 1 frame moi nhat.

        Returns:
            dict: {
                'camera_id': str,
                'frame': np.ndarray (H, W, 3) RGB,
                'timestamp': float (epoch seconds)
            }
            hoac None neu khong co frame.
        """
        pass

    @abstractmethod
    def is_alive(self) -> bool:
        """Kiem tra source con hoat dong khong."""
        pass

    @abstractmethod
    def release(self):
        """Giai phong tai nguyen."""
        pass

    @property
    @abstractmethod
    def camera_id(self) -> str:
        pass


class MultiVideoSource:
    """Quan ly nhieu VideoSource, tra ve frames dong bo."""

    def __init__(self):
        self._sources: dict[str, VideoSource] = {}

    def add_source(self, source: VideoSource):
        self._sources[source.camera_id] = source
        logger.info(f"Added video source: {source.camera_id} ({type(source).__name__})")

    def remove_source(self, camera_id: str):
        source = self._sources.pop(camera_id, None)
        if source:
            source.release()
            logger.info(f"Removed video source: {camera_id}")

    def get_synchronized_frames(self) -> dict:
        """Lay frame moi nhat tu tat ca source.

        Returns:
            dict: {camera_id: {camera_id, frame, timestamp}}
        """
        frames = {}
        for camera_id, source in self._sources.items():
            if source.is_alive():
                frame_data = source.get_frame()
                if frame_data is not None:
                    frames[camera_id] = frame_data
        return frames

    def get_camera_ids(self) -> list[str]:
        return list(self._sources.keys())

    def release_all(self):
        for source in self._sources.values():
            source.release()
        self._sources.clear()


# ---------------------------------------------------------------------------
# CARLA source — dung cho moi truong mo phong
# ---------------------------------------------------------------------------

class CARLAVideoSource(VideoSource):
    """Doc frame tu CARLA camera sensor.
    Chi tra ve frame thuan tuy — KHONG expose actor_id hay bat ky
    thong tin nao ma camera that khong co.
    """

    def __init__(self, carla_camera, camera_id: str):
        """
        Args:
            carla_camera: CARLA camera actor (da spawn)
            camera_id: ten camera ("CAM_001")
        """
        self._camera_id = camera_id
        self._carla_camera = carla_camera
        self._buffer: deque = deque(maxlen=30)
        self._alive = True

        # Dang ky callback nhan frame tu CARLA
        carla_camera.listen(self._on_image)
        logger.info(f"CARLAVideoSource [{camera_id}] listening")

    def _on_image(self, carla_image):
        """Callback tu CARLA — chi luu raw pixels."""
        try:
            # Convert CARLA image -> numpy RGB
            frame = np.array(carla_image.raw_data).reshape(
                carla_image.height, carla_image.width, 4
            )[:, :, :3]  # bo alpha channel

            self._buffer.append({
                'camera_id': self._camera_id,
                'frame': frame,
                'timestamp': time.time(),  # Dung wall clock, khong dung CARLA timestamp
            })
        except Exception as e:
            logger.error(f"CARLAVideoSource [{self._camera_id}] error: {e}")

    def get_frame(self) -> Optional[dict]:
        if len(self._buffer) > 0:
            return self._buffer[-1]
        return None

    def is_alive(self) -> bool:
        return self._alive

    def release(self):
        self._alive = False
        try:
            self._carla_camera.destroy()
        except Exception:
            pass

    @property
    def camera_id(self) -> str:
        return self._camera_id


# ---------------------------------------------------------------------------
# RTSP source — dung cho camera IP that
# ---------------------------------------------------------------------------

class RTSPVideoSource(VideoSource):
    """Doc frame tu camera IP qua RTSP/HTTP stream.

    Dung cho camera that:
        RTSPVideoSource("rtsp://192.168.1.100:554/stream1", "CAM_LOBBY")
        RTSPVideoSource("http://192.168.1.101:8080/video", "CAM_PARKING")
    """

    def __init__(self, url: str, camera_id: str):
        self._camera_id = camera_id
        self._url = url
        self._cap = cv2.VideoCapture(url)

        if not self._cap.isOpened():
            logger.error(f"RTSPVideoSource [{camera_id}] cannot open: {url}")

        logger.info(f"RTSPVideoSource [{camera_id}] connected to {url}")

    def get_frame(self) -> Optional[dict]:
        if not self._cap.isOpened():
            return None

        ret, frame = self._cap.read()
        if not ret:
            return None

        # OpenCV doc BGR, convert sang RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return {
            'camera_id': self._camera_id,
            'frame': frame_rgb,
            'timestamp': time.time(),
        }

    def is_alive(self) -> bool:
        return self._cap.isOpened()

    def release(self):
        self._cap.release()

    @property
    def camera_id(self) -> str:
        return self._camera_id


# ---------------------------------------------------------------------------
# Video file source — dung cho test offline / replay
# ---------------------------------------------------------------------------

class FileVideoSource(VideoSource):
    """Doc frame tu video file (.mp4, .avi, ...).

    Dung cho:
        - Test offline voi video thu san
        - Replay video da ghi
        - Benchmark tren dataset (MOT17, MOT20)
    """

    def __init__(self, filepath: str, camera_id: str, loop: bool = False):
        self._camera_id = camera_id
        self._filepath = filepath
        self._loop = loop
        self._cap = cv2.VideoCapture(filepath)

        if not self._cap.isOpened():
            logger.error(f"FileVideoSource [{camera_id}] cannot open: {filepath}")

        self._fps = self._cap.get(cv2.CAP_PROP_FPS) or 10
        logger.info(f"FileVideoSource [{camera_id}] opened {filepath} ({self._fps:.0f} FPS)")

    def get_frame(self) -> Optional[dict]:
        if not self._cap.isOpened():
            return None

        ret, frame = self._cap.read()

        if not ret:
            if self._loop:
                self._cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                ret, frame = self._cap.read()
                if not ret:
                    return None
            else:
                return None

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return {
            'camera_id': self._camera_id,
            'frame': frame_rgb,
            'timestamp': time.time(),
        }

    def is_alive(self) -> bool:
        return self._cap.isOpened()

    def release(self):
        self._cap.release()

    @property
    def camera_id(self) -> str:
        return self._camera_id


# ---------------------------------------------------------------------------
# Webcam source — dung cho demo nhanh
# ---------------------------------------------------------------------------

class WebcamVideoSource(VideoSource):
    """Doc frame tu webcam.

    Dung cho demo:
        WebcamVideoSource(0, "WEBCAM_0")
    """

    def __init__(self, device_index: int = 0, camera_id: str = "WEBCAM_0"):
        self._camera_id = camera_id
        self._cap = cv2.VideoCapture(device_index)
        logger.info(f"WebcamVideoSource [{camera_id}] opened device {device_index}")

    def get_frame(self) -> Optional[dict]:
        if not self._cap.isOpened():
            return None

        ret, frame = self._cap.read()
        if not ret:
            return None

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return {
            'camera_id': self._camera_id,
            'frame': frame_rgb,
            'timestamp': time.time(),
        }

    def is_alive(self) -> bool:
        return self._cap.isOpened()

    def release(self):
        self._cap.release()

    @property
    def camera_id(self) -> str:
        return self._camera_id


# ---------------------------------------------------------------------------
# CARLA bridge client — nhan frames tu carla_bridge/server.py (Python 3.7)
# qua TCP socket. Dung khi CARLA PythonAPI (cp37) khong tuong thich voi
# moi truong AI (Python 3.10 + torch CUDA cho GPU moi).
# ---------------------------------------------------------------------------

class CARLABridgeClient:
    """Nhan synchronized frames tu carla_bridge server qua TCP socket.

    Cung interface get_synchronized_frames() nhu CameraController, de
    drop-in thay the trong main.py khi chay o che do --source bridge.
    """

    def __init__(self, host: str = '127.0.0.1', port: int = 8765, connect_timeout: float = 60.0):
        logger.info(f"CARLABridgeClient connecting to {host}:{port} ...")
        self._sock = socket.create_connection((host, port), timeout=connect_timeout)
        self._sock.settimeout(None)
        logger.info(f"CARLABridgeClient connected to {host}:{port}")

    def _recv_exact(self, n: int) -> bytes:
        buf = bytearray()
        while len(buf) < n:
            chunk = self._sock.recv(n - len(buf))
            if not chunk:
                raise ConnectionError("CARLA bridge connection closed")
            buf.extend(chunk)
        return bytes(buf)

    def get_synchronized_frames(self) -> dict:
        frame_number = struct.unpack('<I', self._recv_exact(4))[0]
        n_cameras = struct.unpack('<I', self._recv_exact(4))[0]
        frames = {}
        for _ in range(n_cameras):
            id_len = struct.unpack('<I', self._recv_exact(4))[0]
            camera_id = self._recv_exact(id_len).decode('utf-8')
            timestamp = struct.unpack('<d', self._recv_exact(8))[0]
            h, w, c = struct.unpack('<III', self._recv_exact(12))
            frame_bytes = self._recv_exact(h * w * c)
            frame = np.frombuffer(frame_bytes, dtype=np.uint8).reshape(h, w, c)
            frames[camera_id] = {
                'camera_id': camera_id,
                'frame': frame,
                'timestamp': timestamp,
                'frame_number': frame_number,
            }
        return frames

    def cleanup(self):
        try:
            self._sock.close()
        except Exception:
            pass
