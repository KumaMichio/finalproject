"""
Evidence Package Module
Tự động lưu bằng chứng (ảnh crop, video clip, metadata JSON) khi sự cố xảy ra.
"""

import cv2
import os
import json
import logging
import numpy as np
from collections import deque
from datetime import datetime

logger = logging.getLogger(__name__)


class EvidencePackage:
    """
    Khi nhận incident từ IncidentDetector, tự động đóng gói:
      - Ảnh crop đối tượng liên quan (từ frame hiện tại)
      - Snapshot toàn cảnh camera
      - Clip video 30s trước sự cố (từ ring buffer)
      - Clip video 30s sau sự cố (ghi tiếp sau khi incident)
      - metadata.json đầy đủ thông tin

    Cách dùng:
        evidence = EvidencePackage(output_dir="evidence")
        # Mỗi frame: buffer trước
        evidence.buffer_frame(camera_id, frame, timestamp)
        # Khi có incident:
        out_dir, meta = evidence.capture(incident, frames_dict, global_tracks)
    """

    def __init__(self, output_dir: str = "evidence", buffer_seconds: int = 30, fps: int = 10):
        self.output_dir = output_dir
        self.buffer_size = buffer_seconds * fps   # số frame trong ring buffer
        self.fps = fps
        os.makedirs(output_dir, exist_ok=True)

        # Ring buffer per camera: deque of (frame_bgr, timestamp)
        self._buffers: dict[str, deque] = {}

        # Clip đang ghi tiếp sau sự cố:
        # {incident_id: {'camera_id', 'writer', 'frames_left', 'out_dir'}}
        self._pending: dict[str, dict] = {}

        logger.info(f"EvidencePackage initialized — output: {output_dir}, "
                    f"buffer: {buffer_seconds}s @ {fps}fps")

    # ------------------------------------------------------------------
    # Gọi mỗi frame
    # ------------------------------------------------------------------

    def buffer_frame(self, camera_id: str, frame: np.ndarray, timestamp: datetime = None):
        """
        Duy trì ring buffer cho camera này.
        Phải được gọi mỗi frame TRƯỚC khi gọi capture().
        """
        if camera_id not in self._buffers:
            self._buffers[camera_id] = deque(maxlen=self.buffer_size)

        ts = timestamp or datetime.now()
        self._buffers[camera_id].append((frame.copy(), ts))

        # Tiếp tục ghi clip hậu sự cố nếu đang có
        self._write_pending(camera_id, frame)

    # ------------------------------------------------------------------
    # Gọi khi có incident
    # ------------------------------------------------------------------

    def capture(self, incident: dict, frames_dict: dict, global_tracks: list):
        """
        Tạo evidence package cho 1 sự cố.

        Args:
            incident:      dict từ IncidentDetector.update()
            frames_dict:   {camera_id: frame_bgr} — frame hiện tại mỗi camera
            global_tracks: list track hiện tại (để cắt crop)

        Returns:
            (out_dir: str, meta: dict)
        """
        ts_str = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
        incident_id = f"{incident['type']}_{incident['global_id']}_{ts_str}"
        out_dir = os.path.join(self.output_dir, incident_id)
        os.makedirs(out_dir, exist_ok=True)

        camera_id = incident['camera_id']
        frame = frames_dict.get(camera_id)

        crops = {}
        snapshot_path = None

        if frame is not None:
            # --- Ảnh crop các đối tượng liên quan ---
            involved = {incident['global_id']}
            details = incident.get('details', {})
            for key in ('other_id', 'pedestrian_id'):
                if key in details:
                    involved.add(details[key])

            for track in global_tracks:
                if track['global_id'] in involved:
                    crop = self._crop(frame, track['box'])
                    if crop.size > 0:
                        path = os.path.join(out_dir, f"crop_{track['global_id']}.jpg")
                        cv2.imwrite(path, crop)
                        crops[track['global_id']] = path

            # --- Snapshot toàn cảnh ---
            snapshot_path = os.path.join(out_dir, 'snapshot.jpg')
            cv2.imwrite(snapshot_path, frame)

        # --- Clip PRE-incident từ ring buffer ---
        pre_clip_path = self._save_pre_clip(camera_id, out_dir)

        # --- Bắt đầu ghi clip POST-incident ---
        post_clip_path = os.path.join(out_dir, 'clip_post.mp4')
        if frame is not None:
            self._start_post_clip(incident_id, camera_id, frame, post_clip_path)

        # --- metadata.json ---
        meta = {
            'incident_id':    incident_id,
            'type':           incident['type'],
            'severity':       incident['severity'],
            'timestamp':      incident['timestamp'].isoformat(),
            'camera_id':      camera_id,
            'global_id':      incident['global_id'],
            'message':        incident['message'],
            'details':        incident.get('details', {}),
            'evidence': {
                'crops':         crops,
                'snapshot':      snapshot_path,
                'pre_clip':      pre_clip_path,
                'post_clip':     post_clip_path,
            },
        }
        meta_path = os.path.join(out_dir, 'metadata.json')
        with open(meta_path, 'w', encoding='utf-8') as f:
            json.dump(meta, f, ensure_ascii=False, indent=2, default=str)

        logger.info(f"Evidence saved → {out_dir}")
        return out_dir, meta

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _crop(self, frame: np.ndarray, box, padding: int = 20) -> np.ndarray:
        h, w = frame.shape[:2]
        x1, y1, x2, y2 = box
        x1 = max(0, int(x1) - padding)
        y1 = max(0, int(y1) - padding)
        x2 = min(w, int(x2) + padding)
        y2 = min(h, int(y2) + padding)
        return frame[y1:y2, x1:x2]

    def _save_pre_clip(self, camera_id: str, out_dir: str):
        buf = self._buffers.get(camera_id)
        if not buf:
            return None
        frames = list(buf)
        if not frames:
            return None

        clip_path = os.path.join(out_dir, 'clip_pre.mp4')
        h, w = frames[0][0].shape[:2]
        writer = cv2.VideoWriter(
            clip_path,
            cv2.VideoWriter_fourcc(*'mp4v'),
            self.fps, (w, h)
        )
        for f, _ in frames:
            writer.write(f)
        writer.release()
        return clip_path

    def _start_post_clip(self, incident_id: str, camera_id: str,
                         first_frame: np.ndarray, clip_path: str):
        h, w = first_frame.shape[:2]
        writer = cv2.VideoWriter(
            clip_path,
            cv2.VideoWriter_fourcc(*'mp4v'),
            self.fps, (w, h)
        )
        writer.write(first_frame)
        frames_needed = self.buffer_size - 1   # buffer_size = 30s * fps
        self._pending[incident_id] = {
            'camera_id':   camera_id,
            'writer':      writer,
            'frames_left': frames_needed,
            'out_dir':     os.path.dirname(clip_path),
        }

    def _write_pending(self, camera_id: str, frame: np.ndarray):
        done = []
        for inc_id, info in self._pending.items():
            if info['camera_id'] != camera_id:
                continue
            info['writer'].write(frame)
            info['frames_left'] -= 1
            if info['frames_left'] <= 0:
                info['writer'].release()
                done.append(inc_id)
                logger.debug(f"Post-incident clip finalized for {inc_id}")
        for inc_id in done:
            del self._pending[inc_id]

    def flush(self):
        """Giải phóng tất cả VideoWriter đang mở (gọi khi shutdown)."""
        for info in self._pending.values():
            try:
                info['writer'].release()
            except Exception:
                pass
        self._pending.clear()
