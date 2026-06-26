"""
Traffic Light Color Classifier — phan loai mau den giao thong tu camera CCTV
co dinh, khong can train model phat hien.

Vi camera co dinh, vi tri tung bong den (do/vang/xanh) trong khung hinh
khong doi qua thoi gian — chi can do toa do tam BANG TAY mot lan cho moi
camera (xem scripts/eval/test_traffic_light_feasibility.py de tim toa do),
roi moi khung hinh so sanh do sang (HSV value channel) tai cac toa do do —
bong nao sang nhat va vuot nguong la trang thai hien tai. Tranh duoc rui ro
domain-gap cua viec train 1 detector moi tren du lieu khong co san (xem
Chuong 5 muc 5.4 — YOLO catastrophic forgetting khi fine-tune tren domain
hep).

Output dung dinh dang chuoi "Red"/"Yellow"/"Green"/None ma
IncidentDetector._check_red_light_violation() da yeu cau san.
"""

import logging
from collections import defaultdict, deque

import cv2
import numpy as np

logger = logging.getLogger(__name__)

_PATCH_RADIUS = 12       # px quanh tam diem, do tay tu video mau
_MIN_BRIGHTNESS = 120    # nguong V (HSV) toi thieu de coi la "dang sang"
_MIN_MARGIN = 20         # bong sang nhat phai hon bong nhi it nhat bao nhieu V
_SMOOTH_WINDOW = 5       # so khung hinh gan nhat dung de bo phieu da so


class TrafficLightColorClassifier:
    def __init__(self, points_config: dict, smooth_window: int = _SMOOTH_WINDOW):
        """
        Args:
            points_config: {camera_id: {color_name: [x, y]}}, vi du
                {"CAM_PHO_HUE_REAL": {"Red": [1545, 440], "Green": [1545, 490]}}
                Khong nhat thiet phai co du 3 mau — vi du den nguoi-di-bo chi
                co 2 bong (Red/Green), van hop le.
            smooth_window: so khung hinh gan nhat dung lam muot (da so phieu).
        """
        self.points_config = points_config or {}
        self.smooth_window = smooth_window
        self._history: dict = defaultdict(lambda: deque(maxlen=self.smooth_window))
        logger.info(
            "TrafficLightColorClassifier: %d camera(s) co toa do den — %s",
            len(self.points_config), list(self.points_config.keys()),
        )

    def _classify_raw(self, frame: np.ndarray, camera_id: str) -> str | None:
        """Phan loai tren 1 khung hinh don le, khong lam muot."""
        points = self.points_config.get(camera_id)
        if not points:
            return None

        h, w = frame.shape[:2]
        brightness = {}
        for color_name, (cx, cy) in points.items():
            x1, y1 = max(0, cx - _PATCH_RADIUS), max(0, cy - _PATCH_RADIUS)
            x2, y2 = min(w, cx + _PATCH_RADIUS), min(h, cy + _PATCH_RADIUS)
            patch = frame[y1:y2, x1:x2]
            if patch.size == 0:
                continue
            hsv = cv2.cvtColor(patch, cv2.COLOR_RGB2HSV)
            brightness[color_name] = float(hsv[:, :, 2].mean())

        if not brightness:
            return None

        ranked = sorted(brightness.items(), key=lambda kv: kv[1], reverse=True)
        best_color, best_v = ranked[0]
        second_v = ranked[1][1] if len(ranked) > 1 else 0.0

        if best_v < _MIN_BRIGHTNESS or (best_v - second_v) < _MIN_MARGIN:
            # Khong bong nao du sang, hoac 2 bong gan sang nhau (dang chuyen
            # tiep / ca 2 cung toi) — khong chac chan, tra None thay vi doan.
            return None
        return best_color

    def classify(self, frame: np.ndarray, camera_id: str) -> str | None:
        """
        Phan loai mau den hien tai cho 1 camera, da lam muot qua
        smooth_window khung hinh gan nhat (da so phieu, None khong tinh).

        Args:
            frame: khung hinh RGB (H, W, 3) hien tai cua camera_id.
            camera_id: id camera, phai khop key trong points_config.

        Returns:
            "Red" | "Yellow" | "Green" | None (chua xac dinh / camera khong
            co toa do den / dang chuyen tiep).
        """
        raw = self._classify_raw(frame, camera_id)
        hist = self._history[camera_id]
        hist.append(raw)

        votes = [v for v in hist if v is not None]
        if not votes:
            return None
        # Da so phieu trong window — tranh nhieu 1 khung hinh don le.
        counts = defaultdict(int)
        for v in votes:
            counts[v] += 1
        best_color, n = max(counts.items(), key=lambda kv: kv[1])
        if n < (len(votes) // 2 + 1):
            return None  # khong co mau nao chiem da so ro rang
        return best_color
