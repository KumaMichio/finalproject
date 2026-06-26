"""
Goal Classifier Module — real-time turn direction prediction.

Wraps goal_classifier_real.pkl (GradientBoosting + CalibratedCV, 33 features)
to run on live ByteTrack data.  Maintains a per-track bbox history so that
the bbox-normalization features (speed_ms_*, speed_norm_bbox, bbox_size_mean,
px_per_m_mean) can be computed the same way as during training.

Usage (in main.py):
    gc = GoalClassifier('weights/goal_classifier_real.pkl',
                        'weights/goal_scaler_real.pkl',
                        'weights/goal_label_encoder_real.pkl',
                        'weights/goal_feature_names_real.json')
    ...
    result = gc.update_and_predict(track_id, box, cls, timestamp_s)
    # result: {'direction': 'right', 'probabilities': {left:.., right:.., ..}, 'n_frames': 18}
    # or None if not enough history yet
"""

import json
import logging
import math
import pickle
from collections import deque
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

# Must match constants in train_goal_classifier_real.py
_CLASS_SPEED_LIMIT_PX = {
    "car": 350.0, "motorcycle": 300.0, "bus": 400.0,
    "truck": 300.0, "person": 100.0,
}
_CLASS_REAL_AREA_M2 = {
    "motorcycle": 0.7 * 1.8,
    "car":        1.85 * 4.5,
    "bus":        2.5 * 10.0,
    "truck":      2.5 * 7.0,
    "person":     0.5 * 1.7,
}
_CLASS_REAL_SIZE_SQRT = {k: math.sqrt(v) for k, v in _CLASS_REAL_AREA_M2.items()}

_TRAJ_WINDOW        = 4.0   # seconds of history to keep
_MIN_FRAMES         = 10    # minimum frames before first prediction (~1.5s at 6fps)
_HORIZON            = 1.5   # use 1.5 s window (best per-horizon result from training)
# Suppress non-straight predictions when heading change is negligible.
# A track going straight has total_abs_heading < ~15 deg over 1.5s.
# A genuine turn has total_abs_heading > 20-30 deg.
_MIN_HEADING_FOR_TURN_DEG = 12.0


class GoalClassifier:
    def __init__(self, clf_path: str, scaler_path: str,
                 le_path: str, feat_names_path: str):
        clf_path       = Path(clf_path)
        scaler_path    = Path(scaler_path)
        le_path        = Path(le_path)
        feat_names_path = Path(feat_names_path)

        if not all(p.exists() for p in [clf_path, scaler_path, le_path, feat_names_path]):
            raise FileNotFoundError("GoalClassifier: one or more weight files missing")

        with open(clf_path, 'rb') as f:
            self._clf = pickle.load(f)
        with open(scaler_path, 'rb') as f:
            self._scaler = pickle.load(f)
        with open(le_path, 'rb') as f:
            self._le = pickle.load(f)
        with open(feat_names_path, 'r') as f:
            self._feat_names = json.load(f)

        # Per-track history: deque of (t, cx, cy, w, h, speed_px, cls)
        self._history: dict[int, deque] = {}
        logger.info(
            "GoalClassifier loaded: %d classes, %d features",
            len(self._le.classes_), len(self._feat_names)
        )

    @property
    def classes(self):
        return list(self._le.classes_)

    def update_and_predict(self, track_id: int, box: list, cls: str,
                           timestamp_s: float) -> dict | None:
        """
        Append one frame of track data, then attempt prediction.

        Args:
            track_id:    ByteTrack local track ID (or global_id)
            box:         [x1, y1, x2, y2] bounding box in pixels
            cls:         vehicle class string ('car', 'motorcycle', ...)
            timestamp_s: unix timestamp (seconds)

        Returns:
            dict with keys:
                direction      — top-1 class ('left', 'right', 'straight', 'u_turn')
                probabilities  — {class: float} calibrated probabilities
                confidence     — float, top-1 probability
                n_frames       — number of frames used
            or None if not enough history.
        """
        x1, y1, x2, y2 = box
        cx = (x1 + x2) / 2.0
        cy = (y1 + y2) / 2.0
        w  = float(x2 - x1)
        h  = float(y2 - y1)

        if track_id not in self._history:
            self._history[track_id] = deque()
        buf = self._history[track_id]

        # Evict frames older than TRAJ_WINDOW
        while buf and (timestamp_s - buf[0][0]) > _TRAJ_WINDOW:
            buf.popleft()

        buf.append((timestamp_s, cx, cy, w, h, cls))

        if len(buf) < _MIN_FRAMES:
            return None

        feats = self._extract_features(buf, timestamp_s, cls)
        if feats is None:
            return None

        # Suppress non-straight noise: if heading barely changed over the window,
        # the vehicle is going straight regardless of what the model says.
        # total_abs_heading is the sum of absolute heading deltas (deg) — a proxy
        # for "how much did the vehicle turn?" independent of direction.
        if feats.get('total_abs_heading', 0.0) < _MIN_HEADING_FOR_TURN_DEG:
            return {
                'direction':     'straight',
                'probabilities': {'straight': 1.0, 'left': 0.0, 'right': 0.0, 'u_turn': 0.0},
                'confidence':    1.0,
                'n_frames':      len(buf),
            }

        feat_vec = np.array([[feats.get(n, 0.0) for n in self._feat_names]],
                             dtype=np.float32)
        feat_scaled = self._scaler.transform(feat_vec)
        proba = self._clf.predict_proba(feat_scaled)[0]

        class_names = list(self._le.classes_)
        prob_dict   = {c: float(p) for c, p in zip(class_names, proba)}
        top_idx     = int(np.argmax(proba))
        direction   = class_names[top_idx]

        return {
            'direction':     direction,
            'probabilities': prob_dict,
            'confidence':    float(proba[top_idx]),
            'n_frames':      len(buf),
        }

    def clear_track(self, track_id: int):
        self._history.pop(track_id, None)

    def clear_all(self):
        self._history.clear()

    # ------------------------------------------------------------------
    # Internal feature extraction
    # ------------------------------------------------------------------

    def _extract_features(self, buf: deque, t_cutoff: float, cls: str) -> dict | None:
        """Replicate extract_features_px() from training script on live data."""
        entries = [e for e in buf if e[0] <= t_cutoff and e[0] >= t_cutoff - _HORIZON]
        if len(entries) < _MIN_FRAMES:
            return None

        t_arr  = np.array([e[0] for e in entries])
        cx_arr = np.array([e[1] for e in entries])
        cy_arr = np.array([e[2] for e in entries])
        w_arr  = np.array([e[3] for e in entries])
        h_arr  = np.array([e[4] for e in entries])

        dt_total = t_arr[-1] - t_arr[0]
        if dt_total < 0.1:
            return None

        # Velocities (px/s) via finite differences. np.gradient(..., t_arr)
        # divides internally by the local spacing of t_arr — if two
        # consecutive samples share (near-)identical timestamps, that
        # spacing is ~0 and the result is +-inf, which later crashes
        # sklearn's StandardScaler ("Input X contains infinity"). Guard by
        # reconstructing a strictly-increasing time axis from the
        # already-clamped dt_arr instead of passing the raw t_arr through.
        dt_arr = np.diff(t_arr)
        dt_arr = np.where(dt_arr < 1e-6, 1e-6, dt_arr)
        t_safe = np.concatenate([[t_arr[0]], t_arr[0] + np.cumsum(dt_arr)])
        vx_arr = np.gradient(cx_arr, t_safe)
        vy_arr = np.gradient(cy_arr, t_safe)
        speed  = np.sqrt(vx_arr**2 + vy_arr**2)

        # Heading (degrees, 0=right, 90=down in pixel space)
        dx = np.diff(cx_arr)
        dy = np.diff(cy_arr)
        head_deltas = np.degrees(np.arctan2(dy, np.where(np.abs(dx) < 1e-6, 1e-6, dx)))
        head_arr = np.concatenate([[head_deltas[0]], head_deltas])

        # Bbox-based scale
        bbox_size     = np.sqrt(np.maximum(w_arr * h_arr, 1.0))
        real_size     = _CLASS_REAL_SIZE_SQRT.get(cls, 1.12)
        px_per_m      = bbox_size / real_size
        speed_ms      = speed / np.maximum(px_per_m, 0.1)
        speed_norm_bbox = speed / np.maximum(bbox_size, 1.0)
        bbox_size_mean  = float(np.mean(bbox_size))
        px_per_m_mean   = float(np.mean(px_per_m))

        speed_limit = _CLASS_SPEED_LIMIT_PX.get(cls, 300.0)
        speed_norm  = speed / speed_limit

        # Heading rate
        head_diff = np.diff(head_arr)
        head_diff = ((head_diff + 180) % 360) - 180
        head_rate = head_diff / dt_arr

        total_abs_heading  = float(np.sum(np.abs(head_diff)))
        total_sign_heading = float(np.sum(head_diff))
        heading_range      = float(np.max(head_arr) - np.min(head_arr))

        if len(head_diff) > 1:
            signs = np.sign(head_diff)
            signs = signs[signs != 0]
            if len(signs) > 1:
                head_monotonic = float(np.sum(signs[1:] == signs[:-1])) / (len(signs) - 1)
            else:
                head_monotonic = 1.0
        else:
            head_monotonic = 1.0

        speed_min       = float(np.min(speed))
        speed_max       = float(np.max(speed))
        speed_drop      = speed_max - speed_min
        speed_min_ratio = speed_min / max(speed_max, 0.1)
        n_third = max(1, len(speed) // 3)
        speed_final_third = float(np.mean(speed[-n_third:]))

        # Recent 0.5 s slice
        recent_mask = t_arr >= t_cutoff - 0.5
        sp_recent = speed[recent_mask] if recent_mask.any() else speed[-3:]

        # Lateral acceleration. Same zero-spacing hazard as the velocity
        # gradient above — use t_safe, not the raw t_arr.
        if len(vx_arr) >= 3:
            ax     = np.gradient(vx_arr, t_safe)
            ay     = np.gradient(vy_arr, t_safe)
            cross  = np.abs(vx_arr * ay - vy_arr * ax)
            v_mag  = np.sqrt(vx_arr**2 + vy_arr**2)
            safe_mag = np.where(v_mag > 0.5, v_mag, 1.0)
            lat_acc  = np.where(v_mag > 0.5, cross / safe_mag, 0.0)
            lat_acc_mean = float(np.mean(lat_acc))
            lat_acc_max  = float(np.max(lat_acc))
        else:
            lat_acc_mean = lat_acc_max = 0.0

        is_car        = 1 if cls == "car"        else 0
        is_motorcycle = 1 if cls == "motorcycle" else 0
        is_bus        = 1 if cls == "bus"        else 0
        is_truck      = 1 if cls == "truck"      else 0

        return {
            "speed_mean":         float(np.mean(speed)),
            "speed_std":          float(np.std(speed)),
            "speed_last":         float(speed[-1]),
            "speed_recent":       float(np.mean(sp_recent)),
            "speed_norm_mean":    float(np.mean(speed_norm)),
            "speed_change":       float(speed[-1] - speed[0]),
            "speed_change_rate":  float((speed[-1] - speed[0]) / dt_total),
            "head_rate_mean":     float(np.mean(head_rate)),
            "head_rate_std":      float(np.std(head_rate)),
            "head_rate_max":      float(np.max(np.abs(head_rate))),
            "head_sin":           float(math.sin(math.radians(head_arr[-1]))),
            "head_cos":           float(math.cos(math.radians(head_arr[-1]))),
            "total_abs_heading":  total_abs_heading,
            "total_sign_heading": total_sign_heading,
            "heading_range":      heading_range,
            "head_monotonic":     head_monotonic,
            "speed_min":          speed_min,
            "speed_drop":         speed_drop,
            "speed_min_ratio":    speed_min_ratio,
            "speed_final_third":  speed_final_third,
            "lat_acc_mean":       lat_acc_mean,
            "lat_acc_max":        lat_acc_max,
            "is_car":             is_car,
            "is_motorcycle":      is_motorcycle,
            "is_bus":             is_bus,
            "is_truck":           is_truck,
            # Bbox normalization features (v3)
            "speed_ms_mean":      float(np.mean(speed_ms)),
            "speed_ms_std":       float(np.std(speed_ms)),
            "speed_ms_last":      float(speed_ms[-1]),
            "speed_ms_recent":    float(np.mean(speed_ms[recent_mask]) if recent_mask.any() else speed_ms[-1]),
            "speed_norm_bbox":    float(np.mean(speed_norm_bbox)),
            "bbox_size_mean":     bbox_size_mean,
            "px_per_m_mean":      px_per_m_mean,
        }
