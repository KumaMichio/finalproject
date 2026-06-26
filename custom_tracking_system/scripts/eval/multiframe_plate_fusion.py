#!/usr/bin/env python3
"""
Buoc 0 cua plan cai thien bien so (xem plan da duyet): multi-frame fusion.

Vong 5 (test_plate_ocr_feasibility_v3.py) do ty le doc dung TREN TUNG KHUNG
HINH RIENG LE (8% oto, 0.7% xe may). Nhung pipeline thuc te co track on dinh
qua hang chuc khung hinh — MOT track chi can doc dung o MOT khung hinh la du
de gan bien so cho ca track do. Script nay do ty le thanh cong THEO TRACK
(chay detector + ByteTrackWrapper thuc cua project tren 1 doan video, voi
moi track du dai chay lai model bien so vong 5 tren TUNG khung hinh track do
xuat hien, tinh ty le "it nhat 1 khung hinh doc dung/ca track").

Khong train gi ca — chi danh gia xem multi-frame fusion mot minh giup duoc
bao nhieu, truoc khi dau tu vao buoc gan nhan + fine-tune (buoc 1+ cua plan).

Usage:
    venv_tracking\\Scripts\\python.exe scripts\\eval\\multiframe_plate_fusion.py \\
        --n-frames 300 --min-track-len 5
"""

import argparse
import sys
import types
from pathlib import Path

import cv2
import numpy as np

THIS_DIR = Path(__file__).resolve().parent
ROOT = THIS_DIR.parents[1]
sys.path.insert(0, str(THIS_DIR))
sys.path.insert(0, str(ROOT))

sys.modules.setdefault('seaborn', types.ModuleType('seaborn'))

from test_plate_ocr_feasibility import DEFAULT_VIDEO  # noqa: E402
from test_plate_ocr_feasibility_v3 import (  # noqa: E402
    deskew_variants, load_models, read_plate_chars,
)
from modules.detector import ObjectDetector  # noqa: E402
from modules.tracker import ByteTrackWrapper  # noqa: E402

_VEHICLE_CLASSES = {'car', 'motorcycle', 'bus', 'truck'}


def read_plate_for_crop(detector_lp, ocr_lp, vehicle_crop):
    """One Stage1(detect plate)+Stage2(read chars) attempt for one already-
    cropped vehicle image -- same logic as evaluate() in v3, isolated here
    so it can be called per-frame-per-track."""
    if vehicle_crop.size == 0:
        return None

    try:
        det = detector_lp(vehicle_crop, size=640)
    except Exception:
        return None
    plates = det.pandas().xyxy[0].values.tolist()
    if not plates:
        return None
    plates.sort(key=lambda p: -(p[2] - p[0]) * (p[3] - p[1]))
    px1, py1, px2, py2 = (int(v) for v in plates[0][:4])
    plate_crop = vehicle_crop[max(0, py1):py2, max(0, px1):px2]
    if plate_crop.size == 0 or plate_crop.shape[0] < 4 or plate_crop.shape[1] < 4:
        return None

    for variant in deskew_variants(plate_crop):
        try:
            text, n_chars = read_plate_chars(ocr_lp, variant)
        except Exception:
            continue
        if text is not None:
            return text
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--video', default=str(DEFAULT_VIDEO))
    ap.add_argument('--weights', default=str(ROOT / 'weights' / 'yolo11s_vn.pt'))
    ap.add_argument('--n-frames', type=int, default=300)
    ap.add_argument('--min-track-len', type=int, default=5,
                     help='Minimum frames a track must persist to count it')
    ap.add_argument('--max-attempts-per-track', type=int, default=15,
                     help='Cap plate-reading attempts per track (cost control)')
    args = ap.parse_args()

    print("Loading vehicle detector + tracker (project's own modules) ...")
    vehicle_detector = ObjectDetector(model_type=args.weights, conf_threshold=0.35)
    tracker = ByteTrackWrapper(class_map=vehicle_detector.CLASSES, frame_rate=10)

    print("Loading plate detector + character model (round-5 specialized model) ...")
    lp_detector, lp_ocr = load_models()

    cap = cv2.VideoCapture(args.video)
    track_frames = {}   # track_id -> list of (frame_idx, box, class_name)
    track_class = {}

    frame_idx = 0
    print(f"Running detector+tracker over {args.n_frames} frames ...")
    while cap.isOpened() and frame_idx < args.n_frames:
        ok, frame = cap.read()
        if not ok:
            break
        detections = vehicle_detector.detect(frame)
        tracks = tracker.update(detections, frame, timestamp=frame_idx / 10.0)
        for t in tracks:
            cls = t['class']
            if cls not in _VEHICLE_CLASSES:
                continue
            tid = t['track_id']
            x1, y1, x2, y2 = (int(v) for v in t['box'])
            crop = frame[max(0, y1):y2, max(0, x1):x2]
            if crop.size == 0:
                continue
            track_frames.setdefault(tid, []).append((frame_idx, crop.copy()))
            track_class[tid] = cls
        frame_idx += 1
    cap.release()
    print(f"Tracked {len(track_frames)} vehicle tracks over {frame_idx} frames.")

    per_class_total = {}
    per_class_track_success = {}
    per_class_frame_attempts = {}
    per_class_frame_success = {}

    for tid, frames in track_frames.items():
        if len(frames) < args.min_track_len:
            continue
        cls = track_class[tid]
        per_class_total[cls] = per_class_total.get(cls, 0) + 1

        # Sample up to max_attempts_per_track frames evenly across the track's
        # lifetime (cheaper than every single frame, still covers the range
        # of angles/blur the vehicle was seen at).
        sampled = frames[::max(1, len(frames) // args.max_attempts_per_track)]
        sampled = sampled[:args.max_attempts_per_track]

        track_success = False
        for _fi, crop in sampled:
            per_class_frame_attempts[cls] = per_class_frame_attempts.get(cls, 0) + 1
            text = read_plate_for_crop(lp_detector, lp_ocr, crop)
            if text is not None:
                per_class_frame_success[cls] = per_class_frame_success.get(cls, 0) + 1
                track_success = True
                print(f"  track {tid} ({cls}): read '{text}' at one of {len(sampled)} sampled frames")
        if track_success:
            per_class_track_success[cls] = per_class_track_success.get(cls, 0) + 1

    print("\n" + "=" * 70)
    print("MULTI-FRAME FUSION — ty le THEO TRACK vs THEO KHUNG HINH")
    print("=" * 70)
    for cls in sorted(per_class_total):
        n_tracks = per_class_total[cls]
        n_track_ok = per_class_track_success.get(cls, 0)
        n_attempts = per_class_frame_attempts.get(cls, 0)
        n_frame_ok = per_class_frame_success.get(cls, 0)
        track_pct = 100 * n_track_ok / n_tracks if n_tracks else 0
        frame_pct = 100 * n_frame_ok / n_attempts if n_attempts else 0
        print(f"{cls:10s} n_tracks={n_tracks:3d}  "
              f"theo-track={track_pct:5.1f}% ({n_track_ok}/{n_tracks})   "
              f"theo-khung-hinh={frame_pct:5.1f}% ({n_frame_ok}/{n_attempts})")


if __name__ == '__main__':
    main()
