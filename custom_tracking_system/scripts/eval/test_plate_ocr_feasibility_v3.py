#!/usr/bin/env python3
"""
Vong 5 (go/no-go, lan 3) cho OCR bien so — thu model chuyen biet thay EasyOCR.

Vong 4 (test_plate_ocr_feasibility_v2.py) xac nhan: khoanh vung bien (localize)
hoat dong tot (kiem chung bang mat), nhung EasyOCR (OCR van ban canh tong quat)
doc sai/ranh gioi nguong tin cay tren chinh nhung anh ro net. Vong nay thay
EasyOCR bang kien truc 2 giai doan kieu repo tham khao cua nguoi dung
(https://github.com/trungdinh22/License-Plate-Recognition): YOLOv5 detect
vung bien (LP_detector.pt) + mot YOLOv5 RIENG detect TUNG KY TU lam mot class
(LP_ocr.pt, 30 class = 0-9 + chu), ghep ky tu theo dong (tren/duoi) bang
hoi quy tuyen tinh thay vi gia dinh aspect ratio co dinh.

Weights lay tu repo cong dong tren (model/LP_detector.pt, model/LP_ocr.pt,
da clone rieng vao scratchpad, KHONG dua vao repo chinh thuc cua de tai).
torch.load() voi cac file nay mac dinh weights_only=False (yolov5 checkpoint
luu ca object Model, khong chi tensor) — nguoi dung da xac nhan chap nhan rui
ro chuoi cung ung khi quyet dinh thu huong nay.

CANH BAO mot lan: viec load mo hinh yolov5 qua torch.hub kich hoat
hubconf.py cua yolov5, von goi check_requirements() va co the TU Y CHAY PIP
INSTALL de nang cap goi trong venv dang dung (da xay ra 2 lan khi viet script
nay, lam lech tqdm/requests dang PIN trong requirements.txt/requirements_cpu.txt
cua du an — da revert). Script nay monkeypatch
ultralytics.utils.checks.check_requirements truoc khi goi torch.hub.load de
chan han nay tu lap lai.

Usage:
    venv_tracking\\Scripts\\python.exe scripts\\eval\\test_plate_ocr_feasibility_v3.py --n-samples 120
"""

import argparse
import json
import math
import sys
import types
from pathlib import Path

import cv2
import numpy as np

THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(THIS_DIR))

from test_plate_ocr_feasibility import (  # noqa: E402
    DEFAULT_VIDEO, ROOT, sample_from_auto_label, sample_from_video,
)

# yolov5 hubconf imports seaborn at module load for plotting utilities we
# never call; the installed matplotlib is incompatible with this seaborn
# version (register_cmap removed) so stub it out rather than touch versions.
sys.modules.setdefault('seaborn', types.ModuleType('seaborn'))

# Vendored offline-only (see .gitignore): official ultralytics/yolov5 (MIT)
# code, needed only because it's the legacy architecture LP_detector_ref.pt/
# LP_ocr_ref.pt (trungdinh22/License-Plate-Recognition, no LICENSE file --
# reference/pseudo-labeling use only, never shipped in the product) were
# trained on. Re-clone if missing:
#   git clone https://github.com/ultralytics/yolov5 <ROOT>/third_party/yolov5
YOLOV5_DIR = ROOT / 'third_party' / 'yolov5'
LP_DETECTOR_WEIGHTS = ROOT / 'weights' / 'lp_detector_ref.pt'
LP_OCR_WEIGHTS = ROOT / 'weights' / 'lp_ocr_ref.pt'

OUT_DIR = ROOT.parent / 'docs' / 'assets' / 'plate_ocr_feasibility_v3'


def load_models():
    import ultralytics.utils.checks as checks
    checks.check_requirements = lambda *a, **k: True  # see module docstring
    import torch
    detector = torch.hub.load(str(YOLOV5_DIR), 'custom', path=str(LP_DETECTOR_WEIGHTS),
                               source='local', force_reload=False)
    ocr = torch.hub.load(str(YOLOV5_DIR), 'custom', path=str(LP_OCR_WEIGHTS),
                          source='local', force_reload=False)
    ocr.conf = 0.5
    return detector, ocr


def _linear_check(x, y, x1, y1, x2, y2, tol=3):
    if x2 == x1:
        return abs(x - x1) <= tol
    a = (y2 - y1) / (x2 - x1)
    b = y1 - a * x1
    return math.isclose(a * x + b, y, abs_tol=tol)


def read_plate_chars(ocr_model, crop_bgr):
    """Run the character-detector model on a plate crop, reassemble into a
    string (1-line: left-to-right; 2-line: classify via linear-fit check on
    detected character centers, like the reference repo's helper.read_plate).
    Returns None if too few/many characters detected (no partial credit —
    matches the reference repo's own go/no-go threshold of 7-10 chars)."""
    if crop_bgr is None or crop_bgr.size == 0 or crop_bgr.ndim != 3 or crop_bgr.shape[2] != 3:
        return None, 0
    try:
        results = ocr_model(crop_bgr)
    except Exception:
        # Degenerate aspect-ratio crops (e.g. a sliver mis-detected as a
        # "plate") occasionally break yolov5's letterbox preprocessing --
        # skip rather than crash a long-running batch experiment.
        return None, 0
    bb = results.pandas().xyxy[0].values.tolist()
    if len(bb) < 7 or len(bb) > 10:
        return None, len(bb)

    centers = [((b[0] + b[2]) / 2, (b[1] + b[3]) / 2, str(b[-1])) for b in bb]
    l_point = min(centers, key=lambda c: c[0])
    r_point = max(centers, key=lambda c: c[0])

    two_line = any(
        not _linear_check(cx, cy, l_point[0], l_point[1], r_point[0], r_point[1])
        for cx, cy, _ in centers
    )

    if not two_line:
        text = ''.join(c[2] for c in sorted(centers, key=lambda c: c[0]))
        return text, len(bb)

    y_mean = sum(c[1] for c in centers) / len(centers)
    line1 = sorted((c for c in centers if c[1] <= y_mean), key=lambda c: c[0])
    line2 = sorted((c for c in centers if c[1] > y_mean), key=lambda c: c[0])
    text = ''.join(c[2] for c in line1) + ''.join(c[2] for c in line2)
    return text, len(bb)


def _enhance_contrast(image):
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l_channel, a, b = cv2.split(lab)
    l_channel = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8)).apply(l_channel)
    return cv2.cvtColor(cv2.merge((l_channel, a, b)), cv2.COLOR_LAB2BGR)


def _estimate_skew_deg(image, ignore_top_rows: bool):
    gray = cv2.medianBlur(image, 3)
    if gray.ndim == 3:
        gray = cv2.cvtColor(gray, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape[:2]
    edges = cv2.Canny(gray, 30, 100, apertureSize=3, L2gradient=True)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 30,
                             minLineLength=w / 1.5, maxLineGap=h / 3.0)
    if lines is None:
        return 0.0

    best_idx, best_y = None, float('inf')
    for i, (x1, y1, x2, y2) in enumerate(lines[:, 0]):
        ymid = (y1 + y2) / 2
        if ignore_top_rows and ymid < 7:
            continue
        if ymid < best_y:
            best_y, best_idx = ymid, i
    if best_idx is None:
        return 0.0

    x1, y1, x2, y2 = lines[best_idx, 0]
    angle = np.degrees(np.arctan2(y2 - y1, x2 - x1))
    return angle if abs(angle) <= 30 else 0.0


def _rotate(image, angle_deg):
    h, w = image.shape[:2]
    center = (w / 2, h / 2)
    matrix = cv2.getRotationMatrix2D(center, angle_deg, 1.0)
    return cv2.warpAffine(image, matrix, (w, h), flags=cv2.INTER_LINEAR)


def deskew_variants(crop_bgr):
    """Clean-room reimplementation of the reference repo's preprocessing
    idea (not its code): try contrast-enhanced x rotation-corrected
    combinations before giving up on a plate crop. Skew angle comes from the
    most-prominent near-horizontal Hough line; only meaningfully changes
    output on visibly skewed crops."""
    for use_contrast in (False, True):
        base = _enhance_contrast(crop_bgr) if use_contrast else crop_bgr
        for ignore_top in (False, True):
            angle = _estimate_skew_deg(base, ignore_top)
            yield _rotate(base, angle)


def evaluate(samples, detector, ocr, save_crops, out_dir):
    out_dir.mkdir(parents=True, exist_ok=True)
    per_class_total, per_class_detected, per_class_read = {}, {}, {}
    examples = []
    saved = 0

    for frame, box, cls_name, sample_id in samples:
        per_class_total[cls_name] = per_class_total.get(cls_name, 0) + 1
        x1, y1, x2, y2 = (int(v) for v in box)
        vehicle_crop = frame[y1:y2, x1:x2]
        if vehicle_crop.size == 0:
            examples.append({'id': sample_id, 'class': cls_name,
                              'plate_detected': False, 'text': None, 'n_chars': 0})
            continue

        det = detector(vehicle_crop, size=640)
        plates = det.pandas().xyxy[0].values.tolist()
        plate_detected = len(plates) > 0

        text, n_chars = None, 0
        if plate_detected:
            plates.sort(key=lambda p: -(p[2] - p[0]) * (p[3] - p[1]))
            px1, py1, px2, py2 = (int(v) for v in plates[0][:4])
            plate_crop = vehicle_crop[max(0, py1):py2, max(0, px1):px2]
            if plate_crop.size > 0:
                for variant in deskew_variants(plate_crop):
                    text, n_chars = read_plate_chars(ocr, variant)
                    if text is not None:
                        break
                if saved < save_crops:
                    cv2.imwrite(str(out_dir / f"{sample_id}_plate.jpg"), plate_crop)
                    saved += 1

        if plate_detected:
            per_class_detected[cls_name] = per_class_detected.get(cls_name, 0) + 1
        if text is not None:
            per_class_read[cls_name] = per_class_read.get(cls_name, 0) + 1

        examples.append({'id': sample_id, 'class': cls_name,
                          'plate_detected': plate_detected, 'text': text, 'n_chars': n_chars})
        print(f"[{sample_id}] class={cls_name:10s} plate_detected={plate_detected} "
              f"text={text} n_chars={n_chars}")

    summary = {}
    for cls_name in sorted(per_class_total):
        n = per_class_total[cls_name]
        d = per_class_detected.get(cls_name, 0)
        r = per_class_read.get(cls_name, 0)
        summary[cls_name] = {
            'n': n,
            'plate_region_detected_pct': round(100 * d / n, 1),
            'text_read_pct': round(100 * r / n, 1),
        }
    return summary, examples


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--source', choices=['auto_label', 'video'], default='video')
    ap.add_argument('--video', default=str(DEFAULT_VIDEO))
    ap.add_argument('--weights', default=str(ROOT / 'weights' / 'yolo11s_vn.pt'))
    ap.add_argument('--n-samples', type=int, default=120)
    ap.add_argument('--save-crops', type=int, default=40)
    args = ap.parse_args()

    detector, ocr = load_models()
    if args.source == 'auto_label':
        samples = list(sample_from_auto_label(args.n_samples))
    else:
        samples = list(sample_from_video(Path(args.video), args.n_samples, args.weights))

    summary, examples = evaluate(samples, detector, ocr, args.save_crops, OUT_DIR)

    print("\n" + "=" * 70)
    print("VONG 5 — model chuyen biet (YOLOv5 detect bien + detect ky tu)")
    print("=" * 70)
    for cls_name, s in summary.items():
        print(f"{cls_name:10s} n={s['n']:3d}  plate_detected={s['plate_region_detected_pct']:5.1f}%  "
              f"text_read={s['text_read_pct']:5.1f}%")

    out_json = OUT_DIR / 'feasibility_v3_summary.json'
    with open(out_json, 'w', encoding='utf-8') as f:
        json.dump({'summary': summary, 'examples': examples}, f, indent=2, ensure_ascii=False)
    print(f"\nDa luu: {out_json}")


if __name__ == '__main__':
    main()
