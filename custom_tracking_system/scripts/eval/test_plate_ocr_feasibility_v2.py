#!/usr/bin/env python3
"""
Vong 4 (go/no-go, lan 2) cho OCR bien so.

Vong 1-3 (test_plate_ocr_feasibility.py) ket luan 0% bien so xe may doc duoc,
quy cho "khoang cach/do phan giai camera". Kiem tra lai bang mat tren chinh
cac anh crop da luu (docs/assets/plate_ocr_feasibility/video_*.jpg) cho thay
nhieu bien — ca xe may 2 dong — RO NET voi mat nguoi nhung EasyOCR doc sai
hoan toan. Gia thuyet that su: (1) bien 2 dong bi tach thanh nhieu doan text
roi rac, ham is_plausible_vn_plate() kiem tra TUNG doan rieng le nen khong
doan nao du dai de qua; (2) khong dung allowlist (EasyOCR doan ca chu thuong/
ky tu khac lam nhieu); (3) vung crop bottom_half cua ca xe (nguoi + xe may)
khong khoanh sat vao bien, nen nen sau (yem xe, banh xe...) lam nhieu EasyOCR.

Vong nay so sanh CUNG MOT BO MAU (tu sample_from_video/sample_from_auto_label
cua vong 1-3) giua:
  - baseline: dung lai y nguyen PlateOCRReader hien tai (bottom_half, upscale
    2.5x, khong allowlist, kiem tra plausible tren TUNG doan text rieng le)
  - candidate: khoanh vung bien sat hon bang mau (trang/vang) trong crop,
    OCR voi allowlist=alnum, GHEP cac doan text theo thu tu tren-xuong-duoi
    truoc khi kiem tra is_plausible_vn_plate (xu ly dung bien 2 dong)

Khong sua modules/plate_ocr.py — day la kiem chung doc lap truoc khi quyet
dinh co doi cach lam hay khong.

Usage:
    venv_tracking\\Scripts\\python.exe scripts\\eval\\test_plate_ocr_feasibility_v2.py --n-samples 120
"""

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(THIS_DIR))

from test_plate_ocr_feasibility import (  # noqa: E402
    DEFAULT_VIDEO, ROOT, sample_from_auto_label, sample_from_video,
)

sys.path.insert(0, str(ROOT))
from modules.plate_ocr import (  # noqa: E402
    PlateOCRReader, is_plausible_vn_plate, normalize_plate_text,
)

OUT_DIR = ROOT.parent / 'docs' / 'assets' / 'plate_ocr_feasibility_v2'
_ALLOWLIST = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'


def localize_plate(crop_bgr: np.ndarray):
    """Find the most plate-like rectangular blob inside a vehicle crop:
    light (white/yellow) background AND restricted to the bottom band where
    plates actually sit AND containing enough internal edge detail to be
    text, not a flat chrome/headlight reflection. Returns a tight sub-crop,
    or None if nothing plausible found (caller should fall back to the
    original crop)."""
    h, w = crop_bgr.shape[:2]
    if h < 6 or w < 6:
        return None

    y_floor = int(0.35 * h)  # plates sit in the lower part of the box, not the top
    band = crop_bgr[y_floor:, :]
    bh, bw = band.shape[:2]
    if bh < 6 or bw < 6:
        return None

    hsv = cv2.cvtColor(band, cv2.COLOR_BGR2HSV)
    white_mask = cv2.inRange(hsv, (0, 0, 140), (180, 60, 255))
    yellow_mask = cv2.inRange(hsv, (15, 80, 120), (40, 255, 255))
    mask = cv2.bitwise_or(white_mask, yellow_mask)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    gray = cv2.cvtColor(band, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 60, 150)

    best, best_score = None, 0.0
    frame_area = bw * bh
    for c in contours:
        x, y, cw, ch = cv2.boundingRect(c)
        area = cw * ch
        if area < 0.02 * frame_area or area > 0.7 * frame_area:
            continue
        aspect = cw / max(ch, 1)
        if not (0.6 <= aspect <= 4.5):  # VN plate: ~1:1 (2-line) to ~3.5:1 (1-line)
            continue
        edge_density = float(np.count_nonzero(edges[y:y + ch, x:x + cw])) / area
        if edge_density < 0.04:  # flat chrome/reflection has almost no internal edges
            continue
        score = area * edge_density
        if score > best_score:
            best_score, best = score, (x, y, cw, ch)

    if best is None:
        return None
    x, y, cw, ch = best
    pad = max(2, int(0.10 * max(cw, ch)))
    x0, y0 = max(0, x - pad), max(0, y - pad)
    x1, y1 = min(bw, x + cw + pad), min(bh, y + ch + pad)
    sub = band[y0:y1, x0:x1]
    if sub.size == 0 or sub.shape[0] < 6 or sub.shape[1] < 6:
        return None
    return sub


def ocr_concat_rows(reader, image_bgr, upscale, min_confidence):
    """Run EasyOCR, sort fragments top-to-bottom, concatenate, then validate
    as ONE string (handles 2-line plates split into separate detections)."""
    up = cv2.resize(image_bgr, None, fx=upscale, fy=upscale,
                     interpolation=cv2.INTER_CUBIC)
    try:
        raw = reader._reader.readtext(up, allowlist=_ALLOWLIST)
    except Exception:
        return [], None

    frags = []
    for bbox, text, conf in raw:
        if conf < min_confidence:
            continue
        ys = [p[1] for p in bbox]
        frags.append((sum(ys) / len(ys), text, conf))
    frags.sort(key=lambda f: f[0])

    raw_texts = [(t, round(float(c), 2)) for _y, t, c in frags]
    if not frags:
        return raw_texts, None

    concat = ''.join(normalize_plate_text(t) for _y, t, _c in frags)
    mean_conf = sum(c for _y, _t, c in frags) / len(frags)
    if is_plausible_vn_plate(concat):
        return raw_texts, (concat, mean_conf)
    return raw_texts, None


def evaluate(samples, reader, min_confidence, save_crops, out_dir):
    out_dir.mkdir(parents=True, exist_ok=True)
    per_class_total, per_class_baseline, per_class_candidate = {}, {}, {}
    examples = []
    saved = 0

    for frame, box, cls_name, sample_id in samples:
        per_class_total[cls_name] = per_class_total.get(cls_name, 0) + 1

        crop_bottom = reader._crop_region(frame, box)
        crop_full = frame[int(box[1]):int(box[3]), int(box[0]):int(box[2])]

        baseline_ok = False
        if crop_bottom is not None:
            up = cv2.resize(crop_bottom, None, fx=reader.upscale, fy=reader.upscale,
                             interpolation=cv2.INTER_CUBIC)
            try:
                raw = reader._reader.readtext(up)
            except Exception:
                raw = []
            for _b, text, conf in raw:
                if conf >= min_confidence and is_plausible_vn_plate(text):
                    baseline_ok = True
                    break

        # Localize on an already-upscaled crop, not the tiny raw pixels --
        # contour/edge-density analysis needs enough pixels to be reliable,
        # and a 3.5x upscale of an already-tiny raw crop is just blur, not
        # real detail (this was the bug in the first pass of this script).
        localized = None
        for src in (crop_bottom, crop_full):
            if src is None:
                continue
            pre_up = cv2.resize(src, None, fx=reader.upscale, fy=reader.upscale,
                                 interpolation=cv2.INTER_CUBIC)
            localized = localize_plate(pre_up)
            if localized is not None:
                break
        if localized is not None:
            candidate_target, up_scale = localized, 1.5
        elif crop_bottom is not None:
            candidate_target, up_scale = crop_bottom, reader.upscale
        else:
            candidate_target, up_scale = None, 1.0
        candidate_raw, candidate_result = ([], None)
        if candidate_target is not None:
            candidate_raw, candidate_result = ocr_concat_rows(
                reader, candidate_target, up_scale, min_confidence)

        if baseline_ok:
            per_class_baseline[cls_name] = per_class_baseline.get(cls_name, 0) + 1
        if candidate_result is not None:
            per_class_candidate[cls_name] = per_class_candidate.get(cls_name, 0) + 1

        if saved < save_crops and candidate_target is not None:
            cv2.imwrite(str(out_dir / f"{sample_id}_candidate.jpg"), candidate_target)
            saved += 1

        examples.append({
            'id': sample_id, 'class': cls_name,
            'baseline_plausible': baseline_ok,
            'candidate_raw_ocr': candidate_raw,
            'candidate_result': candidate_result,
            'used_localized_crop': localized is not None,
        })
        print(f"[{sample_id}] class={cls_name:10s} baseline={baseline_ok} "
              f"candidate={candidate_result} (localized={localized is not None})")

    summary = {}
    for cls_name in sorted(per_class_total):
        n = per_class_total[cls_name]
        b = per_class_baseline.get(cls_name, 0)
        c = per_class_candidate.get(cls_name, 0)
        summary[cls_name] = {
            'n': n,
            'baseline_plausible_pct': round(100 * b / n, 1),
            'candidate_plausible_pct': round(100 * c / n, 1),
        }
    return summary, examples


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--source', choices=['auto_label', 'video'], default='video')
    ap.add_argument('--video', default=str(DEFAULT_VIDEO))
    ap.add_argument('--weights', default=str(ROOT / 'weights' / 'yolo11s_vn.pt'))
    ap.add_argument('--n-samples', type=int, default=120)
    ap.add_argument('--min-confidence', type=float, default=0.3)
    ap.add_argument('--save-crops', type=int, default=40)
    args = ap.parse_args()

    reader = PlateOCRReader(min_confidence=args.min_confidence)
    if args.source == 'auto_label':
        samples = list(sample_from_auto_label(args.n_samples))
    else:
        samples = list(sample_from_video(Path(args.video), args.n_samples, args.weights))

    summary, examples = evaluate(samples, reader, args.min_confidence,
                                  args.save_crops, OUT_DIR)

    print("\n" + "=" * 70)
    print("VONG 4 — BASELINE (hien tai) vs CANDIDATE (khoanh vung + allowlist + ghep dong)")
    print("=" * 70)
    for cls_name, s in summary.items():
        print(f"{cls_name:10s} n={s['n']:3d}  baseline={s['baseline_plausible_pct']:5.1f}%  "
              f"candidate={s['candidate_plausible_pct']:5.1f}%")

    out_json = OUT_DIR / 'feasibility_v2_summary.json'
    with open(out_json, 'w', encoding='utf-8') as f:
        json.dump({'summary': summary, 'examples': examples}, f, indent=2, ensure_ascii=False)
    print(f"\nDa luu: {out_json}")


if __name__ == '__main__':
    main()
