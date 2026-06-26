#!/usr/bin/env python3
"""
Buoc 0 (go/no-go) cho viec them OCR bien so lam tin hieu bo tro cho Re-ID
(xem plan: them OCR cho cross-camera Re-ID).

Khong dung vao pipeline chinh. Chi do thu: voi cac crop xe THAT (lay tu
bo nhan auto_label hoac tu video CCTV thuc), EasyOCR doc duoc bien so o ti
le nao — tach rieng theo loai xe (oto/xe tai/xe buyt vs xe may), vi gia
thuyet ban dau la xe may se kho doc hon nhieu o khoang cach camera giao
thong thong thuong.

Usage:
    # Nguon mac dinh: bo nhan auto_label (co san box, khong can chay detector)
    venv_tracking\\Scripts\\python.exe scripts\\eval\\test_plate_ocr_feasibility.py --n-samples 60

    # Nguon video thuc + detector YOLO thuc (cham hon, nhung dung dung
    # video/giao lo demo Pho Hue - Tran Khat Chan)
    venv_tracking\\Scripts\\python.exe scripts\\eval\\test_plate_ocr_feasibility.py ^
        --source video --n-samples 40
"""

import argparse
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np


def imread_unicode(path: Path):
    """cv2.imread() can't open paths with non-ASCII (e.g. Vietnamese
    diacritics) characters on Windows -- fopen() uses the system codepage.
    Read raw bytes ourselves and let cv2.imdecode() handle the actual
    image format."""
    try:
        data = np.fromfile(str(path), dtype=np.uint8)
    except OSError:
        return None
    if data.size == 0:
        return None
    return cv2.imdecode(data, cv2.IMREAD_COLOR)

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from modules.plate_ocr import PlateOCRReader, is_plausible_vn_plate  # noqa: E402

_CLASS_NAMES = ['person', 'car', 'motorcycle', 'bus', 'truck']
_VEHICLE_IDX = {1: 'car', 2: 'motorcycle', 3: 'bus', 4: 'truck'}

AUTO_LABEL_DIR = ROOT / 'data' / 'auto_label'
DEFAULT_VIDEO = (ROOT / 'data' / 'datasets' / 'traffic_intersection' /
                 '02. Dữ liệu camera giao thông' / 'Phố Huế Trần Khát Chân' /
                 '12h.10.9.22.mp4')
OUT_DIR = ROOT.parent / 'docs' / 'assets' / 'plate_ocr_feasibility'


def yolo_box_to_pixels(cx, cy, w, h, img_w, img_h):
    x1 = int((cx - w / 2) * img_w)
    y1 = int((cy - h / 2) * img_h)
    x2 = int((cx + w / 2) * img_w)
    y2 = int((cy + h / 2) * img_h)
    return [max(0, x1), max(0, y1), min(img_w, x2), min(img_h, y2)]


# Clip-name substrings confirmed (by visual inspection, see
# docs/assets/plate_ocr_feasibility/diagnose/) to be top-down / highway-gantry
# angle footage where a vehicle's plate is structurally not visible in the
# frame at all (camera looks almost straight down at the roof), unlike the
# oblique CCTV-style intersection cameras this project actually targets.
# Excluding these from the feasibility sample so it reflects the deployment
# camera angle (e.g. the Pho Hue demo camera), not a different sensor type
# that happens to share the same label format.
_BAD_ANGLE_CLIP_SUBSTRINGS = (
    'Giao_thông_nội_đô_HN',  # near-vertical overhead view (looks like a drone/tall mast)
    'NB-LC__ACC_Export',     # highway ACC dashcam-style export
    'Clip_tuyến_cao_tốc',    # highway clip, overhead gantry angle
)


def _is_oblique_cctv_clip(label_path: Path) -> bool:
    name = label_path.stem
    return not any(tag in name for tag in _BAD_ANGLE_CLIP_SUBSTRINGS)


def sample_from_auto_label(n_samples: int, min_box_px: int = 40, seed: int = 0,
                            oblique_only: bool = True):
    """Yield (image_bgr, box, class_name, sample_id) using existing YOLO labels
    (no detector run needed). Prefers larger boxes first (best legibility
    case), then fills the rest with a random sample for representativeness.

    oblique_only: skip label files from clips known to be top-down/highway
        angle (see _BAD_ANGLE_CLIP_SUBSTRINGS) -- those can never show a
        plate regardless of crop_region, and would unfairly drag down the
        legibility estimate for the camera angle this project actually uses.
    """
    label_dir = AUTO_LABEL_DIR / 'labels'
    image_dir = AUTO_LABEL_DIR / 'images'
    label_files = sorted(label_dir.glob('*.txt'))
    if oblique_only:
        label_files = [lp for lp in label_files if _is_oblique_cctv_clip(lp)]

    candidates = []  # (area_norm, label_path, line)
    rng = random.Random(seed)
    rng.shuffle(label_files)

    for lp in label_files[:3000]:  # cap how many label files we even open
        for line in lp.read_text(encoding='utf-8').splitlines():
            parts = line.split()
            if len(parts) != 5:
                continue
            cls_idx = int(parts[0])
            if cls_idx not in _VEHICLE_IDX:
                continue
            cx, cy, w, h = (float(x) for x in parts[1:])
            candidates.append((w * h, lp, (cls_idx, cx, cy, w, h)))
        if len(candidates) >= 4000:
            break

    candidates.sort(key=lambda c: c[0], reverse=True)
    n_top = n_samples // 2
    chosen = candidates[:n_top] + rng.sample(candidates[n_top:], min(n_samples - n_top, max(0, len(candidates) - n_top)))

    sample_id = 0
    for _area, lp, (cls_idx, cx, cy, w, h) in chosen:
        img_path = image_dir / (lp.stem + '.jpg')
        if not img_path.exists():
            continue
        img = imread_unicode(img_path)
        if img is None:
            continue
        img_h, img_w = img.shape[:2]
        box = yolo_box_to_pixels(cx, cy, w, h, img_w, img_h)
        if (box[2] - box[0]) < min_box_px or (box[3] - box[1]) < min_box_px:
            continue
        sample_id += 1
        yield img, box, _VEHICLE_IDX[cls_idx], f"autolabel_{sample_id:03d}_{lp.stem[:30]}"


def sample_from_video(video_path: Path, n_samples: int, weights: str):
    """Run the real YOLO detector on sampled frames of a real CCTV video
    (default: the Pho Hue - Tran Khat Chan demo footage)."""
    from modules.detector import ObjectDetector

    detector = ObjectDetector(model_type=weights, conf_threshold=0.35)
    cap = cv2.VideoCapture(str(video_path))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total <= 0:
        total = 10000
    frame_step = max(1, total // (n_samples * 3))

    sample_id = 0
    frame_idx = 0
    while cap.isOpened() and sample_id < n_samples * 3:
        ok, frame = cap.read()
        if not ok:
            break
        if frame_idx % frame_step == 0:
            detections = detector.detect(frame)
            for det in detections:
                cls = det.get('class')
                if cls not in _VEHICLE_IDX.values():
                    continue
                box = det['box']
                if (box[2] - box[0]) < 40 or (box[3] - box[1]) < 40:
                    continue
                sample_id += 1
                yield frame, box, cls, f"video_{sample_id:03d}_f{frame_idx}"
        frame_idx += 1
    cap.release()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--source', choices=['auto_label', 'video'], default='auto_label')
    ap.add_argument('--video', default=str(DEFAULT_VIDEO))
    ap.add_argument('--weights', default=str(ROOT / 'weights' / 'yolo11s_vn.pt'))
    ap.add_argument('--n-samples', type=int, default=60)
    ap.add_argument('--min-confidence', type=float, default=0.3,
                     help='Lowered from the production default (0.5) for this '
                          'feasibility probe, so we can see near-miss reads too.')
    ap.add_argument('--save-crops', type=int, default=20,
                     help='Save up to this many crop images (with OCR overlay) for visual inspection.')
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    reader = PlateOCRReader(min_confidence=args.min_confidence)

    if args.source == 'auto_label':
        samples = sample_from_auto_label(args.n_samples)
    else:
        samples = sample_from_video(Path(args.video), args.n_samples, args.weights)

    per_class_total = Counter()
    per_class_any_text = Counter()
    per_class_plausible = Counter()
    raw_examples = []
    saved = 0

    for frame, box, cls_name, sample_id in samples:
        per_class_total[cls_name] += 1

        crop = reader._crop_region(frame, box)
        any_text = False
        plausible_result = None
        raw_texts = []

        if crop is not None:
            import cv2 as _cv2
            up = _cv2.resize(crop, None, fx=reader.upscale, fy=reader.upscale,
                              interpolation=_cv2.INTER_CUBIC)
            try:
                raw = reader._reader.readtext(up)
            except Exception as e:  # pragma: no cover - diagnostic path
                raw = []
                print(f"[{sample_id}] OCR ERROR: {e}")
            for _b, text, conf in raw:
                raw_texts.append((text, round(float(conf), 2)))
                if conf >= args.min_confidence:
                    any_text = True
                if conf >= args.min_confidence and is_plausible_vn_plate(text):
                    if plausible_result is None or conf > plausible_result[1]:
                        plausible_result = (text, conf)

            if saved < args.save_crops:
                out_path = OUT_DIR / f"{sample_id}.jpg"
                _cv2.imwrite(str(out_path), up)
                saved += 1

        if any_text:
            per_class_any_text[cls_name] += 1
        if plausible_result is not None:
            per_class_plausible[cls_name] += 1

        raw_examples.append({
            'id': sample_id, 'class': cls_name, 'box': box,
            'raw_ocr': raw_texts, 'plausible_plate': plausible_result,
        })
        print(f"[{sample_id}] class={cls_name:10s} raw_ocr={raw_texts} "
              f"plausible={plausible_result}")

    print("\n" + "=" * 60)
    print("KET QUA BUOC 0 — TI LE OCR DOC DUOC BIEN SO (theo loai xe)")
    print("=" * 60)
    summary = {}
    for cls_name in sorted(per_class_total):
        total = per_class_total[cls_name]
        any_t = per_class_any_text[cls_name]
        plaus = per_class_plausible[cls_name]
        print(f"{cls_name:10s}  n={total:3d}  co_text_OCR={any_t:3d} "
              f"({100*any_t/total:5.1f}%)  bien_so_hop_le={plaus:3d} "
              f"({100*plaus/total:5.1f}%)")
        summary[cls_name] = {
            'n': total, 'any_ocr_text': any_t, 'plausible_plate': plaus,
            'any_ocr_pct': round(100 * any_t / total, 1),
            'plausible_pct': round(100 * plaus / total, 1),
        }

    out_json = OUT_DIR / 'feasibility_summary.json'
    with open(out_json, 'w', encoding='utf-8') as f:
        json.dump({'summary': summary, 'examples': raw_examples}, f, indent=2, ensure_ascii=False)
    print(f"\nDa luu chi tiet: {out_json}")
    print(f"Da luu {saved} anh crop (sau upscale) de kiem tra bang mat: {OUT_DIR}")


if __name__ == '__main__':
    main()
