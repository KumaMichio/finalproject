#!/usr/bin/env python3
"""
Buoc 1-2 cua plan cai thien bien so (xem plan da duyet): pseudo-label.

Dung detector xe co san (yolo11s_vn.pt, 5 class) + 2 model tham khao tu repo
cong dong (LP_detector.pt/LP_ocr.pt, chi dung OFFLINE de tu de xuat nhan,
giong dung "best_xl_ITD_v1.2.pt" lam auto-labeler hien co trong project —
xem .gitignore) de SINH NHAN DE XUAT tren video thuc cua minh. KHONG dung
truc tiep de gan nhan "dung" — nguoi review PHAI sua/xac nhan lai bang
labelImg truoc khi dua vao training, vi day la du lieu se dua vao san pham
thuc (yeu cau nguoi dung: "can so lieu du tin cay de dua vao product thuc").

Xuat 2 dataset rieng, dung dinh dang YOLO txt (cx cy w h, 0-1) khop voi
scripts/train/merge_detection_datasets.py de gop vao dataset hop nhat sau:

  data/plate_pseudolabel/        — full frame, nhan = 5 class xe co san
                                    (tu yolo11s_vn.pt) + class 5 "license_plate"
                                    (tu LP_detector.pt). Dung cho Stage 1
                                    (them class license_plate vao detector
                                    chinh).
  data/plate_chars_pseudolabel/  — crop bien so, nhan = tung ky tu (30 class,
                                    dung dung thu tu class cua LP_ocr.pt).
                                    Dung cho Stage 2 (model nhan dien ky tu
                                    rieng, YOLO11n).

Usage:
    venv_tracking\\Scripts\\python.exe scripts\\eval\\pseudo_label_plates.py \\
        --n-frames 400 --val-fraction 0.15
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
from test_plate_ocr_feasibility_v3 import deskew_variants, load_models  # noqa: E402
from modules.detector import ObjectDetector  # noqa: E402

LICENSE_PLATE_CLASS_ID = 5
PLATE_OUT = ROOT / 'data' / 'plate_pseudolabel'
CHARS_OUT = ROOT / 'data' / 'plate_chars_pseudolabel'

# Same 30-character order as LP_ocr.pt's own `.names` (verified when loading
# the model in round 5) -- keep Stage-2 class ids identical so any review
# notes / muscle memory from inspecting the reference model carries over.
CHAR_CLASS_NAMES = ['1', '2', '3', '4', '5', '6', '7', '8', '9', 'A', 'B', 'C',
                     'D', 'E', 'F', 'G', 'H', 'K', 'L', 'M', 'N', 'P', 'S', 'T',
                     'U', 'V', 'X', 'Y', 'Z', '0']
_CHAR_TO_ID = {name: i for i, name in enumerate(CHAR_CLASS_NAMES)}


def to_yolo_line(cls_id, x1, y1, x2, y2, img_w, img_h):
    cx = (x1 + x2) / 2 / img_w
    cy = (y1 + y2) / 2 / img_h
    w = (x2 - x1) / img_w
    h = (y2 - y1) / img_h
    return f"{cls_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}"


def save_sample(out_root: Path, split: str, stem: str, image, lines: list[str]):
    img_dir = out_root / 'images' / split
    lbl_dir = out_root / 'labels' / split
    img_dir.mkdir(parents=True, exist_ok=True)
    lbl_dir.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(img_dir / f"{stem}.jpg"), image)
    (lbl_dir / f"{stem}.txt").write_text('\n'.join(lines) + ('\n' if lines else ''))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--video', default=str(DEFAULT_VIDEO))
    ap.add_argument('--weights', default=str(ROOT / 'weights' / 'yolo11s_vn.pt'))
    ap.add_argument('--n-frames', type=int, default=400)
    ap.add_argument('--frame-stride', type=int, default=3,
                     help='Sample every Nth frame (consecutive frames of the '
                          'same vehicle add little new signal for labeling)')
    ap.add_argument('--val-fraction', type=float, default=0.15)
    ap.add_argument('--seed', type=int, default=0)
    args = ap.parse_args()

    print("Loading vehicle detector (yolo11s_vn.pt, 5 existing classes) ...")
    vehicle_detector = ObjectDetector(model_type=args.weights, conf_threshold=0.35)

    print("Loading plate detector + character model (reference, offline use only) ...")
    lp_detector, lp_ocr = load_models()

    rng = np.random.RandomState(args.seed)
    cap = cv2.VideoCapture(args.video)
    video_tag = Path(args.video).stem.replace('.', '_')

    n_plate_frames = 0
    n_char_crops = 0
    frame_idx = 0
    sample_idx = 0

    while cap.isOpened() and sample_idx < args.n_frames:
        ok, frame = cap.read()
        if not ok:
            break
        frame_idx += 1
        if frame_idx % args.frame_stride != 0:
            continue
        sample_idx += 1
        split = 'val' if rng.rand() < args.val_fraction else 'train'
        stem = f"{video_tag}_{frame_idx:06d}"

        img_h, img_w = frame.shape[:2]
        vehicle_dets = vehicle_detector.detect(frame)

        full_frame_lines = []
        for det in vehicle_dets:
            x1, y1, x2, y2 = det['box']
            full_frame_lines.append(
                to_yolo_line(det['class_id'], x1, y1, x2, y2, img_w, img_h))

            if det['class'] not in ('car', 'motorcycle', 'bus', 'truck'):
                continue
            vehicle_crop = frame[max(0, y1):y2, max(0, x1):x2]
            if vehicle_crop.size == 0:
                continue

            try:
                plate_det = lp_detector(vehicle_crop, size=640)
            except Exception:
                continue
            plates = plate_det.pandas().xyxy[0].values.tolist()
            if not plates:
                continue
            plates.sort(key=lambda p: -(p[2] - p[0]) * (p[3] - p[1]))
            px1, py1, px2, py2 = (int(v) for v in plates[0][:4])
            # Plate box back to full-frame coords for the Stage-1 label.
            full_frame_lines.append(to_yolo_line(
                LICENSE_PLATE_CLASS_ID, x1 + px1, y1 + py1, x1 + px2, y1 + py2,
                img_w, img_h))
            n_plate_frames += 1

            plate_crop = vehicle_crop[max(0, py1):py2, max(0, px1):px2]
            if plate_crop.size == 0 or plate_crop.shape[0] < 4 or plate_crop.shape[1] < 4:
                continue

            # Best-effort char boxes for Stage-2: take whichever deskew variant
            # finds the most character boxes (no 7-10 gate here -- partial
            # detections are still useful seeds for a human to complete).
            best_variant, best_boxes = None, []
            for variant in deskew_variants(plate_crop):
                try:
                    res = lp_ocr(variant)
                except Exception:
                    continue
                bb = res.pandas().xyxy[0].values.tolist()
                if len(bb) > len(best_boxes):
                    best_variant, best_boxes = variant, bb
            if best_variant is None or not best_boxes:
                continue

            ch, cw = best_variant.shape[:2]
            char_lines = []
            for b in best_boxes:
                char_name = str(b[-1])
                cls_id = _CHAR_TO_ID.get(char_name)
                if cls_id is None:
                    continue
                char_lines.append(to_yolo_line(cls_id, b[0], b[1], b[2], b[3], cw, ch))
            if char_lines:
                save_sample(CHARS_OUT, split, f"{stem}_v{x1}_{y1}", best_variant, char_lines)
                n_char_crops += 1

        save_sample(PLATE_OUT, split, stem, frame, full_frame_lines)

    cap.release()

    for out_root, names in ((PLATE_OUT, ['person', 'car', 'motorcycle', 'bus', 'truck', 'license_plate']),
                             (CHARS_OUT, CHAR_CLASS_NAMES)):
        yaml_path = Path(str(out_root) + '.yaml')
        yaml_path.write_text(
            f"path: {out_root.resolve().as_posix()}\n"
            f"train: images/train\nval: images/val\n"
            f"nc: {len(names)}\nnames: {names}\n"
        )
        print(f"Wrote {yaml_path}")

    print(f"\nSampled {sample_idx} frames -> {PLATE_OUT} "
          f"({n_plate_frames} frames have a candidate plate box)")
    print(f"Plate-character crops -> {CHARS_OUT} ({n_char_crops} crops with >=1 char box)")
    print("\nNHAN DE XUAT, CHUA PHAI NHAN DUNG -- review/sua bang labelImg truoc khi train:")
    print(f"  pip install labelImg")
    print(f"  labelImg {PLATE_OUT / 'images' / 'train'} ")
    print(f"  labelImg {CHARS_OUT / 'images' / 'train'} ")


if __name__ == '__main__':
    main()
