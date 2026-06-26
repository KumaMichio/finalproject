#!/usr/bin/env python3
"""
Buoc bo sung cua plan cai thien bien so: synthetic degradation cho Stage 2
(character detector), bu cho cac class qua hiem trong du lieu pseudo-label
thuc (xem data/plate_chars_pseudolabel — 6 chu hoan toan vang: C,S,T,U,V,Y;
8 chu qua it: B,K,L,M,N,P,X,Z).

Nguon: dataset cong khai cua winter2897
(Real-time-Auto-License-Plate-Recognition-with-Jetson-Nano, anh chup
gan/ro net, 760x560, 36 class = 0-9 + A-Z theo thu tu bang chu cai —
xac nhan bang mat qua anh tham chieu "0.jpg" co san trong dataset).
KHONG dung truc tiep anh ro net de train (khac han do phan giai camera
Pho Hue thuc te) — downscale+blur+JPEG-recompress de mo phong dung muc do
mat chi tiet do khoang cach camera thuc, giu nguyen toa do box (chi giam
chi tiet trong cung khung hinh, khong doi kich thuoc/ti le khung hinh).

Remap 36 class -> dung 30 class cua LP_ocr.pt da dung o Stage 2 (loai
I,J,O,Q,R,W — trung voi cac chu hiem trong CA dataset nay, cung ly do
model tham khao da loai tu dau: de tranh nham trong bien so VN thuc te).

Usage:
    venv_tracking\\Scripts\\python.exe scripts\\train\\prepare_synthetic_char_data.py \\
        --source <duong dan toi yolo_plate_ocr_dataset da giai nen> \\
        --variants-per-image 3
"""

import argparse
import random
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'data' / 'plate_chars_external'

# Same order as scripts/eval/pseudo_label_plates.py::CHAR_CLASS_NAMES --
# keep both in sync, this IS the Stage-2 class scheme.
CHAR_CLASS_NAMES = ['1', '2', '3', '4', '5', '6', '7', '8', '9', 'A', 'B', 'C',
                     'D', 'E', 'F', 'G', 'H', 'K', 'L', 'M', 'N', 'P', 'S', 'T',
                     'U', 'V', 'X', 'Y', 'Z', '0']
_MINE_ID = {name: i for i, name in enumerate(CHAR_CLASS_NAMES)}

_EXT_CHAR_NAMES = [str(d) for d in range(10)] + [chr(ord('A') + i) for i in range(26)]
EXT_TO_MINE = {i: _MINE_ID.get(ch) for i, ch in enumerate(_EXT_CHAR_NAMES)}


def remap_label_lines(label_path: Path) -> list[str]:
    out_lines = []
    for line in label_path.read_text().splitlines():
        parts = line.split()
        if len(parts) != 5:
            continue
        ext_id = int(parts[0])
        mine_id = EXT_TO_MINE.get(ext_id)
        if mine_id is None:
            continue  # I/J/O/Q/R/W -- excluded from our 30-class scheme
        out_lines.append(f"{mine_id} {' '.join(parts[1:])}")
    return out_lines


def degrade(image: np.ndarray, scale: float, jpeg_quality: int) -> np.ndarray:
    """Downscale (destroys detail, area-interpolation ~ sensor averaging)
    then upscale back to original size (cubic) -- this leaves box
    coordinates valid (same frame dimensions) while approximating what a
    real low-resolution capture of the same scene would look like, as
    opposed to just displaying the same detail smaller."""
    h, w = image.shape[:2]
    small = cv2.resize(image, (max(1, int(w * scale)), max(1, int(h * scale))),
                        interpolation=cv2.INTER_AREA)
    back = cv2.resize(small, (w, h), interpolation=cv2.INTER_CUBIC)
    ok, enc = cv2.imencode('.jpg', back, [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality])
    return cv2.imdecode(enc, cv2.IMREAD_COLOR) if ok else back


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--source', required=True,
                     help='Path to the extracted yolo_plate_ocr_dataset directory')
    ap.add_argument('--variants-per-image', type=int, default=3)
    ap.add_argument('--scale-range', type=float, nargs=2, default=(0.12, 0.45),
                     help='Downscale factor range applied before upscaling back '
                          '(lower = more destroyed detail = harder/smaller-equivalent)')
    ap.add_argument('--val-fraction', type=float, default=0.15)
    ap.add_argument('--seed', type=int, default=0)
    args = ap.parse_args()

    src = Path(args.source)
    rng = random.Random(args.seed)

    # The source dataset mixes genuine Vietnamese plates ("xemay*" motorcycle,
    # "CarLongPlate*" car) with foreign plates from other sub-sources
    # ("PlateBaza*" -- Croatian-style, "iwt*" -- Czech, confirmed by visual
    # inspection) -- foreign plate fonts/formats would teach the wrong visual
    # patterns, so only the Vietnamese-confirmed prefixes are used here.
    VN_PREFIXES = ('xemay', 'carlongplate')

    n_images, n_variants, n_skipped, n_foreign = 0, 0, 0, 0
    for label_path in sorted(src.glob('*.txt')):
        if label_path.stem == '0':
            continue  # the A-Z/0-9 reference chart, not a real plate photo
        stem_lower = label_path.stem.lower()
        if not any(p in stem_lower for p in VN_PREFIXES):
            n_foreign += 1
            continue
        img_path = label_path.with_suffix('.jpg')
        if not img_path.exists():
            continue

        lines = remap_label_lines(label_path)
        if not lines:
            n_skipped += 1
            continue
        n_images += 1

        image = cv2.imread(str(img_path))
        if image is None:
            continue

        split = 'val' if rng.random() < args.val_fraction else 'train'
        img_dir = OUT / 'images' / split
        lbl_dir = OUT / 'labels' / split
        img_dir.mkdir(parents=True, exist_ok=True)
        lbl_dir.mkdir(parents=True, exist_ok=True)

        for v in range(args.variants_per_image):
            scale = rng.uniform(*args.scale_range)
            quality = rng.randint(25, 55)
            degraded = degrade(image, scale, quality)
            stem = f"{label_path.stem}_d{v}"
            cv2.imwrite(str(img_dir / f"{stem}.jpg"), degraded)
            (lbl_dir / f"{stem}.txt").write_text('\n'.join(lines) + '\n')
            n_variants += 1

    yaml_path = Path(str(OUT) + '.yaml')
    yaml_path.write_text(
        f"path: {OUT.resolve().as_posix()}\n"
        f"train: images/train\nval: images/val\n"
        f"nc: {len(CHAR_CLASS_NAMES)}\nnames: {CHAR_CLASS_NAMES}\n"
    )

    print(f"Filtered out as non-Vietnamese (PlateBaza/iwt prefixes): {n_foreign}")
    print(f"Source images with >=1 usable (non-excluded-letter) box: {n_images}")
    print(f"Skipped (no usable box after remap, e.g. only I/J/O/Q/R/W): {n_skipped}")
    print(f"Degraded variants written: {n_variants} -> {OUT}")
    print(f"Dataset config: {yaml_path}")


if __name__ == '__main__':
    main()
