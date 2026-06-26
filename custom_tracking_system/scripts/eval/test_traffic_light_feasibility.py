#!/usr/bin/env python3
"""
Kiem tra kha thi nhan dien mau den giao thong tren video CCTV thuc.

Trich vai chuc khung hinh tu video Pho Hue o cac thoi diem khac nhau, luu
toan canh + 1 vung crop quanh vi tri den (toa do tay, do tu 1 khung hinh
mau) de xem bang mat: den co nhin thay duoc khong, co phan biet duoc
do/xanh/vang khong. Theo dung tinh than test_plate_ocr_feasibility.py da co
trong project (kiem tra kha thi truoc khi dau tu xay pipeline).

Usage:
    python scripts/eval/test_traffic_light_feasibility.py --n-samples 30
"""

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

project_root = Path(__file__).resolve().parents[2]
sys.path.append(str(project_root))

DEFAULT_VIDEO = (project_root / 'data' / 'datasets' / 'traffic_intersection' /
                  '02. Dữ liệu camera giao thông' / 'Phố Huế Trần Khát Chân' /
                  '12h.10.9.22.mp4')

# Vung crop quanh den giao thong nhin thay duoc trong anh mau
# (component_for_final/figures_results/02_cctv_pho_hue_sample_frame.jpg) —
# uoc luong ban dau, se chinh lai sau khi xem ket qua round 1.
# Hai vi tri quan sat duoc trong anh mau: 1 den/dong ho dem giay ben phai
# (gan "5 1"), 1 den gan cau vuot phia tren-giua.
CROP_CANDIDATES = {
    'right_pole': (700, 60, 800, 180),   # (x1, y1, x2, y2) — coc den ben phai, gan le duong
    'overpass':   (330, 0, 430, 70),     # den gan cau vuot, phia tren-giua
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--video', type=str, default=str(DEFAULT_VIDEO))
    parser.add_argument('--n-samples', type=int, default=30)
    parser.add_argument('--out-dir', type=str,
                         default='docs/assets/traffic_light_feasibility')
    args = parser.parse_args()

    video_path = Path(args.video)
    if not video_path.exists():
        print(f"Khong tim thay video: {video_path}")
        sys.exit(1)

    out_dir = project_root / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / 'full').mkdir(exist_ok=True)
    (out_dir / 'crops').mkdir(exist_ok=True)

    cap = cv2.VideoCapture(str(video_path))
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"Video: {n_frames} frames, {fps:.1f} fps, {w}x{h}")

    sample_indices = np.linspace(0, n_frames - 1, args.n_samples, dtype=int)

    for i, frame_idx in enumerate(sample_indices):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_idx))
        ret, frame = cap.read()
        if not ret:
            continue

        full_path = out_dir / 'full' / f'frame_{i:03d}_idx{frame_idx}.jpg'
        cv2.imwrite(str(full_path), frame)

        for name, (x1, y1, x2, y2) in CROP_CANDIDATES.items():
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)
            crop = frame[y1:y2, x1:x2]
            if crop.size == 0:
                continue
            # Phong to 4x de de xem bang mat (vung crop rat nho)
            crop_big = cv2.resize(crop, None, fx=4, fy=4,
                                   interpolation=cv2.INTER_NEAREST)
            crop_path = out_dir / 'crops' / f'frame_{i:03d}_idx{frame_idx}_{name}.jpg'
            cv2.imwrite(str(crop_path), crop_big)

    cap.release()
    print(f"Da luu {len(sample_indices)} khung hinh vao {out_dir}")
    print("Xem thu muc 'full/' de kiem tra toa do crop CROP_CANDIDATES co dung "
          "vi tri den khong; xem 'crops/' (da phong to 4x) de danh gia mau "
          "sac co phan biet duoc bang mat khong.")


if __name__ == '__main__':
    main()
