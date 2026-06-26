#!/usr/bin/env python3
"""
Cong cu BAN TU DONG kiem tra/goi y vach son tren duong (vach dung den do,
vach nguoi di bo) tu video CCTV thuc — thay cho cach lam hien tai la click
tay polygon tren DUNG MOT khung hinh mau (xem ghi chu "best-effort, khop truc
quan tu 1 khung hinh" trong camera_config_pho_hue.yaml).

Khong tu dong hoa hoan toan (khong tu xuat ROI moi roi ghi thang vao config)
— ket qua chi la GOI Y de nguoi van hanh doi chieu/xac nhan, vi sai vi tri
vach o day dan toi buoc toi sai mot phuong tien, rui ro cao hon loi ich "tu
dong hoan toan" mang lai.

Cach lam:
  1. Lay median pixel-wise qua N khung hinh cach deu trong video -> anh nen
     gan nhu khong co xe (xe di chuyen nen khong xuat hien o cung vi tri qua
     nhieu khung hinh; rieng doan video co den do/xe dung doi lau co the de
     lai "bong" nhe, van con tot hon mot khung hinh don).
  2. Voi MOI ROI da khai bao tay trong config (vach_dung_den_do,
     vach_nguoi_di_bo, lan_o_to, lan_xe_may): crop anh nen quanh vung do (co
     bien dem) THAY VI tim tren toan khung hinh -> loai bo hau het nhieu tu
     cau vuot, toa nha, xe do ven duong, via he.
  3. Trong vung crop: threshold mau (trang/vang) + Canny + Hough Line
     Transform tim doan thang ung vien, gop cac doan gan nhau cung huong.
  4. Ve canh ROI da khai bao tay (mau xanh) CHONG LEN cac duong ung vien tim
     duoc (mau do/cam) de nguoi van hanh so sanh truc tiep — lech nhieu nghia
     la ROI tay can ve lai.

Usage:
    venv_tracking\\Scripts\\python.exe scripts\\eval\\detect_road_lines.py --n-frames 60
"""

import argparse
from pathlib import Path

import cv2
import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_VIDEO = (ROOT / 'data' / 'datasets' / 'traffic_intersection' /
                  '02. Dữ liệu camera giao thông' / 'Phố Huế Trần Khát Chân' /
                  '12h.10.9.22.mp4')
DEFAULT_CONFIG = ROOT / 'config' / 'camera_config_pho_hue.yaml'
OUT_DIR = ROOT.parent / 'docs' / 'assets' / 'road_lines'
CAM_ID = 'CAM_PHO_HUE_REAL'


def compute_background(video_path: Path, n_frames: int) -> np.ndarray:
    """Pixel-wise median over n_frames evenly spaced samples."""
    cap = cv2.VideoCapture(str(video_path))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total <= 0:
        total = n_frames * 10
    step = max(1, total // n_frames)

    frames = []
    idx = 0
    while cap.isOpened() and len(frames) < n_frames:
        ok, frame = cap.read()
        if not ok:
            break
        if idx % step == 0:
            frames.append(frame)
        idx += 1
    cap.release()

    if not frames:
        raise RuntimeError(f"No frames read from {video_path}")
    stack = np.stack(frames, axis=0)
    return np.median(stack, axis=0).astype(np.uint8)


def load_rois(config_path: Path):
    """Pull every named polygon zone for CAM_ID out of the config -- the
    rois/, lane_discipline/ sections share the same {name, polygon} shape."""
    with open(config_path, encoding='utf-8') as f:
        cfg = yaml.safe_load(f)

    zones = []
    rois_zones = cfg.get('rois', {}).get(CAM_ID, {}).get('zones', [])
    zones.extend(rois_zones)
    # lane_discipline nests under incident_detection in this config, not top-level.
    lane_zones = (cfg.get('incident_detection', {}).get('lane_discipline', {})
                  .get('zones', {}).get(CAM_ID, []))
    zones.extend(lane_zones)
    return zones


def detect_paint_mask(bg_bgr: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(bg_bgr, cv2.COLOR_BGR2HSV)
    white = cv2.inRange(hsv, (0, 0, 150), (180, 50, 255))
    yellow = cv2.inRange(hsv, (15, 60, 130), (40, 255, 255))
    mask = cv2.bitwise_or(white, yellow)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    return mask


def detect_lines_in_crop(crop_bgr, min_line_length, hough_thresh=20, max_line_gap=10):
    mask = detect_paint_mask(crop_bgr)
    masked_gray = cv2.bitwise_and(cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY), mask)
    edges = cv2.Canny(masked_gray, 50, 150, apertureSize=3)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=hough_thresh,
                             minLineLength=min_line_length, maxLineGap=max_line_gap)
    return ([] if lines is None else [tuple(l[0]) for l in lines]), mask


def merge_collinear(lines, angle_tol_deg=10, dist_tol=10):
    def angle(l):
        x1, y1, x2, y2 = l
        return np.degrees(np.arctan2(y2 - y1, x2 - x1)) % 180

    def midpoint(l):
        x1, y1, x2, y2 = l
        return ((x1 + x2) / 2, (y1 + y2) / 2)

    groups = []
    for ln in lines:
        a = angle(ln)
        mx, my = midpoint(ln)
        placed = False
        for g in groups:
            ga = angle(g['rep'])
            gmx, gmy = midpoint(g['rep'])
            angle_diff = min(abs(a - ga), 180 - abs(a - ga))
            if angle_diff <= angle_tol_deg and np.hypot(mx - gmx, my - gmy) <= dist_tol * 3:
                g['members'].append(ln)
                placed = True
                break
        if not placed:
            groups.append({'rep': ln, 'members': [ln]})

    merged = []
    for g in groups:
        pts = np.array([[m[0], m[1]] for m in g['members']] +
                        [[m[2], m[3]] for m in g['members']], dtype=np.float32)
        mean = pts.mean(axis=0)
        centered = pts - mean
        _, _, vt = np.linalg.svd(centered, full_matrices=False)
        direction = vt[0]
        proj = centered @ direction
        p1 = mean + direction * proj.min()
        p2 = mean + direction * proj.max()
        length = np.hypot(*(p2 - p1))
        if length >= 20:
            merged.append((int(p1[0]), int(p1[1]), int(p2[0]), int(p2[1]), len(g['members'])))
    merged.sort(key=lambda m: -m[4])
    return merged


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--video', default=str(DEFAULT_VIDEO))
    ap.add_argument('--config', default=str(DEFAULT_CONFIG))
    ap.add_argument('--n-frames', type=int, default=60)
    ap.add_argument('--padding', type=int, default=60,
                     help='Pixels of context kept around each ROI bounding box')
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Computing background from {args.n_frames} frames ...")
    bg = compute_background(Path(args.video), args.n_frames)
    cv2.imwrite(str(OUT_DIR / 'background.jpg'), bg)
    h, w = bg.shape[:2]

    zones = load_rois(Path(args.config))
    print(f"Loaded {len(zones)} ROI zones from config: {[z['name'] for z in zones]}")

    for zone in zones:
        name = zone['name']
        poly = np.array(zone['polygon'], dtype=np.int32)
        x0, y0 = poly[:, 0].min(), poly[:, 1].min()
        x1, y1 = poly[:, 0].max(), poly[:, 1].max()
        px0, py0 = max(0, x0 - args.padding), max(0, y0 - args.padding)
        px1, py1 = min(w, x1 + args.padding), min(h, y1 + args.padding)

        crop = bg[py0:py1, px0:px1]
        min_len = max(20, int(0.25 * min(crop.shape[0], crop.shape[1])))
        raw_lines, mask = detect_lines_in_crop(crop, min_line_length=min_len)
        merged = merge_collinear(raw_lines)

        overlay = crop.copy()
        manual_poly_local = poly - np.array([px0, py0])
        cv2.polylines(overlay, [manual_poly_local], True, (0, 255, 0), 2)

        for i, (lx1, ly1, lx2, ly2, n_members) in enumerate(merged[:8]):
            color = (0, 0, 255) if n_members >= 2 else (0, 165, 255)
            cv2.line(overlay, (lx1, ly1), (lx2, ly2), color, 2)
            cv2.putText(overlay, str(i), (lx1, ly1 - 5), cv2.FONT_HERSHEY_SIMPLEX,
                        0.45, color, 1, cv2.LINE_AA)

        out_path = OUT_DIR / f"{name}_compare.jpg"
        cv2.imwrite(str(out_path), overlay)
        cv2.imwrite(str(OUT_DIR / f"{name}_mask.jpg"), mask)
        print(f"\n[{name}] crop=({px0},{py0})-({px1},{py1})  "
              f"manual polygon (local coords)={manual_poly_local.tolist()}")
        print(f"  top candidate lines (local coords, sorted by n_segments_merged):")
        for i, (lx1, ly1, lx2, ly2, n_members) in enumerate(merged[:8]):
            length = np.hypot(lx2 - lx1, ly2 - ly1)
            print(f"    [{i}] ({lx1},{ly1})-({lx2},{ly2})  len={length:.0f}px  n_merged={n_members}")
        print(f"  saved: {out_path}")


if __name__ == '__main__':
    main()
