#!/usr/bin/env python3
"""
Cong cu chon toa do polygon bang tay tren 1 khung hinh mau — dung de dinh
nghia stop-line zone (RED_LIGHT_VIOLATION), lane zone (LANE_VIOLATION), hay
no-stop zone (ILLEGAL_STOP_ZONE) trong camera_config_pho_hue.yaml.

Click lien tiep cac diem (theo thu tu quanh polygon), nhan 'u' de undo diem
cuoi, nhan 'q' hoac ESC de in ra danh sach toa do cuoi cung (dang YAML, copy
thang vao file config).

Usage:
    python scripts/tools/pick_polygon_points.py --frame path/to/frame.jpg
    python scripts/tools/pick_polygon_points.py --video path/to/video.mp4 --frame-idx 84
"""

import argparse
import sys
from pathlib import Path

import cv2

project_root = Path(__file__).resolve().parents[2]
sys.path.append(str(project_root))

_points: list[tuple[int, int]] = []


def _on_mouse(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        _points.append((x, y))
        print(f"  + ({x}, {y})")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--frame', type=str, default=None,
                         help='Duong dan anh tinh de chon diem tren do')
    parser.add_argument('--video', type=str, default=None,
                         help='Hoac trich 1 khung hinh tu video')
    parser.add_argument('--frame-idx', type=int, default=0)
    args = parser.parse_args()

    if args.frame:
        img = cv2.imread(args.frame)
        if img is None:
            print(f"Khong doc duoc anh: {args.frame}")
            sys.exit(1)
    elif args.video:
        cap = cv2.VideoCapture(args.video)
        cap.set(cv2.CAP_PROP_POS_FRAMES, args.frame_idx)
        ret, img = cap.read()
        cap.release()
        if not ret:
            print(f"Khong doc duoc frame {args.frame_idx} tu {args.video}")
            sys.exit(1)
    else:
        print("Can --frame hoac --video")
        sys.exit(1)

    print("Click lien tiep cac diem quanh polygon. 'u'=undo, 'q'/ESC=xong.")
    window = "pick_polygon_points — q de ket thuc"
    cv2.namedWindow(window)
    cv2.setMouseCallback(window, _on_mouse)

    while True:
        disp = img.copy()
        for i, p in enumerate(_points):
            cv2.circle(disp, p, 4, (0, 0, 255), -1)
            cv2.putText(disp, str(i), (p[0] + 6, p[1] - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        if len(_points) >= 2:
            cv2.polylines(disp, [cv2.UMat(
                __import__('numpy').array(_points))], True, (0, 255, 0), 1)
        cv2.imshow(window, disp)
        key = cv2.waitKey(20) & 0xFF
        if key in (ord('q'), 27):
            break
        if key == ord('u') and _points:
            removed = _points.pop()
            print(f"  - undo {removed}")

    cv2.destroyAllWindows()

    print("\nToa do da chon (dang YAML, dan thang vao polygon:):")
    print("polygon: [" + ", ".join(f"[{x}, {y}]" for x, y in _points) + "]")


if __name__ == '__main__':
    main()
