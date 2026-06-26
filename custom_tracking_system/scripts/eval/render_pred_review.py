#!/usr/bin/env python3
"""
Render video co ve box + global_id tu chinh file pred_<CAM_ID>.txt da sinh
boi `main.py --eval-output` — dung de nguoi xem phat hien doan ID-switch
truoc khi sua tay thanh gt_<CAM_ID>.txt (xem apply_id_corrections.py).

Doc lai du lieu da co san (khong chay lai model) de dam bao ID ve len video
KHOP CHINH XAC 100% voi ID trong file pred — tranh lap lai loi da gap voi
visualize_goal_predictions.py (script do ve track_id noi bo cua ByteTrack,
khac hoan toan global_id ma pred_<CAM_ID>.txt dung, nen khong dung duoc cho
muc dich nay).

Usage:
    python scripts/eval/render_pred_review.py --video <path.mp4> \
        --pred data/eval_real/pred_CAM_PHO_HUE_REAL.txt \
        --out docs/assets/eval_real/review.mp4
"""

import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

project_root = Path(__file__).resolve().parents[2]
sys.path.append(str(project_root))


def _color_for_id(gid: int):
    rng = np.random.RandomState(gid)
    return tuple(int(c) for c in rng.randint(64, 255, size=3))


def load_pred(pred_path: str):
    by_frame = defaultdict(list)
    with open(pred_path, newline='', encoding='utf-8') as f:
        for row in csv.reader(f):
            if not row:
                continue
            frame, gid, x1, y1, x2, y2, cls = row
            by_frame[int(frame)].append({
                'id': int(gid),
                'box': (float(x1), float(y1), float(x2), float(y2)),
                'class': cls,
            })
    return by_frame


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--video', required=True)
    ap.add_argument('--pred', required=True)
    ap.add_argument('--out', required=True)
    args = ap.parse_args()

    by_frame = load_pred(args.pred)
    if not by_frame:
        print(f"File pred rong hoac khong doc duoc: {args.pred}")
        sys.exit(1)
    max_frame = max(by_frame.keys())
    print(f"Da doc {sum(len(v) for v in by_frame.values())} dong, "
          f"{max_frame} khung hinh tu {args.pred}")

    cap = cv2.VideoCapture(args.video)
    assert cap.isOpened(), f"Khong mo duoc video: {args.video}"
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 10.0

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(args.out, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))

    frame_idx = 0
    while frame_idx <= max_frame:
        ret, frame = cap.read()
        if not ret:
            break
        frame_idx += 1

        for t in by_frame.get(frame_idx, []):
            x1, y1, x2, y2 = (int(v) for v in t['box'])
            color = _color_for_id(t['id'])
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            label = f"ID:{t['id']} {t['class']}"
            cv2.putText(frame, label, (x1, max(0, y1 - 6)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        cv2.putText(frame, f"frame {frame_idx}", (8, h - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        writer.write(frame)

    cap.release()
    writer.release()
    print(f"Da ghi {frame_idx} khung hinh vao {args.out}")


if __name__ == '__main__':
    main()
