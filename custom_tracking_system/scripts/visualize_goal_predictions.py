"""
visualize_goal_predictions.py — chạy pipeline trên video, lưu output video có overlay
goal prediction để kiểm tra thị giác.

Usage:
  python scripts/visualize_goal_predictions.py --video <path.mp4> --frames 300 --out out_goal.mp4
"""

import sys, time, argparse
from pathlib import Path
from collections import defaultdict

import numpy as np
import cv2

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from modules.detector import ObjectDetector
from modules.tracker import ByteTrackWrapper
from modules.goal_classifier import GoalClassifier

_DIR_COLOR_BGR = {
    'straight': (200, 200, 200),
    'left':     (0, 165, 255),
    'right':    (0, 220, 0),
    'u_turn':   (0, 0, 255),
}
_DIR_LABEL = {'straight': 'S', 'left': '<L', 'right': 'R>', 'u_turn': 'U!'}

# Track only high-confidence non-straight predictions for log
LOG_MIN_CONF = 0.50


def run(video_path: str, max_frames: int, out_path: str):
    cap = cv2.VideoCapture(video_path)
    assert cap.isOpened(), f"Cannot open: {video_path}"
    W  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H  = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 6.0

    writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (W, H))

    detector = ObjectDetector(str(ROOT/'weights'/'yolo11s_vn.pt'), conf_threshold=0.35)
    tracker  = ByteTrackWrapper(frame_rate=int(fps))
    gc       = GoalClassifier(
        str(ROOT/'weights'/'goal_classifier_real.pkl'),
        str(ROOT/'weights'/'goal_scaler_real.pkl'),
        str(ROOT/'weights'/'goal_label_encoder_real.pkl'),
        str(ROOT/'weights'/'goal_feature_names_real.json'),
    )

    print(f"Processing {max_frames} frames → {out_path}")
    print("="*60)

    # Per-track prediction history (to detect changes)
    last_pred = {}
    # All non-straight events for summary
    events = []

    frame_idx = 0
    while frame_idx < max_frames:
        ret, frame_bgr = cap.read()
        if not ret:
            break
        frame_idx += 1
        now_ts = frame_idx / fps

        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        detections = detector.detect(frame_rgb)
        tracks = tracker.update(detections, frame_rgb)

        out_frame = frame_bgr.copy()

        for t in tracks:
            tid  = t['track_id']
            box  = t['box']
            cls  = t.get('class', 'car')
            x1, y1, x2, y2 = box

            res = gc.update_and_predict(tid, box, cls, now_ts)

            # Draw bounding box (thin, gray if no pred yet)
            if res is None:
                cv2.rectangle(out_frame, (x1,y1), (x2,y2), (80,80,80), 1)
                cv2.putText(out_frame, f"#{tid}", (x1, y1-4),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (80,80,80), 1)
                continue

            direction = res['direction']
            conf      = res['confidence']
            color     = _DIR_COLOR_BGR[direction]
            lbl       = _DIR_LABEL[direction]

            # Thicker box for non-straight confident predictions
            thickness = 3 if (direction != 'straight' and conf >= 0.50) else 1
            cv2.rectangle(out_frame, (x1,y1), (x2,y2), color, thickness)

            # Label: ID + direction + confidence
            text = f"#{tid} {lbl} {conf:.0%}"
            cv2.putText(out_frame, text, (x1, y1-4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)

            # Log direction change events
            prev = last_pred.get(tid)
            if prev != direction and direction != 'straight' and conf >= LOG_MIN_CONF:
                events.append({
                    'frame': frame_idx,
                    'track_id': tid,
                    'class': cls,
                    'direction': direction,
                    'conf': conf,
                    'probs': {k: round(v,3) for k,v in res['probabilities'].items()},
                })
                print(f"  [f{frame_idx:04d}] track #{tid:4d} ({cls:12s}): "
                      f"{direction:8s}  conf={conf:.0%}  "
                      f"left={res['probabilities']['left']:.2f} "
                      f"right={res['probabilities']['right']:.2f} "
                      f"straight={res['probabilities']['straight']:.2f}")
            last_pred[tid] = direction

        # Frame number overlay
        cv2.putText(out_frame, f"frame {frame_idx}", (8, H-8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200,200,200), 1)

        writer.write(out_frame)

    cap.release()
    writer.release()

    print("="*60)
    print(f"\nSummary: {len(events)} non-straight confident events")
    direction_counts = defaultdict(int)
    for e in events:
        direction_counts[e['direction']] += 1
    for d, c in direction_counts.items():
        print(f"  {d}: {c}")
    print(f"\nOutput: {out_path}")


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--video',  required=True)
    ap.add_argument('--frames', type=int, default=200)
    ap.add_argument('--out',    default='out_goal.mp4')
    args = ap.parse_args()
    run(args.video, args.frames, args.out)
