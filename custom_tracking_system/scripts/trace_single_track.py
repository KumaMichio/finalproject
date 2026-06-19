"""
trace_single_track.py — theo dõi goal prediction của 1 track qua từng frame.
Giúp xem confidence evolve như thế nào từ khi xe xuất hiện đến khi khuất.
"""
import sys, argparse
from pathlib import Path
from collections import defaultdict
import numpy as np
import cv2

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from modules.detector import ObjectDetector
from modules.tracker import ByteTrackWrapper
from modules.goal_classifier import GoalClassifier


def run(video_path, max_frames, target_tid=None):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 6.0

    detector = ObjectDetector(str(ROOT/'weights'/'yolo11s_vn.pt'), conf_threshold=0.35)
    tracker  = ByteTrackWrapper(frame_rate=int(fps))
    gc       = GoalClassifier(
        str(ROOT/'weights'/'goal_classifier_real.pkl'),
        str(ROOT/'weights'/'goal_scaler_real.pkl'),
        str(ROOT/'weights'/'goal_label_encoder_real.pkl'),
        str(ROOT/'weights'/'goal_feature_names_real.json'),
    )

    # Collect per-track prediction histories
    track_history = defaultdict(list)  # tid -> [(frame, direction, conf, left_p, right_p, straight_p)]
    track_class   = {}

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

        for t in tracks:
            tid = t['track_id']
            cls = t.get('class', 'car')
            track_class[tid] = cls
            res = gc.update_and_predict(tid, t['box'], cls, now_ts)
            if res:
                track_history[tid].append((
                    frame_idx,
                    res['direction'],
                    res['confidence'],
                    res['probabilities'].get('left', 0),
                    res['probabilities'].get('right', 0),
                    res['probabilities'].get('straight', 0),
                    res['probabilities'].get('u_turn', 0),
                ))

    cap.release()

    # Filter tracks with >15 predictions (enough history to be interesting)
    long_tracks = {tid: hist for tid, hist in track_history.items() if len(hist) >= 15}
    print(f"\nTracks với ≥15 predictions: {len(long_tracks)}")

    # Find tracks that ever had a non-straight confident prediction
    interesting = {}
    for tid, hist in long_tracks.items():
        max_nonsrt = max((max(h[3], h[4], h[6]) for h in hist), default=0)
        interesting[tid] = max_nonsrt
    interesting = dict(sorted(interesting.items(), key=lambda x: -x[1]))

    print(f"\nTop 5 tracks theo max P(non-straight):")
    print(f"{'TID':>5}  {'class':12}  {'frames':>6}  {'max_left':>8}  {'max_right':>9}  {'max_uturn':>9}  {'final_dir':>10}")
    print("-"*70)
    for tid in list(interesting)[:5]:
        hist = long_tracks[tid]
        max_l = max(h[3] for h in hist)
        max_r = max(h[4] for h in hist)
        max_u = max(h[6] for h in hist)
        final = hist[-1][1]
        cls   = track_class.get(tid, '?')
        print(f"{tid:>5}  {cls:12}  {len(hist):>6}  {max_l:>8.3f}  {max_r:>9.3f}  {max_u:>9.3f}  {final:>10}")

    # Detailed trace for the most interesting track (or user-specified)
    if target_tid is None and interesting:
        target_tid = list(interesting)[0]

    if target_tid and target_tid in long_tracks:
        hist = long_tracks[target_tid]
        cls  = track_class.get(target_tid, '?')
        print(f"\n── Detailed trace: track #{target_tid} ({cls}) — {len(hist)} frames ──")
        print(f"{'frame':>6}  {'dir':>10}  {'conf':>5}  {'left':>6}  {'right':>6}  {'str':>6}  {'uturn':>5}")
        print("-"*55)
        for h in hist:
            fr, direction, conf, lp, rp, sp, up = h
            marker = " <<<" if direction != 'straight' and conf > 0.45 else ""
            print(f"{fr:>6}  {direction:>10}  {conf:>5.2f}  {lp:>6.3f}  {rp:>6.3f}  {sp:>6.3f}  {up:>5.3f}{marker}")


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--video',  required=True)
    ap.add_argument('--frames', type=int, default=200)
    ap.add_argument('--tid',    type=int, default=None, help='specific track ID to trace')
    args = ap.parse_args()
    run(args.video, args.frames, args.tid)
