"""
benchmark_pipeline.py — đo hiệu năng end-to-end pipeline trên video CCTV thực.

Metrics thu thập:
  - FPS trung bình (inference + tracking + goal classify)
  - Số xe detect được / frame
  - Số track duy nhất
  - Phân phối goal prediction (left/right/straight/u_turn)
  - Confidence trung bình của goal prediction
  - ID switch estimate (track born/died count)

Usage:
  python scripts/benchmark_pipeline.py --video <path.mp4> --frames 300
"""

import sys, time, argparse, json
from pathlib import Path
from collections import defaultdict

import numpy as np
import cv2

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from modules.detector import ObjectDetector
from modules.tracker import ByteTrackWrapper
from modules.goal_classifier import GoalClassifier


def run_benchmark(video_path: str, max_frames: int = 300, show: bool = False):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"ERROR: cannot open {video_path}")
        return

    video_fps = cap.get(cv2.CAP_PROP_FPS) or 10.0
    total_video_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"\nVideo: {Path(video_path).name}")
    print(f"  Resolution : {int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))}x{int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))}")
    print(f"  FPS (video): {video_fps:.1f}")
    print(f"  Frames total: {total_video_frames}  |  benchmark: {max_frames}\n")

    print("[INIT] Loading YOLO11s_vn...")
    detector = ObjectDetector(
        model_type=str(ROOT / 'weights' / 'yolo11s_vn.pt'),
        conf_threshold=0.35, half=False,
    )
    print("[INIT] Loading ByteTrack...")
    tracker = ByteTrackWrapper(
        track_activation_threshold=0.25,
        lost_track_buffer=30,
        minimum_matching_threshold=0.8,
        frame_rate=int(video_fps),
    )
    print("[INIT] Loading GoalClassifier...")
    gc = GoalClassifier(
        str(ROOT / 'weights' / 'goal_classifier_real.pkl'),
        str(ROOT / 'weights' / 'goal_scaler_real.pkl'),
        str(ROOT / 'weights' / 'goal_label_encoder_real.pkl'),
        str(ROOT / 'weights' / 'goal_feature_names_real.json'),
    )
    print("[INIT] All modules ready.\n")

    # ── Metrics accumulators ──────────────────────────────────────────────
    frame_times       = []
    det_counts        = []
    track_counts      = []
    all_track_ids     = set()
    goal_direction_counts = defaultdict(int)
    goal_confidences  = []
    frames_with_goals = 0

    frame_idx = 0
    t_start   = time.perf_counter()

    while frame_idx < max_frames:
        ret, frame_bgr = cap.read()
        if not ret:
            break

        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        t0 = time.perf_counter()

        # Detection
        detections = detector.detect(frame_rgb)

        # Tracking
        tracks = tracker.update(detections, frame_rgb)

        # Goal classification
        now_ts = time.time()
        frame_has_goal = False
        for t in tracks:
            all_track_ids.add(t['track_id'])
            res = gc.update_and_predict(
                t['track_id'], t['box'], t.get('class', 'car'), now_ts
            )
            if res:
                goal_direction_counts[res['direction']] += 1
                goal_confidences.append(res['confidence'])
                frame_has_goal = True

        t1 = time.perf_counter()

        frame_times.append(t1 - t0)
        det_counts.append(len(detections))
        track_counts.append(len(tracks))
        if frame_has_goal:
            frames_with_goals += 1

        if show:
            for t in tracks:
                x1,y1,x2,y2 = t['box']
                cv2.rectangle(frame_bgr,(x1,y1),(x2,y2),(0,255,0),2)
            cv2.imshow('benchmark', frame_bgr)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        frame_idx += 1
        if frame_idx % 50 == 0:
            elapsed = time.perf_counter() - t_start
            fps_so_far = frame_idx / elapsed
            print(f"  frame {frame_idx}/{max_frames}  |  pipeline FPS: {fps_so_far:.1f}")

    cap.release()
    if show:
        cv2.destroyAllWindows()

    # ── Report ────────────────────────────────────────────────────────────
    total_elapsed = time.perf_counter() - t_start
    n = len(frame_times)
    if n == 0:
        print("No frames processed.")
        return

    avg_fps      = n / total_elapsed
    avg_ms       = np.mean(frame_times) * 1000
    p50_ms       = np.percentile(frame_times, 50) * 1000
    p95_ms       = np.percentile(frame_times, 95) * 1000
    avg_dets     = np.mean(det_counts)
    avg_tracks   = np.mean(track_counts)
    total_tracks = len(all_track_ids)
    goal_total   = sum(goal_direction_counts.values())
    avg_conf     = np.mean(goal_confidences) if goal_confidences else 0.0

    print("\n" + "="*55)
    print("BENCHMARK RESULTS")
    print("="*55)
    print(f"Frames processed   : {n}")
    print(f"Total time         : {total_elapsed:.1f} s")
    print()
    print("── Latency ──────────────────────────────────")
    print(f"  Pipeline FPS (avg): {avg_fps:.1f}")
    print(f"  Per-frame latency : {avg_ms:.1f} ms  (p50={p50_ms:.0f}ms, p95={p95_ms:.0f}ms)")
    print()
    print("── Detection ────────────────────────────────")
    print(f"  Avg detections/frame: {avg_dets:.1f}")
    print(f"  Max detections/frame: {max(det_counts)}")
    print()
    print("── Tracking ─────────────────────────────────")
    print(f"  Avg active tracks/frame : {avg_tracks:.1f}")
    print(f"  Total unique track IDs  : {total_tracks}")
    print(f"  Frames with >=1 track   : {sum(1 for c in track_counts if c > 0)}/{n}")
    print()
    print("── Goal Classifier ──────────────────────────")
    print(f"  Frames with prediction  : {frames_with_goals}/{n} ({frames_with_goals/n*100:.0f}%)")
    print(f"  Total predictions       : {goal_total}")
    print(f"  Avg confidence          : {avg_conf:.1%}")
    if goal_total > 0:
        print(f"  Direction distribution:")
        for d in ['straight', 'left', 'right', 'u_turn']:
            cnt = goal_direction_counts[d]
            print(f"    {d:10s}: {cnt:5d}  ({cnt/goal_total*100:.1f}%)")
    print("="*55)

    result = {
        "n_frames": n, "total_time_s": round(total_elapsed, 2),
        "fps_avg": round(avg_fps, 1),
        "latency_avg_ms": round(avg_ms, 1),
        "latency_p50_ms": round(p50_ms, 1),
        "latency_p95_ms": round(p95_ms, 1),
        "avg_detections_per_frame": round(avg_dets, 1),
        "avg_tracks_per_frame": round(avg_tracks, 1),
        "total_unique_tracks": total_tracks,
        "goal_frames_pct": round(frames_with_goals / n * 100, 1),
        "goal_avg_confidence": round(avg_conf, 3),
        "goal_direction_counts": dict(goal_direction_counts),
    }
    out_path = ROOT.parent / "docs" / "assets" / "benchmark_pipeline.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, 'w') as f:
        json.dump(result, f, indent=2)
    print(f"\nSaved to: {out_path}")
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--video', required=True)
    parser.add_argument('--frames', type=int, default=300)
    parser.add_argument('--show', action='store_true')
    args = parser.parse_args()
    run_benchmark(args.video, args.frames, args.show)
