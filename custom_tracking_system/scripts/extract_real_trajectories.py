"""
extract_real_trajectories.py
============================
Chạy YOLO11s + ByteTrack trên các frame sequence từ data/auto_label/images/,
xuất trajectory CSV theo từng video (giống format trajectories_hanoi_v2).

Output per video:
    data/trajectories_real_vn/<video_id>.csv
    Columns: t, video_id, track_id, class, cx_px, cy_px, vx_px, vy_px,
             speed_px, heading_deg, frame_id, conf, w_px, h_px

Summary:
    data/trajectories_real_vn/summary.json

Usage:
    python custom_tracking_system/scripts/extract_real_trajectories.py
    python custom_tracking_system/scripts/extract_real_trajectories.py --fps 10 --conf 0.35
    python custom_tracking_system/scripts/extract_real_trajectories.py --video "Cửa_Nam*" --preview
"""

import argparse
import json
import logging
import math
import sys
from collections import defaultdict
from fnmatch import fnmatch
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

# ── Paths ────────────────────────────────────────────────────────────────────
ROOT       = Path(__file__).resolve().parents[1]   # custom_tracking_system/
IMG_DIR    = ROOT / "data" / "auto_label" / "images"
OUT_DIR    = ROOT / "data" / "trajectories_real_vn"
WEIGHTS    = ROOT / "weights" / "yolo11s_vn.pt"

# ── Defaults ─────────────────────────────────────────────────────────────────
DEFAULT_FPS        = 10      # assumed frame rate for auto_label clips
MIN_TRACK_FRAMES   = 8       # discard tracks shorter than this
SMOOTH_WINDOW      = 3       # frames for velocity smoothing (median)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger(__name__)


# ── Helpers ───────────────────────────────────────────────────────────────────

def group_frames_by_video(img_dir: Path, pattern: str = "*") -> dict[str, list[Path]]:
    """
    Group JPG files by video prefix (everything before the last _NNNNNNN.jpg).
    Returns {video_id: [sorted list of frame paths]}.
    """
    groups: dict[str, list[Path]] = defaultdict(list)
    for p in sorted(img_dir.glob("*.jpg")):
        # Filename: <prefix>_<7digits>.jpg
        parts = p.stem.rsplit("_", 1)
        if len(parts) == 2 and parts[1].isdigit():
            prefix = parts[0]
        else:
            prefix = p.stem
        if fnmatch(prefix, pattern):
            groups[prefix].append(p)
    # Sort frames within each group
    for k in groups:
        groups[k].sort()
    return dict(groups)


def smooth_velocity(arr: np.ndarray, window: int = SMOOTH_WINDOW) -> np.ndarray:
    """Apply median filter along each column of a (N, 2) velocity array."""
    if len(arr) < window:
        return arr
    out = arr.copy().astype(float)
    half = window // 2
    for i in range(len(arr)):
        lo = max(0, i - half)
        hi = min(len(arr), i + half + 1)
        out[i] = np.median(arr[lo:hi], axis=0)
    return out


def compute_velocity_heading(cx: np.ndarray, cy: np.ndarray,
                              t: np.ndarray) -> tuple[np.ndarray, np.ndarray,
                                                       np.ndarray, np.ndarray]:
    """
    Compute per-frame vx, vy (px/s), speed (px/s), heading (deg) from pixel centroids.
    Uses central differences; edge frames use forward/backward difference.
    """
    n = len(cx)
    vx = np.zeros(n)
    vy = np.zeros(n)

    if n >= 2:
        dt = np.diff(t)
        dt = np.where(dt < 1e-6, 1e-6, dt)
        dx = np.diff(cx.astype(float))
        dy = np.diff(cy.astype(float))
        vx_diff = dx / dt
        vy_diff = dy / dt

        # Central differences for interior points
        vx[0]    = vx_diff[0]
        vx[-1]   = vx_diff[-1]
        vy[0]    = vy_diff[0]
        vy[-1]   = vy_diff[-1]
        if n > 2:
            vx[1:-1] = 0.5 * (vx_diff[:-1] + vx_diff[1:])
            vy[1:-1] = 0.5 * (vy_diff[:-1] + vy_diff[1:])

        # Smooth
        vel = smooth_velocity(np.stack([vx, vy], axis=1))
        vx, vy = vel[:, 0], vel[:, 1]

    speed   = np.sqrt(vx**2 + vy**2)
    heading = np.degrees(np.arctan2(vy, vx))   # [-180, 180]
    return vx, vy, speed, heading


# ── Core processing ───────────────────────────────────────────────────────────

def process_video(video_id: str, frames: list[Path], detector,
                  fps: float, out_dir: Path, preview: bool = False) -> dict:
    """
    Run detection + tracking on one video sequence (boxmot 19 API).
    Returns stats dict.
    TrackResults columns: [x1, y1, x2, y2, track_id, conf, cls_id, ...]
    """
    from boxmot.trackers import ByteTrack

    tracker = ByteTrack(
        track_thresh=0.25,
        match_thresh=0.8,
        track_buffer=30,
        frame_rate=int(fps),
    )

    cls_map = detector.CLASSES   # {cls_id: class_name}

    # {track_id: {"frames":[], "cx":[], "cy":[], "w":[], "h":[], "conf":[], "class": str}}
    raw_tracks: dict[int, dict] = defaultdict(lambda: {
        "frames": [], "cx": [], "cy": [], "w": [], "h": [], "conf": [], "class": "unknown"
    })

    preview_frames = []

    for frame_idx, img_path in enumerate(frames):
        img = cv2.imread(str(img_path))
        if img is None:
            log.warning("Cannot read %s", img_path)
            continue

        dets_list = detector.detect(img)

        # boxmot 19 expects ndarray [[x1,y1,x2,y2,conf,cls_id], ...]
        if dets_list:
            dets_arr = np.array(
                [[*d["box"], d["confidence"], d.get("class_id", 0)]
                 for d in dets_list],
                dtype=np.float32,
            )
        else:
            dets_arr = np.empty((0, 6), dtype=np.float32)

        result = tracker.update(dets_arr, img)   # TrackResults → ndarray

        if result is not None and len(result) > 0:
            for row in np.asarray(result):
                # cols: x1, y1, x2, y2, track_id, conf, cls_id, ...
                x1, y1, x2, y2 = row[0], row[1], row[2], row[3]
                tid    = int(row[4])
                conf   = float(row[5])
                cls_id = int(row[6])
                cls_name = cls_map.get(cls_id, "unknown")

                cx = (x1 + x2) / 2.0
                cy = (y1 + y2) / 2.0
                w  = x2 - x1
                h  = y2 - y1

                raw_tracks[tid]["frames"].append(frame_idx)
                raw_tracks[tid]["cx"].append(cx)
                raw_tracks[tid]["cy"].append(cy)
                raw_tracks[tid]["w"].append(w)
                raw_tracks[tid]["h"].append(h)
                raw_tracks[tid]["conf"].append(conf)
                raw_tracks[tid]["class"] = cls_name

            if preview and frame_idx < 5:
                vis = img.copy()
                for row in np.asarray(result):
                    x1, y1, x2, y2 = int(row[0]), int(row[1]), int(row[2]), int(row[3])
                    tid     = int(row[4])
                    cls_id  = int(row[6])
                    cls_name = cls_map.get(cls_id, "?")
                    cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(vis, f"{tid}:{cls_name}",
                                (x1, max(0, y1 - 5)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                preview_frames.append(vis)

    if preview and preview_frames:
        preview_path = out_dir / f"{video_id[:40]}_preview.jpg"
        grid = np.hstack(preview_frames[:min(3, len(preview_frames))])
        cv2.imwrite(str(preview_path), grid)
        log.info("Preview saved: %s", preview_path)

    # ── Build output rows ────────────────────────────────────────────────────
    rows = []
    n_short = 0

    for tid, data in raw_tracks.items():
        n_frames = len(data["frames"])
        if n_frames < MIN_TRACK_FRAMES:
            n_short += 1
            continue

        frame_ids = np.array(data["frames"])
        cx_arr    = np.array(data["cx"])
        cy_arr    = np.array(data["cy"])
        w_arr     = np.array(data["w"])
        h_arr     = np.array(data["h"])
        conf_arr  = np.array(data["conf"])
        t_arr     = frame_ids / fps        # seconds from video start

        vx, vy, speed, heading = compute_velocity_heading(cx_arr, cy_arr, t_arr)

        for i in range(n_frames):
            rows.append({
                "t":           round(float(t_arr[i]), 4),
                "video_id":    video_id,
                "track_id":    int(tid),
                "class":       data["class"],
                "cx_px":       round(float(cx_arr[i]), 2),
                "cy_px":       round(float(cy_arr[i]), 2),
                "vx_px":       round(float(vx[i]), 4),
                "vy_px":       round(float(vy[i]), 4),
                "speed_px":    round(float(speed[i]), 4),
                "heading_deg": round(float(heading[i]), 2),
                "frame_id":    int(frame_ids[i]),
                "conf":        round(float(conf_arr[i]), 3),
                "w_px":        round(float(w_arr[i]), 2),
                "h_px":        round(float(h_arr[i]), 2),
            })

    n_kept = len(raw_tracks) - n_short
    stats = {
        "video_id":    video_id,
        "n_frames":    len(frames),
        "n_tracks_raw": len(raw_tracks),
        "n_tracks_kept": n_kept,
        "n_tracks_short": n_short,
        "n_rows":      len(rows),
    }

    if rows:
        df = pd.DataFrame(rows)
        out_path = out_dir / f"{video_id[:80]}.csv"
        df.to_csv(out_path, index=False)
        log.info("  → %d tracks kept (%d short), %d rows → %s",
                 n_kept, n_short, len(rows), out_path.name)
    else:
        log.warning("  → No valid tracks for %s", video_id)

    return stats


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--fps",     type=float, default=DEFAULT_FPS,
                    help="Assumed frame rate of clips (default 10)")
    ap.add_argument("--conf",    type=float, default=0.35,
                    help="YOLO confidence threshold")
    ap.add_argument("--weights", default=str(WEIGHTS),
                    help="Path to YOLO weights")
    ap.add_argument("--img-dir", default=str(IMG_DIR),
                    help="Folder containing auto_label images")
    ap.add_argument("--out",     default=str(OUT_DIR),
                    help="Output folder for trajectory CSVs")
    ap.add_argument("--video",   default="*",
                    help="Glob pattern to select specific videos (e.g. 'Cửa_Nam*')")
    ap.add_argument("--preview", action="store_true",
                    help="Save a preview JPEG for each video (first 3 tracked frames)")
    ap.add_argument("--limit",   type=int, default=0,
                    help="Process only first N videos (0 = all, useful for testing)")
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Load detector ────────────────────────────────────────────────────────
    sys.path.insert(0, str(ROOT))   # custom_tracking_system/ → finds modules/
    from modules.detector import ObjectDetector

    log.info("Loading detector from %s", args.weights)
    detector = ObjectDetector(
        model_type=args.weights,
        conf_threshold=args.conf,
        device="cpu",
    )
    # suppress boxmot INFO noise
    logging.getLogger("boxmot").setLevel(logging.WARNING)

    # ── Group frames ─────────────────────────────────────────────────────────
    img_dir = Path(args.img_dir)
    groups  = group_frames_by_video(img_dir, pattern=args.video)
    video_ids = sorted(groups.keys())

    if args.limit > 0:
        video_ids = video_ids[:args.limit]

    log.info("Found %d video groups matching '%s'", len(video_ids), args.video)

    # ── Process ──────────────────────────────────────────────────────────────
    all_stats = []
    for i, vid in enumerate(video_ids, 1):
        frames = groups[vid]
        log.info("[%d/%d] %s (%d frames)", i, len(video_ids), vid[:60], len(frames))
        try:
            stats = process_video(
                video_id=vid,
                frames=frames,
                detector=detector,
                fps=args.fps,
                out_dir=out_dir,
                preview=args.preview,
            )
            all_stats.append(stats)
        except Exception as exc:
            log.error("  FAILED: %s", exc, exc_info=True)

    # ── Summary ──────────────────────────────────────────────────────────────
    total_tracks = sum(s["n_tracks_kept"] for s in all_stats)
    total_rows   = sum(s["n_rows"] for s in all_stats)
    summary = {
        "n_videos":      len(all_stats),
        "total_tracks":  total_tracks,
        "total_rows":    total_rows,
        "fps_assumed":   args.fps,
        "min_frames":    MIN_TRACK_FRAMES,
        "videos":        all_stats,
    }
    summary_path = out_dir / "summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    log.info("\n=== DONE ===")
    log.info("Videos processed : %d", len(all_stats))
    log.info("Total tracks kept: %d", total_tracks)
    log.info("Total rows       : %d", total_rows)
    log.info("Output           : %s", out_dir)


if __name__ == "__main__":
    main()
