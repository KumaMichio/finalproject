"""
autolabel_turning.py — Phase 2: Auto-label turning direction
=============================================================
Gán nhãn hướng quẹo (straight / left / right / u_turn) cho mỗi track
trong trajectories_real_vn/ dựa trên sự thay đổi heading (heading-change method).

Thuật toán:
  1. Với mỗi track, chia thành 2 pha:
       - Entry phase : 25% đầu tiên (min 5 frames)
       - Exit  phase : 25% cuối   (min 5 frames)
  2. Tính entry_heading = circular mean của heading_deg ở pha đầu
     Tính exit_heading  = circular mean của heading_deg ở pha cuối
     heading_change     = circular_diff(exit, entry)   ∈ (-180, 180]
  3. Phân loại:
       |Δh| < 30°          → straight
       |Δh| > 150°         → u_turn
       Δh  > 0 (clockwise) → right  (thuận chiều kim đồng hồ ≡ rẽ phải trong hệ y-down)
       Δh  < 0             → left
  4. Lọc track:
       - n_frames ≥ MIN_TRACK_FRAMES
       - mean_speed_px ≥ MIN_SPEED_PX  (loại xe đứng yên)
       - std_heading_entry, std_heading_exit ≤ MAX_HEADING_STD  (loại track nhiễu)

Note về tọa độ: heading = atan2(vy, vx) với y-down, x-right.
  Δheading dương = xoay chiều kim đồng hồ trong ảnh = rẽ PHẢI (đúng/trái đường tay phải VN).
  Kết quả độc lập với góc camera (chỉ quan tâm tương đối entry→exit).

Output:
  data/turning_labels.csv        — nhãn mỗi track
  data/turning_labels_stats.json — thống kê

Usage:
  python custom_tracking_system/scripts/autolabel_turning.py
  python custom_tracking_system/scripts/autolabel_turning.py --traj-dir custom_tracking_system/data/trajectories_real_vn
  python custom_tracking_system/scripts/autolabel_turning.py --min-frames 20 --min-speed 3 --show-dist
"""

import argparse
import json
import logging
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# ── Paths ────────────────────────────────────────────────────────────────────
ROOT     = Path(__file__).resolve().parents[1]   # custom_tracking_system/
TRAJ_DIR = ROOT / "data" / "trajectories_real_vn"
OUT_CSV  = ROOT / "data" / "turning_labels.csv"
OUT_JSON = ROOT / "data" / "turning_labels_stats.json"

# ── Defaults ─────────────────────────────────────────────────────────────────
MIN_TRACK_FRAMES  = 15     # tracks shorter than this are skipped
MIN_SPEED_PX      = 5.0    # px/s — xe đứng yên / slow crawl bị loại
MAX_HEADING_STD   = 60.0   # deg — loại track có heading nhiễu cao
ENTRY_EXIT_FRAC   = 0.25   # dùng 25% đầu/cuối cho heading estimation
ENTRY_EXIT_MIN    = 5      # ít nhất 5 frames cho mỗi pha

STRAIGHT_THR = 30.0   # |Δh| < 30° = straight
UTURN_THR    = 150.0  # |Δh| > 150° = u_turn

# Fix Windows console encoding
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger(__name__)


# ── Heading utils ─────────────────────────────────────────────────────────────

def circular_mean(angles_deg: np.ndarray) -> float:
    """Circular mean of angles in degrees."""
    rad = np.radians(angles_deg)
    return float(np.degrees(np.arctan2(np.mean(np.sin(rad)),
                                        np.mean(np.cos(rad)))))


def circular_diff(a_deg: float, b_deg: float) -> float:
    """Signed difference a - b, normalized to (-180, 180]."""
    d = (a_deg - b_deg + 180) % 360 - 180
    return float(d)


def heading_to_label(heading_change: float) -> str:
    """
    Chuyển heading_change → label.
    +  = clockwise (image y-down) = right turn (giao thông tay phải VN)
    -  = counterclockwise           = left  turn
    """
    ha = abs(heading_change)
    if ha < STRAIGHT_THR:
        return "straight"
    if ha > UTURN_THR:
        return "u_turn"
    return "right" if heading_change > 0 else "left"


def confidence_level(n_frames: int, mean_speed: float,
                     entry_std: float, exit_std: float) -> str:
    """Phân loại độ tin cậy nhãn: high / medium / low."""
    if (n_frames >= 30 and mean_speed >= 15
            and entry_std < 25 and exit_std < 25):
        return "high"
    if (n_frames >= MIN_TRACK_FRAMES and mean_speed >= MIN_SPEED_PX
            and entry_std < MAX_HEADING_STD and exit_std < MAX_HEADING_STD):
        return "medium"
    return "low"


# ── Per-track labeling ────────────────────────────────────────────────────────

def label_track(grp: pd.DataFrame) -> dict | None:
    """
    Nhãn 1 track. Input: sub-DataFrame đã sort theo frame_id (t).
    Returns dict hoặc None nếu bị lọc.
    """
    grp = grp.sort_values("t").reset_index(drop=True)
    n = len(grp)

    if n < MIN_TRACK_FRAMES:
        return None

    mean_speed = float(grp["speed_px"].mean())
    if mean_speed < MIN_SPEED_PX:
        return None

    # ── Entry / Exit phase ───────────────────────────────────────────────
    entry_n = max(ENTRY_EXIT_MIN, int(n * ENTRY_EXIT_FRAC))
    exit_n  = max(ENTRY_EXIT_MIN, int(n * ENTRY_EXIT_FRAC))

    entry_df = grp.iloc[:entry_n]
    exit_df  = grp.iloc[-exit_n:]

    entry_heading = circular_mean(entry_df["heading_deg"].values)
    exit_heading  = circular_mean(exit_df["heading_deg"].values)

    # Std deviation as confidence proxy
    entry_rad = np.radians(entry_df["heading_deg"].values)
    exit_rad  = np.radians(exit_df["heading_deg"].values)
    entry_std = float(np.degrees(np.sqrt(-2 * np.log(
        np.clip(np.abs(np.mean(np.exp(1j * entry_rad))), 1e-9, 1)))))
    exit_std  = float(np.degrees(np.sqrt(-2 * np.log(
        np.clip(np.abs(np.mean(np.exp(1j * exit_rad))), 1e-9, 1)))))

    if entry_std > MAX_HEADING_STD or exit_std > MAX_HEADING_STD:
        return None

    heading_change = circular_diff(exit_heading, entry_heading)
    goal_label = heading_to_label(heading_change)
    conf = confidence_level(n, mean_speed, entry_std, exit_std)

    # ── Entry / Exit centroids ───────────────────────────────────────────
    cx_entry = float(entry_df["cx_px"].mean())
    cy_entry = float(entry_df["cy_px"].mean())
    cx_exit  = float(exit_df["cx_px"].mean())
    cy_exit  = float(exit_df["cy_px"].mean())

    # Displacement magnitude
    disp = math.hypot(cx_exit - cx_entry, cy_exit - cy_entry)

    return {
        "video_id":          grp["video_id"].iloc[0],
        "track_id":          int(grp["track_id"].iloc[0]),
        "class":             grp["class"].iloc[0],
        "goal_label":        goal_label,
        "heading_change_deg": round(heading_change, 2),
        "entry_heading":     round(entry_heading, 2),
        "exit_heading":      round(exit_heading, 2),
        "entry_std_deg":     round(entry_std, 2),
        "exit_std_deg":      round(exit_std, 2),
        "t_enter":           round(float(grp["t"].iloc[0]), 4),
        "t_exit":            round(float(grp["t"].iloc[-1]), 4),
        "n_frames":          n,
        "mean_speed_px":     round(mean_speed, 2),
        "disp_px":           round(disp, 2),
        "cx_entry":          round(cx_entry, 2),
        "cy_entry":          round(cy_entry, 2),
        "cx_exit":           round(cx_exit, 2),
        "cy_exit":           round(cy_exit, 2),
        "confidence":        conf,
    }


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    global MIN_TRACK_FRAMES, MIN_SPEED_PX, MAX_HEADING_STD, STRAIGHT_THR, UTURN_THR
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--traj-dir",   default=str(TRAJ_DIR),
                    help="Folder chứa trajectory CSVs từ Phase 1")
    ap.add_argument("--out",        default=str(OUT_CSV),
                    help="Output CSV path")
    ap.add_argument("--min-frames", type=int,   default=MIN_TRACK_FRAMES,
                    help=f"Min frames per track (default {MIN_TRACK_FRAMES})")
    ap.add_argument("--min-speed",  type=float, default=MIN_SPEED_PX,
                    help=f"Min mean speed in px/s (default {MIN_SPEED_PX})")
    ap.add_argument("--max-std",    type=float, default=MAX_HEADING_STD,
                    help=f"Max heading std in deg (default {MAX_HEADING_STD})")
    ap.add_argument("--straight-thr", type=float, default=STRAIGHT_THR,
                    help=f"Threshold for straight label in deg (default {STRAIGHT_THR})")
    ap.add_argument("--uturn-thr",    type=float, default=UTURN_THR,
                    help=f"Threshold for u_turn label in deg (default {UTURN_THR})")
    ap.add_argument("--show-dist",  action="store_true",
                    help="In phân phối nhãn cuối cùng")
    ap.add_argument("--video",      default="*",
                    help="Glob pattern để chọn video (ví dụ 'Cửa_Nam*')")
    args = ap.parse_args()

    # Override thresholds from args
    MIN_TRACK_FRAMES = args.min_frames
    MIN_SPEED_PX     = args.min_speed
    MAX_HEADING_STD  = args.max_std
    STRAIGHT_THR     = args.straight_thr
    UTURN_THR        = args.uturn_thr

    traj_dir = Path(args.traj_dir)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    from fnmatch import fnmatch
    csv_files = [f for f in sorted(traj_dir.glob("*.csv"))
                 if f.stem != "summary" and fnmatch(f.stem, args.video)]
    log.info("Found %d trajectory CSV files", len(csv_files))

    # ── Process all CSVs ──────────────────────────────────────────────────
    all_rows = []
    n_skipped_speed = 0
    n_skipped_frames = 0
    n_skipped_std = 0

    for i, csv_path in enumerate(csv_files, 1):
        try:
            df = pd.read_csv(csv_path)
        except Exception as e:
            log.warning("Cannot read %s: %s", csv_path.name, e)
            continue

        needed = {"t", "video_id", "track_id", "class",
                  "cx_px", "cy_px", "speed_px", "heading_deg"}
        if not needed.issubset(df.columns):
            missing = needed - set(df.columns)
            log.warning("  Skip %s — missing columns: %s", csv_path.name, missing)
            continue

        n_tracks = df["track_id"].nunique()
        if i % 20 == 0 or i == 1:
            log.info("[%d/%d] %s  (%d tracks)", i, len(csv_files),
                     csv_path.stem[:55], n_tracks)

        for _tid, grp in df.groupby("track_id"):
            n = len(grp)
            if n < MIN_TRACK_FRAMES:
                n_skipped_frames += 1
                continue
            row = label_track(grp)
            if row is None:
                # Determine why it was skipped for stats
                if grp["speed_px"].mean() < MIN_SPEED_PX:
                    n_skipped_speed += 1
                else:
                    n_skipped_std += 1
                continue
            all_rows.append(row)

    if not all_rows:
        log.error("No tracks were labeled! Check --traj-dir and filters.")
        sys.exit(1)

    out_df = pd.DataFrame(all_rows)
    out_df.to_csv(out_path, index=False)
    log.info("\n=== OUTPUT ===")
    log.info("Saved %d labeled tracks to %s", len(out_df), out_path)

    # ── Stats ─────────────────────────────────────────────────────────────
    label_counts = out_df["goal_label"].value_counts().to_dict()
    conf_counts  = out_df["confidence"].value_counts().to_dict()
    class_counts = out_df["class"].value_counts().to_dict()
    total = len(out_df)

    log.info("\n--- Label Distribution ---")
    for lbl in ["straight", "left", "right", "u_turn"]:
        cnt = label_counts.get(lbl, 0)
        log.info("  %-10s %5d  (%5.1f%%)", lbl, cnt, 100*cnt/total)

    log.info("\n--- Confidence ---")
    for lvl in ["high", "medium", "low"]:
        cnt = conf_counts.get(lvl, 0)
        log.info("  %-8s %5d  (%5.1f%%)", lvl, cnt, 100*cnt/total)

    log.info("\n--- Class ---")
    for cls, cnt in sorted(class_counts.items(), key=lambda x: -x[1]):
        log.info("  %-12s %5d  (%5.1f%%)", cls, cnt, 100*cnt/total)

    log.info("\n--- Skipped tracks ---")
    log.info("  Too few frames : %d", n_skipped_frames)
    log.info("  Too slow       : %d", n_skipped_speed)
    log.info("  Heading noisy  : %d", n_skipped_std)

    stats = {
        "total_labeled":     total,
        "n_videos":          out_df["video_id"].nunique(),
        "label_counts":      label_counts,
        "confidence_counts": conf_counts,
        "class_counts":      class_counts,
        "skipped": {
            "too_few_frames": n_skipped_frames,
            "too_slow":       n_skipped_speed,
            "heading_noisy":  n_skipped_std,
        },
        "params": {
            "min_track_frames": MIN_TRACK_FRAMES,
            "min_speed_px":     MIN_SPEED_PX,
            "max_heading_std":  MAX_HEADING_STD,
            "straight_thr":     STRAIGHT_THR,
            "uturn_thr":        UTURN_THR,
        },
    }
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)
    log.info("Stats saved to %s", OUT_JSON)

    if args.show_dist:
        print("\nLabel distribution:")
        for lbl in ["straight", "left", "right", "u_turn"]:
            cnt = label_counts.get(lbl, 0)
            bar = "#" * int(40 * cnt / max(label_counts.values()))
            print(f"  {lbl:<10} {cnt:>5} {bar}")


if __name__ == "__main__":
    main()
