"""
train_goal_classifier_real.py — Phase 3: Goal Classifier trên dữ liệu CCTV thực
=================================================================================
Train GradientBoosting classifier dự đoán hướng đi (straight/left/right/u_turn)
từ trajectory pixel-space của CCTV Hà Nội thực tế.

Luồng:
  1. Đọc turning_labels.csv  (Phase 2 output) → goal events
  2. Đọc trajectories_real_vn/*.csv           → trajectory index
  3. Với mỗi track, đặt t_junction = midpoint(t_enter, t_exit)
  4. Với mỗi horizon h ∈ [0.5, 1.0, 1.5, 2.0]:
       Trích feature từ trajectory window [t_junction - h, t_junction]
  5. Train GradientBoosting (primary_horizon=1.0s)
  6. Eval tất cả horizons + per-class accuracy
  7. Lưu toàn bộ số liệu + charts

So sánh với CARLA baseline:
  - world-space speed (m/s)    → pixel-space speed (px/s)
  - CLASS_SPEED_LIMIT m/s      → CLASS_SPEED_LIMIT_PX px/s (tính từ data)
  - epoch_id/actor_id          → video_id/track_id
  - junction entry chính xác  → midpoint estimation

Usage:
  python custom_tracking_system/scripts/train/train_goal_classifier_real.py
  python custom_tracking_system/scripts/train/train_goal_classifier_real.py --horizon 1.0 --model gb
  python custom_tracking_system/scripts/train/train_goal_classifier_real.py --high-conf-only
"""

import argparse
import json
import math
import pickle
import sys
import time
from collections import defaultdict
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import (accuracy_score, classification_report,
                              confusion_matrix, f1_score)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.utils.class_weight import compute_sample_weight

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAS_MPL = True
except ImportError:
    HAS_MPL = False

# ── Paths ────────────────────────────────────────────────────────────────────
ROOT         = Path(__file__).resolve().parents[2]   # custom_tracking_system/
TRAJ_DIR     = ROOT / "data" / "trajectories_real_vn"
LABELS_CSV   = ROOT / "data" / "turning_labels.csv"
WEIGHTS_DIR  = ROOT / "weights"
ASSET_DIR    = Path(__file__).resolve().parents[3] / "docs" / "assets" / "goal_classifier_real"

# ── Constants ────────────────────────────────────────────────────────────────
# px/s speed limits inferred from p95 of trajectory data (camera-dependent approx)
CLASS_SPEED_LIMIT_PX = {
    "car": 350.0, "motorcycle": 300.0, "bus": 400.0,
    "truck": 300.0, "person": 100.0,
}
LABEL_NAMES   = ["straight", "left", "right", "u_turn"]
LABEL_COLORS  = {"straight": "#4fc3f7", "left": "#ff8a65",
                  "right": "#a5d6a7", "u_turn": "#ce93d8"}
EVAL_HORIZONS = [0.5, 1.0, 1.5, 2.0]
TRAJ_WINDOW   = 4.0    # max lookback window (s)

# Custom class weights — moderate boost cho minority classes (không dùng "balanced" vì quá extreme)
# "balanced" cho u_turn weight ~25x; custom này chỉ 6x
CUSTOM_CLASS_WEIGHTS = {"straight": 1.0, "left": 2.5, "right": 2.5, "u_turn": 6.0}

# Approximate real-world footprint area (m²) for bbox normalization
# px_per_m ≈ sqrt(w_px * h_px) / sqrt(real_area)  → camera-agnostic speed
CLASS_REAL_AREA_M2 = {
    "motorcycle": 0.7 * 1.8,   # 1.26 m²
    "car":        1.85 * 4.5,  # 8.33 m²
    "bus":        2.5 * 10.0,  # 25.0 m²
    "truck":      2.5 * 7.0,   # 17.5 m²
    "person":     0.5 * 1.7,   # 0.85 m²
}
CLASS_REAL_SIZE_SQRT = {k: math.sqrt(v) for k, v in CLASS_REAL_AREA_M2.items()}
TRAJ_MIN_FRAMES = 5    # min frames in window

# Fraction of track duration to use as junction estimate
# 0.5 = midpoint. 0.65 = 65% into track (includes more turning motion)
JUNCTION_FRAC = 0.65


# ── Feature extraction (pixel-space) ─────────────────────────────────────────

def extract_features_px(traj: pd.DataFrame, t_cutoff: float,
                         cls: str, window: float = TRAJ_WINDOW) -> dict | None:
    """
    Trích feature pixel-space từ trajectory TRƯỚC t_cutoff.
    Cột cần: t, cx_px, cy_px, vx_px, vy_px, speed_px, heading_deg, w_px, h_px
    """
    sub = traj[(traj["t"] <= t_cutoff) &
               (traj["t"] >= t_cutoff - window)].copy()
    if len(sub) < TRAJ_MIN_FRAMES:
        return None

    sub = sub.sort_values("t")
    speed  = sub["speed_px"].values
    vx     = sub["vx_px"].values
    vy     = sub["vy_px"].values
    head   = sub["heading_deg"].values
    t_vals = sub["t"].values

    # ── Bbox-based scale estimation ────────────────────────────────────────
    has_bbox = "w_px" in sub.columns and "h_px" in sub.columns
    if has_bbox:
        w_arr = sub["w_px"].values
        h_arr = sub["h_px"].values
        bbox_size = np.sqrt(np.maximum(w_arr * h_arr, 1.0))   # geometric mean (px)
        real_size = CLASS_REAL_SIZE_SQRT.get(cls, 1.12)        # sqrt(real area in m²)
        px_per_m  = bbox_size / real_size                      # px/m estimate per frame
        # Camera-agnostic speed: speed_px / px_per_m → approximate m/s
        speed_ms  = speed / np.maximum(px_per_m, 0.1)
        # Purely scale-normalized speed (removes camera zoom effect)
        speed_norm_bbox = speed / np.maximum(bbox_size, 1.0)
        bbox_size_mean  = float(np.mean(bbox_size))
        px_per_m_mean   = float(np.mean(px_per_m))
    else:
        speed_ms        = speed / CLASS_SPEED_LIMIT_PX.get(cls, 300.0)
        speed_norm_bbox = speed_ms
        bbox_size_mean  = 0.0
        px_per_m_mean   = 0.0
    dt     = t_vals[-1] - t_vals[0]
    if dt < 0.1:
        return None

    speed_limit = CLASS_SPEED_LIMIT_PX.get(cls, 300.0)

    # ── Heading deltas ───────────────────────────────────────────────────
    head_diff = np.diff(head)
    head_diff = ((head_diff + 180) % 360) - 180    # wrap to [-180, 180]
    dt_arr    = np.diff(t_vals)
    dt_arr    = np.where(dt_arr < 1e-6, 1e-6, dt_arr)
    head_rate = head_diff / dt_arr                  # deg/s

    # Cumulative heading features
    total_abs_heading  = float(np.sum(np.abs(head_diff)))
    total_sign_heading = float(np.sum(head_diff))
    heading_range      = float(np.max(head) - np.min(head))

    # Heading monotonicity: 1.0 = always same direction
    if len(head_diff) > 1:
        signs = np.sign(head_diff)
        signs = signs[signs != 0]
        if len(signs) > 1:
            same = float(np.sum(signs[1:] == signs[:-1])) / (len(signs) - 1)
        else:
            same = 1.0
    else:
        same = 1.0
    head_monotonic = same

    # ── Speed features ───────────────────────────────────────────────────
    speed_min       = float(np.min(speed))
    speed_max       = float(np.max(speed))
    speed_drop      = speed_max - speed_min
    speed_min_ratio = speed_min / max(speed_max, 0.1)

    n_third = max(1, len(speed) // 3)
    speed_final_third = float(np.mean(speed[-n_third:]))

    # Normalized speed
    speed_norm = speed / speed_limit

    # Recent 0.5s slice
    recent    = sub[sub["t"] >= t_cutoff - 0.5]
    sp_recent = recent["speed_px"].values if len(recent) > 0 else speed[-3:]

    # ── Lateral acceleration proxy ────────────────────────────────────────
    if len(vx) >= 3:
        ax     = np.gradient(vx, t_vals)
        ay     = np.gradient(vy, t_vals)
        cross  = np.abs(vx * ay - vy * ax)
        v_mag  = np.sqrt(vx**2 + vy**2)
        safe_mag = np.where(v_mag > 0.5, v_mag, 1.0)
        lat_acc  = np.where(v_mag > 0.5, cross / safe_mag, 0.0)
        lat_acc_mean = float(np.mean(lat_acc))
        lat_acc_max  = float(np.max(lat_acc))
    else:
        lat_acc_mean = lat_acc_max = 0.0

    f = {
        # ── Speed statistics ─────────────────────────────────────────────
        "speed_mean":         float(np.mean(speed)),
        "speed_std":          float(np.std(speed)),
        "speed_last":         float(speed[-1]),
        "speed_recent":       float(np.mean(sp_recent)),
        "speed_norm_mean":    float(np.mean(speed_norm)),
        "speed_change":       float(speed[-1] - speed[0]),
        "speed_change_rate":  float((speed[-1] - speed[0]) / dt),

        # ── Heading rate statistics ──────────────────────────────────────
        "head_rate_mean":     float(np.mean(head_rate)),
        "head_rate_std":      float(np.std(head_rate)),
        "head_rate_max":      float(np.max(np.abs(head_rate))),

        # Final heading as sin/cos (camera-independent relative features)
        "head_sin":           float(math.sin(math.radians(head[-1]))),
        "head_cos":           float(math.cos(math.radians(head[-1]))),

        # ── Cumulative heading (key for u_turn) ──────────────────────────
        "total_abs_heading":  total_abs_heading,
        "total_sign_heading": total_sign_heading,
        "heading_range":      heading_range,
        "head_monotonic":     head_monotonic,

        # ── Braking depth (u_turn brakes deeper) ────────────────────────
        "speed_min":          speed_min,
        "speed_drop":         speed_drop,
        "speed_min_ratio":    speed_min_ratio,
        "speed_final_third":  speed_final_third,

        # ── Lateral acceleration proxy ───────────────────────────────────
        "lat_acc_mean":       lat_acc_mean,
        "lat_acc_max":        lat_acc_max,

        # ── Class one-hot ────────────────────────────────────────────────
        "is_car":             1 if cls == "car" else 0,
        "is_motorcycle":      1 if cls == "motorcycle" else 0,
        "is_bus":             1 if cls == "bus" else 0,
        "is_truck":           1 if cls == "truck" else 0,

        # ── Bbox-normalized speed (camera-agnostic) ───────────────────────
        # speed_ms_* : approximate m/s từ bbox scale estimation
        "speed_ms_mean":      float(np.mean(speed_ms)),
        "speed_ms_std":       float(np.std(speed_ms)),
        "speed_ms_last":      float(speed_ms[-1]),
        "speed_ms_recent":    float(np.mean(
                                  speed_ms[t_vals >= t_cutoff - 0.5]
                                  if (t_vals >= t_cutoff - 0.5).any()
                                  else speed_ms[-3:])),
        # speed_norm_bbox = speed_px / sqrt(w_px*h_px) — purely zoom-invariant
        "speed_norm_bbox":    float(np.mean(speed_norm_bbox)),
        # bbox scale features
        "bbox_size_mean":     bbox_size_mean,
        "px_per_m_mean":      px_per_m_mean,
    }
    return f


# ── Data loading ──────────────────────────────────────────────────────────────

def load_data(traj_dir: Path, labels_csv: Path,
              primary_horizon: float = 1.0,
              high_conf_only: bool = False) -> tuple:
    """
    Trả về X_train, y_train, meta, horizon_data, feature_names
    """
    # ── Load turning labels ──────────────────────────────────────────────
    labels = pd.read_csv(labels_csv)

    # Filter: vehicle only (no pedestrian), valid classes
    labels = labels[labels["class"].isin(["car", "motorcycle", "bus", "truck"])]
    if high_conf_only:
        labels = labels[labels["confidence"] == "high"]

    print(f"[Data] Goal events: {len(labels)}"
          f"  ({'high-conf only' if high_conf_only else 'all'})")
    print("[Data] Label distribution:")
    for lbl in LABEL_NAMES:
        cnt = (labels["goal_label"] == lbl).sum()
        print(f"         {lbl:<10} {cnt:>5}  ({100*cnt/len(labels):.1f}%)")

    # ── Load trajectories ────────────────────────────────────────────────
    csv_files = sorted(traj_dir.glob("*.csv"))
    csv_files = [f for f in csv_files if f.stem != "summary"]
    print(f"[Data] Loading {len(csv_files)} trajectory CSVs...")

    needed_cols = {"t", "video_id", "track_id", "class",
                   "cx_px", "cy_px", "vx_px", "vy_px", "speed_px", "heading_deg",
                   "w_px", "h_px"}
    traj_map: dict[tuple, pd.DataFrame] = {}

    for tf in csv_files:
        try:
            df = pd.read_csv(tf)
        except Exception:
            continue
        if not needed_cols.issubset(df.columns):
            continue
        for (vid, tid), grp in df.groupby(["video_id", "track_id"]):
            key = (str(vid), int(tid))
            traj_map[key] = grp.sort_values("t").reset_index(drop=True)

    print(f"[Data] Trajectory index: {len(traj_map)} (video_id, track_id) pairs")

    # ── Extract features per horizon ─────────────────────────────────────
    horizon_data: dict[float, tuple] = {}

    for h in EVAL_HORIZONS:
        Xh, yh, mh = [], [], []
        skipped = 0
        for _, row in labels.iterrows():
            key = (str(row["video_id"]), int(row["track_id"]))
            if key not in traj_map:
                skipped += 1
                continue

            traj       = traj_map[key]
            t_enter    = float(row["t_enter"])
            t_exit     = float(row["t_exit"])
            t_junction = t_enter + JUNCTION_FRAC * (t_exit - t_enter)
            t_cutoff   = t_junction

            feat = extract_features_px(traj, t_cutoff, str(row["class"]), window=h)
            if feat is None:
                skipped += 1
                continue

            Xh.append(feat)
            yh.append(row["goal_label"])
            mh.append({
                "cls":       row["class"],
                "video_id":  row["video_id"],
                "track_id":  row["track_id"],
                "horizon":   h,
                "confidence": row["confidence"],
            })

        if Xh:
            df_x = pd.DataFrame(Xh)
            feature_names = list(df_x.columns)
            horizon_data[h] = (df_x.values.astype(np.float32),
                               np.array(yh),
                               mh)
            print(f"[Data] Horizon {h:.1f}s: {len(Xh)} samples  "
                  f"(skipped {skipped})")

    if primary_horizon not in horizon_data:
        available = sorted(horizon_data.keys())
        primary_horizon = available[-1]
        print(f"[Data] Fallback primary horizon → {primary_horizon:.1f}s")

    X_train, y_train, meta = horizon_data[primary_horizon]
    return X_train, y_train, meta, horizon_data, feature_names


# ── ECE ──────────────────────────────────────────────────────────────────────

def compute_ece(y_true, y_prob, n_bins: int = 10) -> float:
    max_prob = y_prob.max(axis=1)
    correct  = (y_prob.argmax(axis=1) == y_true).astype(float)
    bins = np.linspace(0, 1, n_bins + 1)
    ece  = 0.0
    for i in range(n_bins):
        mask = (max_prob >= bins[i]) & (max_prob < bins[i + 1])
        if mask.sum() == 0:
            continue
        ece += mask.mean() * abs(correct[mask].mean() - max_prob[mask].mean())
    return float(ece)


# ── Charts ───────────────────────────────────────────────────────────────────

def save_confusion_matrix(cm: np.ndarray, label_names: list,
                           path: Path, title: str = "Confusion Matrix"):
    if not HAS_MPL:
        return
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    plt.colorbar(im, ax=ax)
    ax.set(xticks=range(len(label_names)), yticks=range(len(label_names)),
           xticklabels=label_names, yticklabels=label_names,
           ylabel="True label", xlabel="Predicted label", title=title)
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
    thresh = cm.max() / 2
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, f"{cm[i,j]}", ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black", fontsize=9)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def save_horizon_accuracy(horizon_accs: dict, path: Path):
    if not HAS_MPL:
        return
    hs = sorted(horizon_accs.keys())
    accs = [horizon_accs[h] for h in hs]
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot([str(h) for h in hs], accs, "o-", color="#4fc3f7", linewidth=2,
            markersize=8)
    for h, a in zip(hs, accs):
        ax.annotate(f"{a:.1%}", (str(h), a), textcoords="offset points",
                    xytext=(0, 8), ha="center", fontsize=9)
    ax.set(xlabel="Horizon trước ngã tư (giây)", ylabel="Accuracy",
           title="Goal Classifier — Accuracy theo Horizon (Real CCTV)",
           ylim=(0, 1))
    ax.axhline(0.25, linestyle="--", color="gray", label="Random baseline")
    ax.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def save_class_accuracy(class_accs: dict, path: Path):
    if not HAS_MPL:
        return
    classes = list(class_accs.keys())
    accs    = [class_accs[c] for c in classes]
    colors  = [LABEL_COLORS.get(c, "#90caf9") for c in classes]
    fig, ax = plt.subplots(figsize=(8, 4))
    bars = ax.bar(classes, accs, color=colors, edgecolor="white", linewidth=0.8)
    for bar, acc in zip(bars, accs):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                f"{acc:.1%}", ha="center", va="bottom", fontsize=9)
    ax.set(xlabel="Goal label", ylabel="Recall (per-class accuracy)",
           title="Goal Classifier — Per-class Accuracy (Real CCTV)",
           ylim=(0, 1.1))
    ax.axhline(0.25, linestyle="--", color="gray", label="Random baseline")
    ax.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def save_feature_importance(clf, feature_names: list, path: Path, top_n: int = 20):
    if not HAS_MPL:
        return
    model = clf.estimator if hasattr(clf, "estimator") else clf
    if not hasattr(model, "feature_importances_"):
        return
    imp = model.feature_importances_
    idx = np.argsort(imp)[-top_n:]
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.barh(range(len(idx)), imp[idx], color="#4fc3f7", edgecolor="white")
    ax.set_yticks(range(len(idx)))
    ax.set_yticklabels([feature_names[i] for i in idx], fontsize=8)
    ax.set(xlabel="Feature importance", title="Top Features — Real CCTV Classifier")
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    global JUNCTION_FRAC
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--traj-dir",      default=str(TRAJ_DIR))
    ap.add_argument("--labels",        default=str(LABELS_CSV))
    ap.add_argument("--out-weights",   default=str(WEIGHTS_DIR))
    ap.add_argument("--out-assets",    default=str(ASSET_DIR))
    ap.add_argument("--horizon",       type=float, default=1.0,
                    help="Primary training horizon in seconds (default 1.0)")
    ap.add_argument("--model",         default="gb",
                    choices=["gb", "rf", "mlp"])
    ap.add_argument("--high-conf-only", action="store_true")
    ap.add_argument("--cv",            type=int, default=5)
    ap.add_argument("--test-size",     type=float, default=0.2,
                    help="Test split fraction (default 0.2 = 20%%)")
    ap.add_argument("--junction-frac", type=float, default=JUNCTION_FRAC)
    ap.add_argument("--no-class-weight", action="store_true",
                    help="Disable custom class weights")
    args = ap.parse_args()

    JUNCTION_FRAC = args.junction_frac
    out_weights = Path(args.out_weights)
    out_assets  = Path(args.out_assets)
    out_weights.mkdir(parents=True, exist_ok=True)
    out_assets.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    print("=" * 60)
    print("Phase 3: Goal Classifier — Real CCTV (proper train/test split)")
    print(f"  model={args.model}  horizon={args.horizon}s  jf={JUNCTION_FRAC}"
          f"  test={args.test_size:.0%}"
          f"  cw={'off' if args.no_class_weight else 'custom'}")
    print("=" * 60)

    # ── Load all feature data ──────────────────────────────────────────────
    X_all, y_all, meta_all, horizon_data, feature_names = load_data(
        traj_dir=Path(args.traj_dir),
        labels_csv=Path(args.labels),
        primary_horizon=args.horizon,
        high_conf_only=args.high_conf_only,
    )
    print(f"\n[Split] Primary horizon {args.horizon:.1f}s: "
          f"{len(X_all)} samples, {len(feature_names)} features")

    # ── Encode labels ──────────────────────────────────────────────────────
    le = LabelEncoder()
    le.fit(LABEL_NAMES)
    y_all_enc = le.transform(y_all)

    # ── Track-level stratified train/test split ────────────────────────────
    # Split by unique (video_id, track_id) to prevent same track in both sets
    meta_all_list = meta_all  # list of dicts with 'video_id', 'track_id'
    track_keys    = [(m["video_id"], m["track_id"]) for m in meta_all_list]
    unique_tracks = list(dict.fromkeys(track_keys))  # ordered unique

    # Map each unique track to its label (from primary horizon)
    track_label_map = {}
    for key, lbl in zip(track_keys, y_all):
        track_label_map[key] = lbl
    unique_labels = [track_label_map[t] for t in unique_tracks]

    train_tracks, test_tracks = train_test_split(
        unique_tracks,
        test_size=args.test_size,
        stratify=unique_labels,
        random_state=42,
    )
    train_set = set(train_tracks)
    test_set  = set(test_tracks)

    # Apply split to primary horizon arrays
    train_mask = np.array([k in train_set for k in track_keys])
    test_mask  = ~train_mask

    X_tr, y_tr = X_all[train_mask], y_all_enc[train_mask]
    X_te, y_te = X_all[test_mask],  y_all_enc[test_mask]

    print(f"[Split] Train: {len(X_tr)}  Test: {len(X_te)}")
    for lbl in LABEL_NAMES:
        li = le.transform([lbl])[0]
        print(f"  {lbl:<10}  train={int((y_tr==li).sum()):4d}  "
              f"test={int((y_te==li).sum()):4d}")

    # ── Scale (fit on TRAIN only) ─────────────────────────────────────────
    scaler   = StandardScaler()
    X_tr_sc  = scaler.fit_transform(X_tr)
    X_te_sc  = scaler.transform(X_te)

    # ── Build classifier ───────────────────────────────────────────────────
    if args.model == "gb":
        base_clf = GradientBoostingClassifier(
            n_estimators=200, max_depth=4, learning_rate=0.1,
            min_samples_leaf=5, random_state=42,
        )
    elif args.model == "rf":
        base_clf = RandomForestClassifier(
            n_estimators=200, max_depth=8, min_samples_leaf=3,
            random_state=42, n_jobs=-1,
        )
    else:
        base_clf = MLPClassifier(
            hidden_layer_sizes=(128, 64), max_iter=500,
            early_stopping=True, random_state=42,
        )
    clf = CalibratedClassifierCV(base_clf, cv=3, method="sigmoid")

    # ── Cross-validation (on TRAIN set only) ──────────────────────────────
    print(f"\n[CV] {args.cv}-fold on train set (balanced_accuracy)...")
    cv_scores = cross_val_score(
        CalibratedClassifierCV(
            GradientBoostingClassifier(n_estimators=200, max_depth=4,
                                        learning_rate=0.1, min_samples_leaf=5,
                                        random_state=42),
            cv=3, method="sigmoid"),
        X_tr_sc, y_tr,
        cv=StratifiedKFold(n_splits=args.cv, shuffle=True, random_state=42),
        scoring="balanced_accuracy",
    )
    print(f"[CV] balanced_accuracy: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
    print(f"[CV] per-fold: {', '.join(f'{s:.3f}' for s in cv_scores)}")

    # ── Compute custom sample weights (TRAIN only) ─────────────────────────
    if args.no_class_weight:
        sw_tr = None
    else:
        label_names_enc = le.transform(LABEL_NAMES)
        sw_map = {le.transform([lbl])[0]: w
                  for lbl, w in CUSTOM_CLASS_WEIGHTS.items()}
        sw_tr = np.array([sw_map.get(y, 1.0) for y in y_tr])
        wt_str = "  ".join(
            f"{lbl}×{CUSTOM_CLASS_WEIGHTS[lbl]}" for lbl in LABEL_NAMES)
        print(f"[Weight] custom: {wt_str}")

    # ── Fit final model on TRAIN set ──────────────────────────────────────
    print(f"\n[Train] Fitting on {len(X_tr_sc)} train samples...")
    clf.fit(X_tr_sc, y_tr, sample_weight=sw_tr)

    # ── Evaluate on HELD-OUT TEST set ─────────────────────────────────────
    y_te_pred = clf.predict(X_te_sc)
    y_te_prob = clf.predict_proba(X_te_sc)

    test_acc = accuracy_score(y_te, y_te_pred)
    test_f1  = f1_score(y_te, y_te_pred, average="macro")
    test_bal = accuracy_score(y_te, y_te_pred)   # overall
    test_ece = compute_ece(y_te, y_te_prob)
    test_bal_acc = float(np.mean([
        accuracy_score(y_te[y_te == le.transform([lbl])[0]],
                       y_te_pred[y_te == le.transform([lbl])[0]])
        for lbl in LABEL_NAMES
        if (y_te == le.transform([lbl])[0]).sum() > 0
    ]))

    print(f"\n[TEST] accuracy      = {test_acc:.4f}")
    print(f"[TEST] balanced_acc  = {test_bal_acc:.4f}")
    print(f"[TEST] f1_macro      = {test_f1:.4f}")
    print(f"[TEST] ECE           = {test_ece:.4f}")

    # Per-class recall on test
    class_accs_test = {}
    print("\n[TEST] Per-class recall:")
    for lbl in LABEL_NAMES:
        li   = le.transform([lbl])[0]
        mask = (y_te == li)
        if mask.sum() == 0:
            class_accs_test[lbl] = 0.0
            print(f"  {lbl:<10}: N/A  (n=0)")
            continue
        recall = accuracy_score(y_te[mask], y_te_pred[mask])
        class_accs_test[lbl] = float(recall)
        print(f"  {lbl:<10}: {recall:.4f}  (n={mask.sum()})")

    report = classification_report(
        y_te, y_te_pred, target_names=le.classes_, output_dict=True)
    print("\n[TEST] Classification report:")
    print(classification_report(y_te, y_te_pred, target_names=le.classes_))

    cm     = confusion_matrix(y_te, y_te_pred)
    cm_pct = cm.astype(float) / np.maximum(cm.sum(axis=1, keepdims=True), 1)

    # ── Per-horizon eval on TEST tracks ───────────────────────────────────
    print("\n[Eval] Per-horizon on TEST tracks:")
    horizon_full = {}
    horizon_accs = {}

    for h in sorted(horizon_data.keys()):
        Xh, yh, mh = horizon_data[h]
        # Filter to test tracks only
        h_mask  = np.array([(m["video_id"], m["track_id"]) in test_set
                             for m in mh])
        if h_mask.sum() < 10:
            continue
        Xh_te   = scaler.transform(Xh[h_mask])
        yh_te   = le.transform(yh[h_mask])
        pred_h  = clf.predict(Xh_te)
        prob_h  = clf.predict_proba(Xh_te)
        acc_h   = accuracy_score(yh_te, pred_h)
        f1_h    = f1_score(yh_te, pred_h, average="macro")
        ece_h   = compute_ece(yh_te, prob_h)
        horizon_accs[h] = acc_h
        horizon_full[h] = {
            "accuracy":   round(acc_h, 4),
            "f1_macro":   round(f1_h, 4),
            "ece":        round(ece_h, 4),
            "n_test":     int(h_mask.sum()),
        }
        print(f"  h={h:.1f}s  n_test={h_mask.sum():4d}  "
              f"acc={acc_h:.4f}  f1={f1_h:.4f}  ece={ece_h:.4f}")

    # ── Save model ─────────────────────────────────────────────────────────
    model_path   = out_weights / "goal_classifier_real.pkl"
    scaler_path  = out_weights / "goal_scaler_real.pkl"
    feat_path    = out_weights / "goal_feature_names_real.json"
    encoder_path = out_weights / "goal_label_encoder_real.pkl"

    with open(model_path,   "wb") as f: pickle.dump(clf, f)
    with open(scaler_path,  "wb") as f: pickle.dump(scaler, f)
    with open(encoder_path, "wb") as f: pickle.dump(le, f)
    with open(feat_path,    "w", encoding="utf-8") as f:
        json.dump(feature_names, f, ensure_ascii=False, indent=2)

    # ── Save metrics JSON ──────────────────────────────────────────────────
    metrics = {
        "run_info": {
            "model_type":       args.model,
            "primary_horizon":  args.horizon,
            "high_conf_only":   args.high_conf_only,
            "class_weight":     "custom" if not args.no_class_weight else "none",
            "custom_weights":   CUSTOM_CLASS_WEIGHTS,
            "junction_frac":    JUNCTION_FRAC,
            "test_size":        args.test_size,
            "eval_on_test_set": True,
            "n_features":       len(feature_names),
            "feature_names":    feature_names,
            "label_names":      list(le.classes_),
        },
        "cross_validation": {
            "n_folds":    args.cv,
            "split":      "train_only",
            "scoring":    "balanced_accuracy",
            "mean":       round(float(cv_scores.mean()), 4),
            "std":        round(float(cv_scores.std()), 4),
            "per_fold":   [round(float(s), 4) for s in cv_scores],
        },
        "test_set": {
            "n_train":          int(len(X_tr)),
            "n_test":           int(len(X_te)),
            "accuracy":         round(test_acc, 4),
            "balanced_accuracy": round(test_bal_acc, 4),
            "f1_macro":         round(test_f1, 4),
            "ece":              round(test_ece, 4),
            "class_recall":     {k: round(v, 4) for k, v in class_accs_test.items()},
            "classification_report": report,
            "confusion_matrix":     cm.tolist(),
            "confusion_matrix_pct": cm_pct.round(3).tolist(),
        },
        "per_horizon_test": horizon_full,
        "label_distribution": {
            lbl: int((np.array(y_all) == lbl).sum()) for lbl in LABEL_NAMES
        },
    }

    metrics_path = out_assets / "metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    # ── Charts ──────────────────────────────────────────────────────────────
    if HAS_MPL:
        save_confusion_matrix(
            cm, list(le.classes_),
            out_assets / "confusion_matrix.png",
            title=f"Confusion Matrix TEST — Real CCTV (h={args.horizon:.1f}s)",
        )
        save_confusion_matrix(
            (cm_pct * 100).astype(int), list(le.classes_),
            out_assets / "confusion_matrix_pct.png",
            title=f"Normalized CM % TEST — Real CCTV (h={args.horizon:.1f}s)",
        )
        save_horizon_accuracy(horizon_accs, out_assets / "horizon_accuracy.png")
        save_class_accuracy(class_accs_test, out_assets / "class_accuracy.png")
        save_feature_importance(clf, feature_names,
                                 out_assets / "feature_importance.png")

    elapsed = time.time() - t0
    print("\n" + "=" * 60)
    print(f"DONE in {elapsed:.1f}s")
    print(f"[HONEST TEST-SET RESULTS]")
    print(f"  accuracy      : {test_acc:.4f}")
    print(f"  balanced_acc  : {test_bal_acc:.4f}")
    print(f"  f1_macro      : {test_f1:.4f}")
    print(f"  straight rec  : {class_accs_test.get('straight',0):.4f}")
    print(f"  left recall   : {class_accs_test.get('left',0):.4f}")
    print(f"  right recall  : {class_accs_test.get('right',0):.4f}")
    print(f"  u_turn recall : {class_accs_test.get('u_turn',0):.4f}")
    print(f"  CV bal_acc    : {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
    print("=" * 60)


if __name__ == "__main__":
    main()
