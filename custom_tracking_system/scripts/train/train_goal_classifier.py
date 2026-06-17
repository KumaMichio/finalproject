"""
train_goal_classifier.py — Goal Direction Classifier cho CCTV Hà Nội
=====================================================================
Dự đoán hướng đi của phương tiện tại ngã tư (straight / left / right / u_turn)
từ lịch sử trajectory TRƯỚC khi vào ngã tư.

Thiết kế hướng CCTV thực tế:
  - Feature hoàn toàn ở world-space (m/s, deg) — độc lập với camera/góc nhìn
  - Đánh giá theo horizon thời gian thực (dự đoán trước 0.5s / 1s / 1.5s / 2s / 3s)
  - Calibration ECE — "confidence 80% có nghĩa là đúng 80% không?"
  - Accuracy per class (xe máy quan trọng nhất trong traffic VN)
  - Inference time cho 30+ xe đồng thời

Usage:
  python scripts/train/train_goal_classifier.py
  python scripts/train/train_goal_classifier.py --data custom_tracking_system/data/trajectories_hanoi
  python scripts/train/train_goal_classifier.py --horizon 1.5 --model rf
"""

import argparse
import json
import math
import pickle
import sys
import time
from collections import defaultdict
from pathlib import Path

# Fix Windows console encoding (cp1252 does not support Vietnamese)
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import (accuracy_score, classification_report,
                              confusion_matrix, f1_score)
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    HAS_MPL = True
except ImportError:
    HAS_MPL = False

# ── Constants ──────────────────────────────────────────────────────────────
CLASS_SPEED_LIMIT = {
    "car": 16.7, "motorcycle": 13.9, "bus": 11.1, "truck": 9.7, "person": 2.0
}
LABEL_NAMES   = ["straight", "left", "right", "u_turn"]
LABEL_COLORS  = {"straight": "#4fc3f7", "left": "#ff8a65",
                  "right": "#a5d6a7", "u_turn": "#ce93d8"}
EVAL_HORIZONS = [0.5, 1.0, 1.5, 2.0, 3.0]   # seconds before junction entry
TRAJ_WINDOW   = 4.0                           # extended 3→4s: thấy đủ giai đoạn giảm tốc u_turn
TRAJ_MIN_FRAMES = 5                           # discard events with < 5 frames


# ── Label ──────────────────────────────────────────────────────────────────
def angle_to_label(angle_deg: float) -> str:
    """
    turn_angle_deg = exit_heading - entry_heading (CARLA convention).
    CARLA dùng hệ tọa độ trái (left-hand): yaw tăng theo chiều kim đồng hồ.
    → dương = rẽ phải (right), âm = rẽ trái (left).
    """
    a = float(angle_deg)
    while a > 180:  a -= 360
    while a < -180: a += 360
    if abs(a) < 30:   return "straight"
    if abs(a) > 150:  return "u_turn"
    return "right" if a > 0 else "left"


# ── Feature extraction ────────────────────────────────────────────────────
def extract_features(traj: pd.DataFrame, t_cutoff: float,
                     cls: str, window: float = TRAJ_WINDOW) -> dict | None:
    """
    Trích feature từ trajectory TRƯỚC thời điểm t_cutoff.
    Tất cả feature ở world-space (m/s, degree) — camera-agnostic.
    """
    sub = traj[(traj["t"] <= t_cutoff) &
               (traj["t"] >= t_cutoff - window)].copy()
    if len(sub) < TRAJ_MIN_FRAMES:
        return None

    sub = sub.sort_values("t")
    speed  = sub["speed"].values
    vx     = sub["vx"].values
    vy     = sub["vy"].values
    head   = sub["heading"].values
    t_vals = sub["t"].values
    dt     = t_vals[-1] - t_vals[0]
    if dt < 0.1:
        return None

    speed_limit = CLASS_SPEED_LIMIT.get(cls, 13.9)

    # Heading delta per step, wrapped to [-180, 180]
    head_diff = np.diff(head)
    head_diff = ((head_diff + 180) % 360) - 180
    dt_arr    = np.diff(t_vals)
    dt_arr    = np.where(dt_arr < 1e-6, 1e-6, dt_arr)
    head_rate = head_diff / dt_arr                 # deg/s

    # ── NEW: cumulative heading features (key for u_turn detection) ──────
    # Total absolute heading change → straight≈0, turn≈90, u_turn≈180+
    total_abs_heading  = float(np.sum(np.abs(head_diff)))
    # Signed cumulative: left→negative, right→positive, u_turn→large magnitude
    total_sign_heading = float(np.sum(head_diff))
    # Heading range (spread): u_turn sweeps wide range, straight stays narrow
    heading_range      = float(np.max(head) - np.min(head))
    # Monotonicity: does heading keep turning same direction?
    # 1.0 = always same direction, 0.0 = oscillates
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

    # ── NEW: braking depth (u_turn requires deeper braking) ─────────────
    speed_min       = float(np.min(speed))
    speed_max       = float(np.max(speed))
    speed_drop      = speed_max - speed_min           # total drop magnitude
    speed_min_ratio = speed_min / max(speed_max, 0.1) # 0=full stop, 1=no brake
    # Final-third speed (last ~1.3s): how slow are they near junction?
    n_third = max(1, len(speed) // 3)
    speed_final_third = float(np.mean(speed[-n_third:]))

    # Lateral acceleration proxy (|cross product of v and a| / |v|)
    if len(vx) >= 3:
        ax = np.gradient(vx, t_vals)
        ay = np.gradient(vy, t_vals)
        cross = np.abs(vx * ay - vy * ax)
        v_mag = np.sqrt(vx**2 + vy**2)
        lat_acc = np.where(v_mag > 0.1, cross / v_mag, 0.0)
        lat_acc_mean = float(np.mean(lat_acc))
        lat_acc_max  = float(np.max(lat_acc))
    else:
        lat_acc_mean = lat_acc_max = 0.0

    # Speed relative to class limit
    speed_norm = speed / speed_limit

    # Last 0.5s slice (most recent behavior)
    recent = sub[sub["t"] >= t_cutoff - 0.5]
    sp_recent = recent["speed"].values if len(recent) > 0 else speed[-3:]

    f = {
        # Speed statistics
        "speed_mean":        float(np.mean(speed)),
        "speed_std":         float(np.std(speed)),
        "speed_last":        float(speed[-1]),
        "speed_recent":      float(np.mean(sp_recent)),
        "speed_norm_mean":   float(np.mean(speed_norm)),
        "speed_change":      float(speed[-1] - speed[0]),
        "speed_change_rate": float((speed[-1] - speed[0]) / dt),

        # Heading rate statistics
        "head_rate_mean":    float(np.mean(head_rate)),
        "head_rate_std":     float(np.std(head_rate)),
        "head_rate_max":     float(np.max(np.abs(head_rate))),

        # Final heading (sin/cos — camera-agnostic)
        "head_sin":          float(math.sin(math.radians(head[-1]))),
        "head_cos":          float(math.cos(math.radians(head[-1]))),

        # ── NEW cumulative heading features ──────────────────────────────
        "total_abs_heading": total_abs_heading,   # u_turn >> straight
        "total_sign_heading":total_sign_heading,  # sign = left/right
        "heading_range":     heading_range,        # sweep width
        "head_monotonic":    head_monotonic,       # consistent direction?

        # ── NEW braking depth features ───────────────────────────────────
        "speed_min":         speed_min,
        "speed_drop":        speed_drop,
        "speed_min_ratio":   speed_min_ratio,      # u_turn brakes deeper
        "speed_final_third": speed_final_third,    # slow near junction?

        # Lateral acceleration
        "lat_acc_mean":      lat_acc_mean,
        "lat_acc_max":       lat_acc_max,

        # Class (one-hot)
        "is_car":            1 if cls == "car" else 0,
        "is_motorcycle":     1 if cls == "motorcycle" else 0,
        "is_bus":            1 if cls == "bus" else 0,
        "is_truck":          1 if cls == "truck" else 0,
    }
    return f


def features_to_array(feat_dict: dict, feature_names: list) -> np.ndarray:
    return np.array([feat_dict[k] for k in feature_names], dtype=np.float32)


# ── Data loading ──────────────────────────────────────────────────────────
def load_data(data_dir: Path, primary_horizon: float = 1.5):
    """
    Đọc tất cả goal event + trajectory → trích feature tại mỗi horizon.
    Trả về:
      X_train: feature matrix cho primary_horizon (training)
      y_train: labels
      horizon_data: {horizon: (X, y)} cho eval
      meta: episode/actor/class info
    """
    goal_files = sorted(data_dir.glob("episode_*_goals.csv"))
    traj_files = sorted(data_dir.glob("episode_*.csv"))
    traj_files = [f for f in traj_files if "_goals" not in f.name]

    print(f"[Data] Found {len(goal_files)} goal files, {len(traj_files)} traj files")

    # Build trajectory index: (episode_id, actor_id) -> DataFrame
    print("[Data] Loading trajectory CSV files...")
    traj_index: dict[tuple, pd.DataFrame] = defaultdict(list)
    for tf in traj_files:
        df = pd.read_csv(tf, usecols=["t","episode_id","actor_id","class",
                                       "x","y","vx","vy","speed","heading"])
        for (ep, aid), grp in df.groupby(["episode_id", "actor_id"]):
            traj_index[(int(ep), int(aid))].append(grp)

    # Concatenate per actor
    traj_map = {}
    for key, frames in traj_index.items():
        traj_map[key] = pd.concat(frames).sort_values("t").reset_index(drop=True)
    print(f"[Data] Trajectory index: {len(traj_map)} actor-episode pairs")

    # Load all goal events
    all_goals = []
    for gf in goal_files:
        all_goals.append(pd.read_csv(gf))
    goals = pd.concat(all_goals, ignore_index=True)
    print(f"[Data] Total goal events: {len(goals)}")

    # Label
    goals["label"] = goals["turn_angle_deg"].apply(angle_to_label)
    print("[Data] Label distribution:")
    for lbl, cnt in goals["label"].value_counts().items():
        print(f"         {lbl:<10} {cnt:>5}  ({100*cnt/len(goals):.1f}%)")

    # Extract features for each horizon
    horizon_data: dict[float, tuple] = {}
    meta_rows = []

    for h in EVAL_HORIZONS:
        Xh, yh, mh = [], [], []
        for _, row in goals.iterrows():
            key = (int(row["episode_id"]), int(row["actor_id"]))
            if key not in traj_map:
                continue
            traj = traj_map[key]
            t_cut = float(row["t_enter"]) - h
            feat = extract_features(traj, t_cut, str(row["class"]))
            if feat is None:
                continue
            Xh.append(feat)
            yh.append(row["label"])
            mh.append({"cls": row["class"], "episode": row["episode_id"],
                        "actor": row["actor_id"], "horizon": h})

        if Xh:
            df_x = pd.DataFrame(Xh)
            feature_names = list(df_x.columns)
            Xh_arr = df_x.values.astype(np.float32)
            yh_arr = np.array(yh)
            horizon_data[h] = (Xh_arr, yh_arr, mh)
            print(f"[Data] Horizon {h:.1f}s: {len(Xh)} samples")

    # Primary horizon for training
    X_train, y_train, meta = horizon_data[primary_horizon]
    return X_train, y_train, meta, horizon_data, feature_names


# ── ECE (calibration) ─────────────────────────────────────────────────────
def compute_ece(y_true, y_prob, n_bins=10) -> float:
    """Expected Calibration Error — thấp = confidence đáng tin cậy hơn."""
    max_prob = y_prob.max(axis=1)
    correct  = (y_prob.argmax(axis=1) == y_true).astype(float)
    bins = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        mask = (max_prob >= bins[i]) & (max_prob < bins[i + 1])
        if mask.sum() == 0:
            continue
        acc = correct[mask].mean()
        conf = max_prob[mask].mean()
        ece += mask.mean() * abs(acc - conf)
    return float(ece)


# ── Training ───────────────────────────────────────────────────────────────
def train_models(X, y, scaler):
    """Train RF + GBT, calibrate, chọn model tốt nhất."""
    le = LabelEncoder()
    le.fit(LABEL_NAMES)
    y_enc = le.transform(y)

    Xs = scaler.transform(X)

    models = {
        "RandomForest": RandomForestClassifier(
            n_estimators=300, max_depth=12, min_samples_leaf=5,
            class_weight="balanced", random_state=42, n_jobs=-1),
        "GradientBoosting": GradientBoostingClassifier(
            n_estimators=200, max_depth=5, learning_rate=0.05,
            subsample=0.8, random_state=42),
    }

    results = {}
    best_f1, best_name, best_model = 0.0, None, None

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    for name, clf in models.items():
        print(f"\n[Train] {name} ...")
        # Calibrate for proper probability output
        cal_clf = CalibratedClassifierCV(clf, cv=3, method="isotonic")
        cal_clf.fit(Xs, y_enc)

        # Cross-val F1
        f1_scores = cross_val_score(cal_clf, Xs, y_enc, cv=cv,
                                    scoring="f1_weighted", n_jobs=-1)
        f1_cv = float(f1_scores.mean())

        # In-sample (for calibration eval)
        y_prob = cal_clf.predict_proba(Xs)
        y_pred = cal_clf.predict(Xs)
        ece = compute_ece(y_enc, y_prob)
        acc = accuracy_score(y_enc, y_pred)

        print(f"  CV F1-weighted: {f1_cv:.3f} (±{f1_scores.std():.3f})")
        print(f"  Train Acc: {acc:.3f}  ECE: {ece:.3f}")

        results[name] = {"model": cal_clf, "f1_cv": f1_cv, "ece": ece}
        if f1_cv > best_f1:
            best_f1, best_name, best_model = f1_cv, name, cal_clf

    print(f"\n[Train] Best model: {best_name}  (CV F1={best_f1:.3f})")
    return best_model, best_name, results, le


# ── Evaluation ─────────────────────────────────────────────────────────────
def evaluate(model, le, scaler, horizon_data, feature_names, out_dir: Path):
    """Đánh giá theo tiêu chí CCTV thực tế."""
    out_dir.mkdir(parents=True, exist_ok=True)
    report = {}

    # ── 1. Primary evaluation (1.5s horizon) ──────────────────────────────
    X_pri, y_pri, meta_pri = horizon_data[1.5]
    Xs_pri = scaler.transform(X_pri)
    y_enc  = le.transform(y_pri)

    y_pred = model.predict(Xs_pri)
    y_prob = model.predict_proba(Xs_pri)
    y_pred_lbl = le.inverse_transform(y_pred)

    acc = accuracy_score(y_enc, y_pred)
    f1  = f1_score(y_enc, y_pred, average="weighted")
    ece = compute_ece(y_enc, y_prob)

    print(f"\n{'='*60}")
    print(f"  PRIMARY EVALUATION (predict {1.5:.1f}s trước ngã tư)")
    print(f"{'='*60}")
    print(f"  Accuracy      : {acc:.3f}  ({acc*100:.1f}%)")
    print(f"  F1 (weighted) : {f1:.3f}")
    print(f"  ECE           : {ece:.4f}  (< 0.05 = tốt cho deploy)")
    print()
    print(classification_report(y_pri, y_pred_lbl,
                                 target_names=LABEL_NAMES, zero_division=0))
    report["primary_1.5s"] = {"accuracy": acc, "f1_weighted": f1, "ece": ece}

    # ── 2. Accuracy by prediction horizon ─────────────────────────────────
    print(f"\n{'─'*55}")
    print("  ACCURACY THEO HORIZON DỰ ĐOÁN (trước ngã tư bao lâu)")
    print(f"{'─'*55}")
    print(f"  {'Horizon':>8}  {'Samples':>8}  {'Accuracy':>9}  {'F1-W':>7}  {'Dist~':>7}")
    horizon_results = []
    for h in sorted(horizon_data.keys()):
        Xh, yh, mh = horizon_data[h]
        Xsh = scaler.transform(Xh)
        yh_enc = le.transform(yh)

        # Estimate approx distance (speed * horizon)
        # Use mean speed from metadata context — rough estimate ~7 m/s mean
        approx_dist = 7.0 * h

        yh_pred = model.predict(Xsh)
        acc_h   = accuracy_score(yh_enc, yh_pred)
        f1_h    = f1_score(yh_enc, yh_pred, average="weighted", zero_division=0)
        horizon_results.append({"horizon_s": h, "accuracy": acc_h,
                                 "f1_weighted": f1_h, "n_samples": len(Xh),
                                 "approx_dist_m": approx_dist})
        print(f"  {h:>7.1f}s  {len(Xh):>8}  {acc_h:>8.1%}  {f1_h:>7.3f}  ~{approx_dist:.0f}m")

    report["horizon_accuracy"] = horizon_results

    # ── 3. Per-class accuracy ──────────────────────────────────────────────
    print(f"\n{'─'*55}")
    print("  ACCURACY PER VEHICLE CLASS (1.5s horizon)")
    print(f"{'─'*55}")
    cls_results = {}
    meta_cls = [m["cls"] for m in meta_pri]
    for veh_cls in ["motorcycle", "car", "bus", "truck"]:
        mask = np.array([c == veh_cls for c in meta_cls])
        if mask.sum() < 5:
            continue
        acc_c = accuracy_score(y_enc[mask], y_pred[mask])
        f1_c  = f1_score(y_enc[mask], y_pred[mask], average="weighted", zero_division=0)
        n     = mask.sum()
        importance = " ← quan trọng nhất VN" if veh_cls == "motorcycle" else ""
        print(f"  {veh_cls:<12}  n={n:>5}  acc={acc_c:.1%}  f1={f1_c:.3f}{importance}")
        cls_results[veh_cls] = {"n": int(n), "accuracy": acc_c, "f1_weighted": f1_c}

    report["per_class"] = cls_results

    # ── 4. Inference time ─────────────────────────────────────────────────
    print(f"\n{'─'*55}")
    print("  INFERENCE TIME (30 xe đồng thời — tiêu chí CCTV thực tế)")
    print(f"{'─'*55}")
    batch_30 = scaler.transform(X_pri[:30] if len(X_pri) >= 30 else X_pri)
    times = []
    for _ in range(200):
        t0 = time.perf_counter()
        _ = model.predict_proba(batch_30)
        times.append((time.perf_counter() - t0) * 1000)

    t_mean = np.mean(times[10:])
    t_p99  = np.percentile(times[10:], 99)
    fps_ok = "✓ OK" if t_mean < 100 else "✗ Quá chậm"
    print(f"  Batch 30 xe : mean={t_mean:.1f}ms  p99={t_p99:.1f}ms  {fps_ok}")
    print(f"  Per vehicle  : {t_mean/30:.2f}ms")
    report["inference_ms"] = {"batch_30_mean": t_mean, "batch_30_p99": t_p99}

    # ── 5. Confusion matrix ────────────────────────────────────────────────
    cm = confusion_matrix(y_enc, y_pred, labels=range(4))
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True).clip(1)

    # ── 6. Plots ──────────────────────────────────────────────────────────
    if HAS_MPL:
        _plot_results(horizon_results, cls_results, cm_norm, y_enc, y_prob,
                      le, out_dir)

    # ── 7. Save report JSON ───────────────────────────────────────────────
    with open(out_dir / "eval_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print(f"\n[Eval] Report → {out_dir / 'eval_report.json'}")
    return report


def _plot_results(horizon_results, cls_results, cm_norm, y_true, y_prob,
                  le, out_dir):
    fig = plt.figure(figsize=(18, 12), facecolor="#0d1117")
    fig.suptitle("Goal Classifier — Đánh giá cho CCTV Hà Nội",
                 fontsize=15, color="white", y=0.98)
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.35)

    ax_color = "#0d1117"
    txt_color = "white"
    grid_color = "#2d2d2d"

    def style(ax, title):
        ax.set_facecolor(ax_color)
        ax.tick_params(colors=txt_color, labelsize=9)
        ax.spines[:].set_color(grid_color)
        ax.set_title(title, color=txt_color, fontsize=10, pad=8)
        for label in ax.get_xticklabels() + ax.get_yticklabels():
            label.set_color(txt_color)

    # 1. Accuracy by horizon
    ax1 = fig.add_subplot(gs[0, 0])
    hs  = [r["horizon_s"] for r in horizon_results]
    acc = [r["accuracy"] for r in horizon_results]
    ax1.plot(hs, acc, "o-", color="#4fc3f7", linewidth=2, markersize=7)
    ax1.axhline(0.5, color="#666", linestyle="--", linewidth=1, label="Baseline 50%")
    ax1.set_xlabel("Horizon (giây trước ngã tư)", color=txt_color, fontsize=9)
    ax1.set_ylabel("Accuracy", color=txt_color, fontsize=9)
    ax1.set_ylim(0, 1.0)
    ax1.invert_xaxis()
    ax1.legend(fontsize=8, facecolor=ax_color, labelcolor=txt_color)
    style(ax1, "Accuracy theo Horizon Dự đoán\n(CCTV: dự đoán sớm = hữu ích hơn)")

    # Annotate approx dist
    for r in horizon_results:
        ax1.annotate(f"~{r['approx_dist_m']:.0f}m",
                     xy=(r["horizon_s"], r["accuracy"]),
                     xytext=(0, 10), textcoords="offset points",
                     fontsize=7, color="#aaa", ha="center")

    # 2. Per-class accuracy
    ax2 = fig.add_subplot(gs[0, 1])
    if cls_results:
        classes = list(cls_results.keys())
        accs    = [cls_results[c]["accuracy"] for c in classes]
        colors  = ["#ff8a65" if c == "motorcycle" else "#4fc3f7" for c in classes]
        bars = ax2.bar(classes, accs, color=colors, edgecolor="#333")
        ax2.axhline(0.5, color="#666", linestyle="--", linewidth=1)
        ax2.set_ylim(0, 1.0)
        for bar, v in zip(bars, accs):
            ax2.text(bar.get_x() + bar.get_width()/2, v + 0.02,
                     f"{v:.1%}", ha="center", va="bottom",
                     color=txt_color, fontsize=9)
        ax2.set_ylabel("Accuracy", color=txt_color, fontsize=9)
    style(ax2, "Accuracy per Vehicle Class\n(cam = xe máy, quan trọng nhất VN)")

    # 3. Confusion matrix
    ax3 = fig.add_subplot(gs[0, 2])
    im = ax3.imshow(cm_norm, cmap="Blues", vmin=0, vmax=1)
    ax3.set_xticks(range(4)); ax3.set_yticks(range(4))
    ax3.set_xticklabels(LABEL_NAMES, rotation=30, ha="right", fontsize=8)
    ax3.set_yticklabels(LABEL_NAMES, fontsize=8)
    for i in range(4):
        for j in range(4):
            ax3.text(j, i, f"{cm_norm[i,j]:.2f}",
                     ha="center", va="center", fontsize=9,
                     color="white" if cm_norm[i,j] > 0.5 else "#333")
    ax3.set_xlabel("Predicted", color=txt_color, fontsize=9)
    ax3.set_ylabel("Actual", color=txt_color, fontsize=9)
    plt.colorbar(im, ax=ax3, fraction=0.046)
    style(ax3, "Confusion Matrix (1.5s horizon, normalized)")

    # 4. Calibration plot (reliability diagram)
    ax4 = fig.add_subplot(gs[1, 0])
    ax4.plot([0, 1], [0, 1], "k--", linewidth=1, label="Perfect calibration")
    for i, lbl in enumerate(LABEL_NAMES):
        y_bin = (y_true == i).astype(int)
        prob_bin = y_prob[:, i]
        if y_bin.sum() < 10:
            continue
        try:
            frac_pos, mean_pred = calibration_curve(y_bin, prob_bin, n_bins=8)
            ax4.plot(mean_pred, frac_pos, "o-", label=lbl,
                     color=list(LABEL_COLORS.values())[i], linewidth=1.5, markersize=5)
        except Exception:
            pass
    ax4.set_xlabel("Mean Predicted Probability", color=txt_color, fontsize=9)
    ax4.set_ylabel("Fraction of Positives", color=txt_color, fontsize=9)
    ax4.set_xlim(0, 1); ax4.set_ylim(0, 1)
    ax4.legend(fontsize=8, facecolor=ax_color, labelcolor=txt_color)
    style(ax4, "Calibration (Reliability Diagram)\n(gần đường chéo = confidence tin cậy)")

    # 5. Label distribution
    ax5 = fig.add_subplot(gs[1, 1])
    unique, counts = np.unique(y_true, return_counts=True)
    labels_present = [LABEL_NAMES[u] for u in unique]
    colors = [list(LABEL_COLORS.values())[u] for u in unique]
    wedges, texts, autotexts = ax5.pie(
        counts, labels=labels_present, colors=colors, autopct="%1.1f%%",
        textprops={"color": txt_color, "fontsize": 9},
        wedgeprops={"edgecolor": ax_color, "linewidth": 2})
    for t in autotexts: t.set_color("white")
    style(ax5, "Phân phối nhãn thực tế\n(distribution of ground-truth turns)")

    # 6. F1 by horizon
    ax6 = fig.add_subplot(gs[1, 2])
    f1s = [r["f1_weighted"] for r in horizon_results]
    ax6.plot(hs, f1s, "s-", color="#a5d6a7", linewidth=2, markersize=7)
    ax6.set_xlabel("Horizon (giây trước ngã tư)", color=txt_color, fontsize=9)
    ax6.set_ylabel("F1 (weighted)", color=txt_color, fontsize=9)
    ax6.set_ylim(0, 1.0)
    ax6.invert_xaxis()
    style(ax6, "F1-Weighted theo Horizon\n(accounts for class imbalance)")

    plt.savefig(out_dir / "goal_classifier_eval.png", dpi=150,
                bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    print(f"[Plot] Saved → {out_dir / 'goal_classifier_eval.png'}")


# ── Feature importance ─────────────────────────────────────────────────────
def print_feature_importance(model, feature_names):
    try:
        base = model.calibrated_classifiers_[0].estimator
        if hasattr(base, "feature_importances_"):
            imp = base.feature_importances_
            pairs = sorted(zip(feature_names, imp), key=lambda x: -x[1])
            print(f"\n{'─'*45}")
            print("  TOP FEATURE IMPORTANCE")
            print(f"{'─'*45}")
            for fname, v in pairs[:12]:
                bar = "█" * int(v * 100)
                print(f"  {fname:<22} {v:.4f}  {bar}")
    except Exception:
        pass


# ── Main ───────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="Train Goal Direction Classifier cho CCTV Hà Nội")
    parser.add_argument("--data", default="custom_tracking_system/data/trajectories_hanoi",
                        help="Thư mục chứa episode_*.csv và episode_*_goals.csv")
    parser.add_argument("--out-weights", default="custom_tracking_system/weights",
                        help="Thư mục lưu model (.pkl)")
    parser.add_argument("--out-eval", default="custom_tracking_system/data/goal_classifier_eval",
                        help="Thư mục lưu kết quả đánh giá")
    parser.add_argument("--horizon", type=float, default=1.5,
                        help="Horizon huấn luyện (giây trước junction, default 1.5)")
    args = parser.parse_args()

    data_dir    = Path(args.data)
    weights_dir = Path(args.out_weights)
    eval_dir    = Path(args.out_eval)

    weights_dir.mkdir(parents=True, exist_ok=True)
    eval_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.time()

    # 1. Load + extract features
    X, y, meta, horizon_data, feature_names = load_data(data_dir, args.horizon)
    print(f"\n[Data] Training set ({args.horizon}s horizon): {len(X)} samples, "
          f"{len(feature_names)} features")

    # 2. Scaler
    scaler = StandardScaler()
    scaler.fit(X)

    # 3. Train
    model, model_name, all_results, le = train_models(X, y, scaler)

    # 4. Feature importance
    print_feature_importance(model, feature_names)

    # 5. Evaluate
    report = evaluate(model, le, scaler, horizon_data, feature_names, eval_dir)

    # 6. Save model
    bundle = {
        "model":         model,
        "scaler":        scaler,
        "label_encoder": le,
        "feature_names": feature_names,
        "model_name":    model_name,
        "train_horizon": args.horizon,
        "eval_report":   report,
    }
    model_path = weights_dir / "goal_classifier.pkl"
    with open(model_path, "wb") as f:
        pickle.dump(bundle, f)

    elapsed = time.time() - t0
    print(f"\n{'='*60}")
    print(f"  DONE  ({elapsed:.1f}s)")
    print(f"{'='*60}")
    print(f"  Model  → {model_path}")
    print(f"  Report → {eval_dir / 'eval_report.json'}")
    if HAS_MPL:
        print(f"  Plot   → {eval_dir / 'goal_classifier_eval.png'}")

    # Print quick summary cho CCTV deployment assessment
    pri = report.get("primary_1.5s", {})
    acc_1s = next((r["accuracy"] for r in report.get("horizon_accuracy", [])
                   if r["horizon_s"] == 1.0), None)
    print(f"\n  ╔══ DEPLOYMENT READINESS ═══════════════════╗")
    print(f"  ║  Accuracy @1.5s : {pri.get('accuracy', 0):.1%}                    ║")
    print(f"  ║  Accuracy @1.0s : {acc_1s:.1%}  (cần ~7m phản ứng)    ║" if acc_1s else "  ║  Accuracy @1.0s : N/A                         ║")
    print(f"  ║  ECE            : {pri.get('ece', 1):.4f}  (< 0.05 = tốt)     ║")
    print(f"  ║  Inference 30xe : {report['inference_ms']['batch_30_mean']:.1f}ms              ║")
    print(f"  ╚═══════════════════════════════════════════╝")


if __name__ == "__main__":
    main()
