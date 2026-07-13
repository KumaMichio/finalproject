"""
verify_goal_labels.py — Hand-verify nhãn hướng đi (Goal Classifier) trên frame thật
====================================================================================
Mục tiêu: đo ACCURACY THẬT của Goal Classifier bằng cách so dự đoán model với
nhãn do NGƯỜI xác nhận (thay vì nhãn auto-label nhiễu).

Cách hoạt động:
  1. Tái tạo CHÍNH XÁC test split của train_goal_classifier_real.py
     (cùng load_data, cùng train_test_split random_state=42) → 1.334 track test.
  2. Lấy mẫu ngẫu nhiên PHÂN TẦNG theo nhãn auto (mặc định 300 track) —
     ước lượng KHÔNG chệch cho accuracy toàn cục.
  3. GUI OpenCV: vẽ đè quỹ đạo (entry→exit, gradient xanh→đỏ) lên khung hình
     thật của track; hiện nhãn auto + heading_change.
  4. Người bấm phím xác nhận / sửa nhãn. Lưu tăng dần → resume được.
  5. Khi thoát: in báo cáo
       - HONEST accuracy  = model_pred vs nhãn NGƯỜI   ← con số cho báo cáo
       - auto vs người    = mức nhiễu của auto-label
       - model vs auto     = sanity (nên ≈ 60% trên mẫu này)

Phím:
  A / Space = đồng ý nhãn auto      S = straight   L = left
  R = right                         U = u_turn     X = không chắc (bỏ qua)
  B = quay lại track trước          Q / Esc = lưu & thoát

Usage:
  python custom_tracking_system/scripts/eval/verify_goal_labels.py
  python custom_tracking_system/scripts/eval/verify_goal_labels.py --n 300 --seed 42
  python custom_tracking_system/scripts/eval/verify_goal_labels.py --report-only
"""

import argparse
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

# ── Paths ──────────────────────────────────────────────────────────────────────
ROOT       = Path(__file__).resolve().parents[2]           # custom_tracking_system/
TRAJ_DIR   = ROOT / "data" / "trajectories_real_vn"
LABELS_CSV = ROOT / "data" / "turning_labels.csv"
IMG_DIR    = ROOT / "data" / "auto_label" / "images"
WEIGHTS    = ROOT / "weights"
OUT_CSV    = ROOT / "data" / "goal_labels_verified.csv"
TRAIN_DIR  = ROOT / "scripts" / "train"
# ── Bằng chứng cho báo cáo ĐATN ─────────────────────────────────────────────────
EVID_DIR    = ROOT.parent / "docs" / "assets" / "goal_classifier_verified"
OVERLAY_DIR = EVID_DIR / "overlays"          # ảnh từng track đã duyệt (audit trail)

LABEL_NAMES = ["straight", "left", "right", "u_turn"]
KEY_TO_LABEL = {ord("s"): "straight", ord("l"): "left",
                ord("r"): "right",    ord("u"): "u_turn"}
LABEL_BGR = {"straight": (247, 195, 79), "left": (101, 138, 255),
             "right": (167, 214, 165),   "u_turn": (216, 147, 206),
             "unsure": (150, 150, 150)}

# ── Import test-split logic from the training script (single source of truth) ──
sys.path.insert(0, str(TRAIN_DIR))
from train_goal_classifier_real import load_data  # noqa: E402
import train_goal_classifier_real as tg           # noqa: E402
from sklearn.model_selection import train_test_split  # noqa: E402
from sklearn.preprocessing import LabelEncoder        # noqa: E402


def reproduce_test_split(horizon=1.5, junction_frac=0.65):
    """Tái tạo đúng test set của train script. Trả về (test_keys, X_all, meta_all, y_all)."""
    tg.JUNCTION_FRAC = junction_frac
    X_all, y_all, meta_all, _hd, feat_names = load_data(
        traj_dir=TRAJ_DIR, labels_csv=LABELS_CSV,
        primary_horizon=horizon, high_conf_only=False,
    )
    track_keys = [(m["video_id"], m["track_id"]) for m in meta_all]
    unique_tracks = list(dict.fromkeys(track_keys))
    tmap = {}
    for k, lbl in zip(track_keys, y_all):
        tmap[k] = lbl
    unique_labels = [tmap[t] for t in unique_tracks]
    _tr, test_tracks = train_test_split(
        unique_tracks, test_size=0.2, stratify=unique_labels, random_state=42)
    return set(test_tracks), X_all, meta_all, y_all, feat_names


def stratified_sample(test_keys, key_label, n, seed):
    """Mẫu phân tầng theo nhãn auto, phân bổ TỶ LỆ (ước lượng không chệch)."""
    rng = np.random.RandomState(seed)
    by_lbl = defaultdict(list)
    for k in test_keys:
        by_lbl[key_label[k]].append(k)
    total = len(test_keys)
    sample = []
    for lbl, keys in by_lbl.items():
        keys = sorted(keys)                      # ổn định trước khi shuffle
        rng.shuffle(keys)
        take = max(1, round(n * len(keys) / total))
        sample.extend(keys[:min(take, len(keys))])
    rng.shuffle(sample)
    return sample


# ── Frame overlay ──────────────────────────────────────────────────────────────

def imread_u(path):
    """Đọc ảnh path Unicode (tên video tiếng Việt) — cv2.imread trả None trên Windows."""
    import cv2
    try:
        return cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_COLOR)
    except Exception:
        return None


def video_frames(video_id, _cache={}):
    if video_id not in _cache:
        imgs = sorted(IMG_DIR.glob(f"{video_id}_*.jpg"))
        _cache[video_id] = imgs
    return _cache[video_id]


def load_track_traj(video_id, track_id, _cache={}):
    if video_id not in _cache:
        f = TRAJ_DIR / f"{video_id}.csv"
        _cache[video_id] = pd.read_csv(f) if f.exists() else None
    df = _cache[video_id]
    if df is None:
        return None
    sub = df[df["track_id"] == track_id].sort_values("t").reset_index(drop=True)
    return sub if len(sub) else None


def render_overlay(video_id, track_id, auto_label, meta_row, idx, total, done, model_pred=None):
    import cv2
    sub = load_track_traj(video_id, track_id)
    frames = video_frames(video_id)
    if sub is None:
        canvas = np.full((720, 1280, 3), 40, np.uint8)
        cv2.putText(canvas, f"NO TRAJ: {video_id} #{track_id}", (30, 360),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
        return canvas

    # Nền = frame lúc thoát khỏi track (nhiều ngữ cảnh nhất)
    last_fid = int(sub["frame_id"].iloc[-1])
    bg = None
    if frames and 0 <= last_fid < len(frames):
        bg = imread_u(frames[last_fid])
    if bg is None:
        h = int(sub["cy_px"].max()) + 60
        w = int(sub["cx_px"].max()) + 60
        bg = np.full((max(h, 480), max(w, 640), 3), 30, np.uint8)
    vis = bg.copy()

    pts = sub[["cx_px", "cy_px"]].values.astype(int)
    npts = len(pts)
    for i in range(1, npts):
        f = i / max(npts - 1, 1)
        color = (int(80 * (1 - f)), int(200 * (1 - f) + 40 * f), int(80 + 175 * f))  # green->red
        cv2.line(vis, tuple(pts[i - 1]), tuple(pts[i]), color, 3)
    cv2.circle(vis, tuple(pts[0]), 9, (80, 220, 80), -1)    # entry
    cv2.circle(vis, tuple(pts[-1]), 9, (60, 60, 230), -1)   # exit
    cv2.putText(vis, "ENTRY", (pts[0][0] + 8, pts[0][1]),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (80, 220, 80), 2)
    cv2.putText(vis, "EXIT", (pts[-1][0] + 8, pts[-1][1]),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (60, 60, 230), 2)

    # Mũi tên heading entry/exit (heading_deg = atan2(vy,vx), y-down)
    for cx, cy, hd, col in [
        (meta_row["cx_entry"], meta_row["cy_entry"], meta_row["entry_heading"], (80, 220, 80)),
        (meta_row["cx_exit"],  meta_row["cy_exit"],  meta_row["exit_heading"],  (60, 60, 230))]:
        rad = np.radians(hd)
        ex = int(cx + 55 * np.cos(rad)); ey = int(cy + 55 * np.sin(rad))
        cv2.arrowedLine(vis, (int(cx), int(cy)), (ex, ey), col, 2, tipLength=0.3)

    # Fit về tối đa 1280 rộng
    H, W = vis.shape[:2]
    scale = min(1280 / W, 800 / H, 1.0)
    if scale < 1.0:
        vis = cv2.resize(vis, (int(W * scale), int(H * scale)))

    # Bảng thông tin
    bar = np.full((175, vis.shape[1], 3), 25, np.uint8)
    ac = LABEL_BGR.get(auto_label, (255, 255, 255))
    mc = LABEL_BGR.get(model_pred, (200, 200, 200))
    cv2.putText(bar, f"[{idx+1}/{total}] verified={done}   {video_id[:42]} #{track_id} ({meta_row['class']})",
                (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (230, 230, 230), 1)
    cv2.putText(bar, f"AUTO-LABEL: {auto_label.upper()}   dHeading={meta_row['heading_change_deg']:+.0f}deg"
                f"   conf={meta_row['confidence']}   frames={meta_row['n_frames']}",
                (12, 66), cv2.FONT_HERSHEY_SIMPLEX, 0.65, ac, 2)
    cv2.putText(bar, f"MODEL du doan: {str(model_pred).upper()}"
                + ("   (khac auto)" if model_pred and model_pred != auto_label else ""),
                (12, 102), cv2.FONT_HERSHEY_SIMPLEX, 0.65, mc, 2)
    cv2.putText(bar, "P=xem xe chay(tua)  A=dong y  S/L/R/U=sua  X=bo  B=lui  Q=luu&thoat",
                (12, 146), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (120, 255, 255), 1)
    return np.vstack([vis, bar])


def play_track(win, video_id, track_id, loops=3, delay=90):
    """Tua lại chu trình: vẽ hộp xe di chuyển qua TỪNG khung hình thật của track."""
    import cv2
    sub = load_track_traj(video_id, track_id)
    frames = video_frames(video_id)
    if sub is None or not frames:
        return
    rows = sub.reset_index(drop=True)
    pts = rows[["cx_px", "cy_px"]].values.astype(int)
    n = len(rows)
    last_bg = None
    for _ in range(loops):
        for i in range(n):
            fid = int(rows["frame_id"].iloc[i])
            bg = imread_u(frames[fid]) if 0 <= fid < len(frames) else None
            bg = bg if bg is not None else last_bg
            if bg is None:
                continue
            last_bg = bg
            vis = bg.copy()
            for j in range(1, i + 1):
                f = j / max(n - 1, 1)
                col = (int(80 * (1 - f)), int(200 * (1 - f) + 40 * f), int(80 + 175 * f))
                cv2.line(vis, tuple(pts[j - 1]), tuple(pts[j]), col, 3)
            cx, cy = pts[i]
            w = int(rows["w_px"].iloc[i]); h = int(rows["h_px"].iloc[i])
            cv2.rectangle(vis, (cx - w // 2, cy - h // 2), (cx + w // 2, cy + h // 2), (0, 230, 255), 2)
            cv2.circle(vis, (cx, cy), 4, (0, 0, 255), -1)
            rad = np.radians(float(rows["heading_deg"].iloc[i]))
            cv2.arrowedLine(vis, (cx, cy),
                            (int(cx + 45 * np.cos(rad)), int(cy + 45 * np.sin(rad))),
                            (0, 230, 255), 2, tipLength=0.3)
            H, W = vis.shape[:2]
            scale = min(1280 / W, 800 / H, 1.0)
            if scale < 1.0:
                vis = cv2.resize(vis, (int(W * scale), int(H * scale)))
            bar = np.full((46, vis.shape[1], 3), 25, np.uint8)
            cv2.putText(bar, f"DANG TUA  {i+1}/{n}  ({rows['class'].iloc[0]})  --  bam phim bat ky de dung",
                        (12, 31), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 230, 255), 2)
            cv2.imshow(win, np.vstack([vis, bar]))
            if (cv2.waitKey(delay) & 0xFF) != 255:
                return


# ── Reporting ──────────────────────────────────────────────────────────────────

def build_predictions(X_all, meta_all):
    """Load model → dự đoán cho mọi (video_id, track_id) ở primary horizon."""
    import pickle
    with open(WEIGHTS / "goal_classifier_real.pkl", "rb") as f: clf = pickle.load(f)
    with open(WEIGHTS / "goal_scaler_real.pkl", "rb") as f: scaler = pickle.load(f)
    with open(WEIGHTS / "goal_label_encoder_real.pkl", "rb") as f: le = pickle.load(f)
    Xs = scaler.transform(X_all)
    pred = le.inverse_transform(clf.predict(Xs))
    key_pred = {}
    for m, p in zip(meta_all, pred):
        key_pred.setdefault((m["video_id"], m["track_id"]), p)  # first occurrence
    return key_pred


def _wilson(k, n, z=1.96):
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    m = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return ((c - m) / d, (c + m) / d)


def build_montage(out_path, max_n=12):
    """Ghép một số ảnh overlay đã duyệt thành 1 hình cho báo cáo."""
    import cv2
    files = sorted(OVERLAY_DIR.glob("*.jpg"))
    if not files:
        return None
    # ưu tiên lấy xen kẽ MISS/OK để hình phản ánh cả đúng lẫn sai
    miss = [f for f in files if f.name.startswith("MISS")]
    ok = [f for f in files if f.name.startswith("OK")]
    pick, i = [], 0
    while len(pick) < min(max_n, len(files)) and (i < len(ok) or i < len(miss)):
        if i < len(ok):
            pick.append(ok[i])
        if len(pick) < max_n and i < len(miss):
            pick.append(miss[i])
        i += 1
    cols = 3
    tiles = []
    for f in pick[:max_n]:
        im = imread_u(f)
        if im is None:
            continue
        h = int(im.shape[0] * 480 / im.shape[1])
        tiles.append(cv2.resize(im, (480, h)))
    if not tiles:
        return None
    hmax = max(t.shape[0] for t in tiles)
    tiles = [cv2.copyMakeBorder(t, 0, hmax - t.shape[0], 0, 0, cv2.BORDER_CONSTANT, value=(25, 25, 25))
             for t in tiles]
    rows = []
    for r in range(0, len(tiles), cols):
        row = tiles[r:r + cols]
        while len(row) < cols:
            row.append(np.full_like(tiles[0], 25))
        rows.append(np.hstack(row))
    cv2.imwrite(str(out_path), np.vstack(rows))
    return out_path


def report(verified_df, key_auto, key_pred):
    v = verified_df[verified_df["human_label"].isin(LABEL_NAMES)].copy()
    n = len(v)
    if n == 0:
        print("\n[report] Chưa có nhãn người nào được xác nhận.")
        return
    v["auto"]  = v.apply(lambda r: key_auto.get((r["video_id"], r["track_id"])), axis=1)
    v["pred"]  = v.apply(lambda r: key_pred.get((r["video_id"], r["track_id"])), axis=1)
    v = v[v["pred"].notna()]
    n = len(v)

    honest = (v["pred"] == v["human_label"]).mean()
    auto_vs_human = (v["auto"] == v["human_label"]).mean()
    model_vs_auto = (v["pred"] == v["auto"]).mean()
    lo, hi = _wilson(int((v["pred"] == v["human_label"]).sum()), n)

    print("\n" + "=" * 64)
    print(f"HAND-VERIFIED REPORT  (n = {n} track đã xác nhận)")
    print("=" * 64)
    print(f"  HONEST accuracy (model vs NGƯỜI) : {honest:.4f}"
          f"   [95% CI {lo:.3f}–{hi:.3f}]   ← số cho báo cáo")
    print(f"  Nhãn auto ĐÚNG (auto vs người)    : {auto_vs_human:.4f}"
          f"   → nhiễu auto-label = {(1-auto_vs_human)*100:.1f}%")
    print(f"  model vs auto (sanity ~0.60)      : {model_vs_auto:.4f}")
    print("\n  Per-class (theo nhãn NGƯỜI):")
    per_class = {}
    for lbl in LABEL_NAMES:
        m = v["human_label"] == lbl
        nb = int(m.sum())
        if nb == 0:
            per_class[lbl] = {"n": 0}
            print(f"    {lbl:<10} n=0"); continue
        k = int((v.loc[m, "pred"] == lbl).sum())
        acc = k / nb
        aacc = (v.loc[m, "auto"] == lbl).mean()
        clo, chi = _wilson(k, nb)
        per_class[lbl] = {"n": nb, "model_recall": round(acc, 4),
                          "wilson95": [round(clo, 3), round(chi, 3)],
                          "auto_correct": round(float(aacc), 4)}
        print(f"    {lbl:<10} n={nb:3d}  model_recall={acc:.3f} "
              f"[{clo:.3f}–{chi:.3f}]  auto_correct={aacc:.3f}")
    print("=" * 64)
    _save_evidence(v, n, honest, (lo, hi), auto_vs_human, model_vs_auto, per_class)


def _save_evidence(v, n, honest, ci, auto_vs_human, model_vs_auto, per_class):
    """Xuất bằng chứng cho báo cáo ĐATN: metrics.json + confusion_matrix.png + montage + summary.md."""
    import json
    from datetime import datetime
    EVID_DIR.mkdir(parents=True, exist_ok=True)

    # Confusion matrix: hàng = nhãn NGƯỜI, cột = dự đoán MODEL
    idx = {l: i for i, l in enumerate(LABEL_NAMES)}
    cm = np.zeros((4, 4), int)
    for _, r in v.iterrows():
        if r["pred"] in idx:
            cm[idx[r["human_label"]], idx[r["pred"]]] += 1

    metrics = {
        "generated": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "description": "Accuracy THẬT của Goal Classifier = dự đoán model vs nhãn NGƯỜI kiểm "
                       "(mẫu phân tầng từ test set held-out). Đây là con số phòng thủ được cho báo cáo.",
        "n_verified": int(n),
        "honest_accuracy": round(float(honest), 4),
        "honest_accuracy_wilson95": [round(ci[0], 4), round(ci[1], 4)],
        "auto_label_correct": round(float(auto_vs_human), 4),
        "auto_label_noise": round(float(1 - auto_vs_human), 4),
        "model_vs_auto_sanity": round(float(model_vs_auto), 4),
        "labels": LABEL_NAMES,
        "confusion_matrix_human_rows_model_cols": cm.tolist(),
        "per_class": per_class,
    }
    (EVID_DIR / "metrics.json").write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")

    # Confusion matrix PNG
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(5, 4.4))
        im = ax.imshow(cm, cmap="Blues")
        ax.set(xticks=range(4), yticks=range(4),
               xticklabels=LABEL_NAMES, yticklabels=LABEL_NAMES,
               xlabel="Model dự đoán", ylabel="Nhãn NGƯỜI kiểm",
               title=f"Goal Classifier — nhãn người (n={n}, acc={honest:.3f})")
        plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
        for a in range(4):
            for b in range(4):
                ax.text(b, a, cm[a, b], ha="center", va="center",
                        color="white" if cm[a, b] > cm.max() / 2 else "black")
        fig.colorbar(im, fraction=0.046)
        fig.tight_layout(); fig.savefig(EVID_DIR / "confusion_matrix_human.png", dpi=130)
        plt.close(fig)
    except Exception as e:
        print(f"  [warn] không vẽ được confusion matrix: {e}")

    montage = build_montage(EVID_DIR / "montage_verified_tracks.png")

    # summary.md — trích thẳng vào báo cáo
    lo, hi = ci
    lines = [
        "# Bằng chứng hand-verify Goal Classifier (accuracy thật)",
        "",
        f"*Sinh tự động {metrics['generated']} bởi `verify_goal_labels.py`.*",
        "",
        "## Phương pháp",
        "- Tái tạo ĐÚNG test set held-out (1.334 track, seed=42) — tách hoàn toàn khỏi train.",
        f"- Lấy mẫu phân tầng ngẫu nhiên, **{n} track** được người kiểm xác nhận hướng đi thật",
        "  bằng cách xem quỹ đạo vẽ đè lên khung hình CCTV thật.",
        "- So dự đoán model với nhãn NGƯỜI (không phải nhãn auto-label nhiễu).",
        "",
        "## Kết quả",
        "",
        "| Chỉ số | Giá trị |",
        "|---|---|",
        f"| **Accuracy thật (model vs người)** | **{honest:.3f}**  (95% CI {lo:.3f}–{hi:.3f}) |",
        f"| Nhãn auto-label đúng (auto vs người) | {auto_vs_human:.3f} (độ nhiễu {(1-auto_vs_human)*100:.1f}%) |",
        f"| Model vs auto (sanity) | {model_vs_auto:.3f} |",
        "",
        "### Recall theo lớp (vs nhãn người)",
        "",
        "| Hướng | n | Recall | 95% CI |",
        "|---|---|---|---|",
    ]
    for lbl in LABEL_NAMES:
        c = per_class.get(lbl, {})
        if c.get("n"):
            lines.append(f"| {lbl} | {c['n']} | {c['model_recall']:.3f} | "
                         f"{c['wilson95'][0]:.3f}–{c['wilson95'][1]:.3f} |")
        else:
            lines.append(f"| {lbl} | 0 | — | — |")
    lines += [
        "",
        "## Tệp kèm theo (bằng chứng)",
        "- `metrics.json` — toàn bộ số liệu máy đọc được.",
        "- `confusion_matrix_human.png` — ma trận nhầm lẫn model vs nhãn người.",
        "- `montage_verified_tracks.png` — ảnh mẫu các track đã kiểm (OK=đúng, MISS=sai).",
        "- `overlays/` — ảnh từng track đã kiểm (dấu vết kiểm chứng đầy đủ).",
        f"- `../../custom_tracking_system/data/{OUT_CSV.name}` — nhãn người từng track (tái lập được).",
    ]
    (EVID_DIR / "summary.md").write_text("\n".join(lines), encoding="utf-8")

    print(f"\n  ĐÃ LƯU BẰNG CHỨNG → {EVID_DIR}")
    print("    - metrics.json, summary.md, confusion_matrix_human.png"
          + (", montage_verified_tracks.png" if montage else "")
          + f", overlays/ ({len(list(OVERLAY_DIR.glob('*.jpg'))) if OVERLAY_DIR.exists() else 0} ảnh)")


# ── Main ────────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n", type=int, default=300, help="Số track lấy mẫu để verify")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--horizon", type=float, default=1.5)
    ap.add_argument("--junction-frac", type=float, default=0.65)
    ap.add_argument("--out", default=str(OUT_CSV))
    ap.add_argument("--report-only", action="store_true",
                    help="Chỉ in báo cáo từ file đã verify, không mở GUI")
    args = ap.parse_args()
    out_path = Path(args.out)

    print("[1/3] Tái tạo test split (giống train script)...")
    test_keys, X_all, meta_all, y_all, _ = reproduce_test_split(
        args.horizon, args.junction_frac)
    key_auto = {}
    for m, y in zip(meta_all, y_all):
        key_auto.setdefault((m["video_id"], m["track_id"]), y)
    # meta row đầy đủ (heading, centroid...) để vẽ overlay
    labels_df = pd.read_csv(LABELS_CSV)
    labels_df["key"] = list(zip(labels_df["video_id"], labels_df["track_id"]))
    meta_by_key = {k: r for k, r in zip(labels_df["key"],
                                        labels_df.to_dict("records"))}
    print(f"      test set = {len(test_keys)} track")

    print("[2/3] Load model + dự đoán...")
    key_pred = build_predictions(X_all, meta_all)

    # Resume
    if out_path.exists():
        done_df = pd.read_csv(out_path)
        done_df["key"] = list(zip(done_df["video_id"], done_df["track_id"]))
        done_keys = set(done_df["key"])
    else:
        done_df = pd.DataFrame(columns=["video_id", "track_id", "auto_label", "human_label"])
        done_keys = set()

    if args.report_only:
        report(done_df, key_auto, key_pred)
        return

    sample = stratified_sample(test_keys, key_auto, args.n, args.seed)
    todo = [k for k in sample if k not in done_keys]
    print(f"[3/3] Mẫu {len(sample)} track — đã verify {len(done_keys & set(sample))}, "
          f"còn {len(todo)}.\n      Mở cửa sổ OpenCV...")
    if not todo:
        print("      Hết track cần verify. In báo cáo:")
        report(done_df, key_auto, key_pred)
        return

    import cv2
    OVERLAY_DIR.mkdir(parents=True, exist_ok=True)
    records = done_df.to_dict("records")
    i = 0
    while 0 <= i < len(todo):
        key = todo[i]
        vid, tid = key
        auto = key_auto[key]
        pred = key_pred.get(key)
        mrow = meta_by_key.get(key)
        if mrow is None:
            i += 1; continue
        img = render_overlay(vid, tid, auto, mrow, i, len(todo),
                             len([r for r in records if r.get("human_label") in LABEL_NAMES]),
                             model_pred=pred)
        # Chờ quyết định; phím P = tua xem xe chạy rồi quay lại khung quyết định
        while True:
            cv2.imshow("verify goal labels", img)
            k = cv2.waitKey(0) & 0xFF
            if k in (ord("p"), ord("f")):
                play_track("verify goal labels", vid, tid)
                continue
            break

        if k in (ord("q"), 27):
            break
        if k == ord("b"):
            i = max(0, i - 1)
            if records and records[-1]["video_id"] == todo[i][0] \
               and records[-1]["track_id"] == todo[i][1]:
                records.pop()
            continue
        if k in (ord("a"), 32):
            human = auto
        elif k in KEY_TO_LABEL:
            human = KEY_TO_LABEL[k]
        elif k == ord("x"):
            human = "unsure"
        else:
            continue  # phím lạ → không chuyển

        records = [r for r in records if not (r["video_id"] == vid and r["track_id"] == tid)]
        records.append({"video_id": vid, "track_id": tid,
                        "auto_label": auto, "human_label": human})
        pd.DataFrame(records).to_csv(out_path, index=False)  # lưu tăng dần
        # Lưu bằng chứng trực quan: mỗi track đã duyệt → 1 ảnh (tên = người_model_auto)
        if human in LABEL_NAMES:
            ok_flag = "OK" if human == pred else "MISS"
            safe_vid = "".join(c if (c.isascii() and c.isalnum()) else "_" for c in str(vid))[:40]
            fn = f"{ok_flag}__h-{human}__m-{pred}__{safe_vid}_{tid}.jpg"
            cv2.imwrite(str(OVERLAY_DIR / fn), img, [cv2.IMWRITE_JPEG_QUALITY, 85])
        i += 1

    cv2.destroyAllWindows()
    final = pd.DataFrame(records)
    final.to_csv(out_path, index=False)
    print(f"\nĐã lưu {len(final)} nhãn → {out_path}")
    report(final, key_auto, key_pred)


if __name__ == "__main__":
    main()
