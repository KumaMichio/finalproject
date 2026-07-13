"""
confirm_goal_classifier.py — Xác nhận số liệu Goal Classifier tái lập được + thêm rigor
========================================================================================
Mục đích: CHỨNG MINH model đã lưu (weights/goal_classifier_real.pkl) thật sự cho đúng
các con số trong docs/assets/goal_classifier_real/metrics.json, và bổ sung:
  - Khoảng tin cậy Wilson 95% cho accuracy tổng thể + từng lớp (u_turn 36 mẫu → CI rộng).
  - Bootstrap 95% CI cho balanced accuracy (resample theo track).
  - Reliability diagram (calibration curve) minh hoạ ECE.

KHÔNG train lại — chỉ load model đã đóng băng + tái tạo ĐÚNG test split (seed=42).
Đây là bằng chứng "số liệu tái lập được"; accuracy THẬT (model vs người) vẫn cần
verify_goal_labels.py (nhãn người).

Usage:
  custom_tracking_system/venv_tracking/Scripts/python \
      custom_tracking_system/scripts/eval/confirm_goal_classifier.py
"""
import json
import sys
from pathlib import Path

import numpy as np

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

HERE = Path(__file__).resolve()
ROOT = HERE.parents[2]                       # custom_tracking_system/
WEIGHTS = ROOT / "weights"
ASSET = ROOT.parent / "docs" / "assets" / "goal_classifier_real"
OUT_JSON = ASSET / "confirm_eval.json"
OUT_PNG = ASSET / "reliability_diagram.png"

# Tái dùng logic tái tạo test split (single source of truth)
sys.path.insert(0, str(HERE.parent))                       # scripts/eval
from verify_goal_labels import reproduce_test_split        # noqa: E402

LABELS = ["left", "right", "straight", "u_turn"]           # = le.classes_ (alphabetical)


def wilson(k, n, z=1.96):
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    m = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return ((c - m) / d, (c + m) / d)


def ece_score(y_true, proba, n_bins=10):
    conf = proba.max(axis=1)
    pred = proba.argmax(axis=1)
    correct = (pred == y_true).astype(float)
    bins = np.linspace(0, 1, n_bins + 1)
    e = 0.0
    rows = []
    for i in range(n_bins):
        m = (conf > bins[i]) & (conf <= bins[i + 1])
        if m.sum() == 0:
            rows.append((0.5 * (bins[i] + bins[i + 1]), None, None, 0))
            continue
        acc = correct[m].mean()
        cf = conf[m].mean()
        e += (m.sum() / len(conf)) * abs(acc - cf)
        rows.append((0.5 * (bins[i] + bins[i + 1]), acc, cf, int(m.sum())))
    return e, rows


def main():
    import pickle
    print("[1/4] Tái tạo test split (seed=42, horizon=1.5, jf=0.65)...")
    test_keys, X_all, meta_all, y_all, _ = reproduce_test_split(1.5, 0.65)

    with open(WEIGHTS / "goal_classifier_real.pkl", "rb") as f:
        clf = pickle.load(f)
    with open(WEIGHTS / "goal_scaler_real.pkl", "rb") as f:
        scaler = pickle.load(f)
    with open(WEIGHTS / "goal_label_encoder_real.pkl", "rb") as f:
        le = pickle.load(f)

    # Lọc về đúng các track thuộc test set (1 hàng/track ở primary horizon)
    keys = [(m["video_id"], m["track_id"]) for m in meta_all]
    idx = [i for i, k in enumerate(keys) if k in test_keys]
    # bỏ trùng track (giữ lần đầu)
    seen, keep = set(), []
    for i in idx:
        if keys[i] not in seen:
            seen.add(keys[i]); keep.append(i)
    Xte = np.asarray(X_all)[keep]
    yte_names = np.asarray(y_all)[keep]
    yte = le.transform(yte_names)

    print(f"      n_test = {len(keep)} track")
    print("[2/4] Dự đoán bằng model đã lưu...")
    Xs = scaler.transform(Xte)
    proba = clf.predict_proba(Xs)
    pred = proba.argmax(axis=1)

    acc = (pred == yte).mean()
    # balanced acc = mean per-class recall
    recalls = {}
    for ci, lbl in enumerate(le.classes_):
        m = yte == ci
        recalls[lbl] = float((pred[m] == ci).mean()) if m.sum() else 0.0
    bal = float(np.mean(list(recalls.values())))
    from sklearn.metrics import f1_score
    f1m = f1_score(yte, pred, average="macro")
    ece, rel_rows = ece_score(yte, proba)

    # Wilson CI overall + per class
    lo, hi = wilson(int((pred == yte).sum()), len(yte))
    per_class = {}
    for ci, lbl in enumerate(le.classes_):
        m = yte == ci
        n = int(m.sum()); k = int((pred[m] == ci).sum())
        clo, chi = wilson(k, n)
        per_class[lbl] = {"n": n, "recall": (k / n if n else 0.0),
                          "wilson95": [round(clo, 3), round(chi, 3)]}

    # Bootstrap CI cho balanced accuracy (resample track)
    rng = np.random.RandomState(42)
    boots = []
    for _ in range(2000):
        s = rng.randint(0, len(yte), len(yte))
        rr = []
        for ci in range(len(le.classes_)):
            m = yte[s] == ci
            rr.append((pred[s][m] == ci).mean() if m.sum() else np.nan)
        boots.append(np.nanmean(rr))
    b_lo, b_hi = np.percentile(boots, [2.5, 97.5])

    # So khớp với metrics.json đã lưu
    ref = json.loads((ASSET / "metrics.json").read_text(encoding="utf-8"))["test_set"]
    match = (abs(acc - ref["accuracy"]) < 0.005 and abs(bal - ref["balanced_accuracy"]) < 0.01)

    print("[3/4] Vẽ reliability diagram...")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    xs = [r[2] for r in rel_rows if r[1] is not None]
    ys = [r[1] for r in rel_rows if r[1] is not None]
    ns = [r[3] for r in rel_rows if r[1] is not None]
    fig, ax = plt.subplots(figsize=(5.2, 5))
    ax.plot([0, 1], [0, 1], "--", color="gray", label="Hiệu chỉnh hoàn hảo")
    ax.plot(xs, ys, "o-", color="#2b6cb0", label="Model")
    for x, y, n in zip(xs, ys, ns):
        ax.annotate(str(n), (x, y), fontsize=7, xytext=(2, 4), textcoords="offset points")
    ax.set(xlabel="Độ tin cậy dự đoán (confidence)", ylabel="Độ chính xác thực tế (accuracy)",
           title=f"Reliability diagram — Goal Classifier (ECE={ece:.3f})", xlim=(0, 1), ylim=(0, 1))
    ax.legend(loc="upper left"); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(OUT_PNG, dpi=130); plt.close(fig)

    result = {
        "note": "Xác nhận model đã lưu tái lập số liệu test + CI. Đo vs nhãn AUTO-LABEL "
                "(chưa phải accuracy thật vs người — cần verify_goal_labels.py).",
        "n_test": len(yte),
        "reproduced": {"accuracy": round(float(acc), 4), "balanced_accuracy": round(bal, 4),
                       "f1_macro": round(float(f1m), 4), "ece": round(float(ece), 4)},
        "reference_metrics_json": {"accuracy": ref["accuracy"],
                                   "balanced_accuracy": ref["balanced_accuracy"],
                                   "f1_macro": ref["f1_macro"], "ece": ref["ece"]},
        "reproducible_match": bool(match),
        "accuracy_wilson95": [round(lo, 4), round(hi, 4)],
        "balanced_accuracy_bootstrap95": [round(float(b_lo), 4), round(float(b_hi), 4)],
        "per_class_recall_ci": per_class,
    }
    OUT_JSON.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")

    print("[4/4] KẾT QUẢ")
    print("=" * 62)
    print(f"  n_test track           : {len(yte)}")
    print(f"  accuracy (tái lập)      : {acc:.4f}   [Wilson95 {lo:.3f}–{hi:.3f}]")
    print(f"  balanced accuracy       : {bal:.4f}   [boot95 {b_lo:.3f}–{b_hi:.3f}]")
    print(f"  f1_macro / ECE          : {f1m:.4f} / {ece:.4f}")
    print(f"  KHỚP metrics.json?      : {'CÓ' if match else 'KHÔNG'} "
          f"(ref acc={ref['accuracy']}, bal={ref['balanced_accuracy']})")
    print("  Per-class recall [Wilson95]:")
    for lbl in le.classes_:
        c = per_class[lbl]
        print(f"    {lbl:<9} n={c['n']:<4} recall={c['recall']:.3f}  CI {c['wilson95'][0]:.3f}–{c['wilson95'][1]:.3f}")
    print("=" * 62)
    print(f"  → {OUT_JSON}")
    print(f"  → {OUT_PNG}")


if __name__ == "__main__":
    main()
