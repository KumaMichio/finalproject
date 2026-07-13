# -*- coding: utf-8 -*-
"""Tính ĐẦY ĐỦ chỉ số Goal Classifier vs NHÃN NGƯỜI (hand-verify) — bỏ nhãn auto.
Nguồn nhãn: data/goal_labels_verified.csv (người chấm). Dự đoán: model đã lưu.
"""
import sys, pickle, json
import numpy as np, pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "eval"))
from verify_goal_labels import reproduce_test_split
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, f1_score,
                             classification_report, confusion_matrix)

W = ROOT / "weights"
VERIFIED = ROOT / "data" / "goal_labels_verified.csv"
OUT = ROOT.parent / "docs" / "assets" / "goal_classifier_verified" / "full_metrics_human.json"
LBL = ["straight", "left", "right", "u_turn"]


def wilson(k, n, z=1.96):
    if n == 0: return (0, 0)
    p = k / n; d = 1 + z * z / n; c = p + z * z / (2 * n)
    m = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return ((c - m) / d, (c + m) / d)


def ece(y, proba, classes, n_bins=10):
    conf = proba.max(1); pred = proba.argmax(1)
    yi = np.array([classes.index(v) for v in y])
    correct = (pred == yi).astype(float)
    b = np.linspace(0, 1, n_bins + 1); e = 0.0
    for i in range(n_bins):
        m = (conf > b[i]) & (conf <= b[i + 1])
        if m.sum(): e += m.mean() * abs(correct[m].mean() - conf[m].mean())
    return e


print("[1] Tái tạo features + dự đoán...")
_, X_all, meta_all, _, _ = reproduce_test_split(1.5, 0.65)
clf = pickle.load(open(W / "goal_classifier_real.pkl", "rb"))
sc = pickle.load(open(W / "goal_scaler_real.pkl", "rb"))
le = pickle.load(open(W / "goal_label_encoder_real.pkl", "rb"))
classes = list(le.classes_)                          # ['left','right','straight','u_turn']
proba_all = clf.predict_proba(sc.transform(X_all))
key_proba = {}
for m, pr in zip(meta_all, proba_all):
    key_proba.setdefault((m["video_id"], m["track_id"]), pr)

print("[2] Nạp nhãn NGƯỜI...")
v = pd.read_csv(VERIFIED)
v = v[v["human_label"].isin(LBL)].copy()
rows = []
for _, r in v.iterrows():
    k = (r["video_id"], r["track_id"])
    if k in key_proba:
        rows.append((r["human_label"], key_proba[k]))
y = [r[0] for r in rows]
proba = np.array([r[1] for r in rows])
pred = [classes[i] for i in proba.argmax(1)]
n = len(y)
print(f"    n = {n} track (nhãn người)")

acc = accuracy_score(y, pred)
bal = balanced_accuracy_score(y, pred)
f1m = f1_score(y, pred, labels=LBL, average="macro", zero_division=0)
e = ece(y, proba, classes)
lo, hi = wilson(int(sum(a == b for a, b in zip(y, pred))), n)
rep = classification_report(y, pred, labels=LBL, output_dict=True, zero_division=0)
cm = confusion_matrix(y, pred, labels=LBL)

out = {"n": n, "note": "TẤT CẢ đo vs nhãn NGƯỜI (hand-verify), không dùng nhãn auto.",
       "accuracy": round(acc, 4), "accuracy_wilson95": [round(lo, 4), round(hi, 4)],
       "balanced_accuracy": round(bal, 4), "f1_macro": round(f1m, 4), "ece": round(float(e), 4),
       "per_class": {c: {"support": int(rep[c]["support"]),
                         "precision": round(rep[c]["precision"], 4),
                         "recall": round(rep[c]["recall"], 4),
                         "f1": round(rep[c]["f1-score"], 4)} for c in LBL},
       "confusion_matrix_true_rows_pred_cols": cm.tolist(), "labels": LBL}
OUT.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")

print("\n" + "=" * 56)
print(f"GOAL CLASSIFIER vs NHÃN NGƯỜI  (n={n})")
print("=" * 56)
print(f"  Accuracy          : {acc*100:.1f}%   [95% CI {lo*100:.1f}–{hi*100:.1f}%]")
print(f"  Balanced accuracy : {bal*100:.1f}%")
print(f"  F1-macro          : {f1m:.3f}")
print(f"  ECE               : {e:.3f}")
print("  Per-class (support | precision | recall | f1):")
for c in LBL:
    d = rep[c]
    print(f"    {c:<9} n={int(d['support']):3d}  P={d['precision']*100:4.1f}  R={d['recall']*100:4.1f}  F1={d['f1-score']:.3f}")
print("  Confusion (hàng=thật, cột=dự đoán):", LBL)
for i, c in enumerate(LBL):
    print(f"    {c:<9}", cm[i].tolist())
print(f"\n  -> {OUT}")
