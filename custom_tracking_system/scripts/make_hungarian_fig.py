# -*- coding: utf-8 -*-
"""Hình minh hoạ bước GHÉP CẶP Hungarian trong ByteTrack:
   trái  = Track (Kalman dự đoán, nét đứt) vs Detection (YOLO, nét liền), IoU chồng lấn.
   phải  = Ma trận chi phí Cost = 1 - IoU, tô đậm phép ghép tối ưu (tổng chi phí nhỏ nhất).
   Không cần ảnh thật — sơ đồ khái niệm, chữ Việt.
"""
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyArrowPatch

OUT = Path(__file__).resolve().parents[2] / "docs" / "assets" / "slide_hungarian.png"

C_TRACK = "#1f77b4"   # xanh dương - track (Kalman)
C_DET   = "#d62728"   # đỏ - detection (YOLO)
C_GOOD  = "#2ca02c"   # xanh lá - ghép được chọn

# IoU giữa track (hàng) và detection (cột)
IOU = np.array([
    [0.85, 0.10, 0.00],
    [0.05, 0.80, 0.15],
    [0.00, 0.20, 0.75],
])
COST = 1.0 - IOU
ASSIGN = [(0, 0), (1, 1), (2, 2)]   # nghiệm Hungarian: tổng chi phí nhỏ nhất

fig, (axL, axR) = plt.subplots(1, 2, figsize=(15, 6.2), gridspec_kw={"width_ratios": [1.15, 1]})

# ---------------- TRÁI: box track vs detection ----------------
axL.set_title("Track (Kalman dự đoán)  vs  Detection (YOLO)", fontsize=15, pad=12)
# (x, y, w, h) cho 3 track (nét đứt) và 3 detection (nét liền) — chồng lấn từng cặp
tracks = [(0.6, 3.2, 1.7, 1.2), (3.6, 2.4, 1.5, 1.3), (6.4, 3.4, 1.6, 1.1)]
dets   = [(0.9, 3.0, 1.7, 1.2), (3.9, 2.2, 1.5, 1.3), (6.7, 3.2, 1.6, 1.1)]
for i, (x, y, w, h) in enumerate(tracks):
    axL.add_patch(Rectangle((x, y), w, h, fill=False, ec=C_TRACK, lw=2.5, ls="--"))
    axL.text(x, y + h + 0.12, f"T{i+1}", color=C_TRACK, fontsize=14, fontweight="bold")
for i, (x, y, w, h) in enumerate(dets):
    axL.add_patch(Rectangle((x, y), w, h, fill=False, ec=C_DET, lw=2.5))
    axL.text(x + w, y - 0.28, f"D{i+1}", color=C_DET, fontsize=14, fontweight="bold", ha="right")
# chú thích IoU cho từng cặp
for i, (x, y, w, h) in enumerate(dets):
    axL.text(x + w / 2, y + h / 2, f"IoU≈{IOU[i, i]:.2f}", ha="center", va="center",
             fontsize=11, color="#333", fontweight="bold",
             bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.75))
axL.plot([], [], color=C_TRACK, lw=2.5, ls="--", label="Track — Kalman dự đoán")
axL.plot([], [], color=C_DET, lw=2.5, label="Detection — YOLO khung mới")
axL.legend(loc="lower center", ncol=2, fontsize=12, frameon=False, bbox_to_anchor=(0.5, -0.09))
axL.set_xlim(0, 8.6); axL.set_ylim(1.4, 5.2); axL.set_aspect("equal")
axL.axis("off")

# ---------------- PHẢI: ma trận chi phí ----------------
axR.set_title("Ma trận chi phí  Cost = 1 − IoU", fontsize=15, pad=12)
im = axR.imshow(COST, cmap="RdYlGn_r", vmin=0, vmax=1)
axR.set_xticks(range(3)); axR.set_yticks(range(3))
axR.set_xticklabels([f"D{j+1}" for j in range(3)], fontsize=13, color=C_DET, fontweight="bold")
axR.set_yticklabels([f"T{i+1}" for i in range(3)], fontsize=13, color=C_TRACK, fontweight="bold")
axR.set_xlabel("Detection (YOLO)", fontsize=12); axR.set_ylabel("Track (Kalman)", fontsize=12)
for i in range(3):
    for j in range(3):
        chosen = (i, j) in ASSIGN
        axR.text(j, i, f"{COST[i, j]:.2f}", ha="center", va="center", fontsize=13,
                 color="black", fontweight="bold" if chosen else "normal")
        if chosen:
            axR.add_patch(Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False, ec=C_GOOD, lw=4))
tot = sum(COST[i, j] for i, j in ASSIGN)
axR.text(1.0, 3.05,
         f"Hungarian → chọn tổ hợp tổng chi phí NHỎ NHẤT = {tot:.2f}\n"
         f"(ô xanh) → giữ ĐÚNG ID phương tiện",
         ha="center", va="top", fontsize=12, color=C_GOOD, fontweight="bold")
cbar = fig.colorbar(im, ax=axR, fraction=0.046, pad=0.04)
cbar.set_label("Chi phí (thấp = giống nhau)", fontsize=10)

fig.suptitle("Hungarian Algorithm — ghép Track ↔ Detection theo chi phí 1 − IoU",
             fontsize=17, fontweight="bold", y=1.0)
fig.tight_layout(rect=[0, 0, 1, 0.96])
OUT.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(OUT, dpi=170, bbox_inches="tight")
print(f"[OK] saved: {OUT}")
