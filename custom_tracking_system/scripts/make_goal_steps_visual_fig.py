# -*- coding: utf-8 -*-
"""Goal Classifier — minh hoạ TRỰC QUAN 6 bước (mỗi bước một hình nhỏ). Chữ Việt.
   Sinh docs/assets/slide_goal_steps_visual.png
"""
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyBboxPatch, Circle, Polygon, FancyArrowPatch

OUT = Path(__file__).resolve().parents[2] / "docs" / "assets" / "slide_goal_steps_visual.png"
G, R, B, P, Y = "#2ca02c", "#d62728", "#1f77b4", "#9467bd", "#ffcc00"

fig, axes = plt.subplots(2, 3, figsize=(16, 8.6))
titles = [
    "1. Lấy quỹ đạo từ tracking",
    "2. Cắt cửa sổ quan sát",
    "3. Trích 33 đặc trưng",
    "4. Chuẩn hóa đặc trưng",
    "5. Gradient Boosting",
    "6. Ra quyết định (argmax)",
]
caps = [
    "chuỗi vị trí, vận tốc, hướng (ByteTrack)",
    "đoạn ~1,5 s trước nút giao (≥5 khung)",
    "đổi hướng · tốc độ · gia tốc · loại xe · tỉ lệ",
    "đưa về cùng thang đo (scaler)",
    "nhiều cây nhỏ, mỗi cây sửa lỗi cây trước",
    "chọn xác suất cao nhất → hướng đi",
]

for ax, t, c in zip(axes.flat, titles, caps):
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.set_xticks([]); ax.set_yticks([])
    ax.set_title(t, fontsize=13.5, fontweight="bold", pad=8)
    ax.text(0.5, 0.06, c, ha="center", va="center", fontsize=9.6, color="#444")
    for s in ax.spines.values():
        s.set_edgecolor("#cccccc")

a1, a2, a3, a4, a5, a6 = axes.flat

# --- 1. tracking: bbox + trail ---
xs = [0.18, 0.32, 0.46, 0.58]; ys = [0.42, 0.48, 0.56, 0.62]
a1.plot(xs, ys, "--", color=G, lw=2, zorder=1)
for x, y in zip(xs, ys):
    a1.plot(x, y, "o", color=G, ms=6, zorder=2)
a1.add_patch(Rectangle((0.55, 0.55), 0.17, 0.17, fill=False, ec=R, lw=2.5, zorder=3))
a1.annotate("", (0.75, 0.72), (0.6, 0.63), arrowprops=dict(arrowstyle="-|>", color=G, lw=2))
a1.text(0.2, 0.72, "khung t-3…t", fontsize=9, color=G)

# --- 2. window ---
cx = np.linspace(0.15, 0.7, 22); cy = 0.35 + 0.45*(cx-0.15) + 0.25*np.sin((cx-0.15)*3)
a2.plot(cx, cy, "-", color=G, lw=2.5)
a2.plot(cx[-8:], cy[-8:], "-", color=Y, lw=6, alpha=0.9, zorder=1)   # window
a2.plot(cx[-8:], cy[-8:], "-", color=G, lw=2.5, zorder=2)
a2.axvline(0.7, ymin=0.15, ymax=0.85, color="#888", ls="--", lw=1.5)
a2.text(0.72, 0.8, "nút giao", fontsize=9, color="#666", rotation=90, va="top")
a2.text(0.42, 0.28, "cửa sổ ~1,5 s", fontsize=9.5, color="#b58900", fontweight="bold")

# --- 3. feature vector (33 ô) ---
a3.annotate("", (0.36, 0.55), (0.24, 0.55), arrowprops=dict(arrowstyle="-|>", color="#333", lw=2))
a3.plot([0.12, 0.2], [0.5, 0.6], "-", color=G, lw=2)  # tiny trajectory
rng = np.linspace(0.2, 0.95, 33)
for i, v in enumerate(rng):
    r, cch = divmod(i, 11)
    a3.add_patch(Rectangle((0.4 + cch*0.05, 0.66 - r*0.11), 0.045, 0.09,
                 fc=plt.cm.Blues(0.3 + 0.6*((i*37 % 11)/11)), ec="white", lw=0.6))
a3.text(0.675, 0.24, "vector 33 chiều", ha="center", fontsize=9.5, color=B, fontweight="bold")

# --- 4. normalize: before/after bars ---
raw = [0.75, 0.28, 0.62, 0.4]; nrm = [0.55, 0.45, 0.58, 0.5]
for i, h in enumerate(raw):
    a4.add_patch(Rectangle((0.16 + i*0.05, 0.35), 0.04, h*0.4, fc="#b0b0b0"))
a4.annotate("", (0.56, 0.5), (0.44, 0.5), arrowprops=dict(arrowstyle="-|>", color=P, lw=2.2))
a4.text(0.5, 0.6, "scaler", ha="center", fontsize=9.5, color=P, fontweight="bold")
for i, h in enumerate(nrm):
    a4.add_patch(Rectangle((0.62 + i*0.05, 0.42), 0.04, h*0.3, fc=B))
a4.text(0.28, 0.28, "chênh lệch lớn", ha="center", fontsize=8.5, color="#777")
a4.text(0.7, 0.28, "cùng thang đo", ha="center", fontsize=8.5, color=B)

# --- 5. gradient boosting trees ---
def tree(ax, cx, cy, s=1.0):
    ax.add_patch(Circle((cx, cy + 0.09*s), 0.03*s, fc=P, ec="none"))
    for dx in (-0.07*s, 0.07*s):
        ax.plot([cx, cx+dx], [cy+0.09*s, cy-0.06*s], color=P, lw=1.5)
        ax.add_patch(Circle((cx+dx, cy-0.07*s), 0.026*s, fc="#c9b6e0", ec="none"))
for i, tx in enumerate([0.24, 0.5, 0.76]):
    tree(a5, tx, 0.55)
    a5.text(tx, 0.34, f"$f_{i+1}$", ha="center", fontsize=11, color=P, fontweight="bold")
    if i < 2:
        a5.text(tx+0.13, 0.55, "+", ha="center", va="center", fontsize=16, color="#666")
a5.annotate("", (0.5, 0.72), (0.24, 0.72), arrowprops=dict(arrowstyle="->", color="#999", lw=1.3))
a5.annotate("", (0.76, 0.72), (0.5, 0.72), arrowprops=dict(arrowstyle="->", color="#999", lw=1.3))
a5.text(0.5, 0.78, "học phần lỗi còn lại", ha="center", fontsize=8.3, color="#888", style="italic")

# --- 6. argmax bars ---
cls = ["thẳng", "trái", "phải", "quay đầu"]; pr = [0.14, 0.70, 0.11, 0.05]
yb = [0.72, 0.60, 0.48, 0.36]
for c_, p_, y_ in zip(cls, pr, yb):
    win = (p_ == max(pr))
    a6.add_patch(Rectangle((0.34, y_), 0.5*p_, 0.08, fc=G if win else "#c0c0c0"))
    a6.text(0.32, y_+0.04, c_, ha="right", va="center", fontsize=9.5,
            fontweight="bold" if win else "normal")
    a6.text(0.36+0.5*p_, y_+0.04, f"{p_:.2f}", ha="left", va="center", fontsize=8.8,
            color=G if win else "#777")
a6.text(0.5, 0.22, "→ RẼ TRÁI", ha="center", fontsize=13, color=G, fontweight="bold")

fig.suptitle("Goal Classifier — minh hoạ 6 bước hoạt động", fontsize=17, fontweight="bold", y=0.99)
fig.tight_layout(rect=[0, 0, 1, 0.95])
OUT.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(OUT, dpi=170, bbox_inches="tight")
print(f"[OK] saved: {OUT}")
