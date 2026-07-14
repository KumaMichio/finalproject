# -*- coding: utf-8 -*-
"""Ảnh GRADIENT BOOSTING — chỉ mô hình cây + công thức + cách ra dự đoán (KHÔNG chữ giải thích).
   Sinh docs/assets/slide_gradient_boosting.png
"""
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle, Rectangle

OUT = Path(__file__).resolve().parents[2] / "docs" / "assets" / "slide_gradient_boosting.png"
P, G, B = "#9467bd", "#2ca02c", "#1f77b4"

fig = plt.figure(figsize=(15.5, 8.5))
ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, 100); ax.set_ylim(0, 100); ax.axis("off")

ax.text(50, 96, "Gradient Boosting", ha="center", fontsize=19, fontweight="bold")
ax.text(50, 89, r"$F_m(x) = F_{m-1}(x) + \nu \cdot h_m(x)$", ha="center", fontsize=18, color=P)

# ============ BĂNG TRÊN: chuỗi cây tuần tự ============
def small_tree(cx, cy, s=1.0):
    ax.add_patch(Circle((cx, cy + 2.0*s), 0.9*s, fc=P, ec="none"))
    for dx in (-2.0*s, 2.0*s):
        ax.plot([cx, cx+dx], [cy+2.0*s, cy-1.5*s], color=P, lw=1.6)
        ax.add_patch(Circle((cx+dx, cy-1.8*s), 0.8*s, fc="#c9b6e0", ec="none"))

ax.text(7, 74, "$F_0$", ha="center", va="center", fontsize=14, color="#555")
ax.add_patch(FancyArrowPatch((11, 74), (16, 74), arrowstyle="-|>", mutation_scale=18, lw=1.8, color="#888"))
xt = [21, 43, 65]
for i, x in enumerate(xt):
    small_tree(x, 74)
    ax.text(x, 67, f"$h_{i+1}$", ha="center", fontsize=13, color=P, fontweight="bold")
    if i < 2:
        ax.add_patch(FancyArrowPatch((x+4, 74), (xt[i+1]-4, 74), arrowstyle="-|>",
                     mutation_scale=18, lw=1.8, color="#888"))
ax.add_patch(FancyArrowPatch((69, 74), (74, 74), arrowstyle="-|>", mutation_scale=18, lw=1.8, color="#888"))
ax.text(79, 74, "+ …", ha="center", va="center", fontsize=16, color="#666")

ax.plot([4, 96], [57, 57], color="#ddd", lw=1)

# ============ DƯỚI-TRÁI: mô hình một cây quyết định ============
def node(x, y, w, h, text, fc):
    ax.add_patch(FancyBboxPatch((x-w/2, y-h/2), w, h, boxstyle="round,pad=0.25", fc=fc, ec="none"))
    ax.text(x, y, text, ha="center", va="center", fontsize=9.5, color="white", fontweight="bold")

def edge(x0, y0, x1, y1, lab):
    ax.plot([x0, x1], [y0, y1], color="#999", lw=1.4, zorder=1)
    ax.text((x0+x1)/2 - 1, (y0+y1)/2 + 0.6, lab, fontsize=8.5, color="#555", ha="center")

node(26, 48, 24, 5.5, "|tổng góc quay| > 40° ?", P)
edge(20, 45.2, 12, 37, "Không"); edge(32, 45.2, 40, 37, "Có")
node(12, 34, 16, 5.5, "→ ĐI THẲNG", G)
node(42, 34, 19, 5.5, "tổng góc có dấu < 0 ?", P)
edge(36, 31.2, 32, 22, "Có"); edge(48, 31.2, 52, 22, "Không")
node(32, 19, 14, 5.5, "→ RẼ TRÁI", G)
node(52, 19, 14, 5.5, "→ RẼ PHẢI", G)

# ============ DƯỚI-PHẢI: Σ cây -> softmax -> xác suất -> argmax ============
ax.text(78, 53, r"$\Sigma$ cây  →  softmax", ha="center", fontsize=12.5, fontweight="bold", color=B)
ax.text(78, 47, r"$p_i = \dfrac{e^{z_i}}{\sum_j e^{z_j}}$", ha="center", fontsize=14, color=B)
cls = ["thẳng", "trái", "phải", "quay đầu"]; pr = [0.14, 0.70, 0.11, 0.05]
yb = [39, 34, 29, 24]; x0b, maxw = 67, 21
for c, p, y in zip(cls, pr, yb):
    win = (p == max(pr))
    ax.add_patch(Rectangle((x0b, y), maxw*p, 3.6, fc=G if win else "#b8b8b8", ec="none"))
    ax.text(x0b-1, y+1.8, c, ha="right", va="center", fontsize=9.3, fontweight="bold" if win else "normal")
    ax.text(x0b+maxw*p+1, y+1.8, f"{p:.2f}", ha="left", va="center", fontsize=8.8,
            color=G if win else "#777", fontweight="bold" if win else "normal")
ax.text(78, 17.5, r"$\hat{y} = \arg\max_i \; p_i$", ha="center", fontsize=14, color=G)
ax.text(78, 12, "→ argmax = RẼ TRÁI", ha="center", fontsize=13, color=G, fontweight="bold")

OUT.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(OUT, dpi=170, bbox_inches="tight")
print(f"[OK] saved: {OUT}")
