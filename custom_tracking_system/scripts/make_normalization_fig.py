# -*- coding: utf-8 -*-
"""Ảnh riêng: CHUẨN HOÁ ĐẶC TRƯNG (standardization) z=(x-μ)/σ. Chữ Việt.
   Sinh docs/assets/slide_normalization.png
"""
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

OUT = Path(__file__).resolve().parents[2] / "docs" / "assets" / "slide_normalization.png"
B, P, G = "#1f77b4", "#9467bd", "#2ca02c"

fig = plt.figure(figsize=(15, 7.4))
ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, 100); ax.set_ylim(0, 100); ax.axis("off")

ax.text(50, 95, "Chuẩn hoá đặc trưng (Standardization)", ha="center", fontsize=18, fontweight="bold")
ax.text(50, 87, r"$z = \dfrac{x - \mu}{\sigma}$", ha="center", fontsize=26, color=P)
ax.text(50, 78.5, "μ = trung bình, σ = độ lệch chuẩn của đặc trưng đó (ước lượng từ tập train)",
        ha="center", fontsize=11, color="#444")

feats = ["hướng (30)", "|góc quay| (90)", "tốc độ (200)", "bbox (5000)"]
raw = [30, 90, 200, 5000]
std = [-0.6, 1.3, 0.7, -1.1]

# ---------- TRƯỚC ----------
ax.text(24, 69, "TRƯỚC — thang đo chênh lệch", ha="center", fontsize=12.5, fontweight="bold", color="#b00")
xL0, xL1 = 6, 44
ax.plot([xL0, xL1], [55, 55], color="#333", lw=1.6)
ax.text(xL0, 51, "0", ha="center", fontsize=9); ax.text(xL1, 51, "5000", ha="center", fontsize=9)
def mapL(v): return xL0 + (v/5000)*(xL1-xL0)
labely = [46, 41, 36, 62]
for f, v, ly in zip(feats, raw, labely):
    x = mapL(v)
    ax.plot(x, 55, "o", color=B, ms=9, zorder=3)
    ax.annotate(f, (x, 55), (x if v > 1000 else 9, ly),
                fontsize=9, color="#333", ha="center",
                arrowprops=dict(arrowstyle="-", color="#bbb", lw=0.8))
ax.text(24, 30, "3 đặc trưng nhỏ dồn sát 0,\nbbox lấn át → model thiên lệch", ha="center",
        fontsize=9.5, color="#b00")

# ---------- mũi tên ----------
ax.add_patch(FancyArrowPatch((46, 50), (54, 50), arrowstyle="-|>", mutation_scale=26, lw=3, color=P))
ax.text(50, 57, "Scaler", ha="center", fontsize=11, color=P, fontweight="bold")
ax.text(50, 44, "z=(x−μ)/σ", ha="center", fontsize=9.5, color=P)

# ---------- SAU ----------
ax.text(76, 69, "SAU — cùng thang đo", ha="center", fontsize=12.5, fontweight="bold", color=G)
xR0, xR1 = 58, 94; xc = (xR0+xR1)/2
ax.plot([xR0, xR1], [55, 55], color="#333", lw=1.6)
for tick, lab in [(xR0, "-3"), (xc, "0"), (xR1, "+3")]:
    ax.plot([tick, tick], [54, 56], color="#333", lw=1.2)
    ax.text(tick, 51, lab, ha="center", fontsize=9)
def mapR(z): return xc + (z/3)*(xR1-xc)
ly2 = [46, 62, 41, 36]
for f, z, ly in zip(feats, std, ly2):
    x = mapR(z)
    ax.plot(x, 55, "o", color=G, ms=9, zorder=3)
    ax.annotate(f.split(" (")[0], (x, 55), (x, ly), fontsize=9, color="#333", ha="center",
                arrowprops=dict(arrowstyle="-", color="#bbb", lw=0.8))
ax.text(76, 30, "mọi đặc trưng ~[-2, 2],\ntrung bình 0, độ lệch chuẩn 1", ha="center",
        fontsize=9.5, color=G)

# ---------- ví dụ + ý nghĩa ----------
ax.add_patch(FancyBboxPatch((6, 8), 88, 14, boxstyle="round,pad=0.4", fc="#f4f7fb", ec="#ccc"))
ax.text(10, 17.5, "Ví dụ:", fontsize=11, fontweight="bold", color=P)
ax.text(22, 17.5, r"tốc độ $x=230$,  $\mu=180$,  $\sigma=40$   →   $z=\dfrac{230-180}{40}=+1{,}25$",
        fontsize=11.5, color="#333", va="center")
ax.text(72, 17.5, "(nhanh hơn TB 1,25 σ)", fontsize=9.5, color="#777", va="center")
ax.text(10, 11, "Ý nghĩa: không đặc trưng nào lấn át chỉ vì con số lớn → mô hình học công bằng giữa 33 đặc trưng.",
        fontsize=10, color="#333")

OUT.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(OUT, dpi=170, bbox_inches="tight")
print(f"[OK] saved: {OUT}")
