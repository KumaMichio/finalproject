# -*- coding: utf-8 -*-
"""Hình minh hoạ IoU (Intersection over Union):
   trái  = 2 box A, B chồng nhau, tô phần GIAO; công thức + ví dụ số.
   phải  = "phân số bằng hình": tử = phần Giao, mẫu = phần Hợp.
   Sơ đồ khái niệm, chữ Việt. Cùng phong cách với make_hungarian_fig.py.
"""
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

OUT = Path(__file__).resolve().parents[2] / "docs" / "assets" / "slide_iou.png"

C_A = "#1f77b4"    # box A - xanh dương
C_B = "#d62728"    # box B - đỏ
C_I = "#2ca02c"    # giao - xanh lá

# Toạ độ (x1, y1, x2, y2)
A = (1.0, 1.5, 4.2, 4.0)
B = (3.0, 0.8, 6.2, 3.3)
# Giao
ix1, iy1 = max(A[0], B[0]), max(A[1], B[1])
ix2, iy2 = min(A[2], B[2]), min(A[3], B[3])

fig = plt.figure(figsize=(15, 6.4))
axL = fig.add_axes([0.03, 0.08, 0.5, 0.82])
axR = fig.add_axes([0.58, 0.08, 0.40, 0.82])


def box(ax, r, ec, label, lx=0, ly=0, ha="left"):
    ax.add_patch(Rectangle((r[0], r[1]), r[2] - r[0], r[3] - r[1],
                            fill=False, ec=ec, lw=3))
    ax.text(r[0] + lx, r[1] + (r[3] - r[1]) + 0.12 + ly, label, color=ec,
            fontsize=16, fontweight="bold", ha=ha)


# -------- TRÁI: định nghĩa --------
axL.set_title("Hai bounding box chồng nhau", fontsize=15, pad=10)
# tô phần giao
axL.add_patch(Rectangle((ix1, iy1), ix2 - ix1, iy2 - iy1, fc=C_I, ec=C_I, alpha=0.45, lw=0))
box(axL, A, C_A, "A")
box(axL, B, C_B, "B", ha="left")
axL.text((ix1 + ix2) / 2, (iy1 + iy2) / 2, "Giao\n(A∩B)", ha="center", va="center",
         fontsize=12, color="#0a4d0a", fontweight="bold")
axL.set_xlim(0.3, 7.2); axL.set_ylim(0.2, 4.8); axL.set_aspect("equal"); axL.axis("off")

# -------- PHẢI: phân số bằng hình + công thức --------
axR.axis("off"); axR.set_xlim(0, 10); axR.set_ylim(0, 10)


def mini(ax, cx, cy, mode):
    """Vẽ cặp box thu nhỏ quanh (cx,cy); mode='inter' tô giao, 'union' tô cả hai."""
    s = 0.9
    a = (cx - 1.5*s, cy - 0.7*s, cx + 0.4*s, cy + 1.1*s)
    b = (cx - 0.4*s, cy - 1.1*s, cx + 1.5*s, cy + 0.7*s)
    jx1, jy1 = max(a[0], b[0]), max(a[1], b[1])
    jx2, jy2 = min(a[2], b[2]), min(a[3], b[3])
    if mode == "union":
        for r in (a, b):
            ax.add_patch(Rectangle((r[0], r[1]), r[2]-r[0], r[3]-r[1], fc=C_I, ec="none", alpha=0.45))
    else:
        ax.add_patch(Rectangle((jx1, jy1), jx2-jx1, jy2-jy1, fc=C_I, ec="none", alpha=0.7))
    ax.add_patch(Rectangle((a[0], a[1]), a[2]-a[0], a[3]-a[1], fill=False, ec=C_A, lw=2))
    ax.add_patch(Rectangle((b[0], b[1]), b[2]-b[0], b[3]-b[1], fill=False, ec=C_B, lw=2))


mini(axR, 2.4, 7.2, "inter")
mini(axR, 2.4, 3.0, "union")
axR.plot([0.3, 4.5], [5.1, 5.1], color="black", lw=2.5)   # gạch phân số
axR.text(2.4, 9.2, "Giao (A∩B)", ha="center", fontsize=13, color=C_I, fontweight="bold")
axR.text(2.4, 0.7, "Hợp (A∪B)", ha="center", fontsize=13, color=C_I, fontweight="bold")
axR.text(5.2, 5.1, "=", ha="center", va="center", fontsize=30)
axR.text(6.0, 5.1, "IoU", ha="left", va="center", fontsize=26, fontweight="bold", color=C_I)

fig.suptitle("IoU = Diện tích GIAO  /  Diện tích HỢP", fontsize=18, fontweight="bold", y=0.99)

# công thức + ví dụ dưới đáy
fig.text(0.03, -0.02,
         "Hợp = S(A) + S(B) − Giao        (trừ Giao để không cộng trùng)\n"
         "Ví dụ: A=(0,0,4,4) S=16 ; B=(2,2,6,6) S=16 ; Giao=2×2=4 → "
         "Hợp = 16+16−4 = 28 → IoU = 4/28 ≈ 0,14",
         fontsize=13, va="top")

OUT.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(OUT, dpi=170, bbox_inches="tight")
print(f"[OK] saved: {OUT}")
