# -*- coding: utf-8 -*-
"""Goal Classifier pipeline illustration (ENGLISH) — faithful to this project:
   Stage 1 = a SCHEMATIC intersection drawn in CCTV PERSPECTIVE (tilted view, NOT top-down),
   with a turning trajectory in the image plane; then 33 hand-crafted features ->
   GradientBoosting (calibrated) -> probability over {straight, left, right, u_turn}.
"""
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle, Polygon

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT.parent / "docs" / "assets" / "slide_goal_pipeline.png"

PRED  = "left"
PROBS = {"straight": 0.14, "left": 0.70, "right": 0.11, "u_turn": 0.05}

C_TRAJ = "#2ca02c"; C_FEAT = "#1f77b4"; C_MODEL = "#9467bd"; C_WIN = "#2ca02c"
ROAD = "#8a8a8a"; GRASS = "#cfe3c6"


def proj(X, Z):
    """Ground-plane point (X lateral, Z depth) -> CCTV-perspective screen coords."""
    return 5 + 6.0 * X / Z, 9 - 8.0 / Z


def road_poly(x_hw, z0, z1):
    pts = [proj(-x_hw, z0), proj(-x_hw, z1), proj(x_hw, z1), proj(x_hw, z0)]
    return Polygon(pts, closed=True, fc=ROAD, ec="none", zorder=1)


def cross_poly(z_hw, x0, x1):
    a, b = 2.4 - z_hw, 2.4 + z_hw
    pts = [proj(x0, a), proj(x0, b), proj(x1, b), proj(x1, a)]
    return Polygon(pts, closed=True, fc=ROAD, ec="none", zorder=1)


fig = plt.figure(figsize=(17, 6.6))
ax_img = fig.add_axes([0.015, 0.09, 0.295, 0.76])
ax = fig.add_axes([0.33, 0.0, 0.67, 1.0]); ax.set_xlim(0, 100); ax.set_ylim(0, 100); ax.axis("off")

# ============ STAGE 1: schematic intersection in CCTV perspective ============
ax_img.add_patch(Rectangle((0, 0), 10, 9.2, fc=GRASS, ec="none", zorder=0))   # ground
ax_img.add_patch(Rectangle((0, 9.0), 10, 1.2, fc="#bcd4e8", ec="none", zorder=0))  # sky above horizon
ax_img.add_patch(road_poly(0.55, 1.15, 7.0))     # depth road (vehicle travels)
ax_img.add_patch(cross_poly(0.42, -3.2, 3.2))    # cross road
# dashed lane centre line (converges to vanishing point)
zc = np.linspace(1.2, 6.8, 24)
lx, ly = zip(*[proj(0.0, z) for z in zc])
ax_img.plot(lx, ly, ls=(0, (5, 5)), color="white", lw=1.4, zorder=2)

# turning trajectory: approach along depth road, turn LEFT onto cross road
world = [(0.16, 6.0), (0.16, 5.0), (0.16, 4.0), (0.16, 3.2), (0.15, 2.6),
         (-0.05, 2.42), (-0.5, 2.4), (-1.1, 2.4), (-1.8, 2.4), (-2.5, 2.4)]
tx, ty = zip(*[proj(X, Z) for X, Z in world])
ax_img.plot(tx, ty, "-", color=C_TRAJ, lw=3, zorder=4)
ax_img.plot(tx, ty, "o", color=C_TRAJ, ms=5, zorder=5)
ax_img.plot(tx[0], ty[0], "o", mfc="white", mec=C_TRAJ, mew=2.5, ms=10, zorder=6)   # start
ax_img.annotate("", (tx[-1], ty[-1]), (tx[-2], ty[-2]),
                arrowprops=dict(arrowstyle="-|>", color=C_TRAJ, lw=3))
# observation window (~1.5 s before junction): last part of approach, yellow
ax_img.plot(tx[3:6], ty[3:6], "-", color="#ffd000", lw=4, zorder=4)

ax_img.set_xlim(0.2, 9.8); ax_img.set_ylim(0.3, 9.6)
ax_img.set_xticks([]); ax_img.set_yticks([])
for s in ax_img.spines.values():
    s.set_edgecolor("#999")
ax_img.set_title("1. Tracked trajectory — CCTV perspective (schematic)", fontsize=12.5, fontweight="bold")
ax_img.text(0.5, -0.07, "tilted view like a traffic camera — NOT top-down\n"
            "yellow = last ≈1.5 s window before junction",
            transform=ax_img.transAxes, ha="center", va="top", fontsize=9.2, color="#333")

# ============ STAGE 2: feature extraction ============
fb = FancyBboxPatch((5, 16), 36, 70, boxstyle="round,pad=0.6", fc=C_FEAT, ec="none", alpha=0.92)
ax.add_patch(fb)
ax.text(23, 80.5, "2. Feature extraction", ha="center", fontsize=13.5, fontweight="bold", color="white")
ax.text(23, 76, "33 hand-crafted features (pixel space)", ha="center", fontsize=9.8, color="white", style="italic")
groups = [
    ("Speed profile", "mean · std · drop · recent · min  (px & m/s)"),
    ("Heading change", "turn rate · total turn · sin/cos · monotonic"),
    ("Lateral accel.", "lat_acc mean · max"),
    ("Vehicle class", "car · motorcycle · bus · truck  (one-hot)"),
    ("Scale", "bbox size · px-per-metre"),
]
y = 69
for name, detail in groups:
    ax.text(8, y, f"• {name}", ha="left", fontsize=11, color="white", fontweight="bold")
    ax.text(10, y - 3.4, detail, ha="left", fontsize=8.9, color="#eaf2fb")
    y -= 10.6

# ============ STAGE 3: model ============
mb = FancyBboxPatch((47, 40), 17, 24, boxstyle="round,pad=0.5", fc=C_MODEL, ec="none", alpha=0.92)
ax.add_patch(mb)
ax.text(55.5, 58.5, "3. Gradient", ha="center", fontsize=12.5, fontweight="bold", color="white")
ax.text(55.5, 54.3, "Boosting", ha="center", fontsize=12.5, fontweight="bold", color="white")
ax.text(55.5, 48.5, "calibrated\nprobabilities", ha="center", fontsize=9.2, color="#f0e9f7")

# ============ STAGE 4: output ============
ax.text(84, 90, "4. Direction probability", ha="center", fontsize=13.5, fontweight="bold")
order = ["straight", "left", "right", "u_turn"]
ybar = [72, 62, 52, 42]; x0b, maxw = 74, 15
for c, yb in zip(order, ybar):
    p = PROBS[c]; win = (c == PRED)
    ax.add_patch(Rectangle((x0b, yb), maxw*p, 6, fc=C_WIN if win else "#b0b0b0", ec="none"))
    ax.text(x0b-0.8, yb+3, c, ha="right", va="center", fontsize=10.5,
            fontweight="bold" if win else "normal")
    ax.text(x0b+maxw*p+0.6, yb+3, f"{p:.2f}", ha="left", va="center", fontsize=10,
            color=C_WIN if win else "#666", fontweight="bold" if win else "normal")
ax.text(84, 31, f"→ Predicted:  {PRED.upper()}", ha="center", fontsize=13, color=C_WIN, fontweight="bold")
ax.text(84, 25.5, "(argmax of the 4 probabilities)", ha="center", fontsize=9, color="#555")

# ============ arrows ============
for xa, xb in [(0.5, 4.5), (41.5, 46.5), (64.5, 73.5)]:
    ax.add_patch(FancyArrowPatch((xa, 52), (xb, 52), arrowstyle="-|>",
                 mutation_scale=22, lw=2.4, color="#333"))

fig.suptitle("Goal Classifier — predict a vehicle's turning direction from its approach trajectory",
             fontsize=16, fontweight="bold", y=1.0)
OUT.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(OUT, dpi=170, bbox_inches="tight")
print(f"[OK] saved: {OUT}")
