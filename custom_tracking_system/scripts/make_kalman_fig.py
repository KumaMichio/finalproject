# -*- coding: utf-8 -*-
"""Kalman Filter illustration (ENGLISH) for the tracking pipeline:
   left  = data flow: INPUT state -> PREDICT -> UPDATE (loop each frame).
   right = one vehicle's trajectory: estimate (green) = predicted (blue) fused with
           YOLO observation (red); during occlusion only Predict runs but the track survives.
   Concept diagram. Same style as make_hungarian_fig.py / make_iou_fig.py.
"""
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle

OUT = Path(__file__).resolve().parents[2] / "docs" / "assets" / "slide_kalman.png"

C_IN   = "#7f7f7f"   # input state - gray
C_PRED = "#1f77b4"   # predict - blue
C_OBS  = "#d62728"   # YOLO observation - red
C_EST  = "#2ca02c"   # corrected estimate - green

fig = plt.figure(figsize=(16, 6.6))
axL = fig.add_axes([0.02, 0.03, 0.40, 0.88])
axR = fig.add_axes([0.47, 0.09, 0.51, 0.80])

# -------- LEFT: data-flow Input -> Predict -> Update --------
axL.set_xlim(0, 10); axL.set_ylim(0, 10); axL.axis("off")
axL.set_title("What Kalman uses each frame", fontsize=15, pad=6)


def rbox(cy, h, color, title, formula, desc):
    w = 9.0
    axL.add_patch(FancyBboxPatch((5 - w/2, cy - h/2), w, h,
                  boxstyle="round,pad=0.12", fc=color, ec="none", alpha=0.92))
    axL.text(5, cy + h/2 - 0.42, title, ha="center", va="center",
             fontsize=14.5, fontweight="bold", color="white")
    if formula:
        axL.text(5, cy + 0.02, formula, ha="center", va="center", fontsize=13.5, color="white")
    axL.text(5, cy - h/2 + 0.42, desc, ha="center", va="center", fontsize=10.3, color="white")


# INPUT — state vector
rbox(8.35, 2.35, C_IN, "INPUT — state  $\\hat{x}_{t-1}$",
     "$[\\,c_x,\\ c_y,\\ w,\\ h,\\ \\ v_x,\\ v_y,\\ v_w,\\ v_h\\,]$",
     "box position (cx,cy) + size (w,h)  +  their velocities")
# PREDICT
rbox(4.95, 2.35, C_PRED, "1. PREDICT",
     "$\\hat{x}_t^{-} = F\\,\\hat{x}_{t-1}$",
     "extrapolate next position from previous\nstate (constant-velocity motion)")
# UPDATE
rbox(1.55, 2.35, C_EST, "2. UPDATE",
     "$\\hat{x}_t = \\hat{x}_t^{-} + K\\,(z_t - H\\hat{x}_t^{-})$",
     "fuse prediction with YOLO detection  $z_t$\nto correct the state")

# straight arrows down
for y0, y1 in [(7.17, 6.13), (3.77, 2.73)]:
    axL.add_patch(FancyArrowPatch((5, y0), (5, y1), arrowstyle="-|>",
                  mutation_scale=20, lw=2.3, color="#333"))
axL.text(5.25, 6.65, "state", fontsize=9.5, color="#333", ha="left")
axL.text(5.25, 3.25, "predicted position", fontsize=9.5, color="#333", ha="left")
# loop arrow: Update -> Input (next frame)
axL.add_patch(FancyArrowPatch((9.55, 1.55), (9.55, 8.35), connectionstyle="arc3,rad=0.42",
              arrowstyle="-|>", mutation_scale=20, lw=2.3, color="#333"))
axL.text(10.15, 5.0, "next\nframe", fontsize=10, color="#333", ha="center", va="center")

# -------- RIGHT: trajectory + occlusion --------
axR.set_title("One vehicle's trajectory over frames", fontsize=15, pad=8)
xs = [1, 2, 3, 4, 5, 6, 7, 8]
ys = [3.0, 3.25, 3.5, 3.75, 4.0, 4.25, 4.5, 4.75]
axR.add_patch(Rectangle((4.55, 2.2), 1.9, 3.6, fc="#bbbbbb", ec="none", alpha=0.35))
axR.text(5.5, 5.55, "OCCLUDED\n(no detection)", ha="center", va="center",
         fontsize=10.5, color="#555", fontweight="bold")

axR.plot(xs, ys, color=C_EST, lw=2.2, zorder=1)
for i in (0, 1, 2, 3, 6, 7):
    axR.plot(xs[i], ys[i], "o", color=C_EST, ms=13, zorder=3)
for i in (4, 5):
    axR.plot(xs[i], ys[i], "o", mfc="white", mec=C_PRED, mew=2.5, ms=13, zorder=3)

xt = 4
axR.plot(xt, ys[3] + 0.55, "o", mfc="white", mec=C_PRED, mew=2.5, ms=12, zorder=4)
axR.plot(xt, ys[3] - 0.5, "X", color=C_OBS, ms=14, mew=3, zorder=4)
axR.annotate("Predicted", (xt, ys[3] + 0.55), (xt - 2.15, ys[3] + 1.35),
             fontsize=10.5, color=C_PRED, fontweight="bold",
             arrowprops=dict(arrowstyle="->", color=C_PRED))
axR.annotate("YOLO observation", (xt, ys[3] - 0.5), (xt + 0.3, ys[3] - 1.5),
             fontsize=10.5, color=C_OBS, fontweight="bold",
             arrowprops=dict(arrowstyle="->", color=C_OBS))
axR.annotate("Estimate = corrected\n(fuse both)", (xt, ys[3]), (xt + 1.0, ys[3] + 1.0),
             fontsize=10.5, color=C_EST, fontweight="bold",
             arrowprops=dict(arrowstyle="->", color=C_EST))

axR.plot(xs[6], ys[6] - 0.5, "X", color=C_OBS, ms=13, mew=3, zorder=4)
axR.annotate("Detection returns\n-> re-acquire ID", (xs[6], ys[6]), (xs[6] - 0.5, ys[6] + 1.15),
             fontsize=10.5, color="#333", arrowprops=dict(arrowstyle="->", color="#333"))

axR.plot([], [], "o", color=C_EST, ms=11, label="Kalman estimate (corrected)")
axR.plot([], [], "o", mfc="white", mec=C_PRED, mew=2.5, ms=11, label="Predict-only (occluded)")
axR.plot([], [], "X", color=C_OBS, ms=12, mew=3, label="YOLO observation (detection)")
axR.legend(loc="lower right", fontsize=10.5, frameon=True)

axR.set_xlim(0.3, 8.7); axR.set_ylim(1.4, 7.2)
axR.set_xlabel("Frame (time) ->", fontsize=12)
axR.set_yticks([])
for s in ("top", "right", "left"):
    axR.spines[s].set_visible(False)

fig.suptitle("Kalman Filter — Predict + Update: smooth the trajectory & keep the track through occlusion",
             fontsize=16, fontweight="bold", y=1.0)
OUT.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(OUT, dpi=170, bbox_inches="tight")
print(f"[OK] saved: {OUT}")
