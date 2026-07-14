# -*- coding: utf-8 -*-
"""Goal Classifier — minh hoạ 6 bước, dùng FRAME CCTV THẬT cho bước 1/2/6.
   Bước 3/4/5 trừu tượng -> giữ sơ đồ tối giản. Chữ Việt.
   Sinh docs/assets/slide_goal_steps_real.png
"""
from pathlib import Path
import numpy as np, cv2, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT.parent / "docs" / "assets" / "slide_goal_steps_real.png"
VIDEO = "Cửa_Nam_-_Điện_Biên_Phủ__16h30_15_9_22_"
TRACK = 4810
G, R, B, P, Y = "#39d353", "#ff3b30", "#1f77b4", "#9467bd", "#ffcc00"


def imread_u(p):
    return cv2.cvtColor(cv2.imdecode(np.fromfile(str(p), np.uint8), cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)


df = pd.read_csv(ROOT / "data/trajectories_real_vn" / f"{VIDEO}.csv")
g = df[df.track_id == TRACK].sort_values("t").reset_index(drop=True)
imgs = sorted((ROOT / "data/auto_label/images").glob(f"{VIDEO}_*.jpg"))
H, W = imread_u(imgs[0]).shape[:2]
cx, cy = g["cx_px"].values, g["cy_px"].values
w_, h_ = g["w_px"].values, g["h_px"].values
fids = g["frame_id"].values


def crop(ax, f_idx, x0, x1, y0, y1):
    ax.imshow(imread_u(imgs[int(fids[f_idx])]))
    ax.set_xlim(max(0, x0), min(W, x1)); ax.set_ylim(min(H, y1), max(0, y0))
    ax.set_xticks([]); ax.set_yticks([])


fig, axes = plt.subplots(2, 3, figsize=(16.5, 8.8))
a1, a2, a3, a4, a5, a6 = axes.flat
titles = ["1. Lấy quỹ đạo từ tracking", "2. Cắt cửa sổ quan sát", "3. Trích 33 đặc trưng",
          "4. Chuẩn hóa đặc trưng", "5. Gradient Boosting", "6. Ra quyết định (argmax)"]
caps = ["bbox + ID bám qua các khung (ByteTrack)", "đoạn ~1,5 s trước nút giao (vàng)",
        "đổi hướng · tốc độ · gia tốc · loại xe · tỉ lệ", "đưa về cùng thang đo (scaler)",
        "nhiều cây nhỏ, mỗi cây sửa lỗi cây trước", "chọn xác suất cao nhất → hướng đi"]
for ax, t, c in zip(axes.flat, titles, caps):
    ax.set_title(t, fontsize=13, fontweight="bold", pad=7)
    ax.text(0.5, -0.09, c, transform=ax.transAxes, ha="center", va="top", fontsize=9.3, color="#444")

# ---- 1. TRACKING (frame thật): bbox khung hiện tại + trail ----
i1 = len(g) * 2 // 5
ci, cyi, wi, hi = cx[i1], cy[i1], w_[i1], h_[i1]
crop(a1, i1, ci - wi*2.2, ci + wi*2.2, cyi - hi*2.2, cyi + hi*2.2)
a1.add_patch(Rectangle((ci - wi/2, cyi - hi/2), wi, hi, fill=False, ec=R, lw=2.5))
a1.text(ci - wi/2, cyi - hi/2 - 6, f"ID {TRACK}", color="white", fontsize=10, fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.15", fc=R, ec="none"))
tr = slice(max(0, i1-5), i1+1)
a1.plot(cx[tr], cy[tr], "-o", color=G, ms=5, lw=2, mec="white", mew=0.6)

# ---- 2. WINDOW (frame thật): toàn quỹ đạo + cửa sổ vàng ----
mx, Mx, my, My = cx.min(), cx.max(), cy.min(), cy.max()
pad = 55
crop(a2, len(g)//2, mx-pad, Mx+pad, my-pad, My+pad)
a2.plot(cx, cy, "-", color=G, lw=3, solid_capstyle="round")
k = max(3, len(cx)//3)
a2.plot(cx[-k:], cy[-k:], "-", color=Y, lw=6, alpha=0.95, solid_capstyle="round", zorder=2)
a2.plot(cx[-k:], cy[-k:], "-", color=G, lw=2, zorder=3)
a2.plot(cx[0], cy[0], "o", mfc="white", mec=G, mew=2.5, ms=11, zorder=4)
a2.annotate("", (cx[-1], cy[-1]), (cx[-2], cy[-2]), arrowprops=dict(arrowstyle="-|>", color=G, lw=3))
a2.text(cx[-k], cy[-k]-14, "cửa sổ ~1,5 s", color="#ffd000", fontsize=10, fontweight="bold")

# ---- 3. FEATURES: chuỗi khung -> thống kê -> vector 33 ----
a3.set_xlim(0, 1); a3.set_ylim(0, 1); a3.set_xticks([]); a3.set_yticks([])
# quỹ đạo = dãy điểm theo khung
tx = [0.06, 0.12, 0.18, 0.24, 0.30]; ty = [0.80, 0.84, 0.88, 0.92, 0.95]
a3.plot(tx, ty, "-o", color=G, ms=5, lw=1.8, mec="white", mew=0.5)
for i, (x, y) in enumerate(zip(tx, ty)):
    a3.text(x, y - 0.055, f"t{i+1}", ha="center", fontsize=6.5, color="#666")
a3.text(0.37, 0.90, "quỹ đạo = chuỗi ~15 khung", fontsize=9, va="center")
a3.text(0.37, 0.83, "mỗi khung: vị trí · tốc độ · hướng · bbox", fontsize=8, color="#555", va="center")
a3.annotate("", (0.6, 0.57), (0.34, 0.57), arrowprops=dict(arrowstyle="-|>", color="#333", lw=2))
a3.text(0.47, 0.65, "mean · std · Σ · max", ha="center", fontsize=9, color="#333", fontweight="bold")
a3.text(0.47, 0.50, "(thống kê trên từng chuỗi)", ha="center", fontsize=7.8, color="#777")
for i in range(33):
    r, cc = divmod(i, 11)
    a3.add_patch(Rectangle((0.30 + cc*0.035, 0.30 - r*0.075), 0.03, 0.062,
                 fc=plt.cm.Blues(0.3 + 0.6*((i*37 % 11)/11)), ec="white", lw=0.5))
a3.text(0.49, 0.08, "→ vector 33 chiều", ha="center", fontsize=9.5, color=B, fontweight="bold")

# ---- 4. NORMALIZE: chuẩn hóa z = (x-μ)/σ ----
a4.set_xlim(0, 1); a4.set_ylim(0, 1); a4.set_xticks([]); a4.set_yticks([])
a4.text(0.5, 0.87, r"$z = \dfrac{x - \mu}{\sigma}$", ha="center", fontsize=21, color=P)
a4.text(0.5, 0.66, "mỗi đặc trưng: trừ trung bình μ, chia độ lệch chuẩn σ",
        ha="center", fontsize=8.6, color="#333")
a4.text(0.5, 0.55, "Trước: speed≈200 · hướng≈30° · bbox≈5000", ha="center", fontsize=8.4, color="#777")
a4.text(0.5, 0.49, "(thang đo chênh nhau rất lớn)", ha="center", fontsize=7.8, color="#777")
a4.annotate("", (0.5, 0.40), (0.5, 0.45), arrowprops=dict(arrowstyle="-|>", color=P, lw=2))
a4.plot([0.2, 0.8], [0.30, 0.30], color="#333", lw=1.4)
for xx in (0.36, 0.44, 0.48, 0.53, 0.58, 0.64):
    a4.plot(xx, 0.30, "o", color=B, ms=6)
a4.plot([0.5, 0.5], [0.285, 0.315], color="#333", lw=1.6)
a4.text(0.2, 0.24, "-2", fontsize=7, ha="center"); a4.text(0.5, 0.24, "0", fontsize=7, ha="center")
a4.text(0.8, 0.24, "+2", fontsize=7, ha="center")
a4.text(0.5, 0.14, "Sau: mọi đặc trưng ~[-2, 2] quanh 0", ha="center", fontsize=8.6, color=B, fontweight="bold")

# ---- 5. GRADIENT BOOSTING: tuần tự, sửa lỗi ----
a5.set_xlim(0, 1); a5.set_ylim(0, 1); a5.set_xticks([]); a5.set_yticks([])
a5.text(0.5, 0.93, r"$F_m = F_{m-1} + \nu\,h_m$", ha="center", fontsize=13, color=P, fontweight="bold")
def tree(cx0, cy0, s=1.0):
    a5.add_patch(Circle((cx0, cy0 + 0.09*s), 0.03*s, fc=P, ec="none"))
    for dx in (-0.07*s, 0.07*s):
        a5.plot([cx0, cx0+dx], [cy0+0.09*s, cy0-0.05*s], color=P, lw=1.5)
        a5.add_patch(Circle((cx0+dx, cy0-0.07*s), 0.026*s, fc="#c9b6e0", ec="none"))
a5.text(0.07, 0.52, "F₀\nđoán\nđầu", ha="center", va="center", fontsize=8, color="#555")
a5.annotate("", (0.2, 0.52), (0.12, 0.52), arrowprops=dict(arrowstyle="-|>", color="#999", lw=1.4))
for i, txp in enumerate([0.31, 0.55, 0.79]):
    tree(txp, 0.5)
    a5.text(txp, 0.31, f"$f_{i+1}$", ha="center", fontsize=10.5, color=P, fontweight="bold")
    if i < 2:
        a5.annotate("", (txp+0.17, 0.52), (txp+0.07, 0.52),
                    arrowprops=dict(arrowstyle="-|>", color="#999", lw=1.4))
a5.text(0.55, 0.70, "mỗi cây học phần LỖI còn lại", ha="center", fontsize=8, color="#888", style="italic")
a5.text(0.31, 0.19, "vd hỏi: |góc quay| > 40° ?", ha="center", fontsize=7.5, color=P, style="italic")
a5.text(0.79, 0.19, "+ …", ha="center", fontsize=12, color="#666")

# ---- 6. OUTPUT (cùng cảnh panel 2): quỹ đạo mờ + mũi tên hướng dự đoán ----
crop(a6, len(g)//2, mx-pad, Mx+pad, my-pad, My+pad)
a6.plot(cx, cy, "-", color=G, lw=2.5, alpha=0.55, solid_capstyle="round")
# mũi tên hướng dự đoán: nối dài theo chiều đi ở cuối quỹ đạo
dx, dy = cx[-1]-cx[-4], cy[-1]-cy[-4]
n = np.hypot(dx, dy) + 1e-6
ex, ey = cx[-1] + dx/n*90, cy[-1] + dy/n*90
a6.annotate("", (ex, ey), (cx[-1], cy[-1]), arrowprops=dict(arrowstyle="-|>", color=G, lw=5))
a6.text(ex, ey - 16, "RẼ TRÁI  (0,70)", color="white", fontsize=11.5, fontweight="bold", ha="center",
        bbox=dict(boxstyle="round,pad=0.3", fc=G, ec="white", lw=1))

fig.suptitle("Goal Classifier — minh hoạ 6 bước trên video thật", fontsize=17, fontweight="bold", y=0.99)
fig.tight_layout(rect=[0, 0, 1, 0.95])
OUT.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(OUT, dpi=170, bbox_inches="tight")
print(f"[OK] saved: {OUT}")
