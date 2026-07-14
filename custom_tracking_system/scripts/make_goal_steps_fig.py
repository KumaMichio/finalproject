# -*- coding: utf-8 -*-
"""Goal Classifier — infographic 6 bước (KHÔNG vẽ quỹ đạo) + ô giải thích Gradient Boosting.
   Chữ Việt. Sinh docs/assets/slide_goal_steps.png
"""
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle, Polygon

OUT = Path(__file__).resolve().parents[2] / "docs" / "assets" / "slide_goal_steps.png"
C_STEP = "#1f77b4"; C_GB = "#9467bd"; C_ACC = "#2ca02c"

fig = plt.figure(figsize=(16, 8.6))
ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, 100); ax.set_ylim(0, 100); ax.axis("off")

# ---------------- LEFT: 6 steps ----------------
steps = [
    ("Lấy quỹ đạo từ tracking (ByteTrack)",
     "chuỗi vị trí (cx,cy), vận tốc, góc hướng, kích thước bbox — pixel-space"),
    ("Cắt cửa sổ quan sát",
     "đoạn ~1.5 s trước nút giao (tối thiểu 5 khung hình)"),
    ("Trích 33 đặc trưng thủ công",
     "đổi hướng · tốc độ · gia tốc ngang · loại xe · tỉ lệ px/m"),
    ("Chuẩn hóa đặc trưng",
     "đưa các đặc trưng về cùng thang đo (scaler)"),
    ("Gradient Boosting (hiệu chỉnh xác suất)",
     "cho ra xác suất 4 lớp: thẳng · trái · phải · quay đầu"),
    ("Chọn kết quả — argmax",
     "hướng có xác suất cao nhất + độ tin cậy"),
]
x0, x1 = 8, 55
ys = [88, 73.5, 59, 44.5, 30, 15.5]
h = 12
for i, ((title, detail), yc) in enumerate(zip(steps, ys)):
    col = C_GB if i == 4 else C_STEP
    ax.add_patch(FancyBboxPatch((x0, yc - h/2), x1 - x0, h, boxstyle="round,pad=0.3",
                 fc=col, ec="none", alpha=0.93))
    ax.add_patch(Circle((x0 - 2.6, yc), 2.6, fc=C_ACC, ec="white", lw=1.5, zorder=5))
    ax.text(x0 - 2.6, yc, str(i + 1), ha="center", va="center", fontsize=13,
            fontweight="bold", color="white", zorder=6)
    ax.text(x0 + 2, yc + 2.4, title, ha="left", va="center", fontsize=12.3,
            fontweight="bold", color="white")
    ax.text(x0 + 2, yc - 2.7, detail, ha="left", va="center", fontsize=9.8, color="#eef4fb")
    if i < len(steps) - 1:
        ax.add_patch(FancyArrowPatch((x0 + (x1 - x0)/2, yc - h/2), (x0 + (x1 - x0)/2, ys[i+1] + h/2),
                     arrowstyle="-|>", mutation_scale=16, lw=2, color="#555"))

# arrow from step 5 -> GB panel
ax.add_patch(FancyArrowPatch((55, 30), (61, 30), arrowstyle="-|>", mutation_scale=20, lw=2.4, color=C_GB))

# ---------------- RIGHT: Gradient Boosting explainer ----------------
ax.add_patch(FancyBboxPatch((61, 8), 37, 84, boxstyle="round,pad=0.5", fc="#f3eef9",
             ec=C_GB, lw=2))
ax.text(79.5, 87, "Gradient Boosting là gì?", ha="center", fontsize=14.5,
        fontweight="bold", color=C_GB)

ax.text(64, 80.5, "• Tập hợp NHIỀU cây quyết định nhỏ, xây TUẦN TỰ.",
        ha="left", fontsize=11, color="#333")
ax.text(64, 76.3, "• Mỗi cây mới học phần LỖI còn lại (residual)",
        ha="left", fontsize=11, color="#333")
ax.text(66, 72.8, "của các cây trước → dần dần sửa sai.",
        ha="left", fontsize=11, color="#333")


def tree(cx, cy, s=1.0):
    ax.add_patch(Circle((cx, cy + 2.2*s), 0.9*s, fc=C_GB, ec="none"))
    for dx in (-2.2*s, 2.2*s):
        ax.plot([cx, cx + dx], [cy + 2.2*s, cy - 1.6*s], color=C_GB, lw=1.6)
        ax.add_patch(Circle((cx + dx, cy - 1.9*s), 0.8*s, fc="#c9b6e0", ec="none"))


tx = [67, 74.5, 82]
for i, cx in enumerate(tx):
    tree(cx, 62)
    ax.text(cx, 56.5, f"$f_{i+1}$", ha="center", fontsize=12, color=C_GB, fontweight="bold")
    if i < 2:
        ax.text((cx + tx[i+1]) / 2, 62, "+", ha="center", va="center", fontsize=18, color="#666")
ax.text(88.5, 62, "+  …", ha="left", va="center", fontsize=14, color="#666")
ax.annotate("", (74.5, 66.5), (67, 66.5), arrowprops=dict(arrowstyle="->", color="#999", lw=1.4))
ax.annotate("", (82, 66.5), (74.5, 66.5), arrowprops=dict(arrowstyle="->", color="#999", lw=1.4))
ax.text(74.5, 68.6, "học phần lỗi còn lại", ha="center", fontsize=8.8, color="#777", style="italic")

ax.text(79.5, 50, "Dự đoán = tổng có trọng số của tất cả cây:",
        ha="center", fontsize=10.5, color="#333")
ax.text(79.5, 45, r"$F_m(x) = F_{m-1}(x) + \nu\,\cdot\,h_m(x)$",
        ha="center", fontsize=14, color=C_GB, fontweight="bold")
ax.text(79.5, 40.3, r"$h_m$ = cây thứ m  ·  $\nu$ = learning rate", ha="center", fontsize=9.3, color="#666")

ax.text(64, 33.5, "• Kết quả cộng dồn → hiệu chỉnh (calibrate)",
        ha="left", fontsize=11, color="#333")
ax.text(66, 30, "thành XÁC SUẤT 4 lớp có ý nghĩa.",
        ha="left", fontsize=11, color="#333")

ax.text(64, 23.5, "Vì sao hợp bài toán này:", ha="left", fontsize=11, fontweight="bold", color=C_GB)
for j, t in enumerate([
    "dữ liệu dạng bảng (33 đặc trưng số)",
    "nắm tốt quan hệ phi tuyến giữa các đặc trưng",
    "mạnh với dữ liệu vừa phải, ít overfit hơn",
]):
    ax.text(66, 19.5 - j*3.6, f"– {t}", ha="left", fontsize=10, color="#444")

fig.suptitle("Goal Classifier — cách hoạt động qua 6 bước",
             fontsize=17, fontweight="bold", y=0.99)
OUT.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(OUT, dpi=170, bbox_inches="tight")
print(f"[OK] saved: {OUT}")
