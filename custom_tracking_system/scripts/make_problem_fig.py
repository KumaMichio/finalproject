# -*- coding: utf-8 -*-
"""Sinh hình minh hoạ ĐẶT VẤN ĐỀ: quỹ đạo quan sát được + 3 hướng rẽ + '?'."""
import numpy as np, cv2
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
VID = "Cửa_Nam_-_Điện_Biên_Phủ__16h15_15_9_22_"
TID = 1967
OBS = 0.74           # tỉ lệ quỹ đạo "đã quan sát"
SKIP = 3             # bỏ vài khung đầu (nhiễu gần camera)
SC = 2.5             # upscale
OUT = ROOT.parent / "docs" / "assets" / "slide_dat_van_de.png"
AR = "C:/Windows/Fonts/arial.ttf"; ARB = "C:/Windows/Fonts/arialbd.ttf"


def imread_u(p): return cv2.imdecode(np.fromfile(str(p), np.uint8), cv2.IMREAD_COLOR)


df = __import__("pandas").read_csv(ROOT / "data/trajectories_real_vn" / f"{VID}.csv")
s = df[df.track_id == TID].sort_values("t").reset_index(drop=True)
imgs = sorted((ROOT / "data/auto_label/images").glob(f"{VID}_*.jpg"))
n = len(s); k = int(n * OBS)
pts = s[["cx_px", "cy_px"]].values * SC
# làm mượt quỹ đạo (moving average) để đường sạch, đỡ zigzag
sm = pts.copy()
for i in range(len(pts)):
    lo = max(0, i - 2); hi = min(len(pts), i + 3)
    sm[i] = pts[lo:hi].mean(axis=0)
pts = sm
fid = int(s.frame_id.iloc[k])
bg = imread_u(imgs[fid])
bg = cv2.resize(bg, (int(bg.shape[1] * SC), int(bg.shape[0] * SC)), interpolation=cv2.INTER_CUBIC)
bg = (bg * 0.52).astype(np.uint8)   # làm tối nền cho overlay nổi
H, W = bg.shape[:2]

ov = bg.copy()
# quỹ đạo quan sát (cyan sáng), dày + chấm
for i in range(SKIP + 1, k):
    cv2.line(ov, tuple(pts[i - 1].astype(int)), tuple(pts[i].astype(int)), (255, 235, 60), 9, cv2.LINE_AA)
for i in range(SKIP, k, 3):
    cv2.circle(ov, tuple(pts[i].astype(int)), 6, (255, 255, 180), -1, cv2.LINE_AA)
P = pts[k - 1]
# hộp xe tại điểm hiện tại
w = s.w_px.iloc[k - 1] * SC; h = s.h_px.iloc[k - 1] * SC
cv2.rectangle(ov, (int(P[0] - w / 2), int(P[1] - h / 2)), (int(P[0] + w / 2), int(P[1] + h / 2)),
              (255, 235, 60), 4, cv2.LINE_AA)
# heading = huong TONG THE (entry -> exit) => chac chan di len
d = pts[-1] - pts[0]; d = d / (np.linalg.norm(d) + 1e-6)


def rot(v, deg):
    a = np.radians(deg); c, s_ = np.cos(a), np.sin(a)
    return np.array([v[0] * c - v[1] * s_, v[0] * s_ + v[1] * c])


L = 190
AMBER = (40, 175, 255)  # BGR (cam)
tips = {}
for deg, key in [(-42, "left"), (0, "straight"), (42, "right")]:
    v = rot(d, deg); tip = P + v * L
    cv2.arrowedLine(ov, tuple(P.astype(int)), tuple(tip.astype(int)), AMBER, 9, cv2.LINE_AA, tipLength=0.16)
    tips[key] = tip
bg = cv2.addWeighted(ov, 0.95, bg, 0.05, 0)

# ── text bằng PIL (tiếng Việt) ──
im = Image.fromarray(cv2.cvtColor(bg, cv2.COLOR_BGR2RGB))
dr = ImageDraw.Draw(im, "RGBA")
fL = ImageFont.truetype(ARB, 34); fM = ImageFont.truetype(ARB, 30); fQ = ImageFont.truetype(ARB, 120)


def label(xy, txt, fnt, fill=(20, 20, 20), bg=(255, 200, 60, 235), pad=8, anchor="mm"):
    b = dr.textbbox((0, 0), txt, font=fnt)
    tw, th = b[2] - b[0], b[3] - b[1]
    x, y = xy
    if anchor == "mm": x -= tw / 2; y -= th / 2
    dr.rounded_rectangle([x - pad, y - pad, x + tw + pad, y + th + pad], 8, fill=bg)
    dr.text((x - b[0], y - b[1]), txt, font=fnt, fill=fill)


for key, txt in [("left", "Rẽ trái?"), ("straight", "Đi thẳng?"), ("right", "Rẽ phải?")]:
    t = tips[key]
    label((t[0], t[1] - 26), txt, fM, fill=(20, 20, 20), bg=(255, 190, 40, 245))
# dau ? — dat giua xe va fan (khoang trong)
q = P + d * 150
label((q[0] + 70, q[1]), "?", fQ, fill=(255, 70, 70), bg=(255, 255, 255, 0), pad=0)
# nhan quy dao quan sat
op = pts[int(k * 0.45)]
label((op[0] + 130, op[1]), "Quỹ đạo quan sát được", ImageFont.truetype(ARB, 27),
      fill=(255, 255, 255), bg=(30, 150, 60, 240))

# thanh caption duoi
dr.rectangle([0, H - 70, W, H], fill=(15, 25, 45, 220))
cap = "Từ quỹ đạo quan sát trong một camera — có thể dự đoán hướng đi của xe sau ngã tư?"
fc = ImageFont.truetype(ARB, 32)
b = dr.textbbox((0, 0), cap, font=fc)
dr.text(((W - (b[2] - b[0])) / 2, H - 52), cap, font=fc, fill=(255, 255, 255))

im.save(OUT)
print("saved", OUT, im.size, "| track", TID, "obs", k, "/", n)
