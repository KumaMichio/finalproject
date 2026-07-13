# -*- coding: utf-8 -*-
"""Hình minh hoạ ByteTrack (khung DỌC): 3 khung liên tiếp xếp dọc, box + track ID giữ nguyên."""
import sys
import numpy as np, cv2, pandas as pd
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
VID = sys.argv[1] if len(sys.argv) > 1 else "Phố_Huế_Trần_Khát_Chân__12h_11_9_22_"
NP = 3                 # số khung liên tiếp
PW = 960               # bề rộng mỗi panel
MIN_AREA = 900         # chỉ bỏ box quá nhỏ (xe rất xa, box hay lệch)
LABEL_AREA = 3800      # chỉ ghi nhãn ID cho xe đủ to (tránh rối chữ)
OUT = ROOT.parent / "docs" / "assets" / "slide_bytetrack.png"
ARB = "C:/Windows/Fonts/arialbd.ttf"
CLS = {"person": "Nguoi", "car": "Oto", "motorcycle": "Xemay", "bus": "Bus", "truck": "Tai"}


def imread_u(p): return cv2.imdecode(np.fromfile(str(p), np.uint8), cv2.IMREAD_COLOR)


def color(tid):
    rng = (tid * 2654435761) & 0xFFFFFFFF
    return (int(70 + rng % 175), int(70 + (rng >> 8) % 175), int(70 + (rng >> 16) % 175))


df = pd.read_csv(ROOT / "data/trajectories_real_vn" / f"{VID}.csv")
imgs = sorted((ROOT / "data/auto_label/images").glob(f"{VID}_*.jpg"))
by_f = {f: set(g.track_id) for f, g in df.groupby("frame_id")}

# chọn cửa sổ NP khung ĐÔNG xe nhất (nhiều box hợp lệ ở cả 3 khung)
def n_valid(F):
    g = df[df.frame_id == F]
    return int((g.w_px * g.h_px >= MIN_AREA).sum())

best_F, best_score = 0, -1
for F in range(0, len(imgs) - NP):
    if not all((F + i) in by_f for i in range(NP)):
        continue
    sc_ = min(n_valid(F + i) for i in range(NP))     # khung ít xe nhất trong cửa sổ
    if sc_ > best_score:
        best_score, best_F = sc_, F
F0 = best_F
print(f"F0={F0}  ve MOI xe duoc track (area>={MIN_AREA}); nhan ID cho xe >={LABEL_AREA}")

panels = []
for i in range(NP):
    F = F0 + i
    im = imread_u(imgs[F]); sc = PW / im.shape[1]
    im = cv2.resize(im, (PW, int(im.shape[0] * sc)))
    g = df[df.frame_id == F].copy()
    g["a"] = g.w_px * g.h_px
    g = g[g.a >= MIN_AREA].sort_values("a")           # vẽ xe to sau (nhãn đè lên trên)
    for _, r in g.iterrows():
        tid = int(r.track_id); c = color(tid)
        x1 = int((r.cx_px - r.w_px / 2) * sc); y1 = int((r.cy_px - r.h_px / 2) * sc)
        x2 = int((r.cx_px + r.w_px / 2) * sc); y2 = int((r.cy_px + r.h_px / 2) * sc)
        cv2.rectangle(im, (x1, y1), (x2, y2), c, 2, cv2.LINE_AA)
        if r.a >= LABEL_AREA:                          # chỉ ghi nhãn xe đủ to
            lab = f"{tid}:{CLS.get(r['class'], r['class'])}"
            (tw, th), _ = cv2.getTextSize(lab, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(im, (x1, y1 - th - 6), (x1 + tw + 4, y1), c, -1)
            cv2.putText(im, lab, (x1 + 2, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
    panels.append(im)

PH = panels[0].shape[0]
GAP = 16
CAP_H = 70
W = PW * NP + GAP * (NP - 1)          # xếp NGANG
H = PH + CAP_H
canvas = np.full((H, W, 3), 245, np.uint8)
for i, p in enumerate(panels):
    x = i * (PW + GAP)
    canvas[0:PH, x:x + PW] = p

im = Image.fromarray(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB))
dr = ImageDraw.Draw(im, "RGBA")
ft = ImageFont.truetype(ARB, 34); fc = ImageFont.truetype(ARB, 32)
titles = ["Khung t", "Khung t + 1", "Khung t + 2"]
for i, ttl in enumerate(titles[:NP]):
    x = i * (PW + GAP)
    b = dr.textbbox((0, 0), ttl, font=ft)
    dr.rounded_rectangle([x + 16, 14, x + 16 + (b[2] - b[0]) + 24, 14 + (b[3] - b[1]) + 16], 8,
                         fill=(11, 40, 80, 235))
    dr.text((x + 28, 20), ttl, font=ft, fill=(255, 255, 255))
cap = "ByteTrack — cùng phương tiện GIỮ NGUYÊN track ID qua các khung"
dr.rectangle([0, H - CAP_H, W, H], fill=(15, 25, 45))
fsz = 32
while fsz > 18:
    fc = ImageFont.truetype(ARB, fsz)
    b = dr.textbbox((0, 0), cap, font=fc)
    if (b[2] - b[0]) <= W - 40:
        break
    fsz -= 1
dr.text(((W - (b[2] - b[0])) / 2, H - CAP_H + (CAP_H - (b[3] - b[1])) // 2 - 4), cap, font=fc, fill=(255, 255, 255))
im.save(OUT)
print("saved", OUT, im.size)
