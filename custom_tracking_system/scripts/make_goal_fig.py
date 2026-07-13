# -*- coding: utf-8 -*-
"""Hình minh hoạ Goal Classifier: quỹ đạo quan sát + hướng MODEL DỰ ĐOÁN (thật) + độ tin cậy."""
import sys, pickle
import numpy as np, cv2, pandas as pd
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
EVAL = ROOT / "scripts" / "eval"
WEIGHTS = ROOT / "weights"
LABELS = ROOT / "data" / "turning_labels.csv"
TRAJ = ROOT / "data" / "trajectories_real_vn"
IMG = ROOT / "data" / "auto_label" / "images"
TARGET = sys.argv[1] if len(sys.argv) > 1 else "left"   # straight/left/right/u_turn
OUT = ROOT.parent / "docs" / "assets" / (
    "slide_goal_classifier.png" if TARGET == "left" else f"slide_goal_{TARGET}.png")
ARB = "C:/Windows/Fonts/arialbd.ttf"; AR = "C:/Windows/Fonts/arial.ttf"
SC = 2.5
FRAC = {"u_turn": 0.80}.get(TARGET, 0.58)   # u_turn: hiện gần hết đường vòng
CONF_MIN = {"straight": 0.55, "left": 0.55, "right": 0.55, "u_turn": 0.38}
VN = {"straight": "Đi thẳng", "left": "Rẽ trái", "right": "Rẽ phải", "u_turn": "Quay đầu"}


def imread_u(p): return cv2.imdecode(np.fromfile(str(p), np.uint8), cv2.IMREAD_COLOR)


# ── model + dự đoán trên toàn bộ track (dùng đúng feature của train script) ──
sys.path.insert(0, str(EVAL))
from verify_goal_labels import reproduce_test_split
print("[1] Tái tạo features + test split...")
test_keys, X_all, meta_all, y_all, feat = reproduce_test_split(1.5, 0.65)
clf = pickle.load(open(WEIGHTS / "goal_classifier_real.pkl", "rb"))
scaler = pickle.load(open(WEIGHTS / "goal_scaler_real.pkl", "rb"))
le = pickle.load(open(WEIGHTS / "goal_label_encoder_real.pkl", "rb"))
proba = clf.predict_proba(scaler.transform(X_all))
classes = list(le.classes_)                       # ['left','right','straight','u_turn']

lab = pd.read_csv(LABELS); lab["key"] = list(zip(lab.video_id, lab.track_id))
geo = {k: r for k, r in zip(lab.key, lab.to_dict("records"))}

# gom dự đoán theo track (lần đầu)
seen = {}
for m, pr in zip(meta_all, proba):
    k = (m["video_id"], m["track_id"])
    if k not in seen:
        seen[k] = pr

# ── chọn track: dự đoán RẼ, tin cậy cao, ĐÚNG (khớp nhãn auto), box đủ to ──
print("[2] Chọn track đẹp...")
def build_cand(require_correct=True, min_frames=16):
    out = []
    for k, pr in seen.items():
        pi = int(pr.argmax()); pcls = classes[pi]; conf = float(pr[pi])
        g = geo.get(k)
        if g is None or pcls != TARGET:
            continue
        if conf < CONF_MIN.get(TARGET, 0.5) or g.get("n_frames", 0) < min_frames:
            continue
        if require_correct and g.get("goal_label") != pcls:
            continue
        if not (TRAJ / f"{k[0]}.csv").exists():
            continue
        out.append((conf, k, pcls, pr, g))
    return out


cand = build_cand(True)
if not cand:
    cand = build_cand(False, 12)     # nới lỏng (u_turn hiếm)
cand.sort(reverse=True, key=lambda x: x[0])
if not cand:
    print("Khong tim thay candidate cho", TARGET); sys.exit(1)

# ưu tiên track có box to (xe gần) trong top
best = None
for conf, k, pcls, pr, g in cand[:40]:
    s = pd.read_csv(TRAJ / f"{k[0]}.csv"); s = s[s.track_id == k[1]]
    area = (s.w_px * s.h_px).median()
    if best is None or area > best[0]:
        best = (area, conf, k, pcls, pr, g, s)
area, conf, key, pcls, pr, g, s = best
s = s.sort_values("t").reset_index(drop=True)
print(f"  chon {key[0][:30]} #{key[1]}  pred={pcls} conf={conf:.2f} n={len(s)} area={area:.0f}")

# ── vẽ ──
imgs = sorted(IMG.glob(f"{key[0]}_*.jpg"))
pts = s[["cx_px", "cy_px"]].values * SC
sm = pts.copy()
for i in range(len(pts)):
    sm[i] = pts[max(0, i - 2):i + 3].mean(0)
pts = sm
k_obs = max(3, int(len(s) * FRAC))
P = pts[k_obs - 1]
fid = int(s.frame_id.iloc[k_obs - 1])
bg = imread_u(imgs[fid])
bg = cv2.resize(bg, (int(bg.shape[1] * SC), int(bg.shape[0] * SC)), interpolation=cv2.INTER_CUBIC)
bg = (bg * 0.5).astype(np.uint8)
H, W = bg.shape[:2]
ov = bg.copy()
for i in range(4, k_obs):
    cv2.line(ov, tuple(pts[i - 1].astype(int)), tuple(pts[i].astype(int)), (255, 235, 60), 9, cv2.LINE_AA)
for i in range(3, k_obs, 3):
    cv2.circle(ov, tuple(pts[i].astype(int)), 6, (255, 255, 180), -1, cv2.LINE_AA)
w = s.w_px.iloc[k_obs - 1] * SC; h = s.h_px.iloc[k_obs - 1] * SC
cv2.rectangle(ov, (int(P[0] - w / 2), int(P[1] - h / 2)), (int(P[0] + w / 2), int(P[1] + h / 2)),
              (255, 235, 60), 4, cv2.LINE_AA)
# mũi tên = hướng model dự đoán (theo exit_heading, khớp lớp dự đoán)
rad = np.radians(float(g["exit_heading"]))
tip = P + np.array([np.cos(rad), np.sin(rad)]) * 280
cv2.arrowedLine(ov, tuple(P.astype(int)), tuple(tip.astype(int)), (255, 255, 255), 18, cv2.LINE_AA, tipLength=0.22)
cv2.arrowedLine(ov, tuple(P.astype(int)), tuple(tip.astype(int)), (40, 175, 255), 10, cv2.LINE_AA, tipLength=0.22)
bg = cv2.addWeighted(ov, 0.95, bg, 0.05, 0)

im = Image.fromarray(cv2.cvtColor(bg, cv2.COLOR_BGR2RGB))
dr = ImageDraw.Draw(im, "RGBA")


def box(xy, txt, fnt, fg, bgc, pad=10, anchor="lt"):
    b = dr.textbbox((0, 0), txt, font=fnt); tw, th = b[2] - b[0], b[3] - b[1]
    x, y = xy
    if anchor == "mm": x -= tw / 2; y -= th / 2
    dr.rounded_rectangle([x - pad, y - pad, x + tw + pad, y + th + pad], 8, fill=bgc)
    dr.text((x - b[0], y - b[1]), txt, font=fnt, fill=fg)


# nhãn quỹ đạo
op = pts[int(k_obs * 0.45)]
box((op[0] + 120, op[1]), "Quỹ đạo quan sát được", ImageFont.truetype(ARB, 27), (255, 255, 255), (30, 150, 60, 240))
# badge dự đoán (cạnh mũi tên)
box((tip[0] + 10, tip[1] - 20), f"Dự đoán: {VN[pcls].upper()}", ImageFont.truetype(ARB, 30), (20, 20, 20), (255, 190, 40, 250))
# panel xác suất 4 lớp (góc trên trái)
px, py, pw = 30, 30, 430
dr.rounded_rectangle([px, py, px + pw, py + 250], 12, fill=(20, 30, 50, 235))
dr.text((px + 18, py + 14), "Goal Classifier — xác suất hướng đi", font=ImageFont.truetype(ARB, 24), fill=(255, 255, 255))
order = ["straight", "left", "right", "u_turn"]
fb = ImageFont.truetype(ARB, 22)
for i, cl in enumerate(order):
    p = float(pr[classes.index(cl)]); yy = py + 60 + i * 46
    hot = (cl == pcls)
    dr.text((px + 18, yy), VN[cl], font=fb, fill=(255, 210, 90) if hot else (210, 220, 235))
    bx0 = px + 150; bw = int((pw - 210) * p)
    dr.rectangle([bx0, yy + 2, bx0 + (pw - 210), yy + 26], fill=(60, 70, 95))
    dr.rectangle([bx0, yy + 2, bx0 + bw, yy + 26], fill=(255, 180, 40) if hot else (90, 150, 235))
    dr.text((bx0 + (pw - 210) + 8, yy), f"{p*100:.0f}%", font=fb, fill=(255, 255, 255))
# caption
CAP = 66
canvas = Image.new("RGB", (W, H + CAP), (15, 25, 45))
canvas.paste(im, (0, 0))
d2 = ImageDraw.Draw(canvas)
cap = "Goal Classifier: từ quỹ đạo quan sát → dự đoán hướng đi của xe tại ngã tư"
fc = ImageFont.truetype(ARB, 32); b = d2.textbbox((0, 0), cap, font=fc)
d2.text(((W - (b[2] - b[0])) / 2, H + 16), cap, font=fc, fill=(255, 255, 255))
canvas.save(OUT)
print("saved", OUT, canvas.size)
