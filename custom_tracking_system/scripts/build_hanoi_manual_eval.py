"""
Tao TAP TEST GAN NHAN TAY tu video Pho Hue that (demo_pho_hue.mp4).

Muc dich: sinh bang chung THAT cho YOLO tren canh CCTV Ha Noi, doi chieu
nhan NGUOI KIEM thay vi nhan auto-label (khac han eval_yolo11s_autolabel.py).

Quy trinh:
  1. Trich N frame trai deu tu video.
  2. Tien gan nhan bang model THAY yolo11x-ITD (KHONG dung yolo11s de tranh
     thien vi model dang do). Nguoi chi can sua/xoa/them.
  3. Xuat prelabels.js + labels YOLO + eval.yaml de annotate.html doc.

Run tu project root:
  custom_tracking_system/venv_tracking/Scripts/python \
      custom_tracking_system/scripts/build_hanoi_manual_eval.py --n 40 --conf 0.25
"""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np

BASE       = Path("custom_tracking_system")
VIDEO      = Path("data/demo_pho_hue.mp4")
TEACHER    = BASE / "weights" / "best_xl_ITD_v1.2.pt"
DEFAULT_OUT = BASE / "data" / "eval_hanoi_manual"
TEMPLATE_HTML = DEFAULT_OUT / "annotate.html"  # tool tai dung cho moi tap

YOLO_CLASSES = {"person": 0, "car": 1, "motorcycle": 2, "bus": 3, "truck": 4}
CLASS_NAMES  = {v: k for k, v in YOLO_CLASSES.items()}
ITD_REMAP = {
    "two wheeler": "motorcycle", "two_wheeler": "motorcycle",
    "autorickshaw": "motorcycle", "car": "car", "bus": "bus",
    "lcv": "truck", "truck": "truck", "bicycle": None,
    "pedestrian": "person", "pedestrain": "person",
}


def remap_itd(raw_name):
    key = raw_name.strip().lower().replace(" ", "_")
    for k, v in ITD_REMAP.items():
        if k.lower().replace(" ", "_") == key:
            return v
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=40, help="So frame trich (default 40)")
    ap.add_argument("--conf", type=float, default=0.25,
                    help="Conf teacher (thap de bat nhieu, nguoi loc bot; default 0.25)")
    ap.add_argument("--video", type=str, default=str(VIDEO))
    ap.add_argument("--out", type=str, default=str(DEFAULT_OUT),
                    help="Thu muc tap eval (moi clip mot thu muc rieng)")
    args = ap.parse_args()

    video = Path(args.video)
    OUT = Path(args.out)
    img_dir = OUT / "images" / "val"
    lbl_dir = OUT / "labels" / "val"
    img_dir.mkdir(parents=True, exist_ok=True)
    lbl_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps   = cap.get(cv2.CAP_PROP_FPS) or 1.0
    print(f"Video: {video}  total={total} frames  fps={fps:.2f}")

    # Trich N frame trai deu
    idxs = sorted(set(int(round(x)) for x in np.linspace(0, total - 1, args.n)))
    frames = {}
    for i in idxs:
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ok, fr = cap.read()
        if ok:
            frames[i] = fr
    cap.release()
    print(f"Trich duoc {len(frames)} frame")

    from ultralytics import YOLO
    print(f"Load teacher: {TEACHER}")
    model = YOLO(str(TEACHER))
    names = model.names

    prelabels = {"class_names": [CLASS_NAMES[i] for i in range(5)], "images": []}
    keys = sorted(frames.keys())
    for n, fidx in enumerate(keys):
        fr = frames[fidx]
        fh, fw = fr.shape[:2]
        stem = f"frame_{n:03d}_src{fidx:04d}"
        cv2.imwrite(str(img_dir / f"{stem}.jpg"), fr, [cv2.IMWRITE_JPEG_QUALITY, 92])

        res = model(fr, imgsz=640, conf=args.conf, verbose=False)[0]
        boxes = []
        lines = []
        if res.boxes is not None:
            for b in res.boxes:
                mapped = remap_itd(names[int(b.cls[0])])
                if mapped is None:
                    continue
                cid = YOLO_CLASSES[mapped]
                x1, y1, x2, y2 = b.xyxy[0].cpu().numpy().astype(float)
                cx = ((x1 + x2) / 2) / fw
                cy = ((y1 + y2) / 2) / fh
                bw = (x2 - x1) / fw
                bh = (y2 - y1) / fh
                boxes.append({"c": cid,
                              "x": round(float(cx), 5), "y": round(float(cy), 5),
                              "w": round(float(bw), 5), "h": round(float(bh), 5)})
                lines.append(f"{cid} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
        (lbl_dir / f"{stem}.txt").write_text("\n".join(lines))
        prelabels["images"].append({"file": f"{stem}.jpg", "w": fw, "h": fh, "boxes": boxes})
        print(f"  [{n+1}/{len(keys)}] {stem}  boxes={len(boxes)}")

    # prelabels.js cho annotate.html (nap qua <script src>, chay duoc tren file://)
    (OUT / "prelabels.js").write_text(
        "window.PRELABELS = " + json.dumps(prelabels, ensure_ascii=False) + ";",
        encoding="utf-8")

    # eval.yaml
    (OUT / "eval.yaml").write_text(
        f"path: {OUT.resolve()}\n"
        f"train: images/val\nval: images/val\n\n"
        f"nc: 5\nnames: {[CLASS_NAMES[i] for i in range(5)]}\n")

    # Copy cong cu duyet nhan vao thu muc tap (neu chua co)
    import shutil as _sh
    dst_html = OUT / "annotate.html"
    if not dst_html.exists() and TEMPLATE_HTML.exists() and TEMPLATE_HTML.resolve() != dst_html.resolve():
        _sh.copy(TEMPLATE_HTML, dst_html)

    tot_boxes = sum(len(im["boxes"]) for im in prelabels["images"])
    print(f"\nXONG. {len(keys)} anh, {tot_boxes} box tien gan nhan (teacher).")
    print(f"Output: {OUT.resolve()}")
    print(f"  -> Mo {dst_html} de duyet/sua nhan.")


if __name__ == "__main__":
    main()
