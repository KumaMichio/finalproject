# -*- coding: utf-8 -*-
"""So sánh YOLO11s GỐC (COCO) vs yolo11s_vn (fine-tune) trên CÙNG tập nhãn người Xã Đàn.
mAP50 tự tính (101-point interp, IoU 0.5) — validate khớp số ultralytics của model fine-tune.
"""
import json
import numpy as np
from pathlib import Path
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[2]
EVAL = ROOT / "data" / "eval_xadan_manual"
IMGDIR = EVAL / "images" / "val"
GT_JSON = EVAL / "corrected_labels.json"
NAMES = ["person", "car", "motorcycle", "bus", "truck"]
COCO2VN = {0: 0, 2: 1, 3: 2, 5: 3, 7: 4}   # COCO id -> 5-class VN id


def iou(a, b):
    x1 = max(a[0], b[0]); y1 = max(a[1], b[1]); x2 = min(a[2], b[2]); y2 = min(a[3], b[3])
    iw = max(0, x2 - x1); ih = max(0, y2 - y1); inter = iw * ih
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def load_gt():
    d = json.loads(GT_JSON.read_text(encoding="utf-8"))
    gt = {}   # file -> list of (cls, x1,y1,x2,y2)
    for im in d["images"]:
        if not im.get("reviewed"):
            continue
        w, h = im["w"], im["h"]
        boxes = []
        for b in im["boxes"]:
            cx, cy, bw, bh = b["x"] * w, b["y"] * h, b["w"] * w, b["h"] * h
            boxes.append((int(b["c"]), cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2))
        gt[im["file"]] = boxes
    return gt


def predict(model_path, files, coco=False):
    m = YOLO(str(model_path))
    preds = {}   # file -> list of (cls, conf, x1,y1,x2,y2)
    for f in files:
        r = m(str(IMGDIR / f), imgsz=640, conf=0.001, iou=0.7, verbose=False)[0]
        out = []
        if r.boxes is not None:
            for c, cf, xy in zip(r.boxes.cls.tolist(), r.boxes.conf.tolist(),
                                 r.boxes.xyxy.tolist()):
                c = int(c)
                if coco:
                    if c not in COCO2VN:
                        continue
                    c = COCO2VN[c]
                out.append((c, float(cf), *xy))
        preds[f] = out
    return preds


def ap_class(preds, gt, cls, iou_thr=0.5):
    """AP 101-point cho 1 lớp (COCO-style)."""
    npos = sum(1 for f in gt for b in gt[f] if b[0] == cls)
    if npos == 0:
        return None
    dets = []   # (conf, file, box)
    for f, ps in preds.items():
        for p in ps:
            if p[0] == cls:
                dets.append((p[1], f, p[2:6]))
    dets.sort(key=lambda x: -x[0])
    used = {f: [False] * len(gt.get(f, [])) for f in gt}
    tp = np.zeros(len(dets)); fp = np.zeros(len(dets))
    for i, (cf, f, box) in enumerate(dets):
        gbs = gt.get(f, [])
        best, bj = iou_thr, -1
        for j, g in enumerate(gbs):
            if g[0] != cls or used[f][j]:
                continue
            v = iou(box, g[1:5])
            if v >= best:
                best, bj = v, j
        if bj >= 0:
            tp[i] = 1; used[f][bj] = True
        else:
            fp[i] = 1
    tpc = np.cumsum(tp); fpc = np.cumsum(fp)
    rec = tpc / npos
    prec = tpc / np.maximum(tpc + fpc, 1e-9)
    ap = 0.0
    for t in np.linspace(0, 1, 101):
        p = prec[rec >= t].max() if (rec >= t).any() else 0
        ap += p / 101
    return ap


def evaluate(name, preds, gt):
    aps = {}
    for ci, cn in enumerate(NAMES):
        a = ap_class(preds, gt, ci)
        if a is not None:
            aps[cn] = a
    mAP = float(np.mean(list(aps.values())))
    print(f"\n[{name}]  mAP50 = {mAP:.4f}")
    for cn, a in aps.items():
        print(f"    {cn:<11} AP50 = {a:.4f}")
    return mAP, aps


def main():
    gt = load_gt()
    files = list(gt.keys())
    print(f"Tap: {len(files)} anh nhan nguoi, {sum(len(v) for v in gt.values())} box GT")

    print("\n>>> Chay model GOC (COCO) ...")
    p_coco = predict(ROOT / "yolo11s.pt", files, coco=True)
    m_coco, ap_coco = evaluate("YOLO11s GOC (COCO, zero-shot VN)", p_coco, gt)

    print("\n>>> Chay model FINE-TUNE (yolo11s_vn) ...")
    p_ft = predict(ROOT / "weights" / "yolo11s_vn.pt", files, coco=False)
    m_ft, ap_ft = evaluate("yolo11s_vn (fine-tune auto-label teacher ITD)", p_ft, gt)

    print("\n" + "=" * 60)
    print("SO SANH mAP50 tren CUNG tap nhan nguoi Xa Dan")
    print("=" * 60)
    print(f"  YOLO11s GOC (COCO)          : {m_coco*100:.1f}%")
    print(f"  yolo11s_vn (sau fine-tune)  : {m_ft*100:.1f}%   (+{(m_ft-m_coco)*100:.1f} diem)")
    print(f"  (validate: yolo11s_vn ultralytics ~93.25%; ham nay = {m_ft*100:.1f}%)")
    out = {"eval": "xadan_manual_human", "n_images": len(files),
           "coco_stock": {"mAP50": round(m_coco, 4), "per_class": {k: round(v, 4) for k, v in ap_coco.items()}},
           "finetuned": {"mAP50": round(m_ft, 4), "per_class": {k: round(v, 4) for k, v in ap_ft.items()}}}
    op = ROOT.parent / "docs" / "assets" / "yolo_coco_vs_finetune.json"
    op.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n  -> {op}")


if __name__ == "__main__":
    main()
