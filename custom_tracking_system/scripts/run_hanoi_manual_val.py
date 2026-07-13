"""
Chay YOLO val tren TAP NHAN TAY (corrected_labels.json tu annotate.html).
Chi lay cac anh da danh dau 'reviewed' -> tap test that, nhan NGUOI KIEM.

Output: docs/assets/yolo11s_vn_manual_hanoi/ (metrics.json + plots)

Run tu project root, SAU KHI da xuat corrected_labels.json vao thu muc eval:
  custom_tracking_system/venv_tracking/Scripts/python \
      custom_tracking_system/scripts/run_hanoi_manual_val.py
"""
import argparse
import json
import shutil
from pathlib import Path

from ultralytics import YOLO

BASE     = Path("custom_tracking_system")
EVAL     = BASE / "data" / "eval_hanoi_manual"
WEIGHTS  = BASE / "weights" / "yolo11s_vn.pt"
NAMES    = ["person", "car", "motorcycle", "bus", "truck"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval-dir", default=str(EVAL),
                    help="Thu muc tap eval chua corrected_labels.json")
    ap.add_argument("--name", default="yolo11s_vn_manual_hanoi",
                    help="Ten thu muc output duoi docs/assets/")
    ap.add_argument("--note", default="Nhan NGUOI KIEM tren video demo Pho Hue that. Cung dia diem co trong train (demo-fidelity).")
    ap.add_argument("--corrected", default=None)
    ap.add_argument("--weights", default=str(WEIGHTS))
    ap.add_argument("--conf", type=float, default=0.001,
                    help="conf thap chuan cho val/mAP (default 0.001)")
    args = ap.parse_args()

    EVAL_DIR = Path(args.eval_dir)
    OUT_DIR = Path("docs/assets") / args.name
    cj = Path(args.corrected) if args.corrected else EVAL_DIR / "corrected_labels.json"
    if not cj.exists():
        raise SystemExit(f"KHONG THAY {cj}. Hay xuat corrected_labels.json tu annotate.html vao {EVAL_DIR}.")

    data = json.loads(cj.read_text(encoding="utf-8"))
    reviewed = [im for im in data["images"] if im.get("reviewed")]
    if not reviewed:
        raise SystemExit("Chua co anh nao duoc danh dau 'reviewed'. Mo annotate.html, bam R tren cac anh da duyet.")

    # Tao tap val loc (chi anh reviewed) voi nhan tay
    vdir = EVAL_DIR / "manual_val"
    img_d = vdir / "images" / "val"
    lbl_d = vdir / "labels" / "val"
    for d in (img_d, lbl_d):
        if d.exists():
            shutil.rmtree(d)
        d.mkdir(parents=True, exist_ok=True)

    n_box = 0
    cls_box = {i: 0 for i in range(5)}
    for im in reviewed:
        stem = Path(im["file"]).stem
        src_img = EVAL_DIR / "images" / "val" / im["file"]
        shutil.copy(src_img, img_d / im["file"])
        lines = []
        for b in im["boxes"]:
            c = int(b["c"])
            lines.append(f"{c} {b['x']:.6f} {b['y']:.6f} {b['w']:.6f} {b['h']:.6f}")
            n_box += 1
            cls_box[c] += 1
        (lbl_d / f"{stem}.txt").write_text("\n".join(lines))

    yaml_p = vdir / "manual.yaml"
    yaml_p.write_text(
        f"path: {vdir.resolve()}\ntrain: images/val\nval: images/val\n\n"
        f"nc: 5\nnames: {NAMES}\n")

    print(f"Tap test nhan tay: {len(reviewed)} anh, {n_box} box")
    print("  Phan bo lop:", {NAMES[i]: cls_box[i] for i in range(5)})
    print(f"Model do: {args.weights}\n")

    model = YOLO(args.weights)
    m = model.val(data=str(yaml_p), imgsz=640, batch=8, device="cpu",
                  conf=args.conf, project="runs/detect",
                  name="val_" + args.name, exist_ok=True, plots=True)

    result = {
        "eval_set": args.name,
        "note": args.note,
        "weights": str(Path(args.weights).name),
        "n_images": len(reviewed),
        "n_boxes": n_box,
        "class_box_counts": {NAMES[i]: cls_box[i] for i in range(5)},
        "mAP50": round(float(m.box.map50), 4),
        "mAP50_95": round(float(m.box.map), 4),
        "precision": round(float(m.box.mp), 4),
        "recall": round(float(m.box.mr), 4),
        "per_class": {},
    }
    # ap50/ap chi co hang cho lop XUAT HIEN trong tap; map dung bang ap_class_index
    ap_idx = [int(c) for c in m.box.ap_class_index]
    for row, cid in enumerate(ap_idx):
        result["per_class"][NAMES[cid]] = {"mAP50": round(float(m.box.ap50[row]), 4),
                                           "mAP50_95": round(float(m.box.ap[row]), 4)}

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "metrics.json").write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"\n===== KET QUA (nhan tay) — {args.name} =====")
    print(f"  Anh / box : {len(reviewed)} / {n_box}")
    print(f"  mAP50     : {result['mAP50']:.4f}")
    print(f"  mAP50-95  : {result['mAP50_95']:.4f}")
    print(f"  Precision : {result['precision']:.4f}   Recall: {result['recall']:.4f}")
    print("  Per-class mAP50 (chi lop co mat):")
    for i in range(5):
        if NAMES[i] in result["per_class"]:
            print(f"    {NAMES[i]:<12}: {result['per_class'][NAMES[i]]['mAP50']:.4f}")

    # copy plots
    import os
    rn = "val_" + args.name
    for base in (Path("runs/detect") / rn,
                 Path.home() / "runs/detect/runs/detect" / rn):
        if base.exists():
            for fn in ("confusion_matrix.png", "confusion_matrix_normalized.png",
                       "BoxPR_curve.png", "BoxF1_curve.png", "val_batch0_pred.jpg",
                       "val_batch0_labels.jpg"):
                if (base / fn).exists():
                    shutil.copy(base / fn, OUT_DIR / fn)
            break
    print(f"\nDa luu: {OUT_DIR / 'metrics.json'} + plots")


if __name__ == "__main__":
    main()
