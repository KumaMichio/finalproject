"""
Đánh giá yolo11s_vn_plate.pt (Stage 1: + license_plate class) theo 2 phần:

  A. Catastrophic-forgetting check: per-class mAP50 của 5 class cũ
     (person/car/motorcycle/bus/truck) trên đúng eval set 800 ảnh dùng để
     đo baseline yolo11s_vn.pt (mAP50=89.5%, mAP50-95=77.0%, xem
     docs/chuong4_phan_tich_thiet_ke_trien_khai_danh_gia.md) — cùng SEED=42
     nên ra đúng 800 ảnh giống lần đo baseline.
  B. license_plate mAP: val split riêng của data/yolo11s_vn_plate_merged
     (2,437 ảnh, held-out thật khỏi training, gồm cả ảnh plate).

Run từ project root:
  custom_tracking_system/venv_tracking/Scripts/python \
      custom_tracking_system/scripts/eval_yolo11s_vn_plate.py
"""
import random
import shutil
import yaml
from pathlib import Path
from ultralytics import YOLO

SEED       = 42
N_SAMPLES  = 800
AUTO_LABEL = Path("custom_tracking_system/data/auto_label")
EVAL_DIR   = Path("custom_tracking_system/data/auto_label_eval")
OUT_DIR    = Path("docs/assets/yolo11s_vn_plate")
WEIGHTS    = Path("custom_tracking_system/weights/yolo11s_vn_plate.pt")
NAMES6     = ["person", "car", "motorcycle", "bus", "truck", "license_plate"]


def main():
    # --- Part A: recreate the exact 800-image baseline eval set ---
    random.seed(SEED)
    all_imgs = sorted((AUTO_LABEL / "images").glob("*.jpg"))
    sampled = random.sample(all_imgs, min(N_SAMPLES, len(all_imgs)))

    img_dir = EVAL_DIR / "images" / "val"
    lbl_dir = EVAL_DIR / "labels" / "val"
    img_dir.mkdir(parents=True, exist_ok=True)
    lbl_dir.mkdir(parents=True, exist_ok=True)

    copied = 0
    for img_path in sampled:
        lbl_path = AUTO_LABEL / "labels" / (img_path.stem + ".txt")
        if lbl_path.exists():
            shutil.copy(img_path, img_dir / img_path.name)
            shutil.copy(lbl_path, lbl_dir / lbl_path.name)
            copied += 1
    print(f"[A] Baseline-comparable eval set: {copied} images copied to {EVAL_DIR}")

    eval_yaml = EVAL_DIR / "eval6.yaml"
    eval_yaml.write_text(yaml.dump({
        "path": str(EVAL_DIR.resolve()),
        "train": "images/val",
        "val": "images/val",
        "nc": 6,
        "names": NAMES6,
    }))

    model = YOLO(str(WEIGHTS))

    metrics_a = model.val(
        data=str(eval_yaml), imgsz=640, batch=8, device=0, workers=0,
        project="runs/detect", name="eval_yolo11s_vn_plate_part_a", exist_ok=True, plots=True,
    )

    print("\n========== PART A — 5 class cu, 800 anh auto_label (so baseline 89.5%/77.0%) ==========")
    print(f"Overall mAP50   : {metrics_a.box.map50:.4f}")
    print(f"Overall mAP50-95: {metrics_a.box.map:.4f}")
    print(f"Precision       : {metrics_a.box.mp:.4f}")
    print(f"Recall          : {metrics_a.box.mr:.4f}")
    print("\nPer-class (idx = class trong metrics.box.ap_class_index):")
    for idx, ap50, ap in zip(metrics_a.box.ap_class_index, metrics_a.box.ap50, metrics_a.box.ap):
        cname = NAMES6[int(idx)]
        drop = 89.5 - ap50 * 100
        flag = "  <<< CATASTROPHIC FORGETTING?" if drop > 10 else ""
        print(f"  {cname:<14}: mAP50={ap50*100:.2f}%  mAP50-95={ap*100:.2f}%  (delta vs 89.5% baseline: {-drop:+.2f}pp){flag}")

    # --- Part B: license_plate on merged held-out val split ---
    metrics_b = model.val(
        data="custom_tracking_system/data/yolo11s_vn_plate_merged.yaml",
        imgsz=640, batch=16, device=0, workers=0,
        project="runs/detect", name="eval_yolo11s_vn_plate_part_b", exist_ok=True, plots=True,
    )

    print("\n========== PART B — toan bo 6 class, val split merged (2,437 anh, held-out) ==========")
    print(f"Overall mAP50   : {metrics_b.box.map50:.4f}")
    print(f"Overall mAP50-95: {metrics_b.box.map:.4f}")
    print("\nPer-class:")
    for idx, ap50, ap in zip(metrics_b.box.ap_class_index, metrics_b.box.ap50, metrics_b.box.ap):
        cname = NAMES6[int(idx)]
        print(f"  {cname:<14}: mAP50={ap50*100:.2f}%  mAP50-95={ap*100:.2f}%")

    # --- Copy plots from Part B (full 6-class, held-out — the more meaningful one) ---
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    src_dir = Path("runs/detect/runs/detect/eval_yolo11s_vn_plate_part_b")
    plot_names = [
        "results.csv", "confusion_matrix.png", "confusion_matrix_normalized.png",
        "BoxPR_curve.png", "BoxF1_curve.png", "BoxP_curve.png", "BoxR_curve.png",
        "val_batch0_labels.jpg", "val_batch0_pred.jpg",
    ]
    for fname in plot_names:
        src = src_dir / fname
        if src.exists():
            shutil.copy(src, OUT_DIR / fname)
            print(f"Saved: {OUT_DIR / fname}")

    # Also pull the training-run plots (loss curves over 30 epochs)
    train_run_dir = Path("runs/detect/runs/detect/yolo11s_vn_plate")
    for fname in ["results.png", "results.csv"]:
        src = train_run_dir / fname
        if src.exists():
            dest = OUT_DIR / f"train_{fname}"
            shutil.copy(src, dest)
            print(f"Saved: {dest}")

    print(f"\nPlots -> {OUT_DIR}")


if __name__ == "__main__":
    main()
