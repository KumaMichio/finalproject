"""
Danh gia plate_chars_detector.pt (Stage 2: 30 ky tu bien so VN) tren val split
cua data/plate_chars_merged (1,026 anh, real + synthetic).

Bao cao tong + per-class mAP, flag rieng cac class hiem trong du lieu thuc
(B, C, K, L, M, P, S, T, Z - chu yeu duoc synthetic data bu vao).

Run tu project root:
  custom_tracking_system/venv_tracking/Scripts/python \
      custom_tracking_system/scripts/eval_plate_chars_detector.py
"""
import shutil
from pathlib import Path
from ultralytics import YOLO

WEIGHTS = Path("custom_tracking_system/weights/plate_chars_detector.pt")
DATA = "custom_tracking_system/data/plate_chars_merged.yaml"
OUT_DIR = Path("docs/assets/plate_chars_detector")
RARE_REAL = {"B", "C", "K", "L", "M", "P", "S", "T", "Z"}
NAMES = ['1', '2', '3', '4', '5', '6', '7', '8', '9', 'A', 'B', 'C', 'D', 'E',
         'F', 'G', 'H', 'K', 'L', 'M', 'N', 'P', 'S', 'T', 'U', 'V', 'X', 'Y',
         'Z', '0']


def main():
    model = YOLO(str(WEIGHTS))
    metrics = model.val(
        data=DATA, imgsz=640, batch=32, device=0, workers=0,
        project="runs/detect", name="eval_plate_chars_detector", exist_ok=True, plots=True,
    )

    print("\n========== Stage 2 - plate_chars_detector (val split, 1,026 anh) ==========")
    print(f"Overall mAP50   : {metrics.box.map50*100:.2f}%")
    print(f"Overall mAP50-95: {metrics.box.map*100:.2f}%")
    print(f"Precision       : {metrics.box.mp*100:.2f}%")
    print(f"Recall          : {metrics.box.mr*100:.2f}%")

    print("\nPer-class:")
    rare_report = []
    for idx, ap50, ap in zip(metrics.box.ap_class_index, metrics.box.ap50, metrics.box.ap):
        cname = NAMES[int(idx)]
        tag = "  <<< RARE in real data (synthetic-backed)" if cname in RARE_REAL else ""
        print(f"  {cname:<4}: mAP50={ap50*100:6.2f}%  mAP50-95={ap*100:6.2f}%{tag}")
        if cname in RARE_REAL:
            rare_report.append((cname, ap50 * 100, ap * 100))

    print("\n-- Rare-class summary (B/C/K/L/M/P/S/T/Z) --")
    for cname, ap50, ap in sorted(rare_report, key=lambda x: x[1]):
        flag = "  <<< LOW" if ap50 < 70 else ""
        print(f"  {cname:<4}: mAP50={ap50:6.2f}%  mAP50-95={ap:6.2f}%{flag}")

    # --- Copy plots ---
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    src_dir = Path("runs/detect/runs/detect/eval_plate_chars_detector")
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

    # Also pull the 50-epoch training-run loss/mAP curves
    train_run_dir = Path("runs/detect/runs/detect/plate_chars_detector")
    for fname in ["results.png", "results.csv"]:
        src = train_run_dir / fname
        if src.exists():
            dest = OUT_DIR / f"train_{fname}"
            shutil.copy(src, dest)
            print(f"Saved: {dest}")

    print(f"\nPlots -> {OUT_DIR}")


if __name__ == "__main__":
    main()
