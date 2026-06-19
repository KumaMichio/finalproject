"""
Validate yolo11s_vn.pt on carla_cam_det val set and save metrics + plots.
Run from project root: custom_tracking_system/venv_tracking/Scripts/python custom_tracking_system/scripts/eval_yolo11s.py
"""
import shutil
from pathlib import Path
from ultralytics import YOLO

WEIGHTS   = Path("custom_tracking_system/weights/yolo11s_vn.pt")
DATA_YAML = Path("custom_tracking_system/data/carla_cam_det.yaml")
OUT_DIR   = Path("docs/assets/yolo11s_vn")
RUN_DIR   = Path("runs/detect/eval_yolo11s_vn")

model = YOLO(str(WEIGHTS))

metrics = model.val(
    data=str(DATA_YAML),
    imgsz=640,
    batch=8,
    device="cpu",       # đổi thành 0 nếu có GPU
    project="runs/detect",
    name="eval_yolo11s_vn",
    exist_ok=True,
    plots=True,
    save_json=False,
)

# In số liệu tổng hợp
print("\n========== YOLO11s-VN Validation Results ==========")
print(f"mAP50       : {metrics.box.map50:.4f}")
print(f"mAP50-95    : {metrics.box.map:.4f}")
print(f"Precision   : {metrics.box.mp:.4f}")
print(f"Recall      : {metrics.box.mr:.4f}")
print("\nPer-class mAP50:")
names = ["person", "car", "motorcycle", "bus", "truck"]
for i, (ap50, ap) in enumerate(zip(metrics.box.ap50, metrics.box.ap)):
    print(f"  {names[i]:<12}: mAP50={ap50:.4f}  mAP50-95={ap:.4f}")

# Copy plots sang docs/assets/yolo11s_vn/
OUT_DIR.mkdir(parents=True, exist_ok=True)
plots = [
    "results.png",
    "confusion_matrix.png",
    "confusion_matrix_normalized.png",
    "PR_curve.png",
    "F1_curve.png",
    "P_curve.png",
    "R_curve.png",
    "results.csv",
]
for fname in plots:
    src = RUN_DIR / fname
    if src.exists():
        shutil.copy(src, OUT_DIR / fname)
        print(f"Saved: {OUT_DIR / fname}")
    else:
        print(f"Not found (may have different name): {src}")

print(f"\nAll plots in: {OUT_DIR}")
