"""
Đánh giá yolo11s_vn.pt trên 800 ảnh ngẫu nhiên từ auto_label (CCTV Hà Nội thực tế).
Vì không có val split gốc từ lúc train, dùng random sample làm proxy test set.

Run từ project root:
  custom_tracking_system/venv_tracking/Scripts/python \
      custom_tracking_system/scripts/eval_yolo11s_autolabel.py
"""
import random
import shutil
import yaml
from pathlib import Path
from ultralytics import YOLO

SEED        = 42
N_SAMPLES   = 800
AUTO_LABEL  = Path("custom_tracking_system/data/auto_label")
EVAL_DIR    = Path("custom_tracking_system/data/auto_label_eval")
OUT_DIR     = Path("docs/assets/yolo11s_vn")
WEIGHTS     = Path("custom_tracking_system/weights/yolo11s_vn.pt")
RUN_NAME    = "eval_yolo11s_autolabel"

# --- Tạo eval dataset ---
random.seed(SEED)
all_imgs = sorted((AUTO_LABEL / "images").glob("*.jpg"))
sampled  = random.sample(all_imgs, min(N_SAMPLES, len(all_imgs)))

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

print(f"Eval set: {copied} images copied to {EVAL_DIR}")

# --- Ghi YAML ---
yaml_path = EVAL_DIR / "eval.yaml"
yaml_path.write_text(yaml.dump({
    "path": str(EVAL_DIR.resolve()),
    "train": "images/val",   # placeholder, không dùng
    "val":   "images/val",
    "nc": 5,
    "names": ["person", "car", "motorcycle", "bus", "truck"],
}))

# --- Chạy validation ---
model = YOLO(str(WEIGHTS))
metrics = model.val(
    data=str(yaml_path),
    imgsz=640,
    batch=8,
    device="cpu",
    project="runs/detect",
    name=RUN_NAME,
    exist_ok=True,
    plots=True,
)

# --- In kết quả ---
names = ["person", "car", "motorcycle", "bus", "truck"]
print("\n========== YOLO11s-VN — Auto-label CCTV Hà Nội (800 ảnh) ==========")
print(f"mAP50       : {metrics.box.map50:.4f}")
print(f"mAP50-95    : {metrics.box.map:.4f}")
print(f"Precision   : {metrics.box.mp:.4f}")
print(f"Recall      : {metrics.box.mr:.4f}")
print("\nPer-class:")
for i, (ap50, ap) in enumerate(zip(metrics.box.ap50, metrics.box.ap)):
    print(f"  {names[i]:<12}: mAP50={ap50:.4f}  mAP50-95={ap:.4f}")

# --- Copy plots ---
OUT_DIR.mkdir(parents=True, exist_ok=True)
run_dir = Path("runs/detect") / RUN_NAME
# ultralytics đôi khi lưu vào C:/Users/PC/runs/...
alt_dir = Path.home() / "runs/detect" / "runs/detect" / RUN_NAME
src_dir = run_dir if run_dir.exists() else alt_dir

plot_names = [
    "results.png", "results.csv",
    "confusion_matrix.png", "confusion_matrix_normalized.png",
    "BoxPR_curve.png", "BoxF1_curve.png", "BoxP_curve.png", "BoxR_curve.png",
    "val_batch0_labels.jpg", "val_batch0_pred.jpg",
]
for fname in plot_names:
    src = src_dir / fname
    if src.exists():
        shutil.copy(src, OUT_DIR / fname)
        print(f"Saved: {OUT_DIR / fname}")
    else:
        # thử tên không có Box prefix
        alt = src_dir / fname.replace("Box", "")
        if alt.exists():
            shutil.copy(alt, OUT_DIR / fname)
            print(f"Saved (alt name): {OUT_DIR / fname}")

print(f"\nPlots → {OUT_DIR}")
