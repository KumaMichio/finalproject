# Nhiệm vụ cần GPU Cloud

> Cập nhật: 2026-06-17  
> Platform khuyến nghị: Kaggle (T4 x2, miễn phí 30h/tuần) hoặc Google Colab Pro

---

## 1. YOLO11m Fine-tune

**Mục đích**: Fine-tune YOLO11m pretrained trên dataset giao thông Việt Nam để phát hiện 5 lớp: `person, car, motorcycle, bus, truck`.

**Dữ liệu đầu vào** (chuẩn bị local trước, upload lên Kaggle):
- `data/auto_label/` — 15.957 ảnh CARLA đã auto-label (cần QA trước)
- `data/visdrone_vn/` — VisDrone dataset lọc lấy 5 class (chạy `prepare_visdrone_dataset.py`)
- Merge: `python merge_detection_datasets.py --datasets visdrone=data/visdrone_vn carla=data/auto_label --out data/visdrone_carla_vn`

**Lệnh chạy trên Kaggle notebook:**
```python
# Cell 1 — cài dependencies
!pip install ultralytics -q

# Cell 2 — train
!python custom_tracking_system/scripts/train/train_yolo_detector.py \
    --data data/visdrone_carla_vn.yaml \
    --model yolo11m.pt \
    --epochs 50 \
    --batch 16 \
    --device 0 \
    --half \
    --out weights/yolo11m_vn.pt

# Cell 3 — download kết quả
import shutil
shutil.make_archive('/kaggle/working/yolo_results', 'zip', 'runs/detect/yolo11m_vn')
```

**Output cần lấy về:**
- `results.png` — training curves (loss, mAP theo epoch)
- `confusion_matrix_normalized.png` — confusion matrix
- `PR_curve.png` — Precision-Recall curve
- `results.csv` — số liệu mAP50, mAP50-95 từng epoch
- `weights/best.pt` → copy thành `weights/yolo11m_vn.pt`

**Số liệu cần ghi vào báo cáo:**
- mAP50 (tổng 5 class)
- mAP50-95
- Precision / Recall trên val set
- mAP per-class (đặc biệt motorcycle)

**Ước tính thời gian:** ~1–2 giờ trên T4 (50 epochs, dataset ~16K ảnh)

---

## 2. OSNet ReID Fine-tune

**Mục đích**: Fine-tune OSNet trên VeRi-776 (vehicle ReID) để cải thiện cross-camera matching accuracy cho xe cộ Việt Nam.

**Dataset**: `VeRi/` (đã có trong project)

**Script**: `custom_tracking_system/train_veri.py`

**Lệnh chạy trên Kaggle:**
```python
# Cell 1
!pip install torchreid -q  # hoặc cài từ deep-person-reid/

# Cell 2
!python custom_tracking_system/train_veri.py \
    --root VeRi \
    --arch osnet_x1_0 \
    --batch-size 64 \
    --max-epoch 60 \
    --save-dir weights/osnet_veri

# Cell 3
import shutil
shutil.make_archive('/kaggle/working/reid_results', 'zip', 'weights/osnet_veri')
```

**Output cần lấy về:**
- `osnet_x1_0_veri.pth` — checkpoint tốt nhất
- Log training: Rank-1 accuracy trên VeRi val set

**Số liệu cần ghi vào báo cáo:**
- Rank-1 accuracy
- mAP (ReID metric)

**Ước tính thời gian:** ~2–3 giờ trên T4 (60 epochs, VeRi ~50K ảnh)

---

## Checklist trước khi lên Kaggle

- [ ] QA ảnh auto-label xong (lọc bbox < 0.4 confidence)
- [ ] Chạy `prepare_visdrone_dataset.py` để tạo `data/visdrone_vn/`
- [ ] Chạy `merge_detection_datasets.py` để tạo `data/visdrone_carla_vn/`
- [ ] Zip và upload dataset lên Kaggle Dataset
- [ ] Bật GPU T4 x2 trong Kaggle Settings → Accelerator
- [ ] `!pip install ultralytics -q` ở cell đầu tiên

---

## Sau khi có kết quả từ cloud

Cập nhật các chỗ sau trong [bao_cao_cuoi_cung.md](../bao_cao_cuoi_cung.md):

| Vị trí trong báo cáo | Cần điền |
|----------------------|---------|
| Chương 5, mục 5.x | mAP50, mAP50-95, Precision, Recall của YOLO |
| Chương 5, mục 5.x | Rank-1 accuracy OSNet cross-camera |
| Danh mục hình | Thêm: training loss curve, confusion matrix, PR curve |
| Bảng so sánh | Thêm cột "YOLO11m fine-tuned" vs "YOLO pretrained" |
