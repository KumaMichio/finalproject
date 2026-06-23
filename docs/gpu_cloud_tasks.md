# Nhiệm vụ cần GPU Cloud

> Cập nhật: 2026-06-22  
> Platform khuyến nghị: Kaggle (T4 x2, miễn phí 30h/tuần) hoặc Google Colab Pro — nhưng xem lưu ý dưới, 2 task đã chạy được trực tiếp trên GPU local (RTX 4060 8GB) nên không còn cần cloud nữa.

---

## ✅ 1. YOLO11s Fine-tune vòng 2 — carla_cam_det_v2 — Đã hoàn thành (2026-06-22)

**Mục đích:** Fine-tune lại `yolo11s_vn.pt` trên `data/carla_cam_det_v2/` để xóa domain gap CARLA (MOTA hiện tại = −23.6%, Recall = 3.96%).

**Chạy local** (không cần Kaggle — dataset 14,791 ảnh train + 1,456 val đã có sẵn trong `custom_tracking_system/data/carla_cam_det_v2/`):
```bash
python scripts/train/train_yolo_detector.py --data carla_cam_det_v2.yaml \
    --model weights/yolo11s_vn.pt --epochs 30 --imgsz 960 --batch 8 \
    --device 0 --half --out weights/yolo11s_carla_v2.pt --name yolo11s_carla_v2
```
batch=8 (không phải 16) vì RTX 4060 chỉ có 8GB VRAM. Job bị treo (hang) một lần ở epoch 20→21 do hết "virtual memory commit limit" của Windows khi dataloader rebuild worker cho giai đoạn `close_mosaic` — đã resume thành công từ checkpoint `last.pt` với `workers=2` (xem [[feedback_autonomous_training_monitoring]]).

**Kết quả trên val set carla_cam_det_v2** (epoch 30/30):

| Metric | Giá trị |
|---|---|
| mAP50 | **0.805** |
| mAP50-95 | 0.696 |
| Precision | 0.897 |
| Recall | 0.749 |

Per-class mAP50: car=0.904, motorcycle=0.539, truck=0.972 (dataset này không có nhãn person/bus — xem cảnh báo dưới).

Charts lưu tại `docs/assets/yolo11s_carla_v2/` (`results.png`, `results.csv`, `confusion_matrix_normalized.png`, `BoxPR_curve.png`).

> ⚠️ **Phát hiện quan trọng — catastrophic forgetting nặng hơn dự kiến:** Đo lại model trên `data/visdrone_carla_vn.yaml` (val set có đủ 5 class, domain VisDrone + CARLA CCTV thật) thì model mới **mất khả năng tổng quát hóa gần như hoàn toàn**, không chỉ với person/bus mà cả car/motorcycle/truck:
>
> | Class | Baseline `yolo11s_vn` mAP50 | Mới `yolo11s_carla_v2` mAP50 |
> |---|---|---|
> | all | 0.507 | 0.006 |
> | person | 0.433 | 0.000 |
> | car | 0.800 | 0.029 |
> | motorcycle | 0.448 | 0.00002 |
> | bus | 0.503 | 0.000 |
> | truck | 0.349 | 0.0004 |
>
> Log đầy đủ: `docs/assets/yolo11s_carla_v2/forgetting_check_vs_visdrone_carla_vn.txt`. Nguyên nhân: 30 epoch, backbone không freeze, dataset train 100% một domain hẹp (CARLA Town10HD, camera TL-mounted cố định, chỉ 3 class) → model overfit nặng vào style hình ảnh CARLA, mất khả năng nhận diện domain khác.
>
> **Quyết định:** Giữ **2 checkpoint riêng biệt**, không overwrite `weights/yolo11s_vn.pt`:
> - `weights/yolo11s_vn.pt` — model tổng quát, dùng cho production/real CCTV (giữ nguyên, không đổi)
> - `weights/yolo11s_carla_v2.pt` — model chuyên biệt domain CARLA, chỉ dùng khi đánh giá/test trên video CARLA simulation (vd đo MOTA)
>
> Cách dùng cho pipeline CARLA: `python main.py --detector weights/yolo11s_carla_v2.pt ...` (flag `--detector` override đã có sẵn trong `main.py`, không cần sửa code).

---

## ✅ 2. OSNet ReID Fine-tune — Đã hoàn thành (2026-06-23)

**Mục đích**: Fine-tune OSNet trên VeRi-776 (vehicle ReID) để cải thiện cross-camera matching accuracy cho xe cộ Việt Nam.

**Chạy local** (GPU RTX 4060, không cần Kaggle — dataset `custom_tracking_system/data/datasets/VeRi/`, 37,782 ảnh train):
```bash
python scripts/train/train_veri.py \
    --data data/datasets/VeRi \
    --out weights/osnet_veri776_v2.pth \
    --epochs 60 \
    --batch 16 \
    --workers 0
```
60 epoch hoàn thành, train_acc/val_acc đạt ~100%/99.9% ở epoch cuối, loss giảm đều không có dấu hiệu phân kỳ (log đầy đủ: `docs/assets/osnet_veri776_v2/results.csv`). 60 checkpoint trung gian lưu tại `weights/osnet_veri776_v2.ckpt_ep01.pth` … `ep60.pth` (đã gitignore, chỉ giữ local).

**Đánh giá trên test set chuẩn VeRi-776** (`scripts/eval/eval_veri_reid.py`, giao thức market1501-style — loại gallery cùng ID+cùng camera với query trước khi rank):
```bash
python scripts/eval/eval_veri_reid.py --data data/datasets/VeRi --weights weights/osnet_veri776_v2.pth \
    --out ../docs/assets/osnet_veri776_v2/reid_eval_veri_test.json
```

| Metric | Giá trị |
|---|---|
| Query / Gallery | 1.678 / 11.579 ảnh |
| mAP | **71,89%** |
| Rank-1 | **94,16%** |
| Rank-5 | 96,84% |
| Rank-10 | 98,15% |
| Rank-20 | 99,05% |

Kết quả lưu tại `docs/assets/osnet_veri776_v2/` (`reid_eval_veri_test.json`, `cmc_curve.png`, `results.csv`/`results.png` — training curve).

> ✅ **Đã wire vào production**: `custom_tracking_system/modules/reid.py` (default `vehicle_weights`) và `server/services/ai_processor.py` đã chuyển từ checkpoint cũ `osnet_veri776.pth` (epoch 12, chưa từng benchmark) sang `osnet_veri776_v2.pth`. `main.py` (chế độ chạy trực tiếp / OpenCV) **đã đổi sang `DualReIDExtractor`** (2026-06-23), cùng cách dùng như `ai_processor.py` — vehicle Re-ID giờ hoạt động ở cả 2 đường chạy.
>
> **Lưu ý cho báo cáo**: đây là benchmark retrieval offline trên ảnh crop chuẩn hoá VeRi-776, đo chất lượng embedding model. Chưa đo matching accuracy end-to-end trong pipeline thật (ảnh crop từ video CCTV/CARLA, có nhiễu detection/tracking) — xem `bao_cao_cuoi_cung.md` mục 5.3 và 6.2.

---

## ✅ YOLO11s — Đã hoàn thành (2026-06-18)

`weights/yolo11s_vn.pt` (19MB) đã train xong. Val trên `data/auto_label_eval/` (800 ảnh, auto-labels):

| Metric | Giá trị |
|---|---|
| mAP50 | **89.5%** |
| mAP50-95 | **77.0%** |
| Precision | 86.7% |
| Recall | 81.3% |

Per-class mAP50: motorcycle=97.5%, car=98.1%, bus=89.3%, person=82.8%, truck=79.7%

> ⚠️ Val set dùng auto-labels (không phải ground truth tay) → số liệu cao hơn thực tế.
> Charts lưu tại `docs/assets/yolo11s_vn/`.

## Checklist chạy local (OSNet) — ✅ hoàn thành 2026-06-23

- [x] Dataset `custom_tracking_system/data/datasets/VeRi/` đã có sẵn local
- [x] torchreid đã cài trong `venv_tracking`
- [x] Train 60 epoch với `--workers 0` — không treo, hoàn thành trong thời gian ước tính
- [x] Eval VeRi-776 test chuẩn (`scripts/eval/eval_veri_reid.py`) — mAP=71.89%, Rank-1=94.16%
- [x] Wire vào production (`modules/reid.py`, `server/services/ai_processor.py` → `osnet_veri776_v2.pth`)
- [x] `main.py` (direct-run mode) đổi sang `DualReIDExtractor` (2026-06-23) — vehicle Re-ID hoạt động ở cả 2 đường chạy
- [ ] Benchmark end-to-end cross-camera trong pipeline thật (không chỉ offline VeRi-776 retrieval) — chưa làm

---

## Sau khi có kết quả

Cập nhật các chỗ sau trong [bao_cao_cuoi_cung.md](../bao_cao_cuoi_cung.md):

| Vị trí trong báo cáo | Cần điền | Trạng thái |
|----------------------|---------|---|
| 4.4 / 5.4 | mAP50, mAP50-95, Precision, Recall của YOLO + cảnh báo catastrophic forgetting | ✅ đã cập nhật |
| 4.5 / 5.3 | mAP, Rank-1 accuracy OSNet ReID trên VeRi-776 | ✅ đã cập nhật |
| Danh mục hình | Thêm: training loss curve, confusion matrix, PR curve YOLO | ✅ đã lưu vào `docs/assets/yolo11s_carla_v2/` |
| Danh mục hình | Thêm: Hình 5.4 CMC curve OSNet | ✅ đã cập nhật |
| Danh mục bảng | Thêm: Bảng 5.4 kết quả OSNet, đổi Test Cases thành Bảng 5.5 | ✅ đã cập nhật |
| Bảng so sánh | Thêm cột "YOLO11s carla_v2 fine-tuned" vs "YOLO11s_vn" — lưu ý 2 model phục vụ 2 mục đích khác nhau, không phải bản thay thế | ✅ đã cập nhật |
