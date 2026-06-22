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

## 2. OSNet ReID Fine-tune — dataset đã sẵn sàng local, đang chờ chạy

**Mục đích**: Fine-tune OSNet trên VeRi-776 (vehicle ReID) để cải thiện cross-camera matching accuracy cho xe cộ Việt Nam.

**Không cần Kaggle nữa** — dataset VeRi-776 đã upload sẵn vào máy local tại `custom_tracking_system/data/datasets/VeRi/` (37,782 ảnh train, verify khớp `train_label.xml`). Sẽ chạy bằng GPU local (RTX 4060) ngay sau khi job YOLO ở mục 1 giải phóng GPU.

**KHÔNG cần** copy `weights/osnet_veri776.pth` — đó là file output của lần train trước (mặc định `--out`), không phải input. Script tự khởi tạo model từ **ImageNet-pretrained** (`torchreid.models.build_model(..., pretrained=True)`).

**Cài torchreid** (đã cài sẵn trong `venv_tracking`, version 0.2.5):
```bash
pip install git+https://github.com/KaiyangZhou/deep-person-reid.git
```

**Lệnh chạy** (đã verify đúng args thật từ `argparse` trong script):
```bash
python train_veri.py \
    --data data/datasets/VeRi \
    --out weights/osnet_veri776_v2.pth \
    --epochs 60 \
    --batch 16 \
    --workers 0
```

> ⚠️ **`--workers 0` là bắt buộc trên máy này**, không phải 4 hoặc default. Máy có 20GB RAM + 8GB pagefile, commit charge thường xuyên ở mức ~29.6/30GB. Mỗi dataloader worker trên Windows phải re-import toàn bộ torch+CUDA (multiprocessing spawn re-run cả script), nên `--workers ≥ 1` đã từng gây crash `OSError: [WinError 1455] The paging file is too small...` / `[WinError 1114] DLL initialization routine failed` ngay cả khi GPU còn nhiều VRAM trống. batch giảm từ 64 → 16 vì lúc chạy đồng thời với job YOLO chỉ còn ~2.4GB VRAM trống (xem [[feedback_autonomous_training_monitoring]]).

Args đầy đủ của script: `--data` (bắt buộc, path tới folder VeRi), `--out` (default `weights/osnet_veri776.pth`), `--epochs` (default 20), `--batch` (default 32), `--lr` (default 0.0003), `--workers` (default 4 — **đặt 0 trên máy này**), `--resume` (checkpoint để tiếp tục train nếu bị ngắt).

**Output cần lấy về:**
- `weights/osnet_veri776_v2.pth` — checkpoint đã fine-tune
- Log training in ra Rank-1 accuracy / loss theo epoch (script tự in ra stdout, lưu lại log file)

**Số liệu cần ghi vào báo cáo:**
- Rank-1 accuracy (val_acc do script tự tính, dùng làm proxy)
- Loss theo epoch

**Ước tính thời gian:** 2–3 giờ trên GPU RTX 4060 (60 epoch, ~38K ảnh, batch 16, workers 0 — chậm hơn ước tính gốc do single-process data loading).

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

## Checklist chạy local (OSNet — còn lại)

- [x] Dataset `custom_tracking_system/data/datasets/VeRi/` đã có sẵn local
- [x] torchreid đã cài trong `venv_tracking`
- [ ] Chờ GPU rảnh (job YOLO mục 1 dùng GPU xong)
- [ ] Chạy với `--workers 0` (xem cảnh báo virtual-memory ở trên), theo dõi log để bắt sớm nếu treo

---

## Sau khi có kết quả

Cập nhật các chỗ sau trong [bao_cao_cuoi_cung.md](../bao_cao_cuoi_cung.md):

| Vị trí trong báo cáo | Cần điền | Trạng thái |
|----------------------|---------|---|
| 4.4 / 5.4 | mAP50, mAP50-95, Precision, Recall của YOLO + cảnh báo catastrophic forgetting | ✅ đã cập nhật |
| Chương 5, mục 5.x | Rank-1 accuracy OSNet cross-camera | ⏳ chờ train OSNet |
| Danh mục hình | Thêm: training loss curve, confusion matrix, PR curve YOLO | ✅ đã lưu vào `docs/assets/yolo11s_carla_v2/` |
| Bảng so sánh | Thêm cột "YOLO11s carla_v2 fine-tuned" vs "YOLO11s_vn" — lưu ý 2 model phục vụ 2 mục đích khác nhau, không phải bản thay thế | ✅ đã cập nhật |
