# Plan: Fine-tune YOLO11s + OSNet ReID trên máy GPU cloud

## Bối cảnh

Đây là project AI giám sát giao thông (CARLA simulation + CCTV thật, Hà Nội). Có 2 model cần fine-tune trên GPU cloud (máy local không đủ VRAM):

1. **YOLO11s detector** — đang bị domain gap nặng trên ảnh CARLA: MOTA = **-23.6%**, recall chỉ **3.96%** khi chạy trên video render từ CARLA (model hiện tại `yolo11s_vn.pt` chỉ được fine-tune trên ảnh CCTV thật, chưa thấy ảnh CARLA). Cần fine-tune tiếp (không train from scratch) trên dataset ảnh CARLA mới.
2. **OSNet ReID** — chưa có Rank-1 accuracy nào được đo cho vehicle re-identification xuyên camera. Cần fine-tune trên VeRi-776.

Tất cả file cần đã có sẵn trong project, chỉ cần copy đúng đường dẫn sang máy cloud.

---

## Việc 1: Fine-tune YOLO11s (ưu tiên cao nhất)

### File cần copy sang máy cloud

| File/folder | Vị trí trên máy local | Dung lượng |
|---|---|---|
| Dataset | `custom_tracking_system/data/carla_cam_det_v2/` (có sẵn `images/train`, `images/val`, `labels/train`, `labels/val`) | 4.0GB |
| Config | `custom_tracking_system/data/carla_cam_det_v2.yaml` | nhỏ |
| Checkpoint xuất phát | `custom_tracking_system/weights/yolo11s_vn.pt` | 19MB |

### ⚠️ Bắt buộc sửa trước khi train

File `carla_cam_det_v2.yaml` đang chứa đường dẫn tuyệt đối Windows:
```yaml
path: E:/School_project/finalproject/custom_tracking_system/data/carla_cam_det_v2
train: images/train
val:   images/val
nc: 5
names: ['person', 'car', 'motorcycle', 'bus', 'truck']
```
Phải sửa dòng `path:` cho khớp vị trí thật của dataset trên máy cloud (vd `/workspace/carla_cam_det_v2`), nếu không training sẽ lỗi không tìm thấy ảnh.

### ⚠️ Cảnh báo dữ liệu (đã verify bằng cách đếm trực tiếp nhãn)

Dataset này **KHÔNG có một box "person" hay "bus" nào** (class 0 và class 3 = 0 mẫu trong cả train và val, đã đếm toàn bộ 117,489 box). Chỉ có car (67.8%), motorcycle (17.8%), truck (14.4%). Fine-tune 30 epoch trên data này có rủi ro làm model **quên khả năng nhận diện person/bus** (catastrophic forgetting). Chấp nhận rủi ro này có chủ đích vì mục tiêu chính là fix domain gap car/motorcycle/truck. **Sau khi train xong, phải đo lại riêng accuracy person/bus để xác nhận có bị suy giảm không.**

### Lệnh chạy

```bash
pip install ultralytics
```

```python
from ultralytics import YOLO

model = YOLO('yolo11s_vn.pt')   # checkpoint xuất phát, KHÔNG train from scratch
model.train(
    data='carla_cam_det_v2.yaml',
    epochs=30,
    batch=16,        # giảm xuống 8 nếu GPU <8GB VRAM
    imgsz=960,
    device=0,
    half=True,
    name='yolo11s_carla_v2',
)
```

Thời gian ước tính: 45-90 phút trên GPU T4.

### Output cần lấy về

- `runs/detect/yolo11s_carla_v2/weights/best.pt` → sẽ thay cho `custom_tracking_system/weights/yolo11s_vn.pt`
- `runs/detect/yolo11s_carla_v2/results.csv` — mAP50, mAP50-95, precision/recall theo epoch
- `runs/detect/yolo11s_carla_v2/confusion_matrix_normalized.png`
- `runs/detect/yolo11s_carla_v2/PR_curve.png`

---

## Việc 2: Fine-tune OSNet ReID

### File cần copy sang máy cloud

| File/folder | Vị trí trên máy local | Dung lượng |
|---|---|---|
| Dataset VeRi-776 | `custom_tracking_system/data/datasets/VeRi/` (chứa `image_train/`, `train_label.xml`, ...) | 1.1GB (37,778 ảnh train) |
| Script train | `custom_tracking_system/train_veri.py` | nhỏ |

**KHÔNG cần** copy `weights/osnet_veri776.pth` — đó là file output của lần train trước (mặc định `--out`), không phải input. Script tự khởi tạo model từ **ImageNet-pretrained** (`torchreid.models.build_model(..., pretrained=True)`).

### Cài torchreid (không cần copy folder `deep-person-reid/`)

```bash
pip install git+https://github.com/KaiyangZhou/deep-person-reid.git
```

### Lệnh chạy (đã verify đúng args thật từ `argparse` trong script — KHÁC với args ghi trong `docs/gpu_cloud_tasks.md` cũ, bản cũ bị sai)

```bash
python train_veri.py \
    --data /path/to/VeRi \
    --out weights/osnet_veri776_v2.pth \
    --epochs 60 \
    --batch 64
```

Args đầy đủ của script: `--data` (bắt buộc, path tới folder VeRi), `--out` (default `weights/osnet_veri776.pth`), `--epochs` (default 20), `--batch` (default 32), `--lr` (default 0.0003), `--workers` (default 4), `--resume` (checkpoint để tiếp tục train nếu bị ngắt).

Thời gian ước tính: 2-3 giờ trên GPU T4 (60 epoch, ~38K ảnh).

### Output cần lấy về

- `weights/osnet_veri776_v2.pth` — checkpoint đã fine-tune
- Log training in ra Rank-1 accuracy / loss theo epoch (script tự in ra stdout, lưu lại log file)

---

## Sau khi xong cả 2 việc

1. Copy `best.pt` (YOLO) → `custom_tracking_system/weights/yolo11s_vn.pt`
2. Copy `osnet_veri776_v2.pth` → `custom_tracking_system/weights/`
3. Chạy lại `custom_tracking_system/evaluate_tracking.py` (MOTA/IDF1) với detector mới — hiện đang -23.6%, kỳ vọng chuyển dương sau fine-tune
4. Đo riêng accuracy person/bus của YOLO mới (cảnh báo catastrophic forgetting ở trên)
5. Ghi Rank-1 accuracy của OSNet vào báo cáo
