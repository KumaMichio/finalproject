# Execute Guide: Multi-Camera CCTV Tracking System

## Mục lục
1. [Yêu cầu hệ thống](#1-yêu-cầu-hệ-thống)
2. [Cài đặt môi trường](#2-cài-đặt-môi-trường)
3. [Dataset và Training](#3-dataset-và-training)
4. [Cách chạy project](#4-cách-chạy-project)
5. [Các kịch bản (Scenarios)](#5-các-kịch-bản-scenarios)
6. [Troubleshooting](#6-troubleshooting)

---

## 1. Yêu cầu hệ thống

| Thành phần | Yêu cầu tối thiểu | Khuyến nghị |
|-----------|-------------------|-------------|
| OS | Windows 10 64-bit | Windows 11 64-bit |
| CPU | Intel i5 / AMD Ryzen 5 | Intel i7 / AMD Ryzen 7 |
| RAM | 16 GB | 32 GB |
| GPU | NVIDIA GTX 1060 (6GB VRAM) | NVIDIA RTX 3070+ (8GB+ VRAM) |
| Disk | 50 GB trống | 100 GB SSD |
| Python | 3.7 | 3.7 (CARLA 0.9.9.4 yêu cầu chính xác Python 3.7) |
| CARLA | 0.9.9.4 | 0.9.9.4 |

---

## 2. Cài đặt môi trường

### Bước 1: Tạo môi trường Python 3.7

```bash
# Tạo virtual environment với Python 3.7
python3.7 -m venv venv_tracking

# Kích hoạt (Windows)
venv_tracking\Scripts\activate
```

### Bước 2: Cài đặt dependencies

```bash
cd e:\finalproject\PythonAPI\custom_tracking_system

pip install -r requirements.txt
```

> **Lưu ý**: `torchreid` có thể cần cài từ source nếu pip thất bại:
> ```bash
> git clone https://github.com/KaiyangZhou/deep-person-reid.git
> cd deep-person-reid
> pip install -r requirements.txt
> python setup.py develop
> ```

### Bước 3: Thêm CARLA Python API vào PYTHONPATH

```bash
# Windows (thêm vào System Environment Variables hoặc chạy mỗi lần)
set PYTHONPATH=%PYTHONPATH%;e:\finalproject\PythonAPI;e:\finalproject\PythonAPI\carla\dist\carla-0.9.9-py3.7-win-amd64.egg
```

### Bước 4: Kiểm tra cài đặt

```bash
# Test CARLA API
python -c "import carla; print('CARLA version:', carla.__version__)"

# Test PyTorch
python -c "import torch; print('CUDA available:', torch.cuda.is_available())"

# Test YOLOv5
python -c "import torch; model = torch.hub.load('ultralytics/yolov5', 'yolov5s'); print('YOLOv5 OK')"
```

---

## 3. Dataset và Training

> **Quan trọng**: Model mặc định (YOLOv5 pretrained, OSNet pretrained) có thể chạy **không cần train lại**. Training chỉ cần khi muốn tăng độ chính xác cho domain CARLA cụ thể.

---

### 3.1 Object Detection — YOLOv5

#### Dataset

| Dataset | Mục đích | Link tải |
|---------|----------|----------|
| **COCO 2017** | Pretrain chính, có đủ class (car, truck, bus, person) | https://cocodataset.org/#download |
| **BDD100K** | Fine-tune cho môi trường đường phố | https://bdd-data.berkeley.edu/ |
| **CARLA Synthetic** | Fine-tune cho domain CARLA (tự tạo) | Xem mục 3.1.3 |

#### 3.1.1 Sử dụng YOLOv5 pretrained (không cần train)

```python
# Trong detector.py — đã được config sẵn
# model_type: 'yolov5s' (nhanh), 'yolov5m' (cân bằng), 'yolov5l' (chính xác)
detector = ObjectDetector(model_type='yolov5s', conf_threshold=0.4)
```

Pretrained YOLOv5s trên COCO đã detect được: `car`, `truck`, `bus`, `person` — đủ dùng ngay.

#### 3.1.2 Fine-tune YOLOv5 trên BDD100K (tùy chọn)

```bash
# Cài YOLOv5 repo
git clone https://github.com/ultralytics/yolov5
cd yolov5
pip install -r requirements.txt

# Chuẩn bị BDD100K theo format YOLO
# Cấu trúc thư mục:
# datasets/bdd100k/
#   images/train/  images/val/
#   labels/train/  labels/val/

# Train fine-tune từ pretrained weights
python train.py \
  --img 640 \
  --batch 16 \
  --epochs 50 \
  --data bdd100k.yaml \
  --weights yolov5s.pt \
  --cache \
  --device 0
```

File `bdd100k.yaml`:
```yaml
path: datasets/bdd100k
train: images/train
val: images/val

nc: 4
names: ['person', 'car', 'bus', 'truck']
```

#### 3.1.3 Tạo synthetic dataset từ CARLA

```python
# Script thu thập dữ liệu detection từ CARLA
# Chạy CARLA server trước, sau đó:
python e:\finalproject\PythonAPI\custom_tracking_system\utils\data_writer.py \
  --mode collect_detection \
  --output datasets/synthetic_data/detection \
  --frames 5000
```

Format output: ảnh JPG + label YOLO (`.txt`) với format `class cx cy w h`.

---

### 3.2 Multi-Object Tracking — Không cần train riêng

Tracker (`SimpleTracker`) dùng IoU-based matching — **không cần train**, hoạt động trực tiếp trên output của detector.

Để nâng cấp lên ByteTrack/DeepSORT:

```bash
# Cài ByteTrack
git clone https://github.com/ifzhang/ByteTrack.git
cd ByteTrack && pip install -r requirements.txt

# Cài DeepSORT
pip install deep-sort-realtime
```

Evaluate tracker trên MOT17:
```bash
# Download MOT17
wget https://motchallenge.net/data/MOT17.zip

# Evaluate (dùng py-motmetrics)
pip install motmetrics
python -m motmetrics.apps.evaluateTracking MOT17/train predictions/
```

---

### 3.3 Re-Identification (ReID) — OSNet pretrained

#### Dataset

| Dataset | Số identity | Số ảnh | Mục đích |
|---------|-------------|--------|----------|
| **Market-1501** | 1,501 người | 32,668 ảnh | Train/eval ReID người |
| **DukeMTMC-reID** | 1,812 người | 36,411 ảnh | Train/eval ReID multi-camera |
| **VehicleID** | 26,267 xe | 221,763 ảnh | ReID cho vehicle |
| **CARLA Synthetic** | Tự tạo | Tuỳ ý | Fine-tune cho domain CARLA |

#### 3.3.1 Sử dụng OSNet pretrained (khuyến nghị, không cần train)

```python
# reid.py đã auto-load OSNet pretrained trên Market-1501 qua torchreid
# Không cần thêm bước nào
reid = ReIDExtractor()  # Tự động dùng OSNet nếu torchreid đã cài
```

Pretrained OSNet cho kết quả Rank-1 > 90% trên Market-1501 — đủ dùng ngay.

#### 3.3.2 Fine-tune OSNet trên CARLA Synthetic (tùy chọn, nâng cao)

```bash
# Bước 1: Thu thập ReID dataset từ CARLA
# Script tạo ảnh crop của vehicle/pedestrian từ nhiều góc camera khác nhau
python utils/data_writer.py \
  --mode collect_reid \
  --output datasets/synthetic_data/reid \
  --cameras 3 \
  --frames_per_id 10 \
  --num_ids 500

# Cấu trúc output:
# datasets/synthetic_data/reid/
#   query/   (1 ảnh/ID từ camera 1)
#   gallery/ (nhiều ảnh/ID từ camera 2,3)

# Bước 2: Fine-tune với torchreid
python -c "
import torchreid
datamanager = torchreid.data.ImageDataManager(
    root='datasets/synthetic_data',
    sources='reid',
    targets='reid',
    height=256, width=128,
    batch_size_train=32,
    batch_size_test=100
)
model = torchreid.models.build_model(
    name='osnet_x1_0',
    num_classes=datamanager.num_train_pids,
    pretrained=True
)
optimizer = torchreid.optim.build_optimizer(model, optim='adam', lr=0.0003)
scheduler = torchreid.optim.build_lr_scheduler(optimizer, lr_scheduler='cosine', max_epoch=30)
engine = torchreid.engine.ImageSoftmaxEngine(
    datamanager, model, optimizer=optimizer, scheduler=scheduler
)
engine.run(save_dir='models/reid/osnet_carla', max_epoch=30, eval_freq=5, print_freq=20)
"
```

---

### 3.4 Trajectory Prediction — Linear Motion (mặc định)

Module `TrajectoryPredictor` dùng linear motion model — **không cần train**, hoạt động ngay.

#### Nâng cấp lên LSTM (tùy chọn, nâng cao)

```bash
# Bước 1: Thu thập trajectory dataset từ CARLA
python utils/data_writer.py \
  --mode collect_trajectory \
  --output datasets/synthetic_data/trajectories \
  --frames 10000

# Format output (CSV):
# frame_id, global_id, x, y

# Bước 2: Train LSTM model
# Tạo file train_lstm.py trong thư mục models/tracking/
python models/tracking/train_lstm.py \
  --data datasets/synthetic_data/trajectories \
  --epochs 100 \
  --seq_len 10 \
  --pred_len 5 \
  --output models/tracking/lstm_trajectory.pth
```

#### Sử dụng dataset công khai để pre-train LSTM

| Dataset | Link | Đặc điểm |
|---------|------|-----------|
| ETH/UCY | https://github.com/agrimgupta92/sgan | Pedestrian trajectory |
| Argoverse | https://www.argoverse.org/av1.html | Vehicle trajectory, có map |
| Waymo Motion | https://waymo.com/open/data/motion/ | Large-scale, multi-agent |

```bash
# Pre-train trên ETH/UCY (đơn giản nhất để bắt đầu)
git clone https://github.com/agrimgupta92/sgan.git
cd sgan
python scripts/train.py \
  --dataset_name eth \
  --pred_len 12 \
  --num_epochs 200
```

---

### 3.5 Tóm tắt: Thứ tự train

```
Để chạy ngay (không cần train):
  ✅ YOLOv5s pretrained (COCO)      → detector.py tự load
  ✅ OSNet pretrained (Market-1501)  → reid.py tự load nếu có torchreid
  ✅ Linear trajectory model         → không cần model file

Để tăng độ chính xác (train thêm):
  🔧 YOLOv5 fine-tune trên BDD100K  → tăng mAP cho đường phố
  🔧 OSNet fine-tune trên CARLA ReID → tăng Rank-1 cho domain CARLA
  🔧 LSTM train trên ETH/UCY        → tăng độ chính xác trajectory
```

---

## 4. Cách chạy project

### Bước 1: Khởi động CARLA Server

```bash
# Mở terminal riêng, chạy CARLA
cd e:\finalproject\WindowsNoEditor
CarlaUE4.exe -windowed -ResX=800 -ResY=600 -quality-level=Low

# Hoặc chạy headless (không cần màn hình, tiết kiệm VRAM)
CarlaUE4.exe -RenderOffScreen
```

Chờ khoảng 15-30 giây cho đến khi CARLA server sẵn sàng.

### Bước 2: Chạy hệ thống tracking

#### Cách 1: Dùng run.bat (đơn giản nhất)

```bash
cd e:\finalproject\PythonAPI\custom_tracking_system

# Chạy với cấu hình mặc định (1000 frames)
run.bat

# Chạy với số frame tùy chỉnh
run.bat 500

# Chạy với config file khác
run.bat 1000 config\camera_config.yaml
```

#### Cách 2: Dùng python trực tiếp (nhiều tuỳ chọn hơn)

```bash
cd e:\finalproject\PythonAPI\custom_tracking_system

# Chạy cơ bản
python main.py

# Chạy với config tùy chỉnh
python main.py --config config/camera_config.yaml

# Giới hạn số frame
python main.py --max-frames 500

# Bật debug log
python main.py --log-level DEBUG

# Kết hợp
python main.py \
  --config config/camera_config.yaml \
  --max-frames 1000 \
  --log-level INFO
```

### Bước 3: Điều khiển trong lúc chạy

| Phím | Hành động |
|------|-----------|
| `Q` | Dừng hệ thống (graceful shutdown) |
| `Ctrl+C` | Dừng ngay lập tức |

### Bước 4: Xem kết quả

- **Cửa sổ OpenCV**: Mỗi camera hiện thị 1 cửa sổ riêng với bounding box + Global ID
- **Log file**: `tracking_system.log` trong thư mục `custom_tracking_system/`
- **Metrics**: In ra console mỗi 100 frames

---

### 4.1 Cấu hình Camera (camera_config.yaml)

Chỉnh vị trí camera trong file `config/camera_config.yaml`:

```yaml
cameras:
  camera_0:
    position: [0, 0, 3]        # Vị trí trong CARLA world (x, y, z mét)
    rotation: [0, 0, 0]        # Hướng nhìn (pitch, yaw, roll độ)
    camera_id: "CAM_001"
    view_angle: 90             # FOV (độ)
    resolution: [1920, 1080]   # Độ phân giải (giảm xuống để tăng FPS)
    fps: 10                    # FPS camera

system:
  reid_threshold: 0.5          # Ngưỡng match ReID (0.3=dễ match, 0.7=khó match)
  trajectory_window: 10        # Số frame lịch sử để predict
  prediction_steps: 5          # Số bước predict tương lai
```

---

## 5. Các kịch bản (Scenarios)

---

### Kịch bản 1: Kiểm thử cơ bản — Single Camera

**Mục tiêu**: Xác nhận detection + tracking hoạt động trên 1 camera trước khi scale up.

**Cấu hình** (`config/camera_config.yaml`):
```yaml
cameras:
  camera_0:
    position: [0, 0, 5]
    rotation: [-15, 0, 0]     # Nghiêng xuống 15 độ nhìn mặt đường
    camera_id: "CAM_001"
    view_angle: 90
    resolution: [1280, 720]   # Giảm resolution để test nhanh
    fps: 10

system:
  synchronous_mode: true
  fixed_delta_seconds: 0.1
```

**Lệnh chạy**:
```bash
python main.py --config config/camera_config.yaml --max-frames 300 --log-level DEBUG
```

**Điều chỉnh trong CARLA** (trước khi chạy main.py):
```python
# Spawn ít vehicle để dễ quan sát
# Trong TrafficGenerator: num_vehicles=5, num_pedestrians=3
```

**Kết quả kỳ vọng**:
- Cửa sổ hiển thị camera với bounding box màu xanh
- Console log: `Detection: N objects detected`
- Track ID bắt đầu từ 0 và tăng dần
- Không có lỗi exception

**Metric cần kiểm tra**:
```
Detection: mAP > 0.7 (trên CARLA domain)
Tracking: MOTA > 0.6, IDF1 > 0.6
```

---

### Kịch bản 2: Multi-Camera — Đồng bộ và Cross-Camera Tracking

**Mục tiêu**: Xác nhận hệ thống xử lý đồng thời 3 camera và gán Global ID nhất quán.

**Cấu hình**: Dùng `camera_config.yaml` mặc định (3 camera, layout giao lộ).

```
Layout:
          CAM_001 (thẳng)
               ↓
    ←  Road  [GIAO LỘ]  Road →
               ↑
          CAM_002 (đối diện)
               
    CAM_003 (góc 45°, nhìn vào giao lộ)
```

**Lệnh chạy**:
```bash
python main.py \
  --config config/camera_config.yaml \
  --max-frames 1000 \
  --log-level INFO
```

**Điều kiện kiểm tra**:
1. Xe di chuyển từ vùng nhìn của CAM_001 sang CAM_002
2. Hệ thống phải gán cùng 1 Global ID cho xe đó ở cả 2 camera

**Quan sát trong log**:
```
[INFO] Global ID 1000 matched across CAM_001 → CAM_002 (cosine score: 0.72)
[INFO] ALERT: Object 1000 approaching intersection_main (ETA: 3 frames)
```

**Metric cần kiểm tra**:
```
ReID Rank-1 Accuracy > 0.7  (số lần match đúng / tổng số lần xe chuyển camera)
Global ID consistency: cùng 1 xe không nên có 2 Global ID khác nhau
```

---

### Kịch bản 3: Dự đoán quỹ đạo và cảnh báo ROI

**Mục tiêu**: Kiểm thử hệ thống cảnh báo khi object sắp vào vùng ROI.

**Cấu hình ROI** (`camera_config.yaml`):
```yaml
rois:
  camera_0:
    zones:
      - name: "danger_zone"
        polygon: [[400, 300], [900, 300], [900, 600], [400, 600]]  # Vùng giữa màn hình
  camera_1:
    zones:
      - name: "exit_zone"
        polygon: [[0, 0], [400, 0], [400, 400], [0, 400]]
```

**Lệnh chạy**:
```bash
python main.py \
  --config config/camera_config.yaml \
  --max-frames 2000 \
  --log-level INFO
```

**Kịch bản diễn ra**:
1. Xe di chuyển thẳng qua giao lộ
2. Trajectory predictor dự đoán 5 bước tiếp theo
3. Alert system phát hiện xe sắp vào `danger_zone`
4. Alert được log và hiển thị

**Quan sát kỳ vọng trong console**:
```
ALERT: {'type': 'ROI_WARNING', 'global_id': 1001, 'camera_id': 'CAM_001',
        'roi_name': 'danger_zone', 'eta': 3}
```

**Điều chỉnh độ nhạy cảnh báo**:
```python
# Trong alert_system.py — giảm để cảnh báo sớm hơn
self.alert_threshold = 3  # frames (mặc định: 5)

# Trong camera_config.yaml — tăng prediction_steps để nhìn xa hơn
prediction_steps: 10  # Dự đoán 10 frames tới thay vì 5
```

---

### Kịch bản 4: Stress Test — Nhiều object, nhiều camera

**Mục tiêu**: Kiểm tra hiệu suất và độ ổn định hệ thống với traffic dày đặc.

**Cấu hình**:
```yaml
# camera_config.yaml
cameras:
  camera_0:
    resolution: [960, 540]    # Giảm resolution để tăng FPS
    fps: 10
  camera_1:
    resolution: [960, 540]
    fps: 10
  camera_2:
    resolution: [960, 540]
    fps: 10
```

```python
# Trong main.py → initialize_modules()
self.modules['traffic_generator'] = TrafficGenerator(
    self.world, num_vehicles=30, num_pedestrians=15)  # Tăng traffic
```

**Lệnh chạy**:
```bash
python main.py \
  --config config/camera_config.yaml \
  --max-frames 5000 \
  --log-level WARNING     # Chỉ log warning trở lên để không làm chậm
```

**Metric cần theo dõi**:
```
FPS xử lý: > 5 FPS/camera là chấp nhận được
RAM usage: < 8 GB
GPU VRAM: < 4 GB (nếu dùng GPU)
Số Global ID bị trùng (ID switch): càng ít càng tốt
```

**Nếu FPS quá thấp**, thử các biện pháp sau theo thứ tự:
1. Giảm `resolution` xuống `[640, 480]`
2. Đổi sang model `yolov5n` (nano, nhanh nhất)
3. Tắt hiển thị OpenCV (comment out `cv2.imshow`)
4. Giảm `num_vehicles` xuống 15

---

### Kịch bản 5: Thu thập Synthetic Dataset từ CARLA

**Mục tiêu**: Tạo dataset để fine-tune model cho domain CARLA.

**Bước 1: Thu thập ảnh và annotation detection**
```python
# Thêm vào main loop trong main.py — lưu frame và detection label
import json, os

save_dir = "datasets/synthetic_data/detection"
os.makedirs(f"{save_dir}/images", exist_ok=True)
os.makedirs(f"{save_dir}/labels", exist_ok=True)

# Trong vòng lặp xử lý mỗi camera:
frame_filename = f"{save_dir}/images/{camera_id}_frame{frame_count:06d}.jpg"
cv2.imwrite(frame_filename, frame)

# Lưu YOLO format label
label_filename = f"{save_dir}/labels/{camera_id}_frame{frame_count:06d}.txt"
with open(label_filename, 'w') as f:
    for det in detections:
        x1,y1,x2,y2 = det['box']
        h, w = frame.shape[:2]
        cx = ((x1+x2)/2) / w
        cy = ((y1+y2)/2) / h
        bw = (x2-x1) / w
        bh = (y2-y1) / h
        class_map = {'person':0, 'car':1, 'bus':2, 'truck':3}
        cls = class_map.get(det['class'], -1)
        if cls >= 0:
            f.write(f"{cls} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}\n")
```

**Lệnh chạy thu thập**:
```bash
python main.py --max-frames 3000 --log-level WARNING
# Kết quả: ~9000 ảnh (3000 frame × 3 camera)
```

**Bước 2: Thu thập ReID crops**
```python
# Lưu crop ảnh của từng local track ID theo camera
import hashlib

reid_dir = "datasets/synthetic_data/reid"
for local_track in local_tracks:
    track_id = local_track['track_id']
    x1,y1,x2,y2 = local_track['box']
    crop = frame[y1:y2, x1:x2]
    if crop.size == 0: continue
    
    person_dir = f"{reid_dir}/{camera_id}/{track_id:04d}"
    os.makedirs(person_dir, exist_ok=True)
    cv2.imwrite(f"{person_dir}/frame{frame_count:06d}.jpg", crop)
```

---

### Kịch bản 6: Đánh giá chất lượng hệ thống

**Mục tiêu**: Đo các metric định lượng để báo cáo kết quả.

**Chạy evaluation**:
```bash
python main.py \
  --config config/camera_config.yaml \
  --max-frames 1000 \
  --log-level INFO

# Sau khi chạy xong, đọc metrics từ log
grep "METRICS" tracking_system.log
```

**Checklist đánh giá**:

| Module | Metric | Ngưỡng chấp nhận | Cách đo |
|--------|--------|-----------------|---------|
| Detection | mAP@0.5 | > 0.70 | So sánh với ground truth CARLA |
| Tracking | MOTA | > 0.60 | Dùng py-motmetrics |
| Tracking | IDF1 | > 0.65 | Dùng py-motmetrics |
| ReID | Rank-1 Accuracy | > 0.70 | Đếm tỷ lệ match đúng cross-camera |
| Trajectory | ADE (pixel) | < 50px | Mean error giữa predicted và actual |
| Trajectory | FDE (pixel) | < 100px | Error tại bước cuối |
| Alert | Response time | < 100ms | Đo thời gian từ detect đến alert |
| System | FPS | > 5 FPS | Đo thực tế |

**Script đánh giá ReID**:
```python
# Đếm số lần xe xuất hiện ở camera mới được match đúng Global ID
correct_matches = 0
total_transitions = 0

# Dựa vào ground truth từ CARLA actor ID
for event in camera_transition_events:  # Từ CARLA actor ground truth
    actor_id = event['actor_id']
    global_id_assigned = event['global_id']
    expected_global_id = actor_to_global_map.get(actor_id)
    
    if expected_global_id == global_id_assigned:
        correct_matches += 1
    total_transitions += 1

rank1_accuracy = correct_matches / total_transitions
print(f"ReID Rank-1 Accuracy: {rank1_accuracy:.2%}")
```

---

## 6. Troubleshooting

### Lỗi: `Connection refused` khi kết nối CARLA

```
Nguyên nhân: CARLA server chưa khởi động hoặc đang khởi động
Giải pháp:
  1. Chờ 30 giây sau khi chạy CarlaUE4.exe
  2. Kiểm tra: netstat -an | findstr 2000
  3. Kiểm tra firewall không chặn port 2000
```

### Lỗi: `CUDA out of memory`

```
Nguyên nhân: GPU không đủ VRAM cho YOLOv5 + OSNet
Giải pháp:
  1. Đổi sang CPU: detector = ObjectDetector(device='cpu')
  2. Dùng model nhỏ hơn: model_type='yolov5n'
  3. Giảm resolution camera xuống [640, 480]
  4. Xử lý tuần tự từng camera thay vì song song
```

### Lỗi: `torchreid not found` — Fallback sang ResNet50

```
Đây là fallback bình thường, reid.py tự động dùng ResNet50 ImageNet.
Hiệu suất ReID sẽ giảm (Rank-1 ~50-60% thay vì ~85-90%).
Cài torchreid để dùng OSNet:
  pip install torchreid
  hoặc: git clone + python setup.py develop
```

### FPS quá thấp (< 3 FPS)

```
Thứ tự thử:
  1. Giảm resolution: [640, 480]
  2. Đổi model: yolov5n (nano)
  3. Tắt hiển thị: comment cv2.imshow
  4. Giảm vehicles: num_vehicles=5
  5. Dùng GPU thay vì CPU
  6. Tắt CARLA rendering: -RenderOffScreen
```

### ID Switch nhiều (cùng 1 xe có nhiều Global ID)

```
Nguyên nhân: ReID threshold quá cao hoặc feature không ổn định
Giải pháp:
  1. Giảm reid_threshold trong camera_config.yaml (0.5 → 0.4)
  2. Tăng số feature lưu trong gallery (10 → 20)
  3. Fine-tune OSNet trên CARLA data (xem mục 3.3.2)
  4. Đảm bảo camera frames đồng bộ (synchronous_mode: true)
```

### CARLA crash hoặc freeze

```
Nguyên nhân: Quá nhiều actor hoặc memory leak
Giải pháp:
  1. Giảm num_vehicles + num_pedestrians
  2. Đảm bảo cleanup() được gọi khi thoát
  3. Restart CARLA server: CarlaUE4.exe
  4. Chạy với -quality-level=Low để giảm tải GPU
```
