# Execute Guide: Multi-Camera CCTV Tracking System

## 0. Quick Start (cập nhật 2026-06-14)

> **Đường dẫn project hiện tại**: `c:\Users\Admin\finalproject\` (các đường dẫn
> `e:\School_project\finalproject\...` trong tài liệu cũ dưới đây chỉ còn giá
> trị lịch sử — thay bằng đường dẫn trên).

### Khởi động toàn bộ hệ thống

```cmd
cd c:\Users\Admin\finalproject
start.bat
```

Đợi tuần tự: CARLA (port 2000, ~30-60s) → Backend (port 8000, AI pipeline
init ~10-15s) → Frontend (port 3000). Mở **http://localhost:3000** để demo.

Dừng toàn bộ: `stop.bat`.

### Khởi động lại thủ công (khi cần restart từng phần / debug)

```powershell
# 1. Kill tat ca
Get-Process | Where-Object {$_.ProcessName -match "Carla|python|node"} | Stop-Process -Force

# 2. CARLA
Start-Process -FilePath "c:\Users\Admin\finalproject\WindowsNoEditor\CarlaUE4.exe" `
  -ArgumentList "-windowed","-ResX=800","-ResY=600","-quality-level=Low"
# doi port 2000 san sang (~30-60s)

# 3. Backend (--with-ai)
Start-Process -FilePath "c:\Users\Admin\finalproject\custom_tracking_system\venv_tracking\Scripts\python.exe" `
  -ArgumentList "app.py","--with-ai" `
  -WorkingDirectory "c:\Users\Admin\finalproject\server"

# 4. Frontend
Start-Process -FilePath "cmd.exe" -ArgumentList "/c","npm run dev" `
  -WorkingDirectory "c:\Users\Admin\finalproject\frontend"
```

### Lỗi "AI processor chưa chạy" (trigger scenario trả HTTP 400)

**Nguyên nhân**: CARLA simulator bị treo/corrupt sau một thời gian chạy +
nhiều lần trigger scenario — `world.tick()` timeout 10s
(`RuntimeError: time-out of 10000ms while waiting for the simulator`).
Exception này không được `_safe_tick()` bắt (chỉ bắt lỗi
`set_actor_collisions`), nên AI processor thread crash, `status` chuyển
`"error"`, và `trigger_scenario()` luôn trả `{"ok": false, "message": "AI
processor chua chay"}`. FastAPI process vẫn sống — camera stream đứng hình ở
frame cuối.

**Cách nhận biết**: `curl http://localhost:8000/api/stats/` → `fps` không
đổi qua nhiều lần gọi; log backend có dòng `AI processor error: time-out of
10000ms while waiting for the simulator`.

**Khắc phục**: restart CARLA + backend theo "Khởi động lại thủ công" ở trên
(không cần restart frontend — Vite tự reconnect). Đây là vấn đề ổn định của
CARLA server, không phải bug ở code AI pipeline.

### Camera bị "đứng hình" (freeze) khi trigger scenario, fps vẫn báo > 10

**Nguyên nhân đã từng gặp**: pedestrian (do `TrafficGenerator` hoặc
`ScenarioController` spawn) dùng `controller.ai.walker` (`AI walker
controller`) — `.start()`/`go_to_location()` gọi RPC `set_actor_collisions`
không tồn tại ở CARLA server 0.9.14 (client 0.9.15). Lỗi này bị delay và bắt
đầu spam ở mỗi `world.tick()` sau đó; `world.tick()` không advance simulation
đúng cách → camera trả frame cũ lặp lại (đứng hình) nhưng loop xử lý lại chạy
nhanh hơn → `fps` báo cáo vọt lên > 10 (dấu hiệu nhận biết).

**Đã fix** (cả 2 nơi spawn pedestrian — `modules/scenario_controller.py` và
`modules/traffic_generator.py`): bỏ hoàn toàn `controller.ai.walker`, dùng
`WalkerControl` trực tiếp (set hướng + tốc độ đi bộ thủ công) — không gọi RPC
`set_actor_collisions`.

---

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
| GPU | NVIDIA GTX 1060 6GB VRAM | NVIDIA RTX 3060+ (6GB+ VRAM, Ampere/Turing) |
| Disk | 70 GB trống (CARLA ~12 GB) | 100 GB SSD |
| Python | 3.7 (`carla-sim`, env CARLA) + 3.10 (`veri-train`, AI pipeline) | như khuyến nghị |
| CARLA | **0.9.14** (`WindowsNoEditor/`, đã cài đặt) | 0.9.14 |
| Node.js | 18+ | 20+ (cho frontend) |

> **Cập nhật (2026-06-12): chỉ dùng 1 venv (`custom_tracking_system/venv_tracking`,
> Python 3.10) để chạy `server/app.py --with-ai` — không cần `carla_bridge`,
> không cần conda env riêng cho CARLA.**
>
> CARLA 0.9.14 (`WindowsNoEditor/`) chỉ ship Python API cho cp37 (file `.egg`
> ở `WindowsNoEditor/PythonAPI/carla/dist/`). Thay vì dùng `.egg` đó, cài
> thẳng `pip install carla==0.9.15` (wheel cp310 trên PyPI) vào
> `venv_tracking` — client 0.9.15 kết nối được server 0.9.14, chỉ in ra
> WARNING version mismatch (vô hại, đã test thực tế).
>
> `venv_tracking` vì vậy có đủ: `carla` (pip, cp310), `torch`/`ultralytics`/
> `boxmot`/`torchreid` (AI pipeline), `fastapi`/`uvicorn`/`sqlalchemy` (server).
> KHÔNG dùng `python`/`python.exe` mặc định trong PATH — thiếu hết các gói
> trên. `start.bat` đã được cập nhật để gọi thẳng
> `custom_tracking_system\venv_tracking\Scripts\python.exe` cho cả backend
> (`--with-ai` và `--no-ai`).
>
> Lưu ý code: `modules/tracker.py` (`ByteTrackWrapper._init_tracker`) dùng API
> `boxmot.trackers.bytetrack.byte_tracker.BYTETracker` — đây là API của
> `boxmot==10.0.16` (bản đang cài trong `venv_tracking`). KHÔNG nâng `boxmot`
> lên >=20 nếu chưa sửa lại import này — boxmot>=20 yêu cầu torch>=2.2 (sẽ kéo
> theo nâng torch lên bản CPU-only mới nhất, mất CUDA).
>
> `scenario_controller.py` dùng `client.get_trafficmanager(8001)` (KHÔNG phải
> 8000 — port 8000 là của FastAPI server, dùng chung sẽ bị bind error khi
> `--with-ai` khởi tạo ScenarioController).
>
> Frontend chạy ở **port 3000** (xem `frontend/vite.config.js`), không phải
> 5173 (default Vite). `start.bat`/`stop.bat` đã sửa theo port 3000.
>
> Cách chạy đơn giản nhất: `start.bat` (full) hoặc `start.bat --no-ai`
> (không cần CARLA) — xem **mục 4**.
>
> Toàn bộ phần `carla_bridge` (TCP socket bridge giữa 2 venv) và phương án
> "nâng cấp CARLA 0.9.15 + Python 3.8" mô tả bên dưới (mục 2.0–2.x) là tài liệu
> **lịch sử**, không còn cần thiết — giữ lại để tham khảo nếu cần debug môi
> trường cũ, nhưng setup hiện tại của project KHÔNG dùng `carla_bridge`.

---

## 2. Cài đặt môi trường

> **Lưu ý đường dẫn:** Project nằm tại `e:\School_project\finalproject\`.

### 2.0 Phương án khuyến nghị: carla_bridge (2 venv, dùng CARLA hiện có)

Không cần tải CARLA mới. Tạo **2 virtual environment** riêng biệt:

#### venv_bridge (Python 3.7 — chạy carla_bridge/server.py)

```cmd
cd e:\School_project\finalproject\custom_tracking_system

py -3.7 -m venv venv_bridge
venv_bridge\Scripts\activate

pip install numpy pyyaml
pip install WindowsNoEditor\PythonAPI\carla\dist\carla-0.9.9-cp37-cp37m-win_amd64.egg
```

> Nếu cài `.egg` bằng pip thất bại, thêm vào PYTHONPATH thay vì cài đặt:
> ```cmd
> set PYTHONPATH=%PYTHONPATH%;e:\School_project\finalproject\WindowsNoEditor\PythonAPI;e:\School_project\finalproject\WindowsNoEditor\PythonAPI\carla\dist\carla-0.9.9-py3.7-win-amd64.egg
> ```

#### venv_tracking (Python 3.10+ — chạy main.py AI pipeline)

```cmd
cd e:\School_project\finalproject\custom_tracking_system

py -3.10 -m venv venv_tracking
venv_tracking\Scripts\activate

# Cài PyTorch GPU trước (CUDA 11.8 cho RTX 3060 / GTX 1050Ti)
pip install torch==2.0.1+cu118 torchvision==0.15.2+cu118 --extra-index-url https://download.pytorch.org/whl/cu118

# Cài các dependency còn lại (KHÔNG cần gói carla — process này không import carla)
pip install -r requirements.txt
```

> **Lưu ý `torchreid`**: nếu pip thất bại, cài từ source:
> ```cmd
> git clone https://github.com/KaiyangZhou/deep-person-reid.git
> cd deep-person-reid && pip install -e .
> ```

Cài đặt dependencies Server (cùng venv_tracking hoặc venv riêng):

```cmd
cd e:\School_project\finalproject\server
pip install -r requirements.txt
```

Cài đặt frontend:

```cmd
cd e:\School_project\finalproject\frontend
npm install
```

Sau khi cài xong, xem **Cách 4 ở mục 4** để chạy carla_bridge + main.py.

---

### Phương án thay thế: Nâng cấp CARLA 0.9.15 + Python 3.8

> Phương án này KHÔNG bắt buộc nếu đã dùng carla_bridge ở mục 2.0. Chỉ cân
> nhắc nếu cần chạy CARLA + AI pipeline trong cùng 1 process Python.

> CARLA 0.9.15 cần được tải và giải nén vào `e:\School_project\finalproject\WindowsNoEditor\`.

### Bước 1: Tải CARLA 0.9.15

Tải `CARLA_0.9.15.zip` (hoặc Windows installer) từ GitHub Releases của CARLA.
Giải nén vào `e:\School_project\finalproject\WindowsNoEditor\`.

Sau khi giải nén, Python API nằm tại:
```
WindowsNoEditor\PythonAPI\carla\dist\carla-0.9.15-py3.8-win-amd64.egg
```

### Bước 2: Tạo môi trường Python 3.8

```cmd
# Tạo virtual environment với Python 3.8
py -3.8 -m venv venv_tracking

# Kích hoạt (Windows cmd)
venv_tracking\Scripts\activate
```

### Bước 3: Cài đặt dependencies AI Pipeline

```cmd
cd e:\School_project\finalproject\custom_tracking_system

# Cài PyTorch GPU trước (CUDA 11.8 cho RTX 3060)
pip install torch==2.0.1+cu118 torchvision==0.15.2+cu118 --extra-index-url https://download.pytorch.org/whl/cu118

# Cài các dependency còn lại
pip install -r requirements.txt
```

> **Lưu ý `torchreid`**: nếu pip thất bại, cài từ source:
> ```cmd
> git clone https://github.com/KaiyangZhou/deep-person-reid.git
> cd deep-person-reid && pip install -e .
> ```

### Bước 4: Cài đặt dependencies Server

```cmd
cd e:\School_project\finalproject\server
pip install -r requirements.txt
```

### Bước 5: Thêm CARLA Python API vào PYTHONPATH

```cmd
# Thêm vào System Environment Variables (một lần duy nhất) hoặc chạy mỗi lần:
set PYTHONPATH=%PYTHONPATH%;e:\School_project\finalproject\WindowsNoEditor\PythonAPI;e:\School_project\finalproject\WindowsNoEditor\PythonAPI\carla\dist\carla-0.9.15-py3.8-win-amd64.egg
```

Hoặc dùng `start.bat` — script này đã tự set PYTHONPATH đúng.

### Bước 6: Cài đặt frontend

```cmd
cd e:\School_project\finalproject\frontend
npm install
```

### Bước 7: Sửa type annotations cho Python 3.8

Thêm dòng sau làm dòng đầu tiên vào 4 file (cần thiết vì syntax `X | Y` là Python 3.10+):

```python
from __future__ import annotations
```

Các file cần sửa:
- `server/services/ai_processor.py`
- `server/services/stream_service.py`
- `server/routers/websocket.py`
- `custom_tracking_system/modules/trajectory_predictor.py`

### Bước 8: Sửa traffic_generator.py cho CARLA 0.9.14+

Mở `custom_tracking_system/modules/traffic_generator.py`, dòng 83, đổi:
```python
vehicle.set_autopilot(True)
```
thành:
```python
vehicle.set_autopilot(True, 8000)
```

### Bước 9: Kiểm tra cài đặt

```cmd
# Test CARLA API
python -c "import carla; print('CARLA version:', carla.__version__)"

# Test PyTorch + CUDA
python -c "import torch; print('CUDA available:', torch.cuda.is_available(), '| Device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"

# Test YOLOv8
python -c "from ultralytics import YOLO; m = YOLO('yolov8s.pt'); print('YOLOv8 OK')"

# Test ByteTrack
python -c "from boxmot import ByteTrack; print('ByteTrack OK')"
```

---

## 3. Dataset và Training

> **Quan trọng**: Model mặc định (YOLOv8s pretrained, OSNet pretrained) có thể chạy **không cần train lại**. Training chỉ cần khi muốn tăng độ chính xác cho domain cụ thể (giao thông VN, motorcycle VN).

---

### 3.1 Object Detection — YOLO11m (5 lớp VN)

`detector.py` (`ObjectDetector`) mặc định dùng `yolo11m` (COCO weights, qua
`CLASSES_COCO`) và **tự động** chuyển sang checkpoint fine-tune
`weights/yolo11m_vn.pt` (5 lớp `CLASSES_VN`: person/car/motorcycle/bus/truck)
nếu file đó tồn tại. Mục này mô tả cách tạo `weights/yolo11m_vn.pt`.

#### Dataset — 2 nguồn, gộp lại bằng script có sẵn

| Dataset | Mục đích | Script |
|---------|----------|--------|
| **VisDrone2019-DET** | Ảnh thực, góc camera trên cao giống CCTV — merge 10 lớp gốc → 5 lớp VN | `prepare_visdrone_dataset.py` |
| **CARLA Synthetic** | Camera CCTV ảo đặt ở góc phố/gắn tường tòa nhà gần giao lộ, nhãn bbox tự động từ ground-truth 3D | `scripts/collect/collect_carla_detection_data.py` |

#### 3.1.1 Sử dụng YOLO11m pretrained (không cần train)

```python
# Trong detector.py — đã config sẵn, chạy ngay không cần fine-tune
detector = ObjectDetector(model_type='yolo11m', conf_threshold=0.35)

# Khi chạy cùng CARLA (VRAM hạn chế), bật FP16:
detector = ObjectDetector(model_type='yolo11m', conf_threshold=0.35, half=True)

# Batched inference cho 6 camera (1 forward pass thay 6):
detections_per_cam = detector.detect_batch([frame_cam1, frame_cam2, ..., frame_cam6])
```

Pretrained YOLO11m trên COCO detect được: `car`, `truck`, `bus`, `person`,
`motorcycle` — đủ dùng ngay nếu chưa fine-tune.

#### 3.1.2 Chuẩn bị dataset VisDrone-VN

Tải + convert VisDrone2019-DET (train ~1.4GB, val ~70MB) sang YOLO format,
merge 10 lớp gốc → 5 lớp VN (`person, car, motorcycle, bus, truck`):

```cmd
cd custom_tracking_system
python prepare_visdrone_dataset.py --out data/visdrone_vn
```

Output: `data/visdrone_vn/images/{train,val}`, `data/visdrone_vn/labels/{train,val}`,
`data/visdrone_vn.yaml`.

#### 3.1.3 Tạo synthetic dataset từ CARLA

`scripts/collect/collect_carla_detection_data.py` (env `carla-sim`, CarlaUE4.exe đang chạy)
spawn traffic + nhiều camera RGB/depth đặt theo kiểu CCTV thật — gần các
junction trong bản đồ, cách tâm 8–15m (≈ góc tòa nhà), cao 5–8m, nghiêng
xuống 10–35°. Mỗi frame, bounding box 3D của mọi vehicle/walker được chiếu
lên ảnh (`modules/calibration.py`), lọc occlusion bằng depth camera, ghi
label YOLO 5 lớp VN.

```cmd
cd custom_tracking_system
:: smoke test trước (verify camera placement + bbox projection)
python scripts/collect/collect_carla_detection_data.py --frames 50 --out data/carla_det_test

:: thu thập thật
python scripts/collect/collect_carla_detection_data.py --frames 5000 --num-cameras 6 --out data/carla_det
```

Output: `data/carla_det/images/{train,val}`, `data/carla_det/labels/{train,val}`,
`data/carla_det.yaml`.

> **Trạng thái (2026-06-12):** script đã viết xong nhưng **chưa chạy thử với
> CARLA thật** — luôn chạy `--frames 50` trước để kiểm tra vị trí camera và
> bbox hợp lý trước khi thu thập số lượng lớn.

#### 3.1.4 Gộp dataset và fine-tune YOLO11m

```cmd
cd custom_tracking_system

:: gộp VisDrone-VN + CARLA thành 1 dataset
python merge_detection_datasets.py ^
    --datasets visdrone=data/visdrone_vn carla=data/carla_det ^
    --out data/visdrone_carla_vn

:: fine-tune (env veri-train hoặc cloud RTX 4060 8GB — torch CUDA riêng,
:: không downgrade torch của pipeline chính)
python train_yolo_detector.py --data data/visdrone_carla_vn.yaml --epochs 50 ^
    --batch 16 --out weights/yolo11m_vn.pt
```

Hardware notes (xem docstring `train_yolo_detector.py`):
- RTX 4060 8GB: `--batch 16-24 --device 0`
- GTX 1050Ti 4GB: `--batch 4-8 --device 0 --half`
- CPU only: `--device cpu --batch 4 --epochs 2` (chỉ để smoke test)

Sau khi train xong, `weights/yolo11m_vn.pt` được `ObjectDetector` tự động
dùng (xem `modules/detector.py`) — không cần sửa code.

---

### 3.2 Multi-Object Tracking — Không cần train riêng

Tracker hiện tại là `ByteTrackWrapper` (dùng `boxmot.ByteTrack`) — **không cần train**, hoạt động trực tiếp trên output của detector. Sử dụng Kalman Filter để dự đoán vị trí khi bị che khuất và Hungarian matching để gán ID tối ưu.

```python
# tracker.py — cấu hình mặc định
tracker = ByteTrackWrapper(
    track_activation_threshold=0.25,
    lost_track_buffer=30,
    minimum_matching_threshold=0.8,
    frame_rate=10
)
```

Để evaluate tracker trên MOT17:
```cmd
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

### 3.4 Trajectory Prediction — Kalman Filter Constant-Acceleration (mặc định)

Module `TrajectoryPredictor` dùng Kalman Filter constant-acceleration (`KalmanFilter2D`,
state `[x, y, vx, vy, ax, ay]`, dt-aware) — **không cần train**, hoạt động ngay,
không phụ thuộc thư viện ngoài (`filterpy` không cần thiết).

#### Nâng cấp lên LSTM (tùy chọn, nâng cao — chỉ cần khi Kalman không đủ)

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
  ✅ YOLOv8s pretrained (COCO)        → detector.py tự download + load lần đầu (~22 MB)
  ✅ OSNet pretrained (Market-1501)    → DualReIDExtractor tự load qua torchreid (person)
  ✅ OSNet VeRi-776 fine-tuned         → weights đã có tại models/reid/osnet_veri776.pth (vehicle)
  ✅ Exp-weighted velocity predictor   → không cần model file
  ✅ ByteTrackWrapper                  → không cần train, cài qua boxmot

Để tăng độ chính xác (train thêm):
  🔧 YOLOv8s fine-tune trên BDD100K + Vietnam Traffic  → motorcycle VN chính xác hơn
  🔧 Re-train VeRi-776 với validation split đúng        → val_acc có nghĩa (đã sửa bug)
  🔧 LSTM train trên ETH/UCY                            → tăng độ chính xác trajectory
```

---

## 4. Cách chạy project

### Cách 1: start.bat (khuyến nghị — khởi động toàn bộ tự động)

```cmd
cd e:\School_project\finalproject
start.bat
```

Script này tự động:
1. Khởi động CARLA server (`WindowsNoEditor\CarlaUE4.exe -quality-level=Low`)
2. Đợi port 2000 ready (timeout 60 giây)
3. Khởi động backend server với `--with-ai` flag
4. Khởi động frontend dev server

Để dừng toàn bộ:
```cmd
stop.bat
```

---

### Cách 2: Chạy thủ công từng thành phần

#### Bước 1: Khởi động CARLA Server

```cmd
cd e:\School_project\finalproject\WindowsNoEditor
CarlaUE4.exe -windowed -ResX=800 -ResY=600 -quality-level=Low

# Hoặc headless (tiết kiệm VRAM, không cần màn hình)
CarlaUE4.exe -RenderOffScreen
```

Chờ 15–30 giây cho đến khi CARLA sẵn sàng (port 2000).

#### Bước 2: Khởi động Backend Server

Dùng python của `venv_tracking` (không dùng `python` mặc định trong PATH —
xem ghi chú ở mục 1):

```cmd
cd e:\School_project\finalproject\server

REM Chạy không có AI (API-only, không cần CARLA)
..\custom_tracking_system\venv_tracking\Scripts\python.exe app.py

REM Chạy với AI pipeline đầy đủ (cần CARLA đang chạy)
..\custom_tracking_system\venv_tracking\Scripts\python.exe app.py --with-ai
```

Server khởi động tại `http://localhost:8000`. API docs tại `http://localhost:8000/docs`.

#### Bước 3: Khởi động Frontend

```cmd
cd e:\School_project\finalproject\frontend
npm run dev
```

Frontend tại `http://localhost:3000` (xem `vite.config.js`). Tự proxy `/api`, `/ws`, `/stream` → `:8000`.

---

### Cách 3: Chạy AI pipeline trực tiếp (không qua server, hiển thị OpenCV)

```cmd
cd e:\School_project\finalproject\custom_tracking_system

# Chạy cơ bản
python main.py

# Chạy với config tùy chỉnh
python main.py --config config/camera_config.yaml

# Giới hạn số frame
python main.py --max-frames 500 --log-level INFO
```

| Phím | Hành động |
|------|-----------|
| `Q` | Dừng hệ thống (graceful shutdown) |
| `Ctrl+C` | Dừng ngay lập tức |

> Cách 3 chạy CARLA in-process (`--source carla`, mặc định) — yêu cầu môi
> trường Python tương thích cả CARLA lẫn AI libs (xem "Phương án thay thế:
> Nâng cấp CARLA 0.9.15" ở mục 2). Nếu dùng kiến trúc carla_bridge (mục 2.0),
> chạy theo **Cách 4** bên dưới.

---

### Cách 4: Chạy với carla_bridge (2 process — không cần nâng cấp CARLA)

Yêu cầu đã tạo 2 venv theo mục 2.0 (`venv_bridge` Python 3.7, `venv_tracking` Python 3.10+).

#### Terminal 1 — CARLA Server

```cmd
cd e:\School_project\finalproject\WindowsNoEditor
CarlaUE4.exe -windowed -ResX=800 -ResY=600 -quality-level=Low
```

#### Terminal 2 — carla_bridge (Python 3.7)

```cmd
cd e:\School_project\finalproject\custom_tracking_system
venv_bridge\Scripts\activate

python carla_bridge\server.py --config config/camera_config.yaml --num-vehicles 10 --num-pedestrians 5
```

Mặc định bridge mở TCP server tại `127.0.0.1:8765` và đợi AI process kết nối.
Tham số `--host` / `--port` để đổi địa chỉ/cổng.

#### Terminal 3 — AI Pipeline (Python 3.10+)

```cmd
cd e:\School_project\finalproject\custom_tracking_system
venv_tracking\Scripts\activate

python main.py --source bridge --bridge-host 127.0.0.1 --bridge-port 8765
```

Các tham số khác (`--config`, `--max-frames`, `--half`, `--log-level`) hoạt động
như Cách 3. `Q` để dừng, `Ctrl+C` để dừng ngay.

> **Thứ tự khởi động quan trọng**: CARLA server → carla_bridge/server.py (đợi
> kết nối CARLA + spawn camera/traffic xong) → main.py --source bridge (kết
> nối tới bridge). Nếu main.py kết nối trước khi bridge sẵn sàng, nó sẽ chờ
> tối đa `connect_timeout=60s` rồi báo lỗi.

---

### Cách 5: Chạy AI pipeline trên video/camera thật (RTSP/file/webcam)

Không cần CARLA. Dùng `venv_tracking` (Python 3.10+, mục 2.0). Chạy pipeline
trên nguồn video thật qua lớp `VideoSource` (`modules/video_source.py`).

#### Webcam (demo nhanh, 1 camera)

```cmd
cd e:\School_project\finalproject\custom_tracking_system
venv_tracking\Scripts\activate

python main.py --source webcam --webcam-index 0 --camera-ids CAM_001
```

#### Video file (replay / benchmark)

```cmd
python main.py --source file --video-path data/test_video.mp4 --camera-ids CAM_001 --loop-video
```

#### Camera IP qua RTSP (nhiều camera)

```cmd
python main.py --source rtsp ^
  --rtsp-url rtsp://192.168.1.100:554/stream1 ^
  --rtsp-url rtsp://192.168.1.101:554/stream1 ^
  --camera-ids CAM_001,CAM_002
```

| Tham số | Ý nghĩa |
|---|---|
| `--rtsp-url` / `--video-path` / `--webcam-index` | Lặp lại (`--rtsp-url ... --rtsp-url ...`) để chạy nhiều camera cùng lúc — mỗi tham số tương ứng 1 nguồn |
| `--camera-ids` | Danh sách ID phân tách bởi dấu phẩy, theo đúng thứ tự các nguồn ở trên. Nếu bỏ qua, mặc định `CAM_001, CAM_002, ...` |
| `--loop-video` | Chỉ áp dụng `--source file` — phát lại từ đầu khi hết video |

> **Lưu ý ROI/camera_topology**: `--rois` đọc từ `config/camera_config.yaml` như
> các chế độ khác. Vì `camera_config.yaml` mặc định định nghĩa ROI cho
> `CAM_001/CAM_002/CAM_003` (kịch bản CARLA), khi chạy với camera thật bạn cần
> sửa `config/camera_config.yaml` (hoặc tạo file config riêng và truyền qua
> `--config`) cho đúng `camera_id` và toạ độ ROI của camera thật.
>
> **Trạng thái**: các class `RTSPVideoSource`/`FileVideoSource`/`WebcamVideoSource`
> và `--source {rtsp,file,webcam}` mới được wire vào `main.py` — **chưa được
> test với camera/luồng thật**, cần kiểm tra thực tế trước khi dùng cho demo.

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

### 4.2 Kịch bản tai nạn demo (ScenarioPanel) — camera-aware spawn

Dashboard (frontend, panel "Kịch bản demo") có 5 nút kích hoạt kịch bản tai
nạn trong CARLA (`POST /api/scenarios/{name}`, chỉ hoạt động khi backend chạy
`--with-ai`): `hit_and_run`, `pedestrian_hit`, `red_light_crash`, `rear_end`,
`sudden_stop`.

- `ScenarioController` (modules/scenario_controller.py) ưu tiên chọn spawn
  point nằm trong FOV của 1 trong 3 camera (vị trí + yaw + view_angle,
  bán kính 40m) cho mỗi kịch bản, để sự kiện xảy ra TRƯỚC camera (thấy được
  trên dashboard) thay vì ở góc khuất.
- Response của `POST /api/scenarios/{name}` trả về thêm `camera_id` — camera
  dự kiến quan sát được sự kiện (`null` nếu không spawn point nào nằm trong
  tầm camera nào, lúc đó kịch bản fallback về spawn ngẫu nhiên). Dashboard
  highlight camera feed tương ứng (viền cyan nhấp nháy ~15s).
- Lưu ý: pedestrian trong `pedestrian_hit` có thể đứng yên (không đi bộ) do
  lỗi `set_actor_collisions` (CARLA client 0.9.15 / server 0.9.14 mismatch —
  xem `custom_tracking_system/docs/error.md` mục 4). Kịch bản vẫn chạy được
  (xe áp sát người đứng yên), không bị lỗi API.

### 4.3 Lọc bớt cảnh báo (incident_detection.disabled_types)

`IncidentDetector` (modules/incident_detector.py) hỗ trợ
`incident_detection.disabled_types` trong `camera_config.yaml` để tắt các
loại incident xảy ra thường xuyên trong giao thông bình thường (không phải
dấu hiệu tai nạn), tránh làm loãng alert feed khi demo. Mặc định đã tắt:
`SUDDEN_ACCEL`, `CAMERA_TRANSITION`, `STOPPED_VEHICLE`, `VEHICLE_PROXIMITY`.
Các loại còn lại (`SUDDEN_STOP`, `OVERSPEED`, `PEDESTRIAN_DANGER`,
`LOITERING`, `CROWD_DENSITY`, `WRONG_WAY`, `RED_LIGHT_VIOLATION`,
`PREDICTED_COLLISION`, `PREDICTED_ROI_ENTRY`) vẫn hoạt động bình thường. Có
thể chỉnh lại danh sách này trực tiếp trong `camera_config.yaml`.

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
2. Đổi sang model `yolov8n` (nano, nhanh nhất)
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
Nguyên nhân: GPU không đủ VRAM cho YOLOv8s + OSNet + CARLA
Ước tính VRAM: YOLOv8s FP16 ~800MB, OSNet ~200MB, CARLA ~1.5-2GB → tổng ~2.5-3GB
RTX 3060 (6GB) đủ chạy toàn bộ pipeline.

Nếu vẫn bị OOM:
  1. Đảm bảo FP16 đã bật: ObjectDetector(model_type='yolov8s', half=True)
  2. Dùng model nhỏ hơn: model_type='yolov8n'
  3. Giảm resolution camera xuống [640, 480]
  4. Chạy CARLA ở quality-level=Low hoặc -RenderOffScreen
  5. Xử lý tuần tự từng camera thay vì song song
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
  2. Đổi model: yolov8n (nano)
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

### Lỗi: `CarlaUE4.exe` báo thiếu "DirectX Runtime"

```
Nguyên nhân: Windows 10 không có sẵn các DLL DirectX legacy (D3DX9, XAudio2.7,
XInput1.3...) mà UE4/CARLA cần (d3dx9_43.dll, xinput1_3.dll, ...).
winget package "Microsoft.DirectX" (dxwebsetup) KHÔNG cài các DLL này.

Giải pháp: tải và cài "DirectX End-User Runtime (June 2010)" — bộ cài đầy đủ
chứa các DLL legacy:
  1. Tải: https://download.microsoft.com/download/8/4/A/84A35BF1-DAFE-4AE8-82AF-AD2AE20B6B14/directx_Jun2010_redist.exe
  2. Giải nén:  directx_Jun2010_redist.exe /Q /T:%TEMP%\dxredist
  3. Cài đặt:   %TEMP%\dxredist\DXSETUP.exe /silent
  4. Kiểm tra: C:\Windows\System32\d3dx9_43.dll và xinput1_3.dll phải tồn tại
```

### Lỗi `boxmot` không có `boxmot.trackers.bbox.bytetrack.bytetrack.ByteTrack`

```
Nguyên nhân: boxmot 10.0.16 (bản đang cài trong venv_tracking) chỉ có API cũ
`boxmot.trackers.bytetrack.byte_tracker.BYTETracker`, không có class
`ByteTrack` ở path `boxmot.trackers.bbox.bytetrack.bytetrack` (API mới, chỉ
có từ boxmot>=20 — yêu cầu torch>=2.2, py>=3.10; nâng lên sẽ kéo torch về
bản CPU-only mới nhất, mất CUDA — KHÔNG làm trừ khi đã chuẩn bị lại toàn bộ
torch GPU stack).

Đây là lỗi TÁI DIỄN (đã từng sửa, rồi modules/tracker.py bị sửa lại dùng API
mới trong một commit sau đó). Nếu gặp lại `ImportError`/`ModuleNotFoundError`
khi `--with-ai` khởi tạo ByteTrackWrapper, kiểm tra
`modules/tracker.py::ByteTrackWrapper._init_tracker`:

  from boxmot.trackers.bytetrack.byte_tracker import BYTETracker
  self.tracker = BYTETracker(**self._init_kwargs)

_init_kwargs (track_thresh, track_buffer, match_thresh, frame_rate) và
update(dets, frame) / format output (cột 0-3=box, 4=id, 5=conf, 6=cls) đã
khớp sẵn với BYTETracker.update(dets, _) của boxmot 10.0.16 — không cần đổi
gì khác ngoài 2 dòng import/instantiate trên.

Lưu ý: cài boxmot/ultralytics không pin version dễ làm pip nâng cấp ngầm
numpy lên 2.x và torch/torchvision về bản CPU (mất CUDA). Dùng file
`custom_tracking_system/constraints.txt` (numpy==1.23.1,
torch==2.0.1+cu118, torchvision==0.15.2+cu118) với `-c constraints.txt`
khi cài các package AI để giữ đúng phiên bản GPU.
```

### Lỗi `ImportError: cannot import name 'Real' from 'sqlalchemy'`

```
Nguyên nhân: server/models/database.py dùng `Real` (không tồn tại trong
SQLAlchemy), đúng ra phải là `REAL`.

Giải pháp: thay tất cả `Real` → `REAL` trong server/models/database.py.
```

### Lỗi `cv2.imshow`: "function not implemented... Rebuild with GTK+/Cocoa"

```
Nguyên nhân: cả opencv-python và opencv-python-headless cùng được cài trong
venv_tracking (headless là dependency ngầm của ultralytics) — bản headless
ghi đè module cv2 và không có GUI (HighGUI).

Giải pháp:
  pip uninstall -y opencv-python opencv-python-headless
  pip install opencv-python==4.11.0.86
```

### Lỗi traffic manager: "trying to create rpc server... bind error"

```
Nguyên nhân: get_trafficmanager(8000) / set_autopilot(True, 8000) dùng port
8000 cho CARLA Traffic Manager, trùng với port 8000 của FastAPI backend
(server/app.py).

Giải pháp (cả 2 nơi tạo Traffic Manager đều phải đổi port):
  - modules/traffic_generator.py: vehicle.set_autopilot(True, 8050)  (đã sửa)
  - modules/scenario_controller.py: client.get_trafficmanager(8001) (đã sửa)

(2 TM port khác nhau cho 2 nhóm actor riêng — chưa gây vấn đề trong demo,
nhưng nên gộp về 1 port nếu cần ScenarioController điều khiển autopilot
của xe do TrafficGenerator spawn.)
```

### Đã xác nhận chạy thành công trên máy thực tế (RTX 4060 8GB)

```
Cấu hình đã test: Windows 10, Python 3.7.9 (venv) + Python 3.10 (venv_tracking),
Node.js 24.x, CARLA 0.9.14, RTX 4060 8GB.

Quy trình chạy (Cách 4 — carla_bridge):
  1. CarlaUE4.exe -RenderOffScreen -quality-level=Low
  2. (venv 3.7)        python carla_bridge/server.py
  3. (venv_tracking)   python main.py --source bridge
  4. (venv_tracking, thư mục server) python app.py        # FastAPI, port 8000
  5. (frontend)        npm run dev                          # Vite, port 3000

Kết quả: pipeline AI chạy ~8.4 FPS với YOLOv8s + ByteTrack + OSNet ReID (CUDA),
phát hiện sự cố (SUDDEN_STOP...) hoạt động đúng. Backend (port 8000) và
frontend (port 3000) chạy độc lập, ổn định ở chế độ --no-ai.

Lưu ý: server/app.py --with-ai KHÔNG chạy được với kiến trúc 2-venv hiện tại
vì ai_processor.py import trực tiếp `carla` (cần py3.7) và torch/ultralytics
(cần py3.10) trong cùng 1 process. Muốn tích hợp AI vào backend cần refactor
ai_processor.py để dùng carla_bridge qua TCP thay vì import carla trực tiếp.
```
