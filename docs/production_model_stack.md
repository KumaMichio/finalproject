# Production Model Stack — Hệ Thống 6 Camera CCTV AI

> **Mục đích tài liệu:** Ghi lại toàn bộ phân tích, lý do chọn mô hình, và kế hoạch
> triển khai AI pipeline cho môi trường production (6 camera liên tục, video thực).
>
> **Liên kết:** Xem thêm `upgrade.md` (lộ trình implement) và `bao_cao_tien_do.md` (tiến độ tổng thể).
>
> **Cập nhật lần cuối:** 2026-05-27

---

## 1. Ràng Buộc Phần Cứng

| Thành phần | Thông số |
|---|---|
| **CPU** | Intel Core i5-9300H — 4 cores / 8 threads, boost 4.1 GHz |
| **GPU** | NVIDIA GTX 1050Ti — **4 GB VRAM**, Pascal (CUDA 6.1) |
| **RAM** | 16 GB |

### Ý nghĩa với AI pipeline

- **4 GB VRAM là ràng buộc cứng** — không thể chạy mô hình lớn (ViT, CLIP full, YOLOv8l+)
- Pascal **không có Tensor Cores** → FP16 không giúp tăng tốc tính toán (chỉ tiết kiệm VRAM ~50%)
- CARLA Simulator chính nó đã dùng ~1.5–2 GB VRAM riêng → khi chạy cùng AI pipeline cần chia sẻ cẩn thận
- Với video thực (RTSP), CARLA không chạy → toàn bộ 4 GB dành cho AI

### Mục tiêu thực tế cho 6 camera

```
6 camera × 30 fps = 180 frame/giây về lý thuyết
GTX 1050Ti thực tế: ~10–15 fps/camera → ~60–90 frame/giây xử lý được
→ Chiến lược: batch 6 frame cùng lúc, 1 lần forward pass cho cả 6 camera
   thay vì 6 lần forward pass riêng lẻ (tốn 6× overhead)
```

---

## 2. Kiến Trúc Pipeline Production

### 2.1 Sơ đồ tổng thể

```
6 Camera Streams (RTSP / File / CARLA)
         │
         ▼  batch 6 frames đồng thời
┌────────────────────────────────────┐
│  [GPU] YOLOv8s — batch_size=6      │  ← 1 model, 1 forward pass cho cả 6 camera
│        ~1.8 GB VRAM                │
└──────────────┬─────────────────────┘
               │ detections per camera (list bboxes)
       ┌───────┼───────────────────────────┐
       ▼       ▼       ▼   (song song CPU) ▼
  ByteTrack  ByteTrack  ...  ByteTrack × 6   ← Kalman built-in, zero VRAM
       │
       ▼  crop patches của tất cả objects (gom lại)
┌────────────────────────────────────┐
│  [GPU] OSNet — batch crops          │  ← 1 model, batch tất cả crop 1 lần
│        ~0.4 GB VRAM                │
└──────────────┬─────────────────────┘
               │ feature vectors
       ┌───────┴──────────────────────┐
       ▼                              ▼
Global ID Matching              Spatio-Temporal
(cosine similarity)             Filter (CPU logic)
       │
       ▼
Kalman Trajectory Prediction (CPU — tái dùng từ ByteTrack state)
       │
       ▼
Rule-based Incident Detector (CPU) ← dùng predicted positions (proactive)
       │
       ▼
FastAPI Backend → WebSocket → React Dashboard (6 camera grid)
```

### 2.2 Budget VRAM

| Thành phần | VRAM |
|---|---|
| YOLOv8s batch=6 @ img_size=640 | ~1.8 GB |
| OSNet batch inference | ~0.4 GB |
| PyTorch overhead + CUDA context | ~0.5 GB |
| Buffer dự phòng | ~1.3 GB |
| **Tổng** | **~3.0 GB ✅ (trong 4 GB)** |

> **Lưu ý khi chạy cùng CARLA:** CARLA chiếm ~1.5 GB VRAM riêng.
> Khi đó tổng = 3.0 + 1.5 = 4.5 GB → vượt 4 GB.
> **Giải pháp:** Giảm batch size YOLOv8 xuống 3–4, hoặc dùng FP16 (`model.half()`).

---

## 3. Chi Tiết Từng Mô Hình

---

### 3.1 Detection — **YOLOv8s** (thay YOLOv5s)

#### Lý do chọn YOLOv8s thay vì các biến thể khác

| Model | mAP COCO | VRAM @ batch=6 | FPS thực tế | Ghi chú |
|---|---|---|---|---|
| YOLOv5s *(hiện tại)* | 37.4 | ~1.2 GB | ~50 | Thiếu nhãn VN |
| **YOLOv8n** | 37.3 | ~1.0 GB | ~55 | Nhanh nhưng mAP thấp |
| **YOLOv8s** ✅ | 44.9 | ~1.8 GB | ~40 | **Khuyến nghị** |
| YOLOv8m | 50.2 | ~3.2 GB | ~25 | Sát ngưỡng 4 GB, không an toàn |
| RT-DETR-L | 53.0 | ~4.0+ GB | ~15 | Vượt VRAM |

YOLOv8s tăng mAP từ 37 → 45 (+20%) trong khi vẫn trong budget VRAM.

#### Inference batched 6 cameras

```python
from ultralytics import YOLO

model = YOLO('yolov8s.pt')

# Batch 6 frame cùng lúc thay vì gọi 6 lần riêng lẻ
frames_batch = [cam1_frame, cam2_frame, cam3_frame,
                cam4_frame, cam5_frame, cam6_frame]

results = model(
    frames_batch,
    imgsz=640,
    conf=0.35,       # hạ nhẹ so với 0.4 để bắt xe máy nhỏ
    classes=[0, 2, 3, 5, 7],  # person, car, motorcycle, bus, truck
    device=0,        # GPU
    half=True        # FP16 để tiết kiệm VRAM (~1.0 GB thay 1.8 GB)
)

# results[i] = kết quả cho camera i
for cam_idx, result in enumerate(results):
    detections[f"CAM_{cam_idx+1:03d}"] = result.boxes
```

> Dùng `half=True` (FP16) khi chạy cùng CARLA — tiết kiệm ~800 MB VRAM.

#### Fine-tune cho giao thông Việt Nam

COCO có nhãn `motorcycle` (class 3) nhưng chủ yếu là xe máy phương Tây.
Xe máy Việt Nam (SH, Wave, Lead) có hình dạng khác.

**Dataset đề xuất:**
- **BDD100K** — 100k ảnh đường thực tế đa dạng (ngày/đêm/mưa)
- **Vietnam Traffic Dataset** trên Roboflow (~3k ảnh)
- Thu thập thêm từ video giao thông VN trên YouTube

**Nhãn cần có sau fine-tune:** `person / car / bus / truck / motorcycle / bicycle`

```python
# Fine-tune với batch nhỏ vừa 4 GB VRAM
model = YOLO('yolov8s.pt')
model.train(
    data='traffic_vn.yaml',
    epochs=50,
    imgsz=640,
    batch=8,           # vừa đủ 4 GB VRAM
    device=0,
    optimizer='AdamW',
    lr0=0.001,
    augment=True,      # mosaic, flip, hsv_v
    val=True
)
```

---

### 3.2 Per-Camera Tracking — **ByteTrack** (thay IoU Greedy)

#### Tại sao ByteTrack là upgrade quan trọng nhất

ByteTrack **chạy hoàn toàn trên CPU**, không tốn thêm VRAM, nhưng giải quyết
vấn đề gốc rễ của IoU Greedy:

| Vấn đề IoU Greedy | Cách ByteTrack giải quyết |
|---|---|
| Không predict vị trí khi bị che khuất | Kalman Filter built-in |
| ID swap khi 2 xe đi sát nhau | Dual-threshold: giữ track có confidence thấp |
| Greedy matching không tối ưu | Hungarian algorithm |

```python
# Tích hợp qua thư viện boxmot (pip install boxmot)
from boxmot import ByteTrack

tracker = ByteTrack(
    track_activation_threshold=0.25,
    lost_track_buffer=30,       # frames trước khi xóa track (~1s @ 30fps)
    minimum_matching_threshold=0.8,
    frame_rate=30               # cập nhật Kalman đúng dt
)

# Mỗi frame, mỗi camera:
tracks = tracker.update(
    dets=np.array([[x1, y1, x2, y2, conf, cls], ...]),
    img=frame
)
# tracks: array [x1, y1, x2, y2, track_id, conf, cls, idx]
```

**Khi chuyển sang video thực:**
- Đặt `frame_rate` đúng với FPS thực của camera IP
- Tăng `lost_track_buffer` nếu camera có frame drop nhiều

---

### 3.3 Cross-Camera Re-ID — **OSNet fine-tune VeRi-776** + Spatio-Temporal Filter

#### Vấn đề cốt lõi với OSNet hiện tại

OSNet hiện tại được pretrain trên **Market-1501** — dataset người đi bộ trong
trung tâm mua sắm. Feature vector học từ quần áo, dáng người — không phù hợp
để phân biệt xe theo màu sơn, hình dạng cabin, đèn xe.

Hai xe cùng màu trắng, cùng sedan → cosine similarity cao dù là 2 xe khác nhau.

#### Stack 2 tầng đề xuất

**Tầng 1 — Feature Matching (GPU):**

```python
# reid.py — mở rộng thành DualReIDExtractor
import torchreid

class DualReIDExtractor:
    def __init__(self):
        # Model cho người — giữ nguyên
        self.person_model = torchreid.models.build_model(
            name='osnet_x1_0',
            num_classes=751,           # Market-1501
            pretrained=True
        )
        # Model cho xe — fine-tune trên VeRi-776
        self.vehicle_model = torchreid.models.build_model(
            name='osnet_x1_0',
            num_classes=576,           # VeRi-776
            pretrained=False
        )
        self.vehicle_model.load_state_dict(
            torch.load('weights/osnet_veri776.pth')
        )

    def extract(self, frame, box, object_class):
        crop = self._crop(frame, box)
        if object_class == 'person':
            return self._forward(self.person_model, crop)
        else:   # car, bus, truck, motorcycle
            return self._forward(self.vehicle_model, crop)
```

**Fine-tune OSNet trên VeRi-776 với 4 GB VRAM:**

```python
datamanager = torchreid.data.ImageDataManager(
    root='reid-data',
    sources='veri776',
    targets='veri776',
    height=256, width=128,
    batch_size_train=32,   # vừa 4 GB VRAM với OSNet
    workers=4
)

model = torchreid.models.build_model(
    name='osnet_x1_0',
    num_classes=datamanager.num_train_pids,
    pretrained=True        # khởi đầu từ ImageNet, không phải từ đầu
)

optimizer = torchreid.optim.build_optimizer(model, optim='adam', lr=0.0003)
scheduler = torchreid.optim.build_lr_scheduler(optimizer, lr_scheduler='cosine', stepsize=20)

engine = torchreid.engine.ImageSoftmaxEngine(
    datamanager, model, optimizer=optimizer, scheduler=scheduler
)
engine.run(max_epoch=20, save_dir='log/osnet_veri')
# ~2-3 giờ trên GTX 1050Ti
```

**Tầng 2 — Spatio-Temporal Filter (CPU, zero VRAM):**

Loại bỏ các match "phi thực tế" — xe không thể teleport:

```python
# global_tracking.py — thêm filter
CAMERA_TOPOLOGY = {
    # (cam_a, cam_b): (min_travel_seconds, max_travel_seconds)
    # Đo thực tế hoặc từ CARLA bounding box distance
    ("CAM_001", "CAM_002"): (5,  30),
    ("CAM_001", "CAM_003"): (10, 60),
    ("CAM_002", "CAM_003"): (5,  25),
    ("CAM_002", "CAM_001"): (5,  30),   # 2 chiều
    ("CAM_003", "CAM_001"): (10, 60),
    ("CAM_003", "CAM_002"): (5,  25),
    # ... thêm cho 6 camera
}

def is_feasible_transition(cam_a: str, t_a: float,
                            cam_b: str, t_b: float) -> bool:
    """
    Kiểm tra xem đối tượng có đủ thời gian đi từ cam_a sang cam_b không.
    t_a, t_b: Unix timestamp (float seconds)
    """
    key = (cam_a, cam_b)
    if key not in CAMERA_TOPOLOGY:
        return False    # không có đường nối trực tiếp
    min_t, max_t = CAMERA_TOPOLOGY[key]
    delta = abs(t_b - t_a)
    return min_t <= delta <= max_t

# Trong GlobalTracker.match_cross_camera():
# Trước khi accept một cross-camera match:
if not is_feasible_transition(cam_a, last_seen_a, cam_b, now):
    similarity = 0.0   # reject match
```

> **Khi dùng video thực:** Đo khoảng cách thực địa giữa các camera và tốc độ
> tối đa cho phép để tính `min_travel_seconds = distance / max_speed`.

---

### 3.4 Trajectory Prediction — **Kalman Filter** (thay Linear Extrapolation)

#### Phân tích lỗi trong code hiện tại

**Lỗi 1 — `window_size` không có tác dụng:**

```python
# trajectory_predictor.py dòng 91-101
# window_size=10 khai báo nhưng _linear_prediction dùng 2 điểm cuối:
velocity = last_pos - prev_pos   # pos[-1] - pos[-2]
# → 8 điểm trong window bị bỏ qua hoàn toàn

# get_velocity() (dòng 157-164) có tính trung bình đúng
# nhưng predict() KHÔNG gọi get_velocity() → window_size là trang trí
```

**Lỗi 2 — Đơn vị pixel/frame, không phải pixel/giây:**

```python
# velocity = pixels/frame — phụ thuộc vào FPS
# Khi FPS thay đổi (rất phổ biến với RTSP thực), velocity sai

# SimpleTracker đã tính đúng (tracker.py dòng 74-80):
dt = (timestamps[-1] - timestamps[-2]).total_seconds()
track['speeds'].append(d / dt)   # px/s ← đúng

# Nhưng TrajectoryPredictor nhận frame_idx, không dùng timestamps
# → Hai module không nhất quán đơn vị
```

**Lỗi 3 — Prediction horizon quá ngắn:**

```python
# pred_steps=5 → 5 frames → ~0.17 giây ở 30fps
# → Gần như vô nghĩa để cảnh báo sớm
# Cần predict 1–3 giây thực tế
```

**Lỗi 4 — Hoàn toàn tách biệt với IncidentDetector:**

```python
# incident_detector.py chỉ nhìn tốc độ hiện tại:
speed = self._current_speed(gid)   # quá khứ → phản ứng sau sự cố

# Không có câu hỏi: "2 giây nữa xe này sẽ ở đâu?"
# → Không có proactive alert
```

#### Giải pháp — Kalman từ ByteTrack + timestamp-based prediction

**Bước 1 — Sửa TrajectoryPredictor dùng timestamp:**

```python
# trajectory_predictor.py — viết lại _linear_prediction thành _time_based_prediction

def update_trajectory(self, global_id, position, timestamp: float):
    """
    Args:
        timestamp: time.time() — Unix float seconds (KHÔNG phải frame_idx)
    """
    if global_id not in self.trajectories:
        self.trajectories[global_id] = {
            'positions':  deque(maxlen=self.window_size),
            'timestamps': deque(maxlen=self.window_size)  # THÊM
        }
    self.trajectories[global_id]['positions'].append(position)
    self.trajectories[global_id]['timestamps'].append(timestamp)

def _time_based_prediction(self, positions, timestamps):
    """
    Tính velocity có trọng số, predict theo giây thực tế.
    Điểm gần nhất có trọng số cao hơn (exponential decay).
    """
    pos_arr = np.array(positions)   # (N, 2)
    t_arr   = np.array(timestamps)  # (N,)

    if len(pos_arr) < 2:
        return None

    dt  = np.diff(t_arr)           # (N-1,) khoảng thời gian giữa điểm
    dxy = np.diff(pos_arr, axis=0) # (N-1, 2) displacement

    # Tránh chia 0 khi dt quá nhỏ
    mask = dt > 1e-6
    if not mask.any():
        return None

    velocities = dxy[mask] / dt[mask, None]  # px/s

    # Trọng số: điểm gần nhất quan trọng hơn
    weights = np.exp(np.linspace(-2, 0, len(velocities)))
    velocity = np.average(velocities, axis=0, weights=weights)  # px/s

    # Predict tại các mốc thời gian thực
    predict_at_seconds = [0.5, 1.0, 1.5, 2.0, 3.0]
    last_pos = pos_arr[-1]
    return [
        {'t': t, 'pos': (last_pos + velocity * t).tolist()}
        for t in predict_at_seconds
    ]
```

**Bước 2 — Tái dùng Kalman state từ ByteTrack (không cần model riêng):**

ByteTrack đã chạy Kalman Filter cho mỗi track. State vector = `[x, y, w, h, vx, vy, vw, vh]`.
Tái dùng `vx, vy` này để predict thay vì tính riêng:

```python
# Sau khi update ByteTrack:
for strack in tracker.tracked_stracks:
    track_id = strack.track_id
    vx = strack.mean[4]   # velocity x (pixels/frame từ Kalman)
    vy = strack.mean[5]   # velocity y
    cx, cy = strack.tlbr_center   # vị trí hiện tại

    # Convert velocity từ px/frame sang px/s
    vx_per_sec = vx * fps
    vy_per_sec = vy * fps

    predictions[track_id] = {
        't+0.5s': [cx + vx_per_sec * 0.5, cy + vy_per_sec * 0.5],
        't+1s':   [cx + vx_per_sec * 1.0, cy + vy_per_sec * 1.0],
        't+2s':   [cx + vx_per_sec * 2.0, cy + vy_per_sec * 2.0],
        't+3s':   [cx + vx_per_sec * 3.0, cy + vy_per_sec * 3.0],
    }
```

> Lợi ích: ByteTrack's Kalman đã được update liên tục, state ổn định, không cần
> maintain thêm một bộ Kalman riêng trong TrajectoryPredictor.

---

### 3.5 Incident Detection — **Rule-based cải tiến** → Isolation Forest

#### Fix ngay — FPS-normalized speed

Vấn đề hiện tại: `speed > 120 px/s` nhạy cảm với FPS không ổn định.

```python
# incident_detector.py — sửa _current_speed()

def _current_speed(self, gid: int) -> float | None:
    history = list(self.speed_history[gid])
    if len(history) < 2:
        return None

    # Lấy N điểm gần nhất, tính trung bình có trọng số
    recent = history[-5:]
    speeds = [h[1] for h in recent if h[1] is not None]
    if not speeds:
        return None

    # Exponential moving average để chống noise
    weights = np.exp(np.linspace(-1, 0, len(speeds)))
    return float(np.average(speeds, weights=weights))
```

#### Proactive alert dùng predicted positions

Thay vì chỉ phản ứng khi va chạm xảy ra, dùng trajectory prediction để cảnh báo trước:

```python
# incident_detector.py — thêm method mới

def check_predicted_collision(self, tracks, predictions, camera_id, now):
    """
    Kiểm tra xem quỹ đạo dự đoán có dẫn đến va chạm trong 1-3 giây không.
    predictions: {global_id: {'t+1s': [x,y], 't+2s': [x,y], ...}}
    """
    incidents = []
    vehicles    = [t for t in tracks if t['class'] in ('car', 'truck', 'bus', 'motorcycle')]
    pedestrians = [t for t in tracks if t['class'] == 'person']

    for v in vehicles:
        gid = v['global_id']
        v_preds = predictions.get(gid, {})

        for ped in pedestrians:
            ped_center = self._box_center(ped['box'])

            for horizon, pred_pos in v_preds.items():
                dist = np.linalg.norm(
                    np.array(pred_pos) - np.array(ped_center)
                )
                # Ngưỡng rộng hơn cho dự đoán (có sai số)
                if dist < self.ped_proximity_px * 1.5:
                    incidents.append(self._make(
                        type='PREDICTED_COLLISION',
                        severity='CRITICAL',
                        gid=gid, cam=camera_id,
                        msg=(f"Dự đoán va chạm: Xe #{gid} → Người #{ped['global_id']} "
                             f"sau {horizon} (dist={dist:.0f}px)"),
                        details={
                            'horizon': horizon,
                            'predicted_pos': pred_pos,
                            'pedestrian_pos': ped_center,
                            'predicted_distance': round(dist, 1),
                        }
                    ))
    return incidents

def check_predicted_roi_entry(self, tracks, predictions, rois, camera_id, now):
    """
    Cảnh báo trước khi đối tượng vào ROI nguy hiểm.
    """
    incidents = []
    for track in tracks:
        gid = track['global_id']
        preds = predictions.get(gid, {})
        for roi in rois.get(camera_id, []):
            for horizon, pred_pos in preds.items():
                if point_in_polygon(pred_pos, roi['polygon']):
                    incidents.append(self._make(
                        type='PREDICTED_ROI_ENTRY',
                        severity='WARNING',
                        gid=gid, cam=camera_id,
                        msg=f"Dự đoán đối tượng #{gid} vào ROI '{roi['name']}' sau {horizon}",
                        details={'horizon': horizon, 'roi': roi['name']}
                    ))
    return incidents
```

#### Nâng cấp dài hạn — Isolation Forest (sau khi có data thực)

```python
from sklearn.ensemble import IsolationForest

# Bước 1: Thu thập 10-30 phút behavior bình thường từ camera thực
# Feature vector mỗi object mỗi frame:
# [speed_px_s, acceleration, turn_rate, density_nearby, time_of_day]

# Bước 2: Fit model
clf = IsolationForest(contamination=0.03, random_state=42)
clf.fit(normal_behavior_features)    # offline
joblib.dump(clf, 'weights/anomaly_detector.pkl')

# Bước 3: Online inference (cực nhẹ, CPU)
clf = joblib.load('weights/anomaly_detector.pkl')
score = clf.predict([[speed, accel, turn, density, hour]])
# -1 = anomaly → trigger incident review
# +1 = normal behavior
```

---

## 4. Tóm Tắt Production Stack

| Tầng | CARLA *(hiện tại)* | **Production Stack** | VRAM | Trạng thái |
|---|---|---|---|---|
| **Detection** | YOLOv5s (COCO) | **YOLOv8s** fine-tuned VN | ~1.8 GB | ⏳ Chưa làm |
| **Tracking** | IoU Greedy | **ByteTrack** | CPU | ⏳ Chưa làm |
| **Re-ID** | OSNet (Market-1501) | **OSNet** fine-tune VeRi-776 + Spatio-Temporal | ~0.4 GB | ⏳ Chưa làm |
| **Trajectory** | Linear Extrap (có lỗi) | **Kalman** từ ByteTrack + time-based | CPU | ⏳ Chưa làm |
| **Incident** | Rule-based cứng | **Rule-based FPS-normalized** + proactive | CPU | ⏳ Fix cần làm |
| **Video input** | CARLA API | **RTSP / File** (VideoSource đã có) | — | ✅ Sẵn sàng |

---

## 5. Lộ Trình Implement Theo Độ Ưu Tiên

| # | Việc | Thời gian | Lợi ích | VRAM thêm |
|---|---|---|---|---|
| 1 | YOLOv5s → **YOLOv8s** (đổi model) | 1 giờ | mAP +20%, thêm motorcycle | 0 |
| 2 | IoU Greedy → **ByteTrack** | 2 ngày | Giảm ID swap, Kalman built-in | CPU only |
| 3 | Sửa **FPS-normalized speed** + proactive incident | 1 ngày | Giảm false positive, thêm predict_collision | CPU only |
| 4 | Thêm **Spatio-Temporal Filter** (50 dòng code) | 1 ngày | Giảm cross-camera false match | CPU only |
| 5 | Sửa `TrajectoryPredictor` dùng **timestamp** | 1 ngày | Velocity đúng đơn vị | CPU only |
| 6 | Fine-tune **YOLOv8s** trên dataset VN | 3–5 ngày | Nhận diện xe máy VN chính xác | ~2 GB |
| 7 | Fine-tune **OSNet trên VeRi-776** | 3–5 ngày | Re-ID xe chính xác hơn | ~2–3 GB |
| 8 | **Isolation Forest** anomaly detection | 2 ngày | Adaptive threshold, ít false positive | CPU only |

> **Thứ tự thực tế:** Làm #1 + #2 trước — đây là 2 upgrade không cần training,
> ít code nhất nhưng lợi ích lớn nhất (track ổn định hơn → Re-ID chính xác hơn → ít false alert).

---

## 6. Migration từ CARLA sang Video Thực

### 6.1 Thay đổi duy nhất cần làm

`video_source.py` đã có lớp trừu tượng đầy đủ. Chỉ cần thay đổi config:

```yaml
# config/cameras.yaml — thay đổi duy nhất khi chuyển sang real CCTV
cameras:
  - id: CAM_001
    name: "Ngã tư Lê Lợi - Hùng Vương"
    # source_type: carla       ← xóa/comment dòng này
    source_type: rtsp           # ← thêm dòng này
    source_url: "rtsp://192.168.1.10:554/stream1"
    resolution: [1920, 1080]
    fps: 25
    reconnect_delay: 5          # giây, tự reconnect khi đứt

  - id: CAM_002
    source_type: rtsp
    source_url: "rtsp://192.168.1.11:554/stream1"
    # ...
```

AI pipeline (detector → tracker → reid → global tracker → incident detector) **không cần sửa**
vì tất cả chỉ nhận `numpy.ndarray` (frame ảnh) — không quan tâm nguồn gốc.

### 6.2 Điều chỉnh khi dùng camera thực

**A. Calibrate tốc độ pixel → km/h thực tế:**

```python
# Đặt 2 điểm tham chiếu trong thực địa với khoảng cách đã biết
# VD: 2 vạch kẻ đường cách nhau 5 mét
REF_POINT_A = (320, 450)   # pixel
REF_POINT_B = (480, 450)   # pixel
REAL_DISTANCE_METERS = 5.0

pixels_per_meter = np.linalg.norm(
    np.array(REF_POINT_A) - np.array(REF_POINT_B)
) / REAL_DISTANCE_METERS

# speed px/s → km/h
def px_per_sec_to_kmh(speed_px_s):
    return speed_px_s / pixels_per_meter * 3.6
```

**B. Đo topology camera cho Spatio-Temporal Filter:**

```python
# Đo thực địa hoặc dùng bản đồ:
CAMERA_TOPOLOGY = {
    ("CAM_001", "CAM_002"): {
        "distance_meters": 150,
        "min_travel_sec": 150 / (120/3.6),   # 120 km/h tối đa
        "max_travel_sec": 150 / (5/3.6),     # 5 km/h tối thiểu
    },
    # ...
}
```

**C. Xử lý ánh sáng thấp (ban đêm):**

```python
# Tiền xử lý frame trước khi đưa vào YOLOv8
import cv2

def enhance_low_light(frame):
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    l = clahe.apply(l)
    enhanced = cv2.merge([l, a, b])
    return cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)

# Chỉ áp dụng khi mean brightness < ngưỡng
if frame.mean() < 80:
    frame = enhance_low_light(frame)
```

**D. Xử lý RTSP stream drop:**

```python
# video_source.py — đã có RTSPVideoSource, cần thêm reconnect logic
class RTSPVideoSource:
    def get_frame(self):
        ret, frame = self.cap.read()
        if not ret:
            logger.warning(f"{self.camera_id}: stream lost, reconnecting...")
            time.sleep(self.reconnect_delay)
            self.cap = cv2.VideoCapture(self.source_url)
            ret, frame = self.cap.read()
        return frame if ret else None
```

### 6.3 Dashboard — Hiển thị Trajectory Production

```
Hiện tại (CARLA):   5 chấm xanh thẳng hàng (trang trí)

Production:
  ●──── Bounding box hiện tại (màu theo loại: xanh=person, vàng=car, đỏ=truck)
  ·····  Đường chấm dự đoán (màu: xanh→vàng→đỏ theo thời gian t+0.5s → t+3s)
  ⚠     Icon tại điểm predicted nếu sẽ va chạm với pedestrian
  🔴    Flash ROI nếu predicted path đi qua ROI nguy hiểm trong 2 giây tới
```

---

## 7. Ghi Chú Kỹ Thuật

### Dependencies cần thêm

```txt
# Tracker
boxmot>=10.0          # ByteTrack và nhiều tracker khác, pip install boxmot

# Kalman Filter (nếu implement riêng)
filterpy>=1.4

# Re-ID fine-tune
torchreid             # cài từ source: https://github.com/KaiyangZhou/deep-person-reid

# Anomaly Detection
scikit-learn>=1.3     # Isolation Forest

# Image Enhancement
opencv-python>=4.8    # đã có

# FP16 inference
torch>=2.0            # đã có, dùng model.half()
```

### Cấu hình môi trường chạy production

```python
# Tối ưu GPU utilization cho GTX 1050Ti
import torch
torch.backends.cudnn.benchmark = True    # tự chọn kernel nhanh nhất
torch.backends.cudnn.deterministic = False
torch.set_float32_matmul_precision('medium')  # tăng tốc matmul nhẹ

# Nếu VRAM sát ngưỡng
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128'
```

---

*Tài liệu tạo: 2026-05-27. Xem thêm: `upgrade.md` (lộ trình chi tiết), `bao_cao_tien_do.md` (tiến độ tổng thể).*
