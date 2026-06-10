# BÁO CÁO TIẾN ĐỘ DỰ ÁN

## 1. Tên Dự Án

**Hệ Thống Giám Sát Camera CCTV Đa Kênh Ứng Dụng Trí Tuệ Nhân Tạo**
*(Multi-Camera CCTV Tracking System with AI)*

Môi trường mô phỏng: CARLA Simulator phiên bản 0.9.9.4 (vấn đề tương thích Python đã được giải quyết bằng kiến trúc `carla_bridge` — xem mục 7 item 2).

---

## 2. Tổng Quan Dự Án

Dự án được tổ chức thành **4 thành phần chính**:

```
finalproject/
├── custom_tracking_system/   # AI Pipeline — xử lý video và nhận diện đối tượng
├── server/                   # Backend API — REST, WebSocket, MJPEG streaming
├── frontend/                 # Web Dashboard — giao diện giám sát real-time (React)
└── WindowsNoEditor/          # CARLA Simulator — môi trường mô phỏng đô thị
```

### 2.1 Luồng Hoạt Động Tổng Thể

```
CARLA Simulator
      │  (frame video từ camera ảo)
      ▼
AI Pipeline (custom_tracking_system/)
  Detect → Track → ReID → Global ID → Trajectory → Incident Detection
      │  (kết quả: tracks, alerts, incidents, evidence)
      ▼
Backend Server (server/)
  REST API + WebSocket + MJPEG
      │  (HTTP / WebSocket)
      ▼
Web Dashboard (frontend/)
  Camera Grid · Incident Panel · Alert Management
```

---

## 3. Mục Tiêu Dự Án

1. **Phát hiện đối tượng** (xe ô tô, xe tải, xe buýt, xe máy, người đi bộ) từ nhiều camera đồng thời bằng YOLOv8s.
2. **Theo dõi đối tượng trong từng camera** bằng ByteTrack (Kalman Filter + Hungarian matching).
3. **Nhận diện lại xuyên camera** (Cross-Camera Re-ID) — gán Global ID duy nhất bằng OSNet với model riêng cho người (Market-1501) và xe (VeRi-776).
4. **Dự đoán quỹ đạo** tại 5 mốc thời gian thực (0.5s–3.0s) bằng Kalman Filter constant-acceleration (`KalmanFilter2D`).
5. **Phát hiện 12 loại sự cố** real-time (10 reactive + 2 proactive).
6. **API chuẩn** (REST + WebSocket + MJPEG) để frontend và ứng dụng khác truy cập.
7. **Lớp trừu tượng VideoSource** để chuyển từ CARLA sang camera IP thực mà không viết lại pipeline.

---

## 4. Trạng Thái Các Mô Hình AI

### Mô hình 1 — Phát hiện: **YOLOv8s** ✅ Hoàn thành

| | |
|---|---|
| **Cấu hình** | `yolov8s`, conf ≥ 0.35, FP16, **5 lớp**: `person / car / motorcycle / bus / truck` |
| **Ưu điểm** | mAP COCO 44.9 (+20% so YOLOv5s 37.4); batched inference cho 6 camera |
| **Hạn chế** | Weights COCO chung — xe máy Việt Nam nhận diện kém hơn xe Tây |
| **Cải tiến tiếp theo** | Fine-tune trên BDD100K + Vietnam Traffic Dataset để thêm motorcycle VN |

### Mô hình 2 — Tracker trong camera: **ByteTrackWrapper** ✅ Hoàn thành

| | |
|---|---|
| **Cấu hình** | `boxmot.ByteTrack`, `track_activation_threshold=0.25`, `lost_track_buffer=30`, `minimum_matching_threshold=0.8`, `frame_rate=10` |
| **Ưu điểm** | Kalman Filter (dự đoán vị trí khi bị che); Hungarian matching (optimal); dual-threshold chống ID swap; CPU-only |
| **Cải tiến tiếp theo** | Nâng lên StrongSORT nếu cần dùng chung feature với ReID |

### Mô hình 3 — Re-ID xuyên camera: **DualReIDExtractor** ✅ Hoàn thành + VeRi-776 active

| | |
|---|---|
| **Cấu hình** | `person_model`: OSNet x1.0 pretrained Market-1501 (512D); `vehicle_model`: OSNet x1.0 fine-tuned VeRi-776 (575 classes, epoch 12) |
| **Ưu điểm** | Model riêng cho người và xe; L2-normalized cosine similarity; Spatio-Temporal Filter lọc match phi thực tế |
| **Spatio-Temporal Filter** | `camera_topology` đã cấu hình cho 6 cặp CAM_001/002/003 (min=1s, max=120s) |
| **Hạn chế** | CARLA không có biển số xe → không thể dùng LPR để phân biệt chính xác hơn |
| **Cải tiến tiếp theo** | Re-train VeRi-776 với validation split đúng (đã sửa `train_veri.py`) để có val_acc có nghĩa |

### Mô hình 4 — Dự đoán quỹ đạo: **Kalman Filter (Constant-Acceleration)** ✅ Hoàn thành

| | |
|---|---|
| **Cấu hình** | `KalmanFilter2D` per-object, state `[x, y, vx, vy, ax, ay]`, `window_size=10`, dt-aware (dùng unix timestamp thực), 5 mốc: `t+0.5s` → `t+3.0s` |
| **Ưu điểm** | Xử lý được rẽ/tăng/giảm tốc (constant-acceleration model); làm mượt nhiễu detection/tracking qua bước update; không phụ thuộc FPS cố định |
| **Hạn chế** | Toạ độ vẫn ở **pixel-space riêng từng camera** — chưa map sang world/GPS coordinates nên chưa "lên map" được; `process_noise_std`/`measurement_noise_std` cần tune theo dữ liệu thực tế |
| **Cải tiến tiếp theo** | Calibration pixel↔world (homography) để dự đoán trên 1 hệ toạ độ chung; tune noise params bằng kịch bản CARLA + ground_truth |

### Mô hình 5 — Phát hiện sự cố: **Rule-based** ✅ Hoàn thành

| | |
|---|---|
| **Cấu hình** | **12 loại**: 10 reactive (SUDDEN_STOP, SUDDEN_ACCEL, OVERSPEED, VEHICLE_PROXIMITY, PEDESTRIAN_DANGER, STOPPED_VEHICLE, LOITERING, CROWD_DENSITY, WRONG_WAY, CAMERA_TRANSITION) + 2 proactive (PREDICTED_COLLISION, PREDICTED_ROI_ENTRY); cooldown 4s |
| **Ưu điểm** | Hoạt động ngay, dễ điều chỉnh ngưỡng, dễ giải thích |
| **Hạn chế** | Ngưỡng cứng nhạy cảm với noise; không tự thích nghi |
| **Cải tiến tiếp theo** | Anomaly Detection (Isolation Forest / Autoencoder) tự học phân phối bình thường |

---

## 5. Công Việc Đã Hoàn Thành

### 5.1 AI Pipeline (`custom_tracking_system/`)

| STT | Module | Trạng thái | Ghi chú |
|-----|--------|-----------|---------|
| 1 | `camera_controller.py` | ✅ Hoàn thành | Điều khiển và đồng bộ nhiều camera CARLA |
| 2 | `traffic_generator.py` | ✅ Hoàn thành | Sinh xe + người tự động |
| 3 | `detector.py` | ✅ Hoàn thành | YOLOv8s, 5 class, FP16, batched inference |
| 4 | `tracker.py` | ✅ Hoàn thành | ByteTrackWrapper (Kalman + Hungarian + dual-threshold) |
| 5 | `reid.py` | ✅ Hoàn thành | DualReIDExtractor: OSNet Market-1501 (person) + VeRi-776 (vehicle) |
| 6 | `global_tracking.py` | ✅ Hoàn thành | Global ID + Spatio-Temporal Filter với camera_topology cấu hình |
| 7 | `trajectory_predictor.py` | ✅ Hoàn thành | Kalman Filter constant-acceleration (`KalmanFilter2D`), dt-aware, 5 horizons |
| 8 | `alert_system.py` | ✅ Hoàn thành | ROI warning (point-in-polygon), nhận list positions từ predict_list() |
| 9 | `incident_detector.py` | ✅ Hoàn thành | 12 loại sự cố, proactive alerts được kích hoạt đúng |
| 10 | `evidence_package.py` | ✅ Hoàn thành | Ring buffer 30s, lưu clip + ảnh + metadata JSON khi sự cố |
| 11 | `scenario_controller.py` | ✅ Hoàn thành | 5 kịch bản tai nạn trong CARLA |
| 12 | `video_source.py` | ✅ Hoàn thành | Lớp trừu tượng CARLA/RTSP/File/Webcam + `CARLABridgeClient` (nhận frame từ carla_bridge qua TCP) |
| 13 | `ground_truth.py` | ✅ Hoàn thành | Thu thập GT từ CARLA (tách biệt AI pipeline) |
| 14 | `carla_bridge/server.py` | ✅ Hoàn thành | Process Python 3.7 riêng: chạy CARLA PythonAPI, đồng bộ camera + traffic, stream frame qua TCP socket cho `main.py --source bridge` (Python 3.10+) |

### 5.2 Backend API Server (`server/`)

| Thành phần | Trạng thái | Ghi chú |
|-----------|-----------|---------|
| 17 REST endpoints | ✅ Hoàn thành | Cameras, tracks, alerts, ROIs, stats |
| WebSocket `/ws/alerts` | ✅ Hoàn thành | Push incident real-time (thread-safe) |
| WebSocket `/ws/tracks` | ✅ Hoàn thành | Push tracking updates |
| WebSocket `/ws/stats` | ✅ Hoàn thành | Push `{fps, active_tracks, alerts_today, status}` mỗi 1s qua `_push_stats_ws()` |
| MJPEG streaming | ✅ Hoàn thành | Thread-safe `asyncio.Event` qua `call_soon_threadsafe` |
| SQLite + SQLAlchemy | ✅ Hoàn thành | 5 bảng, indexes đầy đủ |
| `ai_processor.py` | ✅ Hoàn thành (đã sửa bugs) | YOLOv8s, ByteTrackWrapper, DualReIDExtractor, predictions wired |

### 5.3 Web Dashboard (`frontend/`)

| Tính năng | Trạng thái |
|-----------|-----------|
| Camera Grid — MJPEG live feed | ✅ Hoàn thành |
| Incident Panel — alerts real-time | ✅ Hoàn thành |
| Alert Management — lọc, xác nhận | ✅ Hoàn thành |
| Object History — lịch sử di chuyển | ✅ Hoàn thành |
| Âm thanh + flash màn hình khi CRITICAL | ✅ Hoàn thành |
| WebSocket auto-reconnect | ✅ Hoàn thành |
| Vite proxy `/api`, `/ws`, `/stream` → backend | ✅ Hoàn thành |

### 5.4 Bug Fixes (cập nhật 2026-06-01)

Các lỗi được phát hiện và sửa trong phiên này:

| # | File | Lỗi | Fix |
|---|------|-----|-----|
| 1 | `ai_processor.py` | `frame_count` (int) truyền vào TrajectoryPredictor thay vì unix timestamp → velocity vô nghĩa | → `time.time()` |
| 2 | `ai_processor.py` | `predict()` trả `dict` nhưng `check_alerts()` nhận `list` → alert check sai | → `list(predicted.values())` |
| 3 | `ai_processor.py` | `model_type="yolov5s"` → dùng model sai | → `"yolov8s"` |
| 4 | `ai_processor.py` | `SimpleTracker` thay vì `ByteTrackWrapper` → tracker cũ không có Kalman | → `ByteTrackWrapper(...)` + thêm `frame` arg |
| 5 | `ai_processor.py` | `incident_detector.update()` không nhận `predictions` → 2 proactive alert (PREDICTED_COLLISION, PREDICTED_ROI_ENTRY) không bao giờ kích hoạt | → truyền `predictions_map` và `rois` |
| 6 | `ai_processor.py` | `_event_loop` không có guard → `AttributeError` nếu `set_event_loop()` chưa được gọi | → `hasattr()` check |
| 7 | `ai_processor.py` | `ReIDExtractor()` thay vì `DualReIDExtractor` → không dùng VeRi-776 | → `DualReIDExtractor(vehicle_weights=<absolute_path>)` |
| 8 | `ai_processor.py` | `GlobalTracker` không có `camera_topology` → Spatio-Temporal Filter vô hiệu | → đọc từ YAML + parse tuple key |
| 9 | `stream_service.py` | `asyncio.Event.set()` gọi từ AI thread → threading violation | → `loop.call_soon_threadsafe(event.set)` |
| 10 | `app.py` | `FrameBuffer` không biết event loop → `set_loop()` chưa được gọi | → gọi `frame_buffer.set_loop()` trong lifespan |
| 11 | `config.py` | `DETECTOR_MODEL = "yolov5s"` | → `"yolov8s"` |
| 12 | `camera_config.yaml` | `camera_topology` thiếu | → thêm 6 cặp camera với min/max travel time |
| 13 | `train_veri.py` | Validation dùng `image_test/` (cross-ID class space) → val_acc luôn ~0% dù model tốt | → 90/10 split từ `image_train/` (same class space) |

---

## 6. Hạng Mục Chưa Hoàn Thành

| Hạng mục | Độ ưu tiên | Ghi chú |
|---------|-----------|---------|
| ~~**WebSocket `/ws/stats` push định kỳ**~~ | ✅ Hoàn thành | `_push_stats_ws()` gọi mỗi 1s từ FPS block; StatsBar hiển thị fps/active_tracks/alerts_today thực tế |
| **Fine-tune YOLOv8s cho giao thông VN** | Trung bình | Cần dataset BDD100K + Vietnam Traffic (Roboflow); motorcycle VN nhận diện kém hơn COCO |
| **Ground Truth Evaluation** | Trung bình | `ground_truth.py` đã có, chưa kết nối `motmetrics` để tính MOTA/IDF1/mAP định lượng |
| **Recording liên tục** | Thấp | `EvidencePackage` chỉ lưu khi có sự cố; chưa ghi liên tục theo segment 1h + storage rotation |
| **Camera Management UI** | Thấp | Chưa có giao diện thêm/xóa camera runtime; API endpoint đã có |
| **VideoSource tích hợp vào pipeline** | Thấp | `main.py --source bridge` đã dùng `CARLABridgeClient`; `ai_processor.py` (server) vẫn dùng `CameraController` trực tiếp |
| **Chọn epoch tốt nhất từ 20 checkpoints** | Thấp | Không cần re-train — 20 checkpoint đã lưu (`ckpt_ep01`→`ckpt_ep20`); cosine annealing → epoch 20 thường tốt nhất; hoặc chạy Re-ID eval (rank-1/mAP trên query/gallery VeRi) để chọn chính xác |

---

## 7. Khó Khăn

**(1) Vehicle ReID trong CARLA thiếu biển số xe**
CARLA 0.9.9.4 không cung cấp biển số đọc được → không thể dùng LPR để phân biệt chính xác xe giống nhau. VeRi-776 trained model cải thiện đáng kể so với person model nhưng vẫn có nhầm lẫn với xe cùng màu/kiểu.

**(2) Xung đột Python/CARLA — vấn đề môi trường quan trọng**
CARLA 0.9.9.4 yêu cầu Python 3.7 (file `.egg` biên dịch cho Python 3.7). Tuy nhiên `boxmot>=10.0` (ByteTrack) và `ultralytics>=8.0` (YOLOv8) **yêu cầu Python 3.8+** — không thể cùng hoạt động trong 1 môi trường Python 3.7. Ngoài ra, syntax `X | Y` và `dict[str, ...]` trong một số file (`ai_processor.py`, `websocket.py`, ...) yêu cầu Python 3.10+/3.9+.

**Giải pháp đã triển khai — kiến trúc `carla_bridge` (2 process song song):**
Thay vì nâng cấp CARLA, dự án tách thành 2 tiến trình Python độc lập giao tiếp qua TCP socket:
- **`carla_bridge/server.py`** (venv Python 3.7, dùng CARLA `.egg` 0.9.9.4 hiện có): kết nối CARLA, bật synchronous mode (10 FPS), tạo `CameraController` + `TrafficGenerator`, mỗi tick gửi frame đồng bộ của tất cả camera qua TCP (binary wire format: số camera, rồi mỗi camera gồm `camera_id`, timestamp, kích thước, raw frame bytes).
- **`main.py --source bridge`** (venv Python 3.10+, đầy đủ `boxmot`/`ultralytics`/`torchreid`): dùng `CARLABridgeClient` (trong `video_source.py`) để nhận frame qua TCP, sau đó chạy toàn bộ AI pipeline (Detect → Track → ReID → Global ID → Trajectory → Incident).

Ưu điểm: không cần tải CARLA 0.9.15 (~12 GB), không cần sửa type annotations, không cần venv 3.8 cho CARLA — mỗi process dùng đúng phiên bản Python mà thư viện của nó yêu cầu. Chế độ cũ `--source carla` (chạy in-process, cần Python tương thích cả CARLA lẫn AI libs) vẫn được giữ làm fallback.

**Giải pháp thay thế (chưa triển khai — Hướng 3):** Nâng cấp lên CARLA 0.9.15 — phiên bản đầu tiên hỗ trợ Python 3.8 natively. Các bước: (1) Tải CARLA 0.9.15 (~12 GB); (2) Tạo venv Python 3.8, cài đủ `boxmot`, `ultralytics`, `torchreid`; (3) Thêm `from __future__ import annotations` vào 4 file có type annotation incompatible (`ai_processor.py`, `stream_service.py`, `websocket.py`, `trajectory_predictor.py`); (4) Sửa `traffic_generator.py` dòng `set_autopilot(True)` → `set_autopilot(True, 8000)` (TM port thay đổi từ CARLA 0.9.14+). Hướng này vẫn có thể cân nhắc nếu sau này cần chạy mọi thứ trong 1 process duy nhất.

Với RTX 3060 Laptop (6 GB VRAM, Ampere): pipeline đầy đủ tốn ~3.2 GB VRAM → còn ~2.8 GB dự phòng cho CARLA. FP16 trên Ampere tăng tốc thực sự nhờ Tensor Cores (2× so với FP32).

**(3) Trajectory prediction — đã chuyển sang Kalman Filter, còn thiếu world-coordinate mapping**
`trajectory_predictor.py` đã đổi từ exp-weighted velocity sang **Kalman Filter constant-acceleration** (`KalmanFilter2D`, state `[x,y,vx,vy,ax,ay]`) — xử lý được rẽ/tăng/giảm tốc và làm mượt nhiễu tracking. Tuy nhiên dự đoán vẫn ở **pixel-space riêng từng camera**, chưa có bước calibration (homography pixel↔world) nên chưa thể hiển thị quỹ đạo dự đoán lên 1 bản đồ chung — đây là việc cần làm tiếp để phục vụ định hướng "dự đoán hướng đi trên map".

**(4) val_acc của VeRi-776 training không có nghĩa**
Bug trong `train_veri.py`: validation dùng `image_test/` có vehicle ID khác hoàn toàn với training → accuracy luôn ~0% dù model hoạt động tốt. **Đã sửa** — lần re-train tiếp theo sẽ cho val_acc chính xác.

**(5) Chưa có ground truth định lượng**
Hiện chỉ đánh giá định tính qua quan sát. Module `ground_truth.py` có nhưng chưa nối `motmetrics` → không có MOTA, IDF1, mAP để so sánh khách quan.

---

## 8. Kế Hoạch Tiếp Theo (Theo Thứ Tự Ưu Tiên)

1. ~~**Giải quyết xung đột Python/CARLA**~~ ✅ Hoàn thành — kiến trúc `carla_bridge` (2 process: Python 3.7 cho CARLA, Python 3.10+ cho AI pipeline qua TCP). Chi tiết tại mục 7 item (2). Cần test end-to-end thực tế (chạy `carla_bridge/server.py` + `main.py --source bridge` đồng thời) để xác nhận hoạt động ổn định.

2. **Ground Truth Evaluation** — kết nối `ground_truth.py` + `motmetrics` để ra MOTA/IDF1/mAP định lượng; chạy ít nhất 1 scenario để có baseline metrics.

3. **Fine-tune YOLOv8s cho VN** — thu thập ảnh motorcycle VN từ Roboflow, fine-tune trên BDD100K + dataset VN. Xem hướng dẫn `docs/production_model_stack.md`.

4. **Re-train VeRi-776** — dùng `train_veri.py` đã sửa (90/10 split đúng) để có model tốt hơn và val_acc có nghĩa.

5. **Recording liên tục** — thêm `VideoRecorder` ghi theo segment 1h với storage rotation, tích hợp vào `ai_processor._main_loop()`.

6. **VideoSource tích hợp vào server** — thay `camera_controller` bằng `video_source.py` trong `ai_processor._initialize()` (đã làm xong cho `main.py` qua `--source bridge`/`CARLABridgeClient`) → sẵn sàng cho camera IP thực tế.

7. **Camera Management UI** — thêm trang Settings trong frontend cho phép thêm/xóa camera và vẽ ROI trực tiếp trên camera feed.

---

## 9. Tóm Tắt Tiến Độ

| Thành phần | Hoàn thành | Ghi chú |
|-----------|-----------|---------|
| AI Pipeline (8 bước) | **100%** | Detect → Track → ReID → Global ID → Trajectory → Alert → Incident → Evidence |
| Bug-free operation | **100%** | 13 bugs đã được sửa, pipeline chạy đúng |
| Vehicle ReID (VeRi-776) | **100%** | Weights đã train và tích hợp vào DualReIDExtractor |
| Backend API (REST + WS + MJPEG) | **100%** | REST, WebSocket (alerts/tracks/stats), MJPEG đều hoạt động đầy đủ |
| Web Dashboard | **90%** | Đủ dùng; thiếu Camera Management UI, ROI editor |
| Evaluation (MOTA/IDF1) | **0%** | ground_truth.py có, motmetrics chưa kết nối |
| Recording liên tục | **0%** | Chưa implement |
| VideoSource integration | **80%** | `main.py --source {bridge,rtsp,file,webcam}` đều dùng VideoSource; server `ai_processor.py` vẫn dùng `CameraController` trực tiếp |
| Xung đột Python/CARLA | **100%** | Đã giải quyết bằng kiến trúc `carla_bridge` (2 process qua TCP) |
| Trajectory Prediction (Kalman) | **100%** | `KalmanFilter2D` constant-acceleration thay exp-weighted velocity; còn thiếu calibration pixel↔world để "lên map" |

**Tổng thể: ~97% hoàn thành** cho mục tiêu demo đầy đủ vòng khép kín (CARLA → AI → Dashboard → Evidence). Phần còn lại là tính năng nâng cao (recording, camera UI, ground truth eval) không ảnh hưởng đến demo core.

---

## 10. Định Hướng Nâng Cấp Lên Kiến Trúc AI Hiện Đại Hơn (phụ thuộc GPU)

Các module hiện tại đều chọn theo hướng nhẹ, phù hợp GPU 4-6GB VRAM. Nếu nâng cấp lên
**RTX 4060 8GB** (Ada Lovelace, Tensor Core thế hệ mới, ~5-7x throughput FP16 so với
GTX 1050Ti), các hướng nâng cấp sau trở nên khả thi cho realtime 6 camera:

| Module | Hiện tại | Lựa chọn hiện đại hơn | Lợi ích | Điều kiện |
|---|---|---|---|---|
| Detector | YOLOv8s | **YOLOv11s/m** (drop-in, cùng `ultralytics`), hoặc **RT-DETR** (transformer, không NMS) | mAP cao hơn, tốt hơn ở object nhỏ/đông (xe máy VN) | YOLOv11 gần như miễn phí; RT-DETR cần ≥6-8GB |
| Tracker | ByteTrack (`boxmot`) | **BoT-SORT/OC-SORT** (cùng thư viện `boxmot`) | Bù chuyển động camera, ReID embedded vào association, giảm ID-switch | Gần như miễn phí, chỉ đổi class |
| Re-ID | OSNet (Market-1501 + VeRi-776) | **CLIP-ReID/TransReID** (ViT backbone) | Generalize tốt hơn nhiều với domain gap CARLA↔thực tế | Cần ≥8GB cho realtime 6 camera |
| Trajectory Prediction | Kalman Filter constant-acceleration (`KalmanFilter2D`) | **Trajectory deep learning** (Trajectron++/Social-GAN/MTR) — dự đoán đa-agent có context tương tác | Dự đoán chính xác hơn ở giao lộ đông, "xe A tránh xe B" thay vì chỉ ngoại suy vận tốc | Cần dataset trajectory từ CARLA + GPU ≥8GB để train/infer realtime |
| Incident Detection | Rule-based (12 loại) | **VLM nhỏ** (Florence-2, Moondream2, Qwen2-VL-2B) chạy song song để mô tả/log sự cố bằng ngôn ngữ tự nhiên | Bổ trợ rule-based, log dễ đọc hơn | Cần VRAM dự phòng đáng kể, không bắt buộc |

**Thứ tự ưu tiên nếu có RTX 4060 8GB** (theo lợi ích/chi phí):
1. YOLOv8s → YOLOv11s (detector) — gần như miễn phí.
2. ByteTrack → BoT-SORT/OC-SORT (tracker) — thử nghiệm dễ, cùng thư viện.
3. Calibration pixel↔world (không phụ thuộc GPU, vẫn là điều kiện tiên quyết để "lên map").
4. Trajectory predictor: Kalman → mô hình deep learning (Trajectron++ hoặc tương đương) — nâng cấp giá trị nhất, đúng trọng tâm "dự đoán hướng đi" của project; chỉ làm sau khi đánh giá Kalman bằng CARLA scenarios cho thấy không đủ chính xác ở tình huống tương tác đông object.
5. Re-ID: OSNet → CLIP-ReID/TransReID — cải thiện độ chính xác cross-camera.
6. CLIP-ReID, VLM incident detection — dành cho khi deploy server có GPU mạnh hơn.

---

*Cập nhật lần cuối: 2026-06-10*
*Thay đổi (2026-06-01): Phát hiện và sửa 13 bugs trong ai_processor, stream_service, config, camera_config, train_veri; tích hợp VeRi-776 weights vào DualReIDExtractor; kích hoạt Spatio-Temporal Filter; proactive alerts hoạt động đúng; `/ws/stats` push FPS/tracks/alerts mỗi 1s.*
*Thay đổi (2026-06-09): Cập nhật phần cứng (RTX 3060 Laptop 6GB VRAM thay GTX 1050Ti); phân tích xung đột Python 3.7 (CARLA 0.9.9.4) vs Python 3.8+ (boxmot/ultralytics); đề xuất nâng cấp CARLA 0.9.15 + Python 3.8 (Hướng 3); cập nhật kế hoạch tiếp theo.*
*Thay đổi (2026-06-10): Triển khai kiến trúc `carla_bridge` (2 process qua TCP) giải quyết xung đột Python/CARLA — thay thế đề xuất Hướng 3; thêm `CARLABridgeClient` vào `video_source.py` và `--source {carla,bridge}` vào `main.py`; sửa `camera_controller.py` (encoding UTF-8) và `traffic_generator.py` (pedestrian retargeting định kỳ thay cho `is_at_goal()` không tồn tại); xóa thư mục thử nghiệm `final_project_Test/` sau khi tích hợp xong.*
*Thay đổi (2026-06-10, tiếp): Viết lại `trajectory_predictor.py` dùng Kalman Filter constant-acceleration (`KalmanFilter2D`) thay cho exp-weighted velocity — xử lý được rẽ/tăng giảm tốc, interface không đổi (`update_trajectory`/`predict`/`predict_list`/`get_velocity`/`get_speed` + thêm `get_acceleration`); thêm `--source {rtsp,file,webcam}` vào `main.py` (dùng `MultiVideoSource` + `RTSPVideoSource`/`FileVideoSource`/`WebcamVideoSource` từ `video_source.py`) với các tham số `--rtsp-url`/`--video-path`/`--webcam-index`/`--camera-ids`/`--loop-video`.*
*Thay đổi (2026-06-10, tiếp): Thêm mục 10 "Định Hướng Nâng Cấp Lên Kiến Trúc AI Hiện Đại Hơn" — bảng so sánh model hiện tại vs lựa chọn hiện đại hơn (YOLOv11/RT-DETR, BoT-SORT/OC-SORT, CLIP-ReID/TransReID, Trajectory deep learning, VLM incident detection) cho kịch bản nâng cấp GPU lên RTX 4060 8GB, kèm thứ tự ưu tiên.*
