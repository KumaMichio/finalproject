# BÁO CÁO TIẾN ĐỘ DỰ ÁN

## 1. Tên Dự Án

**Hệ Thống Giám Sát Camera CCTV Đa Kênh Ứng Dụng Trí Tuệ Nhân Tạo**
*(Multi-Camera CCTV Tracking System with AI)*

Môi trường mô phỏng: CARLA Simulator phiên bản 0.9.9.4.

---

## 2. Tổng Quan Dự Án

Dự án được tổ chức thành **4 thành phần chính**, mỗi thành phần nằm trong một thư mục riêng biệt và có thể hoạt động độc lập hoặc kết hợp với nhau:

```
finalproject/
├── custom_tracking_system/   # AI Pipeline — xử lý video và nhận diện đối tượng
├── server/                   # Backend API — REST, WebSocket, MJPEG streaming
├── frontend/                 # Web Dashboard — giao diện giám sát real-time (React)
└── WindowsNoEditor/          # CARLA Simulator — môi trường mô phỏng đô thị
```

### 2.1 AI Pipeline (`custom_tracking_system/`)

Lõi trí tuệ nhân tạo của hệ thống, xử lý video từ nhiều camera theo pipeline 8+ bước:

| Mô-đun | Chức năng |
|--------|-----------|
| `camera_controller.py` | Kết nối và đồng bộ nhiều camera trong CARLA |
| `traffic_generator.py` | Sinh phương tiện và người đi bộ tự động trong môi trường mô phỏng |
| `detector.py` | Phát hiện đối tượng (xe, người) bằng YOLOv5s |
| `tracker.py` | Theo dõi đối tượng trong từng camera (IoU matching) |
| `reid.py` | Trích xuất đặc trưng hình ảnh bằng OSNet/ResNet50 cho Re-ID |
| `global_tracking.py` | Gán Global ID duy nhất xuyên camera bằng cosine similarity |
| `trajectory_predictor.py` | Dự đoán 5 vị trí tương lai của đối tượng |
| `alert_system.py` | Cảnh báo khi đối tượng xâm nhập vùng ROI (Region of Interest) |
| `incident_detector.py` | Phát hiện 10 loại sự cố real-time (đâm xe, tốc độ cao, lảng vảng...) |
| `evidence_package.py` | Ring buffer 30s — tự động lưu clip + ảnh + metadata khi có sự cố |
| `scenario_controller.py` | Tạo kịch bản tai nạn có kiểm soát trong CARLA (5 kịch bản) |
| `video_source.py` | Lớp trừu tượng VideoSource hỗ trợ CARLA / RTSP / File / Webcam |
| `ground_truth.py` | Thu thập ground truth từ CARLA để phục vụ đánh giá độ chính xác |

#### Các mô hình AI đang sử dụng — Ưu/Nhược điểm và Hướng nâng cấp

---

##### Mô hình 1 — Phát hiện đối tượng: **YOLOv5s** (v6.1, pretrained COCO)

| | |
|---|---|
| **Cấu hình hiện tại** | `yolov5s`, confidence ≥ 0.4, 4 lớp: `person / car / bus / truck` |
| **Ưu điểm** | Nhẹ (~14 MB), tốc độ cao (~30–60 FPS trên GPU), dễ tích hợp qua `torch.hub`, không cần fine-tune |
| **Nhược điểm** | Độ chính xác thấp hơn các biến thể lớn (YOLOv5m/l/x); đôi khi bỏ sót xe nhỏ hoặc bị khuất; COCO không có nhãn xe máy / xe đạp phù hợp với giao thông Việt Nam |
| **Hướng nâng cấp** | **YOLOv8n/s** (Ultralytics mới, cùng tốc độ nhưng chính xác hơn) hoặc **YOLO11** (2024); hoặc **RT-DETR-L** nếu cần độ chính xác cao hơn với cơ chế attention |

---

##### Mô hình 2 — Theo dõi trong từng camera: **IoU Greedy Matching** (tự cài đặt)

| | |
|---|---|
| **Cấu hình hiện tại** | `max_age=30`, `min_hits=3`, `iou_threshold=0.3`; so khớp tham lam theo thứ tự track |
| **Ưu điểm** | Không phụ thuộc thêm thư viện, hoàn toàn kiểm soát được logic, đủ ổn định khi đối tượng di chuyển tốt |
| **Nhược điểm** | Không có bộ lọc Kalman → không dự đoán vị trí khi bị che khuất; không dùng appearance → hai đối tượng giống nhau dễ bị hoán đổi ID khi chồng lên nhau (occlusion swap); greedy matching không tối ưu toàn cục |
| **Hướng nâng cấp** | **ByteTrack** (2022) — kết hợp Kalman filter + hai ngưỡng score thấp/cao, xử lý tốt occlusion; hoặc **StrongSORT** (Kalman + ReID appearance) nếu muốn dùng chung đặc trưng với mô hình ReID |

---

##### Mô hình 3 — Nhận diện lại xuyên camera (Re-ID): **OSNet** / **ResNet50 fallback**

| | |
|---|---|
| **Cấu hình hiện tại** | **Chính:** `osnet_x1_0` (torchreid), pretrained Market-1501, output 512D; **Dự phòng:** ResNet50 ImageNet, output 2048D. So khớp bằng cosine similarity, ngưỡng ≥ 0.5 |
| **Ưu điểm** | OSNet được thiết kế riêng cho Re-ID, nhẹ hơn ResNet50 (1M params vs 25M), feature chuẩn hóa L2 giúp so khớp ổn định |
| **Nhược điểm** | Market-1501 là tập dữ liệu **người đi bộ** — mô hình không được huấn luyện cho **xe cộ**, nên hai xe cùng màu/kiểu dễ bị nhầm; ResNet50-ImageNet fallback không phải ReID model (feature không phân biệt cá thể tốt); không dùng thông tin thời gian/không gian để lọc kết quả khớp phi thực tế |
| **Hướng nâng cấp** | **VehicleNet / VeRi-776** — model Re-ID chuyên cho xe; **CLIP** (ViT-B/32) — zero-shot matching theo mô tả; thêm **Spatio-Temporal Reasoning** (lọc match bằng khoảng cách camera + delta thời gian) để loại bỏ các ID khớp phi thực tế |

---

##### Mô hình 4 — Dự đoán quỹ đạo: **Nội suy tuyến tính** (Linear Extrapolation)

| | |
|---|---|
| **Cấu hình hiện tại** | Velocity = `pos[-1] − pos[-2]`; dự đoán 5 bước tiếp theo bằng `pos + velocity × step`; window 10 vị trí gần nhất |
| **Ưu điểm** | Không cần huấn luyện, tính toán cực nhanh, không có dependency bổ sung |
| **Nhược điểm** | Giả định đối tượng đi thẳng → sai khi rẽ, dừng đèn đỏ, hoặc đổi hướng đột ngột; không mô hình hóa được gia tốc; sai số tích lũy nhanh sau 2–3 bước |
| **Hướng nâng cấp** | **Kalman Filter** — mô hình chuyển động hằng vận tốc/gia tốc, chống nhiễu tốt; **LSTM** — học được pattern rẽ từ lịch sử dài hơn; **Social Force Model** — xét tương tác giữa các đối tượng |

---

##### Mô hình 5 — Phát hiện sự cố: **Rule-based Threshold** (heuristic)

| | |
|---|---|
| **Cấu hình hiện tại** | 10 loại sự cố, mỗi loại là một tập quy tắc ngưỡng cứng (ví dụ: `speed > 120 px/s` → OVERSPEED; tốc độ giảm còn 30% trong 1 frame → SUDDEN_STOP); cooldown 4 giây chống duplicate |
| **Ưu điểm** | Hoạt động ngay không cần data training, dễ giải thích, dễ điều chỉnh ngưỡng theo môi trường cụ thể |
| **Nhược điểm** | Ngưỡng cứng nhạy cảm với noise từ tracker (FPS thay đổi → tốc độ px/s dao động); không tự thích nghi với môi trường khác nhau; dễ false positive khi tracker bị swap ID |
| **Hướng nâng cấp** | **Anomaly Detection** (Isolation Forest / Autoencoder) học phân phối hành vi bình thường; **GCN / Transformer** mô hình hóa tương tác không gian giữa nhiều đối tượng đồng thời |

---

##### Tóm tắt so sánh mô hình

| Thành phần | Hiện tại | Production Stack | VRAM | Độ ưu tiên |
|-----------|---------|-----------------|------|-----------|
| Detection | YOLOv5s (COCO, 4 class) | **YOLOv8s** fine-tune VN (6 class + motorcycle) | ~1.8 GB | **Cao** |
| Tracker | IoU Greedy Matching | **ByteTrack** (Kalman built-in) | CPU only | **Cao** |
| Re-ID | OSNet Market-1501 (người) | **OSNet VeRi-776** (xe) + Spatio-Temporal Filter | ~0.4 GB | **Cao** |
| Trajectory | Linear Extrapolation (có lỗi đơn vị) | **Kalman từ ByteTrack** + timestamp-based | CPU only | Trung bình |
| Incident | Rule-based threshold cứng | **FPS-normalized** + **Proactive (dùng predicted pos)** | CPU only | Trung bình |

> ⚠️ **Lỗi đã phát hiện trong `trajectory_predictor.py`:**
> - `window_size=10` không có tác dụng — `_linear_prediction` chỉ dùng 2 điểm cuối
> - Velocity tính bằng `pixels/frame` (phụ thuộc FPS), không phải `pixels/giây`
> - Prediction horizon chỉ 5 frames (~0.17s) — quá ngắn để cảnh báo sớm
> - Không kết nối với `IncidentDetector` → không có proactive alert
>
> → Xem phân tích đầy đủ và hướng sửa tại `docs/production_model_stack.md`

> 📋 **Tài liệu chi tiết production stack:** [`docs/production_model_stack.md`](production_model_stack.md)
> — bao gồm: ràng buộc phần cứng (4 GB VRAM), kiến trúc batch 6 camera,
> fine-tune guide, migration CARLA → RTSP thực tế.

---

### 2.2 Backend API Server (`server/`)

Server FastAPI cung cấp giao diện lập trình chuẩn cho toàn hệ thống:

| Thành phần | Chức năng |
|-----------|-----------|
| **REST API** (17 endpoints) | Quản lý camera, tracks, alerts, ROIs, thống kê |
| **WebSocket** (3 kênh) | Push cảnh báo, tracking update, thống kê real-time |
| **MJPEG Streaming** | Truyền video trực tiếp tới trình duyệt qua HTTP |
| **Database SQLite** | 5 bảng: `cameras`, `alerts`, `tracked_objects`, `tracking_history`, `rois` |
| **AI Processor** | Chạy AI pipeline trong background thread, push kết quả lên WebSocket |

### 2.3 Web Dashboard (`frontend/`)

Giao diện giám sát real-time được xây dựng bằng React + Tailwind CSS:

| Tính năng | Mô tả |
|-----------|-------|
| Camera Grid | Hiển thị live video từ nhiều camera đồng thời |
| Incident Panel | Bảng sự cố real-time, phân loại theo mức độ (INFO / WARNING / CRITICAL) |
| Alert Management | Xem, lọc, xác nhận các cảnh báo |
| Object History | Tra cứu lịch sử di chuyển của đối tượng theo Global ID |
| Real-time Notification | Âm thanh cảnh báo + flash màn hình đỏ khi CRITICAL |

### 2.4 Môi Trường Mô Phỏng (`WindowsNoEditor/`)

CARLA Simulator 0.9.9.4 — engine mô phỏng đô thị 3D chạy trên Unreal Engine 4, đóng vai trò thay thế cho hệ thống camera IP thực tế trong giai đoạn phát triển và thử nghiệm.

### 2.5 Luồng Hoạt Động Tổng Thể

```
CARLA Simulator
      │  (frame video từ camera ảo)
      ▼
AI Pipeline (custom_tracking_system/)
  Detect → Track → ReID → Global ID → Trajectory → Incident Detection
      │  (kết quả xử lý: tracks, alerts, incidents, evidence)
      ▼
Backend Server (server/)
  REST API + WebSocket + MJPEG
      │  (HTTP / WebSocket)
      ▼
Web Dashboard (frontend/)
  Camera Grid · Incident Panel · Alert Management
```

---

## 3. Mục Tiêu Của Dự Án

### 3.1 Vấn đề dự án hướng tới

Các hệ thống camera giám sát truyền thống hiện nay chủ yếu chỉ ghi hình thụ động,
việc theo dõi đối tượng phải dựa vào sức người. Khi hệ thống có nhiều camera trải
trên một khu vực rộng, các vấn đề thường gặp gồm:

- Không tự động phát hiện được phương tiện và người đi bộ trong khung hình.
- Không theo dõi liên tục được một đối tượng khi nó di chuyển qua nhiều camera
  (mỗi camera nhìn đối tượng như một thực thể mới, không có ID chung).
- Không dự đoán được hành vi tiếp theo của đối tượng để cảnh báo sớm.
- Không tự động cảnh báo khi đối tượng đi vào các vùng nguy hiểm hoặc vùng cấm.
- Khó tích hợp với các ứng dụng giám sát khác (web dashboard, mobile app)
  do thiếu giao diện lập trình (API) chuẩn.

### 3.2 Mục tiêu cụ thể của dự án

Dự án xây dựng một hệ thống giám sát đa camera có khả năng:

1. **Phát hiện đối tượng** (xe ô tô, xe tải, xe buýt, người đi bộ) từ nhiều
   camera đồng thời bằng mô hình học sâu YOLOv5.
2. **Theo dõi đối tượng trong từng camera** (single-camera tracking) bằng
   thuật toán so khớp IoU.
3. **Nhận diện lại đối tượng xuyên camera** (Cross-Camera Re-Identification)
   bằng mô hình OSNet/ResNet50, gán **Global ID** duy nhất cho mỗi đối tượng
   trong toàn hệ thống.
4. **Dự đoán quỹ đạo di chuyển** của đối tượng trong vài bước tương lai.
5. **Cảnh báo tự động** khi đối tượng đi vào vùng quan tâm (Region of Interest).
6. **Cung cấp API chuẩn** (REST API + WebSocket + MJPEG streaming) để các
   ứng dụng khác (web dashboard, mobile app) có thể truy cập dữ liệu real-time.
7. **Thiết kế trừu tượng nguồn video** để hệ thống có thể chuyển từ môi trường
   mô phỏng CARLA sang camera IP thực tế mà không cần viết lại pipeline AI.

---

## 4. Công Việc Đã Hoàn Thành

### 4.1 AI Pipeline (`custom_tracking_system/`)

Đã hoàn thành đầy đủ luồng xử lý 8 bước cho một frame video:

| STT | Mô-đun | Trạng thái | Mô tả |
|-----|--------|-----------|-------|
| 1 | `camera_controller.py` | Hoàn thành | Điều khiển và đồng bộ nhiều camera trong CARLA. |
| 2 | `traffic_generator.py` | Hoàn thành | Sinh xe và người đi bộ tự động trong môi trường mô phỏng. |
| 3 | `detector.py` | Hoàn thành | Phát hiện đối tượng bằng YOLOv5s (pretrained COCO). |
| 4 | `tracker.py` | Hoàn thành | Theo dõi đối tượng trong 1 camera bằng IoU greedy matching. |
| 5 | `reid.py` | Hoàn thành | Trích xuất đặc trưng bằng OSNet (512D) hoặc ResNet50 (2048D). |
| 6 | `global_tracking.py` | Hoàn thành | Gán Global ID xuyên camera dựa trên cosine similarity. |
| 7 | `trajectory_predictor.py` | Hoàn thành | Dự đoán 5 vị trí tương lai bằng nội suy tuyến tính. |
| 8 | `alert_system.py` | Hoàn thành | Sinh cảnh báo khi đối tượng vào ROI (point-in-polygon). |
| + | `video_source.py` | Hoàn thành | Lớp trừu tượng hỗ trợ CARLA / RTSP / File / Webcam. |
| + | `ground_truth.py` | Hoàn thành | Thu thập ground truth từ CARLA để phục vụ đánh giá (tách biệt khỏi AI). |

### 4.2 Backend API Server (`server/`)

Đã hoàn thành backend FastAPI bao gồm:

- **17 REST endpoints** cho quản lý: camera, tracks, alerts, ROIs, thống kê.
- **3 WebSocket channels**: push cảnh báo, tracking update, thống kê real-time.
- **MJPEG streaming**: truyền video trực tiếp tới trình duyệt qua HTTP.
- **Database SQLite** với 5 bảng: `cameras`, `alerts`, `tracked_objects`,
  `tracking_history`, `rois` (dùng SQLAlchemy ORM + Pydantic v2).
- **AI Processor** chạy pipeline AI trong background thread của server,
  push kết quả qua WebSocket và lưu vào database.

### 4.3 Cấu hình và công cụ phụ trợ

- File cấu hình YAML cho 3 camera (CAM_001, CAM_002, CAM_003) với 3 vùng ROI.
- Module visualization vẽ bounding box, quỹ đạo, lưới đa camera bằng OpenCV.
- Module thu thập metrics (FPS, số lượng detection/tracking).
- Module xuất dữ liệu ra JSON, CSV và báo cáo tổng kết.
- Logging hệ thống đầy đủ (file `tracking_system.log`).

### 4.4 Ba chế độ chạy

| Chế độ | Lệnh | Mục đích |
|--------|------|---------|
| API-only | `python app.py` | Server hoạt động không cần CARLA, dùng để phát triển frontend. |
| API + AI | `python app.py --with-ai` | Full system với CARLA, AI pipeline chạy nền. |
| Direct | `python main.py` | Hiển thị kết quả qua cửa sổ OpenCV, không qua server. |

### 4.5 Nâng Cấp Hướng Thực Tế (cập nhật 2026-05-24)

Sau khi phân tích mục tiêu "sát thực tế — phát hiện sự cố real-time", đã implement thêm:

| STT | Mô-đun | Trạng thái | Mô tả |
|-----|--------|-----------|-------|
| 9  | `tracker.py` (nâng cấp) | Hoàn thành | Thêm `timestamps` + `speeds` (px/s) vào mỗi track để tính velocity. |
| 10 | `incident_detector.py` | Hoàn thành | Phát hiện 10 loại sự cố real-time (sudden stop, fleeing, overspeed, proximity, loitering...). |
| 11 | `evidence_package.py` | Hoàn thành | Ring buffer 30s — tự động lưu crop ảnh + clip trước/sau + metadata JSON khi sự cố. |
| 12 | `scenario_controller.py` | Hoàn thành | Script 5 kịch bản tai nạn trong CARLA (hit-and-run, đâm người, vượt đèn đỏ, đâm từ sau, dừng đột ngột). |
| 13 | Web Dashboard (React) | Hoàn thành | Camera grid live, incident panel real-time, alert management, object history. |
| 14 | Real-time Notification | Hoàn thành | WebSocket push, âm thanh cảnh báo, flash màn hình đỏ khi CRITICAL. |

**Tích hợp vào pipeline:**
- `ai_processor.py` (server) và `main.py` đã gọi `IncidentDetector` và `EvidencePackage` trong vòng lặp chính.
- Incident CRITICAL → push WebSocket `/ws/alerts` → Dashboard nhận ngay lập tức.
- Evidence tự động lưu vào thư mục `evidence/<incident_id>/`.

### 4.6 Tài liệu

Đã viết đầy đủ 8 tài liệu trong thư mục `docs/`:
`description.md`, `plan.md`, `workflow.md`, `execute.md`,
`development_roadmap.md`, `report.md`, `project_context.md`, `upgrade.md`.

---

## 5. Tiến Độ Hiện Tại

### 5.1 Hạng mục đã hoàn thành

- AI Pipeline đầy đủ 8 bước, chạy ổn định trên môi trường CARLA.
- Backend API Server (FastAPI) với REST + WebSocket + MJPEG streaming.
- Database 5 bảng lưu trữ toàn bộ dữ liệu camera, alerts, tracks, history, ROIs.
- Tích hợp AI pipeline vào server qua background thread.
- Lớp trừu tượng VideoSource (sẵn sàng cho camera thực tế).
- Module Ground Truth tách biệt phục vụ đánh giá.
- Hệ thống cấu hình, logging, xuất dữ liệu.
- **[MỚI]** Tracker nâng cấp: lưu timestamp + tốc độ (px/s) mỗi frame.
- **[MỚI]** Incident Detector: phát hiện 10 loại sự cố real-time.
- **[MỚI]** Evidence Package: tự động lưu clip + ảnh crop + JSON khi sự cố.
- **[MỚI]** Scenario Controller: tạo kịch bản tai nạn có kiểm soát trong CARLA.
- **[MỚI]** Web Dashboard (React): camera grid, incident panel, alert management.
- **[MỚI]** Real-time notification: âm thanh + flash khi CRITICAL alert.

### 5.2 Hạng mục đang/chưa triển khai

| Hạng mục | Trạng thái | Ghi chú |
|---------|-----------|---------|
| Nâng cấp Tracker (ByteTrack/Kalman) | Chưa làm | IoU greedy hiện đủ chạy nhưng bị swap ID khi occlusion. |
| Vehicle Re-ID | Chưa làm | OSNet (Market-1501) chỉ cho người — xe giống nhau dễ nhầm. |
| Spatio-temporal Reasoning | Chưa làm | Cần dùng thời gian + khoảng cách giữa camera để phân biệt xe. |
| Recording liên tục | Chưa làm | Evidence Package chỉ lưu clip khi có sự cố, chưa ghi liên tục. |
| Camera Management UI | Chưa làm | Chưa có giao diện thêm/xóa camera runtime. |
| Ground Truth Evaluation | Chưa làm | Module đã có, chưa tích hợp tính MOTA, IDF1, mAP. |
| Tích hợp VideoSource | Chưa làm | `video_source.py` đã viết nhưng pipeline vẫn dùng `camera_controller`. |

### 5.3 Tóm tắt

Dự án đã hoàn thành **~85%**. Hệ thống có đầy đủ vòng khép kín:
**tạo kịch bản tai nạn trong CARLA → AI phát hiện sự cố → push thông báo real-time → Dashboard hiển thị → lưu bằng chứng tự động**.
Phần còn lại là nâng cấp độ chính xác AI (ByteTrack, Vehicle ReID) và
tính năng quản lý nâng cao (recording liên tục, camera management UI).

---

## 6. Khó Khăn Gặp Phải

### 6.1 Khó khăn về mặt kỹ thuật

**(1) Nhận diện lại các phương tiện có hình dạng giống nhau**

Mô hình OSNet hiện tại được huấn luyện trên tập Market-1501 (chuyên cho người),
nên với hai chiếc xe cùng màu, cùng kiểu dáng, hệ thống không thể phân biệt được
chúng khi chuyển giữa các camera. Đây là hạn chế cố hữu của ReID dựa trên đặc
trưng hình ảnh thuần túy.

→ Hướng giải quyết dự kiến: bổ sung suy luận không-thời gian (spatio-temporal
reasoning) dựa trên khoảng cách giữa camera và thời gian di chuyển, hoặc
huấn luyện một mô hình Vehicle Re-ID riêng (ví dụ trên tập VeRi-776).

**(2) CARLA không hỗ trợ biển số xe**

Phương pháp phân biệt xe chính xác nhất trong thực tế là nhận dạng biển số
(License Plate Recognition). Tuy nhiên môi trường mô phỏng CARLA 0.9.9.4
không cung cấp biển số có thể đọc được, nên không thể áp dụng phương pháp này
trong giai đoạn mô phỏng.

**(3) Thuật toán tracker hiện tại đơn giản**

Tracker đang dùng IoU greedy matching, chưa có Kalman filter hoặc kết hợp
appearance matching. Khi hai đối tượng đi sát nhau rồi tách ra (occlusion),
tracker có thể hoán đổi ID.

→ Hướng giải quyết: nâng cấp lên DeepSORT hoặc ByteTrack.

**(4) Dự đoán quỹ đạo còn cơ bản**

Hiện chỉ dùng nội suy tuyến tính (linear extrapolation) nên không xử lý được
các trường hợp đối tượng đổi hướng đột ngột (rẽ, dừng đèn đỏ).

→ Hướng giải quyết: thay bằng Kalman filter hoặc mô hình LSTM.

### 6.2 Khó khăn về môi trường và công cụ

**(5) CARLA Simulator nặng và đòi hỏi GPU mạnh**

CARLA 0.9.9.4 chạy trên Unreal Engine 4 cần GPU ít nhất 6GB VRAM. Khi vừa chạy
CARLA vừa chạy YOLOv5 + OSNet, máy tính dễ bị giật/nóng nếu cấu hình không đủ.

**(6) Cài đặt `torchreid` phức tạp**

Thư viện `torchreid` (cho OSNet) phải cài từ source code GitHub, không có trên
PyPI ổn định. Yêu cầu phiên bản PyTorch và CUDA cụ thể, đôi khi xung đột với
môi trường có sẵn. Đã làm fallback dùng ResNet50 (ImageNet pretrained) khi
torchreid không cài được.

**(7) Phiên bản Python bị giới hạn bởi CARLA 0.9.9.4**

CARLA 0.9.9.4 chỉ cung cấp file `.egg` cho Python 3.7, gây khó khăn khi tích
hợp với các thư viện AI mới hơn (yêu cầu Python 3.8+).

### 6.3 Khó khăn về đánh giá

**(8) Chưa có ground truth để đo định lượng**

Mặc dù đã viết module `ground_truth.py`, nhưng chưa tích hợp được vào quy trình
đánh giá để tính các chỉ số chuẩn như MOTA (Multi-Object Tracking Accuracy),
IDF1, và mAP. Hiện chỉ đánh giá định tính qua quan sát.

---

## 7. Kế Hoạch Tiếp Theo

Các hạng mục còn lại theo thứ tự ưu tiên:

1. **Nâng cấp Tracker → ByteTrack + Kalman filter** — giảm swap ID khi hai
   đối tượng che khuất nhau, tăng độ bền track.
2. **Vehicle ReID** — thêm model riêng cho xe (VehicleID/VeRi-776 dataset)
   để phân biệt xe xuyên camera chính xác hơn.
3. **Recording liên tục** — ghi video tất cả camera theo segment 1 giờ,
   tự động xoay vòng khi đầy dung lượng.
4. **Spatio-temporal reasoning** — dùng thời gian + khoảng cách giữa camera
   để tăng độ chính xác cross-camera matching.
5. **Ground Truth Evaluation** — kết nối `ground_truth.py` + `motmetrics`
   để có số liệu MOTA, IDF1, mAP định lượng.
6. **Tích hợp VideoSource** — thay `camera_controller` bằng `video_source`
   trong pipeline, sẵn sàng cho camera IP thực tế.

---

*Báo cáo cập nhật ngày 2026-05-24.*
