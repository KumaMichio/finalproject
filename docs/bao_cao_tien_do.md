# BÁO CÁO TIẾN ĐỘ DỰ ÁN (CUỐI CÙNG)

**Tên dự án:** Hệ Thống Giám Sát Camera CCTV Đa Kênh Ứng Dụng Trí Tuệ Nhân Tạo
*(Multi-Camera CCTV Tracking System with AI)*

**Môi trường mô phỏng:** CARLA Simulator 0.9.14 (Town10HD_Opt)

**Cập nhật lần cuối:** 2026-06-12

---

## 1. Đặt Vấn Đề

Tại các đô thị Việt Nam, mật độ giao thông hỗn hợp (xe máy chiếm tỷ trọng rất lớn,
xen lẫn ô tô, xe buýt, xe tải, người đi bộ băng qua đường không đúng nơi quy định)
khiến việc giám sát an toàn giao thông bằng camera CCTV truyền thống gặp nhiều
hạn chế:

- **Số lượng camera lớn, con người không thể theo dõi liên tục mọi màn hình** —
  sự cố (va chạm, dừng xe đột ngột, đi ngược chiều, tụ tập đông người...) thường
  chỉ được phát hiện *sau khi* đã xảy ra, qua xem lại video.
- **Camera hoạt động độc lập, không có "trí nhớ" xuyên camera** — không thể trả
  lời câu hỏi "đối tượng này đã xuất hiện ở camera nào trước đó, đi theo hướng nào".
- **Không có khả năng cảnh báo sớm (proactive)** — hệ thống chỉ phản ứng khi sự
  việc đã xảy ra (reactive), không dự đoán được "1–3 giây nữa điều gì sẽ xảy ra"
  để cảnh báo trước.
- **Mô hình AI có sẵn (pretrained COCO...) không tối ưu cho điều kiện Việt Nam**
  — đặc biệt là xe máy kiểu VN (Honda Wave, SH, Lead...) khác với "motorcycle"
  trong các bộ dữ liệu phương Tây.

Dự án đặt vấn đề: **xây dựng một hệ thống giám sát giao thông đa camera, dùng AI
để tự động phát hiện – theo dõi – nhận diện lại xuyên camera – dự đoán quỹ đạo –
phát hiện sự cố (kể cả dự đoán trước sự cố), với chi phí phần cứng vừa phải và có
khả năng chuyển từ môi trường mô phỏng (CARLA) sang camera thực (RTSP) mà không
cần viết lại pipeline.**

---

## 2. Mục Tiêu, Phạm Vi Đề Tài

### 2.1 Mục tiêu cụ thể

1. **Phát hiện đối tượng** (person, car, motorcycle, bus, truck) từ nhiều camera
   đồng thời bằng mô hình YOLO, fine-tune cho điều kiện giao thông Việt Nam.
2. **Theo dõi đối tượng trong từng camera** (multi-object tracking) ổn định, ít
   ID-switch, xử lý được trường hợp che khuất tạm thời.
3. **Nhận diện lại xuyên camera (Cross-Camera Re-ID)** — gán một "Global ID" duy
   nhất cho mỗi đối tượng dù nó di chuyển qua nhiều camera khác nhau.
4. **Dự đoán quỹ đạo** tại nhiều mốc thời gian thực (0.5s–3.0s), kể cả các tình
   huống rẽ/đổi hướng — không chỉ ngoại suy tuyến tính.
5. **Phát hiện sự cố giao thông real-time** — cả phản ứng (reactive: dừng đột
   ngột, vượt tốc độ, đi ngược chiều...) và dự đoán trước (proactive: dự đoán va
   chạm, dự đoán đi vào vùng nguy hiểm).
6. **Cung cấp API chuẩn** (REST + WebSocket + MJPEG) để dashboard và các ứng dụng
   khác có thể truy cập dữ liệu camera/cảnh báo theo thời gian thực.
7. **Thiết kế kiến trúc có thể chuyển từ CARLA (mô phỏng) sang camera IP thực**
   chỉ bằng cách đổi cấu hình, không cần viết lại AI pipeline.

### 2.2 Phạm vi

- **Môi trường phát triển/kiểm thử:** CARLA Simulator 0.9.14 (bản đồ Town10HD_Opt),
  6 camera CCTV ảo đặt tại các giao lộ.
- **Phạm vi mô hình AI:** Detection, Tracking, Re-ID, Trajectory Prediction,
  Incident Detection — không bao gồm nhận dạng biển số (CARLA không cung cấp
  biển số đọc được) hoặc các bài toán an ninh khác (nhận diện khuôn mặt...).
- **Phạm vi dữ liệu huấn luyện:** Dataset công khai (VisDrone2019-DET — ảnh thực
  từ drone, góc nhìn tương tự CCTV) kết hợp dữ liệu tổng hợp tự thu thập từ CARLA
  (ảnh + nhãn tự động sinh từ ground-truth của simulator).
- **Phạm vi triển khai:** Backend (FastAPI) + Frontend (React) hoàn chỉnh chạy
  trên 1 máy (RTX 4060 8GB), kiến trúc sẵn sàng mở rộng sang RTSP camera thực.

---

## 3. Các Model AI Sử Dụng & Tiến Độ

### 3.1 Object Detection — YOLO11m

| | |
|---|---|
| **Vai trò** | Phát hiện 5 lớp: `person, car, motorcycle, bus, truck` trên mỗi khung hình camera |
| **Model trước đây** | YOLOv8s (pretrained COCO) — mAP COCO 44.9, hoạt động ổn định nhưng nhận diện xe máy VN kém |
| **Model hiện tại** | **YOLO11m** (~20M params), fine-tune từ checkpoint pretrained COCO |
| **Dataset fine-tune** | `data/visdrone_carla_vn` = VisDrone2019-DET (convert 10→5 lớp VN) **+** dữ liệu CARLA tự thu thập (`collect_carla_detection_data.py`, 6 camera CCTV đặt tại góc phố/gắn tường, 184 ảnh/191 box) |
| **Số lượng ảnh** | 6,643 train (6,471 VisDrone + 172 CARLA) / 560 val (548 VisDrone + 12 CARLA) |
| **Tiến độ huấn luyện** | ✅ **Hoàn thành 18/18 epoch** (2026-06-12), batch=8, imgsz=640, RTX 4060 |
| **Kết quả cuối cùng** | mAP50=**0.559**, mAP50-95=**0.334**, precision=**0.677**, recall=**0.529** (so với epoch 1: mAP50=0.346, mAP50-95=0.189 — **mAP50 +61.6%, mAP50-95 +76.7%**) |
| **Sản phẩm** | `weights/yolo11m_vn.pt` (đã copy từ `best.pt`) + `results.png`, `confusion_matrix.png`, `BoxPR_curve.png`, `BoxF1_curve.png`, `BoxP_curve.png`, `BoxR_curve.png`, `results.csv` lưu tại `docs/assets/yolo11m_vn_carla/` |

**Trạng thái:** ✅ **Hoàn thành fine-tune** (2026-06-12). Camera CARLA được đặt lại
vị trí thực tế (góc phố, gắn tường/cột — không xuyên tường, không dưới đất)
trước khi thu thập dữ liệu CARLA bổ sung. Quá trình train bị gián đoạn 2 lần do
máy bị reset (epoch 2→6 và epoch 17→18) nhưng đã resume thành công từ
checkpoint mỗi lần. `ObjectDetector` đã dùng checkpoint mới này.

---

### 3.2 Per-Camera Tracking — ByteTrack

| | |
|---|---|
| **Vai trò** | Theo dõi đối tượng trong từng camera giữa các frame, gán track ID tạm thời |
| **Model** | `boxmot.ByteTrack` — Kalman Filter (built-in) + Hungarian matching + dual-threshold |
| **Cấu hình** | `track_activation_threshold=0.25`, `lost_track_buffer=30`, `minimum_matching_threshold=0.8`, `frame_rate=10` |
| **Chi phí** | CPU-only, không tốn VRAM |

**Trạng thái:** ✅ **Hoàn thành** (2026-05-29). Đã tích hợp vào `ai_processor.py`
và `main.py`, hoạt động ổn định.

> **Còn thiếu:** đánh giá định lượng MOTA/IDF1 (chưa nối `motmetrics`) — hiện chỉ
> đánh giá định tính qua quan sát.

---

### 3.3 Cross-Camera Re-ID — DualReIDExtractor (OSNet)

| | |
|---|---|
| **Vai trò** | Trích xuất feature vector cho mỗi đối tượng để gán "Global ID" xuyên camera |
| **Model người** | OSNet x1.0 pretrained Market-1501 (512D) |
| **Model xe** | OSNet x1.0 fine-tune trên VeRi-776 (575 classes) |
| **Checkpoint hiện có** | `weights/osnet_veri776.pth` (đã train xong 2026-05-31, epoch 12) |
| **Bộ lọc bổ sung** | Spatio-Temporal Filter — loại match "phi thực tế" dựa trên `camera_topology` (thời gian di chuyển tối thiểu/tối đa giữa các cặp camera) |

**Trạng thái:** ✅ Đã train và tích hợp vào pipeline (DualReIDExtractor hoạt động
đúng). ⚠️ **Không thể re-train ngay** — dataset VeRi-776 (image_train/,
train_label.xml...) đã được dùng để train checkpoint hiện có nhưng **không còn
trên máy** (không nằm trong repo do kích thước lớn). `train_veri.py` đã sửa lỗi
validation split (trước đây dùng `image_test/` khác class-space khiến val_acc
luôn ~0%) nhưng cần tải lại VeRi-776 (~1.1GB, https://vehiclereid.github.io/VeRi/)
trước khi re-run được. Checkpoint `weights/osnet_veri776.pth` (train 2026-05-31,
epoch 12) vẫn hoạt động tốt trong pipeline hiện tại.

---

### 3.4 Trajectory Prediction — Kalman Filter + GRU Ensemble

| | |
|---|---|
| **Vai trò** | Dự đoán vị trí đối tượng tại 5 mốc thời gian (0.5s/1s/1.5s/2s/3s) |
| **Tầng 1 — Kalman** | `KalmanFilter2D` constant-acceleration, state `[x,y,vx,vy,ax,ay]`, dt-aware (unix timestamp thực) |
| **Tầng 2 — Learned (mới)** | `TrajectoryGRU` — Seq2Seq GRU đa-mode (3 hypotheses) + intent classification head, blend với Kalman theo mode-probability |
| **Dataset** | `collect_trajectory_data.py` sinh CSV từ CARLA (calibration pixel↔world) |
| **Checkpoint** | `weights/trajectory_gru.pth` — ✅ đã train 50 epoch trên dữ liệu CARLA thật |

**Trạng thái:**
- Kalman Filter constant-acceleration: ✅ **Hoàn thành** (2026-06-10), đã thay thế
  hoàn toàn linear extrapolation cũ (vốn có lỗi đơn vị px/frame và `window_size`
  không có tác dụng).
- TrajectoryGRU (Seq2Seq + Ensemble): ✅ **Hoàn thành** (2026-06-12) — đã chạy
  CARLA headless (`CarlaUE4.exe -RenderOffScreen -quality-level=Low`), thu thập
  9 episode bằng `collect_trajectory_data.py` (~117,609 dòng dữ liệu world-coord
  x,y,vx,vy), tạo 22,148 sample huấn luyện. Train 50 epoch bằng
  `train_trajectory_predictor.py`, loss giảm từ **0.248 → 0.117**. Checkpoint
  `weights/trajectory_gru.pth` đã lưu, log loss đầy đủ tại
  `docs/assets/trajectory_gru/train_log.txt`.
- **Hạn chế chung:** dự đoán vẫn ở pixel-space riêng từng camera, chưa "lên map"
  chung (cần thêm bước calibration homography pixel↔world cho toàn hệ thống).

---

### 3.5 Incident Detection — Rule-based (12 loại)

| | |
|---|---|
| **Vai trò** | Phát hiện sự cố giao thông từ tracks + predictions |
| **Loại reactive (10)** | SUDDEN_STOP, SUDDEN_ACCEL, OVERSPEED, VEHICLE_PROXIMITY, PEDESTRIAN_DANGER, STOPPED_VEHICLE, LOITERING, CROWD_DENSITY, WRONG_WAY, CAMERA_TRANSITION |
| **Loại proactive (2)** | PREDICTED_COLLISION, PREDICTED_ROI_ENTRY (dùng trajectory prediction) |
| **Cooldown** | 4 giây / loại sự cố / đối tượng |

**Trạng thái:** ✅ **Hoàn thành** (2026-06-01) — cả 12 loại đều hoạt động đúng,
proactive alerts đã được kích hoạt (trước đó bị lỗi do thiếu tham số `predictions`).

> **Hướng nâng cấp:** Isolation Forest / Autoencoder để tự học ngưỡng "bình
> thường" thay vì ngưỡng cứng — chưa triển khai.

---

### 3.6 Tổng hợp tiến độ các model AI

| Model | Vai trò | Trạng thái | Ghi chú |
|---|---|---|---|
| **YOLO11m** | Detection (5 lớp VN) | ✅ **Fine-tune hoàn thành (18 + 10 epoch bổ sung)** | mAP50 0.346→0.559→0.5575, mAP50-95 0.189→0.334→0.3342 |
| **ByteTrack** | Tracking trong camera | ✅ Hoàn thành | Chưa có MOTA/IDF1 định lượng |
| **OSNet (Market-1501)** | Re-ID người | ✅ Hoàn thành | Pretrained, không cần train thêm |
| **OSNet/VeRi-776** | Re-ID xe | ✅ Đã train, tích hợp xong | ⚠️ Re-train cần tải lại dataset VeRi-776 (~1.1GB) |
| **Kalman Filter** | Trajectory (baseline) | ✅ Hoàn thành | Constant-acceleration, dt-aware |
| **TrajectoryGRU (Ensemble)** | Trajectory (learned) | ✅ **Train hoàn thành (50 epoch)** | Loss 0.248→0.117, 22,148 sample từ 9 episode CARLA thật |
| **Incident Detector** | Phát hiện sự cố | ✅ Hoàn thành | 12 loại, proactive hoạt động |

---

## 4. Tiến Độ Của Project

### 4.1 Theo thành phần

| Thành phần | Hoàn thành | Ghi chú |
|---|---|---|
| **AI Pipeline** (`custom_tracking_system/`) — 8 bước Detect→Track→ReID→GlobalID→Trajectory→Alert→Incident→Evidence | **100%** (code) | YOLO11m + TrajectoryGRU đã fine-tune/train xong |
| **Backend API** (`server/`) — REST (17 endpoints) + WebSocket (alerts/tracks/stats) + MJPEG + SQLite | **100%** | Đã sửa 13 bugs, hoạt động đúng |
| **Web Dashboard** (`frontend/`) | **90%** | Thiếu Camera Management UI / ROI editor |
| **Dataset chuẩn bị (Detection)** | **100%** | VisDrone-VN (6,471/548) + CARLA CCTV (1,191/159, 8 camera/100 xe/60 người) đã merge thành `data/visdrone_carla_vn2` |
| **Fine-tune YOLO11m** | ✅ **100%** (18 + 10 epoch) | mAP50=0.5575, mAP50-95=0.3342, checkpoint đã deploy vào `weights/yolo11m_vn.pt` |
| **Re-train OSNet/VeRi-776 (lấy metrics)** | ⚠️ Bị chặn | Cần tải lại dataset VeRi-776 (không còn trên máy) |
| **Train TrajectoryGRU đủ epoch** | ✅ **100%** (50/50 epoch) | Loss 0.248→0.117, checkpoint `weights/trajectory_gru.pth` đã lưu |
| **Ground Truth Evaluation (MOTA/IDF1/mAP)** | **0%** | `ground_truth.py` có sẵn, chưa nối `motmetrics` |
| **Recording liên tục** | **0%** | Chưa implement (`EvidencePackage` chỉ lưu khi có sự cố) |
| **VideoSource → RTSP thực** | **80%** | `main.py --source {bridge,rtsp,file,webcam}` đã sẵn sàng; `server/ai_processor.py` vẫn dùng `CameraController` (CARLA) trực tiếp |

### 4.2 Đánh giá tổng thể

- **Vòng khép kín CARLA → AI Pipeline → Dashboard → Evidence: ~97% hoàn thành**
  (đã demo được end-to-end).
- **Nâng cấp chất lượng model AI cho điều kiện Việt Nam:** ✅ Fine-tune YOLO11m
  trên dataset gộp VisDrone + CARLA (18 epoch ban đầu + 10 epoch bổ sung với
  dataset CARLA mở rộng) đã **hoàn thành** (mAP50 0.346→0.559→0.5575,
  mAP50-95 0.189→0.334→0.3342), checkpoint đã được deploy vào pipeline.
- **TrajectoryGRU đã train xong 50 epoch** với dữ liệu trajectory thật thu từ
  CARLA (9 episode, 117,609 dòng, 22,148 sample), loss giảm 0.248→0.117,
  checkpoint đã lưu vào pipeline.
- **Chỉ còn OSNet/VeRi-776 bị chặn** do thiếu tài nguyên bên ngoài (dataset
  VeRi-776 không có trên máy hiện tại) — không phải do lỗi code. Vẫn hoạt động
  ổn định trong pipeline với checkpoint hiện có.
- **Phần còn thiếu chủ yếu là đánh giá định lượng (MOTA/IDF1/mAP cho tracking)
  và các tính năng vận hành dài hạn (recording, camera management UI)** — không
  ảnh hưởng đến khả năng demo lõi của hệ thống.

---

## 5. Khó Khăn Chính & Giải Pháp

| # | Khó khăn | Giải pháp đã áp dụng |
|---|---|---|
| 1 | CARLA không cung cấp biển số xe → Re-ID xe khó phân biệt xe cùng màu/kiểu | Dùng OSNet fine-tune VeRi-776 + Spatio-Temporal Filter |
| 2 | Xung đột Python 3.7 (CARLA) vs Python 3.8+ (boxmot/ultralytics) | Kiến trúc `carla_bridge` — 2 process giao tiếp qua TCP, mỗi process dùng đúng phiên bản Python cần thiết |
| 3 | Trajectory prediction chỉ ở pixel-space riêng từng camera | Đã thêm `CameraCalibration` (pixel↔world); cần áp dụng đồng bộ cho dashboard "lên map" |
| 4 | YOLO pretrained COCO nhận diện kém xe máy VN | Fine-tune YOLO11m trên VisDrone-VN + dữ liệu CARLA tự thu thập (đang chạy) |
| 5 | Huấn luyện trên máy local RAM/VRAM hạn chế gây OOM | Giảm batch size (16→8), giảm dataloader workers (8→4), resume từ checkpoint khi crash |
| 6 | Chưa có đánh giá định lượng MOTA/IDF1/mAP cho tracking | Việc tiếp theo: nối `ground_truth.py` với `motmetrics` |

---

## 6. Kế Hoạch Tiếp Theo (Theo Thứ Tự)

1. ✅ ~~Hoàn thành fine-tune YOLO11m~~ — **Xong** (2026-06-12), 18 + 10 epoch bổ
   sung, mAP50=0.5575.
2. ✅ ~~Train TrajectoryGRU đủ epoch~~ — **Xong** (2026-06-12), 50 epoch, loss
   0.248→0.117, dữ liệu từ 9 episode CARLA thật.
3. ⏳ **Tải lại dataset VeRi-776** (~1.1GB) rồi re-train OSNet/VeRi-776, lưu log
   `val_acc` và biểu đồ vào `docs/assets/osnet_veri776/`.
4. **Ground Truth Evaluation** — nối `motmetrics` để có MOTA/IDF1 cho ByteTrack.
5. **Calibration pixel↔world toàn hệ thống** — hiển thị trajectory dự đoán "lên map".
6. **Recording liên tục + Camera Management UI** — các tính năng vận hành còn thiếu.

---

*Báo cáo này được cập nhật tự động khi các quá trình train (YOLO11m, OSNet/VeRi-776,
TrajectoryGRU) hoàn tất — xem `docs/production_model_stack.md` và `docs/upgrade.md`
để biết chi tiết kỹ thuật và số liệu cuối cùng.*
