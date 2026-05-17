# BÁO CÁO TIẾN ĐỘ DỰ ÁN

## 1. Tên Dự Án

**Hệ Thống Giám Sát Camera CCTV Đa Kênh Ứng Dụng Trí Tuệ Nhân Tạo**
*(Multi-Camera CCTV Tracking System with AI)*

Môi trường mô phỏng: CARLA Simulator phiên bản 0.9.9.4.

---

## 2. Mục Tiêu Của Dự Án

### 2.1 Vấn đề dự án hướng tới

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

### 2.2 Mục tiêu cụ thể của dự án

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

## 3. Công Việc Đã Hoàn Thành

### 3.1 AI Pipeline (`custom_tracking_system/`)

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

### 3.2 Backend API Server (`server/`)

Đã hoàn thành backend FastAPI bao gồm:

- **17 REST endpoints** cho quản lý: camera, tracks, alerts, ROIs, thống kê.
- **3 WebSocket channels**: push cảnh báo, tracking update, thống kê real-time.
- **MJPEG streaming**: truyền video trực tiếp tới trình duyệt qua HTTP.
- **Database SQLite** với 5 bảng: `cameras`, `alerts`, `tracked_objects`,
  `tracking_history`, `rois` (dùng SQLAlchemy ORM + Pydantic v2).
- **AI Processor** chạy pipeline AI trong background thread của server,
  push kết quả qua WebSocket và lưu vào database.

### 3.3 Cấu hình và công cụ phụ trợ

- File cấu hình YAML cho 3 camera (CAM_001, CAM_002, CAM_003) với 3 vùng ROI.
- Module visualization vẽ bounding box, quỹ đạo, lưới đa camera bằng OpenCV.
- Module thu thập metrics (FPS, số lượng detection/tracking).
- Module xuất dữ liệu ra JSON, CSV và báo cáo tổng kết.
- Logging hệ thống đầy đủ (file `tracking_system.log`).

### 3.4 Ba chế độ chạy

| Chế độ | Lệnh | Mục đích |
|--------|------|---------|
| API-only | `python app.py` | Server hoạt động không cần CARLA, dùng để phát triển frontend. |
| API + AI | `python app.py --with-ai` | Full system với CARLA, AI pipeline chạy nền. |
| Direct | `python main.py` | Hiển thị kết quả qua cửa sổ OpenCV, không qua server. |

### 3.5 Tài liệu

Đã viết đầy đủ 7 tài liệu trong thư mục `docs/`:
`description.md`, `plan.md`, `workflow.md`, `execute.md`,
`development_roadmap.md`, `report.md`, `project_context.md`.

---

## 4. Tiến Độ Hiện Tại

### 4.1 Hạng mục đã hoàn thành

- AI Pipeline đầy đủ 8 bước, chạy ổn định trên môi trường CARLA.
- Backend API Server (FastAPI) với REST + WebSocket + MJPEG streaming.
- Database 5 bảng lưu trữ toàn bộ dữ liệu camera, alerts, tracks, history, ROIs.
- Tích hợp AI pipeline vào server qua background thread.
- Lớp trừu tượng VideoSource (sẵn sàng cho camera thực tế).
- Module Ground Truth tách biệt phục vụ đánh giá.
- Hệ thống cấu hình, logging, xuất dữ liệu.

### 4.2 Hạng mục đang/chưa triển khai

| Hạng mục | Trạng thái | Ghi chú |
|---------|-----------|---------|
| Web Dashboard (frontend) | Chưa làm | Hiện chỉ có API, chưa có giao diện người dùng. |
| Anomaly Detection | Chưa làm | Mới có cảnh báo ROI; cần thêm overspeed, wrong-way, loitering, crowd. |
| Nâng cấp Tracker | Chưa làm | Cần thay IoU matching bằng DeepSORT/ByteTrack + Kalman filter. |
| Vehicle Re-ID | Chưa làm | Hiện chỉ có OSNet train cho person, chưa có model riêng cho xe. |
| Spatio-temporal Reasoning | Chưa làm | Cần dùng thời gian + khoảng cách giữa camera để phân biệt xe giống nhau. |
| Recording & Playback | Chưa làm | Chưa ghi video, chưa cắt clip sự cố. |
| Camera Management UI | Chưa làm | Chưa có giao diện thêm/xóa camera khi đang chạy. |
| Ground Truth Evaluation | Chưa làm | Module đã có, chưa tích hợp tính MOTA, IDF1, mAP. |
| Tích hợp VideoSource | Chưa làm | `video_source.py` đã viết nhưng pipeline vẫn dùng `camera_controller` trực tiếp. |

### 4.3 Tóm tắt

Dự án đã hoàn thành **phần lõi xử lý AI** và **phần backend API**.
Hệ thống đã có khả năng phát hiện, theo dõi, nhận diện lại đối tượng và
cảnh báo cơ bản. Phần còn lại là **giao diện người dùng**, **các thuật toán
phát hiện bất thường nâng cao**, và **đánh giá hiệu năng định lượng**.

---

## 5. Khó Khăn Gặp Phải

### 5.1 Khó khăn về mặt kỹ thuật

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

### 5.2 Khó khăn về môi trường và công cụ

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

### 5.3 Khó khăn về đánh giá

**(8) Chưa có ground truth để đo định lượng**

Mặc dù đã viết module `ground_truth.py`, nhưng chưa tích hợp được vào quy trình
đánh giá để tính các chỉ số chuẩn như MOTA (Multi-Object Tracking Accuracy),
IDF1, và mAP. Hiện chỉ đánh giá định tính qua quan sát.

---

## 6. Kế Hoạch Tiếp Theo

Dựa trên `development_roadmap.md`, các công việc dự kiến triển khai trong
thời gian tới được sắp xếp theo thứ tự ưu tiên:

1. **Tích hợp đánh giá định lượng** — kết nối module `ground_truth.py` vào
   pipeline để tính MOTA, IDF1, mAP, có số liệu cụ thể đánh giá chất lượng AI.
2. **Nâng cấp Tracker** — thay thuật toán IoU greedy bằng DeepSORT hoặc
   ByteTrack để giảm hiện tượng hoán đổi ID.
3. **Xây dựng Web Dashboard** — frontend React/Vue hiển thị camera grid,
   panel cảnh báo, bản đồ vị trí đối tượng, trình chỉnh sửa ROI.
4. **Bổ sung Anomaly Detection** — phát hiện overspeed, đi ngược chiều,
   loitering, mật độ đám đông cao.
5. **Spatio-temporal reasoning** — dùng thời gian và khoảng cách giữa các
   camera để phân biệt phương tiện giống nhau xuyên camera.
6. **Recording & Playback** — ghi video liên tục, tự động cắt clip khi xảy
   ra cảnh báo.
7. **Tích hợp VideoSource** — thay `camera_controller` bằng `video_source`
   trong pipeline chính, sẵn sàng cho camera IP thực tế.

---

*Báo cáo được lập ngày 17/05/2026.*
