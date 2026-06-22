# HỆ THỐNG GIÁM SÁT GIAO THÔNG THÔNG MINH ĐA CAMERA TRÊN MÔI TRƯỜNG MÔ PHỎNG CARLA VỚI BẢN ĐỒ HÀ NỘI

**Trường Đại học Bách khoa Hà Nội**  
**Viện Công nghệ Thông tin và Truyền thông (SOICT)**  
**Chuyên ngành: Khoa học Máy tính**

**Sinh viên thực hiện:** [Tên sinh viên]  
**Mã số sinh viên:** [MSSV]  
**Giảng viên hướng dẫn:** [Tên giảng viên]  
**Hà Nội, tháng 6/2026**

---

## TÓM TẮT (Tiếng Việt)

Đề tài nghiên cứu và xây dựng hệ thống giám sát giao thông thông minh đa camera áp dụng các kỹ thuật học sâu (deep learning) và mô phỏng vật lý trên nền tảng CARLA Simulator với bản đồ đô thị Hà Nội được tái tạo từ dữ liệu OpenStreetMap.

Hệ thống tích hợp năm module AI hoạt động theo pipeline: (1) **Phát hiện đối tượng** sử dụng YOLO11s; (2) **Tracking đa camera đơn** dựa trên ByteTrack; (3) **Nhận diện lại xuyên camera** (cross-camera ReID) bằng OSNet; (4) **Phân loại hướng đi tại ngã tư** (Goal Classifier) dùng GradientBoosting với 26 đặc trưng trajectory; (5) **Dự đoán quỹ đạo** bằng mạng Seq2Seq GRU đa modal.

Bộ dữ liệu trajectory được thu thập từ 120 episode mô phỏng CARLA trên bản đồ Hà Nội (quận Hà Đông/Mỗ Lao), gồm 4.743.151 điểm trajectory, 36.997 sự kiện qua ngã tư và 9.703 bản ghi hậu tai nạn với tỉ lệ xe máy 70% phù hợp thực tế Hà Nội. Goal Classifier đạt độ chính xác **70,3%** tại horizon 1,5 giây trước ngã tư, ECE = 0,104, thời gian suy luận 3,97ms cho 30 phương tiện đồng thời.

**Từ khóa:** giám sát giao thông thông minh, CARLA, Goal Classifier, trajectory prediction, ByteTrack, OSNet, Hà Nội.

---

## ABSTRACT (English)

This thesis presents an intelligent multi-camera traffic monitoring system combining deep learning and physics-based simulation on the CARLA platform, with a Hanoi urban map reconstructed from OpenStreetMap data.

The system integrates five AI modules in a unified pipeline: (1) object detection via YOLO11s; (2) single-camera tracking with ByteTrack; (3) cross-camera re-identification using OSNet; (4) intersection goal direction classification (straight/left/right/u-turn) via a calibrated GradientBoosting classifier with 26 trajectory features; and (5) multi-modal trajectory prediction using a Seq2Seq GRU network.

The trajectory dataset was collected from 120 CARLA simulation episodes on a real Hanoi district map (Ha Dong / Mo Lao), comprising 4,743,151 trajectory rows, 36,997 junction goal events, and 9,703 post-accident records. The motorcycle ratio of 70% reflects real Hanoi traffic composition. The Goal Classifier achieves **70.3% accuracy** at the 1.5-second prediction horizon, ECE = 0.104, and inference time of 3.97 ms for 30 simultaneous vehicles.

**Keywords:** intelligent traffic monitoring, CARLA simulation, goal classifier, trajectory prediction, ByteTrack, OSNet, Hanoi.

---

## LỜI CẢM ƠN

Tôi xin gửi lời cảm ơn chân thành đến giảng viên hướng dẫn đã định hướng và hỗ trợ trong suốt quá trình thực hiện đề tài. Cảm ơn các thành viên nhóm nghiên cứu và cộng đồng mã nguồn mở CARLA, Ultralytics, torchreid đã cung cấp nền tảng kỹ thuật vững chắc. Cảm ơn gia đình và bạn bè đã động viên trong thời gian thực hiện luận văn.

---

## MỤC LỤC

- Chương 1: Tổng quan
- Chương 2: Cơ sở lý thuyết
- Chương 3: Phân tích và thiết kế hệ thống
- Chương 4: Cài đặt và triển khai
- Chương 5: Kiểm thử và đánh giá
- Chương 6: Kết luận và hướng phát triển
- Tài liệu tham khảo
- Phụ lục

---

## DANH MỤC HÌNH ẢNH

| STT | Hình | Mô tả |
|-----|------|--------|
| 1 | Hình 1.1 | Kiến trúc tổng thể hệ thống |
| 2 | Hình 1.2 | Pipeline OSM → OpenDRIVE → CARLA |
| 3 | Hình 2.1 | Kiến trúc YOLO11s |
| 4 | Hình 2.2 | Sơ đồ ByteTrack |
| 5 | Hình 2.3 | Kiến trúc OSNet cho ReID |
| 6 | Hình 2.4 | Kiến trúc Seq2Seq GRU trajectory predictor |
| 7 | Hình 3.1 | Use Case Diagram tổng quan |
| 8 | Hình 3.2 | Activity Diagram luồng thu thập dữ liệu |
| 9 | Hình 3.3 | Sequence Diagram tracking xuyên camera |
| 10 | Hình 3.4 | Class Diagram hệ thống |
| 11 | Hình 4.1 | Bản đồ Hanoi District 3 trong CARLA |
| 12 | Hình 4.2 | Ví dụ trajectory data thu thập được |
| 13 | Hình 4.3 | Feature importance của Goal Classifier |
| 14 | Hình 5.1 | Accuracy theo horizon (0.5s – 3.0s) |
| 15 | Hình 5.2 | Confusion matrix Goal Classifier |
| 16 | Hình 5.3 | Reliability diagram (calibration) |

---

## DANH MỤC BẢNG BIỂU

| STT | Bảng | Mô tả |
|-----|------|--------|
| 1 | Bảng 2.1 | So sánh các hệ thống giám sát giao thông hiện có |
| 2 | Bảng 3.1 | Đặc tả Use Case thu thập dữ liệu |
| 3 | Bảng 3.2 | Đặc tả Use Case dự đoán hướng đi |
| 4 | Bảng 4.1 | Thông số phần cứng thực nghiệm |
| 5 | Bảng 4.2 | Thông số thu thập dữ liệu CARLA |
| 6 | Bảng 4.3 | Thống kê bộ dữ liệu trajectories_hanoi_v2 |
| 7 | Bảng 4.4 | Danh sách 26 đặc trưng Goal Classifier |
| 8 | Bảng 4.5 | Kiến trúc mạng GRU trajectory predictor |
| 9 | Bảng 5.1 | Kết quả Goal Classifier theo horizon |
| 10 | Bảng 5.2 | Kết quả Goal Classifier theo loại phương tiện |
| 11 | Bảng 5.3 | Recall từng nhãn tại horizon 1,5s |
| 12 | Bảng 5.4 | Test cases kiểm thử hệ thống |

---

## DANH MỤC TỪ VIẾT TẮT

| Từ viết tắt | Nghĩa đầy đủ |
|-------------|--------------|
| AI | Artificial Intelligence – Trí tuệ nhân tạo |
| CARLA | Car Learning to Act (phần mềm mô phỏng) |
| CCTV | Closed-Circuit Television – Camera an ninh |
| ECE | Expected Calibration Error |
| FPS | Frames Per Second – Khung hình mỗi giây |
| GRU | Gated Recurrent Unit |
| IoU | Intersection over Union |
| MOTA | Multiple Object Tracking Accuracy |
| OSM | OpenStreetMap |
| ReID | Re-Identification – Nhận diện lại |
| ROI | Region of Interest – Vùng quan tâm |
| UE4 | Unreal Engine 4 |
| VRAM | Video Random Access Memory |
| YOLO | You Only Look Once |

---

# CHƯƠNG 1: TỔNG QUAN

## 1.1 Bối cảnh và động lực

Hà Nội là một trong những thành phố có mật độ giao thông cao nhất Đông Nam Á, với đặc thù lưu lượng xe máy chiếm khoảng 70–75% tổng số phương tiện. Hệ thống CCTV giao thông đô thị hiện nay chủ yếu ghi hình thụ động, thiếu khả năng phân tích hành vi phương tiện theo thời gian thực: không thể dự báo hướng đi tại ngã tư, không theo dõi cùng một phương tiện qua nhiều camera, và không tự động phát hiện sự cố.

Việc xây dựng và kiểm thử các thuật toán phân tích giao thông đòi hỏi dữ liệu được gán nhãn chính xác trong các kịch bản đa dạng (bình thường, vượt đèn đỏ, tai nạn, bỏ trốn...). Thu thập dữ liệu thực tế tốn kém và có những rủi ro pháp lý khi ghi hình người thật. Môi trường mô phỏng vật lý như CARLA cung cấp dữ liệu ground-truth hoàn hảo, có thể tái lập kịch bản tuỳ ý, và cho phép kiểm thử các tình huống nguy hiểm một cách an toàn.

## 1.2 Mục tiêu đề tài

Đề tài hướng đến các mục tiêu:

1. Xây dựng pipeline chuyển đổi bản đồ OSM thực tế Hà Nội sang định dạng OpenDRIVE và nạp vào CARLA để mô phỏng giao thông đô thị với thành phần phương tiện đặc trưng Hà Nội.
2. Xây dựng hệ thống thu thập dữ liệu trajectory tự động từ CARLA bao gồm cả kịch bản tai nạn.
3. Thiết kế và huấn luyện **Goal Direction Classifier** dự đoán hướng đi của phương tiện tại ngã tư từ 0,5–1,5 giây trước, phù hợp với dữ liệu xe máy Hà Nội.
4. Xây dựng **Trajectory Predictor** GRU đa modal dự báo vị trí phương tiện theo nhiều kịch bản tương lai.
5. Tích hợp toàn bộ thành pipeline đa camera hoàn chỉnh: phát hiện → tracking → ReID → phân tích hành vi → cảnh báo.

## 1.3 Phạm vi nghiên cứu

- **Phạm vi**: Mô phỏng giao thông đô thị Hà Nội trên CARLA, không phải hệ thống CCTV vật lý triển khai trực tiếp.
- **Loại phương tiện**: Xe máy (≥70%), ô tô con, xe buýt, xe tải, người đi bộ.
- **Khu vực bản đồ**: Quận Hà Đông/Mỗ Lao (hanoi_district3), chuyển đổi từ OSM.
- **Không thuộc phạm vi**: Tích hợp CCTV thực tế, giao tiếp đèn tín hiệu thực, nhận dạng biển số xe.

## 1.4 Các công trình liên quan

**Bảng 2.1: So sánh các hệ thống giám sát giao thông**

| Hệ thống | Phương pháp | Đặc điểm | Hạn chế |
|----------|-------------|----------|---------|
| DeepSORT [1] | YOLO + Deep ReID | Tracking đơn camera chuẩn | Không hỗ trợ đa camera, không dự đoán |
| StrongSORT [2] | YOLO + EMA features | Cải tiến DeepSORT | Chỉ đơn camera |
| ByteTrack [3] | Byte-level association | Xử lý low-confidence detections | Không cross-camera |
| TrafficPredict [4] | LSTM trajectory | Dự đoán giao thông đô thị | Không mô phỏng Việt Nam |
| CityFlow [5] | Graph + attention | Hệ thống đa camera hoàn chỉnh | Dữ liệu phương Tây, không xe máy |
| **Hệ thống đề xuất** | YOLO11s + ByteTrack + OSNet + GRU | Tích hợp Goal+Trajectory+Alert | Môi trường mô phỏng, chưa CCTV thực |

## 1.5 Cấu trúc báo cáo

Báo cáo được tổ chức thành 6 chương: Chương 1 trình bày tổng quan; Chương 2 trình bày cơ sở lý thuyết; Chương 3 phân tích và thiết kế hệ thống; Chương 4 mô tả cài đặt và triển khai; Chương 5 trình bày kết quả kiểm thử; Chương 6 kết luận và hướng phát triển.

---

# CHƯƠNG 2: CƠ SỞ LÝ THUYẾT

## 2.1 CARLA Simulator

CARLA (Car Learning to Act) là nền tảng mô phỏng lái xe tự động mã nguồn mở, xây dựng trên Unreal Engine 4. CARLA cung cấp:

- **Vật lý thực tế**: Dynamics xe cộ, ma sát đường, điều kiện thời tiết.
- **Sensor đa dạng**: Camera RGB, camera độ sâu, LiDAR, radar, GNSS.
- **API Python**: Toàn quyền kiểm soát actor, vật lý, thời tiết.
- **Traffic Manager**: Sinh phương tiện NPC với hành vi giao thông configurable.
- **Hỗ trợ OpenDRIVE**: Nạp bản đồ đường từ file `.xodr` chuẩn.

CARLA chạy ở chế độ **synchronous mode** với `fixed_delta_seconds = 0.1s` (10 Hz), đảm bảo tính tái lập hoàn toàn của simulation.

## 2.2 Object Detection — YOLO11s

YOLO (You Only Look Once) là họ model phát hiện đối tượng một giai đoạn. YOLO11s là phiên bản "small" của dòng YOLO11 (Ultralytics, 2024), cân bằng tốt giữa tốc độ và độ chính xác:

- **Backbone**: C3k2 blocks với attention mechanism
- **Neck**: Bi-directional Feature Pyramid Network (BiFPN-style)
- **Head**: Anchor-free detection với Decoupled head
- **Tham số**: ~9M parameters, phù hợp GPU hạn chế

Đặc trưng YOLO11 so với YOLOv5/v8: cải thiện mAP trên ảnh nhỏ, tích hợp attention không tăng đáng kể latency.

## 2.3 Multi-Object Tracking — ByteTrack

ByteTrack [3] là thuật toán tracking đơn camera theo paradigm tracking-by-detection:

- Liên kết **tất cả** detection box, kể cả confidence thấp (không loại bỏ như SORT).
- Hai giai đoạn matching: high-confidence detections trước, low-confidence sau.
- Motion model: Kalman Filter trên không gian [cx, cy, w, h, vx, vy, vw, vh].
- Ưu điểm: Xử lý tốt phương tiện bị che khuất tạm thời (occlusion).

## 2.4 Cross-Camera ReID — OSNet

OSNet (Omni-Scale Network) [6] là kiến trúc CNN được thiết kế riêng cho vehicle/person re-identification:

- **Omni-scale feature learning**: Các nhánh convolution ở nhiều scale được kết hợp bằng Unified Aggregation Gates.
- **Nhẹ và nhanh**: ~2,2M parameters, phù hợp inference thời gian thực.
- **Pretrained**: Trên Market-1501 và VeRi-776 (vehicle ReID).

Trong hệ thống, OSNet trích xuất embedding vector 512-d từ ảnh crop phương tiện; cosine similarity được dùng để matching xuyên camera.

## 2.5 Goal Direction Classification

**Bài toán**: Cho chuỗi trajectory T-giây cuối của một phương tiện đang tiếp cận ngã tư, dự đoán hướng đi: `{straight, left, right, u_turn}`.

**Tiếp cận**: Thay vì dùng mạng sequence-to-sequence, đề tài dùng GradientBoosting Classifier (scikit-learn) với 26 đặc trưng engineered từ trajectory window 4 giây:

- Đặc trưng hướng: `head_sin`, `head_cos`, `head_rate_mean`, `head_rate_max`, `heading_range`
- Đặc trưng vận tốc: `speed_mean`, `speed_change_rate`, `decel_rate`
- Đặc trưng vị trí: `dist_to_junction`, `lateral_offset`
- Đặc trưng thống kê: sliding statistics trên cửa sổ con 1s, 2s, 4s

**Calibration**: Platt scaling để ECE ≤ 0,15 (confidence đáng tin cậy cho dashboard).

## 2.6 Trajectory Prediction — Seq2Seq GRU

Mô hình dự đoán quỹ đạo được xây dựng theo kiến trúc Seq2Seq đa modal:

**Encoder**: GRU xử lý `T_OBS = 8` bước quan sát (0,8 giây tại 10Hz). Input mỗi bước:
- [x, y, vx, vy] (normalized)
- One-hot class vector (car/motorcycle/person)  
- K=3 láng giềng gần nhất: relative [dx, dy, dvx, dvy] mỗi láng giềng

Tổng input dimension = 4 + 3 + 3×4 = **19**.

**Decoder**: GRUCell unrolled theo `PRED_HORIZONS = (0.5, 1.0, 1.5, 2.0, 3.0)` giây, chạy độc lập cho `NUM_MODES = 3` giả thuyết.

**Auxiliary heads** (từ encoder hidden state):
- Mode probabilities (softmax 3 modes)
- Intent classification (straight/turn_left/turn_right/stop)

**Hidden size**: 96. **Mode embedding**: 8-dim.

## 2.7 Bản đồ OpenDRIVE và Pipeline OSM→CARLA

OpenDRIVE là định dạng XML mô tả hình học đường, làn xe, ngã tư, tín hiệu giao thông. Workflow chuyển đổi:

```
OpenStreetMap (.osm) → osmium/netconvert → OpenDRIVE (.xodr) → CARLA loader
```

File `hanoi_district3.xodr` mô tả khu vực Mỗ Lao, Hà Đông với đầy đủ lane markings và junction connectivity. CARLA tự động sinh waypoint graph từ OpenDRIVE để Traffic Manager điều hướng NPC.

---

# CHƯƠNG 3: PHÂN TÍCH VÀ THIẾT KẾ HỆ THỐNG

## 3.1 Kiến trúc tổng thể

```
┌─────────────────────────────────────────────────────────────┐
│                    CARLA Simulation                          │
│  [Map: hanoi_district3] + [Traffic Manager] + [Scenarios]   │
└────────────────────┬────────────────────────────────────────┘
                     │ Camera frames (10 FPS, 1920×1080)
                     ▼
┌─────────────────────────────────────────────────────────────┐
│              Perception Layer                                │
│  ┌─────────────┐   ┌──────────────┐   ┌─────────────────┐  │
│  │  YOLO11s    │→  │  ByteTrack   │→  │  OSNet ReID     │  │
│  │  Detection  │   │  (per-cam)   │   │  (cross-cam)    │  │
│  └─────────────┘   └──────────────┘   └─────────────────┘  │
└────────────────────┬────────────────────────────────────────┘
                     │ Global tracks + embeddings
                     ▼
┌─────────────────────────────────────────────────────────────┐
│              Behavior Modeling Layer                         │
│  ┌──────────────────────┐   ┌───────────────────────────┐  │
│  │  Goal Classifier     │   │  Trajectory Predictor     │  │
│  │  (GradientBoosting)  │   │  (Seq2Seq GRU)            │  │
│  │  straight/L/R/U      │   │  3 modes × 5 horizons     │  │
│  └──────────────────────┘   └───────────────────────────┘  │
└────────────────────┬────────────────────────────────────────┘
                     │ Predictions + probabilities
                     ▼
┌─────────────────────────────────────────────────────────────┐
│              Application Layer                               │
│  [Alert System] + [Incident Detector] + [Evidence Package]  │
│  [Visualization Dashboard] + [Metrics Collector]            │
└─────────────────────────────────────────────────────────────┘
```

**Hình 1.1: Kiến trúc tổng thể hệ thống**

## 3.2 Use Case Diagram

### 3.2.1 Use Case Tổng quan

Hệ thống phục vụ hai actor chính:
- **Operator** (người vận hành hệ thống giám sát): xem luồng video, nhận cảnh báo, xem trajectory predictions.
- **CARLA Simulation** (nguồn dữ liệu): cung cấp camera frames và ground truth.

```
                    ┌─────────────────────────────┐
                    │         <<System>>           │
                    │   Traffic Monitoring System  │
                    │                             │
Operator ──────────►│  UC1: Xem camera tracking   │
                    │  UC2: Nhận cảnh báo sự cố   │
                    │  UC3: Xem goal prediction    │
                    │  UC4: Xem trajectory pred    │
                    │  UC5: Export evidence        │
CARLA ─────────────►│  UC6: Thu thập trajectory   │
Simulation          │  UC7: Sinh kịch bản tai nạn │
                    └─────────────────────────────┘
```

**Hình 3.1: Use Case Diagram tổng quan**

### 3.2.2 Đặc tả Use Case chính

**Bảng 3.1: Đặc tả UC3 — Xem Goal Prediction tại ngã tư**

| Trường | Nội dung |
|--------|---------|
| Tên | UC3 — Xem dự đoán hướng đi tại ngã tư |
| Tác nhân | Operator |
| Tiền điều kiện | Hệ thống đang chạy, có phương tiện trong ROI ngã tư |
| Kịch bản chính | 1. Phương tiện tiếp cận vùng pre-junction (distance < 30m) |
| | 2. Trajectory Collector thu thập 4s trajectory gần nhất |
| | 3. Feature extractor tính 26 đặc trưng |
| | 4. Goal Classifier trả về phân phối xác suất [P_straight, P_left, P_right, P_uturn] |
| | 5. Visualizer vẽ mũi tên màu sắc tương ứng xác suất lên frame |
| | 6. Dashboard cập nhật thống kê lưu lượng theo hướng |
| Hậu điều kiện | Operator thấy dự đoán real-time, log được lưu |
| Luồng thay thế | Nếu confidence < 0.4: hiển thị "uncertain", không vẽ mũi tên |

**Bảng 3.2: Đặc tả UC6 — Thu thập dữ liệu trajectory từ CARLA**

| Trường | Nội dung |
|--------|---------|
| Tên | UC6 — Thu thập dữ liệu trajectory tự động |
| Tác nhân | CARLA Simulation |
| Tiền điều kiện | CARLA server chạy, bản đồ hanoi_district3 đã nạp |
| Kịch bản chính | 1. Spawn 20 phương tiện (70% xe máy) + 5 người đi bộ |
| | 2. (25% xác suất) Kích hoạt kịch bản tai nạn ngẫu nhiên |
| | 3. Ghi trajectory CSV mỗi tick (0,1s): [t, actor_id, x, y, vx, vy, heading, junction_id, phase] |
| | 4. Ghi goal event khi phương tiện vào/ra junction |
| | 5. Ghi post-accident rows với `phase='post_accident'` |
| | 6. Sau 1800 ticks (180s): lưu episode, spawn lại |
| Hậu điều kiện | episode_XXXX.csv + episode_XXXX_goals.csv được ghi |
| Luồng thay thế | Crash CARLA: resume từ checkpoint.json |

## 3.3 Activity Diagram — Luồng thu thập dữ liệu

```
[Khởi động] → [Load hanoi_district3.xodr] → [Spawn vehicles (70% moto)] 
    → {scenario_ratio=25%?} 
        → [Có] → [Chọn scenario: red_light/rear_end/pedestrian_hit/hit_and_run/sudden_stop]
        → [Không] → [Normal traffic]
    → [Chạy 1800 ticks @ 10Hz]
        → [Mỗi tick: ghi trajectory CSV]
        → [Khi xe vào junction: ghi goal event]
        → [Khi phase=post_accident: ghi post-accident rows]
    → [Lưu episode files]
    → {episodes_done < 120?} → [Có] → loop
    → [Không] → [Ghi run_summary.json] → [Kết thúc]
```

**Hình 3.2: Activity Diagram — Luồng thu thập dữ liệu CARLA**

## 3.4 Sequence Diagram — Tracking xuyên camera

```
Operator    CARLABridge    YOLO11s    ByteTrack    OSNet     GlobalTracker
   |              |           |           |          |             |
   |──start───►  |           |           |          |             |
   |         frame×N         |           |          |             |
   |              |──detect──►|           |          |             |
   |              |          boxes        |          |             |
   |              |──────────────────────►|          |             |
   |              |                    update tracks |             |
   |              |──────────────────────────────────►|             |
   |              |                             embeddings          |
   |              |──────────────────────────────────────────────►  |
   |              |                                          match/assign global_id
   |              |                                                  |
   |◄─ tracks + global_ids + goal_probs + traj_predictions ─────────|
```

**Hình 3.3: Sequence Diagram — Tracking xuyên camera**

## 3.5 Class Diagram

**Bảng cấu trúc lớp chính:**

| Lớp | Thuộc tính chính | Phương thức chính |
|-----|-----------------|-----------------|
| `TrackingSystem` | config_path, source, half, modules{} | initialize(), run(), shutdown() |
| `ObjectDetector` | model (YOLO11s), conf_threshold=0.25, iou_threshold=0.45 | detect(frame) → List[Detection] |
| `ByteTrackWrapper` | tracker, camera_id | update(detections, frame) → List[Track] |
| `ReIDExtractor` | model (OSNet), gallery{} | extract(crop) → np.ndarray[512] |
| `GlobalTracker` | tracks{}, match_threshold=0.7 | update(cam_tracks) → List[GlobalTrack] |
| `GoalClassifier` | model (GBT), scaler, feature_names[26] | predict_proba(traj_window) → Dict |
| `LearnedTrajectoryPredictor` | gru_model, Kalman (ensemble) | predict(obs_sequence) → MultiModalPred |
| `AlertSystem` | rules[], cooldown_map{} | check(global_tracks) → List[Alert] |
| `IncidentDetector` | collision_threshold | detect(tracks, world) → List[Incident] |
| `EvidencePackage` | incident_id, frames[], tracks[] | export(path) → str |

---

# CHƯƠNG 4: CÀI ĐẶT VÀ TRIỂN KHAI

## 4.1 Môi trường phát triển

**Bảng 4.1: Thông số phần cứng**

| Thành phần | Thông số |
|-----------|---------|
| CPU | Intel Core i5-9300H (4 cores, 8 threads, 2.4–4.1 GHz) |
| GPU | NVIDIA GTX 1050Ti, 4GB GDDR5 VRAM |
| RAM | 16GB DDR4 2666MHz |
| Storage | SSD NVMe |
| OS | Windows 10/11 x64 |

**Stack công nghệ:**

| Thành phần | Công nghệ / Phiên bản |
|-----------|----------------------|
| Mô phỏng | CARLA 0.9.x + Unreal Engine 4 |
| Phát hiện | Ultralytics YOLO11s |
| Tracking | ByteTrack (custom port) |
| ReID | deep-person-reid / OSNet |
| Classifier | scikit-learn GradientBoosting |
| Deep learning | PyTorch 2.x + CUDA |
| Ngôn ngữ | Python 3.10 |
| Format bản đồ | OpenDRIVE + OpenStreetMap |

## 4.2 Pipeline OSM → OpenDRIVE → CARLA

**Hình 1.2: Pipeline xây dựng bản đồ Hà Nội**

```
OpenStreetMap.org ──► [Download .osm vùng Hà Đông] 
    ──► osmium + SUMO netconvert ──► hanoi_district3.xodr
    ──► CARLA map loader ──► World trong UE4
    ──► Traffic Manager sử dụng waypoint graph từ .xodr
```

**Chi tiết kỹ thuật:**
- Công cụ chuyển đổi: SUMO `netconvert` + osmium pre-processing
- File output: `Map/hanoi_district3.xodr`
- Đặc điểm bản đồ: ~15 ngã tư, tốc độ giới hạn 40–60 km/h, 2–4 làn/chiều
- Thách thức: OSM Việt Nam thiếu thông tin làn xe → cần bổ sung thủ công topology ngã tư chính

## 4.3 Thu thập dữ liệu trajectory

**Bảng 4.2: Thông số thu thập dữ liệu**

| Tham số | Giá trị |
|---------|---------|
| Map | hanoi_district3 |
| Episodes | 120 |
| Steps/episode | 1.800 ticks (180 giây) |
| Fixed_delta_seconds | 0,1s (10 Hz) |
| Số phương tiện/episode | 20 + 5 người đi bộ |
| Tỉ lệ xe máy | 0,70 (moto_ratio=0.70) |
| Tỉ lệ kịch bản tai nạn | 25% (scenario_ratio=0.25) |
| Thời gian chạy total | 37,7 phút |

**Bảng 4.3: Thống kê bộ dữ liệu trajectories_hanoi_v2**

| Thống kê | Giá trị |
|---------|---------|
| Tổng trajectory rows | 4.743.151 |
| Tổng goal events | 36.997 |
| Post-accident rows | 9.703 |
| Tỉ lệ motorcycle (thực tế) | 72,7% |
| Car | 19,1% |
| Bus | 3,1% |
| Truck | 3,9% |
| Kịch bản tai nạn có | red_light_crash, rear_end, pedestrian_hit, hit_and_run, sudden_stop |

**Schema CSV trajectory:**
```
t, episode_id, actor_id, class, x, y, vx, vy, speed,
heading, road_id, lane_id, junction_id, lat, lon, phase
```

**Schema CSV goals:**
```
episode_id, actor_id, class, t_enter, t_exit, junction_id,
entry_road_id, entry_lane_id, exit_road_id, exit_lane_id, turn_angle_deg
```

## 4.4 Module phát hiện đối tượng — YOLO11s

YOLO11s được sử dụng với:
- **Confidence threshold**: 0,25 (low để ByteTrack xử lý low-score detections)
- **IoU threshold**: 0,45 (NMS)
- **Input resolution**: 640×640 (checkpoint tổng quát) / 960×960 (checkpoint chuyên CARLA)
- **FP16 inference**: Bật trên GTX 1050Ti để giảm VRAM
- **Classes**: car, motorcycle, bus, truck, person

Dữ liệu tự gán nhãn: 15.957 ảnh từ camera CARLA, auto-labeled bằng ITD model (confidence filter ≥ 0,4).

Hệ thống dùng **2 checkpoint riêng biệt** cho 2 mục đích khác nhau (xem 5.4 và [docs/gpu_cloud_tasks.md](docs/gpu_cloud_tasks.md)):
- `weights/yolo11s_vn.pt` — fine-tune trên dữ liệu CCTV Hà Nội thật, dùng cho production/real footage.
- `weights/yolo11s_carla_v2.pt` — fine-tune thêm trên `carla_cam_det_v2` (camera TL-mounted CARLA), dùng riêng khi đánh giá/test trên video CARLA simulation (đo MOTA). Chọn checkpoint qua flag `--detector` của `main.py`.

## 4.5 Module Tracking và ReID

**ByteTrack** hoạt động per-camera:
- High-score threshold: 0,6
- Low-score threshold: 0,1
- Max age (frames trước khi xóa track): 30
- Min hits (frames trước khi confirm track): 3

**OSNet ReID** cross-camera:
- Pretrained: VeRi-776 (vehicle ReID)
- Gallery: lưu 30 embeddings gần nhất mỗi track
- Matching: cosine similarity với threshold 0,7
- Kết quả: Global ID thống nhất xuyên camera

## 4.6 Goal Direction Classifier

### 4.6.1 Thiết kế đặc trưng

**Bảng 4.4: 26 đặc trưng Goal Classifier**

| Nhóm | Đặc trưng | Mô tả |
|------|---------|-------|
| Hướng (5) | head_sin, head_cos | Sine/cosine heading hiện tại |
| | head_rate_mean, head_rate_max | Tốc độ thay đổi hướng |
| | heading_range | Biên độ dao động heading trong window |
| Vận tốc (4) | speed_mean, speed_std | Tốc độ trung bình, độ lệch |
| | speed_change_rate | Tốc độ thay đổi vận tốc |
| | decel_rate | Tỉ lệ deceleration trong 2s cuối |
| Vị trí (3) | dist_to_junction | Khoảng cách đến junction |
| | lateral_offset | Offset ngang so với centerline |
| | approach_angle | Góc tiếp cận junction |
| Cửa sổ con (14) | speed_1s, head_rate_1s, ... | Thống kê 1s, 2s, 4s sliding window |

### 4.6.2 Huấn luyện

- **Model**: GradientBoostingClassifier (sklearn)
- **Calibration**: CalibratedClassifierCV (Platt scaling, cv=5)
- **Data split**: 80% train, 20% test, stratified by class và loại xe
- **Hyperparameters**: n_estimators=200, max_depth=5, learning_rate=0.1, subsample=0.8

```bat
python custom_tracking_system/scripts/train/train_goal_classifier.py ^
    --data custom_tracking_system/data/trajectories_hanoi_v2 ^
    --out-eval custom_tracking_system/data/goal_classifier_eval_v3
```

**Weights output**: `custom_tracking_system/weights/goal_classifier.pkl`

## 4.7 Trajectory Predictor — Seq2Seq GRU

### 4.7.1 Kiến trúc

**Bảng 4.5: Kiến trúc mạng GRU**

| Thành phần | Chi tiết |
|-----------|---------|
| Encoder | GRU(input=19, hidden=96), T_OBS=8 steps (0,8s) |
| Decoder | GRUCell(input=96+8, hidden=96), unroll × 5 horizons × 3 modes |
| Mode prob head | Linear(96, 3) + Softmax |
| Intent head | Linear(96, 4) + Softmax (straight/turn_left/turn_right/stop) |
| Displacement head | Linear(96, 2) per step per mode |
| Loss | NLL (best mode) + CE (intent) + KL (mode diversity) |

### 4.7.2 Training

```python
# Prediction horizons (seconds)
PRED_HORIZONS = (0.5, 1.0, 1.5, 2.0, 3.0)
# Observation window
T_OBS = 8  # steps = 0.8s
# Multi-modal predictions
NUM_MODES = 3
# Neighbor features
NUM_NEIGHBORS = 3
```

Training script: `custom_tracking_system/train_trajectory_predictor.py`  
Dataset: trajectories_hanoi_v2 (training split)

## 4.8 Alert System và Incident Detector

**IncidentDetector** phát hiện:
- Xung đột vật lý (collision từ CARLA event)
- Dừng đột ngột (sudden deceleration > threshold)
- Chạy ngược chiều (wrong-way detection từ heading vs road direction)
- Bỏ trốn sau tai nạn (hit-and-run: vận tốc tăng sau collision)

**EvidencePackage** đóng gói bằng chứng:
- Video clip 5 giây trước + sau sự cố
- Track history của các phương tiện liên quan
- Metadata: time, location, incident_type, actor_ids

---

# CHƯƠNG 5: KIỂM THỬ VÀ ĐÁNH GIÁ

## 5.1 Bộ dữ liệu đánh giá

Goal Classifier được đánh giá trên tập test độc lập (20% dữ liệu trajectories_hanoi_v2, stratified split):

| Tập | Số mẫu goal events | Tỉ lệ |
|-----|---------------------|-------|
| Train | ~29.597 | 80% |
| Test | ~7.400 | 20% |
| **Tổng** | **36.997** | 100% |

## 5.2 Kết quả Goal Classifier

### 5.2.1 Accuracy theo horizon dự đoán

**Bảng 5.1: Kết quả Goal Classifier theo horizon**

| Horizon (s) | Khoảng cách (m) | Accuracy | F1 Weighted | Số mẫu |
|-------------|----------------|---------|------------|--------|
| 0,5 | ~3,5 | 61,6% | 58,8% | 35.785 |
| **1,0** | **~7,0** | **64,8%** | **62,4%** | **35.272** |
| **1,5** ✓ | **~10,5** | **70,3%** | **68,3%** | **35.218** |
| 2,0 | ~14,0 | 56,9% | 54,6% | 35.164 |
| 3,0 | ~21,0 | 44,4% | 42,2% | 35.000 |

*Horizon 1,5s là sweet spot: đủ sớm để cảnh báo (~10m trước ngã tư), đủ chính xác (70,3%).*  
*Horizon 2,0s kém hơn 1,5s do uncertainty tăng khi xe chưa bộc lộ intention rõ.*

### 5.2.2 Kết quả theo loại phương tiện

**Bảng 5.2: Goal Classifier accuracy theo loại phương tiện (tại 1,5s)**

| Loại xe | Số mẫu | Accuracy | F1 Weighted |
|---------|--------|---------|------------|
| Motorcycle | 25.558 | 70,2% | 67,8% |
| Car | 7.081 | 68,7% | 66,8% |
| Truck | 1.437 | 73,1% | 71,9% |
| Bus | 1.142 | 78,2% | 75,7% |

*Xe máy đạt 70,2% — tương đương ô tô, cho thấy mô hình hoạt động tốt trên thành phần xe đặc trưng Hà Nội.*

### 5.2.3 Recall từng nhãn

**Bảng 5.3: Recall từng hướng đi (tại horizon 1,5s)**

| Nhãn | Recall | Đánh giá |
|------|--------|---------|
| straight (đi thẳng) | 75% | Tốt |
| right (rẽ phải) | 86% | Rất tốt |
| left (rẽ trái) | 53% | Cần cải thiện |
| u_turn (quay đầu) | 17% | Kém — cần detect trong junction |

**Phân tích nguyên nhân:**
- **u_turn recall 17%**: Xe quay đầu chỉ lộ signature rõ khi đã vào sâu trong ngã tư, không thể detect từ approach trajectory 4s.
- **left recall 53%**: Tại Hà Nội, rẽ trái thường bị chặn bởi xe ngược chiều → xe máy hay "chuẩn bị" rẽ trái muộn hơn.

### 5.2.4 Calibration và inference

| Metric | Giá trị |
|--------|---------|
| ECE | 0,104 |
| Inference time (batch 30 vehicles, mean) | 3,97ms |
| Inference time (p99) | 5,74ms |

ECE = 0,104 có nghĩa confidence phân loại lệch trung bình ~10% so với tần suất thực — chấp nhận được cho dashboard, cần cải thiện nếu dùng để trigger cảnh báo tự động.

## 5.3 Test Cases hệ thống

**Bảng 5.4: Test cases kiểm thử hệ thống**

| TC | Module | Đầu vào | Kết quả mong đợi | Kết quả thực tế |
|----|--------|---------|-----------------|----------------|
| TC01 | Goal Classifier | Trajectory xe đi thẳng 4s | P_straight > 0.6 | Đạt (75% recall) |
| TC02 | Goal Classifier | Trajectory rẽ phải | P_right > 0.6 | Đạt (86% recall) |
| TC03 | Goal Classifier | Trajectory quay đầu | P_uturn cao | Không đạt (17% recall) |
| TC04 | Inference speed | 30 xe đồng thời | < 10ms | Đạt (3.97ms mean) |
| TC05 | Calibration | Test set 7.400 mẫu | ECE < 0.15 | Đạt (ECE=0.104) |
| TC06 | Data pipeline | 120 episodes CARLA | 36.997 goal events | Đạt |
| TC07 | ByteTrack | Occlusion 10 frames | ID giữ nguyên | Đạt với occlusion ≤ 8 frames |
| TC08 | OSNet ReID | Phương tiện qua 2 camera | Same global ID | Đạt cosine > 0.7 |
| TC09 | Map load | hanoi_district3.xodr | CARLA load thành công | Đạt |
| TC10 | Incident Detector | Red light crash scenario | Alert trong 1s | Đạt |

## 5.4 Thảo luận kết quả

**Điểm mạnh:**
- Goal Classifier đạt 70,3% accuracy — đủ cho ứng dụng dashboard/thống kê lưu lượng theo hướng.
- Motorcycle accuracy 70,2% — hệ thống hoạt động đúng trên thành phần phương tiện đặc thù Hà Nội.
- Inference 4ms cho 30 phương tiện đồng thời — phù hợp yêu cầu CCTV thực tế (≥25 FPS pipeline).
- Calibration ECE=0,104 — confidence output đáng tin cậy hơn mức random (ECE=0 là perfect).

**Giới hạn:**
- **U_turn recall 17%**: Hướng đi quan trọng nhưng không thể detect từ approach trajectory. Giải pháp: dự đoán trong junction (sau khi xe đã vào).
- **Left recall 53%**: Cần thêm road topology features (entry_lane_id, junction connectivity) để phân biệt.
- **YOLO11s fine-tune trên CARLA gây catastrophic forgetting nặng**: Fine-tune `yolo11s_carla_v2.pt` trên `carla_cam_det_v2` (30 epoch, backbone không freeze, 1 domain hẹp, chỉ 3/5 class) đạt mAP50=0,805 trên domain CARLA TL-mounted, nhưng đo lại trên val set đa domain (`visdrone_carla_vn`) thì mAP50 tổng rơi từ 0,507 (baseline) xuống 0,006 — mất khả năng nhận diện gần như hoàn toàn ở domain khác, kể cả với car/motorcycle/truck (chi tiết: `docs/assets/yolo11s_carla_v2/forgetting_check_vs_visdrone_carla_vn.txt`). Giải pháp tạm: giữ 2 checkpoint riêng (mục 4.4), không dùng `yolo11s_carla_v2.pt` cho production.
- **ReID chưa có benchmark**: OSNet matching rate chưa được đo trên test set cross-camera riêng.
- **Dữ liệu mô phỏng**: Có domain gap với CCTV thực tế — texture, góc nhìn, điều kiện ánh sáng khác.

---

# CHƯƠNG 6: KẾT LUẬN VÀ HƯỚNG PHÁT TRIỂN

## 6.1 Kết luận

Đề tài đã hoàn thành các mục tiêu chính:

1. **Pipeline bản đồ**: Thành công xây dựng bản đồ Hà Nội (hanoi_district3) từ OSM sang OpenDRIVE và nạp vào CARLA, tái tạo môi trường giao thông đô thị với tỉ lệ xe máy 70%.

2. **Thu thập dữ liệu**: Thu thập 4.743.151 điểm trajectory, 36.997 goal events trong 120 episode mô phỏng, bao gồm 5 loại kịch bản tai nạn.

3. **Goal Classifier**: Đạt 70,3% accuracy tại horizon 1,5 giây trước ngã tư, ECE=0,104, inference 3,97ms cho 30 phương tiện. Phù hợp ứng dụng dashboard thống kê lưu lượng.

4. **Trajectory Predictor**: Xây dựng mạng Seq2Seq GRU đa modal (3 modes, 5 horizons 0,5–3,0s) với auxiliary intent classification.

5. **Pipeline hoàn chỉnh**: Tích hợp Detection → ByteTrack → OSNet ReID → Goal Classifier → GRU Predictor → Alert System thành một hệ thống coherent, chạy trên phần cứng GTX 1050Ti 4GB.

## 6.2 Hướng phát triển

Theo thứ tự ưu tiên:

1. **Cải thiện u_turn recall**: Dự đoán trong junction (in-junction traversal prediction) thay vì chỉ từ approach — kỳ vọng u_turn recall ~70%.

2. **YOLO11s unified fine-tune chống forgetting**: Train lại một checkpoint duy nhất dùng được cho cả domain CARLA và domain thật — trộn `carla_cam_det_v2` với dữ liệu đa domain (`visdrone_carla_vn`), giảm learning rate/epoch, freeze thêm backbone — để tránh phải duy trì 2 checkpoint riêng như hiện tại.

3. **ReID benchmark**: Xây dựng test set cross-camera để đo Rank-1 accuracy.

4. **LSTM/Transformer Goal Classifier**: Thay GradientBoosting bằng mô hình sequence-aware để tận dụng temporal pattern — kỳ vọng +5–8% accuracy.

5. **Triển khai CCTV thực**: Thu thập dữ liệu từ CCTV thực tế Hà Nội, đánh giá domain gap và fine-tune.

---

# TÀI LIỆU THAM KHẢO

[1] Bewley A. et al., "Simple Online and Realtime Tracking," ICIP 2016.

[2] Du Y. et al., "StrongSORT: Make DeepSORT Great Again," IEEE TITS 2023.

[3] Zhang Y. et al., "ByteTrack: Multi-Object Tracking by Associating Every Detection Box," ECCV 2022.

[4] Ma Y. et al., "TrafficPredict: Trajectory Prediction for Heterogeneous Traffic-Agents," AAAI 2019.

[5] Tang Z. et al., "CityFlow: A City-Scale Benchmark for Multi-Target Multi-Camera Vehicle Tracking," CVPR 2019.

[6] Zhou K. et al., "Omni-Scale Feature Learning for Person Re-Identification," ICCV 2019.

[7] Dosovitskiy A. et al., "CARLA: An Open Urban Driving Simulator," CoRL 2017.

[8] Wang C. et al., "YOLOv7: Trainable bag-of-freebies sets new state-of-the-art," CVPR 2023.

[9] Jocher G. et al., "Ultralytics YOLO11," https://github.com/ultralytics/ultralytics, 2024.

[10] OpenStreetMap contributors, "OpenStreetMap," https://www.openstreetmap.org, 2024.

[11] ASAM e.V., "OpenDRIVE Format Specification v1.7," 2021.

[12] Nair V. et al., "Rectified Linear Units Improve Restricted Boltzmann Machines," ICML 2010.

[13] Chung J. et al., "Empirical Evaluation of Gated Recurrent Neural Networks on Sequence Modeling," NeurIPS Workshop 2014.

[14] Guo C. et al., "On Calibration of Modern Neural Networks," ICML 2017.

[15] Guimaraes C. et al., "Traffic Forecasting using Vehicle Trajectory Data," IEEE ITS 2021.

---

# PHỤ LỤC

## Phụ lục A: Hướng dẫn cài đặt và chạy

### A.1 Yêu cầu phần cứng tối thiểu
- GPU NVIDIA với CUDA ≥ 11.0, VRAM ≥ 4GB
- RAM ≥ 16GB
- Storage ≥ 50GB (CARLA + dữ liệu)

### A.2 Cài đặt CARLA
```bat
cd WindowsNoEditor
CarlaUE4.exe -quality-level=Low
```

### A.3 Cài đặt dependencies
```bat
cd custom_tracking_system
pip install -r requirements.txt
```

### A.4 Thu thập dữ liệu
```bat
python collect_trajectory_data.py ^
  --map hanoi_district3 ^
  --episodes 120 --steps-per-episode 1800 ^
  --num-vehicles 20 --moto-ratio 0.70 ^
  --scenario-ratio 0.25 ^
  --out data/trajectories_hanoi_v2
```

### A.5 Train Goal Classifier
```bat
python scripts/train/train_goal_classifier.py ^
  --data data/trajectories_hanoi_v2 ^
  --out-eval data/goal_classifier_eval_v3
```

### A.6 Chạy hệ thống tracking
```bat
python main.py --config config/camera_config.yaml --source carla
```

### A.7 Đánh giá tracking
```bat
python evaluate_tracking.py --pred-dir output/eval/ --gt-dir output/gt/
```

## Phụ lục B: Cấu trúc thư mục project

```
finalproject/
├── custom_tracking_system/
│   ├── modules/              # Core AI modules
│   │   ├── detector.py       # YOLO11s wrapper
│   │   ├── tracker.py        # ByteTrack
│   │   ├── reid.py           # OSNet ReID
│   │   ├── global_tracking.py
│   │   ├── trajectory_gru.py # Seq2Seq GRU model definition
│   │   ├── trajectory_predictor.py # Ensemble predictor
│   │   ├── alert_system.py
│   │   ├── incident_detector.py
│   │   └── evidence_package.py
│   ├── scripts/train/        # Training scripts
│   │   ├── train_goal_classifier.py
│   │   ├── train_yolo_detector.py
│   │   └── ...
│   ├── data/
│   │   ├── trajectories_hanoi_v2/ # 120 episodes, 36997 goals
│   │   ├── goal_classifier_eval_v3/
│   │   └── auto_label/        # 15957 YOLO training images
│   ├── weights/               # Trained model weights
│   │   └── goal_classifier.pkl
│   ├── config/
│   │   └── camera_config.yaml
│   ├── main.py
│   └── collect_trajectory_data.py
├── Map/
│   └── hanoi_district3.xodr   # Hanoi OpenDRIVE map
├── deep-person-reid/          # OSNet library
├── WindowsNoEditor/           # CARLA binaries
└── things_to_do.md            # Task tracker
```

## Phụ lục C: Phân phối kịch bản tai nạn trong dataset

| Kịch bản | Số episodes | Post-accident rows |
|----------|------------|-------------------|
| none (bình thường) | ~90 | 0 |
| red_light_crash | 7 | ~0* |
| rear_end | 6 | ~20 |
| pedestrian_hit | 10 | ~13 |
| hit_and_run | 4 | 1.350–2.040 |
| sudden_stop | 6 | 33–3.148 |

*Post-accident rows = 0 nếu xe không đủ speed/impact để kích hoạt phase.

## Phụ lục D: So sánh phiên bản Goal Classifier

| Version | Features | Horizon | Accuracy |
|---------|---------|---------|---------|
| v1 | 12 features, window 2s | 1.5s | ~58% |
| v2 | 18 features, window 3s | 1.5s | ~64% |
| **v3** | **26 features, window 4s** | **1.5s** | **70,3%** |

Cải tiến từ v2 → v3: thêm sliding window statistics (1s, 2s sub-windows), `decel_rate`, `heading_range`.
