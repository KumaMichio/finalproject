# Workflow Triển Khai Dự Án Tracking System trên CARLA 0.9.9.4

## Giới Thiệu
Dự án này xây dựng một hệ thống theo dõi đối tượng (vehicle, pedestrian) đa camera trong môi trường mô phỏng CARLA 0.9.9.4. Hệ thống bao gồm phát hiện, theo dõi, nhận diện lại, dự đoán quỹ đạo và cảnh báo. Workflow dưới đây cung cấp hướng dẫn triển khai từng giai đoạn, từ thiết lập cơ bản đến tối ưu hóa.

## 1. Định Nghĩa Bài Toán
### Input
- Nhiều camera CCTV đặt trong môi trường CARLA.
- Stream video đồng thời (multi-view).

### Output
- Theo dõi object (vehicle, pedestrian) xuyên camera.
- Gán ID toàn cục (global identity).
- Dự đoán quỹ đạo di chuyển trong tương lai.
- Cảnh báo cho các camera hoặc khu vực phía trước.

## 2. Kiến Trúc Hệ Thống
Hệ thống gồm 5 lớp chính:
1. **Data Layer**: CARLA simulation (map, actor, camera), Multi-camera synchronized stream.
2. **Perception Layer**: Object Detection, Multi-object Tracking (per-camera).
3. **Cross-camera Fusion Layer**: Re-identification (ReID), Global ID assignment.
4. **Behavior Modeling Layer**: Trajectory prediction, Motion modeling.
5. **Application Layer**: Alert system, Visualization / monitoring.

## 3. Thứ Tự Triển Khai (Các Bước Chính)
Dự án được triển khai theo 6 giai đoạn để đảm bảo tính ổn định và dễ debug.

### Giai Đoạn 1: Thiết Lập Cơ Bản (Single Camera)
**Mục tiêu**: Xây dựng hệ thống cho một camera đơn.
**Các Bước**:
1. Cài đặt CARLA 0.9.9.4 và PythonAPI.
2. Tạo script đặt một camera trong CARLA (sử dụng `camera_controller.py`).
3. Triển khai Object Detection (YOLOv5/YOLOv8 hoặc Faster R-CNN).
   - Train/fine-tune trên dataset như COCO hoặc BDD100K.
   - Output: Bounding box + confidence + class (car, truck, bus, pedestrian).
4. Triển khai Multi-Object Tracking (ByteTrack hoặc DeepSORT).
   - Input: Bounding box từ detection.
   - Output: Track ID (local), Trajectory trong camera.
5. Test trên một camera: Phát hiện và theo dõi object trong video stream.
**Thời gian ước tính**: 2-3 tuần.
**Metric**: mAP cho detection, MOTA/IDF1 cho tracking.

### Giai Đoạn 2: Multi-Camera (Đồng Bộ Hệ Thống)
**Mục tiêu**: Mở rộng sang nhiều camera, đảm bảo đồng bộ.
**Các Bước**:
1. Đặt nhiều camera tại các tuyến đường trong CARLA (sử dụng metadata: camera ID, vị trí, hướng nhìn).
2. Đồng bộ frame (synchronous mode) để tránh lệch thời gian.
3. Chạy detection và tracking cho từng camera độc lập.
4. Thu thập dữ liệu multi-stream video + timestamp.
5. Test đồng bộ: Đảm bảo timestamp chính xác giữa các camera.
**Thời gian ước tính**: 1-2 tuần.
**Thách thức**: Đồng bộ thời gian – sử dụng CARLA's synchronous mode.

### Giai Đoạn 3: ReID và Global Tracking
**Mục tiêu**: Nhận diện lại object giữa các camera.
**Các Bước**:
1. Triển khai Re-identification (OSNet, ResNet50 ReID, hoặc FastReID).
   - Input: Crop ảnh object từ tracking.
   - Output: Vector đặc trưng (embedding).
2. Matching: Cosine similarity với threshold.
3. Global Tracking: Hợp nhất ID từ nhiều camera.
   - Khi object xuất hiện: So sánh embedding với database.
   - Nếu match → gán Global ID; nếu không → tạo ID mới.
   - Output: Global ID, Lịch sử di chuyển đa camera.
4. Train ReID model trên dataset như Market-1501 hoặc DukeMTMC-reID.
5. Tạo synthetic dataset từ CARLA cho ReID (cross-view).
**Thời gian ước tính**: 3-4 tuần.
**Metric**: Rank-1 accuracy, mAP cho ReID.

### Giai Đoạn 4: Trajectory Prediction
**Mục tiêu**: Dự đoán vị trí tương lai.
**Các Bước**:
1. Thu thập chuỗi vị trí theo thời gian từ global tracking.
2. Triển khai model cơ bản: Linear motion.
3. Nâng cao: LSTM/GRU, Social LSTM, hoặc Transformer-based.
4. (Tùy chọn) Map-aware prediction sử dụng topology từ CARLA.
5. Train trên dataset như ETH/UCY, Argoverse, hoặc Waymo.
6. Tạo synthetic trajectory dataset từ CARLA.
**Thời gian ước tính**: 2-3 tuần.
**Metric**: ADE, FDE.

### Giai Đoạn 5: Alert System và Visualization
**Mục tiêu**: Cảnh báo và hiển thị.
**Các Bước**:
1. Alert System: So sánh trajectory dự đoán với vị trí camera khác hoặc ROI.
   - Output: Danh sách camera bị ảnh hưởng, thời gian dự đoán.
2. Visualization: Hiển thị bounding box + ID, trajectory (past + predicted), camera switching, alert notification.
3. Tích hợp vào Application Layer.
**Thời gian ước tính**: 1-2 tuần.

### Giai Đoạn 6: Tối Ưu và Scaling
**Mục tiêu**: Cải thiện hiệu suất và mở rộng.
**Các Bước**:
1. Tối ưu model (quantization, pruning).
2. Xử lý thách thức: ReID sai (calibration), occlusion, domain gap.
3. Scaling: Thêm nhiều camera, test trên map lớn hơn.
4. End-to-end testing.
**Thời gian ước tính**: 2-4 tuần.

## 4. Dataset và Dữ Liệu
- **Detection**: COCO, BDD100K.
- **Tracking**: MOT17, MOT20.
- **ReID**: Market-1501, DukeMTMC-reID.
- **Trajectory**: ETH/UCY, Argoverse, Waymo.
- **Synthetic**: Tự tạo từ CARLA (multi-camera tracking, ReID cross-view, trajectory).

## 5. Pipeline Dữ Liệu
1. Camera → Frame.
2. Detection → Bounding box.
3. Tracking → Local ID.
4. Crop object → Feature embedding.
5. ReID → Global ID.
6. Lưu trajectory.
7. Prediction → Future path.
8. Alert → Output.

## 6. Metric Đánh Giá
- **Detection**: mAP.
- **Tracking**: MOTA, IDF1.
- **ReID**: Rank-1 accuracy, mAP.
- **Trajectory**: ADE, FDE.

## 7. Thách Thức Chính và Giải Pháp
1. **ReID sai**: Camera góc khác → Khó match. Giải pháp: Multi-camera calibration, domain adaptation.
2. **Đồng bộ thời gian**: Lệch frame → Sai trajectory. Giải pháp: Synchronous mode trong CARLA.
3. **Occlusion**: Object bị che. Giải pháp: Robust tracking algorithms.
4. **Domain gap**: Model train trên dataset thật nhưng chạy trên CARLA. Giải pháp: Fine-tune trên synthetic data từ CARLA.

## 8. Hướng Nâng Cao (Research)
- Multi-camera calibration (mapping giữa camera).
- Graph-based tracking.
- Multi-agent trajectory prediction.
- End-to-end learning (joint detection + tracking + ReID).

## Kết Luận
Workflow này đảm bảo triển khai có hệ thống, từ cơ bản đến nâng cao. Bắt đầu từ single camera để validate từng module, sau đó mở rộng. Sử dụng CARLA để tạo synthetic data giúp giảm domain gap. Nếu gặp vấn đề, ưu tiên debug từng giai đoạn.