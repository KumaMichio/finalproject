# Multi-Camera CCTV Tracking System for CARLA

Hệ thống tracking đa camera sử dụng AI để theo dõi và nhận diện lại objects (vehicles, pedestrians) trong môi trường CARLA UE4.

## Tính năng chính

- **Multi-Camera Detection**: Phát hiện objects qua nhiều camera đồng bộ
- **Cross-Camera Re-Identification**: Nhận diện lại objects khi chuyển giữa camera
- **Global ID Assignment**: Gán ID toàn cục duy nhất cho mỗi object
- **Trajectory Prediction**: Dự đoán quỹ đạo di chuyển tương lai
- **Alert System**: Cảnh báo khi object vào vùng quan tâm (ROI)
- **Real-time Visualization**: Hiển thị tracking results

## Cấu trúc dự án

```
custom_tracking_system/
├── config/
│   └── camera_config.yaml      # Cấu hình camera và ROI
├── modules/
│   ├── camera_controller.py    # Quản lý camera sensors
│   ├── traffic_generator.py    # Sinh traffic cho testing
│   ├── detector.py            # Object detection (YOLOv5)
│   ├── tracker.py             # Single-camera tracking
│   ├── reid.py                # Cross-camera ReID
│   ├── global_tracking.py     # Global ID management
│   ├── trajectory_predictor.py # Trajectory prediction
│   └── alert_system.py        # Alert generation
├── utils/
│   ├── visualization.py       # Display utilities
│   ├── metrics.py            # Performance metrics
│   └── data_writer.py        # Data export
├── models/                    # Pre-trained models
├── datasets/                  # Training data
├── main.py                    # Main entry point
├── requirements.txt           # Dependencies
└── __init__.py               # Package init
```

## Cài đặt

1. **Cài đặt dependencies:**
```bash
cd PythonAPI/custom_tracking_system
pip install -r requirements.txt
```

2. **Khởi động CARLA Server:**
```bash
cd WindowsNoEditor
CarlaUE4.exe
```

## Sử dụng

### Chạy hệ thống tracking:

```bash
python main.py --config config/camera_config.yaml --max-frames 1000
```

### Các tùy chọn command line:

- `--config`: Đường dẫn đến file config (default: config/camera_config.yaml)
- `--max-frames`: Số frame tối đa để xử lý (default: None - chạy vô hạn)
- `--log-level`: Mức độ logging (DEBUG, INFO, WARNING, ERROR)

### Ví dụ:

```bash
# Chạy với debug mode
python main.py --log-level DEBUG

# Chạy 500 frames
python main.py --max-frames 500

# Sử dụng config khác
python main.py --config config/my_config.yaml
```

## Cấu hình

### Camera Configuration (camera_config.yaml)

```yaml
cameras:
  camera_0:
    position: [0, 0, 3]      # Vị trí [x, y, z]
    rotation: [0, 0, 0]      # Góc quay [pitch, yaw, roll]
    camera_id: "CAM_001"     # ID camera
    view_angle: 90           # FOV
    resolution: [1920, 1080] # Độ phân giải
    fps: 10                  # Frame rate

rois:
  camera_0:
    zones:
      - name: "intersection_main"
        polygon: [[100, 100], [500, 100], [500, 400], [100, 400]]
```

## Kiến trúc hệ thống

1. **Data Layer**: CARLA simulation với multi-camera streams
2. **Perception Layer**: Object detection và single-camera tracking
3. **Cross-camera Fusion**: ReID và global ID assignment
4. **Behavior Modeling**: Trajectory prediction
5. **Application Layer**: Alert system và visualization

## Goal Direction Classifier

Dự đoán hướng đi của phương tiện tại ngã tư (straight / left / right / u_turn)
từ trajectory approach, thiết kế cho CCTV đời thực Hà Nội.

### Thu thập dữ liệu

```bat
python custom_tracking_system/collect_trajectory_data.py ^
    --map hanoi_district3 --episodes 120 --steps-per-episode 1800 ^
    --num-vehicles 20 --moto-ratio 0.70 ^
    --out custom_tracking_system/data/trajectories_hanoi_v2
```

Tham số `--moto-ratio` kiểm soát tỉ lệ xe máy trong simulation (default 0.70 = thực tế Hà Nội).  
Data hiện có: `data/trajectories_hanoi_v2/` — 36,997 goal events, motorcycle 72.7%.

### Train classifier

```bat
python custom_tracking_system/scripts/train/train_goal_classifier.py ^
    --data custom_tracking_system/data/trajectories_hanoi_v2
```

Model output: `weights/goal_classifier.pkl` — GradientBoosting, calibrated, 26 features, window 4s.

### Kết quả (v3, data Hà Nội thực tế)

| Metric | Giá trị |
|---|---|
| Accuracy @1.5s trước ngã tư | 70.3% |
| Accuracy @1.0s (~7m) | 64.8% |
| ECE (calibration) | 0.104 |
| Inference 30 xe đồng thời | 4.0ms |
| Motorcycle accuracy | 70.2% |

Recall theo lớp: straight 75%, right 86%, left 53%, u_turn 17%.  
U_turn khó detect từ approach trajectory (xe chỉ lộ signature khi đã vào ngã tư).  
Phù hợp cho dashboard/visualization, thống kê lưu lượng. Chưa đủ cho trigger cảnh báo tự động.

---

## Metrics & Evaluation

Hệ thống tự động thu thập các metrics:

- **Detection**: mAP, precision, recall
- **Tracking**: MOTA, IDF1, track accuracy
- **ReID**: Rank-1 accuracy, mAP
- **Goal Classifier**: Accuracy theo horizon, ECE, per-class recall
- **Performance**: FPS, latency

## Output Files

Hệ thống tạo ra các file output trong thư mục `output/`:

- `trajectories/`: Trajectories của từng object
- `snapshots/`: Snapshots của tracking state
- `logs/`: Alert logs và system logs
- `summary_report.txt`: Báo cáo tổng kết

## Development

### Thêm camera mới:

1. Thêm config vào `camera_config.yaml`
2. Hệ thống tự động detect và setup camera mới

### Tùy chỉnh detection model:

```python
# Trong detector.py
self.model = torch.hub.load('ultralytics/yolov5', 'yolov5x')  # Larger model
```

### Tùy chỉnh ReID threshold:

```python
# Trong main.py hoặc config
match_threshold = 0.7  # Higher = more strict matching
```

## Troubleshooting

### Lỗi kết nối CARLA:
- Đảm bảo CARLA server đang chạy trên port 2000
- Kiểm tra firewall settings

### Detection không hoạt động:
- Kiểm tra CUDA availability: `torch.cuda.is_available()`
- Thử CPU mode nếu GPU không khả dụng

### ReID matching kém:
- Tăng gallery size trong ReIDExtractor
- Điều chỉnh similarity threshold
- Thu thập thêm training data

## License

CARLA Tracking System - Internal Use Only

## Contact

For issues and questions, check the logs in `tracking_system.log`