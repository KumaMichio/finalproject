# Implementation Plan: Multi-Camera CCTV Tracking System in CARLA

## Overview
Xây dựng hệ thống tracking đa camera với AI detection, re-identification, và trajectory prediction trên CARLA UE4.

---

## Phase 1: Setup Environment & Infrastructure

### 1.1 Chuẩn bị thư mục dự án
```
CARLA_0.9.9.4/
├── PythonAPI/
│   ├── carla/
│   └── custom_tracking_system/  (TẠO MỚI)
│       ├── config/
│       │   └── camera_config.yaml
│       ├── modules/
│       │   ├── camera_controller.py
│       │   ├── detector.py
│       │   ├── tracker.py
│       │   ├── reid.py
│       │   ├── global_tracking.py
│       │   ├── trajectory_predictor.py
│       │   └── alert_system.py
│       ├── utils/
│       │   ├── data_writer.py
│       │   ├── visualization.py
│       │   └── metrics.py
│       ├── datasets/
│       │   ├── synthetic_data/
│       │   └── ground_truth/
│       ├── models/
│       │   ├── detection/
│       │   ├── tracking/
│       │   └── reid/
│       ├── main.py
│       └── requirements.txt
```

### 1.2 Dependencies
```txt
# requirements.txt
carla==0.9.9
torch==1.10.0
torchvision==0.11.0
opencv-python==4.5.4
numpy==1.21.0
scipy==1.7.2
scikit-learn==1.0.1
pytorch-lightning==1.5.0
yolov5==6.1.0
pycocotools==2.0.2
```

### 1.3 Kiểm tra CARLA Server
- Chạy CARLA: `cd WindowsNoEditor && CarlaUE4.exe`
- Kiểm tra Python API: `python -c "import carla; print(carla.__version__)"`

---

## Phase 2: Camera Setup & Data Acquisition

### 2.1 Thiết kế layout camera trên map
**Tệp: `config/camera_config.yaml`**

```yaml
cameras:
  camera_0:
    position: [0, 0, 3]  # [x, y, z] - world coordinates
    rotation: [0, 0, 0]  # [pitch, yaw, roll]
    camera_id: "CAM_001"
    view_angle: 90
    resolution: [1920, 1080]
    fps: 10
    
  camera_1:
    position: [50, 0, 3]
    rotation: [0, 180, 0]
    camera_id: "CAM_002"
    view_angle: 120
    resolution: [1920, 1080]
    fps: 10
    
  camera_2:
    position: [25, -30, 3]
    rotation: [0, 45, 0]
    camera_id: "CAM_003"
    view_angle: 100
    resolution: [1920, 1080]
    fps: 10

# Túi ROI (Region of Interest) cho mỗi camera
rois:
  camera_0:
    zones:
      - name: "intersection_main"
        polygon: [[100, 100], [500, 100], [500, 400], [100, 400]]
  camera_1:
    zones:
      - name: "side_road"
        polygon: [[0, 0], [300, 0], [300, 200], [0, 200]]
```

### 2.2 Camera Controller Module
**Tệp: `modules/camera_controller.py`**

```python
import carla
import numpy as np
from collections import deque
import yaml

class CameraController:
    def __init__(self, client, world, config_path):
        self.client = client
        self.world = world
        self.cameras = {}
        self.sensor_data = {}
        self.load_config(config_path)
        
    def load_config(self, config_path):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
    
    def setup_cameras(self):
        """Tạo camera sensors tại các vị trí đã define"""
        blueprint_library = self.world.get_blueprint_library()
        camera_bp = blueprint_library.find('sensor.camera.rgb')
        
        for cam_name, cam_cfg in self.config['cameras'].items():
            camera_bp.set_attribute('image_size_x', str(cam_cfg['resolution'][0]))
            camera_bp.set_attribute('image_size_y', str(cam_cfg['resolution'][1]))
            camera_bp.set_attribute('fov', str(cam_cfg['view_angle']))
            
            location = carla.Location(*cam_cfg['position'])
            rotation = carla.Rotation(*cam_cfg['rotation'])
            transform = carla.Transform(location, rotation)
            
            camera = self.world.spawn_actor(camera_bp, transform)
            camera_id = cam_cfg['camera_id']
            
            # Gán callback để lưu data
            camera.listen(lambda img, cam_id=camera_id: 
                         self._store_image(img, cam_id))
            
            self.cameras[camera_id] = camera
            self.sensor_data[camera_id] = deque(maxlen=100)
    
    def _store_image(self, image, camera_id):
        """Lưu image vào buffer"""
        data = {
            'timestamp': image.timestamp,
            'frame': np.array(image.raw_data).reshape(
                image.height, image.width, 4)[:, :, :3],  # RGB
            'camera_id': camera_id
        }
        self.sensor_data[camera_id].append(data)
    
    def get_synchronized_frames(self):
        """Lấy frames đã đồng bộ từ tất cả camera"""
        synchronized_data = {}
        for cam_id, data_buffer in self.sensor_data.items():
            if len(data_buffer) > 0:
                synchronized_data[cam_id] = data_buffer[-1]
        return synchronized_data
    
    def cleanup(self):
        """Xóa camera sensors"""
        for camera in self.cameras.values():
            camera.destroy()
```

### 2.3 Spawn NPC & Vehicles
**Tệp: `modules/traffic_generator.py`**

```python
import carla

class TrafficGenerator:
    def __init__(self, world, num_vehicles=10, num_pedestrians=5):
        self.world = world
        self.num_vehicles = num_vehicles
        self.num_pedestrians = num_pedestrians
        self.vehicle_list = []
        self.pedestrian_list = []
    
    def spawn_actors(self):
        """Spawn vehicles và pedestrians ngẫu nhiên"""
        blueprint_library = self.world.get_blueprint_library()
        spawn_points = self.world.get_map().get_spawn_points()
        
        # Spawn vehicles
        for i in range(self.num_vehicles):
            bp = blueprint_library.random()
            bp.set_attribute('color', '255,0,0')
            spawn_point = spawn_points[i % len(spawn_points)]
            vehicle = self.world.try_spawn_actor(bp, spawn_point)
            if vehicle:
                vehicle.set_autopilot(True)
                self.vehicle_list.append(vehicle)
        
        # Spawn pedestrians
        walker_bp = blueprint_library.find('walker.pedestrian.0001')
        for i in range(self.num_pedestrians):
            spawn_point = spawn_points[i % len(spawn_points)]
            pedestrian = self.world.try_spawn_actor(walker_bp, spawn_point)
            if pedestrian:
                self.pedestrian_list.append(pedestrian)
    
    def cleanup(self):
        for actor in self.vehicle_list + self.pedestrian_list:
            actor.destroy()
```

---

## Phase 3: Object Detection

### 3.1 YOLOv5 Integration
**Tệp: `modules/detector.py`**

```python
import torch
import cv2
import numpy as np

class ObjectDetector:
    def __init__(self, model_type='yolov5s'):
        # Pre-trained YOLOv5
        self.model = torch.hub.load('ultralytics/yolov5', model_type, pretrained=True)
        self.model.conf = 0.4  # Confidence threshold
        self.classes_of_interest = {0: 'person', 2: 'car', 5: 'bus', 7: 'truck'}
    
    def detect(self, frame):
        """
        Detect objects trong frame
        
        Args:
            frame: numpy array (H, W, 3)
        
        Returns:
            detections: list of {'box': [x1,y1,x2,y2], 'conf': float, 'class': str}
        """
        results = self.model(frame)
        detections = []
        
        for det in results.xyxy[0]:
            x1, y1, x2, y2, conf, cls = det
            class_id = int(cls)
            
            if class_id in self.classes_of_interest:
                detections.append({
                    'box': [int(x1), int(y1), int(x2), int(y2)],
                    'confidence': float(conf),
                    'class': self.classes_of_interest[class_id],
                    'class_id': class_id
                })
        
        return detections
    
    def visualize(self, frame, detections):
        """Draw bounding boxes"""
        frame_copy = frame.copy()
        for det in detections:
            x1, y1, x2, y2 = det['box']
            cv2.rectangle(frame_copy, (x1, y1), (x2, y2), (0, 255, 0), 2)
            label = f"{det['class']} ({det['confidence']:.2f})"
            cv2.putText(frame_copy, label, (x1, y1-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        return frame_copy
```

---

## Phase 4: Multi-Object Tracking (Single Camera)

### 4.1 ByteTrack Implementation
**Tệp: `modules/tracker.py`**

```python
import numpy as np
from scipy.spatial.distance import cdist
from collections import defaultdict

class SimpleTracker:
    def __init__(self, max_age=30, min_hits=3):
        self.max_age = max_age
        self.min_hits = min_hits
        self.tracks = {}
        self.next_id = 0
        self.frame_count = 0
    
    def update(self, detections):
        """
        Update tracks với detections mới
        
        Args:
            detections: list of {'box': [...], 'confidence': float, 'class': str}
        
        Returns:
            tracks: list of {'track_id': int, 'box': [...], 'class': str}
        """
        self.frame_count += 1
        
        # IoU-based matching (simplified)
        active_tracks = [t for t in self.tracks.values() if t['age'] < self.max_age]
        
        matched_detections = set()
        matched_tracks = set()
        
        # Matching step
        if len(active_tracks) > 0 and len(detections) > 0:
            iou_matrix = self._compute_iou_matrix(active_tracks, detections)
            
            for track_idx, det_idx in self._greedy_matching(iou_matrix):
                track = active_tracks[track_idx]
                det = detections[det_idx]
                
                # Update track
                track['box'] = det['box']
                track['age'] = 0
                track['hits'] += 1
                track['positions'].append(self._box_center(det['box']))
                
                matched_tracks.add(track_idx)
                matched_detections.add(det_idx)
        
        # Increment age cho unmatched tracks
        for track in active_tracks:
            if len([i for i, t in enumerate(active_tracks) 
                   if self.tracks[t['id']] == track]) not in matched_tracks:
                track['age'] += 1
        
        # Create new tracks từ unmatched detections
        for i, det in enumerate(detections):
            if i not in matched_detections:
                self.tracks[self.next_id] = {
                    'id': self.next_id,
                    'box': det['box'],
                    'class': det['class'],
                    'age': 0,
                    'hits': 1,
                    'positions': [self._box_center(det['box'])]
                }
                self.next_id += 1
        
        # Output: only tracks with hits >= min_hits
        output = []
        for track in self.tracks.values():
            if track['hits'] >= self.min_hits and track['age'] == 0:
                output.append({
                    'track_id': track['id'],
                    'box': track['box'],
                    'class': track['class'],
                    'positions': track['positions']
                })
        
        return output
    
    def _compute_iou_matrix(self, tracks, detections):
        """Tính IoU matrix giữa tracks và detections"""
        n_tracks = len(tracks)
        n_dets = len(detections)
        iou_matrix = np.zeros((n_tracks, n_dets))
        
        for i, track in enumerate(tracks):
            for j, det in enumerate(detections):
                iou_matrix[i, j] = self._iou(track['box'], det['box'])
        
        return iou_matrix
    
    def _iou(self, box1, box2):
        """Intersection over Union"""
        x1_min, y1_min, x1_max, y1_max = box1
        x2_min, y2_min, x2_max, y2_max = box2
        
        inter_x_min = max(x1_min, x2_min)
        inter_y_min = max(y1_min, y2_min)
        inter_x_max = min(x1_max, x2_max)
        inter_y_max = min(y1_max, y2_max)
        
        if inter_x_max < inter_x_min or inter_y_max < inter_y_min:
            return 0.0
        
        inter_area = (inter_x_max - inter_x_min) * (inter_y_max - inter_y_min)
        box1_area = (x1_max - x1_min) * (y1_max - y1_min)
        box2_area = (x2_max - x2_min) * (y2_max - y2_min)
        union_area = box1_area + box2_area - inter_area
        
        return inter_area / (union_area + 1e-6)
    
    def _greedy_matching(self, iou_matrix, threshold=0.3):
        """Greedy matching: gán detections tốt nhất"""
        matches = []
        used_dets = set()
        
        for i in range(iou_matrix.shape[0]):
            best_j = np.argmax(iou_matrix[i, :])
            if iou_matrix[i, best_j] > threshold and best_j not in used_dets:
                matches.append((i, best_j))
                used_dets.add(best_j)
        
        return matches
    
    def _box_center(self, box):
        """Tính tâm của bounding box"""
        x1, y1, x2, y2 = box
        return [(x1 + x2) / 2, (y1 + y2) / 2]
```

---

## Phase 5: Re-Identification (ReID) - Cross-Camera Matching

### 5.1 ReID Feature Extractor
**Tệp: `modules/reid.py`**

```python
import torch
import torchvision.models as models
import torch.nn.functional as F
from torchvision import transforms
import cv2
import numpy as np

class ReIDExtractor:
    def __init__(self, model_name='resnet50', pretrained=True):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        if model_name == 'resnet50':
            self.model = models.resnet50(pretrained=pretrained)
            self.model.fc = torch.nn.Identity()  # Remove classification layer
        
        self.model.to(self.device)
        self.model.eval()
        
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((256, 128)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])
        ])
        
        self.gallery = {}  # {global_id: [features]}
    
    def extract_feature(self, frame, box):
        """
        Trích xuất feature từ crop ảnh
        
        Args:
            frame: numpy array (H, W, 3)
            box: [x1, y1, x2, y2]
        
        Returns:
            feature: tensor (1, 2048)
        """
        x1, y1, x2, y2 = box
        crop = frame[y1:y2, x1:x2]
        
        if crop.size == 0:
            return None
        
        # Preprocess
        input_tensor = self.transform(crop).unsqueeze(0).to(self.device)
        
        # Extract feature
        with torch.no_grad():
            feature = self.model(input_tensor)
            feature = F.normalize(feature, p=2, dim=1)
        
        return feature.cpu().detach().numpy()
    
    def match_with_gallery(self, feature, threshold=0.5):
        """
        So sánh feature với gallery
        
        Returns:
            best_id: global ID nếu match, None nếu không
        """
        if len(self.gallery) == 0:
            return None
        
        best_id = None
        best_score = 0
        
        for global_id, features_list in self.gallery.items():
            # So sánh với tất cả features của object này
            for gallery_feat in features_list:
                score = float(np.dot(feature, gallery_feat.T))
                
                if score > best_score:
                    best_score = score
                    best_id = global_id
        
        if best_score > threshold:
            return best_id
        else:
            return None
    
    def add_to_gallery(self, global_id, feature):
        """Thêm feature vào gallery"""
        if global_id not in self.gallery:
            self.gallery[global_id] = []
        
        self.gallery[global_id].append(feature)
        
        # Keep only last 10 features per ID
        if len(self.gallery[global_id]) > 10:
            self.gallery[global_id].pop(0)
```

---

## Phase 6: Global Tracking (Multi-Camera)

### 6.1 Global ID Management
**Tệp: `modules/global_tracking.py`**

```python
import numpy as np
from collections import defaultdict
from datetime import datetime

class GlobalTracker:
    def __init__(self, reid_extractor):
        self.reid_extractor = reid_extractor
        self.global_tracks = {}  # {global_id: track_info}
        self.next_global_id = 1000
        
        self.match_threshold = 0.5
        self.temporal_consistency = defaultdict(lambda: [])
    
    def process_camera_tracks(self, camera_id, frame, local_tracks):
        """
        Xử lý local tracks từ 1 camera
        
        Args:
            camera_id: str (e.g., 'CAM_001')
            frame: numpy array
            local_tracks: list of {'track_id': int, 'box': [...], 'class': str}
        
        Returns:
            global_tracks: list of {'global_id': int, 'track_id': int, ...}
        """
        global_tracks_output = []
        
        for local_track in local_tracks:
            box = local_track['box']
            
            # Extract ReID feature
            feature = self.reid_extractor.extract_feature(frame, box)
            if feature is None:
                continue
            
            # Try to match với gallery
            matched_global_id = self.reid_extractor.match_with_gallery(
                feature, threshold=self.match_threshold)
            
            if matched_global_id is not None:
                global_id = matched_global_id
            else:
                # Create new global ID
                global_id = self.next_global_id
                self.next_global_id += 1
            
            # Update gallery
            self.reid_extractor.add_to_gallery(global_id, feature)
            
            # Update global track
            if global_id not in self.global_tracks:
                self.global_tracks[global_id] = {
                    'global_id': global_id,
                    'class': local_track['class'],
                    'first_seen': datetime.now(),
                    'last_seen': datetime.now(),
                    'camera_history': defaultdict(list)
                }
            
            self.global_tracks[global_id]['last_seen'] = datetime.now()
            self.global_tracks[global_id]['camera_history'][camera_id].append({
                'timestamp': datetime.now(),
                'box': box,
                'local_track_id': local_track['track_id']
            })
            
            global_tracks_output.append({
                'global_id': global_id,
                'camera_id': camera_id,
                'box': box,
                'class': local_track['class'],
                'local_track_id': local_track['track_id']
            })
        
        return global_tracks_output
    
    def get_active_tracks(self, timeout_seconds=30):
        """Lấy active tracks"""
        now = datetime.now()
        active_tracks = {}
        
        for global_id, track in self.global_tracks.items():
            elapsed = (now - track['last_seen']).total_seconds()
            
            if elapsed < timeout_seconds:
                active_tracks[global_id] = track
        
        return active_tracks
```

---

## Phase 7: Trajectory Prediction

### 7.1 Motion Prediction Module
**Tệp: `modules/trajectory_predictor.py`**

```python
import numpy as np
from collections import deque

class TrajectoryPredictor:
    def __init__(self, window_size=10, pred_steps=5):
        self.window_size = window_size  # Number of past positions
        self.pred_steps = pred_steps    # Number of future frames to predict
        self.trajectories = {}  # {global_id: deque of positions}
    
    def update_trajectory(self, global_id, position, frame_idx):
        """
        Update trajectory cho 1 object
        
        Args:
            global_id: int
            position: [x, y] (center of bounding box)
            frame_idx: int
        """
        if global_id not in self.trajectories:
            self.trajectories[global_id] = {
                'positions': deque(maxlen=self.window_size),
                'frames': deque(maxlen=self.window_size)
            }
        
        self.trajectories[global_id]['positions'].append(position)
        self.trajectories[global_id]['frames'].append(frame_idx)
    
    def predict(self, global_id):
        """
        Dự đoán vị trí tương lai
        
        Returns:
            predicted_positions: list of [x, y]
        """
        if global_id not in self.trajectories or \
           len(self.trajectories[global_id]['positions']) < 2:
            return None
        
        positions = np.array(list(self.trajectories[global_id]['positions']))
        frames = np.array(list(self.trajectories[global_id]['frames']))
        
        # Linear regression (đơn giản)
        predicted_positions = self._linear_prediction(positions, frames)
        
        return predicted_positions
    
    def _linear_prediction(self, positions, frames):
        """Linear motion model"""
        if len(positions) < 2:
            return None
        
        # Calculate velocity
        velocity = positions[-1] - positions[-2]
        
        # Predict future positions
        predicted = []
        for step in range(1, self.pred_steps + 1):
            next_pos = positions[-1] + velocity * step
            predicted.append(next_pos)
        
        return predicted
    
    def _lstm_prediction(self, positions):
        """
        LSTM-based prediction (nâng cao)
        Cần train LSTM model trước
        """
        # TODO: Implement LSTM model
        pass
```

---

## Phase 8: Alert System

### 8.1 Alert Module
**Tệp: `modules/alert_system.py`**

```python
from collections import defaultdict
from datetime import datetime, timedelta

class AlertSystem:
    def __init__(self, trajectory_predictor):
        self.trajectory_predictor = trajectory_predictor
        self.rois = {}  # Define ROIs từ config
        self.alert_history = defaultdict(list)
        self.alert_threshold = 5  # frames
    
    def set_rois(self, rois):
        """
        Set Regions of Interest
        
        Args:
            rois: {camera_id: [{'name': str, 'polygon': [...]}]}
        """
        self.rois = rois
    
    def check_alerts(self, global_id, camera_id, predicted_positions, current_box):
        """
        Check xem object sắp di chuyển đến camera hoặc ROI nào
        
        Returns:
            alerts: list of {'type': str, 'target_camera': str, 'eta': int}
        """
        alerts = []
        
        if predicted_positions is None:
            return alerts
        
        # Check which ROI object will enter
        if camera_id in self.rois:
            for roi in self.rois[camera_id]:
                if self._will_enter_roi(predicted_positions, roi['polygon']):
                    alerts.append({
                        'type': 'ROI_WARNING',
                        'global_id': global_id,
                        'camera_id': camera_id,
                        'roi_name': roi['name'],
                        'timestamp': datetime.now(),
                        'eta': self._estimate_arrival_time(predicted_positions)
                    })
        
        return alerts
    
    def _will_enter_roi(self, predicted_positions, polygon):
        """Check xem trajectory sắp enter ROI"""
        # Simple point-in-polygon check
        for pos in predicted_positions:
            if self._point_in_polygon(pos, polygon):
                return True
        return False
    
    def _point_in_polygon(self, point, polygon):
        """Ray casting algorithm"""
        x, y = point
        n = len(polygon)
        inside = False
        
        p1x, p1y = polygon[0]
        for i in range(1, n + 1):
            p2x, p2y = polygon[i % n]
            if y > min(p1y, p2y):
                if y <= max(p1y, p2y):
                    if x <= max(p1x, p2x):
                        if p1y != p2y:
                            xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                        if p1x == p2x or x <= xinters:
                            inside = not inside
            p1x, p1y = p2x, p2y
        
        return inside
    
    def _estimate_arrival_time(self, predicted_positions):
        """Estimate ETA (số frames)"""
        return len(predicted_positions)
    
    def log_alert(self, alert):
        """Log alert"""
        self.alert_history[alert['global_id']].append(alert)
        print(f"ALERT: {alert}")
```

---

## Phase 9: Main Pipeline

### 9.1 Main Processing Loop
**Tệp: `main.py`**

```python
import carla
import yaml
import cv2
from collections import defaultdict

from modules.camera_controller import CameraController
from modules.traffic_generator import TrafficGenerator
from modules.detector import ObjectDetector
from modules.tracker import SimpleTracker
from modules.reid import ReIDExtractor
from modules.global_tracking import GlobalTracker
from modules.trajectory_predictor import TrajectoryPredictor
from modules.alert_system import AlertSystem
from utils.visualization import Visualizer

def main():
    # 1. SETUP CARLA
    client = carla.Client('localhost', 2000)
    client.set_timeout(10.0)
    world = client.get_world()
    
    # Set synchronous mode
    settings = world.get_settings()
    settings.synchronous_mode = True
    settings.fixed_delta_seconds = 0.1
    world.apply_settings(settings)
    
    # 2. SETUP CAMERAS
    camera_controller = CameraController(
        client, world, 'config/camera_config.yaml')
    camera_controller.setup_cameras()
    
    # 3. SPAWN TRAFFIC
    traffic_gen = TrafficGenerator(world, num_vehicles=10, num_pedestrians=5)
    traffic_gen.spawn_actors()
    
    # 4. INITIALIZE MODULES
    detector = ObjectDetector(model_type='yolov5s')
    
    trackers = {}  # {camera_id: SimpleTracker}
    for cam_id in camera_controller.cameras.keys():
        trackers[cam_id] = SimpleTracker(max_age=30, min_hits=3)
    
    reid_extractor = ReIDExtractor(model_name='resnet50')
    global_tracker = GlobalTracker(reid_extractor)
    trajectory_predictor = TrajectoryPredictor(window_size=10, pred_steps=5)
    
    alert_system = AlertSystem(trajectory_predictor)
    alert_system.set_rois(camera_controller.config['rois'])
    
    visualizer = Visualizer()
    
    # 5. MAIN LOOP
    frame_count = 0
    try:
        while True:
            frame_count += 1
            
            # Get synchronized frames
            sync_frames = camera_controller.get_synchronized_frames()
            
            all_global_tracks = []
            
            # Process each camera
            for camera_id, frame_data in sync_frames.items():
                frame = frame_data['frame']
                
                # Detection
                detections = detector.detect(frame)
                
                # Single-camera tracking
                local_tracks = trackers[camera_id].update(detections)
                
                # Global tracking (cross-camera)
                global_tracks = global_tracker.process_camera_tracks(
                    camera_id, frame, local_tracks)
                
                all_global_tracks.extend(global_tracks)
                
                # Update trajectories
                for g_track in global_tracks:
                    global_id = g_track['global_id']
                    box = g_track['box']
                    center = [(box[0] + box[2])/2, (box[1] + box[3])/2]
                    trajectory_predictor.update_trajectory(
                        global_id, center, frame_count)
                
                # Trajectory prediction & alerts
                for g_track in global_tracks:
                    global_id = g_track['global_id']
                    predicted = trajectory_predictor.predict(global_id)
                    
                    if predicted is not None:
                        alerts = alert_system.check_alerts(
                            global_id, camera_id, predicted, g_track['box'])
                        
                        for alert in alerts:
                            alert_system.log_alert(alert)
                
                # Visualization
                vis_frame = visualizer.draw_tracks(frame, global_tracks)
                
                # Display
                cv2.imshow(f'{camera_id}', vis_frame)
            
            # Advance CARLA simulation
            world.tick()
            
            # Check for exit
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    
    finally:
        cv2.destroyAllWindows()
        camera_controller.cleanup()
        traffic_gen.cleanup()
        
        # Disable synchronous mode
        settings = world.get_settings()
        settings.synchronous_mode = False
        world.apply_settings(settings)

if __name__ == '__main__':
    main()
```

---

## Phase 10: Visualization & Monitoring

### 10.1 Visualization Module
**Tệp: `utils/visualization.py`**

```python
import cv2
import numpy as np

class Visualizer:
    def draw_tracks(self, frame, global_tracks):
        """Draw tracked objects với global IDs"""
        frame_copy = frame.copy()
        
        for track in global_tracks:
            global_id = track['global_id']
            box = track['box']
            class_name = track['class']
            
            x1, y1, x2, y2 = box
            
            # Draw bbox
            color = self._id_to_color(global_id)
            cv2.rectangle(frame_copy, (x1, y1), (x2, y2), color, 2)
            
            # Draw label
            label = f"ID:{global_id} {class_name}"
            cv2.putText(frame_copy, label, (x1, y1 - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        
        return frame_copy
    
    def draw_trajectories(self, frame, trajectories):
        """Draw historical trajectory"""
        frame_copy = frame.copy()
        
        for global_id, traj_data in trajectories.items():
            positions = traj_data.get('positions', [])
            
            if len(positions) > 1:
                color = self._id_to_color(global_id)
                for i in range(len(positions) - 1):
                    pt1 = tuple(map(int, positions[i]))
                    pt2 = tuple(map(int, positions[i + 1]))
                    cv2.line(frame_copy, pt1, pt2, color, 1)
        
        return frame_copy
    
    def _id_to_color(self, global_id):
        """Generate consistent color từ ID"""
        np.random.seed(global_id)
        color = tuple(map(int, np.random.rand(3) * 256))
        return color
```

---

## Implementation Timeline

| Phase | Task | Duration | Dependencies |
|-------|------|----------|--------------|
| 1 | Setup environment & folders | 1 day | None |
| 2 | Camera setup & CARLA integration | 2 days | Phase 1 |
| 3 | Object Detection (YOLOv5) | 1 day | Phase 2 |
| 4 | Single-camera tracking | 2 days | Phase 3 |
| 5 | ReID module | 3 days | Phase 4 |
| 6 | Global tracking | 2 days | Phase 5 |
| 7 | Trajectory prediction | 2 days | Phase 6 |
| 8 | Alert system | 1 day | Phase 7 |
| 9 | Main pipeline integration | 2 days | All |
| 10 | Testing & optimization | 3 days | Phase 9 |
| **Total** | | **~19 days** | |

---

## Testing & Evaluation Checklist

- [ ] Single camera detection accuracy (mAP > 0.8)
- [ ] Single camera tracking (MOTA > 0.7, IDF1 > 0.7)
- [ ] Cross-camera ReID (Rank-1 Accuracy > 0.8)
- [ ] Global tracking consistency across cameras
- [ ] Trajectory prediction error (ADE < pixel threshold)
- [ ] Alert system response time (< 100ms)
- [ ] System stability (24h continuous run)
- [ ] Multi-camera synchronization (< 50ms drift)

---

## Key Resources

- CARLA Docs: https://carla.readthedocs.io/
- YOLOv5: https://github.com/ultralytics/yolov5
- Fast-ReID: https://github.com/JDAI-CV/fast-reid
- MOT Metrics: https://github.com/cheind/py-motmetrics

---

## Notes & Tips

1. **Synchronization**: Đảm bảo tất cả cameras capture frames cùng lúc (CARLA synchronous mode)
2. **ReID Tuning**: Threshold matching cần tune dựa vào test set
3. **Performance**: Nếu FPS thấp, giảm image resolution hoặc sử dụng faster YOLO model
4. **Trajectory Prediction**: Bắt đầu với linear motion, sau đó upgrade lên LSTM nếu cần
5. **Dataset Creation**: Tạo synthetic dataset từ CARLA để train/fine-tune mô hình

---

**Status**: Ready to implement ✓
