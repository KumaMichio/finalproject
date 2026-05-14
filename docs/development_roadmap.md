# Development Roadmap: He Thong Giam Sat Camera AI Real-time

## Tong Quan

Tai lieu nay mo ta chi tiet cac phan can phat trien them de bien project hien tai (terminal + OpenCV)
thanh mot he thong giam sat camera hoan chinh voi giao dien server, AI bao dong real-time.

---

## Trang Thai Hien Tai

### Da hoan thanh (~65%)
- [x] CARLA simulator setup (WindowsNoEditor)
- [x] Camera controller — dat nhieu camera trong CARLA, dong bo frame
- [x] Traffic generator — spawn vehicle + pedestrian voi autopilot
- [x] Object detection — YOLOv5 pretrained (person, car, bus, truck)
- [x] Single-camera tracking — IoU greedy matching
- [x] Cross-camera ReID — OSNet (torchreid) + fallback ResNet50
- [x] Global ID assignment — gan ID duy nhat xuyen camera
- [x] Trajectory prediction — linear extrapolation
- [x] Alert system — ROI entry warning (point-in-polygon)
- [x] Visualization — OpenCV window voi bounding box + ID
- [x] Metrics collector — FPS, detection/tracking counts
- [x] Data writer — export JSON, CSV, summary report
- [x] Camera config — YAML (3 cameras, 3 ROIs)
- [x] Documentation — plan, workflow, execute guide

### Chua hoan thanh
- [ ] Web dashboard (frontend)
- [ ] Backend API server
- [ ] Video streaming pipeline
- [ ] Database & persistence
- [ ] Anomaly detection (phat hien su co thong minh)
- [ ] Nang cap AI pipeline (DeepSORT, Kalman, VehicleReID)
- [ ] Camera management UI
- [ ] Recording & playback
- [ ] Testing & evaluation voi ground truth

---

## Kien Truc Tong The Muc Tieu

```
+----------------------------------------------------------------+
|                        WEB BROWSER                              |
|  +-------------+  +----------+  +-------+  +---------------+   |
|  | Camera Grid |  | Alert    |  | Map   |  | Object Search |   |
|  | (live)      |  | Panel    |  | View  |  | & History     |   |
|  +------+------+  +----+-----+  +---+---+  +-------+-------+   |
+---------+---------------+------------+---------------+----------+
          |  WebSocket    |   REST     |               |
+---------v---------------v------------v---------------v----------+
|  +----------------------------------------------------------+   |
|  |                   FastAPI Server                          |   |
|  |  /ws/stream   /api/alerts   /api/tracks   /api/search    |   |
|  +------------------------+---------------------------------+   |
|                           |                                     |
|  +------------------------v---------------------------------+   |
|  |              AI Processing Engine                         |   |
|  |  +--------+ +-------+ +------+ +--------+ +-----------+  |   |
|  |  |Detector|>|Tracker|>| ReID |>|Global  |>| Anomaly   |  |   |
|  |  |YOLOv5  | |SORT   | |OSNet | |Tracker | | Detector  |  |   |
|  |  +--------+ +-------+ +------+ +--------+ +-----+-----+  |   |
|  +------------------------------------------------------+---+   |
|                           |                              |       |
|  +------------------------v------------------------------v---+   |
|  |              Database + Storage                           |   |
|  |  PostgreSQL (alerts, tracks)  |  Redis (real-time pub/sub)|   |
|  |  File Storage (video clips)   |  Gallery (ReID features)  |   |
|  +-----------------------------------------------------------+   |
|                           |                                      |
|  +------------------------v----------------------------------+   |
|  |              CARLA Simulator / Real Cameras                |   |
|  |  CAM_001        CAM_002        CAM_003       CAM_N        |   |
|  +-----------------------------------------------------------+   |
+------------------------------------------------------------------+
```

---

## PHAN 1: Backend API Server

### 1.1 Muc tieu
Tach he thong tu 1 script Python monolithic thanh kien truc client-server.
Frontend (browser) giao tiep voi backend qua REST API + WebSocket.

### 1.2 Cong nghe
- **Framework:** FastAPI (Python) — async, nhanh, auto-generate docs
- **WebSocket:** FastAPI WebSocket — push real-time data den browser
- **Task queue:** Background tasks hoac Celery (xu ly nang chay background)

### 1.3 API Endpoints can xay dung

#### REST API

| Method | Endpoint | Mo ta |
|--------|----------|-------|
| GET | `/api/cameras` | Danh sach cameras + trang thai |
| GET | `/api/cameras/{id}` | Thong tin chi tiet 1 camera |
| POST | `/api/cameras` | Them camera moi |
| PUT | `/api/cameras/{id}` | Cap nhat config camera |
| DELETE | `/api/cameras/{id}` | Xoa camera |
| GET | `/api/tracks` | Danh sach objects dang tracking |
| GET | `/api/tracks/{global_id}` | Chi tiet 1 object (history, trajectory) |
| GET | `/api/tracks/{global_id}/trajectory` | Du lieu quy dao |
| GET | `/api/alerts` | Danh sach alerts (co filter: time, camera, type, severity) |
| GET | `/api/alerts/{id}` | Chi tiet 1 alert |
| PUT | `/api/alerts/{id}/acknowledge` | Xac nhan da xu ly alert |
| GET | `/api/rois` | Danh sach ROIs |
| POST | `/api/rois` | Them ROI moi |
| PUT | `/api/rois/{id}` | Cap nhat ROI |
| DELETE | `/api/rois/{id}` | Xoa ROI |
| GET | `/api/stats` | System stats (FPS, object count, uptime) |
| GET | `/api/stats/history` | Performance history (chart data) |

#### WebSocket

| Endpoint | Mo ta | Data format |
|----------|-------|-------------|
| `/ws/alerts` | Push alert moi real-time | `{type, global_id, camera_id, severity, timestamp}` |
| `/ws/tracks` | Push tracking updates | `{global_id, camera_id, box, class, frame_count}` |
| `/ws/stats` | Push system stats moi 1s | `{fps, active_tracks, alert_count}` |
| `/ws/stream/{camera_id}` | Video stream 1 camera | Binary frames (MJPEG) |

### 1.4 Cau truc thu muc moi

```
server/
  app.py                  # FastAPI app chinh
  config.py               # Server configuration
  routers/
    cameras.py            # Camera endpoints
    tracks.py             # Tracking endpoints
    alerts.py             # Alert endpoints
    rois.py               # ROI endpoints
    stats.py              # Statistics endpoints
    websocket.py          # WebSocket handlers
  services/
    camera_service.py     # Business logic cameras
    tracking_service.py   # Business logic tracking
    alert_service.py      # Business logic alerts
    stream_service.py     # Video streaming logic
  models/
    database.py           # Database models (SQLAlchemy)
    schemas.py            # Pydantic request/response schemas
  middleware/
    auth.py               # Authentication
    cors.py               # CORS config
```

### 1.5 Tich hop voi AI Pipeline hien tai

```python
# Hien tai: main.py chay 1 vong while True
# Can doi: tach AI processing thanh background task

class AIProcessor:
    """Chay AI pipeline trong background thread/process"""

    def __init__(self):
        self.detector = ObjectDetector()
        self.trackers = {}
        self.reid = ReIDExtractor()
        self.global_tracker = GlobalTracker(self.reid)
        self.trajectory_predictor = TrajectoryPredictor()
        self.alert_system = AlertSystem(self.trajectory_predictor)
        self.anomaly_detector = AnomalyDetector()  # MOI

    def process_frame(self, camera_id, frame):
        """Xu ly 1 frame, tra ve ket qua"""
        detections = self.detector.detect(frame)
        local_tracks = self.trackers[camera_id].update(detections)
        global_tracks = self.global_tracker.process_camera_tracks(...)
        # ... trajectory, alerts, anomaly ...
        return {
            'detections': detections,
            'tracks': global_tracks,
            'alerts': alerts,
            'annotated_frame': annotated_frame
        }
```

---

## PHAN 2: Web Dashboard (Frontend)

### 2.1 Muc tieu
Giao dien web cho operator giam sat nhieu camera dong thoi,
nhan bao dong real-time, tra cuu lich su.

### 2.2 Cong nghe
- **Framework:** React (hoac Vue 3)
- **UI Library:** Tailwind CSS + shadcn/ui (hoac Ant Design)
- **Real-time:** WebSocket client
- **Video:** MJPEG hoac `<img>` tag refresh (don gian), WebRTC (nang cao)
- **Charts:** Chart.js hoac Recharts
- **Map:** Leaflet hoac custom canvas

### 2.3 Cac trang can xay dung

#### Trang 1: Dashboard chinh (/)
```
+----------------------------------------------------------+
| HEADER: Logo | System Status: ONLINE | FPS: 12 | 15:30  |
+----------------------------------------------------------+
| CAMERA GRID (2x2 hoac 3x3)                               |
| +------------+ +------------+ +------------+              |
| | CAM_001    | | CAM_002    | | CAM_003    |              |
| | [live feed]| | [live feed]| | [live feed]|              |
| | 5 objects  | | 3 objects  | | 7 objects  |              |
| +------------+ +------------+ +------------+              |
+----------------------------------------------------------+
| ALERT PANEL (scroll, moi nhat o tren)                     |
| ! CRITICAL 15:29 - Xe #1042 vuot toc do tai CAM_001      |
| ! WARNING  15:28 - Nguoi #1055 vao vung cam CAM_002      |
| i INFO     15:25 - Object #1038 chuyen tu CAM_001>CAM_003|
+----------------------------------------------------------+
| STATS BAR: Active Objects: 15 | Alerts Today: 23 | ...   |
+----------------------------------------------------------+
```

#### Trang 2: Camera detail (/camera/:id)
```
+----------------------------------------------------------+
| CAM_001 - Giao lo chinh                                   |
| +----------------------------------+ +------------------+ |
| |                                  | | Objects:         | |
| |         LIVE FEED LON            | | #1042 Car   [>]  | |
| |    (full HD, voi overlay)        | | #1055 Person[>]  | |
| |                                  | | #1060 Truck [>]  | |
| +----------------------------------+ +------------------+ |
| ROI Zones: [intersection_main] [danger_zone]              |
| [Edit ROIs]  [Toggle Trajectory]  [Screenshot]            |
+----------------------------------------------------------+
```

#### Trang 3: Object detail (/object/:global_id)
```
+----------------------------------------------------------+
| Object #1042 - Car                                        |
| +--------+ +--------+ +--------+                          |
| |Crop    | |Crop    | |Crop    |   First seen: 15:20     |
| |CAM_001 | |CAM_002 | |CAM_003 |   Last seen:  15:29     |
| +--------+ +--------+ +--------+   Cameras: 3            |
|                                                           |
| TRAJECTORY MAP:                                           |
| CAM_001 -----> CAM_003 -----> CAM_002                     |
| 15:20          15:24          15:28                        |
|                                                           |
| ALERT HISTORY:                                            |
| 15:29 - Overspeed tai CAM_001 (85 km/h, limit 60)        |
| 15:24 - Entered ROI "intersection" tai CAM_003            |
+----------------------------------------------------------+
```

#### Trang 4: Alert management (/alerts)
```
+----------------------------------------------------------+
| ALERTS                                                     |
| Filter: [All types v] [All cameras v] [Today v] [Search]  |
+----------------------------------------------------------+
| Time  | Type     | Severity | Camera | Object | Status    |
| 15:29 | Speed    | CRITICAL | CAM_01 | #1042  | NEW      |
| 15:28 | ROI      | WARNING  | CAM_02 | #1055  | ACK      |
| 15:25 | Transfer | INFO     | CAM_03 | #1038  | AUTO     |
| ...                                                        |
+----------------------------------------------------------+
| [Export CSV]  [Clear All]                     Page 1 of 5  |
+----------------------------------------------------------+
```

#### Trang 5: Settings (/settings)
- Quan ly camera (them/sua/xoa)
- Chinh ROI (ve tren hinh)
- Cau hinh alert thresholds
- System parameters (FPS, model, confidence)

### 2.4 Cau truc thu muc frontend

```
frontend/
  public/
    index.html
  src/
    components/
      CameraGrid.jsx         # Grid hien thi nhieu camera
      CameraFeed.jsx          # 1 camera feed voi overlay
      AlertPanel.jsx          # Panel canh bao real-time
      AlertList.jsx           # Danh sach alerts (trang alerts)
      ObjectDetail.jsx        # Chi tiet 1 object
      TrajectoryMap.jsx       # Ban do quy dao
      ROIEditor.jsx           # Ve/sua ROI tren camera
      StatsBar.jsx            # Thanh thong ke
      SystemStatus.jsx        # Trang thai he thong
    pages/
      Dashboard.jsx           # Trang chinh
      CameraDetail.jsx        # Chi tiet camera
      ObjectView.jsx          # Chi tiet object
      AlertManagement.jsx     # Quan ly alerts
      Settings.jsx            # Cai dat
    hooks/
      useWebSocket.js         # WebSocket connection
      useAlerts.js            # Alert state management
      useTracks.js            # Tracking state
    services/
      api.js                  # REST API calls
      websocket.js            # WebSocket client
    App.jsx
    main.jsx
  package.json
```

---

## PHAN 3: Video Streaming Pipeline

### 3.1 Muc tieu
Truyen video tu AI pipeline den browser voi do tre thap,
hien thi bounding box + ID overlay tren video.

### 3.2 Phuong phap

#### Option A: MJPEG over HTTP (don gian, khuyen dung giai doan dau)
```
AI Pipeline --> Annotate frame --> JPEG encode --> HTTP stream --> <img> tag
```
- Uu: Don gian, tuong thich moi browser
- Nhuoc: Bandwidth cao, khong co audio

#### Option B: WebRTC (nang cao, khuyen dung giai doan sau)
```
AI Pipeline --> Annotate frame --> WebRTC encode --> Browser video element
```
- Uu: Do tre thap (~100ms), tiet kiem bandwidth
- Nhuoc: Phuc tap hon

### 3.3 Implementation MJPEG

```python
# server/routers/stream.py

from fastapi.responses import StreamingResponse

async def generate_mjpeg(camera_id: str):
    while True:
        frame = await get_annotated_frame(camera_id)
        _, jpeg = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
        yield (
            b'--frame\r\n'
            b'Content-Type: image/jpeg\r\n\r\n'
            + jpeg.tobytes()
            + b'\r\n'
        )

@router.get("/stream/{camera_id}")
async def video_stream(camera_id: str):
    return StreamingResponse(
        generate_mjpeg(camera_id),
        media_type="multipart/x-mixed-replace; boundary=frame"
    )
```

### 3.4 Recording

| Tinh nang | Mo ta |
|-----------|-------|
| **Continuous recording** | Ghi lien tuc tat ca camera, luu file MP4 theo gio |
| **Event clip** | Khi co alert, cat clip 30s truoc + 30s sau su co |
| **Snapshot** | Chup anh 1 frame khi operator yeu cau |
| **Storage rotation** | Tu dong xoa video cu khi day dung luong |

```python
# modules/recorder.py

class VideoRecorder:
    def __init__(self, output_dir, segment_minutes=60):
        self.output_dir = output_dir
        self.segment_minutes = segment_minutes
        self.writers = {}  # {camera_id: cv2.VideoWriter}

    def write_frame(self, camera_id, frame):
        """Ghi 1 frame vao file video hien tai"""
        ...

    def save_event_clip(self, camera_id, alert, before_sec=30, after_sec=30):
        """Cat clip quanh thoi diem su co"""
        ...

    def take_snapshot(self, camera_id, frame):
        """Chup 1 anh"""
        ...

    def rotate_storage(self, max_gb=50):
        """Xoa file cu khi vuot dung luong"""
        ...
```

---

## PHAN 4: Database & Persistence

### 4.1 Muc tieu
Luu tru du lieu tracking, alerts, config vao database.
Restart he thong khong mat du lieu.

### 4.2 Cong nghe
- **Database:** SQLite (don gian, du dung) hoac PostgreSQL (scale lon)
- **ORM:** SQLAlchemy
- **Migration:** Alembic
- **Cache:** Redis (optional, cho real-time pub/sub)

### 4.3 Database Schema

#### Bang `cameras`
```sql
CREATE TABLE cameras (
    id          TEXT PRIMARY KEY,      -- "CAM_001"
    name        TEXT NOT NULL,         -- "Giao lo chinh"
    position_x  REAL,
    position_y  REAL,
    position_z  REAL,
    rotation_pitch REAL,
    rotation_yaw   REAL,
    rotation_roll  REAL,
    resolution_w INTEGER DEFAULT 960,
    resolution_h INTEGER DEFAULT 540,
    fov         INTEGER DEFAULT 90,
    fps         INTEGER DEFAULT 10,
    status      TEXT DEFAULT 'active', -- active / inactive / error
    created_at  TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at  TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
```

#### Bang `alerts`
```sql
CREATE TABLE alerts (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    type        TEXT NOT NULL,          -- ROI_WARNING, OVERSPEED, WRONG_WAY, ...
    severity    TEXT NOT NULL,          -- info, warning, critical
    global_id   INTEGER,               -- object ID
    camera_id   TEXT REFERENCES cameras(id),
    roi_name    TEXT,
    message     TEXT,
    details     TEXT,                   -- JSON: {speed, direction, eta, ...}
    snapshot_path TEXT,                 -- Duong dan anh chup luc alert
    clip_path   TEXT,                   -- Duong dan video clip
    status      TEXT DEFAULT 'new',    -- new, acknowledged, resolved, false_alarm
    acknowledged_by TEXT,
    created_at  TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    acknowledged_at TIMESTAMP
);
CREATE INDEX idx_alerts_time ON alerts(created_at DESC);
CREATE INDEX idx_alerts_camera ON alerts(camera_id);
CREATE INDEX idx_alerts_type ON alerts(type);
CREATE INDEX idx_alerts_severity ON alerts(severity);
```

#### Bang `tracked_objects`
```sql
CREATE TABLE tracked_objects (
    global_id   INTEGER PRIMARY KEY,
    class       TEXT NOT NULL,          -- car, person, bus, truck
    first_seen  TIMESTAMP NOT NULL,
    last_seen   TIMESTAMP NOT NULL,
    total_cameras INTEGER DEFAULT 1,
    status      TEXT DEFAULT 'active',  -- active, lost, archived
    created_at  TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
```

#### Bang `tracking_history`
```sql
CREATE TABLE tracking_history (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    global_id   INTEGER REFERENCES tracked_objects(global_id),
    camera_id   TEXT REFERENCES cameras(id),
    frame_number INTEGER,
    box_x1      INTEGER,
    box_y1      INTEGER,
    box_x2      INTEGER,
    box_y2      INTEGER,
    center_x    REAL,
    center_y    REAL,
    confidence  REAL,
    timestamp   TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
CREATE INDEX idx_history_object ON tracking_history(global_id);
CREATE INDEX idx_history_camera ON tracking_history(camera_id);
CREATE INDEX idx_history_time ON tracking_history(timestamp DESC);
```

#### Bang `rois`
```sql
CREATE TABLE rois (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    camera_id   TEXT REFERENCES cameras(id),
    name        TEXT NOT NULL,
    polygon     TEXT NOT NULL,          -- JSON: [[x,y], [x,y], ...]
    alert_types TEXT DEFAULT 'entry',   -- entry, exit, loiter, speed
    is_active   BOOLEAN DEFAULT TRUE,
    created_at  TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
```

#### Bang `reid_gallery`
```sql
CREATE TABLE reid_gallery (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    global_id   INTEGER REFERENCES tracked_objects(global_id),
    feature     BLOB NOT NULL,          -- numpy array serialized
    camera_id   TEXT,
    snapshot    BLOB,                   -- JPEG crop cua object
    created_at  TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
CREATE INDEX idx_gallery_object ON reid_gallery(global_id);
```

---

## PHAN 5: He Thong Alert Thong Minh (Anomaly Detection)

### 5.1 Muc tieu
Nang cap tu "chi canh bao khi vao ROI" thanh he thong phat hien
nhieu loai su co khac nhau, phan cap muc do nghiem trong.

### 5.2 Cac loai su co can phat hien

| # | Loai su co | Severity | Logic phat hien |
|---|-----------|----------|-----------------|
| 1 | **ROI entry** | WARNING | Object enters ROI polygon (DA CO) |
| 2 | **ROI exit** | INFO | Object exits ROI polygon |
| 3 | **Overspeed** | CRITICAL | speed > speed_limit (tinh tu pixel velocity + camera calibration) |
| 4 | **Wrong-way** | CRITICAL | Huong di chuyen nguoc voi huong cho phep cua lane |
| 5 | **Stopped vehicle** | WARNING | Xe dung yen > N frames tren duong (khong phai bai do) |
| 6 | **Sudden stop** | CRITICAL | Toc do giam dot ngot (co the tai nan) |
| 7 | **Object disappeared** | WARNING | Object mat tich dot ngot (khong phai ra khoi frame) |
| 8 | **Crowd density** | WARNING | So nguoi trong 1 vung > threshold |
| 9 | **Loitering** | WARNING | 1 nguoi o cung 1 vung qua lau |
| 10 | **Camera transition** | INFO | Object chuyen tu camera nay sang camera khac |

### 5.3 Cau truc module moi

```python
# modules/anomaly_detector.py

class AnomalyDetector:
    """Phat hien cac hanh vi bat thuong"""

    def __init__(self, config):
        self.speed_limit = config.get('speed_limit', 60)       # pixels/frame
        self.stop_threshold = config.get('stop_threshold', 30)  # frames
        self.crowd_threshold = config.get('crowd_threshold', 10)
        self.loiter_threshold = config.get('loiter_threshold', 300)  # frames

    def check_all(self, global_id, camera_id, track_data, all_tracks):
        """Kiem tra tat ca loai anomaly cho 1 object"""
        alerts = []
        alerts += self.check_overspeed(global_id, track_data)
        alerts += self.check_stopped(global_id, track_data)
        alerts += self.check_sudden_stop(global_id, track_data)
        alerts += self.check_wrong_way(global_id, camera_id, track_data)
        alerts += self.check_crowd(camera_id, all_tracks)
        alerts += self.check_loitering(global_id, track_data)
        return alerts

    def check_overspeed(self, global_id, track_data):
        """Kiem tra vuot toc do"""
        ...

    def check_stopped(self, global_id, track_data):
        """Kiem tra xe dung bat thuong"""
        ...

    def check_sudden_stop(self, global_id, track_data):
        """Kiem tra dung dot ngot (nghi tai nan)"""
        ...

    def check_wrong_way(self, global_id, camera_id, track_data):
        """Kiem tra di nguoc chieu"""
        ...

    def check_crowd(self, camera_id, all_tracks):
        """Kiem tra mat do dong nguoi"""
        ...

    def check_loitering(self, global_id, track_data):
        """Kiem tra lang vang"""
        ...
```

### 5.4 Alert Severity & Notification

```
CRITICAL  -->  Am thanh bao dong + popup do + ghi clip tu dong
WARNING   -->  Notification vang + ghi log
INFO      -->  Hien thi trong alert feed, khong bao dong
```

### 5.5 Config cho Alert System

```yaml
# Them vao camera_config.yaml hoac file rieng alert_config.yaml

alerts:
  overspeed:
    enabled: true
    speed_limit: 80          # pixels/frame (can calibrate theo camera)
    severity: critical
  stopped_vehicle:
    enabled: true
    min_frames: 50           # Dung yen 50 frames = 5 giay (10 FPS)
    severity: warning
  sudden_stop:
    enabled: true
    deceleration_threshold: 0.8  # Giam 80% toc do trong 1 frame
    severity: critical
  wrong_way:
    enabled: true
    severity: critical
    lanes:                   # Dinh nghia huong di cho phep per camera
      CAM_001:
        direction: [1, 0]    # Chi cho phep di sang phai
      CAM_002:
        direction: [-1, 0]   # Chi cho phep di sang trai
  crowd_density:
    enabled: true
    max_persons: 15
    zone_size: [200, 200]    # pixels
    severity: warning
  loitering:
    enabled: true
    max_duration_frames: 300 # 30 giay (10 FPS)
    radius: 50               # pixels — di chuyen < 50px = loitering
    severity: warning
```

---

## PHAN 6: Nang Cap AI Pipeline

### 6.1 Tracker: IoU Greedy --> DeepSORT / ByteTrack

#### Van de hien tai
- Greedy matching khong toi uu (co the match sai khi nhieu object gan nhau)
- Khong co Kalman filter de du doan vi tri khi object bi che (occlusion)
- Khong dung appearance feature de phan biet object tuong tu

#### Giai phap: Tich hop ByteTrack

```python
# modules/tracker.py — thay the SimpleTracker

class ByteTracker:
    """ByteTrack: multi-object tracker su dung 2 vong matching"""

    def __init__(self, max_age=30, min_hits=3,
                 high_threshold=0.6, low_threshold=0.1):
        self.max_age = max_age
        self.min_hits = min_hits
        self.high_threshold = high_threshold
        self.low_threshold = low_threshold
        self.kalman_filters = {}  # Kalman filter per track

    def update(self, detections):
        # Vong 1: Match high-confidence detections voi tracks (IoU)
        # Vong 2: Match low-confidence detections voi unmatched tracks
        # Kalman predict cho unmatched tracks
        ...
```

#### Thay the: DeepSORT (co appearance matching)

```python
class DeepSORTTracker:
    """DeepSORT: Kalman filter + appearance feature matching"""

    def __init__(self, reid_model, max_age=30, max_cosine_distance=0.3):
        self.reid_model = reid_model
        self.max_age = max_age
        self.max_cosine_distance = max_cosine_distance
        self.kalman_filters = {}
        self.tracks = []

    def update(self, detections, frame):
        # 1. Kalman predict
        # 2. Extract appearance features
        # 3. Cascade matching (appearance + IoU)
        # 4. Update Kalman state
        ...
```

### 6.2 Trajectory Prediction: Linear --> Kalman Filter / LSTM

#### Option A: Kalman Filter (khuyen dung, khong can training)

```python
# modules/trajectory_predictor.py — them method

class KalmanPredictor:
    """Kalman Filter cho trajectory prediction"""

    def __init__(self):
        # State: [x, y, vx, vy, ax, ay]
        # Measurement: [x, y]
        self.filters = {}  # {global_id: KalmanFilter}

    def update_and_predict(self, global_id, position, pred_steps=5):
        if global_id not in self.filters:
            self.filters[global_id] = self._create_filter()

        kf = self.filters[global_id]
        kf.update(position)

        predictions = []
        state = kf.state.copy()
        for _ in range(pred_steps):
            state = kf.predict_state(state)
            predictions.append([state[0], state[1]])

        return predictions
```

#### Option B: LSTM (can training, do chinh xac cao hon)

```python
# models/tracking/trajectory_lstm.py

class TrajectoryLSTM(nn.Module):
    """LSTM model cho trajectory prediction"""

    def __init__(self, input_dim=2, hidden_dim=64, output_dim=2,
                 num_layers=2, pred_steps=5):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim * pred_steps)
        self.pred_steps = pred_steps

    def forward(self, x):
        # x: (batch, seq_len, 2)
        lstm_out, _ = self.lstm(x)
        prediction = self.fc(lstm_out[:, -1, :])
        return prediction.view(-1, self.pred_steps, 2)
```

Training data: Thu thap tu CARLA (xem Phan 7.3)

### 6.3 Vehicle ReID

#### Van de hien tai
- OSNet / Market-1501 chi train cho person ReID
- Vehicle appearance khac biet nhieu so voi person

#### Giai phap

| Option | Mo ta | Do kho |
|--------|-------|--------|
| **A: VehicleID pretrained** | Dung model pretrained tren VehicleID dataset | Thap |
| **B: Fine-tune OSNet** | Fine-tune OSNet tren CARLA synthetic vehicle data | Trung binh |
| **C: Dual-branch ReID** | 1 model cho person + 1 model cho vehicle | Trung binh |

```python
# modules/reid.py — mo rong

class DualReIDExtractor:
    """ReID voi 2 model rieng cho person va vehicle"""

    def __init__(self):
        self.person_model = self._load_person_model()   # OSNet Market-1501
        self.vehicle_model = self._load_vehicle_model()  # VehicleNet hoac fine-tuned

    def extract_feature(self, frame, box, object_class):
        if object_class == 'person':
            return self._extract(self.person_model, frame, box)
        else:  # car, bus, truck
            return self._extract(self.vehicle_model, frame, box)
```

### 6.4 Metrics thuc (MOTA, IDF1, mAP)

#### Van de hien tai
- MetricsCollector chi dem FPS va counts
- Khong co ground truth de so sanh
- Khong tinh MOTA, IDF1, mAP thuc su

#### Giai phap

```python
# utils/metrics.py — them evaluation thuc

import motmetrics as mm  # pip install motmetrics

class MOTEvaluator:
    """Danh gia tracking voi ground truth tu CARLA"""

    def __init__(self):
        self.accumulator = mm.MOTAccumulator(auto_id=True)

    def update(self, frame_id, gt_objects, predicted_tracks):
        """
        gt_objects: lay tu CARLA world.get_actors()
        predicted_tracks: output tu global tracker
        """
        gt_ids = [obj.id for obj in gt_objects]
        pred_ids = [t['global_id'] for t in predicted_tracks]

        # Tinh distance matrix
        distances = mm.distances.iou_matrix(
            gt_boxes, pred_boxes, max_iou=0.5)

        self.accumulator.update(gt_ids, pred_ids, distances)

    def compute_metrics(self):
        mh = mm.metrics.create()
        summary = mh.compute(
            self.accumulator,
            metrics=['mota', 'idf1', 'precision', 'recall'],
            name='tracking'
        )
        return summary
```

Ground truth lay tu CARLA:
```python
# CARLA cung cap actor ID that (ground truth)
actors = world.get_actors()
vehicles = actors.filter('vehicle.*')
walkers = actors.filter('walker.*')

for vehicle in vehicles:
    gt_id = vehicle.id
    gt_location = vehicle.get_location()
    gt_bbox = vehicle.bounding_box
```

---

## PHAN 7: Camera Management

### 7.1 Them/Xoa camera runtime

```python
# server/services/camera_service.py

class CameraManager:
    def add_camera(self, camera_config):
        """Them camera moi khong can restart he thong"""
        camera = self.camera_controller.spawn_camera(camera_config)
        self.trackers[camera.id] = SimpleTracker()
        self.db.insert_camera(camera_config)
        self.broadcast_event('camera_added', camera.id)

    def remove_camera(self, camera_id):
        """Xoa camera"""
        self.camera_controller.destroy_camera(camera_id)
        del self.trackers[camera_id]
        self.db.deactivate_camera(camera_id)
        self.broadcast_event('camera_removed', camera_id)
```

### 7.2 Camera health monitoring

```python
class CameraHealthCheck:
    def __init__(self, timeout_seconds=5):
        self.last_frame_time = {}

    def check_health(self, camera_id):
        elapsed = time.time() - self.last_frame_time.get(camera_id, 0)
        if elapsed > self.timeout_seconds:
            return {
                'status': 'error',
                'message': f'No frame received for {elapsed:.1f}s'
            }
        return {'status': 'ok'}
```

### 7.3 ROI Editor tren UI

```
Operator co the:
1. Click vao camera view
2. Ve polygon bang chuot (click cac dinh)
3. Dat ten cho ROI
4. Chon loai alert (entry, exit, speed, loiter)
5. Save --> API POST /api/rois --> Database --> Alert system cap nhat
```

---

## PHAN 8: Recording & Playback

### 8.1 Continuous Recording

```python
# modules/recorder.py

class ContinuousRecorder:
    """Ghi video lien tuc, chia file theo gio"""

    def __init__(self, output_dir, segment_minutes=60, fps=10):
        self.output_dir = output_dir
        self.segment_minutes = segment_minutes
        self.fps = fps
        self.writers = {}

    def write_frame(self, camera_id, frame, timestamp):
        writer = self._get_or_create_writer(camera_id, timestamp)
        writer.write(frame)

    def _get_or_create_writer(self, camera_id, timestamp):
        segment_key = timestamp.strftime('%Y%m%d_%H')
        ...
```

### 8.2 Event Clip

```python
class EventClipExtractor:
    """Cat clip quanh thoi diem su co"""

    def __init__(self, buffer_size=300):
        # Ring buffer luu 300 frames gan nhat (30s tai 10FPS)
        self.frame_buffers = {}  # {camera_id: deque}

    def buffer_frame(self, camera_id, frame, timestamp):
        if camera_id not in self.frame_buffers:
            self.frame_buffers[camera_id] = deque(maxlen=self.buffer_size)
        self.frame_buffers[camera_id].append((frame.copy(), timestamp))

    async def save_event_clip(self, camera_id, alert_time,
                               before_sec=30, after_sec=30):
        """Cat clip truoc va sau su co"""
        # Lay frames tu buffer (truoc su co)
        # Tiep tuc ghi them after_sec giay
        # Luu thanh file MP4
        ...
```

### 8.3 Playback API

```
GET /api/recordings?camera=CAM_001&date=2026-05-12
    --> Danh sach file video trong ngay

GET /api/recordings/{file_id}/stream
    --> Stream video file

GET /api/alerts/{alert_id}/clip
    --> Stream clip cua su co
```

---

## Thu Muc Du An Sau Khi Phat Trien

```
finalproject/
  AI_custom/
    custom_tracking_system/
      config/
        camera_config.yaml
        alert_config.yaml         # MOI: config cho anomaly detection
      modules/
        camera_controller.py
        traffic_generator.py
        detector.py
        tracker.py                # NANG CAP: ByteTrack/DeepSORT
        reid.py                   # NANG CAP: DualReID (person + vehicle)
        global_tracking.py
        trajectory_predictor.py   # NANG CAP: Kalman / LSTM
        alert_system.py
        anomaly_detector.py       # MOI: phat hien su co thong minh
        recorder.py               # MOI: recording & event clips
      utils/
        visualization.py
        metrics.py                # NANG CAP: MOTA, IDF1, mAP
        data_writer.py
      models/
        detection/
        reid/
        tracking/
          trajectory_lstm.pth     # MOI: LSTM model (optional)
      datasets/
      main.py                     # NANG CAP: tich hop modules moi
      requirements.txt            # NANG CAP: them dependencies moi
    server/                       # MOI: Backend API server
      app.py
      config.py
      routers/
        cameras.py
        tracks.py
        alerts.py
        rois.py
        stats.py
        stream.py
        websocket.py
      services/
        camera_service.py
        tracking_service.py
        alert_service.py
        stream_service.py
      models/
        database.py
        schemas.py
      middleware/
        auth.py
        cors.py
    frontend/                     # MOI: Web dashboard
      src/
        components/
        pages/
        hooks/
        services/
      public/
      package.json
    docs/
      description.md
      execute.md
      instruction.txt
      plan.md
      workflow.md
      development_roadmap.md      # FILE NAY
  WindowsNoEditor/                # CARLA simulator (khong thay doi)
```

---

## Thu Tu Uu Tien Phat Trien

```
UU TIEN 1 (tuan 1-3):
  Backend API + Database + Video streaming
  --> Tao nen tang server truoc

UU TIEN 2 (tuan 3-6):
  Web Dashboard co ban (camera grid + alert panel)
  --> Operator bat dau dung duoc

UU TIEN 3 (tuan 5-8):
  Nang cap Alert system + Anomaly detection
  --> Phat hien su co thong minh

UU TIEN 4 (tuan 7-10):
  Nang cap AI pipeline (ByteTrack, Kalman, VehicleReID)
  --> Tracking chinh xac hon

UU TIEN 5 (tuan 10-12):
  Camera management + Recording + Playback
  --> Tinh nang nang cao

UU TIEN 6 (tuan 12-15):
  Testing + toi uu + polish UI
  --> Production-ready
```

---

## Dependencies Moi Can Them

```txt
# Server
fastapi>=0.100.0
uvicorn>=0.23.0
websockets>=11.0
sqlalchemy>=2.0
alembic>=1.11
pydantic>=2.0
python-jose>=3.3        # JWT auth
python-multipart>=0.0.6
aiofiles>=23.0

# AI nang cap
motmetrics>=1.4         # MOT evaluation
filterpy>=1.4           # Kalman filter
deep-sort-realtime>=1.3 # DeepSORT (optional)

# Video
aiortc>=1.6             # WebRTC (optional)

# Frontend (package.json)
react>=18
react-router-dom>=6
tailwindcss>=3
recharts>=2             # Charts
```
