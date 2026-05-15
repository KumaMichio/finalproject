# Bao Cao Du An: He Thong Giam Sat Camera Da Kenh Ung Dung AI

## 1. Muc Tieu Du An

### 1.1 Muc tieu tong quat

Xay dung he thong giam sat camera CCTV da kenh (multi-camera) su dung tri tue nhan tao,
co kha nang:

- Phat hien doi tuong (xe co, nguoi di bo) tu nhieu camera dong thoi.
- Theo doi doi tuong lien tuc trong tung camera va xuyen camera (cross-camera tracking).
- Nhan dien lai doi tuong khi doi tuong di chuyen tu vung nhin cua camera nay sang camera khac (Re-Identification).
- Gan ma dinh danh toan cuc (Global ID) duy nhat cho moi doi tuong trong toan bo he thong.
- Du doan quy dao di chuyen tuong lai cua doi tuong.
- Canh bao khi doi tuong di vao cac vung quan tam (Region of Interest).
- Cung cap giao dien API de phuc vu cho cac ung dung giam sat (web dashboard, mobile app).

### 1.2 Moi truong mo phong

Du an su dung CARLA Simulator phien ban 0.9.9.4 lam moi truong mo phong.
CARLA la mot simulator ma nguon mo chuyen dung cho nghien cuu xe tu lai,
cung cap thanh pho ao, phuong tien, nguoi di bo, va cac cam bien (camera, LiDAR, radar).

Moi truong CARLA cho phep:
- Dat nhieu camera tai cac vi tri tuy y trong thanh pho ao.
- Sinh giao thong tu dong (xe va nguoi di bo voi AI autopilot).
- Thu thap du lieu camera dong bo (synchronous mode).
- Kiem soat hoan toan cac dieu kien mo phong (thoi tiet, anh sang, mat do giao thong).

### 1.3 Pham vi hien tai

He thong hien tai gom 2 phan chinh:
- **AI Pipeline** (`custom_tracking_system/`): Xu ly hinh anh tu camera, phat hien + theo doi + nhan dien lai doi tuong.
- **Backend API Server** (`server/`): Cung cap REST API, WebSocket, va video streaming de cac ung dung khac (web dashboard) co the truy cap du lieu tracking va canh bao.

---

## 2. Cau Truc Du An

```
finalproject/
|
|-- AI_custom/                          # Phan code tu phat trien
|   |
|   |-- custom_tracking_system/         # AI Pipeline
|   |   |-- config/
|   |   |   +-- camera_config.yaml      # Cau hinh camera va vung ROI
|   |   |-- modules/
|   |   |   |-- camera_controller.py    # Dieu khien camera trong CARLA
|   |   |   |-- traffic_generator.py    # Sinh xe co + nguoi di bo
|   |   |   |-- detector.py            # Phat hien doi tuong (YOLOv5)
|   |   |   |-- tracker.py             # Theo doi trong 1 camera (IoU matching)
|   |   |   |-- reid.py                # Nhan dien lai xuyen camera (OSNet/ResNet50)
|   |   |   |-- global_tracking.py     # Gan Global ID xuyen camera
|   |   |   |-- trajectory_predictor.py # Du doan quy dao
|   |   |   |-- alert_system.py        # Canh bao vung ROI
|   |   |   |-- video_source.py        # Abstract video source (CARLA/RTSP/File/Webcam)
|   |   |   +-- ground_truth.py        # Ground truth cho evaluation (tach khoi AI pipeline)
|   |   |-- utils/
|   |   |   |-- visualization.py       # Ve bounding box, trajectory, ROI
|   |   |   |-- metrics.py             # Thu thap chi so hieu suat
|   |   |   +-- data_writer.py         # Xuat du lieu JSON, CSV, bao cao
|   |   |-- models/
|   |   |   |-- hub/                   # YOLOv5 model cache (tu dong tai)
|   |   |   |-- detection/             # (chua co model fine-tune)
|   |   |   |-- reid/                  # (chua co model fine-tune)
|   |   |   +-- tracking/              # (chua co model fine-tune)
|   |   |-- datasets/
|   |   |   |-- ground_truth/          # (chua co du lieu)
|   |   |   +-- synthetic_data/        # (chua co du lieu)
|   |   |-- main.py                    # Entry point chay truc tiep (khong qua server)
|   |   |-- requirements.txt           # Dependencies cho AI pipeline
|   |   +-- run.bat                    # Script chay tren Windows
|   |
|   |-- server/                         # Backend API Server
|   |   |-- models/
|   |   |   |-- database.py            # SQLAlchemy ORM (5 bang)
|   |   |   +-- schemas.py             # Pydantic request/response schemas
|   |   |-- routers/
|   |   |   |-- cameras.py             # CRUD camera
|   |   |   |-- tracks.py              # Query doi tuong + quy dao
|   |   |   |-- alerts.py              # Query + xac nhan canh bao
|   |   |   |-- rois.py                # CRUD vung ROI
|   |   |   |-- stats.py               # Thong ke he thong
|   |   |   |-- websocket.py           # WebSocket push real-time
|   |   |   +-- stream.py              # MJPEG video streaming
|   |   |-- services/
|   |   |   |-- ai_processor.py        # Tich hop AI pipeline vao server
|   |   |   |-- camera_service.py      # Logic nghiep vu camera
|   |   |   |-- alert_service.py       # Logic nghiep vu canh bao
|   |   |   |-- tracking_service.py    # Logic nghiep vu tracking
|   |   |   +-- stream_service.py      # Buffer frame cho MJPEG
|   |   |-- middleware/                 # (du tru cho auth, rate limit)
|   |   |-- app.py                     # FastAPI app chinh
|   |   |-- config.py                  # Cau hinh server
|   |   |-- requirements.txt           # Dependencies cho server
|   |   +-- run_server.bat             # Script chay server tren Windows
|   |
|   +-- docs/                           # Tai lieu du an
|       |-- description.md             # Mo ta workspace
|       |-- plan.md                    # Ke hoach phat trien chi tiet
|       |-- workflow.md                # Quy trinh trien khai tung giai doan
|       |-- execute.md                 # Huong dan chay + cac kich ban test
|       |-- development_roadmap.md     # Lo trinh phat trien tiep theo
|       |-- report.md                  # Bao cao du an (file nay)
|       +-- project_context.md         # Context doc cho prompt tren mobile
|
+-- WindowsNoEditor/                    # CARLA Simulator 0.9.9.4
    |-- CarlaUE4.exe                   # File thuc thi CARLA server
    |-- PythonAPI/                     # Thu vien Python API cua CARLA
    |   |-- carla/                     # CARLA Python package
    |   |-- examples/                  # Vi du su dung API
    |   +-- util/                      # Tien ich
    |-- Engine/                        # Unreal Engine runtime
    |-- Co-Simulation/                 # Tich hop SUMO, PTV-Vissim
    +-- HDMaps/                        # Ban do do phan giai cao
```

---

## 3. Tech Stack

### 3.1 Moi truong mo phong

| Thanh phan | Phien ban | Vai tro |
|-----------|---------|--------|
| CARLA Simulator | 0.9.9.4 | Moi truong thanh pho ao, cung cap camera, xe co, nguoi di bo |
| Unreal Engine 4 | (tich hop trong CARLA) | Render do hoa 3D |

### 3.2 AI Pipeline

| Thanh phan | Phien ban / Chi tiet | Vai tro |
|-----------|---------------------|--------|
| Python | 3.7+ | Ngon ngu lap trinh chinh |
| PyTorch | >= 1.10.0 | Framework deep learning |
| YOLOv5s | v6.1 (pretrained COCO) | Phat hien doi tuong (person, car, bus, truck) |
| OSNet | x1_0 (pretrained Market-1501, qua torchreid) | Trich xuat dac trung ReID cho nhan dien lai |
| ResNet50 | (pretrained ImageNet, fallback) | Trich xuat dac trung ReID khi khong co torchreid |
| OpenCV | 4.5.4 | Xu ly anh, ve visualization, encode MJPEG |
| NumPy | 1.21.0 | Tinh toan ma tran, xu ly du lieu |
| SciPy | 1.7.2 | Tinh khoang cach, thuat toan matching |

### 3.3 Backend Server

| Thanh phan | Phien ban | Vai tro |
|-----------|---------|--------|
| FastAPI | >= 0.100.0 | Web framework (REST API + WebSocket) |
| Uvicorn | >= 0.23.0 | ASGI server chay FastAPI |
| SQLAlchemy | >= 2.0 | ORM tuong tac voi database |
| SQLite | (tich hop Python) | Database luu camera, alerts, tracks, ROIs |
| Pydantic | >= 2.0 | Validate du lieu request/response |
| WebSocket | (tich hop FastAPI) | Push du lieu real-time den client |

### 3.4 Giao thuc truyen du lieu

| Giao thuc | Su dung cho |
|----------|------------|
| REST API (HTTP) | CRUD camera, query alerts/tracks, thong ke |
| WebSocket | Push canh bao moi, cap nhat tracking real-time |
| MJPEG over HTTP | Truyen video live tu camera den browser |

---

## 4. Mo Ta Chi Tiet Cac Thanh Phan

### 4.1 AI Pipeline — Luong xu ly chinh

```
Video Source (CARLA / RTSP / File / Webcam)
        |  chi tra ve raw frames (numpy array)
        v
  [1] Object Detection (YOLOv5s)
        |  Output: bounding boxes + class + confidence
        v
  [2] Single-Camera Tracking (IoU Greedy Matching)
        |  Output: local track ID trong 1 camera
        v
  [3] Feature Extraction (OSNet / ResNet50)
        |  Output: vector dac trung 512D hoac 2048D
        v
  [4] Cross-Camera ReID (Cosine Similarity + Gallery)
        |  Output: matched global ID hoac tao ID moi
        v
  [5] Global Tracking (Global ID Assignment)
        |  Output: global ID + lich su camera
        v
  [6] Trajectory Prediction (Linear Extrapolation)
        |  Output: vi tri du doan tuong lai (5 buoc)
        v
  [7] Alert Check (Point-in-Polygon voi ROI)
        |  Output: canh bao neu doi tuong sap vao vung ROI
        v
  [8] Visualization + Streaming
        Output: frame da ve overlay -> MJPEG stream / OpenCV window
```

### 4.2 Backend Server — Kien truc

```
Client (Browser / App)
        |
   HTTP / WebSocket
        |
        v
  +-- FastAPI Server (app.py) ----------------------+
  |                                                   |
  |  REST API:                                        |
  |    /api/cameras   — Quan ly camera                |
  |    /api/tracks    — Query doi tuong               |
  |    /api/alerts    — Query + xac nhan canh bao     |
  |    /api/rois      — Quan ly vung ROI              |
  |    /api/stats     — Thong ke he thong             |
  |                                                   |
  |  WebSocket:                                       |
  |    /ws/alerts     — Push canh bao real-time       |
  |    /ws/tracks     — Push tracking updates         |
  |    /ws/stats      — Push system stats moi 1s      |
  |                                                   |
  |  Stream:                                          |
  |    /stream/{cam}  — MJPEG video live              |
  |                                                   |
  |  Services:                                        |
  |    ai_processor   — Chay AI pipeline (bg thread)  |
  |    camera_service — CRUD camera trong DB          |
  |    alert_service  — Loc, dem, xac nhan alerts     |
  |    tracking_svc   — Query tracks + trajectory     |
  |    stream_service — Buffer frame cho MJPEG        |
  |                                                   |
  |  Database (SQLite):                               |
  |    cameras, alerts, tracked_objects,               |
  |    tracking_history, rois                          |
  +---------------------------------------------------+
        |
   Background Thread
        |
        v
  AI Pipeline (detector -> tracker -> reid -> global -> predict -> alert)
        |
        v
  Video Source (CARLA Simulator / Camera IP / Video File / Webcam)
        |
  Ground Truth (CARLA actor data / MOT file) — TACH BIET, chi dung evaluation
```

### 4.3 Database — 5 bang chinh

| Bang | Muc dich | Cac truong chinh |
|------|---------|-----------------|
| `cameras` | Luu thong tin camera | id, name, position, rotation, resolution, fov, status |
| `alerts` | Luu lich su canh bao | type, severity, global_id, camera_id, roi_name, status, timestamp |
| `tracked_objects` | Luu doi tuong da theo doi | global_id, object_class, first_seen, last_seen, total_cameras |
| `tracking_history` | Luu lich su vi tri doi tuong | global_id, camera_id, frame_number, box, center, timestamp |
| `rois` | Luu vung quan tam | camera_id, name, polygon (JSON), alert_types, is_active |

---

## 5. Cach Su Dung

### 5.1 Yeu cau he thong

| Thanh phan | Yeu cau toi thieu |
|-----------|-------------------|
| OS | Windows 10/11 64-bit |
| CPU | Intel i5 / AMD Ryzen 5 |
| RAM | 16 GB |
| GPU | NVIDIA GTX 1060 (6GB VRAM) — ho tro CUDA |
| Disk | 50 GB trong |
| Python | 3.7+ |

### 5.2 Cai dat

#### Buoc 1: Cai dat dependencies cho AI Pipeline

```bash
cd AI_custom/custom_tracking_system
pip install -r requirements.txt
```

Luu y:
- PyTorch can cai rieng theo phien ban CUDA cua may. Xem huong dan tai https://pytorch.org
- `torchreid` co the can cai tu source:
  ```bash
  git clone https://github.com/KaiyangZhou/deep-person-reid.git
  cd deep-person-reid
  pip install -r requirements.txt
  python setup.py develop
  ```

#### Buoc 2: Cai dat dependencies cho Backend Server

```bash
cd AI_custom/server
pip install -r requirements.txt
```

#### Buoc 3: Them CARLA Python API vao PYTHONPATH

```bash
set PYTHONPATH=%PYTHONPATH%;E:\finalproject\WindowsNoEditor\PythonAPI
set PYTHONPATH=%PYTHONPATH%;E:\finalproject\WindowsNoEditor\PythonAPI\carla\dist\carla-0.9.9-py3.7-win-amd64.egg
```

### 5.3 Chay he thong

#### Che do 1: Chi chay API Server (khong can CARLA, dung de phat trien frontend)

```bash
cd AI_custom/server
python app.py
```

Sau khi chay:
- API docs (Swagger UI): http://localhost:8000/docs
- Trang thai server: http://localhost:8000/

Trong che do nay, API server hoat dong binh thuong nhung khong co du lieu AI.
Co the su dung Swagger UI de tao camera, ROI, test cac endpoint thu cong.

#### Che do 2: Chay API Server + AI Pipeline (can CARLA)

```bash
# Terminal 1: Khoi dong CARLA server
cd WindowsNoEditor
CarlaUE4.exe -windowed -ResX=800 -ResY=600

# Terminal 2: Chay API server kem AI pipeline
cd AI_custom/server
python app.py --with-ai
```

Sau khi chay:
- API docs: http://localhost:8000/docs
- Trang thai + AI info: http://localhost:8000/
- Video live camera: http://localhost:8000/stream/CAM_001
- Danh sach streams: http://localhost:8000/stream/

#### Che do 3: Chay AI Pipeline truc tiep (khong qua server, dung OpenCV window)

```bash
# Terminal 1: Khoi dong CARLA server
cd WindowsNoEditor
CarlaUE4.exe

# Terminal 2: Chay pipeline truc tiep
cd AI_custom/custom_tracking_system
python main.py --config config/camera_config.yaml --max-frames 1000
```

Che do nay hien thi ket qua qua cua so OpenCV, khong co API.

### 5.4 API Endpoints

#### Camera

| Method | URL | Mo ta |
|--------|-----|-------|
| GET | `/api/cameras` | Lay danh sach tat ca camera |
| GET | `/api/cameras/{id}` | Lay thong tin 1 camera |
| POST | `/api/cameras` | Them camera moi |
| PUT | `/api/cameras/{id}` | Cap nhat camera |
| DELETE | `/api/cameras/{id}` | Xoa camera |

#### Tracking

| Method | URL | Mo ta |
|--------|-----|-------|
| GET | `/api/tracks` | Danh sach doi tuong dang theo doi |
| GET | `/api/tracks/{global_id}` | Thong tin chi tiet 1 doi tuong |
| GET | `/api/tracks/{global_id}/trajectory` | Quy dao di chuyen cua doi tuong |

Query params cho `/api/tracks`:
- `status` — loc theo trang thai (`active`, `lost`)
- `object_class` — loc theo loai (`car`, `person`, `bus`, `truck`)
- `limit`, `offset` — phan trang

#### Alerts

| Method | URL | Mo ta |
|--------|-----|-------|
| GET | `/api/alerts` | Danh sach canh bao |
| GET | `/api/alerts/{id}` | Chi tiet 1 canh bao |
| PUT | `/api/alerts/{id}/acknowledge` | Xac nhan da xu ly canh bao |

Query params cho `/api/alerts`:
- `camera_id` — loc theo camera
- `type` — loc theo loai (`ROI_WARNING`, ...)
- `severity` — loc theo muc do (`info`, `warning`, `critical`)
- `status` — loc theo trang thai (`new`, `acknowledged`)
- `hours` — chi lay alerts trong N gio gan nhat

#### ROI (Region of Interest)

| Method | URL | Mo ta |
|--------|-----|-------|
| GET | `/api/rois` | Danh sach vung ROI |
| GET | `/api/rois/{id}` | Chi tiet 1 ROI |
| POST | `/api/rois` | Them vung ROI moi |
| PUT | `/api/rois/{id}` | Cap nhat ROI |
| DELETE | `/api/rois/{id}` | Xoa ROI |

#### Thong ke

| Method | URL | Mo ta |
|--------|-----|-------|
| GET | `/api/stats` | FPS, so camera, so doi tuong, so alert hom nay, uptime |

#### WebSocket

| URL | Mo ta |
|-----|-------|
| `ws://localhost:8000/ws/alerts` | Nhan canh bao moi ngay khi xay ra |
| `ws://localhost:8000/ws/tracks` | Nhan cap nhat tracking real-time |
| `ws://localhost:8000/ws/stats` | Nhan thong ke he thong moi giay |

#### Video Streaming

| URL | Mo ta |
|-----|-------|
| `GET /stream/` | Danh sach camera dang co stream |
| `GET /stream/{camera_id}` | MJPEG video stream cua 1 camera |

Su dung trong HTML:
```html
<img src="http://localhost:8000/stream/CAM_001" />
```

### 5.5 Cau hinh Camera

File `custom_tracking_system/config/camera_config.yaml`:

```yaml
cameras:
  camera_0:
    position: [0, 0, 3]        # Vi tri trong CARLA (x, y, z met)
    rotation: [0, 0, 0]        # Huong nhin (pitch, yaw, roll do)
    camera_id: "CAM_001"       # Ma dinh danh camera
    view_angle: 90             # Goc nhin (do)
    resolution: [960, 540]     # Do phan giai
    fps: 10                    # Tan suat khung hinh

rois:
  camera_0:
    zones:
      - name: "intersection_main"
        polygon: [[100, 100], [500, 100], [500, 400], [100, 400]]

system:
  synchronous_mode: true
  fixed_delta_seconds: 0.1
  reid_threshold: 0.5
  trajectory_window: 10
  prediction_steps: 5
```

Hien tai he thong cau hinh 3 camera (CAM_001, CAM_002, CAM_003) voi 3 vung ROI tuong ung.

---

## 6. Trang Thai Hien Tai Va Han Che

### 6.1 Da hoan thanh

- AI pipeline day du 8 buoc: detect -> track -> ReID -> global ID -> predict -> alert -> visualize -> stream.
- Backend API server voi 17 REST endpoints, 3 WebSocket channels, MJPEG streaming.
- Database 5 bang luu tru camera, alerts, tracks, history, ROIs.
- AI processor tich hop pipeline vao server, chay trong background thread.
- 3 che do chay: API-only, API + AI (CARLA), Direct (OpenCV window).
- Abstract VideoSource layer — tach AI pipeline khoi nguon video cu the (CARLA/RTSP/File/Webcam).
- Ground Truth module — thu thap du lieu CARLA (actor_id, location, velocity) TACH BIET khoi AI pipeline, chi dung cho evaluation. Ho tro doc file MOT Challenge format.

### 6.2 Han che hien tai

| Han che | Chi tiet |
|--------|---------|
| Tracker don gian | Su dung IoU greedy matching, chua co Kalman filter hoac appearance matching. Khi 2 doi tuong di sat nhau roi tach ra, tracker co the hoan doi ID |
| Trajectory prediction co ban | Chi dung linear extrapolation (noi suy tuyen tinh), chua co Kalman filter hoac LSTM |
| ReID chua toi uu cho vehicle | OSNet/Market-1501 duoc train cho person, chua co model rieng cho xe. Voi 2 xe giong mau va hinh dang, ReID khong phan biet duoc |
| Chua co spatio-temporal reasoning | Chua dung thoi gian + khoang cach giua camera de loc match sai khi ReID khong du chinh xac |
| Chua co web dashboard | Frontend chua duoc xay dung, chi co API |
| Chua co anomaly detection | Chi phat hien doi tuong vao ROI, chua co overspeed, wrong-way, crowd, loitering... |
| Chua co recording/playback | Chua ghi video, chua cat clip su co |
| Chua co ground truth evaluation | Module ground_truth.py da co nhung chua tich hop de tinh MOTA, IDF1, mAP thuc su |
| Datasets trong | Chua co du lieu ground truth hoac synthetic data |
| VideoSource chua tich hop | video_source.py da viet nhung main.py va ai_processor.py van dung camera_controller truc tiep |
| CARLA khong ho tro bien so xe | Khong the dung LPR (phuong phap chinh xac nhat) de phan biet xe giong nhau |
