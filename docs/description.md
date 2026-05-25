# Mo Ta Cau Truc Du An — Multi-Camera CCTV Tracking System

## Tong quan

Du an he thong giam sat camera CCTV da kenh su dung AI, chay trong moi truong mo phong
CARLA 0.9.9.4. He thong phat hien, theo doi, nhan dien lai doi tuong (xe, nguoi) va
canh bao khi doi tuong vao vung ROI.

He thong duoc thiet ke de hoat dong voi **bat ky nguon video nao** (CARLA, camera IP,
video file, webcam) thong qua lop abstract VideoSource.

## Cau truc thu muc

```
E:\finalproject\
|
|-- AI_custom\                              # Phan code tu phat trien
|   |
|   |-- custom_tracking_system\             # AI Pipeline
|   |   |-- config\
|   |   |   +-- camera_config.yaml          # Cau hinh 3 camera + 3 ROI + system params
|   |   |-- modules\
|   |   |   |-- camera_controller.py        # Dieu khien camera trong CARLA, dong bo frame
|   |   |   |-- traffic_generator.py        # Spawn xe + nguoi voi autopilot trong CARLA
|   |   |   |-- detector.py                # Phat hien doi tuong (YOLOv5s, COCO pretrained)
|   |   |   |   |-- tracker.py                 # Theo doi trong 1 camera (IoU + timestamp/speed history)
|   |   |   |-- reid.py                    # ReID xuyen camera (OSNet/torchreid + ResNet50 fallback)
|   |   |   |-- global_tracking.py         # Gan Global ID xuyen camera bang ReID gallery
|   |   |   |-- trajectory_predictor.py    # Du doan quy dao (linear extrapolation)
|   |   |   |-- alert_system.py            # Canh bao vung ROI (point-in-polygon)
|   |   |   |-- incident_detector.py       # [MOI] Phat hien 10 loai su co real-time
|   |   |   |-- evidence_package.py        # [MOI] Ring buffer + luu clip/anh/JSON khi su co
|   |   |   |-- scenario_controller.py     # [MOI] Script kich ban tai nan trong CARLA
|   |   |   |-- video_source.py            # Abstract video source (CARLA/RTSP/File/Webcam)
|   |   |   +-- ground_truth.py            # Ground truth cho evaluation (tach biet khoi AI)
|   |   |-- utils\
|   |   |   |-- visualization.py           # Ve bbox, trajectory, ROI, multi-camera grid
|   |   |   |-- metrics.py                 # Thu thap FPS, detection/tracking counts
|   |   |   +-- data_writer.py             # Xuat du lieu JSON, CSV, summary report
|   |   |-- models\
|   |   |   |-- hub\                       # YOLOv5 model cache + ResNet50 checkpoint
|   |   |   |-- detection\                 # (trong — chua fine-tune)
|   |   |   |-- reid\                      # (trong — chua fine-tune)
|   |   |   +-- tracking\                  # (trong — chua fine-tune)
|   |   |-- datasets\
|   |   |   |-- ground_truth\              # (trong — chua co du lieu)
|   |   |   +-- synthetic_data\            # (trong — chua co du lieu)
|   |   |-- main.py                        # Entry point chay truc tiep (khong qua server)
|   |   |-- requirements.txt               # Dependencies: torch, yolov5, opencv, torchreid
|   |   +-- run.bat                        # Script chay tren Windows
|   |
|   |-- server\                             # Backend API Server (FastAPI)
|   |   |-- models\
|   |   |   |-- database.py                # SQLAlchemy ORM, 5 bang, SQLite backend
|   |   |   +-- schemas.py                 # Pydantic v2 request/response schemas
|   |   |-- routers\
|   |   |   |-- cameras.py                 # CRUD camera (GET/POST/PUT/DELETE)
|   |   |   |-- tracks.py                  # Query doi tuong + quy dao
|   |   |   |-- alerts.py                  # Query + filter + xac nhan canh bao
|   |   |   |-- rois.py                    # CRUD vung ROI
|   |   |   |-- stats.py                   # Thong ke he thong (FPS, counts, uptime)
|   |   |   |-- websocket.py               # WebSocket push real-time (alerts, tracks, stats)
|   |   |   +-- stream.py                  # MJPEG video streaming
|   |   |-- services\
|   |   |   |-- ai_processor.py            # Tich hop AI pipeline + IncidentDetector + EvidencePackage vao server
|   |   |   |-- camera_service.py          # CRUD camera trong DB
|   |   |   |-- alert_service.py           # Logic nghiep vu canh bao
|   |   |   |-- tracking_service.py        # Logic nghiep vu tracking
|   |   |   +-- stream_service.py          # FrameBuffer cho MJPEG streaming (async)
|   |   |-- middleware\                     # (du tru — chua implement)
|   |   |-- app.py                         # FastAPI app + CLI entry point
|   |   |-- config.py                      # Paths, CARLA host/port, AI params, streaming params
|   |   |-- requirements.txt               # Dependencies: fastapi, uvicorn, sqlalchemy, pydantic
|   |   +-- run_server.bat                 # Script chay server tren Windows
|   |
|   |-- frontend\                            # [MOI] Web Dashboard (React + Tailwind + Vite)
|   |   |-- src\
|   |   |   |-- pages\
|   |   |   |   |-- Dashboard.jsx           # Trang chinh: camera grid + incident panel
|   |   |   |   |-- IncidentDetail.jsx      # Chi tiet 1 su co
|   |   |   |   |-- AlertManagement.jsx     # Lich su + filter toan bo alerts
|   |   |   |   +-- ObjectHistory.jsx       # Lich su di chuyen 1 doi tuong
|   |   |   |-- components\
|   |   |   |   |-- CameraGrid.jsx          # Grid nhieu camera
|   |   |   |   |-- CameraFeed.jsx          # 1 camera: MJPEG + incident badge
|   |   |   |   |-- IncidentPanel.jsx       # Panel su co real-time
|   |   |   |   |-- IncidentCard.jsx        # 1 su co: severity + actions
|   |   |   |   +-- StatsBar.jsx            # Thanh thong ke FPS/objects/alerts
|   |   |   |-- hooks\
|   |   |   |   |-- useWebSocket.js         # WS hook voi auto-reconnect
|   |   |   |   +-- useIncidents.js         # Incident state + am thanh + flash
|   |   |   +-- services\
|   |   |       +-- api.js                  # REST API calls
|   |   |-- index.html
|   |   |-- package.json                    # React 18, Tailwind 3, Vite 5
|   |   +-- vite.config.js                  # Proxy /api /ws /stream -> :8000
|   |
|   +-- docs\                               # Tai lieu du an
|       |-- description.md                 # Mo ta cau truc du an (file nay)
|       |-- plan.md                        # Ke hoach phat trien 10 phases (ban goc)
|       |-- workflow.md                    # Quy trinh trien khai 6 giai doan
|       |-- execute.md                     # Huong dan chay + 6 kich ban test
|       |-- development_roadmap.md         # Lo trinh phat trien
|       |-- report.md                      # Bao cao du an
|       |-- project_context.md             # Context doc cho prompt tren mobile
|       +-- upgrade.md                     # [MOI] Ke hoach nang cap huong thuc te
|
+-- WindowsNoEditor\                        # CARLA Simulator 0.9.9.4 (khong chinh sua)
    |-- CarlaUE4.exe                       # File thuc thi CARLA server
    |-- PythonAPI\
    |   |-- carla\                         # CARLA Python package (.egg)
    |   |-- examples\                      # Vi du su dung CARLA API
    |   +-- util\                          # Tien ich
    |-- Engine\                            # Unreal Engine 4 runtime
    |-- Co-Simulation\                     # Tich hop SUMO, PTV-Vissim
    +-- HDMaps\                            # Ban do do phan giai cao
```

## Mo ta cac thanh phan chinh

### AI Pipeline (`custom_tracking_system/`)

Luong xu ly 11 buoc (cap nhat 2026-05-24):
1. **Camera Source** — lay frame tu CARLA (hoac RTSP/File/Webcam qua VideoSource)
2. **Detection** — YOLOv5s phat hien person, car, bus, truck
3. **Tracking** — IoU matching + timestamp + speed history (px/s) moi frame
4. **Feature Extraction** — OSNet (512D) hoac ResNet50 (2048D), L2-normalized
5. **ReID Matching** — cosine similarity voi gallery, threshold 0.5
6. **Global ID** — gan ID duy nhat xuyen camera
7. **Trajectory Prediction** — linear extrapolation 5 buoc tuong lai
8. **Alert** — point-in-polygon kiem tra doi tuong vao vung ROI
9. **[MOI] Incident Detection** — phat hien 10 loai su co: sudden stop, fleeing, overspeed, proximity, pedestrian danger, stopped vehicle, loitering, crowd, wrong way, camera transition
10. **[MOI] Evidence Capture** — ring buffer 30s; khi CRITICAL: luu clip truoc/sau + crop anh + metadata JSON
11. **[MOI] WebSocket Push** — day incident len /ws/alerts; Dashboard nhan va hien thi ngay lap tuc

**video_source.py**: Abstract layer tach AI pipeline khoi nguon video cu the.
AI pipeline chi thay raw frames (numpy array), khong biet nguon la CARLA hay camera that.
Ho tro: CARLAVideoSource, RTSPVideoSource, FileVideoSource, WebcamVideoSource.

**ground_truth.py**: Thu thap ground truth tu CARLA (actor_id, location, velocity)
hoac tu file MOT format. TACH BIET hoan toan khoi AI pipeline, chi dung cho evaluation.

### Backend Server (`server/`)

- **17 REST endpoints**: CRUD camera, query tracks/alerts/ROIs, system stats
- **3 WebSocket channels**: push alerts, tracking updates, stats real-time
- **MJPEG streaming**: video live tu camera den browser
- **Database**: SQLite voi 5 bang (cameras, alerts, tracked_objects, tracking_history, rois)
- **AI Processor**: chay AI pipeline trong background thread, push ket qua qua WebSocket + DB

### 3 che do chay

1. **API-only** (`python app.py`): Server hoat dong, khong can CARLA. Dung de phat trien frontend.
2. **API + AI** (`python app.py --with-ai`): Full system voi CARLA. AI pipeline chay background.
3. **Direct** (`python main.py`): Hien thi qua OpenCV window, khong co server.
