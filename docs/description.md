# Mo Ta Cau Truc Du An — Multi-Camera CCTV Tracking System

## Tong quan

Du an he thong giam sat camera CCTV da kenh su dung AI, chay trong moi truong mo phong
CARLA Simulator. He thong phat hien, theo doi, nhan dien lai doi tuong (xe, nguoi) va
canh bao khi doi tuong vao vung cam hoac xay ra su co.

He thong duoc thiet ke de hoat dong voi **bat ky nguon video nao** (CARLA, camera IP,
video file, webcam) thong qua lop abstract VideoSource.

> **Luu y moi truong (2026-06-10):** CARLA 0.9.9.4 yeu cau Python 3.7, nhung
> `boxmot>=10.0` (ByteTrack) va `ultralytics>=8.0` (YOLOv8) yeu cau Python 3.8+.
> Da giai quyet bang kien truc **carla_bridge**: 1 process Python 3.7 chay
> CARLA PythonAPI (`carla_bridge/server.py`) stream frame qua TCP socket cho
> 1 process Python 3.10+ chay AI pipeline (`main.py --source bridge`). Xem chi
> tiet tai `docs/execute.md` phan 2 va `docs/bao_cao_tien_do.md` muc 7 item 2.

## Cau truc thu muc

```
E:\School_project\finalproject\
|
|-- custom_tracking_system\             # AI Pipeline
|   |-- carla_bridge\
|   |   +-- server.py                   # Process Python 3.7: chay CARLA PythonAPI, stream frame qua TCP
|   |-- config\
|   |   +-- camera_config.yaml          # Cau hinh 3 camera + 3 ROI + camera_topology
|   |-- modules\
|   |   |-- camera_controller.py        # Dieu khien camera trong CARLA, dong bo frame
|   |   |-- traffic_generator.py        # Spawn xe + nguoi voi autopilot trong CARLA
|   |   |-- detector.py                 # Phat hien doi tuong (YOLOv8s, FP16, 5 class)
|   |   |-- tracker.py                  # ByteTrackWrapper: Kalman Filter + Hungarian + dual-threshold
|   |   |-- reid.py                     # DualReIDExtractor: OSNet Market-1501 (person) + VeRi-776 (vehicle)
|   |   |-- global_tracking.py          # Global ID xuyen camera + Spatio-Temporal Filter
|   |   |-- trajectory_predictor.py     # Kalman Filter (constant-acceleration), 5 horizons (0.5s-3.0s)
|   |   |-- alert_system.py             # ROI warning (point-in-polygon)
|   |   |-- incident_detector.py        # 12 loai su co: 10 reactive + 2 proactive
|   |   |-- evidence_package.py         # Ring buffer 30s, luu clip/anh/JSON khi CRITICAL
|   |   |-- scenario_controller.py      # 5 kich ban tai nan trong CARLA
|   |   |-- video_source.py             # Abstract video source (CARLA/RTSP/File/Webcam/CARLABridgeClient)
|   |   +-- ground_truth.py             # Ground truth cho evaluation (tach biet khoi AI)
|   |-- utils\
|   |   |-- visualization.py            # Ve bbox, trajectory, ROI, multi-camera grid
|   |   |-- metrics.py                  # Thu thap FPS, detection/tracking counts
|   |   +-- data_writer.py              # Xuat du lieu JSON, CSV, summary report
|   |-- models\
|   |   |-- detection\                  # (trong — chua fine-tune)
|   |   |-- reid\
|   |   |   +-- osnet_veri776.pth       # VeRi-776 weights (20 checkpoints ep01-ep20)
|   |   +-- tracking\                   # (trong)
|   |-- datasets\                       # (trong — chua co du lieu)
|   |-- train_veri.py                   # Script train VeRi-776 (da sua bug validation)
|   |-- requirements.txt                # ultralytics, boxmot, torchreid, opencv, ...
|   +-- run.bat                         # Script chay tren Windows (legacy)
|
|-- server\                             # Backend API Server (FastAPI)
|   |-- models\
|   |   |-- database.py                 # SQLAlchemy ORM, 5 bang, SQLite backend
|   |   +-- schemas.py                  # Pydantic v2 request/response schemas
|   |-- routers\
|   |   |-- cameras.py                  # CRUD camera (GET/POST/PUT/DELETE)
|   |   |-- tracks.py                   # Query doi tuong + quy dao
|   |   |-- alerts.py                   # Query + filter + xac nhan canh bao
|   |   |-- rois.py                     # CRUD vung ROI
|   |   |-- stats.py                    # Thong ke he thong (FPS, counts, uptime)
|   |   |-- websocket.py                # WebSocket push real-time (alerts, tracks, stats)
|   |   +-- stream.py                   # MJPEG video streaming
|   |-- services\
|   |   |-- ai_processor.py             # Tich hop AI pipeline + IncidentDetector + EvidencePackage
|   |   |-- camera_service.py           # CRUD camera trong DB
|   |   |-- alert_service.py            # Logic nghiep vu canh bao
|   |   |-- tracking_service.py         # Logic nghiep vu tracking
|   |   +-- stream_service.py           # FrameBuffer cho MJPEG streaming (async, thread-safe)
|   |-- middleware\                      # (du tru — chua implement)
|   |-- app.py                          # FastAPI app + CLI entry point (--with-ai flag)
|   |-- config.py                       # Paths, CARLA host/port, AI params, streaming params
|   |-- requirements.txt                # fastapi, uvicorn, sqlalchemy, pydantic
|   +-- run_server.bat                  # Script chay server tren Windows
|
|-- frontend\                           # Web Dashboard (React + Tailwind + Vite)
|   |-- src\
|   |   |-- pages\
|   |   |   |-- Dashboard.jsx           # Trang chinh: camera grid + incident panel
|   |   |   |-- IncidentDetail.jsx      # Chi tiet 1 su co
|   |   |   |-- AlertManagement.jsx     # Lich su + filter toan bo alerts
|   |   |   +-- ObjectHistory.jsx       # Lich su di chuyen 1 doi tuong
|   |   |-- components\
|   |   |   |-- CameraGrid.jsx          # Grid nhieu camera (MJPEG live feed)
|   |   |   |-- CameraFeed.jsx          # 1 camera: MJPEG + incident badge
|   |   |   |-- IncidentPanel.jsx       # Panel su co real-time
|   |   |   |-- IncidentCard.jsx        # 1 su co: severity + actions
|   |   |   +-- StatsBar.jsx            # Thanh thong ke FPS/objects/alerts (WebSocket /ws/stats)
|   |   |-- hooks\
|   |   |   |-- useWebSocket.js         # WS hook voi auto-reconnect
|   |   |   +-- useIncidents.js         # Incident state + am thanh + flash CRITICAL
|   |   +-- services\
|   |       +-- api.js                  # REST API calls
|   |-- index.html
|   |-- package.json                    # React 18, Tailwind 3, Vite 5, Recharts 2
|   +-- vite.config.js                  # Proxy /api /ws /stream -> :8000
|
|-- docs\                               # Tai lieu du an
|   |-- description.md                  # Mo ta cau truc du an (file nay)
|   |-- execute.md                      # Huong dan cai dat va chay
|   |-- workflow.md                     # Quy trinh trien khai
|   |-- development_roadmap.md          # Lo trinh phat trien (chi tiet)
|   |-- bao_cao_tien_do.md              # Bao cao tien do chinh (tieng Viet)
|   +-- upgrade.md                      # Ke hoach nang cap
|
|-- WindowsNoEditor\                    # CARLA Simulator (khong chinh sua)
|   |-- CarlaUE4.exe                    # File thuc thi CARLA server
|   |-- PythonAPI\
|   |   |-- carla\                      # CARLA Python package (.egg) — Python 3.7
|   |   +-- examples\
|   +-- ...
|
|-- start.bat                           # Khoi dong toan bo he thong (CARLA + server + frontend)
+-- stop.bat                            # Dung toan bo he thong
```

## Mo ta cac thanh phan chinh

### AI Pipeline (`custom_tracking_system/`)

Luong xu ly 11 buoc:
1. **Camera Source** — lay frame tu CARLA truc tiep, tu carla_bridge (TCP, qua CARLABridgeClient), hoac RTSP/File/Webcam qua VideoSource
2. **Detection** — YOLOv8s phat hien person, car, motorcycle, bus, truck (FP16, conf>=0.35, batched 6 cam)
3. **Tracking** — `ByteTrackWrapper` (boxmot.ByteTrack): Kalman Filter + Hungarian matching + dual-threshold
4. **Feature Extraction** — `DualReIDExtractor`: OSNet Market-1501 (512D, person) hoac OSNet VeRi-776 (vehicle)
5. **ReID Matching** — cosine similarity voi gallery, L2-normalized, threshold 0.5
6. **Global ID** — gan ID duy nhat xuyen camera qua `GlobalTracker`
7. **Spatio-Temporal Filter** — `_is_feasible_match()` loc match phi thuc te dua tren `camera_topology`
8. **Trajectory Prediction** — `TrajectoryPredictor`: Kalman Filter (constant-acceleration, dt-aware), 5 horizons t+0.5s..t+3.0s
9. **Alert** — `AlertSystem`: point-in-polygon kiem tra doi tuong vao vung ROI
10. **Incident Detection** — `IncidentDetector`: 12 loai su co (10 reactive + 2 proactive: PREDICTED_COLLISION, PREDICTED_ROI_ENTRY); cooldown 4s
11. **Evidence Capture** — `EvidencePackage`: ring buffer 30s; khi CRITICAL: luu clip truoc/sau + crop anh + metadata JSON

**video_source.py**: Abstract layer tach AI pipeline khoi nguon video cu the.
Ho tro: CARLAVideoSource, RTSPVideoSource, FileVideoSource, WebcamVideoSource,
CARLABridgeClient (nhan frame dong bo tu carla_bridge/server.py qua TCP socket).
`main.py --source bridge` da dung CARLABridgeClient; server `ai_processor.py`
van dung CameraController truc tiep — tich hop VideoSource vao server la viec con lai.

**carla_bridge/server.py**: Process Python 3.7 doc lap, dung CARLA `.egg` 0.9.9.4
hien co. Ket noi CARLA, bat synchronous mode (10 FPS), tao CameraController +
TrafficGenerator, va moi tick gui frame dong bo cua tat ca camera qua TCP socket
(mac dinh `127.0.0.1:8765`) cho `main.py --source bridge` (Python 3.10+). Day la
giai phap cho xung dot Python 3.7 (CARLA) vs Python 3.8+ (boxmot/ultralytics) —
xem `docs/bao_cao_tien_do.md` muc 7 item 2.

**ground_truth.py**: Thu thap ground truth tu CARLA (actor_id, location, velocity).
Tach biet hoan toan khoi AI pipeline, chi dung cho evaluation (motmetrics MOTA/IDF1).

### Backend Server (`server/`)

- **17 REST endpoints**: CRUD camera, query tracks/alerts/ROIs, system stats
- **3 WebSocket channels**: `/ws/alerts` (incident real-time), `/ws/tracks` (tracking updates), `/ws/stats` (fps/active_tracks/alerts_today moi 1s)
- **MJPEG streaming**: `/stream/{camera_id}` — annotated frame den browser
- **Database**: SQLite voi 5 bang (cameras, alerts, tracked_objects, tracking_history, rois)
- **AI Processor**: chay AI pipeline trong background thread; push ket qua qua WebSocket + DB (thread-safe qua asyncio.run_coroutine_threadsafe va call_soon_threadsafe)

### 3 che do chay

1. **API-only** (`python app.py`): Server hoat dong, khong can CARLA. Dung de phat trien frontend.
2. **API + AI** (`python app.py --with-ai`): Full system voi CARLA. AI pipeline chay background.
3. **start.bat**: Khoi dong tu dong: CARLA → doi port 2000 → server `--with-ai` → frontend dev server.
