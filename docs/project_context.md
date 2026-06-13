# Project Context — Multi-Camera CCTV Tracking System

> Tai lieu nay dung de dan vao dau cuoc hoi thoai voi Claude tren mobile
> hoac bat ky moi truong nao khong doc duoc folder du an.
> Copy toan bo noi dung file nay va paste vao prompt dau tien.

---

## Tong quan

Day la du an he thong giam sat camera CCTV da kenh, su dung AI de phat hien,
theo doi va nhan dien lai doi tuong (xe, nguoi) trong moi truong mo phong CARLA 0.9.9.4.

He thong gom 2 phan chinh:
- **AI Pipeline** — xu ly video: detect (YOLO11m fine-tune VN) -> track (ByteTrack/Kalman) -> ReID (OSNet) -> global ID (+ Spatio-Temporal Filter) -> predict trajectory (Kalman, hoac Kalman+GRU Ensemble + calibration trong main.py) -> incident detection (14 loai) -> alert ROI.
- **Backend API Server** — FastAPI cung cap REST API, WebSocket real-time, MJPEG video streaming, scenario trigger. AI pipeline chay trong background thread cua server.

Da co frontend/web dashboard (React) voi camera grid, incident panel, alert management, scenario panel.

---

## Cau truc thu muc hien tai

```
E:\School_project\finalproject\
  custom_tracking_system\           # AI Pipeline
    config\
      camera_config.yaml            # 3 cameras (CAM_001/002/003), 3 ROIs, system params
    modules\
      camera_controller.py          # Dat camera trong CARLA, dong bo frame
      traffic_generator.py          # Spawn xe + nguoi voi autopilot
      detector.py                   # YOLOv8s pretrained COCO (person,car,motorcycle,bus,truck) — detect_batch()
      tracker.py                    # ByteTrackWrapper (boxmot.ByteTrack, Kalman + Hungarian) + SimpleTracker legacy
      reid.py                       # ReIDExtractor (OSNet Market-1501) + DualReIDExtractor (person + vehicle)
      global_tracking.py            # Gan Global ID xuyen camera + Spatio-Temporal Filter
      trajectory_predictor.py       # TrajectoryPredictor (Kalman constant-accel) + EnsembleTrajectoryPredictor (Kalman+GRU), horizons 0.5-3s
      calibration.py                # CameraCalibration/CalibrationStore — pixel<->world (CARLA intrinsics hoac homography)
      alert_system.py               # Canh bao khi doi tuong vao ROI (point-in-polygon)
      incident_detector.py          # Phat hien 14 loai su co (12 reactive + 2 proactive), bao gom WRONG_WAY, RED_LIGHT_VIOLATION
      evidence_package.py           # Ring buffer 30s — luu clip + crop + JSON khi su co
      scenario_controller.py        # 5 kich ban tai nan trong CARLA, noi qua server/routers/scenarios.py
      video_source.py               # Abstract video source (CARLA/RTSP/File/Webcam)
      ground_truth.py               # Ground truth cho evaluation (tach biet khoi AI pipeline)
      evaluate_tracking.py          # MOTA/IDF1 evaluation qua py-motmetrics
    utils\
      visualization.py              # Ve bbox, trajectory, ROI, multi-camera grid (OpenCV)
      metrics.py                    # Thu thap FPS, counts
      data_writer.py                # Export JSON, CSV, summary report
    models\                         # (chua fine-tune)
    weights\                        # Weights sau khi fine-tune (VeRi-776, etc.)
    train_veri.py                   # Script fine-tune OSNet tren VeRi-776 (Vehicle ReID)
    main.py                         # Entry point chay truc tiep (khong qua server)
    requirements.txt
    run.bat

  server\                           # Backend API Server (FastAPI)
    app.py                          # FastAPI app, lifespan khoi dong AI, CLI entry point
    config.py                       # Paths, CARLA host/port, streaming params
    models\
      database.py                   # SQLAlchemy ORM, 5 bang, SQLite backend
      schemas.py                    # Pydantic request/response schemas
    routers\
      cameras.py                    # GET/POST/PUT/DELETE /api/cameras
      tracks.py                     # GET /api/tracks, /api/tracks/{id}/trajectory
      alerts.py                     # GET /api/alerts, PUT /api/alerts/{id}/acknowledge
      rois.py                       # GET/POST/PUT/DELETE /api/rois
      stats.py                      # GET /api/stats
      websocket.py                  # WS /ws/alerts, /ws/tracks, /ws/stats
      stream.py                     # GET /stream/{camera_id} (MJPEG)
      scenarios.py                  # POST /api/scenarios/{name}/trigger — kich hoat 5 kich ban tai nan
    services\
      ai_processor.py               # Wrap AI pipeline thanh background thread
      camera_service.py             # CRUD camera trong DB
      alert_service.py              # Query + filter + acknowledge alerts
      tracking_service.py           # Query tracks + trajectory + upsert
      stream_service.py             # FrameBuffer singleton cho MJPEG streaming
    requirements.txt
    run_server.bat

  frontend\                         # Web Dashboard (React + Tailwind + Vite)
    src\                            # Components, pages, hooks, services
    index.html
    package.json
    vite.config.js

  VeRi\                             # VeRi-776 dataset — cho fine-tune Vehicle ReID
  deep-person-reid\                 # torchreid library (cai tu source)

  docs\
    description.md
    plan.md
    workflow.md
    execute.md
    development_roadmap.md
    report.md
    bao_cao_tien_do.md              # Bao cao tien do tong the (doc nay)
    production_model_stack.md       # Phan tich rang buoc phan cung + production stack
    project_context.md              # File nay
    upgrade.md                      # Lo trinh nang cap huong thuc te

  WindowsNoEditor\                   # CARLA 0.9.9.4 (khong chinh sua)
    CarlaUE4.exe
    PythonAPI\carla\
    ...
```

---

## Database Schema (SQLite, file: tracking_system.db)

5 bang:

**cameras**: id(PK,Text), name, position_x/y/z, rotation_pitch/yaw/roll, resolution_w/h, fov, fps, status, created_at, updated_at

**alerts**: id(PK,auto), type(Text), severity(Text), global_id(Int), camera_id(FK), roi_name, message, details(JSON string), snapshot_path, clip_path, status(new/acknowledged/resolved), acknowledged_by, created_at, acknowledged_at

**tracked_objects**: global_id(PK,Int), object_class(Text), first_seen, last_seen, total_cameras, status(active/lost/archived)

**tracking_history**: id(PK,auto), global_id(FK), camera_id(FK), frame_number, box_x1/y1/x2/y2, center_x/y, confidence, timestamp

**rois**: id(PK,auto), camera_id(FK), name, polygon(JSON string), alert_types(Text), is_active(Bool), created_at

---

## API Endpoints hien co

### REST

```
GET    /                              — Server status + AI pipeline status
GET    /docs                          — Swagger UI

GET    /api/cameras                   — List cameras (?status=active)
GET    /api/cameras/{id}              — Camera detail
POST   /api/cameras                   — Create camera
PUT    /api/cameras/{id}              — Update camera
DELETE /api/cameras/{id}              — Delete camera

GET    /api/tracks                    — List tracked objects (?status, ?object_class, ?limit, ?offset)
GET    /api/tracks/{global_id}        — Object detail
GET    /api/tracks/{global_id}/trajectory — Trajectory (?camera_id, ?limit)

GET    /api/alerts                    — List alerts (?camera_id, ?type, ?severity, ?status, ?hours, ?limit, ?offset)
GET    /api/alerts/{id}               — Alert detail
PUT    /api/alerts/{id}/acknowledge   — Acknowledge alert (body: {acknowledged_by: "operator"})

GET    /api/rois                      — List ROIs (?camera_id)
GET    /api/rois/{id}                 — ROI detail
POST   /api/rois                      — Create ROI
PUT    /api/rois/{id}                 — Update ROI
DELETE /api/rois/{id}                 — Delete ROI

GET    /api/stats                     — {fps, active_cameras, active_tracks, total_alerts_today, uptime_seconds}
```

### WebSocket

```
ws://host:8000/ws/alerts    — Push: {event:"alert", data:{...AlertResponse}}
ws://host:8000/ws/tracks    — Push: {event:"track_update", global_id, camera_id, box, object_class, frame_count}
ws://host:8000/ws/stats     — Push: {event:"stats", data:{...SystemStats}}
```

### MJPEG Stream

```
GET /stream/               — List camera streams
GET /stream/{camera_id}    — MJPEG stream (dung trong <img src="...">)
```

---

## AI Pipeline — luong xu ly 1 frame

```
1. camera_controller.get_synchronized_frames()
   -> {camera_id: {frame: np.array(H,W,3), timestamp, frame_number}}

2. detector.detect_batch(frames)   # 1 GPU forward pass cho tat ca camera
   -> [[{box, confidence, class, class_id}, ...], ...]   # per camera

3. tracker[camera_id].update(detections, frame)   # ByteTrackWrapper (Kalman + Hungarian)
   -> [{track_id, box, class, positions, timestamps, speeds}, ...]

4. global_tracker.process_camera_tracks(camera_id, frame, local_tracks)
   # OSNet extract feature -> cosine match -> Spatio-Temporal Filter
   -> [{global_id, camera_id, box, class, local_track_id}]

5. trajectory_predictor.update_trajectory(global_id, center, unix_timestamp)
   trajectory_predictor.predict(global_id)
   -> {'t+0.5s': [x,y], 't+1.0s': [x,y], 't+1.5s': [x,y], 't+2.0s': [x,y], 't+3.0s': [x,y]}
   trajectory_predictor.predict_list(global_id)
   -> [[x,y], [x,y], ...]   # cho AlertSystem

6. alert_system.check_alerts(global_id, camera_id, predicted_positions, current_box)
   -> [{type:"ROI_WARNING", global_id, camera_id, roi_name, timestamp}]

7. incident_detector.update(global_tracks, camera_id, predictions, rois, traffic_light_state)
   # 12 reactive (bao gom WRONG_WAY, RED_LIGHT_VIOLATION) + 2 proactive (PREDICTED_COLLISION, PREDICTED_ROI_ENTRY)
   -> [{type, severity, global_id, camera_id, message, details}]
   # incident_detection.disabled_types (camera_config.yaml) loc bot cac loai
   # qua nhieu trong giao thong binh thuong: SUDDEN_ACCEL, CAMERA_TRANSITION,
   # STOPPED_VEHICLE, VEHICLE_PROXIMITY (van duoc tinh, chi khong xuat ra alert).

8. evidence.capture(incident, frames_dict, global_tracks)   # khi CRITICAL
   # Luu crop anh + clip 30s truoc/sau + metadata JSON -> evidence/<id>/
```

### Scenario controller — camera-aware spawn

`ScenarioController` (modules/scenario_controller.py) nhan `cameras_config`
(tu `camera_config.yaml["cameras"]`) va uu tien chon spawn point nam trong
FOV cua 1 camera (vi tri + yaw + view_angle, range 40m) cho moi kich ban
(hit_and_run, pedestrian_hit, red_light_crash, rear_end, sudden_stop). Moi
ham scenario tra ve `{'ok': bool, 'camera_id': str|None, ...actors}` —
`camera_id` la camera du kien quan sat duoc su kien (None neu khong co
spawn point nao trong tam camera nao, se fallback random).

`POST /api/scenarios/{name}` (routers/scenarios.py -> ai_processor.trigger_scenario)
tra ve them `camera_id` trong response JSON. Frontend (ScenarioPanel.jsx)
hien thi camera nay trong message va Dashboard.jsx highlight camera feed
tuong ung (vien cyan nhap nhay 15s) de nguoi dung biet xem o dau.

---

## Cach chay server

Dung python cua `custom_tracking_system/venv_tracking` (Python 3.10) — co du
`carla` (pip carla==0.9.15, ket noi duoc CARLA server 0.9.14), torch/
ultralytics/boxmot/torchreid, fastapi/uvicorn/sqlalchemy. KHONG dung `python`
mac dinh trong PATH (thieu het cac goi nay). Don gian nhat: chay `start.bat`
(full, da tro san venv_tracking) hoac `start.bat --no-ai`.

```bash
# Chi API (khong can CARLA):
cd server
..\custom_tracking_system\venv_tracking\Scripts\python.exe app.py

# API + AI pipeline (can CARLA chay truoc):
cd server
..\custom_tracking_system\venv_tracking\Scripts\python.exe app.py --with-ai

# AI pipeline truc tiep (khong qua server):
cd custom_tracking_system
venv_tracking\Scripts\python.exe main.py --config config/camera_config.yaml --max-frames 1000
```

Frontend: `npm run dev` trong `frontend/`, chay tai `http://localhost:3000`
(khong phai 5173 — xem `vite.config.js`).

---

## Camera config hien tai

3 cameras: CAM_001 (0,0,3 yaw0 fov90), CAM_002 (50,0,3 yaw180 fov120),
CAM_003 (25,38,3 yaw-90 fov100) — CAM_003 da chuyen tu (25,-30,3 yaw45) vi vi
tri cu nhin vao san trong (plaza), khong co xe/nguoi; vi tri moi nhin vao cung
truc duong voi CAM_001/CAM_002 tu phia doi dien.
Resolution: 960x540, FPS: 10, synchronous mode.
3 ROI zones: intersection_main, side_road, parking_area.
LUU Y: ROI "parking_area" + wrong_way.lanes.CAM_003 trong camera_config.yaml
duoc tinh tu vi tri CAM_003 CU (25,-30,3 yaw45) — can recalibrate lai cho vi
tri moi (xem custom_tracking_system/docs/error.md).

---

## Tech stack

- **Simulator**: CARLA 0.9.9.4 (Unreal Engine 4)
- **AI**: Python, PyTorch >= 2.0, YOLOv8s (Ultralytics, COCO pretrained), OSNet (Market-1501 via torchreid), OpenCV 4.5.4, NumPy, SciPy
- **Server**: FastAPI >= 0.100, Uvicorn, SQLAlchemy >= 2.0, SQLite, Pydantic >= 2.0, WebSocket
- **Streaming**: MJPEG over HTTP

---

## Nhung gi DA hoan thanh

1. AI pipeline day du: detect (YOLO11m fine-tune VN, batch) -> track (ByteTrack/Kalman) -> ReID (OSNet) -> global ID (Spatio-Temporal Filter) -> trajectory (Kalman / Kalman+GRU Ensemble + calibration) -> incident (14 loai) -> alert ROI -> evidence
2. Backend API server: 17 REST endpoints + scenarios router, 3 WebSocket channels, MJPEG streaming
3. Database 5 bang (cameras, alerts, tracked_objects, tracking_history, rois)
4. AI processor tich hop pipeline vao FastAPI background thread
5. 3 che do chay: API-only, API+AI (CARLA), Direct (OpenCV window)
6. Camera config YAML, logging, data export (JSON/CSV)
7. Abstract VideoSource layer — tach AI pipeline khoi nguon video (CARLA/RTSP/File/Webcam)
8. Ground Truth module + evaluate_tracking.py — MOTA/IDF1 qua py-motmetrics
9. Web Dashboard (React + Tailwind + Vite) — camera grid, incident panel, alert management, scenario panel
10. IncidentDetector — 14 loai su co (12 reactive bao gom WRONG_WAY/RED_LIGHT_VIOLATION + 2 proactive: PREDICTED_COLLISION, PREDICTED_ROI_ENTRY)
11. EvidencePackage — ring buffer 30s, tu dong luu clip + crop + JSON
12. ScenarioController — 5 kich ban tai nan trong CARLA, noi day du qua server + ScenarioPanel.jsx
13. ByteTrackWrapper — Kalman + Hungarian matching + dual-threshold (thay IoU greedy)
14. TrajectoryPredictor (Kalman) + EnsembleTrajectoryPredictor (Kalman+GRU, dung trong main.py)
15. GlobalTracker — Spatio-Temporal Filter (camera_topology da cau hinh)
16. DualReIDExtractor — 2 model rieng: person + vehicle (VeRi-776 weights, voi calibration bias)
17. CameraCalibration/CalibrationStore — pixel<->world, dung trong main.py
18. train_veri.py + VeRi/ dataset, fine-tune YOLO11m (carla5/carla6) — `train_yolo_detector.py`, `resume_carla6.py`

## Nhung gi CHUA lam (theo development_roadmap.md)

1. **Vehicle ReID hoan thien** — chon/re-train epoch VeRi-776 tot nhat
2. **Fine-tune tiep YOLO tren anh render CARLA** — giam domain gap (Recall thap trong GT eval, xem error.md)
3. **Recording lien tuc** — ghi video lien tuc, cat clip su co (hien chi co ring buffer 30s khi su co)
4. **Camera management UI** — them/xoa camera runtime, health check

(Da xong 2026-06-13: wire EnsembleTrajectoryPredictor + CalibrationStore vao
server ai_processor.py; toi uu FPS server qua detect_batch() + throttle ReID;
calibrate ROI polygon + wrong_way.lanes.direction theo hinh hoc camera — xem
muc "Da hoan thanh" o tren va development_roadmap.md.)

---

## Kien truc VideoSource va GroundTruth

**VideoSource** (`modules/video_source.py`): Abstract layer de AI pipeline chi thay raw frames,
khong biet nguon la CARLA hay camera that. 4 implementations:
- `CARLAVideoSource` — wrap CARLA sensor, chi tra ve raw pixels
- `RTSPVideoSource` — doc tu camera IP (rtsp://...)
- `FileVideoSource` — doc tu video file (.mp4, .avi)
- `WebcamVideoSource` — doc tu webcam

**GroundTruth** (`modules/ground_truth.py`): TACH BIET khoi AI pipeline.
- `GroundTruthCollector` — doc actor_id, location, velocity tu CARLA World
- `FileGroundTruth` — doc tu file MOT Challenge format
- AI pipeline KHONG DUOC import module nay. Chi dung cho evaluation.

**Ly do**: CARLA cung cap oracle knowledge (actor.id) ma camera that khong co.
Neu AI pipeline truy cap thong tin nay, he thong se khong hoat dong khi chuyen sang camera that.

---

## Han che quan trong

- **Xe giong nhau**: Khi 2 xe cung mau + hinh dang, ReID (OSNet) khong phan biet duoc. Can spatio-temporal reasoning (thoi gian + khoang cach giua camera) hoac LPR (bien so xe, CARLA khong ho tro).
- **VideoSource chua tich hop vao server**: main.py da dung VideoSource (--source bridge/rtsp/file/webcam), nhung server/services/ai_processor.py van dung camera_controller truc tiep.
- **wrong_way.lanes.*.direction la gia tri tinh hinh hoc don gian hoa**: ca 3 camera dung [0,-1] (huong "ra xa camera" voi pitch=0), gia dinh xe di dung huong la di xa camera — nen kiem tra lai bang video CARLA thuc te neu lan duong thuc co huong khac.

---

## Luu y khi prompt tiep

- Folder `WindowsNoEditor/` la CARLA simulator, khong can doc/sua.
- Tat ca code tu viet nam trong `AI_custom/`.
- Khi yeu cau viet code, chi ro file path tuong doi tu `AI_custom/` (vi du: `server/routers/cameras.py`).
- Server chay tai `AI_custom/server/`, working directory la `server/` nen import la `from models.database import ...`, `from services.alert_service import ...`.
- AI modules chay tai `AI_custom/custom_tracking_system/`, import la `from modules.detector import ...`, `from utils.visualization import ...`.
- ai_processor.py them `custom_tracking_system/` vao sys.path de import duoc cac module AI.
