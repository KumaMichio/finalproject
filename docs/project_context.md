# Project Context — Multi-Camera CCTV Tracking System

> Tai lieu nay dung de dan vao dau cuoc hoi thoai voi Claude tren mobile
> hoac bat ky moi truong nao khong doc duoc folder du an.
> Copy toan bo noi dung file nay va paste vao prompt dau tien.

---

## Tong quan

Day la du an he thong giam sat camera CCTV da kenh, su dung AI de phat hien,
theo doi va nhan dien lai doi tuong (xe, nguoi) trong moi truong mo phong CARLA 0.9.9.4.

He thong gom 2 phan chinh:
- **AI Pipeline** — xu ly video: detect (YOLOv8s) -> track (ByteTrack/Kalman) -> ReID (OSNet) -> global ID (+ Spatio-Temporal Filter) -> predict trajectory (exp-weighted px/s) -> incident detection (12 loai) -> alert ROI.
- **Backend API Server** — FastAPI cung cap REST API, WebSocket real-time, MJPEG video streaming. AI pipeline chay trong background thread cua server.

Chua co frontend/web dashboard. Dang trong giai doan phat trien.

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
      trajectory_predictor.py       # Exp-weighted velocity (px/s, timestamp-based), horizons 0.5-3s
      alert_system.py               # Canh bao khi doi tuong vao ROI (point-in-polygon)
      incident_detector.py          # Phat hien 12 loai su co (10 reactive + 2 proactive)
      evidence_package.py           # Ring buffer 30s — luu clip + crop + JSON khi su co
      scenario_controller.py        # 5 kich ban tai nan trong CARLA
      video_source.py               # Abstract video source (CARLA/RTSP/File/Webcam)
      ground_truth.py               # Ground truth cho evaluation (tach biet khoi AI pipeline)
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

7. incident_detector.update(global_tracks, camera_id, predictions, rois)
   # 10 reactive + 2 proactive (PREDICTED_COLLISION, PREDICTED_ROI_ENTRY)
   -> [{type, severity, global_id, camera_id, message, details}]

8. evidence.capture(incident, frames_dict, global_tracks)   # khi CRITICAL
   # Luu crop anh + clip 30s truoc/sau + metadata JSON -> evidence/<id>/
```

---

## Cach chay server

```bash
# Chi API (khong can CARLA):
cd AI_custom/server
python app.py

# API + AI pipeline (can CARLA chay truoc):
cd AI_custom/server
python app.py --with-ai

# AI pipeline truc tiep (khong qua server):
cd AI_custom/custom_tracking_system
python main.py --config config/camera_config.yaml --max-frames 1000
```

---

## Camera config hien tai

3 cameras: CAM_001 (0,0,3 fov90), CAM_002 (50,0,3 fov120), CAM_003 (25,-30,3 fov100).
Resolution: 960x540, FPS: 10, synchronous mode.
3 ROI zones: intersection_main, side_road, parking_area.

---

## Tech stack

- **Simulator**: CARLA 0.9.9.4 (Unreal Engine 4)
- **AI**: Python, PyTorch >= 2.0, YOLOv8s (Ultralytics, COCO pretrained), OSNet (Market-1501 via torchreid), OpenCV 4.5.4, NumPy, SciPy
- **Server**: FastAPI >= 0.100, Uvicorn, SQLAlchemy >= 2.0, SQLite, Pydantic >= 2.0, WebSocket
- **Streaming**: MJPEG over HTTP

---

## Nhung gi DA hoan thanh

1. AI pipeline day du: detect (YOLOv8s batch) -> track (ByteTrack/Kalman) -> ReID (OSNet) -> global ID (Spatio-Temporal Filter) -> trajectory (exp-weighted px/s) -> incident (12 loai) -> alert ROI -> evidence
2. Backend API server: 17 REST endpoints, 3 WebSocket channels, MJPEG streaming
3. Database 5 bang (cameras, alerts, tracked_objects, tracking_history, rois)
4. AI processor tich hop pipeline vao FastAPI background thread
5. 3 che do chay: API-only, API+AI (CARLA), Direct (OpenCV window)
6. Camera config YAML, logging, data export (JSON/CSV)
7. Abstract VideoSource layer — tach AI pipeline khoi nguon video (CARLA/RTSP/File/Webcam)
8. Ground Truth module — TACH BIET khoi AI pipeline, dung cho evaluation
9. Web Dashboard (React + Tailwind + Vite) — camera grid, incident panel, alert management
10. IncidentDetector — 12 loai su co (10 reactive + 2 proactive: PREDICTED_COLLISION, PREDICTED_ROI_ENTRY)
11. EvidencePackage — ring buffer 30s, tu dong luu clip + crop + JSON
12. ScenarioController — 5 kich ban tai nan trong CARLA
13. ByteTrackWrapper — Kalman + Hungarian matching + dual-threshold (thay IoU greedy)
14. TrajectoryPredictor — exp-weighted velocity (px/s, timestamp-based), horizons 0.5-3s (da sua loi)
15. GlobalTracker — Spatio-Temporal Filter (logic da co, cho cau hinh camera_topology)
16. DualReIDExtractor — 2 model rieng: person + vehicle (cho VeRi-776 weights)
17. train_veri.py + VeRi/ dataset san sang fine-tune Vehicle ReID

## Nhung gi CHUA lam (theo development_roadmap.md)

1. **Vehicle ReID hoan thien** — train xong VeRi-776, kich hoat DualReIDExtractor trong pipeline
2. **Cau hinh camera_topology** — do khoang cach thuc te giua cac camera de bat Spatio-Temporal Filter
3. **Fine-tune YOLOv8s** — tren BDD100K + Vietnam Traffic Dataset (motorcycle VN chinh xac hon)
4. **Recording lien tuc** — ghi video lien tuc, cat clip su co (hien chi co ring buffer 30s khi su co)
5. **Camera management UI** — them/xoa camera runtime, health check
6. **Ground truth evaluation** — tich hop ground_truth.py + motmetrics (MOTA, IDF1, mAP)
7. **Tich hop VideoSource** — thay camera_controller bang video_source trong main.py + ai_processor.py

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
- **VideoSource chua tich hop**: video_source.py da viet nhung main.py va ai_processor.py van dung camera_controller truc tiep.

---

## Luu y khi prompt tiep

- Folder `WindowsNoEditor/` la CARLA simulator, khong can doc/sua.
- Tat ca code tu viet nam trong `AI_custom/`.
- Khi yeu cau viet code, chi ro file path tuong doi tu `AI_custom/` (vi du: `server/routers/cameras.py`).
- Server chay tai `AI_custom/server/`, working directory la `server/` nen import la `from models.database import ...`, `from services.alert_service import ...`.
- AI modules chay tai `AI_custom/custom_tracking_system/`, import la `from modules.detector import ...`, `from utils.visualization import ...`.
- ai_processor.py them `custom_tracking_system/` vao sys.path de import duoc cac module AI.
