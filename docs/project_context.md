# Project Context — Multi-Camera CCTV Tracking System

> Tai lieu nay dung de dan vao dau cuoc hoi thoai voi Claude tren mobile
> hoac bat ky moi truong nao khong doc duoc folder du an.
> Copy toan bo noi dung file nay va paste vao prompt dau tien.

---

## Tong quan

Day la du an he thong giam sat camera CCTV da kenh, su dung AI de phat hien,
theo doi va nhan dien lai doi tuong (xe, nguoi) trong moi truong mo phong CARLA 0.9.9.4.

He thong gom 2 phan chinh:
- **AI Pipeline** — xu ly video: detect (YOLOv5) -> track (IoU) -> ReID (OSNet) -> global ID -> predict trajectory -> alert ROI.
- **Backend API Server** — FastAPI cung cap REST API, WebSocket real-time, MJPEG video streaming. AI pipeline chay trong background thread cua server.

Chua co frontend/web dashboard. Dang trong giai doan phat trien.

---

## Cau truc thu muc hien tai

```
E:\finalproject\
  AI_custom\
    custom_tracking_system\         # AI Pipeline
      config\
        camera_config.yaml          # 3 cameras (CAM_001/002/003), 3 ROIs, system params
      modules\
        camera_controller.py        # Dat camera trong CARLA, dong bo frame
        traffic_generator.py        # Spawn xe + nguoi voi autopilot
        detector.py                 # YOLOv5s pretrained COCO (person,car,bus,truck)
        tracker.py                  # SimpleTracker — IoU greedy matching, khong co Kalman
        reid.py                     # OSNet (torchreid, Market-1501) hoac fallback ResNet50
        global_tracking.py          # Gan Global ID xuyen camera bang ReID gallery matching
        trajectory_predictor.py     # Linear extrapolation (chua co Kalman/LSTM)
        alert_system.py             # Canh bao khi doi tuong vao ROI (point-in-polygon)
        video_source.py             # Abstract video source (CARLA/RTSP/File/Webcam)
        ground_truth.py             # Ground truth cho evaluation (tach biet khoi AI pipeline)
      utils\
        visualization.py            # Ve bbox, trajectory, ROI, multi-camera grid (OpenCV)
        metrics.py                  # Thu thap FPS, counts (chua co MOTA/IDF1 thuc)
        data_writer.py              # Export JSON, CSV, summary report
      models\
        hub\                        # YOLOv5 model cache (tu dong tai lan dau)
        detection\                  # (trong — chua fine-tune)
        reid\                       # (trong — chua fine-tune)
        tracking\                   # (trong — chua fine-tune)
      datasets\
        ground_truth\               # (trong)
        synthetic_data\             # (trong)
      main.py                       # Entry point chay truc tiep (khong qua server)
      requirements.txt
      run.bat

    server\                          # Backend API Server (FastAPI)
      app.py                        # FastAPI app, lifespan khoi dong AI, CLI entry point
      config.py                     # Paths, CARLA host/port, streaming params
      models\
        database.py                 # SQLAlchemy ORM, 5 bang, SQLite backend
        schemas.py                  # Pydantic request/response schemas
      routers\
        cameras.py                  # GET/POST/PUT/DELETE /api/cameras
        tracks.py                   # GET /api/tracks, /api/tracks/{id}/trajectory
        alerts.py                   # GET /api/alerts, PUT /api/alerts/{id}/acknowledge
        rois.py                     # GET/POST/PUT/DELETE /api/rois
        stats.py                    # GET /api/stats
        websocket.py                # WS /ws/alerts, /ws/tracks, /ws/stats
        stream.py                   # GET /stream/{camera_id} (MJPEG)
      services\
        ai_processor.py             # Wrap AI pipeline thanh background thread
        camera_service.py           # CRUD camera trong DB
        alert_service.py            # Query + filter + acknowledge alerts
        tracking_service.py         # Query tracks + trajectory + upsert
        stream_service.py           # FrameBuffer singleton cho MJPEG streaming
      middleware\                    # (du tru, chua implement)
      requirements.txt
      run_server.bat

    docs\
      description.md
      plan.md                       # Plan goc 10 phases
      workflow.md                   # Quy trinh trien khai 6 giai doan
      execute.md                    # Huong dan chay + 6 kich ban test
      development_roadmap.md        # Lo trinh phat trien con lai (8 phan)
      report.md                     # Bao cao du an (muc tieu, cau truc, tech, cach dung)
      project_context.md            # File nay

  WindowsNoEditor\                   # CARLA 0.9.9.4 (khong chinh sua)
    CarlaUE4.exe
    PythonAPI\carla\
    PythonAPI\examples\
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
1. camera_controller.get_synchronized_frames()  (hien tai)
   hoac MultiVideoSource.get_synchronized_frames()  (video_source.py, chua tich hop)
   -> {camera_id: {frame: np.array(H,W,3), timestamp, frame_number}}

2. detector.detect(frame)
   -> [{box:[x1,y1,x2,y2], confidence:float, class:str, class_id:int}]

3. tracker.update(detections)
   -> [{track_id:int, box:[x1,y1,x2,y2], class:str, positions:[[x,y],...]}]

4. reid_extractor.extract_feature(frame, box)
   -> np.array(1, 512) L2-normalized   (OSNet)
   -> np.array(1, 2048) L2-normalized  (ResNet50 fallback)

5. reid_extractor.match_with_gallery(feature, threshold=0.5)
   -> matched_global_id (int) hoac None

6. global_tracker.process_camera_tracks(camera_id, frame, local_tracks)
   -> [{global_id, camera_id, box, class, local_track_id}]

7. trajectory_predictor.update_trajectory(global_id, center, frame_idx)
   trajectory_predictor.predict(global_id)
   -> [[x,y], [x,y], ...] (5 buoc tuong lai) hoac None

8. alert_system.check_alerts(global_id, camera_id, predicted_positions, current_box)
   -> [{type:"ROI_WARNING", global_id, camera_id, roi_name, eta_frames, timestamp}]
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
- **AI**: Python, PyTorch >= 1.10, YOLOv5s v6.1 (COCO pretrained), OSNet (Market-1501 via torchreid), OpenCV 4.5.4, NumPy, SciPy
- **Server**: FastAPI >= 0.100, Uvicorn, SQLAlchemy >= 2.0, SQLite, Pydantic >= 2.0, WebSocket
- **Streaming**: MJPEG over HTTP

---

## Nhung gi DA hoan thanh

1. AI pipeline day du 8 buoc (detect -> track -> reid -> global -> predict -> alert -> visualize -> stream)
2. Backend API server: 17 REST endpoints, 3 WebSocket channels, MJPEG streaming
3. Database 5 bang (cameras, alerts, tracked_objects, tracking_history, rois)
4. AI processor tich hop pipeline vao FastAPI background thread
5. 3 che do chay: API-only, API+AI (CARLA), Direct (OpenCV window)
6. Camera config YAML, logging, data export (JSON/CSV)
7. Abstract VideoSource layer — tach AI pipeline khoi nguon video (CARLA/RTSP/File/Webcam)
8. Ground Truth module — thu thap CARLA actor data + doc file MOT format, TACH BIET khoi AI pipeline

## Nhung gi CHUA lam (theo development_roadmap.md)

1. **Web Dashboard (frontend)** — React/Vue, camera grid, alert panel, object detail, map view, ROI editor
2. **Anomaly detection** — overspeed, wrong-way, stopped vehicle, crowd density, loitering (hien chi co ROI entry)
3. **Nang cap tracker** — ByteTrack hoac DeepSORT thay SimpleTracker
4. **Nang cap trajectory** — Kalman filter hoac LSTM thay linear extrapolation
5. **Vehicle ReID** — model rieng cho xe (hien chi co person ReID, OSNet/Market-1501)
6. **Spatio-temporal reasoning** — dung thoi gian + khoang cach giua camera de phan biet doi tuong giong nhau
7. **Recording & playback** — ghi video lien tuc, cat clip su co
8. **Camera management UI** — them/xoa camera runtime, health check
9. **Ground truth evaluation** — module ground_truth.py da co nhung chua tich hop tinh MOTA, IDF1, mAP
10. **Tich hop VideoSource** — video_source.py da viet, can thay camera_controller trong main.py va ai_processor.py

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
