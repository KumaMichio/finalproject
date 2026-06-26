"""
AI Processor — chay AI pipeline trong background thread.

Wrap cac module hien tai (detector, tracker, reid, global_tracking,
trajectory_predictor, alert_system) thanh 1 service chay lien tuc.

Ket qua duoc day vao:
  - frame_buffer  (cho video streaming)
  - database      (cho REST API queries)
  - ws_manager    (cho WebSocket push)
"""

import sys
import time
import json
import math
import asyncio
import logging
import threading
import cv2
from collections import deque
from pathlib import Path
from datetime import datetime

logger = logging.getLogger(__name__)

# Them tracking system vao sys.path de import cac module AI
_tracking_dir = Path(__file__).resolve().parent.parent.parent / "custom_tracking_system"
if str(_tracking_dir) not in sys.path:
    sys.path.insert(0, str(_tracking_dir))


class AIProcessor:
    """Quan ly toan bo AI pipeline, chay trong background thread."""

    def __init__(self, config_path: str, carla_host: str = "localhost",
                 carla_port: int = 2000, source: str = "carla",
                 video_paths: list[str] | None = None,
                 camera_ids: list[str] | None = None,
                 loop_video: bool = True):
        self.config_path = config_path
        self.carla_host = carla_host
        self.carla_port = carla_port

        # source='carla': spawn camera trong CARLA (mac dinh, hanh vi cu).
        # source='file': doc tu video file thuc qua VideoSource (khong can CARLA),
        # cung pattern da chay on trong custom_tracking_system/main.py --source file.
        self.source = source
        self.video_paths = video_paths or []
        self.camera_ids = camera_ids
        self.loop_video = loop_video

        self._thread: threading.Thread | None = None
        self._running = False
        self._lock = threading.Lock()

        # Shared state (doc tu API/WebSocket)
        self.frame_count = 0
        self.fps = 0.0
        self.status = "stopped"  # stopped / starting / running / error
        self._active_track_count = 0

        # ReID throttle: chi chay extract_feature() cho track da biet moi
        # REID_INTERVAL frame (track moi luon duoc chay ReID ngay)
        self.REID_INTERVAL = 3
        self._seen_track_ids: dict = {}  # {camera_id: set(local_track_id)}

        # Hien thi banner "VA CHAM!" tren stream cua camera trong vai giay
        # sau khi ScenarioController ghi nhan 1 va cham (de nguoi xem thay
        # ro su kien dang xay ra, khong chi qua DB/alert).
        self._collision_overlay_until: dict = {}  # {camera_id: expiry_timestamp}
        self.COLLISION_OVERLAY_SECONDS = 3.0

        # Tu dong don dep actor do scenario tao ra sau 1 khoang thoi gian,
        # de camera tro lai trang thai giao thong binh thuong.
        self._scenario_cleanup_timer: threading.Timer | None = None
        self.SCENARIO_CLEANUP_DELAY = 30.0

        # Goal Classifier / Route Predictor / map state — gan gia tri thuc o
        # _initialize() (chay trong background thread), nhung khoi tao truoc
        # o day de get_map_state()/mark_object() khong AttributeError neu duoc
        # goi truoc khi _initialize() hoan tat.
        self.goal_classifier = None
        self.route_predictor = None
        self.georef = None
        self.calibration: dict = {}
        self.marked_global_id: int | None = None
        self._marked_snapshot_saved: bool = False
        self._world_hist: dict = {}        # {global_id: deque[(t, x, y)]}
        self._world_pos: dict = {}         # {global_id: {lat, lon, heading_deg, ...}}
        self._goal_predictions: dict = {}  # {global_id: {direction, probabilities, confidence}}
        self._marked_route: list = []

    def start(self):
        """Khoi dong AI pipeline trong background thread."""
        if self._running:
            logger.warning("AI processor already running")
            return
        self._running = True
        self.status = "starting"
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        logger.info("AI processor thread started")

    def stop(self):
        """Dung AI pipeline."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=10)
        self.status = "stopped"
        logger.info("AI processor stopped")

    def _run(self):
        """Main loop — chay trong background thread."""
        try:
            self._initialize()
            self.status = "running"
            self._main_loop()
        except Exception as e:
            logger.error(f"AI processor error: {e}", exc_info=True)
            self.status = "error"
        finally:
            self._cleanup()

    def _initialize(self):
        """Khoi tao nguon video (CARLA hoac file) + cac module AI."""
        logger.info("Initializing AI pipeline (source=%s)...", self.source)

        # Import AI modules (dung chung ca 2 nhanh source)
        from modules.detector import ObjectDetector
        from modules.tracker import ByteTrackWrapper
        from modules.reid import DualReIDExtractor
        from modules.global_tracking import GlobalTracker
        from modules.trajectory_predictor import EnsembleTrajectoryPredictor
        from modules.calibration import CalibrationStore
        from modules.alert_system import AlertSystem
        from modules.incident_detector import IncidentDetector
        from modules.evidence_package import EvidencePackage
        from utils.visualization import Visualizer

        if self.source == "carla":
            # Import CARLA
            import carla
            self.client = carla.Client(self.carla_host, self.carla_port)
            self.client.set_timeout(10.0)
            self.world = self.client.get_world()

            # Synchronous mode
            settings = self.world.get_settings()
            settings.synchronous_mode = True
            settings.fixed_delta_seconds = 0.1
            self.world.apply_settings(settings)

            from modules.camera_controller import CameraController
            from modules.traffic_generator import TrafficGenerator
            from modules.scenario_controller import ScenarioController

            self.camera_controller = CameraController(
                self.client, self.world, self.config_path)
            self.camera_controller.setup_cameras()
            self.config = self.camera_controller.config
            self.frame_source = self.camera_controller

            self.traffic_generator = TrafficGenerator(
                self.world, num_vehicles=10, num_pedestrians=5)
            self.traffic_generator.spawn_actors()

            self.scenario_controller = ScenarioController(
                self.world, self.client,
                cameras_config=self.config.get("cameras", {}))
            self.camera_traffic_lights = self._find_nearby_traffic_lights()
            self._cam_ids = list(self.camera_controller.cameras.keys())
        else:
            # source == 'file': video CCTV thuc qua VideoSource, khong can CARLA
            # (cung pattern da chay on trong custom_tracking_system/main.py).
            import yaml
            from modules.video_source import MultiVideoSource, FileVideoSource

            with open(self.config_path, "r", encoding="utf-8") as f:
                self.config = yaml.safe_load(f)

            cam_ids = self.camera_ids or [
                f"CAM_{i + 1:03d}" for i in range(len(self.video_paths))
            ]
            if len(cam_ids) != len(self.video_paths):
                raise ValueError(
                    f"camera_ids ({len(cam_ids)}) phai khop so luong voi "
                    f"video_paths ({len(self.video_paths)})"
                )

            self.frame_source = MultiVideoSource()
            for cam_id, path in zip(cam_ids, self.video_paths):
                self.frame_source.add_source(
                    FileVideoSource(path, cam_id, loop=self.loop_video))

            # Khong co CARLA world/scenario/traffic-light o che do video thuc.
            self.world = None
            self.scenario_controller = None
            self.camera_traffic_lights = {}
            self._cam_ids = cam_ids

        _yolo_s = _tracking_dir / "weights" / "yolo11s_vn.pt"
        _yolo_m = _tracking_dir / "weights" / "yolo11m_vn.pt"
        _yolo_weights = str(_yolo_s if _yolo_s.exists() else _yolo_m)
        self.detector = ObjectDetector(model_type=_yolo_weights, half=True)

        self.trackers = {}
        for cam_id in self._cam_ids:
            self.trackers[cam_id] = ByteTrackWrapper(
                track_activation_threshold=0.25,
                lost_track_buffer=30,
                minimum_matching_threshold=0.8,
                frame_rate=10,
                class_map=self.detector.CLASSES,
            )

        vehicle_weights = str(_tracking_dir / "weights" / "osnet_veri776_v2.pth")
        self.reid_extractor = DualReIDExtractor(vehicle_weights=vehicle_weights)

        # Đọc camera_topology từ config (cặp key dùng "__" làm phân cách vì YAML không hỗ trợ tuple key)
        raw_topology = self.config.get("camera_topology", {})
        camera_topology = {
            tuple(k.split("__")): tuple(v)
            for k, v in raw_topology.items()
        }
        self.global_tracker = GlobalTracker(self.reid_extractor, camera_topology=camera_topology)

        try:
            calibration = CalibrationStore.load_from_config(self.config_path)
        except Exception as e:
            logger.warning("CalibrationStore.load_from_config failed (%s) "
                            "-> trajectory predictor will be Kalman-only", e)
            calibration = {}
        self.trajectory_predictor = EnsembleTrajectoryPredictor(
            window_size=10, calibration=calibration,
            gru_checkpoint=str(_tracking_dir / "weights" / "trajectory_gru.pth"))

        self.alert_system = AlertSystem(self.trajectory_predictor)
        if "rois" in self.config:
            self.alert_system.set_rois(self.config["rois"])

        incident_cfg = self.config.get("incident_detection", {})
        self.incident_detector = IncidentDetector(config=incident_cfg)
        self.evidence = EvidencePackage(output_dir="evidence", buffer_seconds=30, fps=10)

        # Mau den giao thong tu camera thuc (source != 'carla') — xem
        # modules/traffic_light_classifier.py. Khong tao gi neu config
        # khong co toa do diem do (camera_config*.yaml: incident_detection.
        # traffic_light.points).
        self.traffic_light_classifier = None
        tl_points = (incident_cfg.get("traffic_light", {}) or {}).get("points", {})
        if tl_points:
            from modules.traffic_light_classifier import TrafficLightColorClassifier
            self.traffic_light_classifier = TrafficLightColorClassifier(tl_points)

        # Goal Classifier — huong re (left/right/straight/u_turn). Generic,
        # khong phu thuoc map/georef, hoat dong tu bbox history nen tai luon
        # bat ke nguon nao (~4ms cho 30 xe, da benchmark).
        from modules.goal_classifier import GoalClassifier
        try:
            self.goal_classifier = GoalClassifier(
                str(_tracking_dir / "weights" / "goal_classifier_real.pkl"),
                str(_tracking_dir / "weights" / "goal_scaler_real.pkl"),
                str(_tracking_dir / "weights" / "goal_label_encoder_real.pkl"),
                str(_tracking_dir / "weights" / "goal_feature_names_real.json"),
            )
        except FileNotFoundError as e:
            logger.warning("GoalClassifier weights khong tim thay (%s) - bo qua", e)
            self.goal_classifier = None

        # Route Predictor + GeoReference — du doan huong di dai han + ve len
        # map thuc (lat/lon). Chi kich hoat khi config co section
        # 'route_prediction' tro dung 1 map da georeference (vd pho_hue_tkc).
        # camera_config.yaml mac dinh (CARLA Town10HD_Opt) KHONG co section nay
        # vi vi tri camera o do khong khop dia ly voi map nao (hanoi_district3
        # la khu vuc gia lap rieng, khac he toa do) — map/route se tu tat.
        route_cfg = self.config.get("route_prediction")
        self.route_n_hops = 4
        self.route_min_confidence = 0.40
        if route_cfg:
            try:
                from georeference import GeoReference
                from route_predictor import RoutePredictor

                map_name = route_cfg["map_name"]
                map_dir = _tracking_dir.parent / "Map"
                graph_path = map_dir / route_cfg.get("graph_path", "junction_graph.json")
                self.georef = GeoReference.from_map(map_name)
                self.route_predictor = RoutePredictor.from_map(
                    map_name, georef=self.georef, graph_path=graph_path)
                self.route_n_hops = route_cfg.get("n_hops", 4)
                self.route_min_confidence = route_cfg.get("min_confidence", 0.40)
                logger.info("RoutePredictor enabled (map=%s)", map_name)
            except Exception as e:
                logger.warning("RoutePredictor init failed (%s) - map/route prediction se tat", e)
                self.route_predictor = None
                self.georef = None

        # Giu lai calibration (da load o tren cho trajectory_predictor) de
        # dung chung cho world-position/route prediction.
        self.calibration = calibration

        # CalibrationStore keys tu cam_cfg['camera_id'] khai bao trong YAML,
        # hoan toan doc lap voi self._cam_ids (vd mac dinh CAM_001, CAM_002...
        # neu --camera-ids khong duoc truyen). Neu lech, calibration.get(camera_id)
        # o vong xu ly chinh luon tra None va ÂM THẦM tat world-position/map/
        # route prediction cho camera do, khong co loi/canh bao nao khac de
        # phat hien — canh bao ngay luc khoi dong de tranh debug mu.
        if self.calibration:
            uncalibrated = [c for c in self._cam_ids if c not in self.calibration]
            if uncalibrated:
                logger.warning(
                    "Camera(s) %s khong co calibration tuong ung (calibration "
                    "co cho: %s) -> world-position/map/route prediction se TAT "
                    "AM THAM cho camera nay. Kiem tra --camera-ids co khop dung "
                    "'camera_id' khai bao trong --config khong.",
                    uncalibrated, list(self.calibration.keys()),
                )

        self.visualizer = Visualizer()

        # Sync cameras vao database
        self._sync_cameras_to_db()

        logger.info("AI pipeline initialized successfully")

    def _sync_cameras_to_db(self):
        """Dong bo camera config vao database."""
        from models.database import SessionLocal, Camera

        db = SessionLocal()
        try:
            for cam_name, cam_cfg in self.config["cameras"].items():
                cam_id = cam_cfg["camera_id"]
                existing = db.query(Camera).filter(Camera.id == cam_id).first()
                if not existing:
                    db.add(Camera(
                        id=cam_id,
                        name=cam_name,
                        position_x=cam_cfg["position"][0],
                        position_y=cam_cfg["position"][1],
                        position_z=cam_cfg["position"][2],
                        rotation_pitch=cam_cfg["rotation"][0],
                        rotation_yaw=cam_cfg["rotation"][1],
                        rotation_roll=cam_cfg["rotation"][2],
                        resolution_w=cam_cfg["resolution"][0],
                        resolution_h=cam_cfg["resolution"][1],
                        fov=cam_cfg["view_angle"],
                        fps=cam_cfg.get("fps", 10),
                        status="active",
                    ))
            db.commit()
        finally:
            db.close()

    def _find_nearby_traffic_lights(self) -> dict:
        """Map camera_id -> list traffic light actors gan camera (CARLA world),
        dung de check RED_LIGHT_VIOLATION (incident_detection.red_light)."""
        red_light_cfg = self.config.get("incident_detection", {}).get("red_light", {})
        radius = red_light_cfg.get("search_radius_m", 40)

        try:
            lights = list(self.world.get_actors().filter("traffic.traffic_light*"))
        except Exception as e:
            logger.warning("Khong lay duoc traffic lights tu CARLA: %s", e)
            return {}

        mapping = {}
        for cam_cfg in self.config["cameras"].values():
            cam_id = cam_cfg["camera_id"]
            cx, cy, _cz = cam_cfg["position"]
            nearby = []
            for tl in lights:
                loc = tl.get_location()
                dist = ((loc.x - cx) ** 2 + (loc.y - cy) ** 2) ** 0.5
                if dist <= radius:
                    nearby.append(tl)
            mapping[cam_id] = nearby
            logger.info("Camera %s: %d traffic light(s) trong radius %dm",
                         cam_id, len(nearby), radius)
        return mapping

    def _get_traffic_light_state(self, camera_id: str, frame=None):
        """Tra ve 'Red'/'Yellow'/'Green'/None cho camera (uu tien Red neu co
        >=1 traffic light gan camera dang Red).

        source='carla': doc truc tiep tu CARLA actor (tl.get_state()).
        Nguon khac (file/rtsp/webcam): khong co actor de doc, dung
        TrafficLightColorClassifier (modules/traffic_light_classifier.py) —
        can `frame` hien tai cua camera_id va config co toa do diem do."""
        if self.source != "carla":
            if self.traffic_light_classifier is None or frame is None:
                return None
            return self.traffic_light_classifier.classify(frame, camera_id)

        lights = self.camera_traffic_lights.get(camera_id, [])
        if not lights:
            return None
        states = set()
        for tl in lights:
            try:
                states.add(tl.get_state().name)
            except Exception:
                continue
        for preferred in ("Red", "Yellow", "Green"):
            if preferred in states:
                return preferred
        return None

    def _main_loop(self):
        """Vong lap xu ly chinh."""
        from services.stream_service import frame_buffer
        from services.tracking_service import upsert_tracked_object, add_history, set_marked_snapshot
        from services.alert_service import create_alert
        from models.database import SessionLocal

        fps_start = time.time()
        fps_frame_count = 0

        while self._running:
            self.frame_count += 1
            fps_frame_count += 1

            # Tinh FPS moi 1 giay + push stats
            elapsed = time.time() - fps_start
            if elapsed >= 1.0:
                self.fps = fps_frame_count / elapsed
                fps_frame_count = 0
                fps_start = time.time()
                self._push_stats_ws()

            # Lay frames tu cameras (CARLA hoac video file thuc — cung interface)
            sync_frames = self.frame_source.get_synchronized_frames()
            if not sync_frames:
                if self.source == "carla":
                    self._safe_tick()
                continue

            # Lay va cham moi tu ScenarioController NGAY tu dau frame, danh
            # dau camera lien quan de ve banner "VA CHAM!" len stream trong
            # vai giay (nguoi xem thay ro su kien dang xay ra). Khong co
            # ScenarioController o che do video thuc (source='file').
            new_collisions = (
                self.scenario_controller.get_new_collisions()
                if self.scenario_controller else []
            )
            if new_collisions:
                expiry = time.time() + self.COLLISION_OVERLAY_SECONDS
                for col in new_collisions:
                    cam_id = col.get("camera_id")
                    if cam_id:
                        self._collision_overlay_until[cam_id] = expiry

            # Mo 1 DB session cho frame nay
            db = SessionLocal()
            try:
                all_global_tracks = []

                # 1. Batch detection — 1 GPU forward pass cho tat ca camera
                cam_ids = list(sync_frames.keys())
                frames = [sync_frames[c]["frame"] for c in cam_ids]
                cam_timestamps = [sync_frames[c]["timestamp"] for c in cam_ids]
                batch_detections = self.detector.detect_batch(frames)

                for camera_id, frame, detections, cam_ts in zip(
                        cam_ids, frames, batch_detections, cam_timestamps):

                    # 2. Single-camera tracking (ByteTrackWrapper cần thêm frame).
                    # timestamp = dong ho cua frame nguon (video-content-time cho
                    # FileVideoSource, wall-clock cho CARLA/RTSP) — KHONG dung
                    # time.time() o day, neu khong speed se bi nhieu theo do tre
                    # xu ly CPU (xem video_source.py FileVideoSource).
                    local_tracks = self.trackers[camera_id].update(
                        detections, frame, timestamp=cam_ts)

                    # 3. Cross-camera global tracking — throttle ReID: chi chay
                    # extract_feature() cho track moi hoac moi REID_INTERVAL frame
                    seen = self._seen_track_ids.setdefault(camera_id, set())
                    reid_track_ids = set()
                    for lt in local_tracks:
                        tid = lt["track_id"]
                        if tid not in seen or self.frame_count % self.REID_INTERVAL == 0:
                            reid_track_ids.add(tid)
                        seen.add(tid)

                    global_tracks = self.global_tracker.process_camera_tracks(
                        camera_id, frame, local_tracks, reid_track_ids=reid_track_ids
                    )
                    all_global_tracks.extend(global_tracks)

                    # 4. Update trajectory + check alerts — dung cung cam_ts
                    # (dong ho cua frame nguon) de nhat quan voi tracker o tren,
                    # khong dung time.time() (xem ly do o comment phia tren).
                    now_ts = cam_ts
                    predictions_map = {}  # {global_id: dict} cho incident detector
                    predictions_for_viz = {}  # {global_id: [[x,y],...]} cho visualizer
                    for g_track in global_tracks:
                        gid = g_track["global_id"]
                        box = g_track["box"]
                        center = [(box[0] + box[2]) / 2, (box[1] + box[3]) / 2]

                        self.trajectory_predictor.update_trajectory(
                            gid, center, now_ts,
                            camera_id=camera_id,
                            obj_class=g_track.get("class", "car"),
                            all_tracks=all_global_tracks,
                        )

                        predicted = self.trajectory_predictor.predict(gid)
                        if predicted:
                            predictions_map[gid] = predicted
                            pred_list = list(predicted.values())
                            predictions_for_viz[gid] = pred_list
                            # check_alerts() cần list[[x,y]]
                            alerts = self.alert_system.check_alerts(
                                gid, camera_id, pred_list, box
                            )
                            for alert in alerts:
                                self.alert_system.log_alert(alert)
                                self._save_alert(db, alert)

                        # 4b. World position (pixel->world qua calibration camera) -
                        # dung cho object marking + ve vi tri/route len map thuc.
                        calib = self.calibration.get(camera_id)
                        if calib is not None:
                            self._update_world_position(
                                gid, camera_id, calib, box,
                                g_track.get("class", "car"), now_ts)

                        # 4c. Goal Classifier - huong re (left/right/straight/u_turn)
                        if self.goal_classifier is not None:
                            goal_res = self.goal_classifier.update_and_predict(
                                gid, box, g_track.get("class", "car"), now_ts)
                            if goal_res is not None:
                                self._goal_predictions[gid] = goal_res

                        # 4d. Route Predictor - chi tinh cho object da duoc "mark"
                        # (tranh chi phi tinh route cho moi xe moi frame).
                        if gid == self.marked_global_id:
                            self._update_marked_route(gid)

                            # Luu anh crop luc danh dau - chi 1 lan/lan mark
                            # (mark_object() reset co nay), de co the "truy
                            # quet lai" object sau (xem ghi chu mark_object()).
                            if not self._marked_snapshot_saved:
                                h, w = frame.shape[:2]
                                x1, y1, x2, y2 = box
                                x1 = max(0, int(x1) - 20)
                                y1 = max(0, int(y1) - 20)
                                x2 = min(w, int(x2) + 20)
                                y2 = min(h, int(y2) + 20)
                                crop = frame[y1:y2, x1:x2]
                                if crop.size > 0:
                                    out_dir = Path("evidence") / "marked"
                                    out_dir.mkdir(parents=True, exist_ok=True)
                                    snap_path = str(out_dir / f"marked_{gid}.jpg")
                                    cv2.imwrite(snap_path, crop)
                                    set_marked_snapshot(
                                        db, gid, snap_path, datetime.fromtimestamp(now_ts)
                                    )
                                    self._marked_snapshot_saved = True

                        # Luu tracking data vao DB (moi 10 frames de giam tai)
                        if self.frame_count % 10 == 0:
                            upsert_tracked_object(
                                db, gid, g_track["class"], camera_id
                            )
                            add_history(
                                db, gid, camera_id, self.frame_count, box
                            )

                    # 5. Incident detection — truyền predictions để kích hoạt proactive alerts
                    incidents = self.incident_detector.update(
                        global_tracks, camera_id,
                        predictions=predictions_map or None,
                        rois=self.alert_system.rois or None,
                        traffic_light_state=self._get_traffic_light_state(camera_id, frame),
                    )
                    for incident in incidents:
                        self._save_incident(db, incident)
                        self._push_incident_ws(incident)

                    # 6. Evidence capture cho incident nghiêm trọng
                    if incidents:
                        critical = [i for i in incidents if i['severity'] == 'CRITICAL']
                        if critical:
                            self.evidence.capture(
                                incident=critical[0],
                                frames_dict={camera_id: frame},
                                global_tracks=global_tracks,
                            )

                    # 7. Buffer frame cho evidence ring buffer
                    self.evidence.buffer_frame(camera_id, frame)

                    # 8. Ve visualization len frame
                    vis_frame = self.visualizer.draw_tracks(frame, global_tracks)
                    if predictions_for_viz:
                        vis_frame = self.visualizer.draw_predictions(
                            vis_frame, predictions_for_viz, global_tracks)

                    # 8b. Banner "VA CHAM!" neu camera nay vua co va cham
                    if self._collision_overlay_until.get(camera_id, 0) > time.time():
                        vis_frame = self.visualizer.draw_collision_banner(vis_frame)

                    # 9. Day frame vao buffer cho MJPEG streaming
                    frame_buffer.put_frame(camera_id, vis_frame)

                # 10. Va cham tu ScenarioController (hit_and_run, rear_end, ...)
                self._check_scenario_collisions(db, sync_frames, all_global_tracks, new_collisions)

                db.commit()

                self._active_track_count = len(all_global_tracks)
                self._prune_stale_tracking_state(all_global_tracks)
                self._push_ws_updates(all_global_tracks)

            except Exception as e:
                logger.error(f"Frame processing error: {e}", exc_info=True)
                db.rollback()
            finally:
                db.close()

            # Tick CARLA (khong can voi video file — source tu doc frame ke tiep)
            if self.source == "carla":
                self._safe_tick()

    def _safe_tick(self):
        """world.tick() bao boc loi rpc 'set_actor_collisions'.

        Client carla 0.9.15 vs server 0.9.14: pedestrian AI controller.start()
        gui 1 RPC chua duoc server ho tro, loi nay khong xay ra ngay ma bi
        tre va noi len o lan world.tick() ke tiep -> neu khong bat se lam chet
        ca AI processor thread. Loi nay khong anh huong toi simulation/tracking
        nen chi log warning va tiep tuc.
        """
        try:
            self.world.tick()
        except RuntimeError as e:
            if "set_actor_collisions" in str(e):
                logger.warning(f"Ignored non-fatal CARLA RPC error on tick: {e}")
            else:
                raise

    def _check_scenario_collisions(self, db, sync_frames: dict, all_global_tracks: list,
                                    new_collisions: list):
        """Day cac va cham moi (da lay tu ScenarioController dau frame) thanh incident CRITICAL."""
        for col in new_collisions:
            cam_id = col.get("camera_id")
            incident = {
                "type":      "VEHICLE_COLLISION",
                "severity":  "CRITICAL",
                "global_id": col["attacker_id"],
                "camera_id": cam_id,
                "timestamp": datetime.now(),
                "message":   (f"Phát hiện {col['scenario_desc']} tại {cam_id}: "
                               f"xe #{col['attacker_id']} đâm {col['strength_desc']} "
                               f"vào {col['victim_desc']}"),
                "details":   col,
            }
            self._save_incident(db, incident)
            self._push_incident_ws(incident)

            if cam_id in sync_frames:
                frames_dict = {cam_id: sync_frames[cam_id]["frame"]}
            else:
                frames_dict = {cid: sync_frames[cid]["frame"] for cid in sync_frames}

            self.evidence.capture(
                incident=incident,
                frames_dict=frames_dict,
                global_tracks=all_global_tracks,
            )

    def _save_incident(self, db, incident: dict):
        """Lưu incident vào bảng alerts trong database."""
        from models.database import Alert
        db.add(Alert(
            type=incident['type'],
            severity=incident['severity'],
            global_id=incident['global_id'],
            camera_id=incident['camera_id'],
            message=incident['message'],
            details=json.dumps(incident.get('details', {}), default=str),
            created_at=incident['timestamp'],
        ))

    def _push_incident_ws(self, incident: dict):
        """Push incident qua WebSocket /ws/alerts."""
        try:
            from routers.websocket import ws_manager
            if not hasattr(self, '_event_loop') or ws_manager.get_count("alerts") == 0:
                return
            msg = {
                "event":     "incident",
                "type":      incident['type'],
                "severity":  incident['severity'],
                "global_id": incident['global_id'],
                "camera_id": incident['camera_id'],
                "message":   incident['message'],
                "details":   incident.get('details', {}),
                "timestamp": incident['timestamp'].isoformat(),
            }
            asyncio.run_coroutine_threadsafe(
                ws_manager.broadcast("alerts", msg),
                self._event_loop,
            )
        except Exception:
            pass

    def _save_alert(self, db, alert_dict: dict):
        """Luu 1 alert vao database."""
        from models.database import Alert

        db_alert = Alert(
            type=alert_dict.get("type", "ROI_WARNING"),
            severity="warning",
            global_id=alert_dict.get("global_id"),
            camera_id=alert_dict.get("camera_id"),
            roi_name=alert_dict.get("roi_name"),
            message=(f"Phương tiện #{alert_dict.get('global_id')} sắp tiến vào "
                     f"khu vực giao lộ tại {alert_dict.get('camera_id')}"),
            details=json.dumps({
                "eta_frames": alert_dict.get("eta_frames"),
            }),
            created_at=datetime.utcnow(),
        )
        db.add(db_alert)

    def _push_ws_updates(self, global_tracks: list):
        """Push tracking updates qua WebSocket (non-blocking)."""
        try:
            from routers.websocket import ws_manager
            if not hasattr(self, '_event_loop') or ws_manager.get_count("tracks") == 0:
                return
            for track in global_tracks:
                msg = {
                    "event": "track_update",
                    "global_id": track["global_id"],
                    "camera_id": track["camera_id"],
                    "box": track["box"],
                    "object_class": track["class"],
                    "frame_count": self.frame_count,
                }
                asyncio.run_coroutine_threadsafe(
                    ws_manager.broadcast("tracks", msg),
                    self._event_loop,
                )
        except Exception:
            pass  # WebSocket push la best-effort

    def _push_stats_ws(self):
        """Push system stats qua /ws/stats moi 1 giay."""
        try:
            from routers.websocket import ws_manager
            if not hasattr(self, '_event_loop') or ws_manager.get_count("stats") == 0:
                return
            from services.alert_service import count_alerts_today
            from models.database import SessionLocal
            db = SessionLocal()
            try:
                alerts_today = count_alerts_today(db)
            finally:
                db.close()
            msg = {
                "event":         "stats",
                "fps":           round(self.fps, 1),
                "frame_count":   self.frame_count,
                "active_tracks": self._active_track_count,
                "alerts_today":  alerts_today,
                "status":        self.status,
            }
            asyncio.run_coroutine_threadsafe(
                ws_manager.broadcast("stats", msg),
                self._event_loop,
            )
        except Exception:
            pass

    def set_event_loop(self, loop: asyncio.AbstractEventLoop):
        """Goi tu FastAPI lifespan de AI processor co the push WS."""
        self._event_loop = loop

    # ------------------------------------------------------------------
    # Scenario triggers (demo kich ban tai nan)
    # ------------------------------------------------------------------

    SCENARIOS = {
        "hit_and_run": "hit_and_run",
        "pedestrian_hit": "pedestrian_hit",
        "red_light_crash": "red_light_crash",
        "rear_end": "rear_end",
        "sudden_stop": "sudden_stop",
    }

    def trigger_scenario(self, name: str) -> dict:
        """Kich hoat 1 kich ban tai nan trong CARLA (goi tu API).

        Tra ve dict {ok, message}. An toan goi tu thread khac
        (chi spawn actor + set autopilot, khong dung chung state voi _main_loop).
        """
        if self.status != "running":
            return {"ok": False, "message": "AI processor chua chay"}

        if self.scenario_controller is None:
            return {"ok": False,
                    "message": "Kich ban demo chi kha dung o che do CARLA (--source carla)"}

        method_name = self.SCENARIOS.get(name)
        if method_name is None:
            return {"ok": False, "message": f"Kich ban khong hop le: {name}"}

        try:
            with self._lock:
                result = getattr(self.scenario_controller, method_name)()
            camera_id = (result or {}).get("camera_id")
            if camera_id:
                message = f"Da kich hoat kich ban '{name}' - quan sat tai camera {camera_id}"
            else:
                message = (f"Da kich hoat kich ban '{name}' - khong xac dinh duoc "
                            f"camera quan sat (vi tri nam ngoai tam camera)")
            logger.info("Scenario triggered: %s (camera_id=%s)", name, camera_id)
            self._schedule_scenario_cleanup()
            return {"ok": True, "message": message, "camera_id": camera_id}
        except Exception as e:
            logger.error("Scenario trigger error: %s", e, exc_info=True)
            return {"ok": False, "message": str(e)}

    def _schedule_scenario_cleanup(self):
        """Hen gio xoa cac actor (xe/nguoi do scenario tao ra) sau
        SCENARIO_CLEANUP_DELAY giay, de camera tro lai binh thuong. Neu co
        scenario moi duoc trigger truoc khi het gio, hen lai tu dau."""
        if self._scenario_cleanup_timer is not None:
            self._scenario_cleanup_timer.cancel()

        def _do_cleanup():
            try:
                with self._lock:
                    self.scenario_controller.cleanup()
                logger.info("Scenario actors cleaned up, camera tro lai binh thuong")
            except Exception as e:
                logger.warning("Scenario cleanup error (non-fatal): %s", e)

        self._scenario_cleanup_timer = threading.Timer(
            self.SCENARIO_CLEANUP_DELAY, _do_cleanup)
        self._scenario_cleanup_timer.daemon = True
        self._scenario_cleanup_timer.start()

    def _cleanup(self):
        """Don dep khi dung."""
        try:
            if self._scenario_cleanup_timer is not None:
                self._scenario_cleanup_timer.cancel()
            if hasattr(self, "evidence"):
                self.evidence.flush()
            # scenario_controller/world co the ton tai nhung = None o che do
            # video file (source='file') — dung getattr(..., None) thay hasattr
            # de tranh AttributeError tren None.cleanup()/None.get_settings().
            if getattr(self, "scenario_controller", None):
                self.scenario_controller.cleanup()
            if hasattr(self, "camera_controller"):
                self.camera_controller.cleanup()
            if hasattr(self, "traffic_generator"):
                self.traffic_generator.cleanup()
            if getattr(self, "world", None):
                settings = self.world.get_settings()
                settings.synchronous_mode = False
                self.world.apply_settings(settings)
            if self.source != "carla" and hasattr(self, "frame_source"):
                self.frame_source.release_all()
        except Exception as e:
            logger.error(f"Cleanup error: {e}")

    def get_status(self) -> dict:
        return {
            "status": self.status,
            "frame_count": self.frame_count,
            "fps": round(self.fps, 1),
            "source": self.source,
            "map_enabled": self.route_predictor is not None,
        }

    # ------------------------------------------------------------------
    # World position / Goal Classifier / Route Predictor (map view)
    # ------------------------------------------------------------------

    _MAX_RANGE_M = 150.0  # bo qua diem ground-plane qua xa camera (gan horizon ->
                          # toa do the gioi bi "no" do phep chieu ground-plane, khong
                          # phai loi detect thuc).

    def _update_world_position(self, gid: int, camera_id: str, calib, box: list,
                                obj_class: str, now_ts: float):
        """Chieu pixel -> world (met) qua calibration camera, uoc luong
        heading/speed tu lich su vi tri gan nhat, va lat/lon neu co georef.
        Ket qua luu vao self._world_pos[gid] de get_map_state() doc."""
        x1, y1, x2, y2 = box
        cx, cy_bottom = (x1 + x2) / 2.0, y2  # ground-contact point, chinh xac
                                              # hon center khi chieu len ground-plane
        wx, wy = calib.pixel_to_world(cx, cy_bottom)

        cam_cfg = next((c for c in self.config["cameras"].values()
                         if c["camera_id"] == camera_id), None)
        if cam_cfg is not None:
            cam_x, cam_y, _ = cam_cfg["position"]
            if math.hypot(wx - cam_x, wy - cam_y) > self._MAX_RANGE_M:
                return  # pixel gan horizon -> diem the gioi khong dang tin, bo qua

        hist = self._world_hist.setdefault(gid, deque(maxlen=5))
        hist.append((now_ts, wx, wy))

        heading_deg, speed_ms = 0.0, 0.0
        if len(hist) >= 2:
            t0, x0, y0 = hist[0]
            t1, x1w, y1w = hist[-1]
            if math.hypot(x1w - x0, y1w - y0) > 0.3:
                heading_deg = math.degrees(math.atan2(y1w - y0, x1w - x0))
            dt = t1 - t0
            if dt > 1e-3:
                speed_ms = math.hypot(x1w - x0, y1w - y0) / dt

        lat = lon = None
        if self.georef is not None:
            lat, lon = self.georef.carla_to_latlon(wx, wy)

        self._world_pos[gid] = {
            "global_id": gid,
            "camera_id": camera_id,
            "class": obj_class,
            "x": round(wx, 2), "y": round(wy, 2),
            "lat": round(lat, 6) if lat is not None else None,
            "lon": round(lon, 6) if lon is not None else None,
            "heading_deg": round(heading_deg, 1),
            "speed_ms": round(speed_ms, 2),
        }

    def _update_marked_route(self, gid: int):
        """Du doan route nhieu hop cho object da duoc mark, dung huong re
        moi nhat tu Goal Classifier (hoac 'straight' neu chua du tin)."""
        if self.route_predictor is None:
            return
        pos = self._world_pos.get(gid)
        if pos is None:
            return

        direction = "straight"
        probs = {"straight": 1.0, "left": 0.0, "right": 0.0, "u_turn": 0.0}
        confidence = 1.0
        goal = self._goal_predictions.get(gid)
        if goal and goal["confidence"] >= self.route_min_confidence:
            direction = goal["direction"]
            probs = goal["probabilities"]
            confidence = goal["confidence"]

        self._marked_route = self.route_predictor.predict(
            pos["x"], pos["y"], pos["heading_deg"],
            direction=direction, n_hops=self.route_n_hops, direction_probs=probs,
        )
        pos["direction"] = direction
        pos["confidence"] = round(confidence, 3)
        pos["probabilities"] = {k: round(v, 3) for k, v in probs.items()}

    def _prune_stale_tracking_state(self, all_global_tracks: list):
        """Xoa world-position/goal-prediction cua global_id khong con active
        trong frame nay, tranh memory tang vo han qua thoi gian chay dai."""
        active_ids = {t["global_id"] for t in all_global_tracks}
        for store in (self._world_hist, self._world_pos, self._goal_predictions):
            for gid in list(store.keys()):
                if gid not in active_ids:
                    store.pop(gid, None)

    def get_map_state(self) -> dict:
        """Trang thai cho map view: vi tri camera (lat/lon), tat ca track
        dang active (lat/lon + huong re), va route du doan cho object da
        duoc danh dau (neu co). enabled=False neu nguon hien tai khong co
        georef hop le (vd CARLA Town10HD_Opt mac dinh)."""
        if self.georef is None:
            return {
                "enabled": False,
                "cameras": [], "tracks": [], "marked": None,
                "message": ("Map/route prediction chua duoc cau hinh cho nguon nay - "
                             "can section 'route_prediction' trong config camera, tro "
                             "dung 1 map da georeference."),
            }

        cameras = []
        for cam_cfg in self.config["cameras"].values():
            cx, cy, _ = cam_cfg["position"]
            lat, lon = self.georef.carla_to_latlon(cx, cy)
            cameras.append({
                "camera_id": cam_cfg["camera_id"],
                "lat": round(lat, 6), "lon": round(lon, 6),
            })

        tracks = []
        for gid, pos in self._world_pos.items():
            item = dict(pos)
            item["is_marked"] = (gid == self.marked_global_id)
            tracks.append(item)

        marked = None
        if self.marked_global_id is not None:
            pos = self._world_pos.get(self.marked_global_id)
            if pos is not None:
                marked = {
                    "global_id": self.marked_global_id,
                    "lat": pos["lat"], "lon": pos["lon"],
                    "direction": pos.get("direction", "straight"),
                    "confidence": pos.get("confidence", 0.0),
                    "probabilities": pos.get("probabilities", {}),
                    "route": self._marked_route,
                }

        return {"enabled": True, "cameras": cameras, "tracks": tracks, "marked": marked}

    def mark_object(self, global_id: int) -> dict:
        """Danh dau 1 object de theo doi + du doan route dai han (goi tu API,
        vd khi nguoi van hanh click 1 xe tren map / dashboard)."""
        if self.route_predictor is None:
            return {"ok": False,
                    "message": "Route prediction chua duoc cau hinh cho nguon nay"}
        if global_id not in self._world_pos:
            return {"ok": False,
                    "message": f"Object #{global_id} khong ton tai hoac chua co vi tri the gioi"}
        with self._lock:
            self.marked_global_id = global_id
            self._marked_route = []
            self._marked_snapshot_saved = False
        return {"ok": True, "message": f"Da danh dau object #{global_id}",
                "global_id": global_id}

    def unmark_object(self) -> dict:
        with self._lock:
            self.marked_global_id = None
            self._marked_route = []
        return {"ok": True, "message": "Da bo danh dau"}


# Singleton
ai_processor: AIProcessor | None = None


def get_ai_processor() -> AIProcessor | None:
    return ai_processor


def create_ai_processor(config_path: str, **kwargs) -> AIProcessor:
    global ai_processor
    ai_processor = AIProcessor(config_path, **kwargs)
    return ai_processor
