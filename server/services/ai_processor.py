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
import asyncio
import logging
import threading
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
                 carla_port: int = 2000):
        self.config_path = config_path
        self.carla_host = carla_host
        self.carla_port = carla_port

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
        """Khoi tao CARLA + cac module AI."""
        logger.info("Initializing AI pipeline...")

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

        # Import AI modules
        from modules.camera_controller import CameraController
        from modules.traffic_generator import TrafficGenerator
        from modules.detector import ObjectDetector
        from modules.tracker import ByteTrackWrapper
        from modules.reid import DualReIDExtractor
        from modules.global_tracking import GlobalTracker
        from modules.trajectory_predictor import EnsembleTrajectoryPredictor
        from modules.calibration import CalibrationStore
        from modules.alert_system import AlertSystem
        from modules.incident_detector import IncidentDetector
        from modules.evidence_package import EvidencePackage
        from modules.scenario_controller import ScenarioController
        from utils.visualization import Visualizer

        # Khoi tao modules
        self.camera_controller = CameraController(
            self.client, self.world, self.config_path)
        self.camera_controller.setup_cameras()

        self.traffic_generator = TrafficGenerator(
            self.world, num_vehicles=10, num_pedestrians=5)
        self.traffic_generator.spawn_actors()

        self.detector = ObjectDetector(model_type="yolov8s", half=True)

        self.trackers = {}
        for cam_id in self.camera_controller.cameras:
            self.trackers[cam_id] = ByteTrackWrapper(
                track_activation_threshold=0.25,
                lost_track_buffer=30,
                minimum_matching_threshold=0.8,
                frame_rate=10,
            )

        vehicle_weights = str(_tracking_dir / "weights" / "osnet_veri776.pth")
        self.reid_extractor = DualReIDExtractor(vehicle_weights=vehicle_weights)

        # Đọc camera_topology từ config (cặp key dùng "__" làm phân cách vì YAML không hỗ trợ tuple key)
        raw_topology = self.camera_controller.config.get("camera_topology", {})
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
        if "rois" in self.camera_controller.config:
            self.alert_system.set_rois(self.camera_controller.config["rois"])

        self.incident_detector = IncidentDetector(
            config=self.camera_controller.config.get("incident_detection", {})
        )
        self.camera_traffic_lights = self._find_nearby_traffic_lights()
        self.evidence = EvidencePackage(output_dir="evidence", buffer_seconds=30, fps=10)

        self.scenario_controller = ScenarioController(
            self.world, self.client,
            cameras_config=self.camera_controller.config.get("cameras", {}))

        self.visualizer = Visualizer()

        # Sync cameras vao database
        self._sync_cameras_to_db()

        logger.info("AI pipeline initialized successfully")

    def _sync_cameras_to_db(self):
        """Dong bo camera config vao database."""
        from models.database import SessionLocal, Camera

        db = SessionLocal()
        try:
            for cam_name, cam_cfg in self.camera_controller.config["cameras"].items():
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
        red_light_cfg = self.camera_controller.config.get("incident_detection", {}).get("red_light", {})
        radius = red_light_cfg.get("search_radius_m", 40)

        try:
            lights = list(self.world.get_actors().filter("traffic.traffic_light*"))
        except Exception as e:
            logger.warning("Khong lay duoc traffic lights tu CARLA: %s", e)
            return {}

        mapping = {}
        for cam_cfg in self.camera_controller.config["cameras"].values():
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

    def _get_traffic_light_state(self, camera_id: str):
        """Tra ve 'Red'/'Yellow'/'Green'/None cho camera (uu tien Red neu co
        >=1 traffic light gan camera dang Red)."""
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
        from services.tracking_service import upsert_tracked_object, add_history
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

            # Lay frames tu cameras
            sync_frames = self.camera_controller.get_synchronized_frames()
            if not sync_frames:
                self.world.tick()
                continue

            # Mo 1 DB session cho frame nay
            db = SessionLocal()
            try:
                all_global_tracks = []

                # 1. Batch detection — 1 GPU forward pass cho tat ca camera
                cam_ids = list(sync_frames.keys())
                frames = [sync_frames[c]["frame"] for c in cam_ids]
                batch_detections = self.detector.detect_batch(frames)

                for camera_id, frame, detections in zip(cam_ids, frames, batch_detections):

                    # 2. Single-camera tracking (ByteTrackWrapper cần thêm frame)
                    local_tracks = self.trackers[camera_id].update(detections, frame)

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

                    # 4. Update trajectory + check alerts
                    now_ts = time.time()
                    predictions_map = {}  # {global_id: dict} cho incident detector
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
                            # check_alerts() cần list[[x,y]]
                            alerts = self.alert_system.check_alerts(
                                gid, camera_id,
                                self.trajectory_predictor.predict_list(gid), box
                            )
                            for alert in alerts:
                                self.alert_system.log_alert(alert)
                                self._save_alert(db, alert)

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
                        traffic_light_state=self._get_traffic_light_state(camera_id),
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

                    # 9. Day frame vao buffer cho MJPEG streaming
                    frame_buffer.put_frame(camera_id, vis_frame)

                db.commit()

                self._active_track_count = len(all_global_tracks)
                self._push_ws_updates(all_global_tracks)

            except Exception as e:
                logger.error(f"Frame processing error: {e}", exc_info=True)
                db.rollback()
            finally:
                db.close()

            # Tick CARLA
            self.world.tick()

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
            message=f"Object {alert_dict.get('global_id')} entering {alert_dict.get('roi_name')}",
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
            return {"ok": True, "message": message, "camera_id": camera_id}
        except Exception as e:
            logger.error("Scenario trigger error: %s", e, exc_info=True)
            return {"ok": False, "message": str(e)}

    def _cleanup(self):
        """Don dep khi dung."""
        try:
            if hasattr(self, "evidence"):
                self.evidence.flush()
            if hasattr(self, "scenario_controller"):
                self.scenario_controller.cleanup()
            if hasattr(self, "camera_controller"):
                self.camera_controller.cleanup()
            if hasattr(self, "traffic_generator"):
                self.traffic_generator.cleanup()
            if hasattr(self, "world"):
                settings = self.world.get_settings()
                settings.synchronous_mode = False
                self.world.apply_settings(settings)
        except Exception as e:
            logger.error(f"Cleanup error: {e}")

    def get_status(self) -> dict:
        return {
            "status": self.status,
            "frame_count": self.frame_count,
            "fps": round(self.fps, 1),
        }


# Singleton
ai_processor: AIProcessor | None = None


def get_ai_processor() -> AIProcessor | None:
    return ai_processor


def create_ai_processor(config_path: str, **kwargs) -> AIProcessor:
    global ai_processor
    ai_processor = AIProcessor(config_path, **kwargs)
    return ai_processor
