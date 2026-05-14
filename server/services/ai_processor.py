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
        from modules.tracker import SimpleTracker
        from modules.reid import ReIDExtractor
        from modules.global_tracking import GlobalTracker
        from modules.trajectory_predictor import TrajectoryPredictor
        from modules.alert_system import AlertSystem
        from utils.visualization import Visualizer

        # Khoi tao modules
        self.camera_controller = CameraController(
            self.client, self.world, self.config_path)
        self.camera_controller.setup_cameras()

        self.traffic_generator = TrafficGenerator(
            self.world, num_vehicles=10, num_pedestrians=5)
        self.traffic_generator.spawn_actors()

        self.detector = ObjectDetector(model_type="yolov5s")

        self.trackers = {}
        for cam_id in self.camera_controller.cameras:
            self.trackers[cam_id] = SimpleTracker(max_age=30, min_hits=3)

        self.reid_extractor = ReIDExtractor()
        self.global_tracker = GlobalTracker(self.reid_extractor)
        self.trajectory_predictor = TrajectoryPredictor(window_size=10, pred_steps=5)

        self.alert_system = AlertSystem(self.trajectory_predictor)
        if "rois" in self.camera_controller.config:
            self.alert_system.set_rois(self.camera_controller.config["rois"])

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

            # Tinh FPS moi 1 giay
            elapsed = time.time() - fps_start
            if elapsed >= 1.0:
                self.fps = fps_frame_count / elapsed
                fps_frame_count = 0
                fps_start = time.time()

            # Lay frames tu cameras
            sync_frames = self.camera_controller.get_synchronized_frames()
            if not sync_frames:
                self.world.tick()
                continue

            # Mo 1 DB session cho frame nay
            db = SessionLocal()
            try:
                all_global_tracks = []

                for camera_id, frame_data in sync_frames.items():
                    frame = frame_data["frame"]

                    # 1. Detection
                    detections = self.detector.detect(frame)

                    # 2. Single-camera tracking
                    local_tracks = self.trackers[camera_id].update(detections)

                    # 3. Cross-camera global tracking
                    global_tracks = self.global_tracker.process_camera_tracks(
                        camera_id, frame, local_tracks
                    )
                    all_global_tracks.extend(global_tracks)

                    # 4. Update trajectory + check alerts
                    for g_track in global_tracks:
                        gid = g_track["global_id"]
                        box = g_track["box"]
                        center = [(box[0] + box[2]) / 2, (box[1] + box[3]) / 2]

                        self.trajectory_predictor.update_trajectory(
                            gid, center, self.frame_count
                        )

                        predicted = self.trajectory_predictor.predict(gid)
                        if predicted:
                            alerts = self.alert_system.check_alerts(
                                gid, camera_id, predicted, box
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

                    # 5. Ve visualization len frame
                    vis_frame = self.visualizer.draw_tracks(frame, global_tracks)

                    # 6. Day frame vao buffer cho MJPEG streaming
                    frame_buffer.put_frame(camera_id, vis_frame)

                db.commit()

                # 7. Push WebSocket updates (async, fire-and-forget)
                self._push_ws_updates(all_global_tracks)

            except Exception as e:
                logger.error(f"Frame processing error: {e}", exc_info=True)
                db.rollback()
            finally:
                db.close()

            # Tick CARLA
            self.world.tick()

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

            # Chi push neu co clients dang ket noi
            if ws_manager.get_count("tracks") > 0:
                for track in global_tracks:
                    msg = {
                        "event": "track_update",
                        "global_id": track["global_id"],
                        "camera_id": track["camera_id"],
                        "box": track["box"],
                        "object_class": track["class"],
                        "frame_count": self.frame_count,
                    }
                    # Schedule coroutine len event loop cua FastAPI
                    asyncio.run_coroutine_threadsafe(
                        ws_manager.broadcast("tracks", msg),
                        self._event_loop,
                    )
        except Exception:
            pass  # WebSocket push la best-effort

    def set_event_loop(self, loop: asyncio.AbstractEventLoop):
        """Goi tu FastAPI lifespan de AI processor co the push WS."""
        self._event_loop = loop

    def _cleanup(self):
        """Don dep khi dung."""
        try:
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
