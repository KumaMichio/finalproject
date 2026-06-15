#!/usr/bin/env python3
"""
Multi-Camera CCTV Tracking System for CARLA
Main entry point for the tracking system
"""

import sys
import os
import time
import argparse
import logging
import cv2
import yaml
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.append(str(project_root))

from modules.detector import ObjectDetector
from modules.tracker import ByteTrackWrapper
from modules.reid import ReIDExtractor
from modules.global_tracking import GlobalTracker
from modules.trajectory_predictor import EnsembleTrajectoryPredictor
from modules.calibration import CalibrationStore
from modules.ground_truth import project_actors_to_cameras
from modules.alert_system import AlertSystem
from modules.incident_detector import IncidentDetector
from modules.evidence_package import EvidencePackage
from modules.video_source import (
    CARLABridgeClient, MultiVideoSource,
    RTSPVideoSource, FileVideoSource, WebcamVideoSource,
)
from utils.visualization import Visualizer
from utils.metrics import MetricsCollector

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('tracking_system.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# CARLA fixed_delta_seconds = 0.1 → effective simulation FPS = 10
_CARLA_FPS = 10


class TrackingSystem:
    def __init__(self, config_path, half: bool = False, source: str = 'carla',
                 bridge_host: str = '127.0.0.1', bridge_port: int = 8765,
                 video_paths: list = None, rtsp_urls: list = None,
                 webcam_indices: list = None, camera_ids: list = None,
                 loop_video: bool = False, eval_output: str = None):
        self.config_path = config_path
        self.half = half        # FP16 inference — saves ~800 MB VRAM alongside CARLA
        self.source = source    # 'carla'/'bridge' (CARLA) or 'rtsp'/'file'/'webcam' (real cameras)
        self.bridge_host = bridge_host
        self.bridge_port = bridge_port
        self.video_paths = video_paths or []
        self.rtsp_urls = rtsp_urls or []
        self.webcam_indices = webcam_indices or []
        self.camera_ids = camera_ids
        self.loop_video = loop_video
        self.eval_output = eval_output   # directory to write pred_<CAM_ID>.txt for evaluate_tracking.py
        self.carla_client = None
        self.world = None
        # camera_controller (carla) / CARLABridgeClient (bridge) / MultiVideoSource (rtsp/file/webcam)
        self.frame_source = None
        self.modules = {}
        self._eval_calibrations = None  # {camera_id: CameraCalibration}, set if --eval-output + --source carla

    # ------------------------------------------------------------------
    # CARLA setup
    # ------------------------------------------------------------------

    def initialize_carla(self):
        try:
            import carla
            self.carla_client = carla.Client('localhost', 2000)
            self.carla_client.set_timeout(10.0)
            self.world = self.carla_client.get_world()
            logger.info("CARLA client initialized")
            return True
        except Exception as e:
            logger.error("Failed to initialize CARLA: %s", e)
            return False

    def setup_synchronous_mode(self):
        settings = self.world.get_settings()
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = 1.0 / _CARLA_FPS
        self.world.apply_settings(settings)
        logger.info("Synchronous mode enabled at %d fps", _CARLA_FPS)

    # ------------------------------------------------------------------
    # Module initialisation
    # ------------------------------------------------------------------

    def initialize_modules(self):
        def step(name):
            print(f"[INIT] {name}...", flush=True)
            logger.info("Initializing: %s", name)

        try:
            if self.source == 'carla':
                from modules.camera_controller import CameraController
                from modules.traffic_generator import TrafficGenerator

                step("CameraController")
                self.modules['camera_controller'] = CameraController(
                    self.carla_client, self.world, self.config_path)
                self.modules['camera_controller'].setup_cameras()
                cam_ids = list(self.modules['camera_controller'].cameras.keys())
                self.frame_source = self.modules['camera_controller']
                rois_config = self.modules['camera_controller'].config['rois']

                step("TrafficGenerator")
                self.modules['traffic_generator'] = TrafficGenerator(
                    self.world, num_vehicles=10, num_pedestrians=5)
                self.modules['traffic_generator'].spawn_actors()
            elif self.source == 'bridge':
                step("CARLABridgeClient")
                self.frame_source = CARLABridgeClient(
                    host=self.bridge_host, port=self.bridge_port)
                with open(self.config_path, 'r', encoding='utf-8') as f:
                    cfg = yaml.safe_load(f)
                cam_ids = [c['camera_id'] for c in cfg['cameras'].values()]
                rois_config = cfg['rois']
            else:
                # 'rtsp' / 'file' / 'webcam' — real camera sources via VideoSource
                step(f"VideoSource ({self.source})")
                if self.source == 'rtsp':
                    sources_input = self.rtsp_urls
                elif self.source == 'file':
                    sources_input = self.video_paths
                else:
                    sources_input = self.webcam_indices

                if not sources_input:
                    raise ValueError(
                        f"--source {self.source} requires at least one "
                        f"--rtsp-url / --video-path / --webcam-index")

                cam_ids = self.camera_ids or [
                    f"CAM_{i + 1:03d}" for i in range(len(sources_input))]
                if len(cam_ids) != len(sources_input):
                    raise ValueError(
                        "--camera-ids must list exactly one ID per source "
                        f"({len(sources_input)} source(s), {len(cam_ids)} ID(s))")

                self.frame_source = MultiVideoSource()
                for cam_id, src in zip(cam_ids, sources_input):
                    if self.source == 'rtsp':
                        self.frame_source.add_source(RTSPVideoSource(src, cam_id))
                    elif self.source == 'file':
                        self.frame_source.add_source(
                            FileVideoSource(src, cam_id, loop=self.loop_video))
                    else:
                        self.frame_source.add_source(WebcamVideoSource(src, cam_id))

                with open(self.config_path, 'r', encoding='utf-8') as f:
                    cfg = yaml.safe_load(f)
                rois_config = cfg.get('rois', {})

            step("ObjectDetector")
            # Use the VN-traffic fine-tuned checkpoint if available, otherwise
            # fall back to the stock YOLO11m (COCO) weights.
            yolo_vn_weights = project_root / 'weights' / 'yolo11m_vn.pt'
            detector_model = str(yolo_vn_weights) if yolo_vn_weights.exists() else 'yolo11m'
            self.modules['detector'] = ObjectDetector(
                model_type=detector_model,
                conf_threshold=0.35,
                half=self.half,   # FP16 when running alongside CARLA
            )

            step("ByteTrackWrapper (per camera)")
            self.modules['trackers'] = {
                cam_id: ByteTrackWrapper(
                    track_activation_threshold=0.25,
                    lost_track_buffer=30,
                    minimum_matching_threshold=0.8,
                    frame_rate=_CARLA_FPS,   # 10 fps, not 30
                    class_map=self.modules['detector'].CLASSES,
                )
                for cam_id in cam_ids
            }

            step("ReIDExtractor")
            # osnet_x1_0 (Market-1501) — proper ReID model, not ImageNet ResNet50
            self.modules['reid_extractor'] = ReIDExtractor(model_name='osnet_x1_0')

            step("GlobalTracker")
            self.modules['global_tracker'] = GlobalTracker(
                self.modules['reid_extractor'],
                match_threshold=0.5,
                # camera_topology — leave empty for CARLA (no real physical distances).
                # For real CCTV, set: {('CAM_001','CAM_002'): (5, 60), ...}
                camera_topology={},
            )

            step("TrajectoryPredictor")
            try:
                calibration = CalibrationStore.load_from_config(self.config_path)
            except Exception as e:
                logger.warning("CalibrationStore.load_from_config failed (%s) "
                                "-> trajectory predictor will be Kalman-only", e)
                calibration = {}
            self.modules['trajectory_predictor'] = EnsembleTrajectoryPredictor(
                window_size=10, calibration=calibration,
                gru_checkpoint='weights/trajectory_gru.pth')

            step("AlertSystem")
            self.modules['alert_system'] = AlertSystem(self.modules['trajectory_predictor'])
            self.modules['alert_system'].set_rois(rois_config)

            step("IncidentDetector")
            self.modules['incident_detector'] = IncidentDetector()

            step("EvidencePackage")
            self.modules['evidence'] = EvidencePackage(
                output_dir='evidence', buffer_seconds=30, fps=_CARLA_FPS)

            step("Visualizer")
            self.modules['visualizer'] = Visualizer()

            step("MetricsCollector")
            self.modules['metrics'] = MetricsCollector()

            if self.eval_output:
                step("Eval output (pred_<CAM_ID>.txt)")
                os.makedirs(self.eval_output, exist_ok=True)
                self.modules['eval_pred_files'] = {
                    cam_id: open(os.path.join(self.eval_output, f'pred_{cam_id}.txt'), 'w')
                    for cam_id in cam_ids
                }

                # --source carla: this process has direct CARLA world access,
                # so it can also export ground truth (gt_<CAM_ID>.txt) for
                # evaluate_tracking.py. Used for evaluation logging only —
                # never fed back into detection/tracking decisions.
                if self.source == 'carla' and calibration:
                    self._eval_calibrations = calibration
                    self.modules['eval_gt_files'] = {
                        cam_id: open(os.path.join(self.eval_output, f'gt_{cam_id}.txt'), 'w')
                        for cam_id in cam_ids
                    }
                else:
                    self._eval_calibrations = None

            print(f"[INIT] All modules OK — {len(cam_ids)} cameras: {cam_ids}", flush=True)
            logger.info("All modules initialised successfully")
            return True

        except Exception as e:
            print(f"[INIT ERROR] {e}", flush=True)
            logger.error("Failed to initialize modules: %s", e, exc_info=True)
            return False

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------

    def run(self, max_frames=None):
        frame_count = 0
        _fps_t0     = time.perf_counter()
        _fps_count  = 0
        logger.info("Starting tracking system...")

        try:
            while True:
                if max_frames and frame_count >= max_frames:
                    logger.info("Reached max frames (%d), stopping.", max_frames)
                    break

                frame_count += 1
                _fps_count  += 1

                # ----------------------------------------------------------
                # Grab frames from all cameras
                # ----------------------------------------------------------
                sync_frames = self.frame_source.get_synchronized_frames()

                if not sync_frames:
                    logger.warning("No frames received, skipping...")
                    if self.source == 'carla':
                        self.world.tick()
                    continue

                cam_ids = list(sync_frames.keys())
                frames  = [sync_frames[c]['frame'] for c in cam_ids]

                # Ground-truth export (--source carla + --eval-output only)
                if self._eval_calibrations is not None:
                    gt_data = project_actors_to_cameras(self.world, self._eval_calibrations)
                    for cam_id, items in gt_data.items():
                        f = self.modules['eval_gt_files'].get(cam_id)
                        if f is None:
                            continue
                        for item in items:
                            x1, y1, x2, y2 = item['box']
                            f.write(f"{frame_count},{item['id']},"
                                    f"{x1:.2f},{y1:.2f},{x2:.2f},{y2:.2f},{item['class']}\n")

                # ----------------------------------------------------------
                # Phase 1 — Batch detection (1 GPU forward pass for all cameras)
                # Replaces N separate detect() calls → N× faster on GPU
                # ----------------------------------------------------------
                batch_detections = self.modules['detector'].detect_batch(frames)

                # ----------------------------------------------------------
                # Phase 2 — Per-camera CPU processing
                # Single timestamp for the whole frame so all cameras are
                # consistent when TrajectoryPredictor computes velocity.
                # ----------------------------------------------------------
                all_global_tracks = []
                now_ts = time.time()

                for camera_id, frame, detections in zip(cam_ids, frames, batch_detections):

                    # Single-camera tracking (Kalman + Hungarian via ByteTrack)
                    local_tracks = self.modules['trackers'][camera_id].update(
                        detections, frame)

                    # Cross-camera Re-ID → assign Global ID
                    global_tracks = self.modules['global_tracker'].process_camera_tracks(
                        camera_id, frame, local_tracks)

                    all_global_tracks.extend(global_tracks)

                    # Ground-truth evaluation: dump predicted boxes for this frame
                    if self.eval_output:
                        eval_frame = sync_frames[camera_id].get('frame_number', frame_count)
                        f = self.modules['eval_pred_files'][camera_id]
                        for g in global_tracks:
                            x1, y1, x2, y2 = g['box']
                            f.write(f"{eval_frame},{g['global_id']},"
                                    f"{x1:.2f},{y1:.2f},{x2:.2f},{y2:.2f},{g.get('class', '')}\n")

                    # Trajectory update (real unix time, not frame index)
                    for g in global_tracks:
                        box = g['box']
                        self.modules['trajectory_predictor'].update_trajectory(
                            g['global_id'],
                            [(box[0] + box[2]) / 2, (box[1] + box[3]) / 2],
                            now_ts,
                            camera_id=camera_id,
                            obj_class=g.get('class', 'car'),
                            all_tracks=all_global_tracks,
                        )

                    # Build predictions dict once — reused by both alert checks below
                    predictions = {}
                    for g in global_tracks:
                        pred = self.modules['trajectory_predictor'].predict(g['global_id'])
                        if pred is not None:
                            predictions[g['global_id']] = pred

                    # ROI entry alerts (AlertSystem expects plain list of [x,y])
                    for g in global_tracks:
                        pred_list = self.modules['trajectory_predictor'].predict_list(
                            g['global_id'])
                        for alert in self.modules['alert_system'].check_alerts(
                                g['global_id'], camera_id, pred_list, g['box']):
                            self.modules['alert_system'].log_alert(alert)

                    # Incident detection — reactive + proactive (predicted collision/ROI)
                    incidents = self.modules['incident_detector'].update(
                        global_tracks, camera_id,
                        predictions=predictions,
                        rois=self.modules['alert_system'].rois,
                    )

                    for incident in incidents:
                        logger.warning(
                            "[INCIDENT] %s | %s | obj=%d | %s",
                            incident['severity'], incident['type'],
                            incident['global_id'], incident['message'],
                        )
                        if incident['severity'] == 'CRITICAL':
                            # Capture evidence from ALL cameras, not just the triggering one
                            self.modules['evidence'].capture(
                                incident=incident,
                                frames_dict={cid: sync_frames[cid]['frame']
                                             for cid in cam_ids},
                                global_tracks=all_global_tracks,
                            )

                    # Ring-buffer frame for evidence pre-event clip
                    self.modules['evidence'].buffer_frame(camera_id, frame)

                    # Visualisation
                    vis_frame = self.modules['visualizer'].draw_tracks(frame, global_tracks)
                    if predictions:
                        predictions_for_viz = {gid: list(pred.values())
                                                for gid, pred in predictions.items()}
                        vis_frame = self.modules['visualizer'].draw_predictions(
                            vis_frame, predictions_for_viz, global_tracks)
                    cv2.imshow(camera_id, vis_frame)

                # ----------------------------------------------------------
                # End-of-frame housekeeping
                # ----------------------------------------------------------
                self.modules['metrics'].update(frame_count, all_global_tracks)
                if self.source == 'carla':
                    self.world.tick()

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    logger.info("Exit signal received, stopping.")
                    break

                # FPS report every 100 frames
                if _fps_count == 100:
                    elapsed = time.perf_counter() - _fps_t0
                    logger.info("Pipeline FPS: %.1f  (frame %d)", 100 / elapsed, frame_count)
                    _fps_t0    = time.perf_counter()
                    _fps_count = 0

        except KeyboardInterrupt:
            logger.info("Interrupted by user")
        except Exception as e:
            logger.error("Error in main loop: %s", e, exc_info=True)
        finally:
            self.cleanup()

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------

    def cleanup(self):
        logger.info("Cleaning up...")
        cv2.destroyAllWindows()

        if self.frame_source is not None:
            if hasattr(self.frame_source, 'cleanup'):
                self.frame_source.cleanup()
            elif hasattr(self.frame_source, 'release_all'):
                self.frame_source.release_all()
        if 'evidence' in self.modules:
            self.modules['evidence'].flush()

        if 'eval_pred_files' in self.modules:
            for f in self.modules['eval_pred_files'].values():
                f.close()
        if 'eval_gt_files' in self.modules:
            for f in self.modules['eval_gt_files'].values():
                f.close()

        if self.source == 'carla':
            if 'traffic_generator' in self.modules:
                self.modules['traffic_generator'].cleanup()

            if self.world:
                settings = self.world.get_settings()
                settings.synchronous_mode = False
                self.world.apply_settings(settings)

        logger.info("Cleanup completed")


# ------------------------------------------------------------------
# Entry point
# ------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description='Multi-Camera CCTV Tracking System')
    parser.add_argument('--config', default='config/camera_config.yaml',
                        help='Path to configuration file')
    parser.add_argument('--max-frames', type=int, default=None,
                        help='Stop after N frames (default: run until Ctrl-C or q)')
    parser.add_argument('--half', action='store_true',
                        help='FP16 inference — saves ~800 MB VRAM when running with CARLA')
    parser.add_argument('--log-level', default='INFO',
                        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'])
    parser.add_argument('--source', choices=['carla', 'bridge', 'rtsp', 'file', 'webcam'],
                        default='carla',
                        help="'carla': connect to CARLA directly (in-process). "
                             "'bridge': receive frames from carla_bridge/server.py over TCP. "
                             "'rtsp'/'file'/'webcam': real camera sources via VideoSource")
    parser.add_argument('--bridge-host', default='127.0.0.1',
                        help='Host of carla_bridge/server.py (--source bridge)')
    parser.add_argument('--bridge-port', type=int, default=8765,
                        help='Port of carla_bridge/server.py (--source bridge)')
    parser.add_argument('--rtsp-url', action='append', default=None,
                        help='RTSP/HTTP stream URL (--source rtsp). Repeat for multiple cameras.')
    parser.add_argument('--video-path', action='append', default=None,
                        help='Path to a video file (--source file). Repeat for multiple cameras.')
    parser.add_argument('--webcam-index', action='append', type=int, default=None,
                        help='Webcam device index (--source webcam). Repeat for multiple cameras.')
    parser.add_argument('--camera-ids', default=None,
                        help='Comma-separated camera IDs matching --rtsp-url/--video-path/'
                             '--webcam-index order (default: CAM_001, CAM_002, ...)')
    parser.add_argument('--loop-video', action='store_true',
                        help='Loop video file(s) when reaching the end (--source file)')
    parser.add_argument('--eval-output', default=None,
                        help='Directory to write predicted-track MOT files (pred_<CAM_ID>.txt) '
                             'for evaluate_tracking.py. Pair with carla_bridge/server.py '
                             '--eval-output (--source bridge) to get gt_<CAM_ID>.txt')
    args = parser.parse_args()

    logging.getLogger().setLevel(getattr(logging, args.log_level))

    camera_ids = args.camera_ids.split(',') if args.camera_ids else None

    system = TrackingSystem(args.config, half=args.half, source=args.source,
                             bridge_host=args.bridge_host, bridge_port=args.bridge_port,
                             video_paths=args.video_path, rtsp_urls=args.rtsp_url,
                             webcam_indices=args.webcam_index, camera_ids=camera_ids,
                             loop_video=args.loop_video, eval_output=args.eval_output)

    if system.source == 'carla':
        if not system.initialize_carla():
            sys.exit(1)

        system.setup_synchronous_mode()

    if not system.initialize_modules():
        sys.exit(1)

    system.run(max_frames=args.max_frames)


if __name__ == '__main__':
    main()
